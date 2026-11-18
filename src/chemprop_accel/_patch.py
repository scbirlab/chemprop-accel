"""Reversible, idempotent patches of a handful of Chemprop methods."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from . import _compat, _ops

_LOCK = threading.RLock()
_NATIVE_FLOATS = (torch.float32, torch.float64)


def _exporting() -> bool:
    """True while exporting/tracing for ONNX, TorchScript or ``torch.export``.

    The replacements use a custom operator that exporters cannot serialise, so Chemprop's own
    (exportable) code is used instead. ``torch.compile`` is unaffected.
    """
    try:
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            return True
    except Exception:  # noqa: BLE001
        pass
    f = getattr(torch.compiler, "is_exporting", None)
    return bool(f()) if f is not None else False


def _fast_ok(H: torch.Tensor, *idx: torch.Tensor) -> bool:
    """Replacement kernels run only for plain CPU float tensors; anything else uses the original."""
    return (
        not _exporting()
        and H.device.type == "cpu"
        and H.dtype in _NATIVE_FLOATS
        and H.dim() == 2
        and all(i.device.type == "cpu" and i.dtype == torch.long for i in idx)
    )


# ----------------------------------------------------------------------------------------------
# replacement factories: each takes the *original* function and returns the replacement
# ----------------------------------------------------------------------------------------------
def _bond_message(orig: Callable) -> Callable:
    @functools.wraps(orig)
    def message(self, H, bmg):
        ei, rev = bmg.edge_index, bmg.rev_edge_index
        if not _fast_ok(H, ei, rev):
            return orig(self, H, bmg)
        return _ops.edge_message(H, ei[1], ei[0], rev, bmg.V.shape[0])

    return message


def _atom_message(orig: Callable) -> Callable:
    @functools.wraps(orig)
    def message(self, H, bmg):
        ei = bmg.edge_index
        if not _fast_ok(H, ei) or bmg.E.dtype != H.dtype:
            return orig(self, H, bmg)
        H = torch.cat((H, bmg.E), dim=1)
        return _ops.edge_message(H, ei[1], ei[0], None, bmg.V.shape[0])

    return message


def _mp_forward(orig: Callable) -> Callable:
    @functools.wraps(orig)
    def forward(self, bmg, V_d=None):
        ei = bmg.edge_index
        if not _fast_ok(bmg.V, ei):
            return orig(self, bmg, V_d)
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

        if not _fast_ok(H, bmg.edge_index):
            M = torch.zeros(bmg.V.shape[0], H.shape[1], dtype=H.dtype, device=H.device)
            M = M.scatter_reduce_(
                0,
                bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1]),
                H,
                reduce="sum",
                include_self=False,
            )
        else:
            M = _ops.scatter_sum(H, bmg.edge_index[1], bmg.V.shape[0])
        return self.finalize(M, bmg.V, V_d)

    return forward


def _mean_forward(orig: Callable) -> Callable:
    @functools.wraps(orig)
    def forward(self, H, batch):
        if self.dim != 0 or not _fast_ok(H, batch):
            return orig(self, H, batch)
        n = int(batch.max()) + 1
        sums = _ops.scatter_sum(H, batch, n)
        counts = torch.bincount(batch, minlength=n).to(H.dtype).unsqueeze(1)
        return sums / counts.clamp(min=1)

    return forward


def _sum_forward(orig: Callable) -> Callable:
    @functools.wraps(orig)
    def forward(self, H, batch):
        if self.dim != 0 or not _fast_ok(H, batch):
            return orig(self, H, batch)
        return _ops.scatter_sum(H, batch, int(batch.max()) + 1)

    return forward


@dataclass
class _Patch:
    name: str
    resolve: Callable[[], tuple[type, str]]
    factory: Callable[[Callable], Callable]
    # filled during enable()
    state: str = "inactive"
    original: Callable | None = None
    replacement: Callable | None = None
    owner: type | None = None
    attr: str | None = None


def _chemprop_targets() -> list[_Patch]:
    def bond():
        from chemprop.nn.message_passing.mixins import _BondMessagePassingMixin as C

        return C, "message"

    def atom():
        from chemprop.nn.message_passing.mixins import _AtomMessagePassingMixin as C

        return C, "message"

    def fwd():
        from chemprop.nn.message_passing.base import _MessagePassingBase as C

        return C, "forward"

    def mean():
        from chemprop.nn.agg import MeanAggregation as C

        return C, "forward"

    def sm():
        from chemprop.nn.agg import SumAggregation as C

        return C, "forward"

    return [
        _Patch("BondMessagePassing.message", bond, _bond_message),
        _Patch("AtomMessagePassing.message", atom, _atom_message),
        _Patch("MessagePassing.forward", fwd, _mp_forward),
        _Patch("MeanAggregation.forward", mean, _mean_forward),
        _Patch("SumAggregation.forward", sm, _sum_forward),
    ]


@dataclass
class _State:
    enabled: bool = False
    patches: list[_Patch] = field(default_factory=list)
    reason: str = "not yet enabled"


STATE = _State()


def apply(selfcheck: Callable[[list[_Patch]], str | None]) -> None:
    """Apply every patch whose target passes the gates; roll everything back if the self-check fails."""
    with _LOCK:
        if STATE.enabled:
            return  # idempotent
        v = _compat.chemprop_version()
        ok, why = _compat.version_supported(v)
        if not ok:
            STATE.reason = why
            return
        try:
            patches = _chemprop_targets()
            for p in patches:
                owner, attr = p.resolve()
                p.owner, p.attr = owner, attr
                fn = owner.__dict__.get(attr)
                if fn is None:
                    p.state = f"skipped: {owner.__name__}.{attr} not defined on the class itself"
                    continue
                fp = _compat.fingerprint(fn)
                if fp is None:
                    p.state = "skipped: source unavailable, cannot verify"
                elif fp not in _compat.FINGERPRINTS[p.name]:
                    p.state = f"skipped: unrecognised source fingerprint {fp}"
                else:
                    p.original, p.replacement = fn, p.factory(fn)
                    setattr(owner, attr, p.replacement)
                    p.state = "active"
        except Exception as exc:  # noqa: BLE001 - never break the user's import
            _rollback(patches if "patches" in locals() else [])
            STATE.reason = f"patching failed: {exc!r}"
            return
        STATE.patches = patches
        active = [p for p in patches if p.state == "active"]
        if not active:
            STATE.reason = "no patch target matched a verified fingerprint"
            return
        problem = None
        try:
            problem = selfcheck(active)
        except Exception as exc:  # noqa: BLE001
            problem = f"self-check raised {exc!r}"
        if problem:
            _rollback(patches)
            for p in patches:
                if p.state == "active":
                    p.state = "skipped: self-check failed"
            STATE.reason = f"self-check failed: {problem}"
            return
        STATE.enabled, STATE.reason = True, "ok"


def _rollback(patches: list[_Patch]) -> None:
    for p in patches:
        if p.state == "active" and p.owner is not None and p.original is not None:
            if p.owner.__dict__.get(p.attr) is p.replacement:
                setattr(p.owner, p.attr, p.original)
            p.state = "inactive"


def revert() -> list[str]:
    """Restore original methods. Returns warnings for patches that someone else has re-wrapped."""
    with _LOCK:
        warnings = []
        for p in STATE.patches:
            if p.state != "active":
                continue
            if p.owner.__dict__.get(p.attr) is p.replacement:
                setattr(p.owner, p.attr, p.original)
                p.state = "inactive"
            else:
                warnings.append(f"{p.name} was re-wrapped by other code; left in place")
        STATE.enabled = False
        STATE.reason = "disabled"
        return warnings
