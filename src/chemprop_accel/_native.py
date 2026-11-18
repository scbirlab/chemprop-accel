"""Loading of the optional native extension (``_C``) and registration of its autograd formulas.

The shared library is compiled against one PyTorch minor version. Loading it into a different
minor version can crash the interpreter, so the version recorded at build time is checked *before*
``torch.ops.load_library`` is called. Any failure is recorded in :data:`REASON` and the package
falls back to pure PyTorch.
"""

from __future__ import annotations

import glob
import os
import sys

import torch

AVAILABLE: bool = False
REASON: str = "not attempted"
LIBRARY_PATH: str | None = None
BUILT_AGAINST: str | None = None


def _minor(v: str) -> str:
    return ".".join(v.split("+")[0].split(".")[:2])


def _find_library() -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    patterns = ["_C.so", "_C.pyd", "_C.dylib", "_C.*.so", "_C.*.pyd"]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(here, pat)))
        if hits:
            return hits[0]
    return None


def _diff_backward(g, dst, src, rev, n, variant):
    """Backward of ``edge_message`` expressed with differentiable ops (for higher-order AD).

    The op is linear in ``H``, so its transpose is again an ``edge_message`` when ``rev`` is an
    involution (always true for Chemprop graphs); that call is itself autograd-registered, so any
    order of derivative works. Otherwise a composite of ``index_add_``/``index_select`` is used.
    """
    ops = torch.ops.chemprop_accel
    if rev.numel() == 0:
        return ops.edge_message(g, src, dst, rev, n, variant)
    idx = torch.arange(rev.numel(), device=rev.device)
    if bool((rev[rev] == idx).all()):
        return ops.edge_message(g, src, dst, rev, n, variant)
    A = g.new_zeros((n, g.shape[1])).index_add(0, src, g)
    return A.index_select(0, dst).index_add(0, rev, -g)


def _register_autograd() -> None:
    ops = torch.ops.chemprop_accel

    # -- fake (meta) kernels so the ops work under torch.compile / FakeTensor ------------------
    @torch.library.register_fake("chemprop_accel::edge_message")
    def _(H, dst, src, rev, n, variant=2):
        return torch.empty_like(H)

    @torch.library.register_fake("chemprop_accel::edge_message_backward")
    def _(g, dst, src, rev, n, variant=2):
        return torch.empty_like(g)

    # -- autograd ---------------------------------------------------------------------------
    def _edge_setup(ctx, inputs, output):
        H, dst, src, rev, n, variant = inputs
        ctx.save_for_backward(dst, src, rev)
        ctx.n = n
        ctx.variant = variant

    def _edge_bwd(ctx, grad):
        dst, src, rev = ctx.saved_tensors
        if torch.is_grad_enabled():
            # create_graph=True (double backward / Hessians): use a differentiable formulation.
            dH = _diff_backward(grad, dst, src, rev, ctx.n, ctx.variant)
        else:
            dH = ops.edge_message_backward(grad, dst, src, rev, ctx.n, ctx.variant)
        return dH, None, None, None, None, None

    torch.library.register_autograd(
        "chemprop_accel::edge_message", _edge_bwd, setup_context=_edge_setup
    )


def load() -> bool:
    """Try to load the native library once. Returns ``True`` on success."""
    global AVAILABLE, REASON, LIBRARY_PATH, BUILT_AGAINST
    if AVAILABLE:
        return True
    if os.environ.get("CHEMPROP_ACCEL_NO_NATIVE") == "1":
        REASON = "disabled by CHEMPROP_ACCEL_NO_NATIVE=1"
        return False
    path = _find_library()
    if path is None:
        REASON = "native extension not built/installed (pure-PyTorch fallback in use)"
        return False
    try:
        from . import _build_info  # generated at build time

        BUILT_AGAINST = _build_info.TORCH_BUILT_AGAINST
    except Exception:  # noqa: BLE001
        REASON = "missing _build_info; cannot verify PyTorch compatibility"
        return False
    if _minor(BUILT_AGAINST) != _minor(torch.__version__):
        REASON = (
            f"native extension built against torch {BUILT_AGAINST} but torch "
            f"{torch.__version__} is installed; reinstall a matching wheel or build from source"
        )
        return False
    try:
        torch.ops.load_library(path)
        _register_autograd()
    except Exception as exc:  # noqa: BLE001
        REASON = f"failed to load native extension: {exc!r}"
        return False
    AVAILABLE, REASON, LIBRARY_PATH = True, "ok", path
    return True


def python_tag() -> str:  # pragma: no cover - debugging helper
    return f"py{sys.version_info.major}{sys.version_info.minor}"
