"""Graph operations used by the Chemprop patches.

Each public function dispatches to the native CPU kernel when it is available and applicable,
and otherwise to an equivalent pure-PyTorch implementation (which still avoids the expanded
``E x d`` index tensors that Chemprop's ``scatter_reduce_`` formulation allocates).
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from . import _native

#: Algorithm for the atom-level reduction in the native kernels (see docs/optimisations).
#: 0 = column-tiled scatter, 1 = CSR + atom-parallel, 2 = direct per-edge neighbour sum (default).
VARIANT: int = int(os.environ.get("CHEMPROP_ACCEL_VARIANT", "2"))

_EMPTY_LONG = torch.empty(0, dtype=torch.long)
_NATIVE_DTYPES = (torch.float32, torch.float64)


def set_variant(variant: int) -> None:
    """Select the native algorithm variant (0, 1 or 2). Intended for benchmarking."""
    global VARIANT
    if variant not in (0, 1, 2):
        raise ValueError("variant must be 0, 1 or 2")
    VARIANT = variant


def _transform_active(tensors) -> bool:
    """True under torch.func transforms or forward-mode AD.

    The native op registers reverse-mode autograd only (any order). ``vmap``/``jvp``/``jacrev``
    etc. use the equivalent pure-PyTorch path, which supports every transform.
    """
    try:
        if torch._C._are_functorch_transforms_active():
            return True
    except AttributeError:  # pragma: no cover
        pass
    try:
        import torch.autograd.forward_ad as fwAD

        return any(fwAD.unpack_dual(t).tangent is not None for t in tensors)
    except Exception:  # noqa: BLE001
        return False


def use_native(*tensors: Tensor) -> bool:
    """True when the native kernel can handle these tensors (CPU, float32/64)."""
    if not _native.AVAILABLE:
        return False
    if _transform_active(tensors):
        return False
    x = tensors[0]
    return x.device.type == "cpu" and x.dtype in _NATIVE_DTYPES and x.dim() == 2


def edge_message(H: Tensor, dst: Tensor, src: Tensor, rev: Tensor | None, n: int) -> Tensor:
    r"""``M[e] = sum_{k : dst[k] == src[e]} H[k] - H[rev[e]]`` (``rev`` optional).

    For Chemprop's directed message passing, ``dst = edge_index[1]``, ``src = edge_index[0]`` and
    ``rev = rev_edge_index``.
    """
    if use_native(H):
        r = _EMPTY_LONG if rev is None else rev
        return torch.ops.chemprop_accel.edge_message(H, dst, src, r, n, VARIANT)
    A = H.new_zeros((n, H.shape[1])).index_add_(0, dst, H)
    out = A.index_select(0, src)
    if rev is not None:
        out = out - H.index_select(0, rev)
    return out


def scatter_sum(X: Tensor, index: Tensor, n: int) -> Tensor:
    """Row-wise scatter-add of ``X`` into ``n`` buckets given by ``index``.

    Uses ``index_add_`` (no expanded ``R x d`` index tensor). A native kernel for this was measured
    and found no faster than ``index_add_``, so it was removed (see docs/optimisations.md).
    """
    return X.new_zeros((n, X.shape[1])).index_add_(0, index, X)
