"""Kernel-level tests: native vs pure vs Chemprop's original formulation."""

import pytest
import torch

from chemprop_accel import _native, _ops

native = pytest.mark.skipif(not _native.AVAILABLE, reason="native extension not available")


def reference(H, dst, src, rev, n):
    """Chemprop's original (scatter_reduce_) formulation."""
    idx = dst.unsqueeze(1).repeat(1, H.shape[1])
    A = torch.zeros(n, H.shape[1], dtype=H.dtype).scatter_reduce_(
        0, idx, H, reduce="sum", include_self=False
    )
    out = A[src]
    return out - H[rev] if rev is not None else out


def random_graph(n_atoms=40, n_bonds=55, seed=0, isolated=3):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, n_atoms - isolated, (n_bonds,), generator=g)
    b = torch.randint(0, n_atoms - isolated, (n_bonds,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    src = torch.stack([a, b], 1).flatten()  # edges (a->b),(b->a) interleaved
    dst = torch.stack([b, a], 1).flatten()
    rev = torch.arange(src.numel()).view(-1, 2).flip(1).flatten()
    return dst, src, rev, n_atoms


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("d", [1, 7, 128, 300])
@native
def test_edge_message_matches_reference(variant, dtype, d):
    dst, src, rev, n = random_graph()
    H = torch.randn(src.numel(), d, dtype=dtype)
    got = torch.ops.chemprop_accel.edge_message(H, dst, src, rev, n, variant)
    tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(got, reference(H, dst, src, rev, n), **tol)


@pytest.mark.parametrize("variant", [0, 1, 2])
@native
def test_no_rev(variant):
    dst, src, _, n = random_graph(seed=1)
    H = torch.randn(src.numel(), 33, dtype=torch.float64)
    torch.testing.assert_close(
        torch.ops.chemprop_accel.edge_message(
            H, dst, src, torch.empty(0, dtype=torch.long), n, variant
        ),
        reference(H, dst, src, None, n),
    )


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("with_rev", [True, False])
@native
def test_gradcheck_double(variant, with_rev):
    dst, src, rev, n = random_graph(n_atoms=12, n_bonds=14, seed=2)
    H = torch.randn(src.numel(), 5, dtype=torch.double, requires_grad=True)
    r = rev if with_rev else torch.empty(0, dtype=torch.long)
    assert torch.autograd.gradcheck(
        lambda h: torch.ops.chemprop_accel.edge_message(h, dst, src, r, n, variant), (H,)
    )


@native
def test_backward_with_non_involution_rev():
    """Repeated / non-involutive reverse indices exercise the general backward path."""
    dst, src, _, n = random_graph(n_atoms=10, n_bonds=12, seed=3)
    E = src.numel()
    rev = torch.randint(0, E, (E,))  # arbitrary, with repeats
    H = torch.randn(E, 6, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda h: torch.ops.chemprop_accel.edge_message(h, dst, src, rev, n, 0), (H,)
    )


@native
def test_grad_matches_reference_autograd():
    dst, src, rev, n = random_graph(seed=4)
    H1 = torch.randn(src.numel(), 64, dtype=torch.double, requires_grad=True)
    H2 = H1.detach().clone().requires_grad_()
    w = torch.randn(src.numel(), 64, dtype=torch.double)
    (torch.ops.chemprop_accel.edge_message(H1, dst, src, rev, n, 0) * w).sum().backward()
    (reference(H2, dst, src, rev, n) * w).sum().backward()
    torch.testing.assert_close(H1.grad, H2.grad, atol=1e-12, rtol=1e-12)


@native
def test_empty_graph_and_isolated_atoms():
    z = torch.empty(0, 8)
    e = torch.empty(0, dtype=torch.long)
    for v in (0, 1, 2):
        assert torch.ops.chemprop_accel.edge_message(z, e, e, e, 5, v).shape == (0, 8)


@native
@pytest.mark.parametrize("which", ["dst", "src", "rev"])
def test_out_of_range_indices_raise(which):
    H = torch.randn(4, 3)
    bad = torch.tensor([0, 1, 2, 9])
    ok = torch.tensor([0, 1, 2, 3])
    empty = torch.empty(0, dtype=torch.long)
    args = {"dst": (bad, ok, empty), "src": (ok, bad, empty), "rev": (ok, ok, bad)}[which]
    with pytest.raises(RuntimeError, match="out of range"):
        torch.ops.chemprop_accel.edge_message(H, *args, 4, 0)


@native
def test_noncontiguous_and_wrong_dtype_inputs():
    dst, src, rev, n = random_graph(seed=5)
    Hbig = torch.randn(src.numel(), 40)
    H = Hbig[:, ::2]  # non-contiguous view
    torch.testing.assert_close(
        torch.ops.chemprop_accel.edge_message(H, dst, src, rev, n, 0),
        reference(H, dst, src, rev, n),
        atol=1e-5,
        rtol=1e-5,
    )
    with pytest.raises(RuntimeError):
        torch.ops.chemprop_accel.edge_message(H, dst.int(), src, rev, n, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_ops_match_reference_including_pure_fallback(dtype, monkeypatch):
    dst, src, rev, n = random_graph(seed=6)
    H = torch.randn(src.numel(), 50, dtype=dtype, requires_grad=True)
    want = reference(H, dst, src, rev, n)
    tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(_ops.edge_message(H, dst, src, rev, n), want, **tol)
    monkeypatch.setattr(_native, "AVAILABLE", False)  # force the pure-PyTorch path
    got = _ops.edge_message(H, dst, src, rev, n)
    torch.testing.assert_close(got, want, **tol)
    got.sum().backward()  # autograd through the fallback works
    assert H.grad is not None


def test_bfloat16_uses_fallback():
    dst, src, rev, n = random_graph(seed=7)
    H = torch.randn(src.numel(), 16).bfloat16()
    assert not _ops.use_native(H)
    assert _ops.edge_message(H, dst, src, rev, n).dtype == torch.bfloat16


@pytest.mark.parametrize("variant", [0, 1, 2])
@native
def test_threads_do_not_change_result(variant):
    dst, src, rev, n = random_graph(n_atoms=900, n_bonds=1200, seed=8)
    H = torch.randn(src.numel(), 300)
    prev = torch.get_num_threads()
    try:
        outs = []
        for t in (1, 2, 4):
            torch.set_num_threads(t)
            outs.append(torch.ops.chemprop_accel.edge_message(H, dst, src, rev, n, variant))
    finally:
        torch.set_num_threads(prev)
    # every variant gives each output element to exactly one thread, summed in a fixed order
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


# --- higher-order autograd (needed by Hessian tools such as scbirlab/duvida) -----------------
def _small_graph():
    src = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    dst = torch.tensor([1, 0, 2, 1, 3, 2, 4, 3, 5, 4])
    rev = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8])
    return dst, src, rev, 6


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("use_rev", [True, False])
def test_gradgradcheck(variant, use_rev):
    dst, src, rev, n = _small_graph()
    H = torch.randn(10, 4, dtype=torch.double, requires_grad=True)
    _ops.set_variant(variant)
    try:
        f = lambda H: _ops.edge_message(H, dst, src, rev if use_rev else None, n)  # noqa: E731
        assert torch.autograd.gradcheck(f, (H,))
        assert torch.autograd.gradgradcheck(f, (H,))
    finally:
        _ops.set_variant(2)


def test_gradgradcheck_non_involution_rev():
    dst, src, _, n = _small_graph()
    rev = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])  # a cycle, not an involution
    H = torch.randn(10, 3, dtype=torch.double, requires_grad=True)
    f = lambda H: _ops.edge_message(H, dst, src, rev, n)  # noqa: E731
    assert torch.autograd.gradgradcheck(f, (H,))


def test_torch_func_transforms_match_pure_path():
    from torch.func import hessian, jacrev, jvp, vmap

    dst, src, rev, n = _small_graph()
    H = torch.randn(10, 4, dtype=torch.double)
    loss = lambda H: (torch.tanh(_ops.edge_message(H, dst, src, rev, n)) ** 2).sum()  # noqa: E731
    Hf = hessian(loss)(H)
    Hp = torch.autograd.functional.hessian(loss, H)  # plain double backward (native path)
    torch.testing.assert_close(Hf, Hp)
    vs = torch.randn(3, 10, 4, dtype=torch.double)
    hv = vmap(lambda v: jvp(jacrev(loss), (H,), (v,))[1])(vs)  # duvida's HVP pattern
    torch.testing.assert_close(hv, torch.einsum("abcd,ncd->nab", Hf, vs))
