"""Original vs accelerated Chemprop: forward, gradients, training, checkpoints."""

import copy

import pytest
import torch
from chemprop import models, nn

import chemprop_accel as ca
from conftest import accel, make_batches, make_bmg

TOL64 = dict(atol=1e-9, rtol=1e-9)
TOL32 = dict(atol=2e-5, rtol=2e-5)


def build_mp(kind, **kw):
    cls = nn.BondMessagePassing if kind == "bond" else nn.AtomMessagePassing
    return cls(d_h=32, **kw)


def run(mp, bmg, V_d=None, on=True):
    mp = copy.deepcopy(mp)
    with accel(on):
        out = mp(bmg, V_d)
        w = torch.linspace(-1, 1, out.numel()).view_as(out).to(out.dtype)
        (out * w).sum().backward()
    return out.detach(), {k: p.grad.clone() for k, p in mp.named_parameters()}


@pytest.mark.parametrize("kind", ["bond", "atom"])
@pytest.mark.parametrize("depth", [2, 3, 5])
@pytest.mark.parametrize("undirected", [False, True])
@pytest.mark.parametrize("activation", ["relu", "leakyrelu", "tanh", "elu", "prelu"])
def test_message_passing_forward_and_param_grads_float64(kind, depth, undirected, activation):
    torch.manual_seed(1)
    mp = build_mp(kind, depth=depth, undirected=undirected, activation=activation).double()
    bmg = make_bmg(dtype=torch.float64)
    o1, g1 = run(mp, bmg, on=False)
    o2, g2 = run(mp, bmg, on=True)
    torch.testing.assert_close(o2, o1, **TOL64)
    for k in g1:
        torch.testing.assert_close(g2[k], g1[k], **TOL64, msg=k)


@pytest.mark.parametrize("kind", ["bond", "atom"])
def test_float32_within_reduction_order_tolerance(kind):
    torch.manual_seed(2)
    mp = build_mp(kind, depth=3)
    bmg = make_bmg()
    o1, g1 = run(mp, bmg, on=False)
    o2, g2 = run(mp, bmg, on=True)
    torch.testing.assert_close(o2, o1, **TOL32)
    for k in g1:
        torch.testing.assert_close(g2[k], g1[k], atol=1e-4, rtol=1e-4, msg=k)


def test_input_gradients_for_graph_features():
    """Gradients flow to atom and bond features (V, E) identically."""
    torch.manual_seed(3)
    mp = build_mp("bond", depth=3).double()
    res = []
    for on in (False, True):
        bmg = make_bmg(dtype=torch.float64)
        bmg.V.requires_grad_(True)
        bmg.E.requires_grad_(True)
        with accel(on):
            mp(bmg).square().sum().backward()
        res.append((bmg.V.grad.clone(), bmg.E.grad.clone()))
    for a, b in zip(*res):
        torch.testing.assert_close(a, b, **TOL64)


def test_dropout_train_mode_same_seed_matches():
    mp = build_mp("bond", depth=3, dropout=0.3).double()
    bmg = make_bmg(dtype=torch.float64)
    outs = []
    for on in (False, True):
        torch.manual_seed(11)
        with accel(on):
            outs.append(mp(bmg).detach())
    torch.testing.assert_close(outs[0], outs[1], **TOL64)


def test_vertex_descriptors():
    torch.manual_seed(4)
    mp = build_mp("bond", d_vd=6).double()
    bmg = make_bmg(dtype=torch.float64)
    V_d = torch.randn(bmg.V.shape[0], 6, dtype=torch.double)
    o1, g1 = run(mp, bmg, V_d, on=False)
    o2, g2 = run(mp, bmg, V_d, on=True)
    torch.testing.assert_close(o2, o1, **TOL64)
    for k in g1:
        torch.testing.assert_close(g2[k], g1[k], **TOL64)


@pytest.mark.parametrize("agg", ["mean", "sum", "norm"])
def test_aggregations(agg):
    cls = {"mean": nn.MeanAggregation, "sum": nn.SumAggregation, "norm": nn.NormAggregation}[agg]
    bmg = make_bmg(dtype=torch.float64)
    H = torch.randn(bmg.V.shape[0], 9, dtype=torch.double, requires_grad=True)
    outs = []
    for on in (False, True):
        H.grad = None
        with accel(on):
            o = cls()(H, bmg.batch)
            o.square().sum().backward()
        outs.append((o.detach(), H.grad.clone()))
    for a, b in zip(*outs):
        torch.testing.assert_close(a, b, **TOL64)


def _mpnn(kind="bond", seed=0, **kw):
    torch.manual_seed(seed)
    return models.MPNN(
        build_mp(kind, **kw), nn.MeanAggregation(), nn.RegressionFFN(input_dim=32, hidden_dim=16)
    )


def _train(model, batches, steps, on):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    with accel(on):
        for i in range(steps):
            b = batches[i % len(batches)]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(b.bmg), b.Y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return losses


@pytest.mark.parametrize("kind", ["bond", "atom"])
def test_training_trajectories_match_and_converge(kind):
    batches = make_batches()
    m1, m2 = _mpnn(kind), _mpnn(kind)
    l1 = _train(m1, batches, 40, on=False)
    l2 = _train(m2, batches, 40, on=True)
    assert l2[-1] < l2[0] * 0.8, "accelerated training should converge"
    torch.testing.assert_close(torch.tensor(l2), torch.tensor(l1), atol=2e-4, rtol=2e-3)
    for (k, p1), (_, p2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        torch.testing.assert_close(p2, p1, atol=5e-4, rtol=5e-3, msg=k)


def test_state_dict_keys_and_init_unchanged():
    with accel(False):
        a = _mpnn()
    with accel(True):
        b = _mpnn()
    assert list(a.state_dict()) == list(b.state_dict())
    for k in a.state_dict():
        assert torch.equal(a.state_dict()[k], b.state_dict()[k])


def test_model_built_before_enable_is_accelerated_too():
    ca.disable()
    m = _mpnn()
    b = make_batches(1)[0]
    ref = m(b.bmg).detach()
    ca.enable()
    torch.testing.assert_close(m(b.bmg).detach(), ref, **TOL32)
    assert ca.status().enabled


def test_checkpoints_interchangeable(tmp_path):
    batches = make_batches()
    m = _mpnn()
    _train(m, batches, 10, on=True)
    path = tmp_path / "accel.pt"
    torch.save(m.state_dict(), path)
    plain = _mpnn(seed=99)
    plain.load_state_dict(torch.load(path))
    m.eval(), plain.eval()
    b = batches[0]
    with accel(False):
        p_plain = plain(b.bmg)
    with accel(True):
        p_accel = m(b.bmg)
    torch.testing.assert_close(p_plain, p_accel, **TOL32)
    # and the reverse: checkpoint trained *without* acceleration, loaded and run with it
    m2 = _mpnn()
    _train(m2, batches, 10, on=False)
    torch.save(m2.state_dict(), path)
    m3 = _mpnn(seed=5)
    m3.load_state_dict(torch.load(path))
    m2.eval(), m3.eval()
    with accel(False):
        a = m2(b.bmg)
    with accel(True):
        c = m3(b.bmg)
    torch.testing.assert_close(a, c, **TOL32)


def test_freeze_unfreeze_parameters():
    m = _mpnn()
    b = make_batches(1)[0]
    for p in m.message_passing.parameters():
        p.requires_grad_(False)
    with accel(True):
        torch.nn.functional.mse_loss(m(b.bmg), b.Y).backward()
    assert all(p.grad is None for p in m.message_passing.parameters())
    assert all(p.grad is not None for p in m.predictor.parameters())
    for p in m.message_passing.parameters():
        p.requires_grad_(True)
    with accel(True):
        torch.nn.functional.mse_loss(m(b.bmg), b.Y).backward()
    assert all(p.grad is not None for p in m.message_passing.parameters())


def test_unsupported_dtype_falls_back_to_original():
    mp = build_mp("bond")
    bmg = make_bmg()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = mp(bmg)  # H is bf16 inside -> original code path
    assert out.dtype in (torch.bfloat16, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_uses_original_path_and_matches():
    mp = build_mp("bond").cuda()
    bmg = make_bmg()
    bmg.to("cuda")
    with accel(False):
        a = mp(bmg)
    with accel(True):
        b = mp(bmg)
    torch.testing.assert_close(a, b)


@pytest.mark.slow
def test_lightning_fit_matches():
    import lightning.pytorch as pl

    batches = make_batches(4)
    ds_loader = [tuple(b) for b in batches]

    class DL(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds_loader)

        def __getitem__(self, i):
            return ds_loader[i]

    results = []
    for on in (False, True):
        m = _mpnn()
        with accel(on):
            tr = pl.Trainer(
                max_epochs=2,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            tr.fit(m, torch.utils.data.DataLoader(DL(), batch_size=None, shuffle=False))
        results.append(copy.deepcopy(m.state_dict()))
    for k in results[0]:
        torch.testing.assert_close(results[1][k], results[0][k], atol=1e-4, rtol=1e-3, msg=k)


# --- Hessians through a Chemprop model (the pattern used by scbirlab/duvida) -----------------
def _param_hessian_diag(model, bmg, on):
    from torch.func import functional_call, hessian

    params = {k: v.detach().double() for k, v in model.named_parameters()}
    keys = [k for k in params if k.endswith("weight") and "predictor" in k][-1:]  # last layer
    fixed = {k: v for k, v in params.items() if k not in keys}

    def loss(p):
        out = functional_call(model, {**fixed, **p}, (bmg,))
        return (out**2).sum()

    with accel(on):
        H = hessian(loss)({k: params[k] for k in keys})
    return H


def test_hessian_through_message_passing_matches_original():
    torch.manual_seed(0)
    model = (
        models.MPNN(
            build_mp("bond", activation="elu"),
            nn.NormAggregation(),
            nn.RegressionFFN(input_dim=32, hidden_dim=16),
        )
        .double()
        .eval()
    )
    bmg = make_bmg(dtype=torch.float64)
    a = _param_hessian_diag(model, bmg, on=True)
    b = _param_hessian_diag(model, bmg, on=False)
    for k in a:
        for kk in a[k]:
            torch.testing.assert_close(a[k][kk], b[k][kk], rtol=1e-8, atol=1e-10)


def test_hvp_wrt_message_passing_weights_matches_original():
    """Double backward through the fused op to an *early* (message-passing) parameter."""
    torch.manual_seed(1)
    model = (
        models.MPNN(
            build_mp("bond", activation="elu"),
            nn.MeanAggregation(),
            nn.RegressionFFN(input_dim=32, hidden_dim=16),
        )
        .double()
        .eval()
    )
    bmg = make_bmg(dtype=torch.float64)
    p = model.message_passing.W_h.weight

    def hv(on):
        with accel(on):
            loss = (model(bmg) ** 2).sum()
            (g,) = torch.autograd.grad(loss, p, create_graph=True)
            v = torch.ones_like(p)
            (h,) = torch.autograd.grad((g * v).sum(), p)
        return h

    torch.testing.assert_close(hv(True), hv(False), rtol=1e-8, atol=1e-10)
