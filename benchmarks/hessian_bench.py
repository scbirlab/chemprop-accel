"""Second-order benchmark: Hessian-vector products through a Chemprop MPNN, accel on vs off.

Measures (a) plain double-backward HVPs (``create_graph=True``; uses the native kernel) and
(b) Duvida-style ``torch.func`` HVPs (``jvp`` of ``jacrev`` under ``vmap``; uses the pure path),
and checks the results agree with the original Chemprop code. CPU only, built-in molecules only.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch
from chemprop import models, nn
from torch.func import functional_call, jacrev, jvp, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import chemprop_accel as ca  # noqa: E402
from conftest import SMILES, make_bmg  # noqa: E402

H, DEPTH, NV, REPEATS = 300, 3, 8, 5


def build(seed=0):
    torch.manual_seed(seed)
    mp = nn.BondMessagePassing(d_h=H, depth=DEPTH, activation="elu")
    return (
        models.MPNN(mp, nn.NormAggregation(), nn.RegressionFFN(input_dim=H, hidden_dim=16))
        .double()
        .eval()
    )


def timeit(fn):
    fn()
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    model = build()
    bmg = make_bmg(SMILES * 6, dtype=torch.float64)
    params = {k: v.detach() for k, v in model.named_parameters()}
    key = "message_passing.W_h.weight"
    vs = torch.randn(NV, *params[key].shape, dtype=torch.float64)

    def loss(p):
        return (functional_call(model, {**params, key: p}, (bmg,)) ** 2).sum()

    def hvp_autograd():
        p = params[key].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(loss(p), p, create_graph=True)
        return torch.stack(
            [torch.autograd.grad((g * v).sum(), p, retain_graph=True)[0] for v in vs]
        )

    def hvp_func():
        return vmap(lambda v: jvp(jacrev(loss), (params[key],), (v,))[1])(vs)

    res = {}
    for name, fn in (
        ("autograd double-backward (native)", hvp_autograd),
        ("torch.func jvp(jacrev) + vmap (pure path)", hvp_func),
    ):
        out, tm = {}, {}
        for on in (False, True):
            (ca.enable if on else ca.disable)()
            out[on] = fn()
            tm[on] = timeit(fn)
        ca.enable()
        err = (out[True] - out[False]).abs().max().item()
        scale = out[False].abs().max().item()
        res[name] = (tm[False], tm[True], err, scale)
    print(
        f"{NV} HVPs, {len(SMILES) * 6} molecules, d_h={H}, depth={DEPTH}, float64, "
        f"threads={torch.get_num_threads()}"
    )
    for k, (a, b, e, s) in res.items():
        print(
            f"{k:45s} original {a * 1e3:8.1f} ms  accel {b * 1e3:8.1f} ms  "
            f"speedup {a / b:4.2f}x  max|diff| {e:.2e} (scale {s:.2e})"
        )


if __name__ == "__main__":
    main()
