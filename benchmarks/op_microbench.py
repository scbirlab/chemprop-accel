"""Micro-benchmark of the directed-message aggregation op in isolation.

Compares (a) Chemprop's original scatter_reduce_/gather formulation, (b) the pure-PyTorch
fallback (index_add_/index_select, no expanded index), and (c) the native kernels (variants 0-2),
for forward and forward+backward, over several graph sizes. Interleaved repeats, medians.

    python benchmarks/op_microbench.py [--threads 1 2] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import torch

from chemprop_accel import _native
from chemprop_accel.bench._molecules import SMILES


def make_graph(B, d, seed=0):
    from chemprop.data import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    rng = random.Random(seed)
    f = SimpleMoleculeMolGraphFeaturizer()
    bmg = BatchMolGraph([f(Chem.MolFromSmiles(rng.choice(SMILES))) for _ in range(B)])
    E = bmg.edge_index.shape[1]
    return bmg, torch.randn(E, d)


def original(H, bmg):
    idx = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
    M = torch.zeros(bmg.V.shape[0], H.shape[1], dtype=H.dtype).scatter_reduce_(
        0, idx, H, reduce="sum", include_self=False
    )[bmg.edge_index[0]]
    return M - H[bmg.rev_edge_index]


def impl(name):
    if name == "original":
        return original
    if name == "pure":

        def f(H, bmg):
            A = H.new_zeros((bmg.V.shape[0], H.shape[1])).index_add_(0, bmg.edge_index[1], H)
            return A.index_select(0, bmg.edge_index[0]) - H.index_select(0, bmg.rev_edge_index)

        return f
    v = int(name[-1])

    def f(H, bmg):
        return torch.ops.chemprop_accel.edge_message(
            H, bmg.edge_index[1], bmg.edge_index[0], bmg.rev_edge_index, bmg.V.shape[0], v
        )

    return f


def bench(fns, bmg, H, backward, reps):
    ts = {k: [] for k in fns}
    for r in range(reps + 5):
        for k, fn in fns.items():  # interleave to share any drift/noise equally
            h = H.clone().requires_grad_(backward)
            t0 = time.perf_counter()
            out = fn(h, bmg)
            if backward:
                out.sum().backward()
            dt = time.perf_counter() - t0
            if r >= 5:
                ts[k].append(dt)
    return {k: statistics.median(v) for k, v in ts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", nargs="+", type=int, default=[1, torch.get_num_threads()])
    ap.add_argument("--batch", nargs="+", type=int, default=[8, 64, 256])
    ap.add_argument("--hidden", nargs="+", type=int, default=[128, 300, 600])
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--json")
    a = ap.parse_args()
    names = ["original", "pure"] + (["native0", "native1", "native2"] if _native.AVAILABLE else [])
    fns = {n: impl(n) for n in names}
    rows = []
    print(
        f"{'threads':>7} {'B':>4} {'d':>4} {'E':>6} {'pass':>4} "
        + " ".join(f"{n:>9}" for n in names)
        + "   (µs; speedup of best native vs original)"
    )
    for T in sorted(set(a.threads)):
        torch.set_num_threads(T)
        for B in a.batch:
            for d in a.hidden:
                bmg, H = make_graph(B, d)
                for bw in (False, True):
                    res = bench(fns, bmg, H, bw, a.reps)
                    rows.append(
                        dict(
                            threads=T,
                            batch=B,
                            hidden=d,
                            edges=H.shape[0],
                            backward=bw,
                            us={k: v * 1e6 for k, v in res.items()},
                        )
                    )
                    best = min((v for k, v in res.items() if k.startswith("native")), default=None)
                    sp = f"{res['original'] / best:.2f}x" if best else ""
                    print(
                        f"{T:>7} {B:>4} {d:>4} {H.shape[0]:>6} {'f+b' if bw else 'fwd':>4} "
                        + " ".join(f"{res[n] * 1e6:9.0f}" for n in names)
                        + f"   {sp}"
                    )
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
