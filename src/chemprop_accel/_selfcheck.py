"""Cheap numerical self-check run when patches are enabled (tens of milliseconds)."""

from __future__ import annotations

import torch

_SMILES = ["CCO", "c1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C", "[Na+].[Cl-]", "C1CC1"]
_ATOL = 1e-9


def _batch():
    from chemprop.data import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    f = SimpleMoleculeMolGraphFeaturizer()
    bmg = BatchMolGraph([f(Chem.MolFromSmiles(s)) for s in _SMILES])
    bmg.V, bmg.E = bmg.V.double(), bmg.E.double()
    return bmg


def _close(a, b):
    return a.shape == b.shape and torch.allclose(a, b, atol=_ATOL, rtol=_ATOL)


def run(active) -> str | None:
    """Return ``None`` if all active replacements agree with the originals, else a message."""
    from chemprop import nn

    by_name = {p.name: p for p in active}
    with torch.random.fork_rng(devices=[]):  # never disturb the user's global RNG state
        torch.manual_seed(0)
        bmg = _batch()
        d = 16
        for cls, key in (
            (nn.BondMessagePassing, "BondMessagePassing.message"),
            (nn.AtomMessagePassing, "AtomMessagePassing.message"),
        ):
            mp = cls(d_h=d, depth=3).double()
            if key in by_name:
                p = by_name[key]
                H = torch.randn(bmg.E.shape[0], d, dtype=torch.double, requires_grad=True)
                new, ref = p.replacement(mp, H, bmg), p.original(mp, H, bmg)
                if not _close(new, ref):
                    return f"{key}: forward mismatch"
                w = torch.randn_like(new)
                (g_new,) = torch.autograd.grad((new * w).sum(), H)
                (g_ref,) = torch.autograd.grad((ref * w).sum(), H)
                if not _close(g_new, g_ref):
                    return f"{key}: gradient mismatch"
            if "MessagePassing.forward" in by_name:
                p = by_name["MessagePassing.forward"]
                new = p.replacement(mp, bmg)
                ref = p.original(mp, bmg)
                if not _close(new, ref):
                    return f"{cls.__name__}.forward mismatch"
        for key, agg in (
            ("MeanAggregation.forward", nn.MeanAggregation()),
            ("SumAggregation.forward", nn.SumAggregation()),
        ):
            if key in by_name:
                p = by_name[key]
                H = torch.randn(bmg.V.shape[0], d, dtype=torch.double)
                if not _close(p.replacement(agg, H, bmg.batch), p.original(agg, H, bmg.batch)):
                    return f"{key}: mismatch"
    return None
