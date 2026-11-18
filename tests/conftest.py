from __future__ import annotations

import contextlib
import random

import pytest
import torch

chemprop = pytest.importorskip("chemprop")
import chemprop_accel as ca  # noqa: E402

SMILES = [
    "CCO",
    "c1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "C",
    "[Na+].[Cl-]",
    "C1CC1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2c(c1)ccc1ccccc12",
    "O",
    "CN1CCN(CC1)c1ccc(cc1)C(=O)Nc1ccc(C)c(Nc2nccc(n2)-c2cccnc2)c1",
]


@contextlib.contextmanager
def accel(on: bool):
    """Temporarily run with the accelerator on/off, restoring the previous state afterwards."""
    was = ca.status().enabled
    (ca.enable if on else ca.disable)()
    try:
        yield
    finally:
        (ca.enable if was else ca.disable)()


@pytest.fixture(autouse=True)
def _ensure_enabled():
    ca.enable()
    yield
    ca.enable()


def make_bmg(smiles=SMILES, dtype=torch.float32, seed=0):
    from chemprop.data import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem

    f = SimpleMoleculeMolGraphFeaturizer()
    bmg = BatchMolGraph([f(Chem.MolFromSmiles(s)) for s in smiles])
    bmg.V, bmg.E = bmg.V.to(dtype), bmg.E.to(dtype)
    return bmg


def make_batches(n_batches=3, batch=8, seed=0):
    from chemprop import data, featurizers
    from chemprop.data import collate_batch

    rng = random.Random(seed)
    f = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dps = [
        data.MoleculeDatapoint.from_smi(rng.choice(SMILES), [rng.random()])
        for _ in range(n_batches * batch)
    ]
    ds = data.MoleculeDataset(dps, f)
    return [
        collate_batch([ds[j] for j in range(i * batch, (i + 1) * batch)]) for i in range(n_batches)
    ]


@pytest.fixture
def bmg64():
    return make_bmg(dtype=torch.float64)


@pytest.fixture
def batches():
    return make_batches()
