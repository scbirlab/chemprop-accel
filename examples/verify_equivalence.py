"""Train the same Chemprop model with and without chemprop_accel and verify equivalence.

    python examples/verify_equivalence.py

Both runs use the same seed, data and optimiser. The accelerated model should track the original to within
floating-point summation-order noise; checkpoints are interchangeable.
"""

import random
import time

import torch
from chemprop import data, featurizers, models, nn
from chemprop.data import collate_batch

import chemprop_accel as ca

SMILES = [
    "CCO",
    "c1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2c(c1)ccc1ccccc12",
    "CC(=O)Nc1ccc(O)cc1",
    "C1CC1",
]
rng = random.Random(0)
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
dps = [data.MoleculeDatapoint.from_smi(rng.choice(SMILES), [rng.random()]) for _ in range(256)]
ds = data.MoleculeDataset(dps, feat)
batches = [collate_batch([ds[j] for j in range(i, i + 32)]) for i in range(0, 256, 32)]


def build():
    torch.manual_seed(0)
    return models.MPNN(
        nn.BondMessagePassing(d_h=300), nn.MeanAggregation(), nn.RegressionFFN(input_dim=300)
    )


def train(model, steps=40):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0, losses = time.perf_counter(), []
    for i in range(steps):
        b = batches[i % len(batches)]
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(b.bmg), b.Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, time.perf_counter() - t0


print(ca.status(), end="\n\n")

ca.disable()
m_ref = build()
l_ref, t_ref = train(m_ref)

ca.enable()
m_acc = build()
l_acc, t_acc = train(m_acc)

m_ref.eval(), m_acc.eval()
with torch.no_grad():
    p_ref = torch.cat([m_ref(b.bmg) for b in batches])
    ca.disable()
    p_cross = torch.cat(
        [m_acc(b.bmg) for b in batches]
    )  # accelerated-trained weights, original code
    ca.enable()
    p_acc = torch.cat([m_acc(b.bmg) for b in batches])

print(f"original      : {t_ref:6.2f} s  final loss {l_ref[-1]:.6f}")
print(
    f"chemprop_accel: {t_acc:6.2f} s  final loss {l_acc[-1]:.6f}   ({t_ref / t_acc:.2f}x; one run, this machine)"
)
print(f"max |loss diff| over training   : {max(abs(a - b) for a, b in zip(l_ref, l_acc)):.2e}")
print(
    f"max |prediction diff| (ref vs accel-trained, both run accelerated/original): "
    f"{(p_ref - p_acc).abs().max():.2e} / {(p_ref - p_cross).abs().max():.2e}"
)
print(
    f"accelerated and original code agree on accel-trained weights: {(p_acc - p_cross).abs().max():.2e}"
)
assert torch.allclose(p_ref, p_acc, atol=1e-3, rtol=1e-3), "predictions diverged!"
assert list(m_ref.state_dict()) == list(m_acc.state_dict())
print("OK: equivalent within tolerance; state_dict keys identical.")
