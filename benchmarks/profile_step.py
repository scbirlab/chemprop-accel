"""Profile one Chemprop training step with torch.profiler and print the top operators as a Markdown table.

python benchmarks/profile_step.py [--accel/--no-accel] [--batch 64] [--hidden 300] [--threads 2]
"""

import argparse
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--accel", action=argparse.BooleanOptionalAction, default=True)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--hidden", type=int, default=300)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--kind", choices=["bond", "atom"], default="bond")
ap.add_argument("--top", type=int, default=9)
a = ap.parse_args()
if not a.accel:
    os.environ["CHEMPROP_ACCEL_DISABLE"] = "1"

import torch  # noqa: E402
from chemprop import data, featurizers, models, nn  # noqa: E402
from chemprop.data import collate_batch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import chemprop_accel  # noqa: E402,F401
from chemprop_accel.bench._molecules import SMILES  # noqa: E402

if a.threads:
    torch.set_num_threads(a.threads)
rng = random.Random(0)
f = featurizers.SimpleMoleculeMolGraphFeaturizer()
ds = data.MoleculeDataset(
    [data.MoleculeDatapoint.from_smi(rng.choice(SMILES), [rng.random()]) for _ in range(a.batch)], f
)
b = collate_batch([ds[i] for i in range(a.batch)])
torch.manual_seed(0)
mp = (nn.BondMessagePassing if a.kind == "bond" else nn.AtomMessagePassing)(d_h=a.hidden)
m = models.MPNN(
    mp, nn.MeanAggregation(), nn.RegressionFFN(input_dim=a.hidden, hidden_dim=a.hidden, n_layers=1)
)
opt = torch.optim.Adam(m.parameters())


def step():
    opt.zero_grad()
    torch.nn.functional.mse_loss(m(b.bmg), b.Y).backward()
    opt.step()


for _ in range(5):
    step()
with profile(activities=[ProfilerActivity.CPU]) as p:
    for _ in range(20):
        step()
ev = sorted(p.key_averages(), key=lambda e: -e.self_cpu_time_total)
tot = sum(e.self_cpu_time_total for e in ev)
print(
    f"accel={'on' if a.accel else 'off'} · {a.kind} · B={a.batch} · d_h={a.hidden} · threads={torch.get_num_threads()} · "
    f"total self-CPU {tot / 20 / 1e3:.1f} ms/step (profiler overhead included)\n"
)
print("| Operator | Self CPU (ms/step) | Share |\n| --- | ---: | ---: |")
for e in ev[: a.top]:
    print(
        f"| `{e.key}` | {e.self_cpu_time_total / 20 / 1e3:.2f} | {100 * e.self_cpu_time_total / tot:.0f}% |"
    )
