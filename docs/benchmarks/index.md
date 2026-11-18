# Benchmarks

```{admonition} Read this first
:class: warning
Speed-ups depend on the CPU, core count, batch size, message-passing type and thread count. Numbers measured on
one machine do not transfer to another. Everything below is measured, reproducible with the scripts in this
repository, and labelled with the hardware it came from. **Run the benchmark on your own machine** before drawing
conclusions.
```

## Run it yourself

```bash
pip install chemprop-accel chemprop       # (or your editable install)
python -m chemprop_accel.bench            # quick preset, ~2-4 minutes
python -m chemprop_accel.bench --preset standard --rounds 3   # more configurations, median of 3 interleaved rounds
```

It writes `chemprop_accel_bench/results-<host>.json` and a self-contained `report-<host>.html` (open it in a browser).
The exit status is non-zero if an accelerated run's predictions differ from the original's beyond tolerance, so
please report that.

What it does, and what it deliberately does not do:

- Every configuration runs in a **fresh subprocess** with its own timeout; a global `--max-minutes` budget (default 15)
  stops the run. Nothing is installed or modified.
- **No network access, downloads or external data**: a built-in pool of ~55 drug-like molecules is featurised with
  Chemprop's own featuriser and batched with Chemprop's own collate function, giving realistic, heterogeneous graphs.
- Writes only into `--output`. `--dry-run` prints the plan and runs nothing. The `full` preset (the complete
  batch/hidden/depth/thread grid) needs `--yes`.
- **GPU**: with `--device auto` (default) and a CUDA device present, GPU reference runs are added. `chemprop_accel`
  deliberately does not change GPU execution, so the GPU rows are a *comparison point* ("is my CPU+accel competitive with
  my GPU for this batch size?"). Batches are small enough to need well under 1 GB of device memory.
- Reports forward-only time, full training step (forward + loss + backward + Adam), molecules/s, peak process
  memory, cold-start (first step, including `torch.compile` time) and, optionally (`--e2e`, `--lightning`), a complete
  epoch including featurisation, DataLoader and Lightning.
- Each accelerated run is paired with an identical eager run (same seed, data and optimiser) to produce a
  **predictive-equivalence** table.

| Preset | Configurations | Typical time |
| --- | --- | --- |
| `quick` (default) | bond MPNN, batch 32/128, all threads, original vs accelerated, equivalence + epoch | 2-4 min |
| `standard` | bond+atom, batch 8/64/256, 1 and all threads, + `torch.compile` modes, epoch | 15-40 min |
| `full` | the grid from the project spec: batch 1-256, d_h 128/300/600, depth 3/5, threads 1/2/4/8/all, fused Adam, compile | hours |

Modes (`--modes`): `eager` (original Chemprop), `accel`, `accel_pure` (patches active, native kernels disabled),
`accel_v0/v1` (alternative native algorithms), `eager_fused`/`accel_fused` (fused Adam), `eager_compile`/`accel_compile`
(`torch.compile`).

## Micro-benchmark of the fused op

```bash
python benchmarks/op_microbench.py --threads 1 2      # original vs pure fallback vs native variants 0/1/2, fwd and fwd+bwd
```

## Continuous benchmark report

Every push to `main` (and weekly) CI runs the `ci` preset on Linux x86-64, Linux arm64, macOS arm64 and Windows with
Python 3.11 and 3.13, checks predictive equivalence, and publishes a combined report.
```{raw} html
<p><a href="report/index.html"><b>Open the latest CI benchmark report</b></a> (published by the docs workflow; absent in local builds).</p>
```

GitHub-hosted runners have 2-4 shared vCPUs and are noisy; treat that report as
a regression and portability signal, not as headline performance.

## Results from the development machine

```{include} results_generated.md
```

```{raw} html
<p><a href="../_static/report-sandbox-2vcpu.html">Full HTML report for the development machine</a> (all tables, ranges across rounds, equivalence).</p>
```
