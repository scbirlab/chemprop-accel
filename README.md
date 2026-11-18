<h1 align="center">chemprop-accel</h1>
<p align="center"><b>Drop-in multicore CPU acceleration for <a href="https://github.com/chemprop/chemprop">Chemprop</a> v2.</b><br>
<code>import chemprop_accel</code> — and your existing training and inference code runs faster on CPU.</p>

<p align="center">
<a href="https://github.com/scbirlab/chemprop-accel/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/scbirlab/chemprop-accel/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://scbirlab.github.io/chemprop-accel/"><img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-blue"></a>
<a href="https://scbirlab.github.io/chemprop-accel/benchmarks/report/"><img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-live%20report-informational"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11%E2%80%933.14-blue">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

```python
import chemprop_accel  # activates automatically (CHEMPROP_ACCEL_DISABLE=1 to prevent)
import chemprop

from chemprop import nn, models

model = models.MPNN(nn.BondMessagePassing(), nn.MeanAggregation(), nn.RegressionFFN())  # unchanged
```

No new model classes. No changes to checkpoints, `state_dict` keys, initialisation, hyper-parameters or public APIs.
No fork of Chemprop or PyTorch. Models created before *or* after the import are accelerated; checkpoints trained with
or without the package are interchangeable.

## What it does

Profiling shows that on CPU a Chemprop training step is dominated not by matrix multiplication but by graph plumbing:
`scatter_reduce_` with an `E × d` expanded int64 index tensor, zero-filled intermediates, separate gathers and
subtractions, and their backward counterparts. `chemprop_accel` replaces five private methods
(directed-bond and atom message aggregation, the readout scatter, mean/sum aggregation) with

- a **fused C++ operator** `M[e] = Σ_{k: dst[k]=src[e]} H[k] − H[rev[e]]` with an analytic backward pass that reuses the same
  kernel, no expanded index, no atomics, thread-parallel with PyTorch's own pool, and run-time AVX-512/AVX2 dispatch;
- a **pure-PyTorch fallback** (`index_add_`) used when the native library is unavailable, which already removes most of
  the overhead.

Everything else, the whole dense network, stays PyTorch. See the [design notes](https://scbirlab.github.io/chemprop-accel/optimisations.html)
for what was measured, what was fused and what was rejected.

## Measured speed-up (one machine; yours will differ)

| Bond MPNN, d_h=300, depth 3 | Original: train step | chemprop_accel | Train speed-up | Forward speed-up | accel mol/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch 8, 2 threads | 9.8 ms | 5.3 ms | **1.84×** | 1.93× | 1,500 |
| batch 64, 1 thread | 64.2 ms | 29.1 ms | **2.21×** | 2.43× | 2,198 |
| batch 64, 2 threads | 43.2 ms | 19.1 ms | **2.26×** | 2.44× | 3,352 |
| batch 256, 2 threads | 168.3 ms | 65.5 ms | **2.57×** | 2.77× | 3,909 |
| *atom* MPNN, batch 64, 2 threads | 39.1 ms | 18.1 ms | **2.16×** | 2.36× | 3,530 |

*Machine: shared 2-vCPU Intel Xeon @ 2.3 GHz (AVX-512), Linux, PyTorch 2.15, Chemprop 2.3.1, Python 3.13.* Complete training steps (forward, loss, backward, Adam).
A full epoch over 1024 molecules including the DataLoader took 0.77 s → 0.34 s (2.30×); Lightning `Trainer.fit` 1 epoch: 0.81 s → 0.37 s.
Predictions after 15 training steps agree to ≤ 7e-7 (relative 6e-7), bit-identical at 1 thread.

**Where the gain comes from:** removing Chemprop's expanded-index scatter/gather pattern gives most of it (the pure-PyTorch fallback alone reached 2.21× vs 2.26× native at 2 threads and
1.93× vs 2.21× at 1 thread on this machine); the native operator is 5–17× faster than Chemprop's version at op level, but the dense matrix products then dominate the step.

Numbers are medians from fresh-process runs on the machine named above; shared-VM noise was large, so the range
across repeated rounds is in the [full results](https://scbirlab.github.io/chemprop-accel/benchmarks/index.html). Speed-up
depends on CPU, core count, batch size and architecture. **Measure on your own machine** (next section) before relying on any of it.

## Benchmark it yourself — safely

```bash
python -m chemprop_accel.bench               # ~2–4 min; writes ./chemprop_accel_bench/report-<host>.html
python -m chemprop_accel.bench --dry-run     # see exactly what would run
```

Each configuration runs in a fresh subprocess with a timeout and a global time budget; it uses a built-in molecule
pool (no network, no downloads), writes only into its output directory, and also compares predictions with the original
implementation (non-zero exit status if they differ). If CUDA is available it also times the GPU for reference
(this package does not touch GPU execution). More: [benchmark docs](https://scbirlab.github.io/chemprop-accel/benchmarks/index.html).

## Install

```bash
pip install torch chemprop            # your usual way; CPU or CUDA build both fine
pip install chemprop-accel            # pure-Python wheel: PyTorch fallback, no compiler needed
```

For the native kernels install the wheel whose build tag matches your **PyTorch minor version** (e.g. `…-1torch2.15-py3-none-<platform>.whl`) from the GitHub release assets, or build
from source against your installed torch (needs a C++17 compiler; `--no-build-isolation` is required):

```bash
CHEMPROP_ACCEL_REQUIRE_NATIVE=1 pip install --no-build-isolation "chemprop-accel @ git+https://github.com/scbirlab/chemprop-accel"
```

Native wheels are tied to a torch minor version (C++ ABI) and a platform, **not** to a Python version, so there is no
"universal" wheel and you should check `chemprop_accel.status()`. Importing never compiles anything and never fails: if
the native library is missing or built for another torch minor, the pure-PyTorch path is used and `status()` says why.
Details: [installation](https://scbirlab.github.io/chemprop-accel/installation.html).

## Control and inspection

```python
import chemprop_accel as ca

print(
    ca.status()
)  # enabled?, Chemprop/torch versions, native extension, per-patch state, fallback reasons
ca.disable()  # restores the original methods (parameters untouched)
ca.enable()  # idempotent
```

| Environment variable | Effect |
| --- | --- |
| `CHEMPROP_ACCEL_DISABLE=1` | don't activate on import |
| `CHEMPROP_ACCEL_NO_NATIVE=1` | use the pure-PyTorch implementation only |

Importing the package never changes thread counts, seeds, precision, determinism or hyper-parameters.

## Compatibility and safety

| | Supported / tested |
| --- | --- |
| Chemprop | 2.2.0 – 2.3.x (CI: 2.2.0, 2.2.4, 2.3.1, `main`); pinned dev commit in [`CHEMPROP_COMMIT`](CHEMPROP_COMMIT) |
| Python | 3.11 – 3.14 (Chemprop's range); CI 3.11–3.13, 3.14 experimental |
| PyTorch | ≥ 2.4 |
| OS | Linux x86-64/arm64, macOS arm64, Windows (AVX dispatch is Linux/GCC x86-64 only) |
| GPU | CUDA tensors use Chemprop's original code |

Patches are applied only if the installed Chemprop is in the tested range **and** the exact source of each replaced
method matches a verified fingerprint; a numerical self-check then compares every replacement with the original and
rolls everything back on disagreement. Unsupported versions, GPU tensors and non-float32/64 dtypes run Chemprop's own
code. Higher-order autograd works (`gradgradcheck`-verified), including `torch.func.hessian`/`jacrev`/`jvp`/`vmap`, so Hessian tools such as [scbirlab/duvida](https://github.com/scbirlab/duvida) can be used on accelerated models. See
[compatibility](https://scbirlab.github.io/chemprop-accel/compatibility.html).

## Verify equivalence

`python examples/verify_equivalence.py` trains the same model with and without the package and compares losses,
predictions and `state_dict` keys. The test-suite (`pytest`) covers forward values, input and parameter gradients,
double-precision `gradcheck`, mixed graph sizes (including single-atom and disconnected molecules), directed/undirected
and atom/bond message passing, dropout, activations, vertex descriptors, aggregations, freezing, checkpoint
interchange, Lightning fitting, `torch.compile`, and repeated enable/disable cycles.

## Documentation

<https://scbirlab.github.io/chemprop-accel/> · [Benchmark report (CI, multiple runners)](https://scbirlab.github.io/chemprop-accel/benchmarks/report/) ·
[CONTRIBUTING](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) · [CHANGELOG](CHANGELOG.md)

## Status

Alpha (0.1.0). Development and the measurements above were done on a single Linux x86-64 VM with PyTorch 2.15 and
Python 3.13; the CI configuration for the wider matrix (Python 3.11–3.14, older PyTorch, macOS, Windows, arm64) is
included but has not yet been run by the author. Please report results from your hardware.

## Citing

If this helps your work, please cite Chemprop (Heid et al., *J. Chem. Inf. Model.* 2024, "Chemprop: A Machine Learning
Package for Chemical Property Prediction") and this repository.

## License

MIT. Chemprop is MIT-licensed by its authors; this project is not affiliated with the Chemprop developers.
