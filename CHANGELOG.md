# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- ONNX / TorchScript / `torch.export` tracing now uses Chemprop's original code (the custom operator cannot be
  exported), so enabling the accelerator no longer adds export failures.

### Added
- Benchmark: inference (no-grad forward) speedup chart and a same-weights inference-equivalence column in the report.
- Differentiable backward for the fused op: double backward, Hessian-vector products and Hessians now work, with
  automatic pure-PyTorch dispatch under `torch.func` transforms and forward-mode AD (Duvida-style Hessian usage).
- Second-order tests (`gradgradcheck`, `torch.func.hessian`, vmapped forward-over-reverse HVPs, Chemprop-level Hessians).

## [0.1.0] - 2026-11-18

### Added
- `import chemprop_accel` activates fused CPU implementations of Chemprop v2 directed-bond and atom message
  aggregation, the message-passing readout scatter and the mean/sum aggregations, via reversible, idempotent,
  fingerprint-gated patches of five methods.
- Native C++ operator `chemprop_accel::edge_message` (+ analytic backward) with three algorithm variants and
  runtime ISA dispatch (AVX-512 / AVX2 / baseline) on x86-64 Linux; pure-PyTorch fallback.
- `chemprop_accel.status() / enable() / disable()` and the `CHEMPROP_ACCEL_DISABLE` environment variable.
- `python -m chemprop_accel.bench`: safe, self-contained benchmark with equivalence checks and HTML report
  (CPU, and CUDA reference when available).
- Test-suite (kernel, gradcheck, equivalence, checkpoints, Lightning, torch.compile, patching), GitHub Actions
  (test matrix across Python x Chemprop, docs to GitHub Pages, benchmark report on multiple runners, wheels).

### Tested against
- Chemprop 2.2.0 - 2.3.1 and development commit `934d43b` (see `CHEMPROP_COMMIT`), PyTorch 2.15 (local),
  Python 3.13 (local); the CI matrix covers Python 3.11 - 3.14.
