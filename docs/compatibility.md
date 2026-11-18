# Compatibility

## Supported versions

| Component | Supported | Tested in CI | Notes |
| --- | --- | --- | --- |
| Chemprop | `>=2.2.0, <2.4.0` | 2.2.0, 2.2.4, 2.3.1, `main` | development commit pinned in `CHEMPROP_COMMIT` |
| Python | 3.11 – 3.14 (Chemprop's own range) | 3.11, 3.12, 3.13 (+ 3.14 experimental) | the native library is Python-version independent |
| PyTorch | `>=2.4` | oldest (2.4) and latest | `torch.library.register_autograd` is needed for the native ops |
| OS / arch | Linux x86-64 and arm64, macOS arm64, Windows x86-64 | all four runner types | AVX-512/AVX2 run-time dispatch is x86-64 Linux/GCC only |
| GPU | CUDA tensors use Chemprop's original code | CPU-only CI | no GPU kernels are provided; the benchmark can compare against your GPU |

Versions outside these ranges are **not** patched: `status()` reports the reason and Chemprop runs unmodified.

## How unsupported setups fail safe

Each patch is applied only if *both* conditions hold:

1. the installed Chemprop version is inside the tested range; and
2. a whitespace-insensitive hash of the source of the exact method being replaced matches a fingerprint that the
   replacement was verified against (`chemprop_accel._compat.FINGERPRINTS`).

After patching, a numerical self-check (float64, six small molecules including a single-atom and a disconnected
molecule) compares every replacement with the original it replaced, forward and backward. If anything disagrees, all
patches are rolled back. Each patch also falls back **per call** to the original implementation for non-CPU tensors
and dtypes other than float32/float64.

## Patched methods (and only these)

| Target | Replacement |
| --- | --- |
| `_BondMessagePassingMixin.message` | fused `M[e] = Σ_{k: dst[k]=src[e]} H[k] − H[rev[e]]` |
| `_AtomMessagePassingMixin.message` | same fused op without the reverse-edge term |
| `_MessagePassingBase.forward` | identical, but the final readout scatter uses `index_add_` |
| `MeanAggregation.forward`, `SumAggregation.forward` | `index_add_` (plus `bincount` for the mean) |

Classes, parameter names, `state_dict` keys, initialisation and checkpoint formats are untouched.
`MulticomponentMessagePassing` simply calls ordinary (patched) message-passing modules. The mol-atom-bond classes
(`MABBondMessagePassing`, `MABAtomMessagePassing`) inherit the patched mixin `message` methods but define their own
`forward`, which is **not** patched; they are not covered by dedicated tests yet.

## Interaction with other features

- **PyTorch Lightning**: training/validation loops are unchanged; covered by a test comparing weights after a
  `Trainer.fit`.
- **`torch.compile`**: the native ops have fake kernels and autograd formulas, so Inductor/AOT autograd trace through
  them as opaque operators (tested with `inductor` and `aot_eager`). Enable the package *before* compiling.
  The op takes the node count as a plain `int`, so dynamic-shape compilation specialises per batch size.
- **Higher-order autograd and Hessians**: the fused op's backward is itself differentiable (the transpose of the op is the
  same op with `src`/`dst` swapped; a composite fallback handles a non-involutive `rev`), so double backward,
  `torch.autograd.functional.hessian` and Hessian-vector products work and are `gradgradcheck`-tested. Under
  `torch.func` transforms (`vmap`, `jvp`, `jacrev`, `hessian`) and forward-mode AD, the pure-PyTorch implementation is
  used automatically, since it supports every transform; results are identical. Tests mirror the forward-over-reverse
  HVP and `vmap` patterns of [scbirlab/duvida](https://github.com/scbirlab/duvida) (including `functional_call` over a
  Chemprop MPNN with ELU activations). Second-order passes through the native kernel run only for plain autograd
  (`create_graph=True`).
- **Export (ONNX, `torch.jit.trace`, `torch.export`)**: while tracing/exporting, the replacements step aside and
  Chemprop's own code runs, because exporters cannot serialise the custom operator. Export behaviour is therefore
  identical to Chemprop's (including any upstream limitations, e.g. `scatter_reduce(include_self=False)` on older
  exporters). Call `chemprop_accel.disable()` if you prefer to be explicit.
- **Determinism**: results are bitwise independent of thread count for variant 0; variants 1 and 2 also own disjoint
  outputs per thread. Results differ from Chemprop's `scatter_reduce_` only by floating-point summation order.
- **bf16 / autocast, fp16**: use the original code path.
- **Multiprocessing**: patches live in the importing process; DataLoader workers don't run the model, so nothing is
  needed there. With `spawn`, import `chemprop_accel` in the module that runs the model.
