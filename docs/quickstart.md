# Quickstart

## One import

```python
import chemprop_accel  # activates automatically
from chemprop import nn, models

mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
ffn = nn.RegressionFFN()
model = models.MPNN(mp, agg, ffn)  # trains/infers exactly as before, just faster on CPU
```

Import order does not matter: `import chemprop` first also works, and models that already exist when the package
is imported are accelerated too, because the patches live on the (private) Chemprop classes. Things that capture the
old behaviour early, such as a model you already wrapped in `torch.compile` and ran, keep their traced graph; compile
*after* importing `chemprop_accel`.

## Inspect and control

```python
import chemprop_accel as ca

print(
    ca.status()
)  # enabled?, supported versions, native extension, per-patch state, fallback reasons
ca.disable()  # restore the original Chemprop methods (parameters are never touched)
ca.enable()  # idempotent; re-applies the patches
```

```text
chemprop_accel 0.1.0: ENABLED (ok)
  chemprop: 2.3.1 (supported >=2.2.0,<2.4.0: True)
  torch: 2.15.0
  native extension: available - ok [built against torch 2.15.0]
  native variant: 2
  patch BondMessagePassing.message: active
  ...
```

To keep a process from being patched at import, set `CHEMPROP_ACCEL_DISABLE=1` and call `chemprop_accel.enable()`
yourself where you want it.

## What is *not* touched

The package never changes thread counts, seeds, precision, determinism flags or hyper-parameters; it does not
change `state_dict` keys or initialisation; and it falls back to Chemprop's own code for GPU tensors, dtypes other
than float32/float64 (e.g. autocast bf16), unsupported Chemprop versions and anything whose source it does not
recognise.

## Verify equivalence yourself

```{literalinclude} ../examples/verify_equivalence.py
:language: python
```

## Benchmark your machine

```bash
python -m chemprop_accel.bench            # ~2 minutes; CPU, plus CUDA reference if available
```

See {doc}`benchmarks/index`.
