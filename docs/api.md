# API reference

```{eval-rst}
.. module:: chemprop_accel

.. autofunction:: chemprop_accel.enable
.. autofunction:: chemprop_accel.disable
.. autofunction:: chemprop_accel.status
.. autofunction:: chemprop_accel.set_variant
.. autoclass:: chemprop_accel.Status
   :members:
   :undoc-members:
```

## Environment variables

See {doc}`installation`.

## Command-line tools

```{eval-rst}
.. program:: chemprop-accel-bench
```

`python -m chemprop_accel.bench` (also installed as `chemprop-accel-bench`): the safe benchmark runner, documented in
{doc}`benchmarks/index`.

`chemprop-accel-report results-*.json -o report.html`: render one or more result files into a single self-contained
HTML report (standard library only).
