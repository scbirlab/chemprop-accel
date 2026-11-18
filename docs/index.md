---
html_theme.sidebar_secondary.remove: true
---

# chemprop-accel

**Drop-in multicore CPU acceleration for [Chemprop](https://github.com/chemprop/chemprop) v2.**

```python
import chemprop_accel  # that's it
import chemprop
```

Existing Chemprop training and inference code then runs on fused CPU implementations of directed-bond and atom
message aggregation. No new model classes, no changes to checkpoints, `state_dict` keys, initialisation or public
APIs, and no fork of Chemprop or PyTorch.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Install
:link: installation
:link-type: doc
Wheels, source builds, and the pure-PyTorch fallback.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
One import, `status()`, `disable()`, and a train-with-and-without equivalence check.
:::

:::{grid-item-card} How it works
:link: optimisations
:link-type: doc
What was profiled, what was fused, what did *not* help.
:::

:::{grid-item-card} Benchmarks
:link: benchmarks/index
:link-type: doc
Run them on your own machine (CPU and CUDA), and see the latest CI report.
:::

:::{grid-item-card} Compatibility
:link: compatibility
:link-type: doc
Supported Chemprop, PyTorch and Python versions; how unsupported setups fail safe.
:::

:::{grid-item-card} API
:link: api
:link-type: doc
`status()`, `enable()`, `disable()`.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
optimisations
compatibility
benchmarks/index
api
contributing
changelog
```
