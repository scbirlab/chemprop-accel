# AGENTS.md — instructions for AI coding agents

Project: **chemprop-accel** — `import chemprop_accel` makes Chemprop v2 faster on multicore CPUs by patching five
private methods with fused implementations (native C++ operator + pure-PyTorch fallback). Human-oriented docs are in
`README.md` and `CONTRIBUTING.md`; this file is the machine-oriented summary. Keep it in sync with the code.

## Commands (run from the repo root)

```bash
CHEMPROP_ACCEL_REQUIRE_NATIVE=1 pip install --no-build-isolation --no-deps -e .   # (re)build the extension
pytest -q                                   # full suite, ~25 s; must pass natively
CHEMPROP_ACCEL_NO_NATIVE=1 pytest -q        # must also pass on the pure-PyTorch fallback
ruff check . && ruff format --check .
sphinx-build -W -b html docs docs/_build/html
python -W ignore benchmarks/op_microbench.py --threads 1 2        # kernel timings
python -m chemprop_accel.bench --preset quick --output /tmp/b     # end-to-end speed + equivalence
```

Prerequisites: Python 3.11–3.14, `torch>=2.4`, `chemprop>=2.2,<2.4`, a C++17 compiler. Install torch before building
and always use `--no-build-isolation` (the extension must be compiled against the installed torch).

## Architecture in one paragraph

`csrc/graph_ops.cpp` registers `chemprop_accel::edge_message(H, dst, src, rev, n, variant)` and
`edge_message_backward`. `_native.py` loads it (refusing if built for a different torch minor), registers fake kernels
and autograd. `_ops.py` exposes `edge_message` / `scatter_sum` (native if CPU+float32/64, else `index_add_`/`index_select`).
`_patch.py` replaces `_BondMessagePassingMixin.message`, `_AtomMessagePassingMixin.message`,
`_MessagePassingBase.forward`, `MeanAggregation.forward`, `SumAggregation.forward`, each only if the original source's
fingerprint is in `_compat.FINGERPRINTS`, then runs `_selfcheck.run`; any failure rolls everything back.
`__init__.py` auto-enables on import unless `CHEMPROP_ACCEL_DISABLE=1`.

## Invariants — do not break these

- Math: `M[e] = Σ_{k: dst[k]=src[e]} H[k] − H[rev[e]]` (`rev` optional). Backward uses the same fused kernel with `src`/`dst`
  swapped when `rev` is an involution, and a general scatter otherwise.
- Importing the package must not change torch threads, seeds/RNG state, dtype, determinism, parameters or `state_dict`.
- `enable()` is idempotent; `disable()` restores the exact original function objects; patches never wrap twice.
- Unsupported Chemprop version, unknown source fingerprint, non-CPU device, or dtype other than float32/64 ⇒ the
  **original** Chemprop code runs. Never widen patches to unrelated classes or to torch itself.
- Native code validates dtype/device/shape/index range and never retains pointers between calls.
- Public Chemprop class identities, signatures and checkpoint formats are unchanged.

## Things agents commonly get wrong here

- Don't add a fingerprint to `FINGERPRINTS` without reading the diff of that Chemprop method and re-verifying the
  replacement; the fingerprint list *is* the safety mechanism.
- Don't claim a speedup without numbers from `python -m chemprop_accel.bench` or `benchmarks/op_microbench.py`
  on a stated CPU. Timings on shared CI runners are a regression signal only. Never fabricate or extrapolate timings.
- Benchmark runs share a machine with your tests: don't run `pytest`/builds while timing.
- Wheels are tied to a torch *minor* version, not to a Python version; don't claim universal wheels.
- `src/chemprop_accel/_build_info.py` and `_C.*` are generated; don't commit them.
- Negative results belong in `docs/optimisations.md` (don't delete them).

## Style

Ruff (`line-length = 100`), type hints on public API, NumPy-style docstrings, English (British spellings fine).
Commit messages: short imperative subject (`perf: …`, `fix: …`). Don't push, tag or publish without the maintainer's
instruction.
