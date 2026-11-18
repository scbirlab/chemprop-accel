# Contributing to chemprop-accel

Thanks for helping! This project replaces a handful of private Chemprop internals with faster equivalents, so the
bar for correctness is high and every performance claim must come with a measurement. This guide is for humans;
automated agents should also read [`AGENTS.md`](https://github.com/scbirlab/chemprop-accel/blob/main/AGENTS.md).

## Quick start

```bash
git clone https://github.com/scbirlab/chemprop-accel && cd chemprop-accel
python -m venv .venv && source .venv/bin/activate
# Install PyTorch first (CPU build is enough), then Chemprop, then this package *without* build isolation
# so the C++ extension is compiled against the torch you actually have:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "chemprop>=2.2,<2.4" pytest ruff setuptools wheel ninja
CHEMPROP_ACCEL_REQUIRE_NATIVE=1 pip install --no-build-isolation --no-deps -e .
pytest -q                                   # ~25 s
CHEMPROP_ACCEL_NO_NATIVE=1 pytest -q        # pure-PyTorch fallback
ruff check . && ruff format .
pip install -e ".[docs]" && sphinx-build -W -b html docs docs/_build/html
```

Needs a C++17 compiler (GCC/Clang/MSVC). After editing `csrc/graph_ops.cpp`, re-run the `pip install -e .` line.

## Repository map

| Path | What lives there |
| --- | --- |
| `csrc/graph_ops.cpp` | native kernels; registered as `torch.ops.chemprop_accel.*` |
| `src/chemprop_accel/_native.py` | loads the library, version-checks it, registers autograd + fake kernels |
| `src/chemprop_accel/_ops.py` | public op wrappers: native when applicable, pure-PyTorch otherwise |
| `src/chemprop_accel/_patch.py` | the reversible, idempotent patches of Chemprop methods |
| `src/chemprop_accel/_compat.py` | supported-version range and **source fingerprints** of patched methods |
| `src/chemprop_accel/_selfcheck.py` | numerical self-check executed on `enable()` |
| `src/chemprop_accel/bench/` | benchmark worker, CLI orchestrator, HTML report |
| `benchmarks/` | op-level micro-benchmark used for algorithm decisions |
| `tests/` | kernel, gradcheck, equivalence, patching, compile tests |
| `docs/` | Sphinx (pydata theme) sources |

## Ground rules

1. **Equivalence first.** A change must keep original-vs-accelerated forward outputs, input gradients and parameter
   gradients equal (float64: ~1e-9; float32: reduction-order tolerance) and checkpoints interchangeable.
2. **Measure before you optimise, and keep negative results.** Use `benchmarks/op_microbench.py` for kernels and
   `python -m chemprop_accel.bench` for end-to-end. Interleave repeats, report medians, state the CPU. If an idea does
   not help, add it to `docs/optimisations.md` under "Negative results" rather than deleting the evidence.
3. **Fail safe.** Unknown Chemprop versions or method sources must leave Chemprop untouched. Never patch more than the
   targeted methods; never change global PyTorch state (threads, seeds, dtype, determinism) at import time.
4. **No hidden cost at import.** No compilation, network or large allocations on `import chemprop_accel`.
5. **Native code safety.** Validate shapes, dtypes, devices and index ranges; no raw cached pointers across calls;
   no data races (threads must own disjoint outputs or use per-thread buffers).

## Supporting a new Chemprop release

1. Install the release, then compute fingerprints of the five patch targets:
   ```python
   from chemprop_accel import _compat, _patch

   for p in _patch._chemprop_targets():
       owner, attr = p.resolve()
       print(p.name, _compat.fingerprint(owner.__dict__[attr]))
   ```
2. If a fingerprint is new, **read the diff of that method** and confirm the replacement in `_patch.py` still has
   identical semantics; only then add the hash to `FINGERPRINTS` in `_compat.py`.
3. Raise `MAX_CHEMPROP_EXCLUSIVE`, extend the CI matrix (`.github/workflows/ci.yml`), update
   `docs/compatibility.md` and `CHANGELOG.md`, and bump `CHEMPROP_COMMIT` if you re-pin the development commit.

## Changing a kernel

- Edit `csrc/graph_ops.cpp`, keep the three variants consistent, add/adjust tests in `tests/test_ops.py`
  (including `gradcheck` in double precision and the non-involutive `rev` path).
- Attach `benchmarks/op_microbench.py` output (before/after) to the PR.

## Pull requests

- Small, focused PRs; conventional commit-style titles are appreciated (`fix:`, `feat:`, `perf:`, `docs:`).
- CI must be green: lint, the Python × Chemprop matrix, other OSes, docs build.
- Update `CHANGELOG.md` under *Unreleased*.

## Releasing (maintainers)

1. Update `CHANGELOG.md` and `version` in `pyproject.toml`; merge to `main`.
2. `git tag vX.Y.Z && git push --tags` triggers `wheels.yml` (wheels per torch minor, sdist + pure wheel to PyPI via
   trusted publishing, GitHub release).
3. Docs and the benchmark report redeploy from `main` automatically (`docs.yml`).

## Code of conduct

See [`CODE_OF_CONDUCT.md`](https://github.com/scbirlab/chemprop-accel/blob/main/CODE_OF_CONDUCT.md). Security issues: [`SECURITY.md`](https://github.com/scbirlab/chemprop-accel/blob/main/SECURITY.md).
