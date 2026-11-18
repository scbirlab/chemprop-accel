## What and why

## Checklist
- [ ] `ruff check . && ruff format --check .` pass
- [ ] `pytest` passes locally (native + `CHEMPROP_ACCEL_NO_NATIVE=1`)
- [ ] New/changed behaviour has tests (equivalence vs original Chemprop where relevant)
- [ ] Performance claims are backed by `python -m chemprop_accel.bench` output (attach JSON / table, state the CPU)
- [ ] Docs / CHANGELOG updated
- [ ] If a Chemprop method fingerprint changed: `FINGERPRINTS` updated **and** the replacement re-verified
