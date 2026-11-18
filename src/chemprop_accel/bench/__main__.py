"""Safe, self-contained benchmark runner.

    python -m chemprop_accel.bench                 # quick preset (~1-3 min), writes ./chemprop_accel_bench/
    python -m chemprop_accel.bench --preset standard
    python -m chemprop_accel.bench --dry-run       # show the plan, run nothing

What it does / does not do
--------------------------
* Runs every configuration in a **fresh subprocess** with a per-run timeout and a global time budget.
* Uses only a small built-in molecule pool: **no network access, no downloads, no external data**.
* Writes only into ``--output`` (JSON + HTML report). It never modifies your Chemprop install.
* Touches the GPU only if CUDA is available and ``--device`` asks for it, with batches of at most a
  few hundred small molecules (well under 1 GB of device memory).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .core import host_info

SCHEMA = 1

PRESETS = {
    #            kinds,            batch,            hidden,  depth, threads,       modes
    "quick": dict(
        kinds=["bond"],
        batch=[32, 128],
        hidden=[300],
        depth=[3],
        threads=["all"],
        modes=["eager", "accel"],
        repeats=15,
        budget_s=6,
        check_steps=10,
        e2e=True,
        e2e_mols=256,
    ),
    "standard": dict(
        kinds=["bond", "atom"],
        batch=[8, 64, 256],
        hidden=[300],
        depth=[3],
        threads=["1", "all"],
        modes=["eager", "accel", "eager_compile", "accel_compile"],
        repeats=20,
        budget_s=8,
        check_steps=15,
        e2e=True,
        e2e_mols=1024,
    ),
    "ci": dict(
        kinds=["bond", "atom"],
        batch=[32, 128],
        hidden=[300],
        depth=[3],
        threads=["1", "all"],
        modes=["eager", "accel", "eager_compile", "accel_compile"],
        repeats=15,
        budget_s=6,
        check_steps=15,
        e2e=True,
        e2e_mols=512,
    ),
    "full": dict(
        kinds=["bond", "atom"],
        batch=[1, 8, 32, 64, 128, 256],
        hidden=[128, 300, 600],
        depth=[3, 5],
        threads=["1", "2", "4", "8", "all"],
        modes=["eager", "eager_fused", "accel", "accel_fused", "eager_compile", "accel_compile"],
        repeats=20,
        budget_s=8,
        check_steps=0,
        e2e=False,
        e2e_mols=1024,
    ),
}
MODE_FLAGS = {  # accel, compile, fused_optim, native variant (None = default), force pure-PyTorch fallback
    "eager": (False, False, False, None, False),
    "eager_fused": (False, False, True, None, False),
    "accel": (True, False, False, None, False),
    "accel_fused": (True, False, True, None, False),
    "eager_compile": (False, True, False, None, False),
    "accel_compile": (True, True, False, None, False),
    # ablations
    "accel_pure": (True, False, False, None, True),  # patches active, native kernels disabled
    "accel_v0": (True, False, False, 0, False),  # column-tiled scatter
    "accel_v1": (True, False, False, 1, False),  # CSR + atom-parallel
    "accel_v2": (True, False, False, 2, False),  # direct per-edge neighbour sum
}


def _threads_list(spec, avail):
    out = []
    for t in spec:
        n = avail if t == "all" else int(t)
        if 1 <= n <= avail and n not in out:
            out.append(n)
    return out or [avail]


def build_plan(a, avail_cpus, cuda):
    p = dict(PRESETS[a.preset])
    for k in ("kinds", "batch", "hidden", "depth", "modes"):
        v = getattr(a, k, None)
        if v:
            p[k] = v
    if a.threads:
        p["threads"] = a.threads
    for k in ("repeats", "budget_s", "check_steps", "e2e_mols"):
        if getattr(a, k, None) is not None:
            p[k] = getattr(a, k)
    if a.e2e is not None:
        p["e2e"] = a.e2e
    devices = {
        "auto": ["cpu"] + (["cuda"] if cuda else []),
        "cpu": ["cpu"],
        "cuda": ["cuda"],
        "both": ["cpu", "cuda"],
    }[a.device]
    if "cuda" in devices and not cuda:
        print("note: CUDA requested but not available; skipping GPU runs", file=sys.stderr)
        devices = [d for d in devices if d != "cuda"]
    runs = []
    for dev in devices:
        threads = _threads_list(p["threads"], avail_cpus) if dev == "cpu" else [avail_cpus]
        modes = p["modes"] if dev == "cpu" else [m for m in p["modes"] if "compile" not in m][:2]
        # on CUDA, chemprop_accel falls back to the original ops, so eager vs accel should match
        for kind, B, H, D, T, mode in itertools.product(
            p["kinds"], p["batch"], p["hidden"], p["depth"], threads, modes
        ):
            acc, comp, fused, variant, pure = MODE_FLAGS[mode]
            runs.append(
                dict(
                    accel=acc,
                    compile=comp,
                    fused_optim=fused,
                    device=dev,
                    kind=kind,
                    batch=B,
                    hidden=H,
                    depth=D,
                    threads=T,
                    repeats=p["repeats"],
                    budget_s=p["budget_s"],
                    check_steps=p["check_steps"],
                    e2e=p["e2e"] and dev == "cpu" and B >= 8,
                    e2e_mols=p["e2e_mols"],
                    lightning=a.lightning,
                    mode=mode,
                    variant=variant if variant is not None else a.variant,
                    pure=pure,
                )
            )
    return runs


def run_one(cfg, timeout):
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore"
    if cfg["accel"]:
        env.pop("CHEMPROP_ACCEL_DISABLE", None)
    else:
        env["CHEMPROP_ACCEL_DISABLE"] = "1"
    if cfg.get("pure"):
        env["CHEMPROP_ACCEL_NO_NATIVE"] = "1"
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-W", "ignore", "-m", "chemprop_accel.bench.core", json.dumps(cfg)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"cfg": cfg, "ok": False, "error": f"timeout after {timeout}s"}
    for line in r.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            res = json.loads(line[len("@@RESULT@@") :])
            res["wall_s"] = time.time() - t0
            return res
    return {
        "cfg": cfg,
        "ok": False,
        "error": f"worker crashed (rc={r.returncode}): {(r.stderr or '')[-400:]}",
    }


def equivalence(results):
    """Compare each accelerated run with its eager twin (same everything except accel)."""

    def key(c):
        return (
            c["device"],
            c["kind"],
            c["batch"],
            c["hidden"],
            c["depth"],
            c["threads"],
            c["compile"],
            c["fused_optim"],
        )

    eager = {
        key(r["cfg"]): r
        for r in results
        if r.get("ok") and not r["cfg"]["accel"] and r.get("check")
    }
    rows, seen = [], {}
    for r in results:
        c = r["cfg"]
        if r.get("ok") and c["accel"] and r.get("check"):
            seen[(c["mode"], key(c))] = r  # repeated rounds of the same config: keep the last
    for (_, k), r in seen.items():
        c = r["cfg"]
        e = eager.get(k)
        if not e:
            continue
        pa, pe = r["check"]["preds"], e["check"]["preds"]
        la, le = r["check"]["losses"], e["check"]["losses"]
        scale = max(1e-12, max(abs(x) for x in pe))
        dmax = max(abs(a - b) for a, b in zip(pa, pe))
        ia, ie = r["check"].get("infer_preds"), e["check"].get("infer_preds")
        idmax = max(abs(a - b) for a, b in zip(ia, ie)) if ia and ie else None
        iscale = max(1e-12, max(abs(x) for x in ie)) if ie else None
        rows.append(
            dict(
                key=list(key(c)),
                mode=c["mode"],
                steps=r["check"]["steps"],
                max_abs_pred_diff=dmax,
                max_rel_pred_diff=dmax / scale,
                max_abs_loss_diff=max(abs(a - b) for a, b in zip(la, le)),
                pred_scale=scale,
                max_abs_infer_diff=idmax,
                max_rel_infer_diff=None if idmax is None else idmax / iscale,
                passed=dmax / scale <= 1e-3 and (idmax is None or idmax / iscale <= 1e-3),
            )
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m chemprop_accel.bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--preset", choices=PRESETS, default="quick")
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "both"],
        default="auto",
        help="auto = CPU, plus CUDA if available (default)",
    )
    ap.add_argument("--kinds", nargs="+", choices=["bond", "atom"])
    ap.add_argument("--batch", nargs="+", type=int)
    ap.add_argument("--hidden", nargs="+", type=int)
    ap.add_argument("--depth", nargs="+", type=int)
    ap.add_argument("--threads", nargs="+", help="e.g. 1 2 4 all")
    ap.add_argument("--modes", dest="modes", nargs="+", choices=list(MODE_FLAGS))
    ap.add_argument("--repeats", type=int)
    ap.add_argument(
        "--budget-s", dest="budget_s", type=float, help="max seconds of timing per measurement"
    )
    ap.add_argument(
        "--check-steps", dest="check_steps", type=int, help="train steps for the equivalence probe"
    )
    ap.add_argument(
        "--e2e",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also time featurisation + DataLoader + a full training epoch",
    )
    ap.add_argument("--e2e-mols", dest="e2e_mols", type=int)
    ap.add_argument(
        "--lightning",
        action="store_true",
        help="also time a 1-epoch Lightning fit (adds ~10 s/run)",
    )
    ap.add_argument(
        "--variant", type=int, choices=[0, 1, 2], default=None, help="native algorithm variant"
    )
    ap.add_argument("--output", default="chemprop_accel_bench", help="output directory (created)")
    ap.add_argument(
        "--name", default=None, help="label for this machine in reports (default: hostname)"
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="repeat the whole plan N times (interleaved); reports use the median across rounds. "
        "Use 3+ on shared/noisy machines",
    )
    ap.add_argument(
        "--max-minutes",
        type=float,
        default=15.0,
        help="global time budget; remaining runs are skipped",
    )
    ap.add_argument("--run-timeout", type=float, default=600.0, help="per-run timeout in seconds")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--yes", action="store_true", help="skip the confirmation required by --preset full"
    )
    a = ap.parse_args(argv)

    import platform

    try:
        import chemprop  # noqa: F401
        import torch
    except ImportError as exc:
        print(f"error: {exc}. Install chemprop and torch first.", file=sys.stderr)
        return 2
    avail = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    runs = build_plan(a, avail, torch.cuda.is_available())
    print(
        f"chemprop_accel benchmark: {len(runs)} runs, preset={a.preset}, budget={a.max_minutes} min, "
        f"cpus available={avail}, cuda={torch.cuda.is_available()}"
    )
    if a.dry_run:
        for r in runs:
            print(
                " ",
                r["device"],
                r["mode"],
                r["kind"],
                f"B={r['batch']} H={r['hidden']} D={r['depth']} T={r['threads']}",
            )
        return 0
    if a.preset == "full" and not a.yes:
        print(
            "The 'full' preset can take hours. Re-run with --yes to confirm, or narrow it with --batch/--threads."
        )
        return 2
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or platform.node() or "machine"
    results, t_start = [], time.time()
    schedule = [(rnd, i, cfg) for rnd in range(max(1, a.rounds)) for i, cfg in enumerate(runs, 1)]
    for n_done, (rnd, i, cfg) in enumerate(schedule):
        if time.time() - t_start > a.max_minutes * 60:
            print(f"time budget reached after {n_done} runs; remaining runs skipped")
            break
        print(
            f"[round {rnd + 1} · {i}/{len(runs)}] {cfg['device']:4s} {cfg['mode']:14s} {cfg['kind']} B={cfg['batch']} H={cfg['hidden']} "
            f"D={cfg['depth']} T={cfg['threads']} ... ",
            end="",
            flush=True,
        )
        res = run_one(cfg, a.run_timeout * (3 if cfg["compile"] else 1))
        res["round"] = rnd
        results.append(res)
        if res.get("ok"):
            print(
                f"{res['train_step_s']['median'] * 1e3:8.2f} ms/step  {res['mols_per_s']:9.0f} mol/s"
            )
        else:
            print(f"FAILED: {res.get('error')}")
    doc = dict(
        schema=SCHEMA,
        name=name,
        created=dt.datetime.now(dt.UTC).isoformat(),
        host=host_info(),
        preset=a.preset,
        rounds=max(1, a.rounds),
        results=results,
        equivalence=equivalence(results),
        wall_minutes=(time.time() - t_start) / 60,
    )
    jpath = out / f"results-{name}.json"
    jpath.write_text(json.dumps(doc, indent=1))
    from .report import write_report

    hpath = write_report([doc], out / f"report-{name}.html")
    print(f"\nWrote {jpath}\nWrote {hpath}")
    bad = [r for r in doc["equivalence"] if not r["passed"]]
    if bad:
        print(
            f"WARNING: {len(bad)} accelerated run(s) differ from eager beyond tolerance - please report this!"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
