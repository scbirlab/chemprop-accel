"""Benchmark worker: runs one configuration in the current (fresh) process and returns a dict.

Not intended to be imported by user code; use ``python -m chemprop_accel.bench``.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time


def _pcts(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)

    def q(p):
        k = (n - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {"median": statistics.median(xs), "min": xs[0], "p25": q(0.25), "p75": q(0.75), "n": n}


def _peak_rss_mb() -> float | None:
    try:
        import resource

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024
    except Exception:  # noqa: BLE001
        return None


def _timeit(fn, warmup: int, repeats: int, budget_s: float, sync=None) -> dict:
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    t_end = time.perf_counter() + budget_s
    while len(ts) < repeats or (len(ts) < 3):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
        if time.perf_counter() > t_end and len(ts) >= 3:
            break
    return _pcts(ts)


def run_worker(cfg: dict) -> dict:
    """``cfg`` keys: accel, compile, fused_optim, device, kind, batch, hidden, depth, threads, repeats,
    budget_s, check_steps, e2e, lightning, e2e_mols, variant."""
    import random

    import torch

    out: dict = {"cfg": cfg, "ok": False}
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    if cfg["accel"]:
        import chemprop_accel

        if cfg.get("variant") is not None:
            chemprop_accel.set_variant(int(cfg["variant"]))
        st = chemprop_accel.status()
        out["accel_status"] = str(st)
        if not st.enabled:
            out["error"] = f"chemprop_accel not enabled: {st.reason}"
            return out
    from chemprop import data, featurizers, models, nn
    from chemprop.data import collate_batch

    from ._molecules import SMILES

    dev = torch.device(cfg["device"])
    kind, B, H, depth = cfg["kind"], cfg["batch"], cfg["hidden"], cfg["depth"]
    rng = random.Random(0)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    n_batches = 6

    def make_ds(n):
        smis = [SMILES[rng.randrange(len(SMILES))] for _ in range(n)]
        dps = [
            data.MoleculeDatapoint.from_smi(s, [0.1 * (len(s) % 17) + 0.01 * rng.random()])
            for s in smis
        ]
        return data.MoleculeDataset(dps, featurizer)

    t0 = time.perf_counter()
    ds = make_ds(B * n_batches)
    for i in range(len(ds)):
        ds[i]
    out["featurise_s_per_mol"] = (time.perf_counter() - t0) / len(ds)
    batches = [collate_batch([ds[j] for j in range(i * B, (i + 1) * B)]) for i in range(n_batches)]
    n_atoms = sum(b.bmg.V.shape[0] for b in batches) / n_batches
    n_edges = sum(b.bmg.E.shape[0] for b in batches) / n_batches
    out["avg_atoms_per_batch"], out["avg_directed_edges_per_batch"] = n_atoms, n_edges
    for b in batches:
        b.bmg.to(dev)
    Ys = [b.Y.to(dev) for b in batches]

    def build():
        torch.manual_seed(0)
        mp = (nn.BondMessagePassing if kind == "bond" else nn.AtomMessagePassing)(
            d_h=H, depth=depth
        )
        m = models.MPNN(
            mp, nn.MeanAggregation(), nn.RegressionFFN(input_dim=H, hidden_dim=H, n_layers=1)
        )
        return m.to(dev)

    model = build()
    opt_kwargs = {"fused": True} if cfg.get("fused_optim") else {}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, **opt_kwargs)
    net = model
    if cfg["compile"]:
        net = torch.compile(model)
    mse = torch.nn.MSELoss()
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    it = {"i": 0}

    def step():
        i = it["i"] % n_batches
        it["i"] += 1
        opt.zero_grad(set_to_none=True)
        loss = mse(net(batches[i].bmg), Ys[i])
        loss.backward()
        opt.step()
        return loss

    # cold start: first step includes compilation / first-call overheads
    t0 = time.perf_counter()
    try:
        step()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"first step failed: {exc!r}"[:500]
        return out
    if sync:
        sync()
    out["cold_first_step_s"] = time.perf_counter() - t0

    rep, bud = cfg["repeats"], cfg["budget_s"]
    # compiled graphs are specialised per input shape: warm up over every distinct batch (twice)
    warm = 3 + (2 * n_batches if cfg["compile"] else 0)
    train = _timeit(step, warm, rep, bud, sync)
    out["train_step_s"] = train
    out["mols_per_s"] = B / train["median"]

    model.eval()

    def fwd():
        with torch.no_grad():
            net(batches[0].bmg)

    out["forward_s"] = _timeit(fwd, 3 + (2 if cfg["compile"] else 0), rep, bud, sync)
    model.train()
    out["peak_rss_mb"] = _peak_rss_mb()
    if dev.type == "cuda":
        out["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 2**20

    # --- predictive-equivalence probe: identical seed, identical data, fixed number of steps -----
    K = int(cfg.get("check_steps", 0))
    if K:
        model = build()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        net = torch.compile(model) if cfg["compile"] else model
        it["i"] = 0
        # inference equivalence: identical (untrained, seeded) weights, eval mode, no_grad
        model.eval()
        with torch.no_grad():
            infer = torch.cat([net(b.bmg) for b in batches]).flatten().cpu().tolist()
        model.train()
        losses = [float(step()) for _ in range(K)]
        model.eval()
        with torch.no_grad():
            preds = torch.cat([net(b.bmg) for b in batches]).flatten().cpu().tolist()
        out["check"] = {"steps": K, "losses": losses, "preds": preds, "infer_preds": infer}

    # --- end-to-end epoch: featurise + DataLoader + train, optionally Lightning -------------------
    if cfg.get("e2e"):
        model = build()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        net = torch.compile(model) if cfg["compile"] else model
        n = int(cfg.get("e2e_mols", 512))
        t0 = time.perf_counter()
        ds2 = make_ds(n)
        ds2.cache = True
        for i in range(len(ds2)):
            ds2[i]
        feat_s = time.perf_counter() - t0
        dl = data.build_dataloader(ds2, batch_size=B, num_workers=0, shuffle=True, seed=0)
        model.train()
        t0 = time.perf_counter()
        data_s = 0.0
        t_fetch = time.perf_counter()
        for b in dl:
            data_s += time.perf_counter() - t_fetch
            b.bmg.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = mse(net(b.bmg), b.Y.to(dev))
            loss.backward()
            opt.step()
            t_fetch = time.perf_counter()
        if sync:
            sync()
        epoch_s = time.perf_counter() - t0
        out["e2e"] = {
            "mols": n,
            "featurise_s": feat_s,
            "epoch_s": epoch_s,
            "dataloader_s": data_s,
            "mols_per_s": n / epoch_s,
        }
        if cfg.get("lightning"):
            import lightning.pytorch as pl

            m2 = build()
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator="gpu" if dev.type == "cuda" else "cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                deterministic=False,
            )
            t0 = time.perf_counter()
            trainer.fit(
                m2, data.build_dataloader(ds2, batch_size=B, num_workers=0, shuffle=True, seed=0)
            )
            out["e2e"]["lightning_fit_1_epoch_s"] = time.perf_counter() - t0
    out["ok"] = True
    return out


def host_info() -> dict:
    """Hardware / software environment (no network access, no subprocesses beyond ``cc --version``)."""
    import torch

    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_num_threads_default": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "cpus_available_to_process": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "env": {
            k: os.environ[k]
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_WAIT_POLICY",
                "KMP_AFFINITY",
                "GITHUB_ACTIONS",
                "CHEMPROP_ACCEL_VARIANT",
            )
            if k in os.environ
        },
    }
    try:
        from importlib import metadata

        info["chemprop"] = metadata.version("chemprop")
        info["chemprop_accel"] = metadata.version("chemprop-accel")
        info["lightning"] = metadata.version("lightning")
        info["rdkit"] = metadata.version("rdkit")
    except Exception:  # noqa: BLE001
        pass
    model = None
    flags: set[str] = set()
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name") and model is None:
                    model = line.split(":", 1)[1].strip()
                if line.startswith(("flags", "Features")) and not flags:
                    flags = set(line.split(":", 1)[1].split())
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    info["memory_gb"] = round(int(line.split()[1]) / 2**20, 1)
                    break
        elif sys.platform == "darwin":
            model = (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                or platform.processor()
            )
            mem = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            ).stdout
            info["memory_gb"] = round(int(mem) / 2**30, 1) if mem.strip().isdigit() else None
        else:
            model = platform.processor()
    except Exception:  # noqa: BLE001
        pass
    info["cpu_model"] = model or platform.processor() or "unknown"
    info["isa_extensions"] = sorted(
        f
        for f in flags
        if f
        in {
            "sse4_2",
            "avx",
            "avx2",
            "fma",
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx512_bf16",
            "amx_bf16",
            "neon",
            "asimd",
        }
    )
    try:
        cc = subprocess.run(["c++", "--version"], capture_output=True, text=True, timeout=5)
        info["compiler"] = (
            (cc.stdout or cc.stderr).splitlines()[0] if (cc.stdout or cc.stderr) else None
        )
    except Exception:  # noqa: BLE001
        info["compiler"] = None
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


if __name__ == "__main__":  # worker entry: python -m chemprop_accel.bench.core '<json cfg>'
    cfg = json.loads(sys.argv[1])
    try:
        res = run_worker(cfg)
    except Exception as exc:  # noqa: BLE001
        res = {"cfg": cfg, "ok": False, "error": repr(exc)[:500]}
    sys.stdout.write("\n@@RESULT@@" + json.dumps(res) + "\n")
