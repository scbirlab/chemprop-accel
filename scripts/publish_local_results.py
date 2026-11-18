"""Merge benchmark result files from several ``python -m chemprop_accel.bench`` runs on ONE machine and
write (a) a merged JSON, (b) a standalone HTML report and (c) the Markdown tables used by the docs.

    python scripts/publish_local_results.py DIR [DIR ...] --name sandbox-2vcpu --out docs/benchmarks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemprop_accel.bench.__main__ import equivalence
from chemprop_accel.bench.report import MODE_LABEL, collapse, write_report

ROWS = [
    "eager",
    "eager_fused",
    "eager_compile",
    "accel_pure",
    "accel_v0",
    "accel_v1",
    "accel",
    "accel_fused",
    "accel_compile",
]


def load(dirs):
    docs = [json.loads(p.read_text()) for d in dirs for p in sorted(Path(d).glob("results-*.json"))]
    merged = dict(docs[0])
    merged["results"] = [r for d in docs for r in d["results"]]
    merged["equivalence"] = equivalence(merged["results"])
    merged["wall_minutes"] = sum(d.get("wall_minutes", 0) for d in docs)
    merged["preset"] = "merged: " + ", ".join(sorted({d.get("preset", "?") for d in docs}))
    return merged


def md_table(res, kind, B, T):
    sel = {
        r["cfg"]["mode"]: r
        for r in res
        if r.get("ok")
        and r["cfg"]["kind"] == kind
        and r["cfg"]["batch"] == B
        and r["cfg"]["threads"] == T
        and r["cfg"]["device"] == "cpu"
        and r["cfg"]["hidden"] == 300
    }
    if "eager" not in sel:
        return ""
    base = sel["eager"]["train_step_s"]["median"]
    lines = [
        "| Configuration | Forward (ms) | Training step (ms) | Molecules/s | Peak RSS (MB) | Cold first step (s) | Speedup (train step) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in ROWS:
        if m not in sel:
            continue
        r = sel[m]
        lines.append(
            f"| {MODE_LABEL[m]} | {r['forward_s']['median'] * 1e3:.2f} | {r['train_step_s']['median'] * 1e3:.2f} | "
            f"{r['mols_per_s']:,.0f} | {r['peak_rss_mb']:,.0f} | {r['cold_first_step_s']:.2f} | "
            f"{base / r['train_step_s']['median']:.2f}× |"
        )
    return "\n".join(lines)


def scaling_table(res):
    rows = {}
    for r in res:
        c = r["cfg"]
        if (
            r.get("ok")
            and c["device"] == "cpu"
            and c["mode"] in ("eager", "accel")
            and c["hidden"] == 300
        ):
            rows.setdefault((c["kind"], c["batch"], c["threads"]), {})[c["mode"]] = r
    out = [
        "| Message passing | Batch | Threads | Original (ms/step) | chemprop_accel (ms/step) | Train speedup | Forward speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (kind, B, T), d in sorted(rows.items()):
        if len(d) == 2:
            e, a = d["eager"], d["accel"]
            out.append(
                f"| {kind} | {B} | {T} | {e['train_step_s']['median'] * 1e3:.2f} | {a['train_step_s']['median'] * 1e3:.2f} | "
                f"{e['train_step_s']['median'] / a['train_step_s']['median']:.2f}× | "
                f"{e['forward_s']['median'] / a['forward_s']['median']:.2f}× |"
            )
    return "\n".join(out)


def e2e_table(res):
    d = {
        r["cfg"]["mode"]: r
        for r in res
        if r.get("ok") and r.get("e2e") and r["cfg"]["mode"] in ("eager", "accel")
    }
    if len(d) < 2:
        return ""
    e, a = d["eager"]["e2e"], d["accel"]["e2e"]
    rows = [
        "| Stage (1024 molecules, B=64, bond MPNN) | Original (s) | chemprop_accel (s) |",
        "| --- | ---: | ---: |",
        f"| RDKit featurisation (cached after first pass) | {e['featurise_s']:.2f} | {a['featurise_s']:.2f} |",
        f"| DataLoader + collate (time waiting on batches) | {e['dataloader_s']:.2f} | {a['dataloader_s']:.2f} |",
        f"| Training epoch, total (DataLoader + steps) | {e['epoch_s']:.2f} | {a['epoch_s']:.2f} |",
    ]
    if "lightning_fit_1_epoch_s" in e and "lightning_fit_1_epoch_s" in a:
        rows.append(
            f"| Lightning `Trainer.fit`, 1 epoch incl. setup | {e['lightning_fit_1_epoch_s']:.2f} | {a['lightning_fit_1_epoch_s']:.2f} |"
        )
    return "\n".join(rows)


def equiv_table(eq):
    out = [
        "| Run | Steps | max abs Δ prediction | max rel Δ prediction | max abs Δ loss | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for e in eq:
        dev, kind, B, H, D, T, comp, fused = e["key"]
        out.append(
            f"| {kind}, B={B}, {T} thr{', compiled' if comp else ''} ({e['mode']}) | {e['steps']} | {e['max_abs_pred_diff']:.1e} | "
            f"{e['max_rel_pred_diff']:.1e} | {e['max_abs_loss_diff']:.1e} | {'pass' if e['passed'] else 'FAIL'} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="docs/benchmarks")
    a = ap.parse_args()
    doc = load(a.dirs)
    doc["name"] = a.name
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / f"results-{a.name}.json").write_text(json.dumps(doc, indent=1))
    static = out.parent / "_static"
    static.mkdir(exist_ok=True)
    write_report([doc], static / f"report-{a.name}.html")
    h = doc["host"]
    res = [r for r in collapse(doc["results"]) if r.get("ok")]
    T = h["cpus_available_to_process"]
    md = [
        f"<!-- generated by scripts/publish_local_results.py from {len(res)} runs; do not edit by hand -->",
        f"**Machine:** {h['cpu_model']} · {h['cpus_available_to_process']} vCPU available · {h.get('memory_gb')} GB RAM · "
        f"{h['platform']} · Python {h['python']} · PyTorch {h['torch']} · Chemprop {h.get('chemprop')} · {h.get('compiler')}  \n"
        f"**ISA extensions seen:** {', '.join(h['isa_extensions'])} · **Run:** {doc['created'][:10]}\n",
        f"## Bond message passing, batch 64, d_h=300, depth 3, {T} threads",
        md_table(res, "bond", 64, T),
        "",
        "## Bond message passing, batch 64, 1 thread",
        md_table(res, "bond", 64, 1),
        "",
        f"## Atom message passing, batch 64, {T} threads",
        md_table(res, "atom", 64, T),
        "",
        "## Scaling with batch size and message-passing type",
        scaling_table(res),
        "",
        "## End-to-end epoch",
        e2e_table(res),
        "",
        "## Predictive equivalence",
        equiv_table(doc["equivalence"]),
    ]
    (out / "results_generated.md").write_text("\n".join(md))
    print("wrote", out / "results_generated.md", "and report", static / f"report-{a.name}.html")


if __name__ == "__main__":
    main()
