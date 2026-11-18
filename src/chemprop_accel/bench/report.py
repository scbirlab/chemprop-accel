"""Render one or more benchmark result files into a single self-contained HTML report.

chemprop-accel-report results-*.json -o index.html --title "CI benchmarks"
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CSS = """
:root{--fg:#1f2430;--muted:#667;--bg:#fff;--line:#e3e6ec;--good:#0a7d3b;--bad:#b3261e;--acc:#2a5db0;--eager:#9aa4b5}
*{box-sizing:border-box}body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);max-width:1180px;margin:2rem auto;padding:0 1.2rem}
h1{font-size:1.7rem;margin-bottom:.2rem}h2{margin-top:2.4rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}h3{margin-top:1.6rem}
table{border-collapse:collapse;width:100%;margin:.6rem 0 1.2rem;font-size:13.5px}th,td{padding:.35rem .6rem;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th.l,td.l{text-align:left}th{background:#f6f7fa;position:sticky;top:0}
.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}.muted{color:var(--muted)}
.note{background:#fff8e6;border:1px solid #f0dca0;padding:.7rem 1rem;border-radius:6px;font-size:14px}
.kv td:first-child{color:var(--muted);width:210px}.scroll{overflow-x:auto}
svg text{font:11px sans-serif;fill:var(--fg)}code{background:#f1f3f7;padding:0 .25rem;border-radius:3px}
"""

MODE_ORDER = [
    "eager",
    "eager_fused",
    "accel_pure",
    "accel_v0",
    "accel",
    "accel_v1",
    "accel_v2",
    "accel_fused",
    "eager_compile",
    "accel_compile",
]
MODE_LABEL = {
    "eager": "Original Chemprop",
    "eager_fused": "Original + fused Adam",
    "accel": "chemprop_accel (default, variant 2)",
    "accel_fused": "chemprop_accel + fused Adam",
    "eager_compile": "torch.compile",
    "accel_compile": "chemprop_accel + compile",
    "accel_pure": "accel: pure-PyTorch fallback",
    "accel_v0": "accel: tiled-scatter variant (0)",
    "accel_v1": "accel: CSR variant (1)",
    "accel_v2": "accel: direct variant (2)",
}


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return (
        None
        if not xs
        else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)
    )


def collapse(results: list[dict]) -> list[dict]:
    """Merge repeated rounds of the same configuration into one result (median across rounds).

    ``train_step_s`` / ``forward_s`` gain ``lo``/``hi`` (min/max of the per-round medians).
    """
    ident = (
        "mode",
        "accel",
        "compile",
        "fused_optim",
        "device",
        "kind",
        "batch",
        "hidden",
        "depth",
        "threads",
        "variant",
        "pure",
    )
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(
            json.dumps({k: r["cfg"].get(k) for k in ident}, sort_keys=True), []
        ).append(r)
    out = []
    for grp in groups.values():
        ok = [r for r in grp if r.get("ok")]
        if not ok:
            out.append(grp[0])
            continue
        m = dict(ok[0])
        m["rounds"] = len(ok)
        for k in ("train_step_s", "forward_s"):
            meds = [r[k]["median"] for r in ok]
            m[k] = {**ok[0][k], "median": _med(meds), "lo": min(meds), "hi": max(meds)}
        m["mols_per_s"] = m["cfg"]["batch"] / m["train_step_s"]["median"]
        for k in ("cold_first_step_s", "peak_rss_mb", "peak_cuda_mb"):
            if ok[0].get(k) is not None:
                m[k] = _med([r.get(k) for r in ok])
        with_e2e = [r for r in ok if r.get("e2e")]
        if with_e2e:
            m["e2e"] = {k: _med([r["e2e"].get(k) for r in with_e2e]) for k in with_e2e[0]["e2e"]}
        with_check = [r for r in ok if r.get("check")]
        if with_check:
            m["check"] = with_check[0]["check"]
        out.append(m)
    return out


def _e(x):
    return html.escape(str(x))


def _ms(x):
    return f"{x * 1e3:.2f}"


def _key(c):
    return (c["device"], c["kind"], c["batch"], c["hidden"], c["depth"], c["threads"])


def _fmt_key(k):
    dev, kind, B, H, D, T = k
    return f"{dev} · {kind} · B={B} · d_h={H} · depth={D} · {T} thread{'s' if T != 1 else ''}"


def _spd(base, x):
    if not base or not x:
        return "", None
    s = base / x
    cls = "good" if s > 1.05 else ("bad" if s < 0.95 else "muted")
    return f' <span class="{cls}">{s:.2f}×</span>', s


def _bars(items, title, width=900):
    """items: list of (label, speedup). Horizontal SVG bars with a 1.0× baseline."""
    if not items:
        return ""
    h, pad = 20, 6
    mx = max(2.0, max(s for _, s in items) * 1.1)
    lab_w = 420
    H = (h + pad) * len(items) + 30
    x1 = lab_w + (width - lab_w - 60) / mx
    rows = []
    for i, (lab, s) in enumerate(items):
        y = 24 + i * (h + pad)
        w = (width - lab_w - 60) * s / mx
        color = "var(--acc)" if s >= 1 else "var(--bad)"
        rows.append(
            f'<text x="{lab_w - 8}" y="{y + 14}" text-anchor="end">{_e(lab)}</text>'
            f'<rect x="{lab_w}" y="{y}" width="{max(w, 1):.1f}" height="{h}" fill="{color}" rx="2"/>'
            f'<text x="{lab_w + w + 5:.1f}" y="{y + 14}">{s:.2f}×</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {H}" width="100%" role="img" aria-label="{_e(title)}">'
        f'<text x="0" y="12" style="font-weight:600">{_e(title)}</text>'
        f'<line x1="{x1:.1f}" x2="{x1:.1f}" y1="18" y2="{H - 4}" stroke="#888" stroke-dasharray="3 3"/>'
        + "".join(rows)
        + "</svg>"
    )


def _host_table(h):
    rows = [
        ("Platform", h.get("platform")),
        ("CPU", h.get("cpu_model")),
        ("ISA extensions", ", ".join(h.get("isa_extensions", [])) or "n/a"),
        (
            "Logical CPUs / available to process",
            f"{h.get('logical_cpus')} / {h.get('cpus_available_to_process')}",
        ),
        ("Memory (GB)", h.get("memory_gb")),
        ("Python", h.get("python")),
        ("PyTorch", h.get("torch")),
        ("Chemprop", h.get("chemprop")),
        ("chemprop-accel", h.get("chemprop_accel")),
        ("Lightning / RDKit", f"{h.get('lightning')} / {h.get('rdkit')}"),
        ("C++ compiler", h.get("compiler")),
        (
            "CUDA",
            f"{h.get('cuda_device')} (CUDA {h.get('cuda_version')})"
            if h.get("cuda_available")
            else "not available",
        ),
        ("Threading env", ", ".join(f"{k}={v}" for k, v in h.get("env", {}).items()) or "none set"),
    ]
    return (
        '<table class="kv">'
        + "".join(f"<tr><td>{_e(k)}</td><td class='l'>{_e(v)}</td></tr>" for k, v in rows)
        + "</table>"
    )


def _train_getter(r):
    return r["train_step_s"]["median"]


def _machine_section(doc, idx):
    collapsed = collapse(doc["results"])
    res = [r for r in collapsed if r.get("ok")]
    failed = [r for r in collapsed if not r.get("ok")]
    by = {}
    for r in res:
        by.setdefault(_key(r["cfg"]), {})[r["cfg"]["mode"]] = r
    modes = [m for m in MODE_ORDER if any(m in v for v in by.values())]
    out = [
        f"<h2 id='m{idx}'>{_e(doc.get('name', 'machine'))}</h2>",
        f"<p class='muted'>preset <code>{_e(doc.get('preset'))}</code> · {len(doc['results'])} runs"
        f"{' (medians pooled over up to ' + str(max(r.get('rounds', 1) for r in res)) + ' repeated runs per configuration)' if res and max(r.get('rounds', 1) for r in res) > 1 else ''} · "
        f"{doc.get('wall_minutes', 0):.1f} min · {_e(doc.get('created', ''))}</p>",
        _host_table(doc["host"]),
    ]

    def table(title, getter, fmt, spd=True, lower_better=True):
        head = "".join(f"<th>{_e(MODE_LABEL[m])}</th>" for m in modes)
        rows = []
        for k, d in sorted(by.items()):
            base = getter(d["eager"]) if "eager" in d else None
            cells = []
            for m in modes:
                if m not in d or getter(d[m]) is None:
                    cells.append("<td>–</td>")
                    continue
                v = getter(d[m])
                extra = _spd(base, v)[0] if (spd and m != "eager" and lower_better) else ""
                rng = ""
                if (
                    isinstance(d[m].get("train_step_s"), dict)
                    and d[m]["train_step_s"].get("hi")
                    and getter is _train_getter
                    and d[m]["train_step_s"]["hi"] != d[m]["train_step_s"]["lo"]
                ):
                    rng = f" <span class='muted'>[{_ms(d[m]['train_step_s']['lo'])}–{_ms(d[m]['train_step_s']['hi'])}]</span>"
                cells.append(f"<td>{fmt(v)}{rng}{extra}</td>")
            rows.append(f"<tr><td class='l'>{_e(_fmt_key(k))}</td>{''.join(cells)}</tr>")
        return f"<h3>{_e(title)}</h3><div class='scroll'><table><tr><th class='l'>Configuration</th>{head}</tr>{''.join(rows)}</table></div>"

    bars = []
    for k, d in sorted(by.items()):
        if "eager" in d and "accel" in d and k[0] == "cpu":
            bars.append(
                (
                    _fmt_key(k)[4:],
                    d["eager"]["train_step_s"]["median"] / d["accel"]["train_step_s"]["median"],
                )
            )
    out.append(
        _bars(
            bars, "Training-step speedup: chemprop_accel vs original Chemprop (CPU; >1 is faster)"
        )
    )
    out.append(
        table(
            "Training step (median ms [range across rounds]; speedup vs original)",
            _train_getter,
            _ms,
        )
    )
    out.append(
        table(
            "Throughput (molecules / s)",
            lambda r: r["mols_per_s"],
            lambda v: f"{v:,.0f}",
            spd=False,
        )
    )
    fbars = []
    for k, d in sorted(by.items()):
        if "eager" in d and "accel" in d and k[0] == "cpu":
            fbars.append(
                (
                    _fmt_key(k)[4:],
                    d["eager"]["forward_s"]["median"] / d["accel"]["forward_s"]["median"],
                )
            )
    out.append(
        _bars(
            fbars,
            "Inference (no-grad forward) speedup: chemprop_accel vs original Chemprop (CPU; >1 is faster)",
        )
    )
    out.append(
        table("Forward / inference (median ms; no-grad)", lambda r: r["forward_s"]["median"], _ms)
    )
    out.append(
        table(
            "Cold start: first training step incl. compilation (s)",
            lambda r: r.get("cold_first_step_s"),
            lambda v: f"{v:.2f}",
            spd=False,
        )
    )
    out.append(
        table(
            "Peak process memory (MB RSS, whole process)",
            lambda r: r.get("peak_rss_mb"),
            lambda v: f"{v:,.0f}",
            spd=False,
        )
    )
    if any(r.get("e2e") for r in res):
        out.append(
            table(
                "End-to-end epoch: DataLoader + training (s; excludes featurisation)",
                lambda r: (r.get("e2e") or {}).get("epoch_s"),
                lambda v: f"{v:.2f}",
            )
        )
        out.append(
            table(
                "Featurisation of the epoch's molecules (s; RDKit + Chemprop, same for all modes)",
                lambda r: (r.get("e2e") or {}).get("featurise_s"),
                lambda v: f"{v:.2f}",
                spd=False,
            )
        )
        if any((r.get("e2e") or {}).get("lightning_fit_1_epoch_s") for r in res):
            out.append(
                table(
                    "Lightning fit, 1 epoch incl. trainer setup (s)",
                    lambda r: (r.get("e2e") or {}).get("lightning_fit_1_epoch_s"),
                    lambda v: f"{v:.2f}",
                )
            )
    eq = doc.get("equivalence", [])
    if eq:
        rows = "".join(
            f"<tr><td class='l'>{_e(_fmt_key(tuple(e['key'][:6])))}{' · compiled' if e['key'][6] else ''}</td>"
            f"<td>{e['steps']}</td><td>{e['max_abs_pred_diff']:.2e}</td><td>{e['max_rel_pred_diff']:.2e}</td>"
            f"<td>{e['max_abs_loss_diff']:.2e}</td>"
            f"<td>{'n/a' if e.get('max_rel_infer_diff') is None else format(e['max_rel_infer_diff'], '.2e')}</td>"
            f"<td class='{'good' if e['passed'] else 'bad'}'>{'PASS' if e['passed'] else 'FAIL'}</td></tr>"
            for e in eq
        )
        out.append(
            "<h3>Predictive equivalence (same seed, same data, N Adam steps, float32)</h3>"
            "<p class='muted'>Max difference between predictions of the accelerated and original model after "
            "training, relative to the largest prediction magnitude. Threshold: 1e-3. Differences arise from "
            "floating-point summation order.</p>"
            "<div class='scroll'><table><tr><th class='l'>Configuration</th><th>steps</th><th>max |Δ pred|</th>"
            f"<th>max rel Δ pred</th><th>max |Δ loss|</th><th>inference max rel Δ</th><th>result</th></tr>{rows}</table></div>"
        )
    if failed:
        out.append(
            "<h3>Failed or skipped runs</h3><ul>"
            + "".join(
                f"<li><code>{_e(f['cfg'].get('mode'))}</code> {_e(_fmt_key(_key(f['cfg'])))}: {_e(f.get('error'))}</li>"
                for f in failed
            )
            + "</ul>"
        )
    return "\n".join(out)


def _summary(docs):
    if len(docs) < 2:
        return ""
    keys = {}
    for i, d in enumerate(docs):
        by = {}
        for r in d["results"]:
            if r.get("ok"):
                by.setdefault(_key(r["cfg"]), {})[r["cfg"]["mode"]] = r
        for k, v in by.items():
            if "eager" in v and "accel" in v and k[0] == "cpu":
                keys.setdefault(k, {})[i] = (
                    v["eager"]["train_step_s"]["median"] / v["accel"]["train_step_s"]["median"]
                )
    head = "".join(f"<th>{_e(d.get('name'))}</th>" for d in docs)
    rows = []
    for k, d in sorted(keys.items()):
        cells = ""
        for i in range(len(docs)):
            if i not in d:
                cells += "<td>–</td>"
                continue
            v = d[i]
            cls = "good" if v > 1.05 else ("bad" if v < 0.95 else "muted")
            cells += f'<td><span class="{cls}">{v:.2f}×</span></td>'
        rows.append(f"<tr><td class='l'>{_e(_fmt_key(k))}</td>{cells}</tr>")
    return (
        "<h2>Cross-machine summary</h2><p>Training-step speedup of <code>chemprop_accel</code> over original Chemprop "
        "(median step time ratio; &gt;1 is faster). Shared CI runners are noisy and have few cores: treat these as "
        "a regression signal, not as headline performance.</p>"
        f"<div class='scroll'><table><tr><th class='l'>Configuration</th>{head}</tr>{''.join(rows)}</table></div>"
    )


def render(docs: list[dict], title: str = "chemprop_accel benchmark report") -> str:
    nav = " · ".join(f"<a href='#m{i}'>{_e(d.get('name'))}</a>" for i, d in enumerate(docs))
    all_eq = [e for d in docs for e in d.get("equivalence", [])]
    verdict = ""
    if all_eq:
        nfail = sum(not e["passed"] for e in all_eq)
        verdict = (
            f"<p><b>Predictive equivalence:</b> <span class='{'bad' if nfail else 'good'}'>"
            f"{len(all_eq) - nfail}/{len(all_eq)} checks passed</span>.</p>"
        )
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body><h1>{_e(title)}</h1>"
        f"<p class='muted'>Machines: {nav}</p>{verdict}"
        "<div class='note'><b>How to read this.</b> Each row is a fresh process with a fixed seed and a built-in molecule pool. "
        "Timings are medians over repeated steps after warm-up and are specific to the hardware listed; speedups on "
        "one CPU do not generalise to others. Speedup = original time ÷ accelerated time.</div>"
        f"{_summary(docs)}{''.join(_machine_section(d, i) for i, d in enumerate(docs))}"
        "<hr><p class='muted'>Generated by <code>chemprop_accel.bench</code>.</p></body></html>"
    )


def write_report(docs: list[dict], path, title: str = "chemprop_accel benchmark report") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(docs, title), encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="results JSON files")
    ap.add_argument("-o", "--output", default="report.html")
    ap.add_argument("--title", default="chemprop_accel benchmark report")
    a = ap.parse_args(argv)
    docs = [json.loads(Path(p).read_text()) for p in sorted(a.inputs)]
    print(write_report(docs, a.output, a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
