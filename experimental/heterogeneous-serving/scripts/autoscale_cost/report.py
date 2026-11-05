#!/usr/bin/env python3
"""Assemble results/autoscale-cost/<rev>/summary.json and the README tables (markdown between
<!-- cost-tables:begin --> / <!-- cost-tables:end --> in README.md) from the per-step outputs.

    /tmp/cm-venv/bin/python scripts/autoscale_cost/report.py --dir results/autoscale-cost/<rev>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

LABELS = ["gemma-x0.35", "gemma-raw", "gemma-x0.35-speed0.965", "mistral24b-raw", "qwen-raw"]


def f(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def validation_table(rows, title):
    out = [f"**{title}**", "",
           "| run | n | metric | burst SLO viol. % | run SLO viol. % | burst TTFT P99 (s) | max queue | queue recovery (s) "
           "| instance-s | 429 | triggers (s) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for who in ("real", "sim"):
            m = r[who]
            out.append(f"| {r['rid'] if who == 'real' else ''} | {r['n'] if who == 'real' else ''} | {who} "
                       f"| {f(m['burst_viol_pct'])} | {f(m['run_viol_pct'])} | {f(m['burst_ttft_p99'])} "
                       f"| {m['max_queue']} | {f(m['queue_recovery_s'], 0)} | {f(m['instance_s'], 0)} | {m['n_429']} "
                       f"| {', '.join(str(int(round(t))) for t in m['triggers_s'])} |")
        e = r["err"]
        out.append("| | | error | " + " | ".join(
            (f"{e[k]['abs']:+.1f}" + (f" ({100 * e[k]['rel']:+.0f}%)" if e[k]["rel"] is not None and k not in
                                       ("burst_viol_pct", "run_viol_pct") else "")) if e.get(k) else "-"
            for k in ("burst_viol_pct", "run_viol_pct", "burst_ttft_p99", "max_queue", "queue_recovery_s",
                      "instance_s", "n_429")) + " | |")
    return out


def savings_table(a, family="reactive"):
    out = ["| boot pair | target | stock chip-h/day (config) | PM chip-h/day (config) | saving (single config) "
           "| saving (convex hull) | PM attainment at stock's chip-h |", "|---|---|---|---|---|---|---|"]
    for pname, P in a["pairs"].items():
        for t in (99.0, 98.0, 95.0, 90.0):
            r = P.get(f"{family}@{t}")
            if not r or not r.get("stock") or not r.get("pm"):
                out.append(f"| {pname} | {t:.0f}% | n/a | n/a | | | |")
                continue
            h = r.get("hull") or {}
            pm_b = r.get("pm_at_stock_budget") or {}
            out.append(f"| {pname} | {t:.0f}% | {f(r['stock']['chip_h_day'])} ({r['stock']['cfg']}) "
                       f"| {f(r['pm']['chip_h_day'])} ({r['pm']['cfg']}) | {f(r.get('saving_pct'))}% "
                       f"({f(r.get('saving_chip_h_day'))}) | {f(h.get('saving_pct'))}% ({f(h.get('saving_chip_h_day'))}) "
                       f"| {f(pm_b.get('att'), 2)}% vs {f(r['stock']['att'], 2)}% "
                       f"(+{f(r.get('attainment_gain_pp'), 2)} pp) |")
    return out


def decomposition_table(a):
    out = ["| pair @ target | who | total | load-following | headroom | keep-alive | draining | billed boot "
           "| scale-ups/day |", "|---|---|---|---|---|---|---|---|---|"]
    for pname, P in a["pairs"].items():
        for t in (99.0, 95.0):
            r = P.get(f"reactive@{t}")
            if not r or not r.get("stock") or not r.get("pm"):
                continue
            for who in ("stock", "pm"):
                x = r[who]
                out.append(f"| {pname} @ {t:.0f}% | {who} | {f(x['chip_h_day'])} | {f(x['base'])} | {f(x['head'])} "
                           f"| {f(x['keep'])} | {f(x['drain'])} | {f(x['boot'])} | {f(x['acq_per_day'])} |")
            s, p = r["stock"], r["pm"]
            out.append(f"| | **stock - PM** | **{f(s['chip_h_day'] - p['chip_h_day'])}** "
                       f"| {f(s['base'] - p['base'])} | {f(s['head'] - p['head'])} | {f(s['keep'] - p['keep'])} "
                       f"| {f(s['drain'] - p['drain'])} | {f(s['boot'] - p['boot'])} | |")
    return out


def boot_curve_table(a):
    out = ["| B (s) | " + " | ".join(str(c["B"]) for c in a["boot_curve"]) + " |",
           "|---|" + "---|" * len(a["boot_curve"])]
    for t in (99.0, 95.0):
        out.append(f"| chip-h/day @ {t:.0f}% (hull) | " + " | ".join(f(c.get(f"hull@{t}"), 0) for c in a["boot_curve"])
                   + " |")
    return out


def per_day_table(a):
    out = ["| pair @ target | stock cfg | PM cfg | median saving chip-h/day [p10, p90] | median % [p10, p90] "
           "| days PM cheaper | min daily attainment stock / PM |", "|---|---|---|---|---|---|---|"]
    for k, d in a["per_day"].items():
        s, p = d["saving_chip_h"], d["saving_pct"]
        out.append(f"| {k} | {d['stock_cfg']} | {d['pm_cfg']} | {f(s['median'])} [{f(s['p10'])}, {f(s['p90'])}] "
                   f"| {f(p['median'])}% [{f(p['p10'])}, {f(p['p90'])}] | {s['days_positive']}/{s['days']} "
                   f"| {f(d['attainment_stock']['min'], 2)} / {f(d['attainment_pm']['min'], 2)} |")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    summ = {"rev": d.name, "slo": {"ttft_s": cm.SLO_TTFT, "tpot_s": cm.SLO_TPOT}, "C_rps_gemma": cm.C_RPS,
            "chips_per_instance": cm.CHIPS, "cap": cm.CAP}
    summ["calibration"] = json.loads((d / "calibration_fit.json").read_text())
    summ["validation"] = json.loads((d / "validation.json").read_text())
    if (d / "validation-speed0.965.json").exists():
        summ["validation_speed0.965"] = json.loads((d / "validation-speed0.965.json").read_text())
    summ["analyses"] = {}
    for lab in LABELS:
        p = d / f"analysis-{lab}.json"
        if p.exists():
            x = json.loads(p.read_text())
            for k in x.get("per_day", {}).values():  # keep the per-day arrays out of the summary (in analysis-*.json)
                for kk in ("days", "stock_chip_h", "pm_chip_h", "stock_att", "pm_att"):
                    k.pop(kk, None)
            summ["analyses"][lab] = x
    for name in ("comparator.json", "capacity-mistral24b.json", "capacity-qwen.json"):
        if (d / name).exists():
            summ[name.removesuffix(".json")] = json.loads((d / name).read_text())
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    # tables
    T = []
    T += validation_table(summ["validation"], "Validation, calibrated model (speed 1.0)")
    if "validation_speed0.965" in summ:
        T += [""] + validation_table(summ["validation_speed0.965"], "Validation, capacity-matched model (speed 0.965)")
    for lab, x in summ["analyses"].items():
        T += ["", f"### {lab}: reactive headroom policy, TPUFlex vs stock at matched attainment "
                  f"({x['days_included']} days, {x['n_requests']} requests)", ""]
        T += savings_table(x)
        T += ["", f"Best of reactive + HPA-fast + HPA-WVA120 + static fleet ({lab}):", ""]
        T += savings_table(x, "best")
        if lab in ("gemma-x0.35", "gemma-raw"):
            T += ["", f"Decomposition ({lab}, chip-hours/day):", ""] + decomposition_table(x)
            T += ["", f"Per-day distribution ({lab}):", ""] + per_day_table(x)
            T += ["", f"Cost vs boot time ({lab}, reactive policy, convex hull):", ""] + boot_curve_table(x)
            b = x["bounds"]
            T += ["", f"Bounds ({lab}): static fleet @99% {b['static@99.0'] and b['static@99.0']['chip_h_day']} "
                      f"chip-h/day ({b['static@99.0'] and b['static@99.0']['cfg']}), @95% "
                      f"{b['static@95.0'] and b['static@95.0']['chip_h_day']}; oracle (B=0, clairvoyant) @99% "
                      f"{b['oracle@99.0'] and b['oracle@99.0']['chip_h_day']}, @95% "
                      f"{b['oracle@95.0'] and b['oracle@95.0']['chip_h_day']}."]
    G = ["| workload | days | requests | C (req/s) | boot pair | reactive saving @99% | @95% | @90% "
         "| best-of saving @99% / @95% (incl. static) | PM attainment at stock's chip-h @95% (reactive) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ("gemma-x0.35", "gemma-raw", "mistral24b-raw", "qwen-raw"):
        x = summ["analyses"].get(lab)
        if not x:
            continue
        for pname in ("standby", "new capacity (e2e GKE)"):
            P = x["pairs"][pname]
            cell = lambda t: (f"{f((P[f'reactive@{t}'].get('hull') or {}).get('saving_pct'))}%"  # noqa: E731
                              if P.get(f"reactive@{t}") else "-")
            r95 = P.get("reactive@95.0") or {}
            g = (f"{f((r95.get('pm_at_stock_budget') or {}).get('att'), 2)}% vs {f((r95.get('stock') or {}).get('att'), 2)}%"
                 if r95.get("stock") else "-")
            bcell = lambda t: (f"{f((P[f'best@{t}'].get('hull') or {}).get('saving_pct'))}%"  # noqa: E731
                               if P.get(f"best@{t}") else "-")
            G.append(f"| {lab} | {x['days_included']} | {x['n_requests']} | {f(x.get('C', cm.C_RPS), 3)} | {pname} "
                     f"| {cell(99.0)} | {cell(95.0)} | {cell(90.0)} | {bcell(99.0)} / {bcell(95.0)} | {g} |")
    gmd = "\n".join(G)
    (d / "generality.md").write_text(gmd + "\n")
    md = "\n".join(T)
    (d / "tables.md").write_text(md + "\n")
    rd = d / "README.md"
    if rd.exists():
        txt = rd.read_text()
        new = re.sub(r"<!-- cost-tables:begin -->.*<!-- cost-tables:end -->",
                     "<!-- cost-tables:begin -->\n" + md + "\n<!-- cost-tables:end -->", txt, flags=re.S)
        new = re.sub(r"<!-- generality:begin -->.*<!-- generality:end -->",
                     "<!-- generality:begin -->\n" + gmd + "\n<!-- generality:end -->", new, flags=re.S)
        rd.write_text(new)
    print(md)


if __name__ == "__main__":
    main()
