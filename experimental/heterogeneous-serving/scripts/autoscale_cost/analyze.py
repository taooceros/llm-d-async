#!/usr/bin/env python3
"""Cost-vs-SLO analysis of sweep results: frontiers, savings at matched attainment / chip-hours, per-day
distribution, decomposition (headroom / keep-alive / billed boot), boot-time curve.

    /tmp/cm-venv/bin/python scripts/autoscale_cost/analyze.py --sweep /tmp/cm/sweep-g035.npz --label gemma-x0.35 \
        --out results/autoscale-cost/<rev>
Writes <out>/analysis-<label>.json and <out>/figures/*-<label>.{png,pdf}.

Definitions
  attainment   % of requests (arrival in an included day) meeting TTFT <= 0.788 s and TPOT <= 66.3 ms; 429s
               (600 s admission timeout) and unfinished requests count as misses
  chip-hours   8 chips x billed instance-hours (acquisition -> release, booting and draining included), included days
  frontier     per (policy family, boot scenario): the cheapest single configuration with attainment >= target
               (no time-sharing between configurations)
  included day a day of the trace with <= 2 empty hours in the unthinned trace (drops the 49 h data gap of days
               303-305 and the partial last day)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
import sweep as sw  # noqa: E402

TARGETS = [99.0, 98.0, 95.0, 90.0]
PAIRS = [("standby", "stock-standby", "pm-standby"), ("new capacity (e2e GKE)", "stock-new", "pm-new"),
         ("new capacity, stockout GKE", "stock-new-stockout", "pm-new-stockout")]
SCEN_B = {k: (lo + hi) / 2 for k, (lo, hi) in sw.SCENARIOS.items()}
COL = {"stock-standby": "#d62728", "pm-standby": "#1f77b4", "stock-new": "#ff7f0e", "pm-new": "#9467bd",
       "stock-new-stockout": "#8c564b", "pm-new-stockout": "#17becf"}


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return meta, {k: z[k] for k in z.files if k != "meta"}


def included_days(meta):
    w = sw.load_workload(meta["workload"], 1.0)
    nd = meta["ndays"]
    if nd == 1:  # a single short trace (qwen: 2 h): keep it; "per day" then means "per trace"
        return np.ones(1, bool)
    h = np.bincount(np.minimum((w["t"] // 3600).astype(int), nd * 24 - 1), minlength=nd * 24).reshape(nd, 24)
    return (h == 0).sum(1) <= 2


def totals(R, inc):
    n = R["n"][:, inc].sum(1)
    met = R["met"][:, inc].sum(1)
    nd = inc.sum()
    f = lambda k: cm.CHIPS * R[k][:, inc].sum(1) / 3600.0 / nd  # chip-hours per day  # noqa: E731
    return {"att": 100.0 * met / np.maximum(n, 1), "chip_h_day": f("bill_s"), "boot": f("boot_s"), "base": f("base_s"),
            "head": f("head_s"), "keep": f("keep_s"), "drain": f("drain_s"),
            "rej_pct": 100.0 * R["rej"][:, inc].sum(1) / np.maximum(n, 1),
            "acq_day": R["n_acq"] / R["n"].shape[1]}


def family_idx(cfgs, family, boot):
    return [j for j, c in enumerate(cfgs) if c["policy"] == family and (boot is None or c["boot"] == boot)]


def best_at(T, idx, target):
    """Cheapest config in idx with attainment >= target -> j or None."""
    ok = [j for j in idx if T["att"][j] >= target - 1e-9]
    return min(ok, key=lambda j: T["chip_h_day"][j]) if ok else None


def hull_cost(T, idx, target):
    """Cost at `target` on the lower convex hull of (attainment, chip-h/day): reachable by time-sharing two
    configurations (e.g. alternating by day); smooths the grid's staircase."""
    pts = sorted({(float(T["att"][j]), float(T["chip_h_day"][j])) for j in idx})
    if not pts:
        return None
    hull = []
    for p in pts:  # lower hull, x ascending
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= \
                (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    # keep the increasing part (cost rises with attainment)
    lo = int(np.argmin([h[1] for h in hull]))
    hull = hull[lo:]
    if not hull or target > hull[-1][0]:
        return None
    if target <= hull[0][0]:
        return hull[0][1]
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        if x0 <= target <= x1:
            return y0 + (y1 - y0) * (target - x0) / (x1 - x0)
    return None


def best_under(T, idx, budget):
    ok = [j for j in idx if T["chip_h_day"][j] <= budget * 1.001]  # 0.1%: same (h, T) bills ~identically
    return max(ok, key=lambda j: T["att"][j]) if ok else None


def pareto(T, idx):
    pts = sorted(idx, key=lambda j: (-T["att"][j], T["chip_h_day"][j]))
    out, best = [], math.inf
    for j in pts:
        if T["chip_h_day"][j] < best - 1e-12:
            out.append(j)
            best = T["chip_h_day"][j]
    return sorted(out, key=lambda j: T["att"][j])


def cdesc(c):
    if c["policy"] in ("headroom", "oracle"):
        return f"{c['policy']} h={c['h_c']}C T={c['T']}s"
    if c["policy"].startswith("hpa"):
        return f"{c['policy']} u={c['u']}"
    return f"static N={c['n']}"


def row(T, cfgs, j):
    if j is None:
        return None
    return {"cfg": cdesc(cfgs[j]), "att": round(float(T["att"][j]), 3), "chip_h_day": round(float(T["chip_h_day"][j]), 2),
            "boot": round(float(T["boot"][j]), 2), "base": round(float(T["base"][j]), 2),
            "head": round(float(T["head"][j]), 2), "keep": round(float(T["keep"][j]), 2),
            "drain": round(float(T["drain"][j]), 2), "acq_per_day": round(float(T["acq_day"][j]), 1),
            "rej_pct": round(float(T["rej_pct"][j]), 4)}


def families(cfgs):
    return {"reactive": lambda b: family_idx(cfgs, "headroom", b),
            "hpa-fast": lambda b: family_idx(cfgs, "hpa-fast", b),
            "hpa-wva120": lambda b: family_idx(cfgs, "hpa-wva120", b),
            "best": lambda b: family_idx(cfgs, "headroom", b) + family_idx(cfgs, "hpa-fast", b)
            + family_idx(cfgs, "hpa-wva120", b) + family_idx(cfgs, "static", None)}  # static never boots


def analyze(meta, R, out, label):
    cfgs = meta["configs"]
    inc = included_days(meta)
    T = totals(R, inc)
    fam = families(cfgs)
    res = {"label": label, "workload": meta["workload"], "frac": meta["frac"], "speed": meta.get("speed", 1.0),
           "C": meta.get("C", cm.C_RPS),
           "n_requests": meta["n_requests"], "days_total": int(meta["ndays"]), "days_included": int(inc.sum()),
           "days_excluded": [300 + int(d) for d in np.where(~inc)[0]] if meta["workload"] != "qwen" else
           [int(d) for d in np.where(~inc)[0]]}
    # bounds
    stat = family_idx(cfgs, "static", None)
    orc = family_idx(cfgs, "oracle", None)
    res["bounds"] = {f"static@{t}": row(T, cfgs, best_at(T, stat, t)) for t in TARGETS}
    res["bounds"].update({f"oracle@{t}": row(T, cfgs, best_at(T, orc, t)) for t in TARGETS})
    res["bounds"]["static_max"] = row(T, cfgs, stat[-1]) if stat else None
    # headline pairs
    res["pairs"] = {}
    for pname, sb, pb in PAIRS:
        P = {}
        for fname, f in fam.items():
            for t in TARGETS:
                js, jp = best_at(T, f(sb), t), best_at(T, f(pb), t)
                r = {"stock": row(T, cfgs, js), "pm": row(T, cfgs, jp)}
                if js is not None and jp is not None:
                    cs, cp = T["chip_h_day"][js], T["chip_h_day"][jp]
                    r["saving_chip_h_day"] = round(float(cs - cp), 2)
                    r["saving_pct"] = round(float(100 * (cs - cp) / cs), 2)
                hs, hp = hull_cost(T, f(sb), t), hull_cost(T, f(pb), t)
                if hs is not None and hp is not None:
                    r["hull"] = {"stock": round(hs, 2), "pm": round(hp, 2), "saving_chip_h_day": round(hs - hp, 2),
                                 "saving_pct": round(100 * (hs - hp) / hs, 2)}
                if js is not None:
                    jm = best_under(T, f(pb), T["chip_h_day"][js])
                    r["pm_at_stock_budget"] = row(T, cfgs, jm)
                    if jm is not None:
                        r["attainment_gain_pp"] = round(float(T["att"][jm] - T["att"][js]), 3)
                        r["violation_reduction_x"] = round(float((100 - T["att"][js]) / max(100 - T["att"][jm], 1e-6)), 2)
                P[f"{fname}@{t}"] = r
        res["pairs"][pname] = P
    # boot-time curve (reactive family; hpa families at B0 and the named scenarios only)
    curve = []
    for b in sw.B_SWEEP:
        idx = fam["reactive"](f"B{b}")
        if not idx:  # --no-bsweep runs
            continue
        curve.append({"B": b, **{f"hull@{t}": (round(v, 2) if (v := hull_cost(T, idx, t)) is not None else None)
                                 for t in TARGETS},
                      **{f"chip_h_day@{t}": (round(float(T["chip_h_day"][j]), 2) if (j := best_at(T, idx, t))
                                                      is not None else None) for t in TARGETS},
                      **{f"cfg@{t}": (cdesc(cfgs[j]) if (j := best_at(T, idx, t)) is not None else None)
                         for t in TARGETS}})
    res["boot_curve"] = curve
    res["scenario_points"] = {s: {"B_mean": SCEN_B[s], **{f"hull@{t}": (round(v, 2) if (v := hull_cost(
                                      T, fam["reactive"](s), t)) is not None else None) for t in TARGETS}, **{f"chip_h_day@{t}": (round(float(T["chip_h_day"][j]), 2) if
                                  (j := best_at(T, fam["reactive"](s), t)) is not None else None) for t in TARGETS}}
                              for s in sw.SCENARIOS}
    # per-day distribution at matched attainment (reactive family)
    res["per_day"] = {}
    nd = inc.sum()
    day_ids = np.where(inc)[0]
    for pname, sb, pb in PAIRS:
        for t in (99.0, 95.0):
            js, jp = best_at(T, fam["reactive"](sb), t), best_at(T, fam["reactive"](pb), t)
            if js is None or jp is None:
                continue
            cs = cm.CHIPS * R["bill_s"][js, inc] / 3600
            cp = cm.CHIPS * R["bill_s"][jp, inc] / 3600
            att_s = 100 * R["met"][js, inc] / np.maximum(R["n"][js, inc], 1)
            att_p = 100 * R["met"][jp, inc] / np.maximum(R["n"][jp, inc], 1)
            sv = cs - cp
            res["per_day"][f"{pname}@{t}"] = {
                "stock_cfg": cdesc(cfgs[js]), "pm_cfg": cdesc(cfgs[jp]),
                "saving_chip_h": {"median": float(np.median(sv)), "p10": float(np.percentile(sv, 10)),
                                  "p90": float(np.percentile(sv, 90)), "min": float(sv.min()), "max": float(sv.max()),
                                  "mean": float(sv.mean()), "days_positive": int((sv > 0).sum()), "days": int(nd)},
                "saving_pct": {"median": float(np.median(100 * sv / cs)), "p10": float(np.percentile(100 * sv / cs, 10)),
                               "p90": float(np.percentile(100 * sv / cs, 90))},
                "attainment_stock": {"median": float(np.median(att_s)), "min": float(att_s.min())},
                "attainment_pm": {"median": float(np.median(att_p)), "min": float(att_p.min())},
                "days": [int(300 + d) for d in day_ids], "stock_chip_h": cs.round(2).tolist(),
                "pm_chip_h": cp.round(2).tolist(), "stock_att": att_s.round(3).tolist(), "pm_att": att_p.round(3).tolist()}
    # best keep-alive / headroom vs B at 99% (what the policy needs)
    res["policy_vs_B"] = [{"B": c["B"], "cfg@99": c["cfg@99.0"], "cfg@95": c["cfg@95.0"]} for c in curve]
    res["derivation"] = derivations(meta, res, inc)
    (out / f"analysis-{label}.json").write_text(json.dumps(res, indent=1))
    plots(meta, R, T, res, out / "figures", label, inc)
    return res


def derivations(meta, res, inc):
    """Closed-form views of the three cost terms, from the trace itself (no simulation).

    headroom: a reactive policy sees load R(t) (60 s window) and its new capacity serves from t + B, so covering
      the load growth within B needs h >= rise_B(t) = max_{u in (t, t+B]} R(u) - R(t); we report quantiles of
      rise_B over time (in units of C) and the matching cost 24 * 8 * h / C chip-hours/day.
    billed boot: every scale-up bills 8 chips for B before serving: 8 * B * n_up / 3600 chip-hours.
    keep-alive: ski rental; releasing an idle instance that is needed again g seconds later costs one more boot
      (8 B chip-seconds billed plus a B-long capacity gap), keeping it costs 8 g: break-even keep-alive T* ~ B
      (the deterministic 2-competitive rule), so the keep-alive surplus that buys back an SLO miss scales with B.
    """
    from scipy.ndimage import maximum_filter1d
    w = sw.load_workload(meta["workload"], meta["frac"])
    C = meta.get("C", cm.C_RPS)
    nsec = int(np.ceil(w["t"][-1])) + 1
    cnt = np.bincount(w["t"].astype(np.int64), minlength=nsec)[:nsec].astype(float)
    R = np.convolve(cnt, np.ones(60) / 60.0, "full")[:nsec]
    day = np.minimum(np.arange(nsec) // 86400, meta["ndays"] - 1)
    keep = inc[day]
    out = {"rise_over_B_in_C": {}}
    for name, B in (("pm-standby", 63.0), ("stock-standby", 377.0), ("pm-new", 264.7), ("stock-new", 580.6)):
        b = int(B)
        mf = maximum_filter1d(R, size=b, mode="nearest")  # centred: covers [c - b//2, c + (b-1)//2]
        fut = mf[np.minimum(np.arange(nsec) + 1 + b // 2, nsec - 1)]  # max of R over (t, t + b]
        rise = np.clip(fut - R, 0, None)[keep] / C
        out["rise_over_B_in_C"][name] = {"B": B, **{f"q{q}": round(float(np.percentile(rise, q)), 3)
                                                     for q in (90, 95, 99, 99.9)}}
    bt = {}
    for pname, sb, pb in PAIRS:
        for t in (99.0, 95.0):
            r = res["pairs"][pname].get(f"reactive@{t}")
            if not r or not r["stock"] or not r["pm"]:
                continue
            for who, sc_ in (("stock", sb), ("pm", pb)):
                x = r[who]
                bt[f"{pname}@{t}/{who}"] = {"acq_per_day": x["acq_per_day"], "B_mean": SCEN_B[sc_],
                                            "formula_chip_h_day": round(8 * SCEN_B[sc_] * x["acq_per_day"] / 3600, 2),
                                            "simulated_boot_chip_h_day": x["boot"]}
    out["billed_boot"] = bt
    return out


def plots(meta, R, T, res, figdir, label, inc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    cfgs = meta["configs"]
    fam = families(cfgs)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / f"{name}-{label}.png", dpi=130)
        fig.savefig(figdir / f"{name}-{label}.pdf")
        plt.close(fig)

    # frontiers
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    stat = pareto(T, family_idx(cfgs, "static", None))
    orc = pareto(T, family_idx(cfgs, "oracle", None))
    for ax, (pname, sb, pb) in zip(axes, PAIRS):
        for s in (sb, pb):
            for fname, ls in (("reactive", "-"), ("hpa-fast", ":"), ("hpa-wva120", "--")):
                pts = pareto(T, fam[fname](s))
                if not pts:
                    continue
                ax.plot([T["att"][j] for j in pts], [T["chip_h_day"][j] for j in pts], ls, color=COL[s], marker="o"
                        if fname == "reactive" else None, ms=3, lw=1.6 if fname == "reactive" else 1.0,
                        label=f"{s} {fname} (B~{SCEN_B[s]:.0f}s)")
        ax.plot([T["att"][j] for j in stat], [T["chip_h_day"][j] for j in stat], "-", color="k", lw=0.8, marker=".",
                label="static fleet")
        ax.plot([T["att"][j] for j in orc], [T["chip_h_day"][j] for j in orc], "-", color="tab:green", lw=1.0,
                label="oracle (B=0, clairvoyant)")
        ax.set_xlim(85, 100)
        ax.set_xlabel("SLO attainment (% requests meeting TTFT and TPOT SLO)")
        ax.set_title(pname, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper left")
        for t in (95, 99):
            ax.axvline(t, color="0.7", lw=0.6)
    axes[0].set_ylabel("chip-hours per day (TPU v5e, 8 chips/instance)")
    ymax = max([T["chip_h_day"][j] for j in pareto(T, fam["reactive"]("stock-new-stockout")) if T["att"][j] >= 85] or [1])
    axes[0].set_ylim(0, ymax * 1.15)
    fig.suptitle(f"Cost vs SLO frontier ({label}, {res['days_included']} days)", fontsize=11)
    save(fig, "frontier")

    # boot-time curve
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bs = [c["B"] for c in res["boot_curve"]]
    for t, mk in ((99.0, "o"), (95.0, "s")):
        ax.plot(bs, [c[f"chip_h_day@{t}"] for c in res["boot_curve"]], "-", marker=mk, ms=4, color="0.3",
                label=f"reactive, cheapest single config @ {t:.0f}%")
        ax.plot(bs, [c[f"hull@{t}"] for c in res["boot_curve"]], ":", color="0.3", lw=1,
                label=f"reactive, convex hull (time-sharing) @ {t:.0f}%")
        for s, p in res["scenario_points"].items():
            if p[f"chip_h_day@{t}"] is not None:
                ax.scatter([p["B_mean"]], [p[f"chip_h_day@{t}"]], color=COL[s], marker=mk, s=60, zorder=5,
                           label=s if t == 99.0 else None, edgecolor="k", lw=0.5)
    ax.set_xlabel("boot time B: trigger -> first served (s)")
    ax.set_ylabel("chip-hours per day")
    ax.set_title(f"Cost vs boot time ({label})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "boot_curve")

    # per-day saving
    keys = [k for k in res["per_day"]]
    if keys:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        ax = axes[0]
        data = [np.array(res["per_day"][k]["stock_chip_h"]) - np.array(res["per_day"][k]["pm_chip_h"]) for k in keys]
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(keys) + 1), [k.replace("new capacity", "new").replace(", stockout GKE", "/stockout")
                                                 for k in keys], rotation=20, fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("chip-hours saved per day (stock - PM config)")
        ax.set_title(f"Per-day saving, configs matched on whole-trace attainment ({label})", fontsize=9)
        ax = axes[1]
        k = keys[0]
        d = res["per_day"][k]
        ax.bar(np.arange(len(d["days"])) - 0.2, d["stock_chip_h"], 0.4, color=COL["stock-standby"], label="stock")
        ax.bar(np.arange(len(d["days"])) + 0.2, d["pm_chip_h"], 0.4, color=COL["pm-standby"], label="TPUFlex")
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(d["days"])), d["stock_att"], color=COL["stock-standby"], lw=0.8)
        ax2.plot(np.arange(len(d["days"])), d["pm_att"], color=COL["pm-standby"], lw=0.8)
        ax2.set_ylabel("daily attainment (%)")
        ax.set_xlabel("day index")
        ax.set_ylabel("chip-hours")
        ax.set_title(f"{k}: {d['stock_cfg']} vs {d['pm_cfg']}", fontsize=8)
        ax.legend(fontsize=7)
        save(fig, "per_day")

    # decomposition
    fig, ax = plt.subplots(figsize=(9, 4.5))
    comps = [("base", "load-following ceil(rate/C)", "#999999"), ("head", "headroom", "#f4a582"),
             ("keep", "keep-alive surplus", "#92c5de"), ("drain", "draining", "#d1e5f0"),
             ("boot", "billed boot", "#b2182b")]
    bars, labels = [], []
    for pname, sb, pb in PAIRS:
        for t in (99.0, 95.0):
            r = res["pairs"][pname].get(f"reactive@{t}")
            if not r or not r["stock"] or not r["pm"]:
                continue
            for who in ("stock", "pm"):
                bars.append(r[who])
                labels.append(f"{pname.split(' ')[0]}{'/so' if 'stockout' in pname else ''}\n{who}@{t:.0f}")
    x = np.arange(len(bars))
    bottom = np.zeros(len(bars))
    for key, name, col in comps:
        v = np.array([b[key] for b in bars])
        ax.bar(x, v, bottom=bottom, color=col, label=name, edgecolor="k", lw=0.3)
        bottom += v
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.set_ylabel("chip-hours per day")
    ax.set_title(f"Where the chip-hours go (reactive frontier configs, {label})", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, "decomposition")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta, R = load(a.sweep)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res = analyze(meta, R, Path(a.out), a.label)
    for pname in res["pairs"]:
        for k, r in res["pairs"][pname].items():
            if k.startswith(("reactive", "best")):
                print(pname, k, r.get("stock", {}) and r["stock"]["chip_h_day"], r.get("pm", {}) and r["pm"]["chip_h_day"],
                      r.get("saving_pct"), r.get("attainment_gain_pp"),
                      (r.get("stock") or {}).get("cfg"), (r.get("pm") or {}).get("cfg"))


if __name__ == "__main__":
    main()
