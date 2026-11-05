#!/usr/bin/env python3
"""Real-world comparator (descriptive only): how Chutes itself provisioned unsloth/gemma-3-27b-it over days 300-366
of its trace, against demand.

Per minute: demand = requests with token counts started in the minute; active instances = distinct instance_id
that started >= 1 request (any row, incl. failed) in the minute (a lower bound: idle instances are invisible);
provisioned instances = instances between their first and last request (lifetime proxy; an upper-ish bound if an
instance was idle-but-gone in between). Chutes' hardware per instance is unknown (heterogeneous GPUs), so
over-provisioning is relative to the per-instance load Chutes itself sustained: c_obs = 99th percentile over minutes
of demand / active instances. Ratio = provisioned / ceil(demand / c_obs). Burst lag: for each burst of
gemma3_bursts.csv (pm-render burst list, all in days 300+), minutes from the burst start until the active /
provisioned count first exceeds its pre-burst level (median of the 10 minutes before).

    /tmp/cm-venv/bin/python scripts/autoscale_cost/comparator.py --out results/autoscale-cost/<rev>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as dm  # noqa: E402


def per_minute(s_valid, s_all, inst_all, m0, nmin):
    dem = np.bincount((s_valid // 60).astype(np.int64) - m0, minlength=nmin)[:nmin]
    mins = (s_all // 60).astype(np.int64) - m0
    key = np.unique(mins * (inst_all.max() + 1) + inst_all)
    act = np.bincount(key // (inst_all.max() + 1), minlength=nmin)[:nmin]
    # lifetime-based provisioned count
    order = np.argsort(inst_all, kind="stable")
    ii, mm = inst_all[order], mins[order]
    starts = np.r_[0, np.where(np.diff(ii))[0] + 1]
    first = np.minimum.reduceat(mm, starts)
    last = np.maximum.reduceat(mm, starts)
    delta = np.zeros(nmin + 1, np.int64)
    np.add.at(delta, first, 1)
    np.add.at(delta, last + 1, -1)
    prov = np.cumsum(delta)[:nmin]
    return dem, act, prov, (last - first + 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    d = dm.load_gemma(with_instances=True)
    m0 = 300 * 1440
    nmin = int(np.ceil(d["s_all"].max() / 60)) - m0 + 1
    dem, act, prov, life = per_minute(d["s"], d["s_all"], d["inst_all"], m0, nmin)
    covered = act > 0
    per_inst = dem[covered] / act[covered]
    c_obs = float(np.percentile(per_inst, 99))
    need = np.ceil(dem / c_obs)
    sel = covered & (need > 0)
    ratio = prov[sel] / need[sel]
    ratio_act = act[sel] / need[sel]
    # demand vs provisioned: peak/median
    res = {"minutes": int(nmin), "minutes_with_traffic": int(covered.sum()),
           "c_obs_req_per_inst_min": c_obs,
           "demand_rpm": {"median": float(np.median(dem[covered])), "p99": float(np.percentile(dem[covered], 99)),
                          "max": int(dem.max())},
           "active_instances": {"median": float(np.median(act[covered])), "p99": float(np.percentile(act[covered], 99)),
                                "max": int(act.max())},
           "provisioned_instances": {"median": float(np.median(prov[covered])),
                                     "p99": float(np.percentile(prov[covered], 99)), "max": int(prov.max())},
           "instance_lifetime_min": {"median": float(np.median(life)), "p10": float(np.percentile(life, 10)),
                                     "p90": float(np.percentile(life, 90)), "n_instances": int(life.size)},
           "overprov_ratio_provisioned": {"median": float(np.median(ratio)), "mean_time_weighted":
                                          float(prov[sel].sum() / need[sel].sum()),
                                          "p10": float(np.percentile(ratio, 10)), "p90": float(np.percentile(ratio, 90))},
           "overprov_ratio_active": {"median": float(np.median(ratio_act)),
                                     "mean_time_weighted": float(act[sel].sum() / need[sel].sum())},
           "peak_to_median": {"demand": float(np.percentile(dem[covered], 99) / max(np.median(dem[covered]), 1)),
                              "provisioned": float(np.percentile(prov[covered], 99) / max(np.median(prov[covered]), 1))}}
    # demand-instance cross-correlation of 1-min changes (lag in minutes where instances follow demand)
    x = dem.astype(float) - np.convolve(dem, np.ones(361) / 361, "same")
    y = prov.astype(float) - np.convolve(prov, np.ones(361) / 361, "same")
    lags = list(range(0, 121, 1))
    cc = [float(np.corrcoef(x[: nmin - L], y[L:])[0, 1]) for L in lags]
    res["xcorr_demand_to_provisioned"] = {"best_lag_min": int(lags[int(np.argmax(cc))]), "max_corr": float(max(cc)),
                                          "corr_at_0": cc[0]}
    # bursts
    rows = []
    with open(dm.TRACES / "gemma3_bursts.csv") as f:
        for r in csv.DictReader(f):
            st = int(r["start_min"]) - m0
            if st < 10 or st >= nmin - 60:
                continue
            dur = int(r["dur_min"])
            b_act, b_prov = np.median(act[st - 10:st]), np.median(prov[st - 10:st])
            win = slice(st, min(nmin, st + max(dur, 1) + 60))
            la = np.where(act[win] > b_act)[0]
            lp = np.where(prov[win] > b_prov)[0]
            rows.append({"start_min": int(r["start_min"]), "x": float(r["x"]), "dur_min": dur,
                         "base_rpm": float(r["base_rpm"]), "peak_rpm": float(r["peak_rpm"]),
                         "active_before": float(b_act), "active_max": int(act[win].max()),
                         "prov_before": float(b_prov), "prov_max": int(prov[win].max()),
                         "lag_active_min": int(la[0]) if la.size else None,
                         "lag_prov_min": int(lp[0]) if lp.size else None,
                         "ttft_p90_before": r["ttft_p90_before"], "ttft_p90_burst": r["ttft_p90_burst"]})
    la = np.array([r["lag_active_min"] for r in rows if r["lag_active_min"] is not None], float)
    lp = np.array([r["lag_prov_min"] for r in rows if r["lag_prov_min"] is not None], float)
    res["bursts"] = {"n": len(rows), "no_active_increase": sum(r["lag_active_min"] is None for r in rows),
                     "no_prov_increase": sum(r["lag_prov_min"] is None for r in rows),
                     "lag_active_min": {"median": float(np.median(la)), "p75": float(np.percentile(la, 75))}
                     if la.size else None,
                     "lag_prov_min": {"median": float(np.median(lp)), "p75": float(np.percentile(lp, 75))}
                     if lp.size else None,
                     "peak_rpm_over_c_obs_x_active_before": float(np.median(
                         [r["peak_rpm"] / (c_obs * max(r["active_before"], 1)) for r in rows])),
                     "rows": rows}
    (out / "comparator.json").write_text(json.dumps(res, indent=1))
    plot(dem, act, prov, need, rows, out / "figures", m0)
    print(json.dumps({k: v for k, v in res.items() if k != "bursts"}, indent=1))
    print(json.dumps({k: v for k, v in res["bursts"].items() if k != "rows"}, indent=1))


def plot(dem, act, prov, need, rows, figdir, m0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5))
    ax = axes[0]
    h = np.arange(dem.size) / 1440 + 300
    k = 10
    sm = lambda v: np.convolve(v, np.ones(k) / k, "same")  # noqa: E731
    ax.plot(h, sm(need), color="k", lw=0.6, label="demand / c_obs (instances needed at Chutes' own per-instance load)")
    ax.plot(h, sm(prov), color="tab:orange", lw=0.6, label="provisioned (lifetime proxy)")
    ax.plot(h, sm(act), color="tab:blue", lw=0.6, label="active (>= 1 request in the minute)")
    ax.set_xlabel("trace day")
    ax.set_ylabel("instances (10-min mean)")
    ax.set_yscale("symlog")
    ax.legend(fontsize=7)
    ax.set_title("Chutes gemma-3-27b-it: instances vs demand, days 300-366", fontsize=10)
    ax = axes[1]
    st = 464487 - m0
    w = slice(st - 60, st + 120)
    t = np.arange(-60, 120)
    ax.plot(t, dem[w], color="k", lw=1, label="demand (rpm)")
    ax2 = ax.twinx()
    ax2.step(t, act[w], color="tab:blue", lw=1, label="active instances")
    ax2.step(t, prov[w], color="tab:orange", lw=1, label="provisioned instances")
    ax.set_xlabel("minutes from burst 464487 start (the campaign's trace-A window)")
    ax.set_ylabel("requests / min")
    ax2.set_ylabel("instances")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "comparator.png", dpi=130)
    fig.savefig(figdir / "comparator.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
