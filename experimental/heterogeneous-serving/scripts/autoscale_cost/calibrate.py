#!/usr/bin/env python3
"""Fit the engine step-time model to the one-instance calibration runs of the real campaign.

Inputs (read-only): ~/hetero-wt/AutoscaleBurst/results/autoscale-burst/calibration/<run>/loadgen/client.jsonl[.gz]
  unloaded            80 trace-A-stratified requests one at a time (+ warm-up)       -> step time at batch 1
  rate-0.4 .. 1.0     Poisson 240 s, router cap 16                                   -> batching + queueing
  cap8-rate-0.7/0.8   Poisson 240 s, cap 8
  cap-8 / 16 / 32     240-request instantaneous burst                                -> saturated throughput
Each run is replayed in the simulator with its recorded router arrival times and token counts; the fit minimises
log errors of per-request TTFT/TPOT (unloaded) and of TTFT/TPOT quantiles, SLO attainment and saturated
throughput (loaded). Output: <out>/calibration_fit.json.

    /tmp/cm-venv/bin/python scripts/autoscale_cost/calibrate.py --out results/autoscale-cost/<rev>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
import simcore as sc  # noqa: E402

RUNS = {  # name -> router cap
    "unloaded": 8, "rate-0.4": 16, "rate-0.5": 16, "rate-0.6": 16, "rate-0.7": 16, "rate-0.8": 16, "rate-0.9": 16,
    "rate-1.0": 16, "cap8-rate-0.7": 8, "cap8-rate-0.8": 8, "cap-8": 8, "cap-16": 16, "cap-32": 32,
}
P0 = np.array([0.005, 0.0003, 0.072, 0.015, 0.0002, 0.003, 0.3, 0.007, 0.003])


def load_runs():
    out = {}
    for name, cap in RUNS.items():
        recs = cm.read_jsonl(cm.CAL / name / "loadgen/client.jsonl")
        if recs:
            d = cm.client_arrays(recs)
            d["cap"] = cap
            out[name] = d
    return out


def sat_rps(done_rel: np.ndarray) -> float:
    """Completion rate between the 10% and 90% completion quantiles (asb_metrics.throughput_window)."""
    d = np.sort(done_rel[~np.isnan(done_rel)])
    if d.size < 10:
        return float("nan")
    i0, i1 = int(d.size * 0.1), int(d.size * 0.9) - 1
    return (i1 - i0 + 1) / (d[i1] - d[i0])


def sim_run(d, p, cap=None, pb=None):
    eng = sc.make_eng(p, cap=cap or d["cap"], pb=pb)
    pol = sc.make_pol(mode=3, n0=1, max_inst=1)
    return sc.run(d["t"], d["in"], d["out"], eng, pol)


def metrics(ttft, tpot, ok, t, out):
    m = cm.meets(ttft, tpot, ok)
    e2e = ttft + np.where(np.isnan(tpot), 0, tpot) * np.maximum(out - 1, 0)
    return {"ttft_p50": cm.pct(ttft[ok], 50), "ttft_p90": cm.pct(ttft[ok], 90), "ttft_p99": cm.pct(ttft[ok], 99),
            "tpot_p50": cm.pct(tpot[ok], 50), "tpot_p90": cm.pct(tpot[ok], 90), "tpot_p99": cm.pct(tpot[ok], 99),
            "attain": 100.0 * m.mean(), "sat_rps": sat_rps(np.where(ok, t + e2e, np.nan))}


def real_metrics(d):
    done = d["done_at"] - d["t0"]
    m = metrics(d["ttft"], d["tpot"], d["ok"], d["t"], d["out"])
    m["sat_rps"] = sat_rps(np.where(d["ok"], done, np.nan))
    return m


def loss(p, runs, verbose=False, pb=None):
    if np.any(p < 0) or p[6] > 1.0:
        return 1e9
    L = 0.0
    rows = {}
    for name, d in runs.items():
        r = sim_run(d, p, pb=pb)
        ok = (r["status"] == 1) & d["ok"]
        if name == "unloaded":
            sel = ok & (d["out"] >= 2)
            e1 = np.log(r["ttft"][sel] / d["ttft"][sel])
            e2 = np.log(r["tpot"][sel] / d["tpot"][sel])
            L += 4 * (np.mean(e1 ** 2) + np.mean(e2 ** 2))
            rows[name] = {"ttft_logrmse": float(np.sqrt(np.mean(e1 ** 2))),
                          "tpot_logrmse": float(np.sqrt(np.mean(e2 ** 2)))}
            continue
        ms = metrics(r["ttft"].astype(float), r["tpot"].astype(float), r["status"] == 1, d["t"], d["out"])
        mr = real_metrics(d)
        rows[name] = {"sim": ms, "real": mr}
        burst = name.startswith("cap-")
        keys = ["sat_rps", "ttft_p50", "tpot_p50", "tpot_p90"] if burst else \
            ["ttft_p50", "ttft_p90", "ttft_p99", "tpot_p50", "tpot_p90", "tpot_p99"]
        for k in keys:
            if np.isfinite(ms[k]) and np.isfinite(mr[k]) and mr[k] > 0 and ms[k] > 0:
                w = 10.0 if k == "sat_rps" else 3.0
                L += w * np.log(ms[k] / mr[k]) ** 2 / len(keys)
        if not burst:
            L += ((ms["attain"] - mr["attain"]) / 10.0) ** 2
    if verbose:
        return L, rows
    return L


class _LossFn:
    def __init__(self, runs, pb):
        self.runs = runs
        self.pb = pb

    def __call__(self, x):
        return loss(np.abs(np.asarray(x)), self.runs, pb=self.pb)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--method", default="de", choices=["de", "nm"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pb", type=int, default=None, help="prefill budget per step while decoding")
    ap.add_argument("--init", default=None, help="start Nelder-Mead from this calibration_fit.json")
    a = ap.parse_args(argv)
    from scipy.optimize import minimize
    runs = load_runs()
    print("runs:", {k: len(v["t"]) for k, v in runs.items()})
    best = None
    starts = [P0, P0 * np.array([1, 2, 1, 0.5, 1, 1, 0.2, 1, 2]), P0 * np.array([0.6, 0.5, 1.0, 1.2, 0.5, 1, 2.5, 1.2, 0.3]),
              P0 * np.array([1.2, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0])]
    if a.init:
        starts = [np.array(json.loads(Path(a.init).read_text())["params"])]
    if a.method == "de":
        from scipy.optimize import differential_evolution
        bounds = [(0.0, 0.012), (0.0, 0.002), (0.05, 0.09), (0.0, 0.04), (0.0, 0.001), (0.0, 0.006), (0.0, 1.0),
                  (0.0, 0.012), (0.0, 0.02)]
        res = differential_evolution(_LossFn(runs, a.pb), bounds, seed=1, maxiter=a.maxiter // 4, popsize=20, tol=1e-6,
                                     workers=a.workers, updating="deferred", polish=False, init="sobol",
                                     x0=np.clip(P0, [b[0] for b in bounds], [b[1] for b in bounds]))
        print("DE ->", res.x, res.fun)
        starts = [res.x]
    for x0 in starts:
        res = minimize(_LossFn(runs, a.pb), x0, method="Nelder-Mead",
                       options={"maxiter": a.maxiter, "xatol": 1e-6, "fatol": 1e-5})
        res = minimize(_LossFn(runs, a.pb), np.abs(res.x), method="Nelder-Mead",
                       options={"maxiter": a.maxiter, "xatol": 1e-6, "fatol": 1e-5})  # restart
        print("start", x0, "->", np.abs(res.x), res.fun)
        if best is None or res.fun < best.fun:
            best = res
    p = np.abs(best.x)
    L, rows = loss(p, runs, verbose=True, pb=a.pb)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"params": p.tolist(), "prefill_budget_while_decoding": a.pb,
           "param_names": ["p0 step base (s)", "p1 per decoding request (s)", "p2 per 1k prefill tokens (s)",
                           "p3 per 1e5 tokens of the longest decoding context (s)", "p4 prefill attention per 1e6 token^2 (s)",
                           "p5 TTFT constant overhead (s)", "p6 overlap: weight of min(P, D)",
                           "p7 decode memory floor: weight read per step (s)",
                           "p3 = per 1e5 tokens of the longest decoding context; p8 per 1e5 total decoding context (s)"],
           "loss": float(L), "runs": rows, "max_batched_tokens": 2048, "kv_cap_tokens": 205 * 256}
    (out / "calibration_fit.json").write_text(json.dumps(rec, indent=1))
    for k, v in rows.items():
        print(k, json.dumps(v, default=lambda x: round(x, 4)))
    print("params", p)


if __name__ == "__main__":
    main()
