#!/usr/bin/env python3
"""Policy x boot-time sweep over the 67-day Chutes gemma-3-27b trace (or another workload) with the validated
simulator. Each configuration simulates the whole trace once (state carries across days); results are per-day
aggregates: requests, requests meeting both SLOs, 429s, billed / booting / decomposed instance-seconds.

    /tmp/cm-venv/bin/python scripts/autoscale_cost/sweep.py --fit F --workload gemma --frac 0.35 --out /tmp/cm/sweep-g035.npz

Workloads: gemma (thinned by --frac, timestamps unscaled), qwen (Bailian to-C trace A, non-bursty control),
chutes:<file.npz> (another model extracted by extract_model.py).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
import data as dm  # noqa: E402
import simcore as sc  # noqa: E402

DAY = 86400.0
# boot scenarios: name -> (lo, hi) seconds, trigger -> first served (uniform per scale-up)
SCENARIOS = {
    "stock-standby": (363.0, 391.0),      # matrix fresh host 362.6 s; campaign triggers 387.0/390.6/391.4 s
    "pm-standby": (54.0, 72.0),           # matrix 53.8 s; campaign 70.0/72.0/71.1 s
    "stock-new": (580.6, 580.6),          # GKE-trigger e2e (GKE ~170-176 s + boot)
    "pm-new": (264.7, 264.7),
    "stock-new-stockout": (738.0, 866.0),  # smoke: GKE + Ray join 331-459 s today + 407 s stock boot
    "pm-new-stockout": (429.0, 557.0),     # 331-459 s + 98 s PM boot (smoke new-pm 557 s at GKE 459 s)
}
B_SWEEP = [0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 390, 480, 600, 750, 900]
H_C = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 2.5, 3, 4, 6]   # headroom in units of C
T_S = [0, 30, 60, 120, 180, 300, 450, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200]  # keep-alive (s)
U = [round(0.3 + 0.05 * i, 2) for i in range(15)]                               # HPA target utilisation
HPA_VARIANTS = {"hpa-fast": (0, 0), "hpa-wva120": (120, 120)}

G = {}


def load_workload(name: str, frac: float):
    if name == "gemma":
        d = dm.load_gemma()
        d = dm.thin(d, frac)
        s0 = 300 * DAY
    elif name == "qwen":
        d = dm.load_qwen()
        d = dm.thin(d, frac)
        s0 = 0.0
    elif name.startswith("chutes:"):
        z = np.load(name.split(":", 1)[1])
        d = dm.thin({"s": z["s"], "in": z["inp"], "out": z["out"]}, frac)
        s0 = 300 * DAY if z["s"].min() >= 300 * DAY else float(np.floor(z["s"].min() / DAY) * DAY)
    else:
        raise SystemExit(f"unknown workload {name}")
    t = d["s"] - s0
    return {"t": np.ascontiguousarray(t), "in": d["in"].astype(np.int64), "out": d["out"].astype(np.int64)}


def configs(max_inst: int, quick: bool = False, bsweep: bool = True):
    out = []
    boots = [(n, lo, hi) for n, (lo, hi) in SCENARIOS.items()]
    boots += [(f"B{b}", b, b) for b in (B_SWEEP if bsweep else [0])]
    hs = H_C if not quick else [0, 1, 4]
    ts = T_S if not quick else [0, 600]
    for (bn, lo, hi) in boots:
        for h, T in itertools.product(hs, ts):
            out.append({"policy": "headroom", "boot": bn, "blo": lo, "bhi": hi, "h_c": h, "T": T})
        if bn in SCENARIOS or bn in ("B0",):
            for (vn, (su, sd)), u in itertools.product(HPA_VARIANTS.items(), U):
                out.append({"policy": vn, "boot": bn, "blo": lo, "bhi": hi, "u": u, "Su": su, "Sd": sd})
    for h, T in itertools.product(hs, ts):
        out.append({"policy": "oracle", "boot": "B0", "blo": 0.0, "bhi": 0.0, "h_c": h, "T": T})
    for n in range(1, max_inst + 1):
        out.append({"policy": "static", "boot": "-", "blo": 0.0, "bhi": 0.0, "n": n})
    return out


def make_pol(c, max_inst, seed):
    C = G.get("C", cm.C_RPS)
    if c["policy"] in ("headroom", "oracle"):
        return sc.make_pol(mode=1, h=c["h_c"] * C, T=c["T"], W=60, C=C, min_inst=1, max_inst=max_inst,
                           clair=1 if c["policy"] == "oracle" else 0, n0=G.get("n0", 1), blo=c["blo"], bhi=c["bhi"],
                           seed=seed)
    if c["policy"].startswith("hpa"):
        return sc.make_pol(mode=2, u=c["u"], Su=c["Su"], Sd=c["Sd"], W=60, C=C, min_inst=1, max_inst=max_inst,
                           scrape=10, sync=15, tol=0.1, n0=G.get("n0", 1), blo=c["blo"], bhi=c["bhi"], seed=seed,
                           uplim=4)
    return sc.make_pol(mode=3, n0=c["n"], max_inst=c["n"])


def run_cfg(job):
    j, c = job
    w = G["w"]
    max_inst = G["max_inst"] if c["policy"] != "static" else c["n"]
    pol = make_pol(c, max_inst, seed=1000 + j)
    t0 = time.time()
    r = sc.run(w["t"], w["in"], w["out"], G["eng"], pol, n_ticks=G["n_ticks"])
    return j, aggregate(r, w, G["ndays"]), time.time() - t0


def aggregate(r, w, ndays):
    day = np.minimum((w["t"] // DAY).astype(np.int64), ndays - 1)
    ok = r["status"] == 1
    met = cm.meets(r["ttft"].astype(float), r["tpot"].astype(float), ok)
    n = np.bincount(day, minlength=ndays)[:ndays]
    nm = np.bincount(day, weights=met, minlength=ndays)[:ndays]
    nrej = np.bincount(day, weights=(r["status"] == 2), minlength=ndays)[:ndays]
    nttft = np.bincount(day, weights=ok & (r["ttft"] <= cm.SLO_TTFT), minlength=ndays)[:ndays]

    def per_day(x):
        x = x.astype(np.float64)
        pad = ndays * int(DAY) - x.size
        if pad > 0:
            x = np.concatenate([x, np.zeros(pad)])
        return x[: ndays * int(DAY)].reshape(ndays, int(DAY)).sum(1)

    bill = per_day(r["bill"])
    boot = per_day(r["boot"])
    act = per_day(r["act"])
    tg, tg0 = r["tgt"].astype(np.int64), r["tgt0"].astype(np.int64)
    a = r["act"].astype(np.int64)
    base = per_day(np.minimum(a, tg0))
    head = per_day(np.clip(np.minimum(a, tg) - tg0, 0, None))
    keep = per_day(np.clip(a - np.maximum(tg, tg0), 0, None))
    qmax = np.zeros(ndays)
    q = r["q"]
    m = min(q.size, ndays * int(DAY))
    qmax[: (m + int(DAY) - 1) // int(DAY)] = [q[i:i + int(DAY)].max() for i in range(0, m, int(DAY))]
    return {"n": n, "met": nm, "rej": nrej, "ttft_ok": nttft, "bill_s": bill, "boot_s": boot, "act_s": act,
            "drain_s": bill - boot - act, "base_s": base, "head_s": head, "keep_s": keep, "qmax": qmax,
            "n_acq": r["n_acq"], "n_cancel": r["n_cancel"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--workload", default="gemma")
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--max-inst", type=int, default=64)
    ap.add_argument("--procs", type=int, default=110)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-bsweep", action="store_true", help="named scenarios + B0 only")
    ap.add_argument("--C", type=float, default=cm.C_RPS, help="per-instance capacity of the workload's mix (req/s)")
    ap.add_argument("--warm-start", action="store_true",
                    help="dynamic policies start with ceil(first-minute rate / C) active instances (short traces)")
    ap.add_argument("--speed", type=float, default=1.0, help="step-time scale (capacity sensitivity)")
    ap.add_argument("--only", default=None, help="comma list of policies to run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fit = json.loads(Path(a.fit).read_text())
    G["eng"] = cm.engine_from_fit(fit, a.speed)
    G["w"] = load_workload(a.workload, a.frac)
    G["C"] = a.C
    if a.warm_start:
        G["n0"] = max(1, int(np.ceil(np.sum(G["w"]["t"] < 60.0) / 60.0 / a.C)))
    G["max_inst"] = a.max_inst
    G["ndays"] = int(np.ceil(G["w"]["t"][-1] / DAY))
    G["n_ticks"] = int(G["ndays"] * DAY) + 2
    cfgs = configs(a.max_inst, a.quick, not a.no_bsweep)
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c["policy"] in keep]
    print(f"workload {a.workload} frac {a.frac}: {G['w']['t'].size} requests, {G['ndays']} days, "
          f"{len(cfgs)} configs", flush=True)
    # compile once in the parent so forked workers share the cache
    sc.run(G["w"]["t"][:2000], G["w"]["in"][:2000], G["w"]["out"][:2000], G["eng"],
           make_pol({"policy": "headroom", "blo": 60, "bhi": 60, "h_c": 1, "T": 60}, 8, 1))
    # longest first (static fleets with many instances, then the rest)
    order = sorted(range(len(cfgs)), key=lambda j: -(cfgs[j].get("n", 0)))
    res = [None] * len(cfgs)
    t0 = time.time()
    with get_context("fork").Pool(a.procs) as pool:
        for k, (j, agg, dt) in enumerate(pool.imap_unordered(run_cfg, [(j, cfgs[j]) for j in order])):
            res[j] = agg
            if k % 100 == 0:
                print(f"  {k}/{len(cfgs)} done, last {dt:.1f}s, elapsed {time.time() - t0:.0f}s", flush=True)
    keys = [k for k in res[0] if isinstance(res[0][k], np.ndarray)]
    out = {k: np.stack([r[k] for r in res]) for k in keys}
    out["n_acq"] = np.array([r["n_acq"] for r in res])
    out["n_cancel"] = np.array([r["n_cancel"] for r in res])
    meta = {"workload": a.workload, "frac": a.frac, "max_inst": a.max_inst, "n_requests": int(G["w"]["t"].size),
            "ndays": G["ndays"], "fit": a.fit, "params": fit["params"],
            "pb": fit.get("prefill_budget_while_decoding"), "speed": a.speed, "C": a.C, "n0": G.get("n0", 1), "configs": cfgs, "elapsed_s": time.time() - t0}
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
