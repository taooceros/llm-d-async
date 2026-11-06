#!/usr/bin/env python3
"""Per-instance capacity C of a workload's length mix with the campaign's rule, on the calibrated engine model:
the highest Poisson rate (step 0.025 req/s) whose SLO attainment is >= 90% on one TP8 instance at router cap 8
(240 s windows after a 30 s warm-up, averaged over 8 seeds of lengths sampled from the workload).
For gemma-3-27b's trace-A mix the campaign measured C = 0.7 req/s.

    /tmp/cm-venv/bin/python scripts/autoscale_cost/capacity.py --fit F --workload chutes:/tmp/cm/mistral_small_24b.npz
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
import sweep as sw  # noqa: E402


def attainment_at(eng, ins, outs, rate, seed, dur=270.0, warm=30.0):
    rng = np.random.Generator(np.random.PCG64(seed))
    n = int(rate * dur * 1.3) + 10
    t = np.cumsum(rng.exponential(1.0 / rate, n))
    t = t[t < dur]
    k = rng.integers(0, ins.size, t.size)
    r = sc.run(t, ins[k], outs[k], eng, sc.make_pol(mode=3, n0=1, max_inst=1))
    sel = t >= warm
    met = cm.meets(r["ttft"].astype(float), r["tpot"].astype(float), r["status"] == 1)
    return 100.0 * met[sel].mean()


def capacity(eng, ins, outs, seeds=8, lo=0.1, hi=2.0, step=0.025):
    rates = np.arange(lo, hi + 1e-9, step)
    curve = []
    best = lo
    for rate in rates:
        a = float(np.mean([attainment_at(eng, ins, outs, rate, s) for s in range(seeds)]))
        curve.append((round(float(rate), 3), round(a, 2)))
        if a >= 90.0:
            best = float(rate)
        elif a < 80.0:
            break
    return best, curve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    fit = json.loads(Path(a.fit).read_text())
    eng = cm.engine_from_fit(fit)
    w = sw.load_workload(a.workload, 1.0)
    C, curve = capacity(eng, w["in"], w["out"])
    rec = {"workload": a.workload, "C_rps": C, "curve": curve, "in_mean": float(w["in"].mean()),
           "out_mean": float(w["out"].mean())}
    print(json.dumps(rec))
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
