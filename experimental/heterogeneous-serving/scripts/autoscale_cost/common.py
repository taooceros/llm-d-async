"""Shared paths, run loaders and metrics for the autoscale cost model."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASB = Path(os.path.expanduser("~/hetero-wt/AutoscaleBurst/results/autoscale-burst"))  # read-only input
CAL = ASB / "calibration"
CAMPAIGN = ASB / "4daec1af"
OUT_ROOT = ROOT / "results/autoscale-cost"
SLO_TTFT = 0.7882183790206909
SLO_TPOT = 0.06634625338868454
C_RPS = 0.7
CAP = 8
CHIPS = 8


def read_jsonl(path: Path) -> list[dict]:
    for p in (path, Path(str(path) + ".gz")):
        if p.exists():
            op = gzip.open if str(p).endswith(".gz") else open
            with op(p, "rt") as f:
                return [json.loads(x) for x in f if x.strip()]
    return []


def read_json(path: Path):
    for p in (path, Path(str(path) + ".gz")):
        if p.exists():
            op = gzip.open if str(p).endswith(".gz") else open
            with op(p, "rt") as f:
                return json.load(f)
    return None


def client_arrays(recs: list[dict]) -> dict:
    """Real client records -> arrays sorted by router arrival (s from the first arrival)."""
    recs = sorted(recs, key=lambda r: r.get("arrived_at") or r["send_at"])
    arr = np.array([r.get("arrived_at") or r["send_at"] for r in recs], float)
    t0 = arr.min()
    ok = np.array([r.get("status") == 200 and r.get("ttft_s") is not None and not r.get("mismatch") for r in recs])
    f = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in recs], float)  # noqa: E731
    return {"t": arr - t0, "t0": t0, "in": np.array([r["in"] for r in recs], np.int64),
            "out": np.array([r["out"] for r in recs], np.int64), "ok": ok, "ttft": f("ttft_s"), "tpot": f("tpot_s"),
            "queue": f("queue_s"), "status": [r.get("status") for r in recs], "sched_t": f("sched_t"),
            "send_at": f("send_at"), "done_at": f("done_at")}


def meets(ttft, tpot, ok):
    tp = np.where(np.isnan(tpot), 0.0, tpot)
    return ok & (ttft <= SLO_TTFT) & (tp <= SLO_TPOT)


def pct(a, q):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


STEP_TERMS = [0, 1, 2, 3, 4, 7, 8]  # step-time parameters (p5 = TTFT overhead, p6 = overlap weight are not times)


def engine_from_fit(fit: dict, speed: float = 1.0, cap: int = CAP):
    """Engine vector from calibration_fit.json; speed < 1 scales every step-time term (capacity sensitivity)."""
    import simcore as sc
    p = np.array(fit["params"], float)
    p[STEP_TERMS] *= speed
    return sc.make_eng(p, cap=cap, pb=fit.get("prefill_budget_while_decoding"))
