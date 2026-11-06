#!/usr/bin/env python3
"""Validate the simulator against the real AutoscaleBurst runs (read-only input).

Every counted run under ~/hetero-wt/AutoscaleBurst/results/autoscale-burst/4daec1af/runs/<rid>/ that has a run.json is
replayed with its own recorded router arrivals and token counts, the same controller (queue > 2 in every 1 s sample
for 10 s since the previous trigger, +1 instance per trigger, <= 3) and the measured trigger -> joined boot time of
each of its triggers. Static arms run the controller dry (triggers computed, nothing actuated). Metrics use the
campaign's definitions (scripts/asb_report.py): burst/run SLO violation %, burst TTFT P99, max router queue, queue
recovery (trigger 1 -> queue <= 2 for 10 s), instance-seconds (always-on from client start, scaled-up from their
trigger, to the last completion), 429 count, trigger times.

    /tmp/cm-venv/bin/python scripts/autoscale_cost/validate.py --fit <calibration_fit.json> --out <dir>
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

THR, SUSTAIN = 2, 10


def burst_window(meta: dict):
    if meta.get("kind") == "trace-a":
        k = meta["factor_k"]
        lead = -meta["rel_window_min"][0]
        end = min(meta["burst_dur_min"], meta["rel_window_min"][1])
        return lead * 60.0 / k, (lead + end) * 60.0 / k
    t, best = 0.0, None
    for seg in meta.get("schedule_rps") or []:
        dur, rate = float(seg["dur_s"]), float(seg["rps"])
        if best is None or rate > best[2]:
            best = (t, t + dur, rate)
        t += dur
    return (best[0], best[1]) if best else (0.0, float("inf"))


def queue_recovery(ts, q, t1, thr=THR):
    low = None
    for t, x in zip(ts, q):
        if t < t1:
            continue
        if x <= thr:
            low = t if low is None else low
            if t - low >= 10.0:
                return low - t1
        else:
            low = None
    return None


def dry_triggers(ts, q, n=3, thr=THR, sustain=SUSTAIN):
    """The campaign rule on a 1 s queue series (as in simcore mode 0)."""
    out, high = [], None
    for t, x in zip(ts, q):
        if x > thr:
            high = t if high is None else high
        else:
            high = None
        if len(out) < n and high is not None and t - high > sustain + 0.5:
            out.append(t)
            high = t
    return out


def load_real(d: Path):
    rec = cm.read_json(d / "run.json")
    lg = d / "loadgen"
    client = cm.read_jsonl(lg / "client.jsonl")
    if rec is None or not client:
        return None
    samples = cm.read_jsonl(lg / "router/samples.jsonl")
    events = cm.read_jsonl(lg / "router/events.jsonl")
    decisions = cm.read_jsonl(lg / "router/decisions.jsonl")
    meta = (cm.read_json(lg / "trace.json") or {}).get("meta", {})
    t0 = rec.get("client_start_at")
    a = cm.client_arrays(client)
    a["t"] = a["t"] + a["t0"] - t0  # relative to the client start
    trig = [x for x in decisions if x.get("kind") == "trigger"]
    joins = []
    for x in trig:
        if x.get("mode") != "live":
            continue
        ln = x.get("lane")
        js = [e["t_router"] for e in events if e.get("kind") == "joined" and e.get("name") in (ln, f"{ln}-ep")]
        joins.append(min(js) - x["t"] if js else None)
    return {"rid": rec["rid"], "arm": rec["arm"], "trace": rec["trace"], "t0": t0, "a": a, "meta": meta,
            "q_t": np.array([s["t"] - t0 for s in samples]), "q": np.array([s["queue_depth"] for s in samples]),
            "trig": [x["t"] - t0 for x in trig], "live": [x.get("mode") == "live" for x in trig], "boot": joins,
            "n_static": 1 + (len(rec.get("lanes_order", [])) if rec["arm"] == "static-peak" else 0)}


def summarize(ttft, tpot, ok, rejected, sched_t, done, arr, bw, q_t, q, trig, live, n_static):
    met = cm.meets(ttft, tpot, ok)
    burst = (sched_t >= bw[0]) & (sched_t < bw[1])
    t_end = np.nanmax(done)
    inst_s = n_static * t_end + sum(max(0.0, t_end - t) for t, lv in zip(trig, live) if lv)
    return {"burst_viol_pct": 100.0 * (1 - met[burst].mean()), "run_viol_pct": 100.0 * (1 - met.mean()),
            "burst_ttft_p99": cm.pct(ttft[burst & ok], 99), "max_queue": int(np.max(q)) if len(q) else None,
            "queue_integral": float(np.sum(q)),
            "queue_recovery_s": queue_recovery(q_t, q, trig[0]) if trig else None,
            "instance_s": float(inst_s), "n_429": int(rejected.sum()), "triggers_s": [round(t, 1) for t in trig],
            "t_end_s": float(t_end)}


def run_one(R, eng):
    a = R["a"]
    static = R["arm"] in ("static-base", "static-peak")
    n_ticks = int(np.ceil(a["t"].max())) + 4000
    if static:
        pol = sc.make_pol(mode=3, n0=R["n_static"], max_inst=R["n_static"])
        boots = None
    else:
        pol = sc.make_pol(mode=0, n0=1, max_inst=4, thr=THR, sustain=SUSTAIN, max_trig=len(R["boot"]))
        boots = [b if b is not None else 1e9 for b in R["boot"]]
    r = sc.run(a["t"], a["in"], a["out"], eng, pol, boots, n_ticks=n_ticks)
    ok = r["status"] == 1
    ttft = r["ttft"].astype(float)
    tpot = r["tpot"].astype(float)
    done = np.where(ok, a["t"] + ttft + np.where(np.isnan(tpot), 0, tpot) * np.maximum(a["out"] - 1, 0), np.nan)
    q_t = np.arange(len(r["q"]), dtype=float)
    last = int(np.nanmax(done)) + 2
    q_t, q = q_t[1:last], r["q"][1:last]
    if static:
        trig = dry_triggers(q_t, q)
        live = [False] * len(trig)
    else:
        trig = list(r["trig_t"])
        live = [True] * len(trig)
    bw = burst_window(R["meta"])
    sim = summarize(ttft, tpot, ok, r["status"] == 2, a["sched_t"], done, a["t"], bw, q_t, q, trig, live,
                    R["n_static"])
    sim["join_s"] = [round(float(j - t), 1) for j, t in zip(r["join_t"], r["trig_t"])]
    # real
    ra = R["a"]
    real_done = ra["done_at"] - R["t0"]
    real = summarize(ra["ttft"], ra["tpot"], ra["ok"], np.array([s == 429 for s in ra["status"]]), ra["sched_t"],
                     np.where(ra["ok"], real_done, np.nan), ra["t"], bw, R["q_t"], R["q"], R["trig"], R["live"],
                     R["n_static"])
    real["join_s"] = [None if b is None else round(b, 1) for b in R["boot"]]
    series = {"sim_q_t": q_t, "sim_q": q, "real_q_t": R["q_t"], "real_q": R["q"],
              "sim_ttft": ttft, "real_ttft": ra["ttft"], "t": a["t"], "sim_act": r["act"][1:last]}
    return sim, real, series


def err_row(sim, real):
    out = {}
    for k in ("burst_viol_pct", "run_viol_pct", "max_queue", "queue_recovery_s", "instance_s", "burst_ttft_p99",
              "n_429"):
        s, r = sim.get(k), real.get(k)
        if s is None or r is None:
            out[k] = None
            continue
        out[k] = {"abs": s - r, "rel": (s - r) / r if r else None}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    fit = json.loads(Path(a.fit).read_text())
    eng = cm.engine_from_fit(fit, a.speed)
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, series = [], {}
    import subprocess
    tracked = set(subprocess.run(["git", "-C", str(cm.CAMPAIGN), "ls-files", "runs"], capture_output=True,
                                 text=True).stdout.split())
    for d in sorted((cm.CAMPAIGN / "runs").iterdir()):
        if f"runs/{d.name}/run.json" not in tracked:  # committed (counted) runs only
            continue
        R = load_real(d)
        if R is None:
            continue
        sim, real, ser = run_one(R, eng)
        rows.append({"rid": R["rid"], "arm": R["arm"], "trace": R["trace"], "n": int(len(R["a"]["t"])),
                     "boot_s": R["boot"], "sim": sim, "real": real, "err": err_row(sim, real)})
        series[R["rid"]] = ser
        print(R["rid"], json.dumps({"sim": sim, "real": real}, default=str))
    (out / f"validation{a.tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    plot(rows, series, out / "figures", a.tag)
    return rows


def plot(rows, series, figdir, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), squeeze=False)
    for i, row in enumerate(rows):
        s = series[row["rid"]]
        ax = axes[i][0]
        ax.plot(s["real_q_t"], s["real_q"], color="k", lw=1.2, label="real router queue")
        ax.plot(s["sim_q_t"], s["sim_q"], color="tab:red", lw=1.0, ls="--", label="simulated")
        for t in row["real"]["triggers_s"]:
            ax.axvline(t, color="0.6", lw=0.6)
        ax.set_title(f"{row['rid']}: router queue (req)", fontsize=9)
        ax.set_xlabel("replay time (s)")
        ax.legend(fontsize=7)
        ax = axes[i][1]
        ok = ~np.isnan(s["real_ttft"])
        ax.scatter(s["t"][ok], s["real_ttft"][ok], s=1, color="k", alpha=0.4, label="real TTFT")
        ok = ~np.isnan(s["sim_ttft"])
        ax.scatter(s["t"][ok], s["sim_ttft"][ok], s=1, color="tab:red", alpha=0.4, label="simulated TTFT")
        ax.set_yscale("log")
        ax.axhline(cm.SLO_TTFT, color="tab:blue", lw=0.8)
        ax.set_title(f"{row['rid']}: per-request TTFT (s) by send time", fontsize=9)
        ax.set_xlabel("replay time (s)")
        ax.legend(fontsize=7, markerscale=6)
    fig.tight_layout()
    fig.savefig(figdir / f"validation_timelines{tag}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
