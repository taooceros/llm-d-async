#!/usr/bin/env python3
"""Production-C re-cut of the sealed controlled H->S warm/cold transitions.

Reads (read-only) the sealed OverlapCompile campaign controlled-hs-baseline-20261002b
plus the rcm telemetry log, splits the published dispatch gap into contiguous
wall-clock segments, and classifies each segment as
  harness   - exists only because of the measurement harness (pilot/result I/O,
              correctness re-check, benchmark warm-up, 30 MB evidence dumps)
  controller- real today but produced by this controller shape (one Ray job per
              step, runtime_env upload, kubectl apply/rollout, job exit/polling)
  engine    - real engine/serving work on the critical path.
Validation: segment sum must equal the published dispatch gap within 1 s.

Usage: production_c_recut.py [--telemetry /tmp/omp_rcm/telemetry.jsonl] [--out DIR]
"""
import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
from pathlib import Path

CAMPAIGN = Path.home() / "hetero-wt/OverlapCompile/results/overlap/OverlapCompile/controlled-hs-baseline-20261002b"
PHASES = ["init_device", "load_model", "determine_available_memory", "initialize_from_config",
          "compile_or_warm_up_model"]
LOAD_SUB = ["snapshot_manifest_verify", "snapshot_read_tmpfs", "snapshot_verify", "snapshot_device_placement"]
REMOTE_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\s+INFO (job_manager|worker)\.py:\d+ -- (.*)")


def load(name):
    return json.load(open(CAMPAIGN / name))


def timeline():
    return {e["event"]: e["t"] for e in map(json.loads, open(CAMPAIGN / "timeline.jsonl"))}


def telemetry_window(path, lo, hi):
    """Lines in [lo, hi] (ingestion time). rg on the integer prefix keeps this fast."""
    prefixes = sorted({str(int(t))[:7] for t in range(int(lo), int(hi) + 1, 1000)} |
                      {str(int(lo))[:7], str(int(hi))[:7]})
    pattern = "|".join(f'^\\{{"timestamp": {p}' for p in prefixes)
    out = subprocess.run(["rg", pattern, str(path)], capture_output=True, text=True).stdout
    rows = [json.loads(line) for line in out.splitlines()]
    return [r for r in rows if lo <= r["timestamp"] <= hi]


def remote_stamp(message, tz_offset_s):
    """Ray job_manager/worker lines carry the remote wall clock to the millisecond."""
    m = REMOTE_TS.search(message)
    if not m:
        return None
    naive = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return naive.timestamp() + int(m.group(2)) / 1000 - tz_offset_s, m.group(4)


def ray_job_bounds(rows, stage, entry_substr, tz):
    """(submitted_ingest, entrypoint_remote, ray_connected_remote) for one Ray job."""
    submitted = entry = connected = None
    for r in rows:
        if r["stage"] != stage:
            continue
        msg = r["message"]
        if submitted is None and "submitted successfully" in msg and entry_substr in msg:
            submitted = r["timestamp"]
        if submitted is not None and entry is None and "Runtime env is setting up" in msg:
            s = remote_stamp(msg, tz)
            entry = s[0] if s else None
        if entry is not None and connected is None and "Connected to Ray cluster" in msg:
            s = remote_stamp(msg, tz)
            connected = s[0] if s else None
    return submitted, entry, connected


def deploy_anatomy(dep):
    m = dep["measurements"]
    insts = m["deployment"]["instances"]
    per_worker, inst_ready = [], {}
    for inst in insts:
        init = inst["initialization"]
        e0 = init["engine_init_started_at"]
        ends = []
        for w in init["worker_init_phases"]:
            ph = w["phases"]
            row = {"instance": inst["descriptor"].get("instance_id", init["instance_id"]),
                   "rank": w["executor_rank"], "engine_init_started_at": e0,
                   "init_device_start": ph["init_device"]["start"],
                   "end": ph["compile_or_warm_up_model"]["end"]}
            for p in PHASES + LOAD_SUB:
                if p in ph:
                    row[p] = ph[p]["end"] - ph[p]["start"]
            lm = ph["load_model"]
            sub = sum(row.get(p, 0) for p in LOAD_SUB)
            row["load_model_unattributed"] = (lm["end"] - lm["start"]) - sub
            per_worker.append(row)
            ends.append(row["end"])
        inst_ready[init["instance_id"]] = max(ends)
    gw = m["gateway"]["status"]["bindings"]
    registered = max(b["registered_at"] for b in gw.values())
    return {
        "engine_init_started_at_min": min(w["engine_init_started_at"] for w in per_worker),
        "engine_init_started_at_max": max(w["engine_init_started_at"] for w in per_worker),
        "init_device_start_min": min(w["init_device_start"] for w in per_worker),
        "init_device_start_max": max(w["init_device_start"] for w in per_worker),
        "last_engine_ready": max(inst_ready.values()),
        "gateway_registered_at": registered,
        "stage_clocks": m["deployment"]["stage_clocks"],
        "remote_elapsed_s": m["elapsed_s"],
        "saved_at": dep["saved_at"],
        "phase_median_s": {p: round(statistics.median(w[p] for w in per_worker if p in w), 2)
                           for p in PHASES + LOAD_SUB + ["load_model_unattributed"]},
        "phase_max_s": {p: round(max(w[p] for w in per_worker if p in w), 2)
                        for p in PHASES + LOAD_SUB + ["load_model_unattributed"]},
    }


def recut(treatment, tel_path, tz):
    obs = load(f"{treatment}-transition-observed.json")
    tl = timeline()
    T0 = obs["source"]["last_finished_request"]
    first = obs["target"]["first_dispatch"]
    gap = obs["dispatch_gap_s"]
    src = "source-cold" if treatment == "cold" else "source-warm"
    tgt = f"target-{treatment}"
    ctrl = "source-cold-control" if treatment == "cold" else "source-warm-control"
    rows = telemetry_window(tel_path, T0 - 5, first + 5)
    dep = load(f"{tgt}-deploy.json")
    ana = deploy_anatomy(dep)
    pilot = load(f"{tgt}-pilot.json")["measurements"]["pilot"]
    started_wall = pilot["started_wall"]

    # Teardown: three Ray jobs; each has saved_at (client) and elapsed_s (remote work).
    jobs = []
    prev = tl[f"pilot_end:{ctrl}"]
    for step in ("drain", "gateway-down", "release"):
        j = load(f"{src}-{step}.json")
        jobs.append({"step": step, "wall_s": j["saved_at"] - prev, "work_s": j["measurements"]["elapsed_s"]})
        prev = j["saved_at"]
    teardown_tail = tl[f"released:{src}"] - prev

    sub_dep, entry_dep, conn_dep = ray_job_bounds(rows, "hetero_command", "hetero-deploy", tz)
    sub_p, entry_p, conn_p = ray_job_bounds(rows, "overlap_pilot", "overlap-precompile-pilot", tz)
    # Remote command start: cli.main stamps `started` before ray.init; bound it by the
    # Ray job entrypoint start (remote clock). Remote end = start + elapsed_s.
    remote_begin = entry_dep
    remote_end = remote_begin + ana["remote_elapsed_s"]

    seg = []

    def add(name, a, b, cls, note=""):
        seg.append({"segment": name, "start_rel": round(a - T0, 2), "end_rel": round(b - T0, 2),
                    "s": round(b - a, 2), "class": cls, "note": note})

    add("source pilot tail (result emit, artifact tar)", T0, tl[f"pilot_end:{ctrl}"], "harness")
    t = tl[f"pilot_end:{ctrl}"]
    for j in jobs:
        add(f"teardown {j['step']}: remote work", t, t + j["work_s"], "engine",
            "drain/gateway-down/release real work")
        add(f"teardown {j['step']}: Ray job submit/env/exit", t + j["work_s"], t + j["wall_s"], "controller")
        t += j["wall_s"]
    add("teardown script tail", t, tl[f"released:{src}"], "controller")
    add("deploy: kubectl apply + rollout status + job submit + runtime_env", tl[f"deploy_start:{tgt}"],
        remote_begin, "controller", "gateway_resources stage then Ray job start")
    sc = ana["stage_clocks"]
    add("deploy: ray.init + backend/placement/actor create", remote_begin, ana["engine_init_started_at_min"],
        "controller", f"measured backend {sc['backend_s']}+placement {sc['placement_s']}+actor {sc['actor_s']} s; rest ray.init/inventory")
    add("engine: EngineCore + RayDistributedExecutor + worker proc start", ana["engine_init_started_at_min"],
        ana["init_device_start_min"], "engine", "process/actor start before init_device")
    add("engine: init_device..compile_or_warm_up (slowest worker)", ana["init_device_start_min"],
        ana["last_engine_ready"], "engine", "see phase_median_s / phase_max_s")
    add("engine->serving endpoint->gateway registered", ana["last_engine_ready"], ana["gateway_registered_at"],
        "engine", f"serving_endpoint_s={sc['serving_endpoint_s']}")
    add("gateway actor create/bind + status (until remote command end)", ana["gateway_registered_at"],
        remote_end, "controller", "gateway recreated per deploy")
    add("deploy job exit + 30 MB evidence stream + client poll", remote_end, ana["saved_at"], "harness",
        "MEASUREMENTS_EMITTED ~30 MB")
    add("deploy client: artifact fetch + mark", ana["saved_at"], tl[f"deploy_end:{tgt}"], "harness")
    add("controls check + snapshot_validate check job", tl[f"deploy_end:{tgt}"], tl[f"pilot_start:{tgt}"],
        "harness", "correctness re-check, not production")
    add("pilot: rcm/Ray job submit + runtime_env + ray.init", tl[f"pilot_start:{tgt}"], conn_p, "harness")
    add("pilot: catalogue/tokenizer load, 8-req warm-up, idle wait, capture arm", conn_p, started_wall,
        "harness", "contains benchmark warm-up; no sub-clocks")
    add("pilot: started_wall -> first dispatch", started_wall, first, "harness")

    total = sum(s["s"] for s in seg)
    by_class = {}
    for s in seg:
        by_class[s["class"]] = round(by_class.get(s["class"], 0) + s["s"], 2)
    ramp = obs["ramp_deficit_s"]
    result = {
        "treatment": treatment,
        "T0_source_last_finished": T0,
        "published_gap_s": round(gap, 2), "segment_sum_s": round(total, 2),
        "reconciles_within_1s": abs(total - gap) < 1.0,
        "published_C_s": round(obs["C_s"], 2), "ramp_deficit_s": round(ramp, 2),
        "by_class_s": by_class,
        "C_engine_only_s": round(by_class.get("engine", 0) + ramp, 2),
        "C_production_today_s": round(by_class.get("engine", 0) + by_class.get("controller", 0) + ramp, 2),
        "deploy_anatomy": ana,
        "job_bounds": {"deploy_submitted_ingest": sub_dep, "deploy_entry_remote": entry_dep,
                       "deploy_connected_remote": conn_dep, "pilot_submitted_ingest": sub_p,
                       "pilot_entry_remote": entry_p, "pilot_connected_remote": conn_p},
        "segments": seg,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default="/tmp/omp_rcm/telemetry.jsonl")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/production_c_recut"))
    ap.add_argument("--remote-utc-offset-h", type=float, default=0.0,
                    help="remote Ray log clock offset from UTC (Ray pods log UTC); checked against ingestion")
    args = ap.parse_args()
    tz = args.remote_utc_offset_h * 3600
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for treatment in ("warm", "cold"):
        r = recut(treatment, args.telemetry, tz)
        # Clock sanity: remote entry stamp must precede its ingestion and follow submission.
        jb = r["job_bounds"]
        r["clock_check_ok"] = all(0 <= jb[f"{k}_entry_remote"] - jb[f"{k}_submitted_ingest"] + 0.5 <= 5.5
                                  for k in ("deploy", "pilot"))
        allres[treatment] = r
        print(f"== {treatment}: gap {r['published_gap_s']} sum {r['segment_sum_s']} ok={r['reconciles_within_1s']} "
              f"clock_ok={r['clock_check_ok']} by_class={r['by_class_s']} C_engine={r['C_engine_only_s']} "
              f"C_prod_today={r['C_production_today_s']}")
        for s in r["segments"]:
            print(f"  {s['start_rel']:8.2f} {s['end_rel']:8.2f} {s['s']:7.2f} {s['class']:10} {s['segment']}")
        print("  phase_median_s", r["deploy_anatomy"]["phase_median_s"])
        print("  phase_max_s   ", r["deploy_anatomy"]["phase_max_s"])
    (out / "recut.json").write_text(json.dumps(allres, indent=2) + "\n")


if __name__ == "__main__":
    main()
