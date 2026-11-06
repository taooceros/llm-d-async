#!/usr/bin/env python3
"""Local end-to-end test for asb_trace.py / asb_loadgen.py / asb_stub_engine.py.

  .venv/bin/python scripts/test_asb_loadgen.py [--keep] [--stress-rps 60 --stress-s 30]

Phase 1 (autoscale): stub A joined at start, stub B registered `pending` (healthy after 8 s),
router cap 4, controller live threshold 2 / sustain 3 s / max_triggers 1, ~40 s Poisson
trace that overloads A. Then exercises /controller/close (410 semantics, no new triggers).
Phase 2 (stress): open-loop client at --stress-rps for --stress-s against a wide stub,
reports client lateness.
Outputs go to /tmp/asb-test-<pid>/ (deleted unless --keep or a failure occurs).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TRACE_A = "/tmp/asb-trace-a.json"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return e.code, raw


def wait_http(url, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except urllib.error.HTTPError:
            return  # server is up (e.g. 503 health)
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError(f"timeout waiting for {url}")


def jl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


class Procs:
    def __init__(self, d):
        self.d = d
        self.ps = []

    def start(self, name, args):
        log = open(os.path.join(self.d, name + ".log"), "w")
        p = subprocess.Popen([PY] + args, stdout=log, stderr=subprocess.STDOUT)
        self.ps.append(p)
        return p

    def kill(self):
        for p in self.ps:
            if p.poll() is None:
                p.terminate()
        for p in self.ps:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


def check(cond, msg, fails):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


def ensure_lengths(d):
    if os.path.exists(TRACE_A):
        return TRACE_A
    # fallback: tiny synthetic length source
    p = os.path.join(d, "lengths.json")
    reqs = [{"i": i, "t": float(i), "in": 200 + (i * 97) % 3000, "out": 20 + (i * 53) % 400} for i in range(200)]
    with open(p, "w") as f:
        json.dump({"meta": {"kind": "synthetic"}, "requests": reqs}, f)
    return p


def phase1(d, procs, fails):
    lengths = ensure_lengths(d)
    trace = os.path.join(d, "poisson.json")
    subprocess.check_call([PY, os.path.join(HERE, "asb_trace.py"), "poisson", "--lengths-from", lengths,
                           "--rate-unit-rps", "1.0", "--schedule", "8:0.5,22:4,10:0.5",
                           "--seed", "7", "--label", "test", "--out", trace], stdout=subprocess.DEVNULL)
    pa, pb, pr = free_port(), free_port(), free_port()
    stub = os.path.join(HERE, "asb_stub_engine.py")
    common = ["--slots", "2", "--prefill-tps", "40000", "--tpot-s", "0.004"]
    procs.start("stubA", [stub, "--port", str(pa), "--name", "A"] + common)
    t_b = time.time()
    procs.start("stubB", [stub, "--port", str(pb), "--name", "B", "--start-healthy-after", "8"] + common)
    rdir = os.path.join(d, "router1")
    ctl = {"mode": "live", "threshold": 2, "sustain_s": 3, "period_s": 1, "max_triggers": 1, "lanes": ["B"]}
    procs.start("router1", [os.path.join(HERE, "asb_loadgen.py"), "router", "--port", str(pr),
                            "--out-dir", rdir, "--cap", "4", "--controller", json.dumps(ctl), "--run-id", "t1"])
    R = f"http://127.0.0.1:{pr}"
    for u in (f"http://127.0.0.1:{pa}/health", f"http://127.0.0.1:{pb}/health", R + "/health"):
        wait_http(u)
    check(http("GET", f"http://127.0.0.1:{pb}/health")[0] == 503, "stub B unhealthy at start", fails)
    http("POST", R + "/admin/instances", {"name": "A", "url": f"http://127.0.0.1:{pa}", "mode": "join"})
    http("POST", R + "/admin/instances", {"name": "B", "url": f"http://127.0.0.1:{pb}", "mode": "pending",
                                          "meta": {"lane": "B"}})
    # long-poll for trigger 1 in background (actuator-style)
    lp = {}

    def poll():
        lp["res"] = http("GET", R + "/controller/trigger/1?wait=60", timeout=70)
        lp["t"] = time.time()
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    http("POST", R + "/admin/event", {"kind": "test_marker", "note": "client start"})
    out = os.path.join(d, "client1.jsonl")
    t_client = time.time()
    subprocess.check_call([PY, os.path.join(HERE, "asb_loadgen.py"), "client", "--router", R, "--trace", trace,
                           "--out", out, "--run-id", "t1", "--seed", "3", "--model", "stub"],
                          stdout=open(os.path.join(d, "client1.log"), "w"), timeout=300)
    th.join(5)
    # ---- close semantics
    st = http("GET", R + "/admin/state")[1]
    code1 = http("GET", R + "/controller/trigger/1?wait=0")[0]
    lp2 = {}

    def poll2():
        t0 = time.time()
        lp2["res"] = http("GET", R + "/controller/trigger/2?wait=30", timeout=40)
        lp2["dt"] = time.time() - t0
    th2 = threading.Thread(target=poll2, daemon=True)
    th2.start()
    time.sleep(1.0)
    c_close = http("POST", R + "/controller/close", {})[0]
    th2.join(10)
    code2 = http("GET", R + "/controller/trigger/2?wait=5")[0]
    code1b = http("GET", R + "/controller/trigger/1")[0]
    # try to provoke more triggers after close: always-true condition
    http("POST", R + "/controller/config", {"mode": "live", "threshold": -1, "sustain_s": 1, "period_s": 0.5,
                                            "max_triggers": 5, "lanes": []})
    time.sleep(2.5)
    st2 = http("GET", R + "/admin/state")[1]
    http("POST", R + "/admin/shutdown", {})
    time.sleep(1.5)

    recs = jl(out)
    summ = json.load(open(out + ".summary.json"))
    tr = json.load(open(trace))
    rr = jl(os.path.join(rdir, "router_requests.jsonl"))
    samples = jl(os.path.join(rdir, "samples.jsonl"))
    events = jl(os.path.join(rdir, "events.jsonl"))
    decs = jl(os.path.join(rdir, "decisions.jsonl"))
    n = len(tr["requests"])
    print(json.dumps({k: summ[k] for k in ("n", "ok", "errors_by_status", "mismatch", "max_lateness_s",
                                           "duration_s", "by_instance")}))
    print("ttft", summ["ttft_s"], "tpot", summ["tpot_s"], "queue", summ["queue_s"])
    check(len(recs) == n and summ["ok"] == n, f"all {n} requests 200 (ok={summ['ok']})", fails)
    check(summ["mismatch"] == 0, "no token-count mismatch", fails)
    check(all(r["queue_s"] is not None and r["ttft_s"] is not None for r in recs), "queue_s/ttft_s recorded", fails)
    check(all(r["tpot_s"] is not None for r in recs if r["out"] > 1), "tpot_s recorded", fails)
    check(len(rr) == n and all(r["status"] == 200 for r in rr), "router_requests.jsonl complete", fails)
    check(st["counters"]["arrived"] == n and st["counters"]["completed"] == n, "router counters", fails)
    trig = [x for x in decs if x.get("kind") == "trigger"]
    check(len(trig) == 1 and trig[0]["actuate"] and trig[0]["lane"] == "B", "exactly one live trigger (lane B)", fails)
    if trig:
        t0 = trig[0]
        win = t0["window"]
        span = win[-1][0] - win[0][0]
        check(all(q > 2 for _, q in win) and len(win) >= 3, f"trigger window all > 2 ({len(win)} samples)", fails)
        # queue_depth above threshold continuously for >= sustain before the trigger (from samples.jsonl)
        above_since = None
        for s in samples:
            if s["t"] > t0["t"]:
                break
            above_since = (above_since or s["t"]) if s["queue_depth"] > 2 else None
        cont = t0["t"] - above_since if above_since else 0
        print(f"trigger at +{t0['t'] - t_client:.1f}s after client start, window span {span:.2f}s, "
              f"sampler-observed continuous q>2 for {cont:.2f}s")
        check(t0["t"] - t0["high_since"] >= 3.0 - 1e-6 and cont >= 3.0 - 1e-6,
              "queue stayed above threshold >= sustain", fails)
        check(lp.get("res", (None,))[0] == 200 and lp["res"][1]["k"] == 1 and lp["t"] - t0["t"] < 1.0,
              "long-poll /controller/trigger/1 released promptly with 200", fails)
    ev = {e["kind"]: e for e in events if e["kind"] in ("joined", "healthy", "first_dispatch", "registered")}
    joinedB = [e for e in events if e["kind"] == "joined" and e.get("name") == "B"]
    healthyB = [e for e in events if e["kind"] == "healthy" and e.get("name") == "B"]
    fdB = [e for e in events if e["kind"] == "first_dispatch" and e.get("name") == "B"]
    check(bool(joinedB) and joinedB[0]["t_router"] >= t_b + 8.0, "B joined only after it became healthy (>= 8 s)",
          fails)
    check(bool(healthyB) and healthyB[0]["polls"] > 1, "B health polled while 503", fails)
    b_disp = [r["dispatched_at"] for r in rr if r["instance"] == "B"]
    check(len(b_disp) > 0 and min(b_disp) >= joinedB[0]["t_router"] and bool(fdB),
          f"B received requests after joining ({len(b_disp)})", fails)
    check(any(e["kind"] == "test_marker" for e in events), "external /admin/event logged", fails)
    dts = [b["t"] - a["t"] for a, b in zip(samples, samples[1:])]
    check(len(samples) >= 30 and all(0.8 < x < 1.3 for x in dts), f"samples ~1/s ({len(samples)} rows, "
          f"dt {min(dts):.3f}..{max(dts):.3f})", fails)
    check(any((s["instances"].get("A") or {}).get("metrics", {}) and
              s["instances"]["A"]["metrics"].get("num_running") is not None for s in samples),
          "engine metrics scraped", fails)
    # close semantics
    check(code1 == 200 and code1b == 200, "fired trigger 1 returns 200 before and after close", fails)
    check(c_close == 200 and lp2.get("res", (None,))[0] == 410 and lp2["dt"] < 3.0,
          f"waiting long-poll for unfired k=2 released with 410 on close ({lp2.get('dt', -1):.2f}s)", fails)
    check(code2 == 410, "trigger/2 after close -> 410 immediately", fails)
    check(len(st2["triggers"]) == 1 and st2["controller_closed_at"] is not None, "no triggers after close", fails)
    check(any(x.get("kind") == "closed" for x in decs), "decisions.jsonl has closed record", fails)
    return summ


def phase2(d, procs, fails, rps, dur):
    lengths = ensure_lengths(d)
    trace = os.path.join(d, "stress.json")
    subprocess.check_call([PY, os.path.join(HERE, "asb_trace.py"), "poisson", "--lengths-from", lengths,
                           "--schedule", f"{dur}:{rps}", "--seed", "11", "--out", trace], stdout=subprocess.DEVNULL)
    ps, pr = free_port(), free_port()
    procs.start("stubS", [os.path.join(HERE, "asb_stub_engine.py"), "--port", str(ps), "--name", "S",
                          "--slots", "100000", "--prefill-tps", "1000000", "--tpot-s", "0.001"])
    rdir = os.path.join(d, "router2")
    procs.start("router2", [os.path.join(HERE, "asb_loadgen.py"), "router", "--port", str(pr), "--out-dir", rdir,
                            "--cap", "100000", "--run-id", "t2"])
    R = f"http://127.0.0.1:{pr}"
    wait_http(f"http://127.0.0.1:{ps}/health")
    wait_http(R + "/health")
    http("POST", R + "/admin/instances", {"name": "S", "url": f"http://127.0.0.1:{ps}", "mode": "join"})
    out = os.path.join(d, "client2.jsonl")
    subprocess.check_call([PY, os.path.join(HERE, "asb_loadgen.py"), "client", "--router", R, "--trace", trace,
                           "--out", out, "--run-id", "t2", "--model", "stub"],
                          stdout=open(os.path.join(d, "client2.log"), "w"), timeout=dur + 300)
    http("POST", R + "/admin/shutdown?now=1", {})
    s = json.load(open(out + ".summary.json"))
    print(json.dumps({"stress_rps": rps, "n": s["n"], "ok": s["ok"], "max_lateness_s": s["max_lateness_s"],
                      "p99_lateness_s": s["p99_lateness_s"], "errors": s["errors_by_status"]}))
    check(s["ok"] == s["n"], "stress: all ok", fails)
    check(s["max_lateness_s"] < 0.5, f"stress: max lateness {s['max_lateness_s']:.3f}s < 0.5s", fails)
    return s


def phase3(d, procs, fails, n_burst=400):
    """Instantaneous burst (every t == 0) in open-loop mode + `client --sequential`."""
    lengths = ensure_lengths(d)
    src = json.load(open(lengths))["requests"]
    burst = os.path.join(d, "burst.json")
    with open(burst, "w") as f:
        json.dump({"meta": {"kind": "burst"}, "requests": [
            {"i": i, "t": 0.0, "in": src[i % len(src)]["in"], "out": min(64, src[i % len(src)]["out"])}
            for i in range(n_burst)]}, f)
    ps, pr = free_port(), free_port()
    procs.start("stubC", [os.path.join(HERE, "asb_stub_engine.py"), "--port", str(ps), "--name", "C",
                          "--slots", "16", "--prefill-tps", "200000", "--tpot-s", "0.001"])
    rdir = os.path.join(d, "router3")
    procs.start("router3", [os.path.join(HERE, "asb_loadgen.py"), "router", "--port", str(pr), "--out-dir", rdir,
                            "--cap", "32", "--run-id", "t3"])
    R = f"http://127.0.0.1:{pr}"
    wait_http(f"http://127.0.0.1:{ps}/health")
    wait_http(R + "/health")
    http("POST", R + "/admin/instances", {"name": "C", "url": f"http://127.0.0.1:{ps}", "mode": "join"})
    out = os.path.join(d, "client3.jsonl")
    subprocess.check_call([PY, os.path.join(HERE, "asb_loadgen.py"), "client", "--router", R, "--trace", burst,
                           "--out", out, "--run-id", "t3b", "--model", "stub"],
                          stdout=open(os.path.join(d, "client3.log"), "w"), timeout=300)
    s = json.load(open(out + ".summary.json"))
    recs = jl(out)
    sends = [r["send_at"] for r in recs]
    print(json.dumps({"burst_n": s["n"], "ok": s["ok"], "max_lateness_s": s["max_lateness_s"],
                      "send_spread_s": max(sends) - min(sends), "queue_p99_s": s["queue_s"]["p99"]}))
    check(s["ok"] == n_burst and s["mismatch"] == 0, f"t==0 burst of {n_burst}: all ok", fails)
    check(s["max_lateness_s"] < 1.0, f"t==0 burst: max lateness {s['max_lateness_s']:.3f}s", fails)
    out = os.path.join(d, "client4.jsonl")
    subprocess.check_call([PY, os.path.join(HERE, "asb_loadgen.py"), "client", "--router", R, "--trace", burst,
                           "--out", out, "--run-id", "t3s", "--model", "stub", "--sequential", "--limit", "8"],
                          stdout=open(os.path.join(d, "client4.log"), "w"), timeout=300)
    recs = sorted(jl(out), key=lambda r: r["send_at"])
    s = json.load(open(out + ".summary.json"))
    check(s["ok"] == 8 and s["sequential"] is True, "sequential: 8 ok", fails)
    check(all(b["send_at"] >= a["done_at"] for a, b in zip(recs, recs[1:])), "sequential: no overlap", fails)
    check(all(r["lateness_s"] is None for r in recs) and [r["i"] for r in recs] == list(range(8))
          and all(b["sched_t"] > a["sched_t"] for a, b in zip(recs, recs[1:])),
          "sequential: lateness null, sched_t = actual offset, in order", fails)
    check(all(r["queue_s"] < 0.05 for r in recs), "sequential: no router queueing", fails)
    http("POST", R + "/admin/shutdown?now=1", {})


def phase4(d, procs, fails):
    """Controller timing: with an always-true condition (threshold -1) the first trigger must wait a
    full sustain_s after start / config change, and consecutive triggers are >= sustain_s apart."""
    pr = free_port()
    rdir = os.path.join(d, "router4")
    ctl = {"mode": "live", "threshold": -1, "sustain_s": 3, "period_s": 1, "max_triggers": 2, "lanes": ["a", "b"]}
    procs.start("router4", [os.path.join(HERE, "asb_loadgen.py"), "router", "--port", str(pr), "--out-dir", rdir,
                            "--cap", "1", "--controller", json.dumps(ctl)])
    R = f"http://127.0.0.1:{pr}"
    wait_http(R + "/health")
    time.sleep(1.5)
    t_cfg = time.time()
    http("POST", R + "/controller/config", ctl)  # restart the window mid-stream
    c1, r1 = http("GET", R + "/controller/trigger/1?wait=15", timeout=20)
    c2, r2 = http("GET", R + "/controller/trigger/2?wait=15", timeout=20)
    http("POST", R + "/admin/shutdown?now=1", {})
    ok = c1 == 200 and c2 == 200
    d1 = r1["t"] - t_cfg if ok else -1
    d12 = r2["t"] - r1["t"] if ok else -1
    print(f"controller timing: trigger1 {d1:.2f}s after config change, trigger2 {d12:.2f}s after trigger1")
    eps = 1e-6
    check(ok and d1 >= 3.0 - eps and d12 >= 3.0 - eps and r1["t"] - r1["high_since"] >= 3.0 - eps
          and r2["t"] - r2["high_since"] >= 3.0 - eps,
          "controller: sustain enforced after config change and between triggers", fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--stress-rps", type=float, default=40.0)
    ap.add_argument("--stress-s", type=float, default=30.0)
    ap.add_argument("--skip-stress", action="store_true")
    a = ap.parse_args()
    d = f"/tmp/asb-test-{os.getpid()}"
    os.makedirs(d)
    procs = Procs(d)
    fails = []
    try:
        s1 = phase1(d, procs, fails)
        print(f"phase1 max client lateness {s1['max_lateness_s']:.4f}s")
        phase3(d, procs, fails)
        phase4(d, procs, fails)
        if not a.skip_stress:
            phase2(d, procs, fails, a.stress_rps, a.stress_s)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        fails.append(f"exception: {e}")
    finally:
        procs.kill()
    print(("ALL PASS" if not fails else f"FAILED ({len(fails)}): {fails}") + f"  dir={d}")
    if not fails and not a.keep:
        shutil.rmtree(d, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
