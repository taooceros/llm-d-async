#!/usr/bin/env python3
"""Build the long-context rate table (W6 on H / M_TP16, W0 on M_2xTP8) from pilot artifacts.

Run naming (scripts/run_pilot.sh): results/pilot/<run_id>_<W>_<arm>.json with run_id
w6-<sha7>-plain-r<N>; arm H for H (workload W6), arm M0 for M (workload MIX_W6_W0, composite
W6 + W0 running concurrently).  Per-class finish on M = max(client_finished_at_s) over the class's
requests minus min(client_started_at_s) of the run, joining each record's canonical_request_id
with the expanded composite manifest (class labeller is pluggable: composite part or the
manifest's request_class field).  Refuses incomplete or token-mismatched runs.
Output: <out> (throughput_table format: base table + W6 + W0 M_2xTP8) and <out>.runs.json.

Mid-band mode (--w7-s-runs / --w7-h-runs / --w0h131k-runs, comma-separated run ids, no --commit):
reads results/pilot/<run_id>_W7_S.json[.gz], <run_id>_W7_H.json[.gz] and <run_id>_W0_H.json[.gz]
(run ids <tag>-<sha7>-plain-r<N> from scripts/band_campaign.sh), checks completeness and exact
requests/tokens against data/manifests/W7.json (W0.json) under --repo-root, and writes a mid table
(W7 makespan_s S/H, W0 makespan_s H_131k) usable by bulk_cost_model.py --mid-table, plus <out>.runs.json.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import gzip
import json
import pathlib
import statistics
import sys

HOME = pathlib.Path.home()
DEFAULT_REPO = HOME / "hetero-wt/Bf16Kv"


# ---------------------------------------------------------------- manifests and class labels
def expand_manifest(name, root):
    """Request rows (request_id, + phase_workload for composites, request_class if the part has it)
    in the id scheme of hetero.manifest._expand_composite, without importing hetero."""
    d = json.loads((pathlib.Path(root) / "data/manifests" / f"{name}.json").read_text())
    if "composite" not in d:
        return d, [dict(r) for r in d["requests"]]
    rows = []
    for index, (part, repeats) in enumerate(d["composite"]):
        _, base = expand_manifest(part, root)
        for rep in range(repeats):
            for r in base:
                rows.append({**{k: v for k, v in r.items() if k in ("request_class", "max_tokens")},
                             "request_id": f"{name}-p{index}r{rep}-{r['request_id']}",
                             "phase_workload": part})
    if "num_requests" in d and d["num_requests"] != len(rows):
        raise ValueError(f"{name}: expanded {len(rows)} rows != descriptor num_requests {d['num_requests']}")
    return d, rows


def label_composite_part(rows):
    return {r["request_id"]: r["phase_workload"] for r in rows}


def label_request_class(rows):
    return {r["request_id"]: r["request_class"] for r in rows if "request_class" in r}


LABELLERS = {"composite_part": label_composite_part, "request_class": label_request_class}


def canonical_id(rec):
    return rec.get("canonical_request_id") or rec["request_id"].split(":")[-1]


def class_finish_times(records, labels):
    """{class: {n, finish_s (max finish - run start), p50_s, first_start_s}}, plus unlabelled count."""
    t0 = min(r["client_started_at_s"] for r in records)
    by, unl = {}, 0
    for r in records:
        cls = labels.get(canonical_id(r))
        if cls is None:
            unl += 1
            continue
        by.setdefault(cls, []).append(r["client_finished_at_s"] - t0)
    out = {}
    for cls, v in by.items():
        v.sort()
        out[cls] = {"n": len(v), "finish_s": round(v[-1], 3), "p50_s": round(v[len(v) // 2], 3)}
    return out, unl


# ---------------------------------------------------------------- per-run artifacts
def find_artifact(art_root, run_id, workload, arm, suffix):
    pat = f"{art_root}/{run_id}/{workload}_{arm}_r[0-9]_{suffix}"
    p = sorted(glob.glob(pat))
    return p[0] if p else None


def read_result(path):
    p = pathlib.Path(path)
    text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
    return json.loads(text)["measurements"]["result"]


def find_result(pilot_dir, run_id, workload, arm):
    """Pilot result file of one run: .json or .json.gz (run_compress gzips files >= 1 MB)."""
    for suffix in (".json", ".json.gz"):
        p = pathlib.Path(pilot_dir) / f"{run_id}_{workload}_{arm}{suffix}"
        if p.exists():
            return str(p)
    return None


def mid_table(a, root, errors_out):
    """{table, runs} for the mid-band flags; appends refusal reasons to errors_out."""
    table, runs = {}, {}
    wanted = (("W7", "S", a.w7_s_runs, "S"), ("W7", "H", a.w7_h_runs, "H"), ("W0", "H", a.w0h131k_runs, "H_131k"))
    for workload, arm, ids, label in wanted:
        if not ids:
            continue
        tok, req, src = expected_totals(workload, root, None, None)
        spans, used = [], []
        for rid in [x for x in ids.split(",") if x]:
            path = find_result(a.pilot_dir, rid, workload, arm)
            if not path:
                errors_out.append(f"{rid}: no result file {a.pilot_dir}/{rid}_{workload}_{arm}.json[.gz]")
                continue
            r = read_result(path)
            e = check(path, r, tok, req)
            if e:
                errors_out.append(e)
                continue
            spans.append(r["makespan_s"])
            used.append(rid)
            runs.setdefault(f"{workload}_{label}", []).append({
                "run_id": rid, "path": path, "makespan_s": r["makespan_s"], "num_requests": r.get("num_requests"),
                "useful_output_tokens": r.get("useful_output_tokens"),
                "total_preemptions": r.get("total_preemptions"), "expected_totals": src})
        if spans and len(spans) == len([x for x in ids.split(",") if x]):
            table.setdefault(workload, {"output_tokens": tok, "requests": req, "makespan_s": {}})
            table[workload]["makespan_s"][label] = cell(
                spans, f"measured {', '.join(used)} {workload} makespan on {arm}" + (" at max_model_len 131072" if label == "H_131k" else ""))
        elif not spans and not any(label in e for e in errors_out):
            errors_out.append(f"{workload} {label}: no usable runs")
    return table, runs


def run_extras(art_root, run_id, workload, arm, tp16="large", deploy=None):
    """KV pool, max scheduled/running, preemptions, iteration p50 per instance (+ TP16 focus)."""
    ex = {}
    sp = find_artifact(art_root, run_id, workload, arm, "summary.json")
    if not sp:
        return {"error": "summary artifact not found"}
    s = json.loads(pathlib.Path(sp).read_text())
    per = (s.get("iteration_stats") or {}).get("per_instance") or {}
    idle = s.get("idle_before") or {}
    pools = deploy_pools(deploy) if deploy else {}
    block = next((b for _, (_, b) in pools.items()), 256)
    insts = {}
    for inst, st in per.items():
        d = {"preemptions": st.get("preemptions"), "iteration_p50_s": st.get("iteration_latency_p50_s"),
             "max_scheduled_requests": st.get("scheduled_requests_max"),
             "p50_scheduled_requests": st.get("scheduled_requests_p50")}
        if inst in pools:
            nb, bs = pools[inst]
            d["kv_pool_tokens"] = nb * bs
            d["kv_pool_source"] = f"deploy num_gpu_blocks {nb} x block_size {bs}"
        elif inst in idle and idle[inst].get("free_blocks") is not None:
            d["kv_pool_tokens"] = idle[inst]["free_blocks"] * block
            d["kv_pool_source"] = (f"summary idle_before free_blocks {idle[inst]['free_blocks']} x block_size "
                                   f"{block} (free at idle, <= pool)")
        insts[inst] = d
    ip = find_artifact(art_root, run_id, workload, arm, "iterations.csv.gz")
    if ip:
        mx = {}
        with gzip.open(ip, "rt") as f:
            for row in csv.DictReader(f):
                v = int(row["running_requests"] or 0)
                i = row["instance_id"]
                mx[i] = max(mx.get(i, 0), v)
        for i, v in mx.items():
            insts.setdefault(i, {})["max_running_requests"] = v
    ex["instances"] = insts
    ex["tp16"] = {"instance": tp16, **insts.get(tp16, {})} if tp16 in insts else None
    ex["total_preemptions"] = s.get("total_preemptions")
    ex["summary"] = sp
    return ex


def deploy_pools(path):
    """{instance_id: (num_gpu_blocks, block_size)} from a deploy artifact."""
    p = pathlib.Path(path)
    d = json.loads(gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text())
    out = {}
    for inst in d.get("measurements", {}).get("deployment", {}).get("instances", []):
        cfg = (inst.get("initialization") or {}).get("resolved_engine_config") or {}
        iid = (inst.get("descriptor") or {}).get("instance_id") or inst.get("instance_id")
        if iid and cfg.get("num_gpu_blocks"):
            out[iid] = (cfg["num_gpu_blocks"], cfg.get("block_size", 256))
    return out


def expected_totals(name, root, cli_tokens, cli_reqs):
    p = pathlib.Path(root) / "data/manifests" / f"{name}.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d["useful_output_tokens"], d["num_requests"], f"manifest {p}"
    if cli_tokens is None:
        raise SystemExit(f"no manifest {p} and no --expected-{name}-tokens given")
    return cli_tokens, cli_reqs, "CLI"


def check(path, r, toks, reqs):
    if r.get("all_requests_complete") is not True:
        return f"{path}: all_requests_complete={r.get('all_requests_complete')!r}, not True"
    if r.get("useful_output_tokens") != toks:
        return f"{path}: useful_output_tokens={r.get('useful_output_tokens')} != expected {toks}"
    if reqs is not None and r.get("num_requests") != reqs:
        return f"{path}: num_requests={r.get('num_requests')} != expected {reqs}"
    return None


def fmt_reps(vals):
    return f"({vals[0]:.1f}, {vals[1]:.1f})" if len(vals) >= 2 else f"({vals[0]:.1f}, {vals[0]:.1f})"


def cell(vals, src):
    m = round(sum(vals) / len(vals), 1)
    return {"value": m, "source": f"{src} {fmt_reps(vals)}"}


def inspect(a):
    """Join records of one existing run to its manifest and print per-class finish times."""
    root = pathlib.Path(a.manifest_root)
    _, rows = expand_manifest(a.manifest, root)
    labels = LABELLERS[a.classifier](rows)
    rp = find_artifact(a.artifact_dir, a.inspect, a.workload or a.manifest, a.arm, "records.json.gz")
    if not rp:
        raise SystemExit(f"records not found under {a.artifact_dir}/{a.inspect}")
    recs = json.load(gzip.open(rp))
    fin, unl = class_finish_times(recs, labels)
    print(f"records: {rp}\n  rows in manifest {len(rows)}, records {len(recs)}, unlabelled {unl}")
    for cls, v in sorted(fin.items()):
        print(f"  class {cls}: n={v['n']} finish={v['finish_s']} s p50={v['p50_s']} s")
    ex = run_extras(a.artifact_dir, a.inspect, a.workload or a.manifest, a.arm, a.tp16_instance,
                    a.deploy_json)
    print("  extras:", json.dumps(ex.get("tp16"), default=str), "total_preemptions", ex.get("total_preemptions"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-dir", default=str(DEFAULT_REPO / "results/pilot"))
    ap.add_argument("--commit", help="SHA7 in run ids w6-<sha>-plain-r<N>")
    ap.add_argument("--base-table", default="results/bulk_cost_model/throughput_table_fullbf16.json")
    ap.add_argument("--out")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO), help="holds data/manifests")
    ap.add_argument("--artifact-dir", default=None, help="default <repo-root>/results/hetero/artifacts")
    ap.add_argument("--classifier", choices=sorted(LABELLERS), default="composite_part")
    ap.add_argument("--w6-class", default="W6", help="class label of W6 rows in the mix")
    ap.add_argument("--w0-class", default="W0", help="class label of W0 rows in the mix")
    ap.add_argument("--mix", default="MIX_W6_W0")
    ap.add_argument("--tp16-instance", default="large", help="instance id of the TP16 engine in M runs")
    ap.add_argument("--deploy-json", default=None, help="deploy artifact with num_gpu_blocks/block_size")
    ap.add_argument("--expected-W6-tokens", type=int, default=None)
    ap.add_argument("--expected-W6-requests", type=int, default=None)
    ap.add_argument("--expected-mix-tokens", type=int, default=None)
    ap.add_argument("--expected-mix-requests", type=int, default=None)
    ap.add_argument("--w7-s-runs", default=None, help="comma-separated W7 run ids on S -> mid table t7S")
    ap.add_argument("--w7-h-runs", default=None, help="comma-separated W7 run ids on H -> mid table t7H")
    ap.add_argument("--w0h131k-runs", default=None,
                    help="comma-separated W0 run ids on H at max_model_len 131072 -> mid table a0H_131k")
    ap.add_argument("--inspect", metavar="RUN_ID", help="join one existing run's records and print per-class finish")
    ap.add_argument("--manifest", help="with --inspect: manifest name to join against")
    ap.add_argument("--manifest-root", default=None)
    ap.add_argument("--workload", default=None, help="with --inspect: artifact workload prefix (default manifest)")
    ap.add_argument("--arm", default="M0")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.repo_root)
    a.artifact_dir = a.artifact_dir or str(root / "results/hetero/artifacts")
    a.manifest_root = a.manifest_root or a.repo_root
    if a.inspect:
        return inspect(a)
    if a.w7_s_runs or a.w7_h_runs or a.w0h131k_runs:
        if not a.out:
            ap.error("--out is required")
        errors = []
        table, runs = mid_table(a, root, errors)
        if errors:
            print("REFUSING to write table:", file=sys.stderr)
            for e in errors:
                print("  " + e, file=sys.stderr)
            return 1
        out = pathlib.Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=1))
        pathlib.Path(str(out) + ".runs.json").write_text(json.dumps(runs, indent=1))
        print(f"wrote {out} and {out}.runs.json")
        for w, row in table.items():
            for lay, c in row["makespan_s"].items():
                print(f"  {w} {lay} {c['value']}  <- {c['source']}")
        return 0
    if not (a.commit and a.out):
        ap.error("--commit and --out are required")
    pd = pathlib.Path(a.pilot_dir)
    w6_tok, w6_req, w6_src = expected_totals("W6", root, a.expected_W6_tokens, a.expected_W6_requests)
    mx_tok, mx_req, mx_src = expected_totals(a.mix, root, a.expected_mix_tokens, a.expected_mix_requests)
    errors, runs = [], {"W6_H": [], "M": []}
    base = json.loads(pathlib.Path(a.base_table).read_text())
    table = copy.deepcopy(base)

    h_paths = sorted(glob.glob(str(pd / f"w6-{a.commit}-plain-r*_W6_H.json")))
    m_paths = sorted(glob.glob(str(pd / f"w6-{a.commit}-plain-r*_{a.mix}_M0.json")))
    if not h_paths:
        errors.append(f"no W6 H runs matching {pd}/w6-{a.commit}-plain-r*_W6_H.json")
    if not m_paths:
        errors.append(f"no M runs matching {pd}/w6-{a.commit}-plain-r*_{a.mix}_M0.json")
    h_ms, h_ids = [], []
    for p in h_paths:
        rid = pathlib.Path(p).name[: -len("_W6_H.json")]
        r = read_result(p)
        e = check(p, r, w6_tok, w6_req)
        if e:
            errors.append(e)
            continue
        h_ms.append(r["makespan_s"])
        h_ids.append(rid)
        runs["W6_H"].append({"run_id": rid, "path": p, "makespan_s": r["makespan_s"],
                             "num_requests": r.get("num_requests"),
                             "useful_output_tokens": r.get("useful_output_tokens"),
                             "total_preemptions": r.get("total_preemptions"),
                             "iteration_latency_p50_s": r.get("iteration_latency_p50_s"),
                             "extras": run_extras(a.artifact_dir, rid, "W6", "H", a.tp16_instance, a.deploy_json)})
    _, rows = expand_manifest(a.mix, root) if (root / "data/manifests" / f"{a.mix}.json").exists() else (None, None)
    labels = LABELLERS[a.classifier](rows) if rows else None
    w6_f, w0_f, m_ids = [], [], []
    for p in m_paths:
        rid = pathlib.Path(p).name[: -len(f"_{a.mix}_M0.json")]
        r = read_result(p)
        e = check(p, r, mx_tok, mx_req)
        if e:
            errors.append(e)
            continue
        if labels is None:
            errors.append(f"composite manifest {a.mix} not found under {root}/data/manifests")
            break
        rp = find_artifact(a.artifact_dir, rid, a.mix, "M0", "records.json.gz")
        if not rp:
            errors.append(f"{rid}: records artifact not found under {a.artifact_dir}")
            continue
        fin, unl = class_finish_times(json.load(gzip.open(rp)), labels)
        if unl or a.w6_class not in fin or a.w0_class not in fin:
            errors.append(f"{rid}: join incomplete (unlabelled={unl}, classes={sorted(fin)})")
            continue
        w6_f.append(fin[a.w6_class]["finish_s"])
        w0_f.append(fin[a.w0_class]["finish_s"])
        m_ids.append(rid)
        runs["M"].append({"run_id": rid, "path": p, "makespan_s": r["makespan_s"],
                          "class_finish": fin, "records": rp,
                          "total_preemptions": r.get("total_preemptions"),
                          "extras": run_extras(a.artifact_dir, rid, a.mix, "M0", a.tp16_instance, a.deploy_json)})
    if h_paths and not h_ms:
        errors.append("W6 H: no usable runs")
    if m_paths and not w6_f:
        errors.append("M: no usable runs")
    if errors:
        print("REFUSING to write table:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    table["W6"] = {"output_tokens": w6_tok, "requests": w6_req,
                   "makespan_s": {"H": cell(h_ms, f"measured {', '.join(h_ids)} W6 makespan on H plain"),
                                  "M_TP16": cell(w6_f, f"measured {', '.join(m_ids)} {a.mix} W6-class finish on M")}}
    table["W0"]["makespan_s"]["M_2xTP8"] = cell(
        w0_f, f"measured {', '.join(m_ids)} {a.mix} W0-class finish on M")
    runs["expected_totals_source"] = {"W6": w6_src, a.mix: mx_src}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    pathlib.Path(str(out) + ".runs.json").write_text(json.dumps(runs, indent=1))
    print(f"wrote {out} and {out}.runs.json")
    print(f"  W6 H {table['W6']['makespan_s']['H']['value']}  M_TP16 {table['W6']['makespan_s']['M_TP16']['value']}"
          f"  W0 M_2xTP8 {table['W0']['makespan_s']['M_2xTP8']['value']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
