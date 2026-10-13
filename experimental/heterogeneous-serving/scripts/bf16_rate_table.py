#!/usr/bin/env python3
"""Build a bf16 throughput table (throughput_table_measured.json format) from pilot result JSONs.

Writes <out> (table) and <out>.runs.json (per-file measurements). Refuses incomplete or
token-mismatched runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

EXPECTED = {"W0": (721595, 1763), "W5": (982899, 250), "W4": (387975, 904)}
FIELDS = ("num_requests", "useful_output_tokens", "makespan_s", "all_requests_complete",
          "total_preemptions", "iteration_latency_p50_s")


def read_run(path):
    r = json.loads(pathlib.Path(path).read_text())["measurements"]["result"]
    d = {"path": str(path)}
    for f in FIELDS:
        d[f] = r.get(f)
    return d


def check(w, d):
    toks, reqs = EXPECTED[w]
    if d["all_requests_complete"] is not True:
        return f"{d['path']}: all_requests_complete={d['all_requests_complete']!r}, not True"
    if d["useful_output_tokens"] != toks:
        return f"{d['path']}: useful_output_tokens={d['useful_output_tokens']} != expected {toks} ({w})"
    if d["num_requests"] is not None and d["num_requests"] != reqs:
        return f"{d['path']}: num_requests={d['num_requests']} != expected {reqs} ({w})"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-dir", default=str(pathlib.Path.home() / "hetero-wt/Bf16Kv/results/pilot"))
    ap.add_argument("--rc-commit", required=True, help="SHA7 in run ids rc-<sha>-<L>-s<N>")
    ap.add_argument("--w4-commit", required=True, help="SHA7 in run ids w4-<sha>-plain-r<N>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="bf16-KV", help="numeric profile named in each source string")
    a = ap.parse_args(argv)
    pd = pathlib.Path(a.pilot_dir)

    # (W, L) -> list of (run_id, path)
    plan = {}
    for lay in ("S", "H"):
        for w, slots in (("W0", (1, 4)), ("W5", (2, 3))):
            plan[(w, lay)] = [(f"rc-{a.rc_commit}-{lay}-s{n}",
                               pd / f"rc-{a.rc_commit}-{lay}-s{n}_{w}_{lay}.json") for n in slots]
    for lay in ("S", "H"):  # W4 on S needs bf16 KV (a TP8 pool must hold a 36,864-token request)
        w4 = sorted(glob.glob(str(pd / f"w4-{a.w4_commit}-plain-r*_W4_{lay}.json")))
        if w4:
            plan[("W4", lay)] = [(pathlib.Path(p).name[: -len(f"_W4_{lay}.json")], pathlib.Path(p))
                                 for p in w4]

    table, runs, errors = {}, {}, []
    for w, (toks, reqs) in EXPECTED.items():
        table[w] = {"output_tokens": toks, "requests": reqs, "makespan_s": {"S": None, "H": None}}
        runs[w] = {}
    for (w, lay), items in plan.items():
        if not items:
            continue
        ds, ids = [], []
        for rid, p in items:
            if not p.exists():
                errors.append(f"missing file {p}")
                continue
            d = read_run(p)
            d["run_id"] = rid
            err = check(w, d)
            if err:
                errors.append(err)
            ds.append(d)
            ids.append(rid)
        if not ds:
            continue
        runs[w][lay] = ds
        if any(d["makespan_s"] is None for d in ds):
            errors.append(f"{w} {lay}: makespan_s missing")
            continue
        ms = [d["makespan_s"] for d in ds]
        mean = round(sum(ms) / len(ms), 1)
        reps = f"({ms[0]:.1f}, {ms[1]:.1f})" if len(ms) >= 2 else f"({ms[0]:.1f}, {ms[0]:.1f})"
        extra = "".join(f", {m:.1f}" for m in ms[2:])
        kind = "plain" if w == "W4" else "blind"
        src = f"measured {a.label} {', '.join(ids)} {reps}{' + extra reps' + extra if extra else ''} {kind}"
        table[w]["makespan_s"][lay] = {"value": mean, "source": src}
    # W0/W5 on S and H are required
    for w in ("W0", "W5"):
        for lay in ("S", "H"):
            if table[w]["makespan_s"][lay] is None:
                errors.append(f"{w} {lay}: no usable runs")
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
        print(" ", w, {l: (c["value"] if c else None) for l, c in row["makespan_s"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
