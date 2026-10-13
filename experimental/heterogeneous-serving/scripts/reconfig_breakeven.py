#!/usr/bin/env python3
"""Break-even phase size for switching serving layout between bulk-inference phases.

Switching from layout A to B for a phase of N output tokens pays when
    N / r_A - N / r_B > C   <=>   N > C * r_A * r_B / (r_B - r_A)
where r is the layout's useful-token throughput on that phase and C the reconfiguration cost
(seconds of lost slice time).  Inputs are a JSON table (see --table); every entry carries its
source so simulated values can never be mistaken for measurements.

Usage: reconfig_breakeven.py --table TABLE.json [--cost 360 555] [--json OUT.json]
"""
import argparse
import itertools
import json


def breakeven(r_from, r_to, cost_s):
    if r_to <= r_from:
        return None  # switching never pays
    return cost_s * r_from * r_to / (r_to - r_from)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--cost", type=float, nargs="+", default=[360.0, 555.0])
    ap.add_argument("--json")
    args = ap.parse_args()
    table = json.load(open(args.table))
    rows = []
    for workload, entry in table.items():
        tokens = entry["output_tokens"]
        layouts = {k: v for k, v in entry["makespan_s"].items() if v is not None}
        for a, b in itertools.permutations(layouts, 2):
            r_a, r_b = tokens / layouts[a]["value"], tokens / layouts[b]["value"]
            for cost in args.cost:
                n = breakeven(r_a, r_b, cost)
                rows.append({
                    "workload": workload, "from": a, "to": b, "cost_s": cost,
                    "r_from": round(r_a, 1), "r_to": round(r_b, 1),
                    "gain_per_phase_s": round(layouts[a]["value"] - layouts[b]["value"], 1),
                    "breakeven_tokens": None if n is None else round(n),
                    "breakeven_phase_multiple": None if n is None else round(n / tokens, 2),
                    "breakeven_requests": None if n is None else round(n / tokens * entry["requests"]),
                    "sources": [layouts[a]["source"], layouts[b]["source"]],
                })
    for r in rows:
        if r["breakeven_tokens"] is not None:
            print(f"{r['workload']:3} {r['from']}->{r['to']} C={r['cost_s']:4.0f}s "
                  f"r {r['r_from']:6.1f}->{r['r_to']:6.1f} tok/s  gain/phase {r['gain_per_phase_s']:6.1f}s  "
                  f"N*={r['breakeven_tokens'] / 1e6:5.2f}M tok = {r['breakeven_phase_multiple']:5.2f}x phase "
                  f"({r['breakeven_requests']} req)  [{', '.join(r['sources'])}]")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
