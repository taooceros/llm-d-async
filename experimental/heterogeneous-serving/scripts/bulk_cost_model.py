#!/usr/bin/env python3
"""Bulk-job makespan model for one-way layout phases on the 32-chip v5e slice.

Phase time for k manifests of class w on layout L is k * T[L][w] (the published break-even
convention; per-manifest makespans include their own ramp and drain tail).  A one-way switch
adds its effective cost C_eff and forfeits the phase overlap O that a static backlog gets.

  full switch     C_eff = C_regime
  partial step    C_eff = gap + D_regime * (1 - f) + join_tail        (kept engines serve at f)
  staged S->Mb->H C_eff = sum of two partial steps

Long-context job (--long-table / --long-placeholder): k_s W0 then k_l W6 manifests; static H =
k_s a0H + k_l t6H - O, S->H = k_s a0S + C + k_l t6H (a switch adds C only, as in scenario(); so
k_s* = (C + O)/(a0H - a0S)), static M = max(k_s a0M, k_l t6M), S->M partial (see s_to_m).

Mid-band job (--mid-table / --mid-placeholder): k_s W0, then k_m W7 (16k-48k-token GovReport),
then optionally k_l W6 manifests, all known at t=0.  Rates a0S, a0H, t7S, t7H (W7 makespan per
manifest on S / H), t6H.  Switching adds C, a static arm keeps the overlap O once (only when it has
at least two non-empty phases, as in long_arms):
  static S          = k_s a0S + k_m t7S - O                      (k_l = 0 only; W6 does not fit S)
  static H          = k_s a0H + k_m t7H + k_l t6H - O
  S->H before W7    = k_s a0S + C + k_m t7H + k_l t6H
  S->H before W6    = k_s a0S + k_m t7S + C + k_l t6H            (k_l > 0 only)
Break-evens: k_m* = (C + O)/(t7S - t7H) (S->H before W7 beats static S at k_l = 0; none if
t7S <= t7H); k_s* = (C + O)/(a0H - a0S) vs static H as for the long job.  With --c-full / --c-partial
the S->H arms (long and mid) use c_full and the S->M arm uses c_partial; --a0h-override replaces
a0H (W0 on H at max_model_len 131,072) in both jobs.

Every measured input carries its source; inferred inputs are tagged "INF".  The C regimes are
parameters (override with --regimes JSON) because they are still being re-verified.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

EVID = pathlib.Path.home() / ("heterogeneous-serving-tpu/llm-d/experimental/heterogeneous-serving/"
                              "docs/experiments/reconfiguration_evidence")

# name: (C full-switch s, D joiner start-up s, provenance)
REGIMES = {
    "full_redeploy": (420.0, 363.7, "C 407.5-445.0 measured (breakeven.typ:217-220), midpoint 420 as "
                      "published; D = joiner start-up 361.7-364.7 measured (breakeven.typ:391-395)"),
    "warm_snapshot_harness": (225.26, 136.2, "C measured (verified-c.md:24); D = deploy-job start -> "
                              "engine ready 160.8-24.62 (verified-c.md:13-19), INF for a joiner"),
    "warm_production": (160.5, 136.2, "C verified (verified-c.md:24); D as above, INF"),
    "projected_levers": (80.0, 70.7, "INF (verified-c.md:37-41): controller + pre-spawn + "
                         "FastSnapshot/FastGraph; D = 136.2-12.5 job start-18 pre-spawn-35 load"),
    "projected_no_trace": (45.0, 34.7, "INF (verified-c.md:42): also no per-start trace+lower (-36)"),
}
PARTIAL_GAP = 9.3      # s, drain+relabel+retarget, mean of mix-A partial arms (breakeven.typ:348)
# The joiner readiness wait (0.7-9.4 s, breakeven.typ:391-395) is inside the calibrated join tail.
O_MIXA = 115.075       # s, static-backlog overlap forfeited, mix A (mixA_model.json)
O_TAIL_S = 32.6        # s, output-equivalent drain tail of S, INF (verified-c.md:45)
O_TAIL_H = 21.1        # s, same for H, INF


def load_table(path=None):
    t = json.loads((pathlib.Path(path) if path else EVID / "throughput_table_measured.json").read_text())
    out = {}
    for w, row in t.items():
        out[w] = {"tokens": row["output_tokens"], "T": {}, "reps": {}, "sources": {}}
        for lay, cell in row["makespan_s"].items():
            if cell is None:
                continue
            out[w]["T"][lay] = cell["value"]
            out[w]["sources"][lay] = cell["source"]
            reps = re.search(r"\(([\d.]+), ([\d.]+)\)", cell["source"])
            out[w]["reps"][lay] = ([float(reps.group(1)), float(reps.group(2))] if reps
                                   else [cell["value"]])
    return out


def validate_breakeven(tab):
    pub = json.loads((EVID / "breakeven_ranges.json").read_text())
    rows = []
    for r in pub:
        w, a, b = r["workload"], r["frm"], r["to"]
        gain = tab[w]["T"][a] - tab[w]["T"][b]
        diffs = [x - y for x in tab[w]["reps"][a] for y in tab[w]["reps"][b]]
        k = 420.0 / gain
        k_lo, k_hi = 395.0 / max(diffs), 445.0 / min(diffs)
        tok = k * tab[w]["tokens"] / 1e6
        rows.append({"row": f"{w} {a}->{b}", "k": round(k, 2), "k_pub": r["k"],
                     "res_k": round(k - r["k"], 3), "range": [round(k_lo, 2), round(k_hi, 2)],
                     "range_pub": [r["k_lo"], r["k_hi"]], "tokens_M": round(tok, 2),
                     "tokens_pub": r["N_M"]})
    return rows


def validate_mixes(tab):
    e2e = {e["runs"][0]: e for e in json.loads((EVID / "e2e.json").read_text())}
    trans = json.loads((EVID / "mix_transitions.json").read_text())
    gap_ms = [t["first_dispatch"] for t in trans if t["to"] == "S"]
    gap_mh = [t["first_dispatch"] for t in trans if t["to"] == "H"]
    T = lambda w, l: tab[w]["T"][l]
    # S W0 backlog decomposition from two independent runs: 1 manifest (343.3) and x4 (1229.4)
    s4 = e2e["mix-583f2da-static_S_W0x4-221542"]["job_wall_s"]
    t_ss = (s4 - T("W0", "S")) / 3.0
    tau = T("W0", "S") - t_ss
    r_ss = tab["W0"]["tokens"] / t_ss
    n4 = 4 * tab["W0"]["tokens"]
    meas = lambda key: [e["job_wall_s"] for k, e in e2e.items() if key in k]
    arms = []
    # Mix A, full M->S redeploy: W4 on M + measured gap + W0x4 on S (independent x4 run)
    pred = T("W4", "M") + sum(gap_ms) / len(gap_ms) + s4
    arms.append(("mixA full M->S", pred, meas("reconf_M_W4_S_W0x4"), "out-of-sample"))
    pred_lin = T("W4", "M") + sum(gap_ms) / len(gap_ms) + 4 * T("W0", "S")
    arms.append(("mixA full M->S (linear k*T)", pred_lin, meas("reconf_M_W4_S_W0x4"),
                 "out-of-sample"))
    # Mix A, partial: kept pair serves until join, rest at S steady rate + S drain tail
    f_obs, preds = [], []
    for key in ("partial-b2458d2-mixA-r1", "partial-2d003a4-mixA-r2c"):
        e = e2e[key]
        tl, pre = e["transition"]["timeline"], e["transition"]["join_partition"]["pre_join"]
        join_at = tl["join_after_phase2_start_s"]
        kept_tok = pre["useful_output_tokens"]
        f_obs.append(kept_tok / (join_at * r_ss))
        preds.append(T("W4", "M") + PARTIAL_GAP + join_at + (n4 - kept_tok) / r_ss + tau)
    pm = [x for k, x in ((k, e["job_wall_s"]) for k, e in e2e.items()) if "mixA-r" in k]
    arms.append(("mixA partial M->S", sum(preds) / 2, pm,
                 "kept-pair tokens and join time from the same runs; makespan not fitted"))
    # Mix A, static M from blind single-manifest M rates (policy confound expected)
    arms.append(("mixA static M (blind rates, -O)", T("W4", "M") + 4 * T("W0", "M") - O_MIXA,
                 meas("static_M_MIX_W4_W0x4"), "out-of-sample; O in-sample"))
    # Control
    arms.append(("control full M->H (blind W5 on H)",
                 T("W4", "M") + sum(gap_mh) / len(gap_mh) + T("W5", "H"),
                 meas("reconf_M_W4_H_W5"), "out-of-sample"))
    arms.append(("control static M (blind, O=0)", T("W4", "M") + T("W5", "M"),
                 meas("static_M_MIX_W4_W5"), "out-of-sample"))
    out = []
    for name, p, m, note in arms:
        mm = sum(m) / len(m)
        out.append({"arm": name, "pred_s": round(p, 1), "meas_s": round(mm, 1), "reps": m,
                    "res_s": round(p - mm, 1), "res_pct": round(100 * (p - mm) / mm, 2),
                    "note": note})
    # Join tail implied by the measured effective partial cost 186.47 s (mixA_model.json)
    c_eff_meas = json.loads((EVID / "mixA_model.json").read_text())["C_partial_effective_s"]
    f_mean = sum(f_obs) / len(f_obs)
    d_mean = 363.7
    join_tail = c_eff_meas - PARTIAL_GAP - d_mean * (1 - f_mean)
    calib = {"S_W0_t_ss_s": round(t_ss, 2), "S_W0_tau_s": round(tau, 2),
             "S_W0_r_ss_tok_s": round(r_ss, 1), "f_W0_pair_vs_S": [round(x, 3) for x in f_obs],
             "C_partial_eff_meas_s": round(c_eff_meas, 2), "join_tail_implied_s": round(join_tail, 1)}
    return out, calib


def partial_cost(gap, d, f, tail):
    return gap + d * (1.0 - f) + tail


def scenario(tab, regimes, calib, f_steps, overlaps):
    T = lambda w, l: tab[w]["T"][l]
    has = lambda w, *ls: w in tab and all(l in tab[w]["T"] for l in ls)
    gains = {}
    if has("W0", "H", "S"):
        gains["W0 on S vs H (small phase)"] = T("W0", "H") - T("W0", "S")
    if has("W5", "H", "S"):
        gains["W5 on H vs S (middle phase)"] = T("W5", "S") - T("W5", "H")
    if has("W3", "H", "S"):
        gains["W3 on H vs S (middle phase)"] = T("W3", "S") - T("W3", "H")
    if has("W4", "H", "S"):
        gains["W4 on H vs S"] = T("W4", "S") - T("W4", "H")
    tail = calib["join_tail_implied_s"]
    res = {"gain_per_manifest_s": {k: round(v, 1) for k, v in gains.items()}, "regimes": {}}
    for name, (c, d, prov) in regimes.items():
        staged = {}
        for tag, tail_v in (("join_tail_meas", tail), ("join_tail_0", 0.0)):
            staged[tag] = round(sum(partial_cost(PARTIAL_GAP, d, f, tail_v)
                                    for f in f_steps), 1)
        row = {"C_full_s": c, "D_s": d, "provenance": prov, "C_staged_S_Mb_H_s": staged,
               "breakeven_manifests": {}, "rho_star_TP32_vs_H": {}}
        for o_name, o in overlaps.items():
            for sw_name, c_eff in (("full", c), ("staged", staged["join_tail_meas"]),
                                   ("staged_tail0", staged["join_tail_0"]),
                                   ("staged_keepO", staged["join_tail_meas"])):
                # staged_keepO: joiners released as they go idle, so the drain tail is not lost (INF)
                o_eff = 0.0 if sw_name == "staged_keepO" else o
                for g_name, g in gains.items():
                    if sw_name != "full" and "small" in g_name:
                        continue  # staged applies to the S->H switch, judged on the middle phase
                    row["breakeven_manifests"][f"{g_name} | {sw_name} | O={o_name}"] = round(
                        (c_eff + o_eff) / g, 2)
            # final H->TP32 switch, not capacity-forced: needs T_H/T_32 > T_H / (T_H - C - O)
            th_h = tab["W4"]["T"]["H"] if "W4" in tab and "H" in tab["W4"]["T"] else 347.1
            for th in (th_h, 1000.0, 2000.0, 4000.0):
                denom = th - c - o
                row["rho_star_TP32_vs_H"][f"T_H={th:g}s | O={o_name}"] = (
                    round(th / denom, 3) if denom > 0 else None)
        res["regimes"][name] = row
    return res


def job_makespan(tab, plan, c_switch, o, rho32=None):
    """plan: list of (class, manifests, layout); layout 'TP32' uses T_H / rho32."""
    total, prev = 0.0, None
    for w, k, lay in plan:
        t = tab[w]["T"].get("H")
        t = t / rho32 if lay == "TP32" and t is not None else tab[w]["T"].get(lay)
        if t is None:
            return None
        total += k * t
        if prev is not None:
            total += (c_switch + o) if lay != prev else -o
        prev = lay
    return round(total, 1)


def kv_capacity():
    """Per-instance KV pool by TP from two measured pools plus head-padding rules (DERIVED).

    Per chip and token: 50 sliding layers x ceil(16/tp) heads x 256 + 10 global layers x 1 head
    x 512 (4 global KV heads padded/replicated to tp), K and V, kv_cache_dtype float32 in the
    campaign deploys (deploy-bab4f9e-H.json:1500).  Sliding layers allocate full length
    (tpu-inference kv_cache_manager.py:537-538).  Free HBM per chip F(tp) = U - W/tp.
    """
    w = 62_546_338_248  # checkpoint bytes (init_tp16.json:194-200)
    per_chip = lambda tp, by: (50 * max(16 // tp, 1) * 256 + 10 * 512) * 2 * by
    meas = {8: 26_368, 16: 72_192}  # fp32 pools at max_model_len 16,384/36,864 (w4 typ:143-144)
    f = {tp: n * per_chip(tp, 4) for tp, n in meas.items()}
    u = f[16] + w / 16
    check = (f[16] - f[8]) / (w / 8 - w / 16) - 1.0
    out = {"consistency_residual_pct": round(100 * check, 2), "U_GB_per_chip": round(u / 1e9, 3)}
    for tp in (8, 16, 32):
        free = u - w / tp
        out[f"TP{tp}"] = {"weights_GB_per_chip": round(w / tp / 1e9, 2),
                          "fp32_tokens": int(free / per_chip(tp, 4)),
                          "bf16_tokens": int(free / per_chip(tp, 2)),
                          "measured_fp32": meas.get(tp)}
    out["TP16"]["measured_bf16_8192"] = 145_152  # decode_allocation_20260913/init_tp16.json:1325
    return out



# ---------------------------------------------------------------------------------------------
# Long-context job: k_s W0 manifests (small phase) then k_l W6 manifests, all known at t=0.
# W6 does not fit a TP8 (KV pool 26,368 < one request), so static S is infeasible.
# ---------------------------------------------------------------------------------------------
# Long-job convention: static H = k_s a0H + k_l t6H - O; S->H = k_s a0S + C + k_l t6H (switch adds C
# only, as in scenario()), so k_s* = (C + O)/(a0H - a0S).
LONG_C = {  # name: (C s, note)
    "full_warm_production": (160.5, "full warm production switch (verified-c.md:24)"),
    "staged_effective": (186.0, "EFFECTIVE cost of a staged/partial transition (mixA_model.json "
                                "C_partial_effective_s 186.47), not a full-switch C"),
    "projected_levers": (80.0, "INF projected with levers (verified-c.md:37-41)"),
    "projected_no_trace": (45.0, "INF projected, also no per-start trace+lower (verified-c.md:42)"),
}
LONG_ARMS = ("H", "M", "h", "m")  # static H, static M, S->H, S->M
LONG_ARM_NAMES = {"H": "static H", "M": "static M", "h": "S->H", "m": "S->M"}
PLACEHOLDER_TAG = "PLACEHOLDER, not measured"
MID_ARMS = ("S", "H", "h7", "h6")  # static S, static H, S->H before W7, S->H before W6
MID_ARM_NAMES = {"S": "static S", "H": "static H", "h7": "S->H before W7", "h6": "S->H before W6"}
MID_PLACEHOLDERS = {"t7S": 300.0, "t7H": 240.0}  # INF, tagged
LONG_SPEC = {"a0S": ("W0", "S"), "a0H": ("W0", "H"), "a0M": ("W0", "M_2xTP8"),
             "t6H": ("W6", "H"), "t6M": ("W6", "M_TP16")}


def long_rates(tab, ltab=None, placeholder=False):
    """Resolve a0S, a0H, a0M, t6H, t6M (+ source strings).  Cells come from the long table, then
    the main table; with placeholder=True only the missing ones become tagged placeholders
    (a0M = 2 a0S, t6M = 2 t6H, t6H = 420 s)."""
    r, src, ph = {}, {}, []
    for k, (w, lay) in LONG_SPEC.items():
        r[k], src[k] = None, None
        for t in (ltab, tab):
            if t and lay in t.get(w, {}).get("T", {}):
                r[k] = t[w]["T"][lay]
                src[k] = t[w].get("sources", {}).get(lay) or f"table {w}[{lay}]"
                break
    if placeholder:
        if r["a0S"] is None or r["a0H"] is None:
            raise ValueError("long-job needs measured W0 S and H rates even with --long-placeholder")
        if r["t6H"] is None:
            r["t6H"], src["t6H"] = 420.0, f"t6H = 420 s ({PLACEHOLDER_TAG})"
            ph.append("t6H")
        if r["a0M"] is None:
            r["a0M"], src["a0M"] = 2.0 * r["a0S"], f"a0M = 2*a0S ({PLACEHOLDER_TAG})"
            ph.append("a0M")
        if r["t6M"] is None:
            r["t6M"], src["t6M"] = 2.0 * r["t6H"], f"t6M = 2*t6H ({PLACEHOLDER_TAG})"
            ph.append("t6M")
    missing = [k for k, v in r.items() if v is None]
    if missing:
        raise ValueError(f"long-job rates missing: {missing}; pass --long-table or --long-placeholder")
    return r, src, ph


def s_to_m(r, ks, kl, c):
    """Partial S->M: at tau 2 of 4 TP8 become the TP16 (W6 starts at tau+C); the other 2 keep
    W0.  t0(tau) = tau + (ks - tau/a0S) a0M, t6(tau) = tau + C + kl t6M, tau in [0, ks a0S].
    With rho = a0M/a0S > 1, t0 falls and t6 rises in tau, so the optimum is the crossing
        tau* = (ks a0M - C - kl t6M)/rho = (ks - (C + kl t6M)/a0M) a0S,  clipped at 0
    (never above ks a0S because C + kl t6M > 0).  Makespan there
        ks a0S + (C + kl t6M)(1 - a0S/a0M)   if ks a0M > C + kl t6M, else C + kl t6M.
    rho <= 1: t0 does not fall, tau = 0 is optimal."""
    a0s, a0m, lt = r["a0S"], r["a0M"], kl * r["t6M"]
    rho = a0m / a0s
    tau = min(max((ks * a0m - c - lt) / rho, 0.0), ks * a0s) if rho > 1.0 else 0.0
    t0 = tau + (ks - tau / a0s) * a0m
    t6 = tau + c + lt
    return {"tau": tau, "t0": t0, "t6": t6, "makespan": max(t0, t6)}


def s_to_m_numeric(r, ks, kl, c, n=20001):
    """Brute-force min over tau of max(t0, t6) on an n-point grid (test oracle)."""
    tmax = ks * r["a0S"]
    return min(max(tau + (ks - tau / r["a0S"]) * r["a0M"], tau + c + kl * r["t6M"])
               for tau in (tmax * i / (n - 1) for i in range(n)))


def long_arms(r, ks, kl, c, o, cp=None):
    """Makespans (s) of the four arms.  Convention (scenario()/published R3): a switch adds C only;
    static H keeps the same-layout overlap credit -O, applied only where both phases exist."""
    ob = o if (ks > 0 and kl > 0) else 0.0
    cp = c if cp is None else cp  # S->M cost: c_partial when given, else C
    return {"H": ks * r["a0H"] + kl * r["t6H"] - ob,
            "M": max(ks * r["a0M"], kl * r["t6M"]),
            "h": ks * r["a0S"] + c + kl * r["t6H"],
            "m": s_to_m(r, ks, kl, cp)["makespan"]}


def long_winner(r, ks, kl, c, o, cp=None):
    ms = long_arms(r, ks, kl, c, o, cp)
    order = sorted(LONG_ARMS, key=lambda a: (ms[a], LONG_ARMS.index(a)))
    return order[0], ms[order[1]] - ms[order[0]], ms


def frange(spec):
    """'start:stop:step' (inclusive) or a JSON list."""
    s = str(spec).strip()
    if s.startswith("["):
        return [float(x) for x in json.loads(s)]
    a, b, st = (float(x) for x in s.split(":"))
    return [round(a + i * st, 10) for i in range(int(round((b - a) / st)) + 1)]


def region_map(r, kss, kls, c, o, cp=None):
    rows, cells = [], {}
    for kl in kls:
        line = ""
        for ks in kss:
            w, m, ms = long_winner(r, ks, kl, c, o, cp)
            line += w
            cells[f"{ks:g},{kl:g}"] = {"winner": LONG_ARM_NAMES[w], "margin_s": round(m, 2),
                                      "makespan_s": {LONG_ARM_NAMES[a]: round(ms[a], 1) for a in LONG_ARMS}}
        rows.append(f"{kl:6g} {line}")
    return {"ascii": rows[::-1], "cells": cells}


def winner_ranges(r, kl, c, o, ks_max, step=0.01, cp=None):
    """k_s intervals (fixed k_l) where each arm wins; breakpoints refined by bisection."""
    xs = [i * step for i in range(int(round(ks_max / step)) + 1)]
    win = lambda x: long_winner(r, x, kl, c, o, cp)[0]
    out, start, cur = [], 0.0, win(xs[0])
    for a, b in zip(xs, xs[1:]):
        if win(b) != cur:
            lo, hi = a, b
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if win(mid) == cur else (lo, mid)
            out.append({"arm": LONG_ARM_NAMES[cur], "k_s": [round(start, 4), round(hi, 4)]})
            start, cur = hi, win(b)
    out.append({"arm": LONG_ARM_NAMES[cur], "k_s": [round(start, 4), ks_max]})
    return out


def long_breakevens(r, c, o, kl_list, ks_max, cp=None):
    cm = c if cp is None else cp  # S->M cost
    a0s, a0h, a0m, t6h, t6m = (r[k] for k in ("a0S", "a0H", "a0M", "t6H", "t6M"))
    d = a0h - a0s
    res = {
        # S->H < static H  <=>  ks (a0H - a0S) > C + O  (switch adds C; static H has -O)
        "S->H_vs_static_H_k_s_star": (c + o) / d if d > 0 else None,
        "S->H_formula": "k_s* = (C + O)/(a0H - a0S); independent of k_l",
        "S->M_vs_static_M_formula": "S->M < static M iff a0M > a0S and k_s > (C + k_l t6M)/a0M",
        "S->M_vs_static_M_k_s_by_k_l": {f"{kl:g}": ((cm + kl * t6m) / a0m if a0m > a0s else None)
                                        for kl in kl_list},
        "tau_star_formula": "tau* = max(0, (k_s - (C + k_l t6M)/a0M) a0S)",
        "static_M_vs_static_H_formula": "M < H: W6-bound (k_s a0M <= k_l t6M): k_s > (k_l (t6M - t6H) + O)/a0H; "
                                        "W0-bound: k_s (a0M - a0H) < k_l t6H - O",
        "static_M_vs_static_H_by_k_l": {},
        "ranges_by_k_l": {},
    }
    for kl in kl_list:
        pieces = []
        kb = kl * t6m / a0m  # W6-bound below this k_s, W0-bound above
        xb = (kl * (t6m - t6h) + o) / a0h
        if xb <= kb:
            pieces.append({"regime": "W6-bound", "M_better_above_k_s": xb})
        if a0m > a0h:
            xa = (kl * t6h - o) / (a0m - a0h)
            if xa >= kb:
                pieces.append({"regime": "W0-bound", "M_better_below_k_s": xa})
        elif kl * t6h - o > 0:
            pieces.append({"regime": "W0-bound", "M_better_for_all_k_s": True})
        res["static_M_vs_static_H_by_k_l"][f"{kl:g}"] = {"k_s_W6_W0_split": kb, "pieces": pieces}
        res["ranges_by_k_l"][f"{kl:g}"] = winner_ranges(r, kl, c, o, ks_max, cp=cp)
    return res


def long_job(r, src, placeholders, c_map, kss, kls, o_vals, kl_list=(0.5, 1, 2, 4, 8), cp=None):
    out = {"inputs": {k: {"value": round(v, 3), "source": src[k], "placeholder": k in placeholders}
                      for k, v in r.items()},
           "placeholder_inputs": list(placeholders), "any_placeholder": bool(placeholders),
           "O_values_s": o_vals, "C_values_s": {k: v[0] for k, v in c_map.items()},
           "C_notes": {k: v[1] for k, v in c_map.items()},
           "grid": {"k_s": kss, "k_l": kls},
           "legend": {"H": "static H", "M": "static M (W0 on 2xTP8 || W6 on TP16, INF linear scaling)",
                      "h": "S->H", "m": "S->M (partial, 2 of 4 TP8 -> TP16)"},
           "static_S": "infeasible: W6 does not fit a TP8 KV pool",
           "k_l=0": "pure W0 job -> static S applies; the grid starts at k_l > 0",
           "convention": "scenario()/published R3: a switching arm adds C only (no +O); static H keeps the "
                         "same-layout overlap credit -O (only when both phases exist); S->M has no O term",
           "k_s=0": "S->X arms still pay C (job assumed to start on S)",
           "results": {}}
    for oname, o in o_vals.items():
        out["results"][oname] = {}
        for cname, (c, _) in c_map.items():
            out["results"][oname][cname] = {
                "C_s": c, "O_s": o, "map": region_map(r, kss, kls, c, o, cp),
                "break_even": long_breakevens(r, c, o, kl_list, max(kss), cp)}
            if cp is not None:
                out["results"][oname][cname]["C_partial_s"] = cp
    return out


def print_long_job(lj):
    tag = (" [PLACEHOLDER INPUTS: " + ", ".join(lj["placeholder_inputs"]) + "]") if lj["any_placeholder"] else ""
    print(f"\nlong-context job (k_s W0 then k_l W6){tag}")
    for k, v in lj["inputs"].items():
        print(f"  {k} = {v['value']} s  <- {v['source']}")
    print("  legend: " + "  ".join(f"{k}={v}" for k, v in lj["legend"].items()))
    first = next(iter(lj["results"]))
    for oname, per_c in lj["results"].items():
        for cname, res in per_c.items():
            be = res["break_even"]
            print(f"\n  O={oname} ({res['O_s']} s), C={cname} ({res['C_s']} s){tag}")
            print(f"  S->H beats static H for k_s > {be['S->H_vs_static_H_k_s_star']:.2f}")
            if oname != first:
                continue
            ks = lj["grid"]["k_s"]
            print(f"  rows k_l (top {lj['grid']['k_l'][-1]:g}), cols k_s {ks[0]:g}..{ks[-1]:g}")
            for row in res["map"]["ascii"]:
                print("  " + row)
            for kl, rg in be["ranges_by_k_l"].items():
                print(f"  k_l={kl}: " + "; ".join(f"{x['arm']} {x['k_s'][0]:g}-{x['k_s'][1]:g}" for x in rg))


def mid_rates(tab, ltab=None, mtab=None, placeholder=False):
    """Resolve a0S, a0H, t7S, t7H, t6H (+ sources).  Cells come from the mid table, the long table,
    then the main table.  With placeholder=True missing t7S/t7H (300/240 s) and t6H (420 s) become
    tagged INF placeholders; a0S/a0H must always be measured."""
    spec = {"a0S": ("W0", "S"), "a0H": ("W0", "H"), "t7S": ("W7", "S"), "t7H": ("W7", "H"),
            "t6H": ("W6", "H")}
    r, src, ph = {}, {}, []
    for k, (w, lay) in spec.items():
        r[k], src[k] = None, None
        for t in (mtab, ltab, tab):
            if t and lay in t.get(w, {}).get("T", {}):
                r[k] = t[w]["T"][lay]
                src[k] = t[w].get("sources", {}).get(lay) or f"table {w}[{lay}]"
                break
    if placeholder:
        for k, v in {**MID_PLACEHOLDERS, "t6H": 420.0}.items():
            if r[k] is None:
                r[k], src[k] = v, f"{k} = {v:g} s (INF {PLACEHOLDER_TAG})"
                ph.append(k)
    missing = [k for k, v in r.items() if v is None]
    if missing:
        raise ValueError(f"mid-job rates missing: {missing}; pass --mid-table or --mid-placeholder")
    return r, src, ph


def mid_arms(r, ks, km, kl, c, o):
    """Makespans (s) of the mid-job arms; None marks an infeasible arm (static S with W6, S->H before
    W6 without W6).  A static arm keeps overlap O once, only if it has >= 2 non-empty phases."""
    ob = o if sum(x > 0 for x in (ks, km, kl)) >= 2 else 0.0
    return {"S": ks * r["a0S"] + km * r["t7S"] - ob if kl == 0 else None,
            "H": ks * r["a0H"] + km * r["t7H"] + kl * r["t6H"] - ob,
            "h7": ks * r["a0S"] + c + km * r["t7H"] + kl * r["t6H"],
            "h6": ks * r["a0S"] + km * r["t7S"] + c + kl * r["t6H"] if kl > 0 else None}


def mid_winner(r, ks, km, kl, c, o):
    ms = mid_arms(r, ks, km, kl, c, o)
    order = sorted((a for a in MID_ARMS if ms[a] is not None), key=lambda a: (ms[a], MID_ARMS.index(a)))
    margin = ms[order[1]] - ms[order[0]] if len(order) > 1 else None
    return order[0], margin, ms


def mid_breakevens(r, c, o):
    d7, dh = r["t7S"] - r["t7H"], r["a0H"] - r["a0S"]
    return {"k_m_star": (c + o) / d7 if d7 > 0 else None,
            "k_m_formula": "k_m* = (C + O)/(t7S - t7H): S->H before W7 beats static S (k_l = 0); none if t7S <= t7H",
            "k_s_star": (c + o) / dh if dh > 0 else None,
            "k_s_formula": "k_s* = (C + O)/(a0H - a0S): S->H beats static H; independent of k_m, k_l",
            "h6_vs_h7": "S->H before W6 - S->H before W7 = k_m (t7S - t7H): W7-first wins iff t7S > t7H"}


def mid_job(r, src, placeholders, c_map, kss, kms, kls, o_vals):
    out = {"inputs": {k: {"value": round(v, 3), "source": src[k], "placeholder": k in placeholders}
                      for k, v in r.items()},
           "placeholder_inputs": list(placeholders), "any_placeholder": bool(placeholders),
           "O_values_s": o_vals, "C_values_s": {k: v[0] for k, v in c_map.items()},
           "C_notes": {k: v[1] for k, v in c_map.items()},
           "grid": {"k_s": kss, "k_m": kms, "k_l": kls},
           "legend": {"S": "static S (k_l = 0 only)", "H": "static H", "h7": "S->H before W7",
                      "h6": "S->H before W6 (k_l > 0 only)"},
           "convention": "a switching arm adds C only; a static arm keeps O once when it has >= 2 non-empty phases",
           "results": {}}
    for oname, o in o_vals.items():
        out["results"][oname] = {}
        for cname, (c, _) in c_map.items():
            maps, cells = {}, {}
            for kl in kls:
                rows = []
                for km in kms:
                    line = ""
                    for ks in kss:
                        w, m, ms = mid_winner(r, ks, km, kl, c, o)
                        line += {"S": "S", "H": "H", "h7": "a", "h6": "b"}[w]
                        cells[f"{ks:g},{km:g},{kl:g}"] = {
                            "winner": MID_ARM_NAMES[w], "margin_s": None if m is None else round(m, 2),
                            "makespan_s": {MID_ARM_NAMES[a]: (None if ms[a] is None else round(ms[a], 1))
                                           for a in MID_ARMS}}
                    rows.append(f"{km:6g} {line}")
                maps[f"{kl:g}"] = rows[::-1]
            out["results"][oname][cname] = {"C_s": c, "O_s": o, "ascii_by_k_l": maps, "cells": cells,
                                            "break_even": mid_breakevens(r, c, o)}
    return out


def print_mid_job(mj):
    tag = (" [PLACEHOLDER INPUTS: " + ", ".join(mj["placeholder_inputs"]) + "]") if mj["any_placeholder"] else ""
    print(f"\nmid-band job (k_s W0, k_m W7, k_l W6){tag}")
    for k, v in mj["inputs"].items():
        print(f"  {k} = {v['value']} s  <- {v['source']}")
    print("  legend: S=static S  H=static H  a=S->H before W7  b=S->H before W6 (rows k_m, cols k_s)")
    first = next(iter(mj["results"]))
    for oname, per_c in mj["results"].items():
        for cname, res in per_c.items():
            be = res["break_even"]
            ks_star, km_star = be["k_s_star"], be["k_m_star"]
            print(f"\n  O={oname} ({res['O_s']} s), C={cname} ({res['C_s']} s){tag}")
            print("  S->H beats static H for k_s > " + ("none" if ks_star is None else f"{ks_star:.2f}")
                  + "; S->H before W7 beats static S (k_l=0) for k_m > "
                  + ("none" if km_star is None else f"{km_star:.2f}"))
            if oname != first:
                continue
            ks, kms = mj["grid"]["k_s"], mj["grid"]["k_m"]
            for kl, rows in res["ascii_by_k_l"].items():
                print(f"  k_l={kl}: rows k_m (top {kms[-1]:g}), cols k_s {ks[0]:g}..{ks[-1]:g}")
                for row in rows:
                    print("  " + row)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regimes", help="JSON {name: [C, D, provenance]} to replace defaults")
    ap.add_argument("--f-steps", default="0.55,0.56",
                    help="kept fraction of H's middle-phase rate during each staged step (INF: "
                         "2xTP8 0.47-0.62, 1xTP16 0.50-0.62)")
    ap.add_argument("--rho32", default="0.6,0.8,1.0,1.2", help="TP32/H rate ratios to tabulate")
    ap.add_argument("--json", default="results/bulk_cost_model/model.json")
    ap.add_argument("--table", default=None,
                    help="alternative throughput table (same format as throughput_table_measured.json)")
    ap.add_argument("--long-table", default=None,
                    help="rate table JSON with W0 S/H/M_2xTP8 and W6 H/M_TP16 cells (may equal --table); "
                         "absent -> long_job skipped")
    ap.add_argument("--long-placeholder", action="store_true",
                    help="dry run: fill missing W6/M cells with tagged placeholders (a0M=2 a0S, t6M=2 t6H, t6H=420)")
    ap.add_argument("--long-c", default=None,
                    help="JSON {name: C or [C, note]} replacing the long-job switch costs")
    ap.add_argument("--long-ks", default="0:8:0.25", help="k_s grid start:stop:step or JSON list")
    ap.add_argument("--long-kl", default="0.25:8:0.25", help="k_l grid start:stop:step or JSON list")
    ap.add_argument("--mid-table", default=None,
                    help="rate table JSON (same format) with W7 S/H cells (t7S, t7H), from long_rate_table.py --w7-*")
    ap.add_argument("--mid-placeholder", action="store_true",
                    help="dry run: fill missing t7S=300, t7H=240 (and t6H=420) with tagged INF placeholders")
    ap.add_argument("--mid-ks", default="0:8:0.25", help="mid job k_s grid start:stop:step or JSON list")
    ap.add_argument("--mid-km", default="0:8:0.25", help="mid job k_m grid start:stop:step or JSON list")
    ap.add_argument("--mid-kl", default="[0,1,2]", help="mid job k_l values (JSON list or start:stop:step)")
    ap.add_argument("--c-full", type=float, default=None,
                    help="measured full-switch cost (s): replaces the C regimes for S->H arms of the long and mid jobs")
    ap.add_argument("--c-partial", type=float, default=None,
                    help="measured partial-switch cost (s): used by the S->M arm of the long job")
    ap.add_argument("--c-full-src", default="CLI --c-full", help="provenance of --c-full")
    ap.add_argument("--c-partial-src", default="CLI --c-partial", help="provenance of --c-partial")
    ap.add_argument("--a0h-override", type=float, default=None,
                    help="W0 on H at max_model_len 131,072 (s/manifest): replaces a0H in the long and mid jobs")
    ap.add_argument("--a0h-src", default="CLI --a0h-override", help="provenance of --a0h-override")
    ap.add_argument("--kv-measured", default=None,
                    help="JSON of measured KV pools, e.g. '{\"TP8\": 52480, \"TP16\": 144896, \"source\": \"...\"}'")
    a = ap.parse_args(argv)
    regimes = {k: tuple(v) for k, v in json.loads(a.regimes).items()} if a.regimes else REGIMES
    pub_tab = load_table()
    tab = load_table(a.table) if a.table else pub_tab
    be = validate_breakeven(pub_tab)
    arms, calib = validate_mixes(pub_tab)
    f_steps = [float(x) for x in a.f_steps.split(",")]
    overlaps = {"0": 0.0, "S_tail_32.6": O_TAIL_S, "mixA_115": O_MIXA}
    sc = scenario(tab, regimes, calib, f_steps, overlaps)
    # Example job: 4 W0 (small) -> 2 W5 (middle) -> final batch of W4-like work, one H manifest
    ex = {}
    for name, (c, _, _) in regimes.items():
        for rho in [float(x) for x in a.rho32.split(",")]:
            ex[f"{name} rho={rho:g}"] = {
                "static H": job_makespan(tab, [("W0", 4, "H"), ("W5", 2, "H"), ("W4", 1, "H")], c, O_TAIL_S),
                "static M": job_makespan(tab, [("W0", 4, "M"), ("W5", 2, "M"), ("W4", 1, "M")], c, O_TAIL_S),
                "S->H": job_makespan(tab, [("W0", 4, "S"), ("W5", 2, "H"), ("W4", 1, "H")], c, O_TAIL_S),
                "S->H->TP32": job_makespan(tab, [("W0", 4, "S"), ("W5", 2, "H"), ("W4", 1, "TP32")], c, O_TAIL_S, rho),
                "S->TP32": job_makespan(tab, [("W0", 4, "S"), ("W5", 2, "S"), ("W4", 1, "TP32")], c, O_TAIL_S, rho),
            }
    kv = kv_capacity()
    pub_path = str(EVID / "throughput_table_measured.json")
    out = {"breakeven_validation": be, "mix_validation": arms, "calibration": calib,
           "kv_capacity": kv, "scenario": sc, "example_job_4W0_2W5_1W4": ex, "regimes": regimes,
           "table": str(a.table or pub_path), "validation_table": pub_path}
    if a.kv_measured:
        kv["measured_bf16_current"] = json.loads(a.kv_measured)
    def c_map_for(long_c=None):
        if a.c_full is not None:
            return {"measured_full": (a.c_full, f"--c-full: {a.c_full_src}")}
        return LONG_C if not long_c else {
            k: ((float(v[0]), v[1]) if isinstance(v, list) else (float(v), "user override"))
            for k, v in json.loads(long_c).items()}

    def measured_c():
        return {"c_full_s": a.c_full, "c_full_source": a.c_full_src if a.c_full is not None else None,
                "c_partial_s": a.c_partial,
                "c_partial_source": a.c_partial_src if a.c_partial is not None else None,
                "routing": "S->H arms use c_full (replacing the C regimes); the S->M arm uses c_partial"}

    def apply_a0h(r, src):
        if a.a0h_override is not None:
            r["a0H"], src["a0H"] = a.a0h_override, f"a0H@131k override: {a.a0h_src}"

    lj = None
    if a.long_table or a.long_placeholder:
        ltab = load_table(a.long_table) if a.long_table else None
        lr, lsrc, lph = long_rates(tab, ltab, a.long_placeholder)
        apply_a0h(lr, lsrc)
        lj = long_job(lr, lsrc, lph, c_map_for(a.long_c), frange(a.long_ks), frange(a.long_kl),
                      {f"S_tail_{O_TAIL_S:g}": O_TAIL_S, "0": 0.0}, cp=a.c_partial)
        if a.c_full is not None or a.c_partial is not None:
            lj["measured_C"] = measured_c()
        lj["C_staged_note"] = "186 s is an EFFECTIVE cost for staged/partial transitions; every switching arm uses C as given"
        out["long_job"] = lj
    mj = None
    if a.mid_table or a.mid_placeholder:
        mtab = load_table(a.mid_table) if a.mid_table else None
        ltab = load_table(a.long_table) if a.long_table else None
        mr, msrc, mph = mid_rates(tab, ltab, mtab, a.mid_placeholder)
        apply_a0h(mr, msrc)
        mj = mid_job(mr, msrc, mph, c_map_for(a.long_c), frange(a.mid_ks), frange(a.mid_km),
                     frange(a.mid_kl), {f"S_tail_{O_TAIL_S:g}": O_TAIL_S, "0": 0.0})
        if a.c_full is not None or a.c_partial is not None:
            mj["measured_C"] = measured_c()
        out["mid_job"] = mj
    if a.a0h_override is not None:
        out["a0h_override"] = {"value_s": a.a0h_override, "source": a.a0h_src,
                               "applied_to": [k for k, v in (("long_job", lj), ("mid_job", mj)) if v]}
    p = pathlib.Path(a.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    worst = max(abs(r["res_k"]) for r in be)
    print(f"break-even rows: {len(be)}; max |k residual| = {worst:.3f} manifests")
    for r in be:
        print(f"  {r['row']:<10} k={r['k']:<6} pub={r['k_pub']:<6} range={r['range']} pub={r['range_pub']}")
    for r in arms:
        print(f"  {r['arm']:<36} pred {r['pred_s']:>7}  meas {r['meas_s']:>7}  res {r['res_s']:>+7} ({r['res_pct']:+.2f}%)")
    print("calibration:", json.dumps(calib))
    print("KV pool per instance (DERIVED):", json.dumps(kv))
    if a.kv_measured:
        print("KV pool measured (bf16, current):", json.dumps(kv["measured_bf16_current"]))
    print("\nregime: C | staged S->Mb->H (tail meas / tail 0) | k* W0 small, W5 full, W5 staged, "
          "W5 staged_tail0 (O = S tail) | rho* TP32/H at T_H = 1000 / 2000 s (O = S tail)")
    key = "| O=S_tail_32.6"
    for name, row in sc["regimes"].items():
        b, r = row["breakeven_manifests"], row["rho_star_TP32_vs_H"]
        st = row["C_staged_S_Mb_H_s"]
        g = lambda k: b.get(k, "-")
        print(f"  {name:<22} {row['C_full_s']:>6} | {st['join_tail_meas']:>6} / {st['join_tail_0']:>6} | "
              f"{g('W0 on S vs H (small phase) | full ' + key):>5} "
              f"{g('W5 on H vs S (middle phase) | full ' + key):>5} "
              f"{g('W5 on H vs S (middle phase) | staged ' + key):>5} "
              f"{g('W5 on H vs S (middle phase) | staged_tail0 ' + key):>5} | "
              f"{r.get('T_H=1000s ' + key)} / {r.get('T_H=2000s ' + key)}")
    print("\nexample job (4 W0, 2 W5, 1 W4-like final; O = S tail 32.6 s):")
    for k, v in ex.items():
        print(f"  {k}: {v}")
    if lj:
        print_long_job(lj)
    if mj:
        print_mid_job(mj)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
