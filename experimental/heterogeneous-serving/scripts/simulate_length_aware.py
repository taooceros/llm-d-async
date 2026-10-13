#!/usr/bin/env python3
"""Step-level replay of a bulk backlog on TPU engines: decode-length-blind vs length-aware scheduling.

Step time is fitted on the measured W3 iterations (mean absolute error about 1 ms per step):
  t = c0 + c1 * decode_requests + c2 * prefill_tokens + c3 * context_tokens
`blind` mimics the measured system: the gateway hands every request to an engine up front
(round robin; W3 prompts are small), and each engine's vLLM scheduler runs FCFS with a 2048-token
step budget.  When a running request cannot grow, the newest running request is preempted and
later recomputed (prompt plus generated tokens).
`aware_*` policies know each request's output length.  An engine admits a request only if its
future KV trajectory (every running request grows to its known end, then frees its blocks) stays
within capacity, so nothing is ever preempted.  Engines pull from one shared queue whenever they
can admit, which also balances work across engines of different speed.

Usage: simulate_length_aware.py <manifest.json> [--layouts H,S,M] [--policies ...] [--kv-scale X]
"""
import argparse
import json

import numpy as np

# (c0 s, c1 s per decode request, c2 s per prefill token, c3 s per context token), w3-c7edd5a-r1.
STEP_MODEL = {"tp16": (11.58e-3, 1.014e-3, 139.5e-6, 0.062e-6),
              "tp8": (16.17e-3, 0.845e-3, 200.7e-6, 0.046e-6)}
KV_BLOCKS = {"tp16": 1134, "tp8": 412}  # 64-token blocks at max_model_len 16384, as deployed for W3
BLOCK = 64
LAYOUTS = {"H": ["tp16", "tp16"], "S": ["tp8"] * 4, "M": ["tp16", "tp8", "tp8"]}
TOKEN_BUDGET = 2048
RETRY_STEPS = 16  # aware engines retry admission after a finish, or every RETRY_STEPS steps


def nblocks(tokens):
    return -(-tokens // BLOCK)


class Req:
    __slots__ = ("rid", "prompt", "decode", "done", "prefilled", "target", "preemptions")

    def __init__(self, rid, prompt, decode):
        self.rid, self.prompt, self.decode = rid, prompt, decode
        self.done = 0         # output tokens emitted
        self.prefilled = 0    # tokens of the current (re)prefill computed so far
        self.target = prompt  # tokens the current prefill must reach
        self.preemptions = 0

    def decoding(self):
        return self.prefilled >= self.target

    def ctx(self):  # tokens whose KV the attention reads this step
        return self.prompt + self.done if self.decoding() else self.prefilled

    def held(self):  # blocks allocated: a (re)prefill reserves its whole target on admission
        return nblocks(self.prompt + self.done if self.decoding() else self.target)


class Engine:
    def __init__(self, name, kind, kv_scale):
        self.name, self.kind = name, kind
        self.cap = int(KV_BLOCKS[kind] * kv_scale)
        self.model = STEP_MODEL[kind]
        self.running, self.waiting = [], []
        self.t = 0.0
        self.last_finish = 0.0
        self.preempted = self.replayed = self.steps_since_try = self.overflows = 0
        self.migrations = 0
        self.migrate = None  # None | "free" | "recompute"
        self.dirty = True
        self.busy_weighted_batch = 0.0
        self.victim = "newest"
        self.horizon = None  # decode steps the admission trajectory must fit; None = to the end

    def used(self):
        return sum(r.held() for r in self.running)

    def fits(self, cand):
        """Whether the KV trajectory with `cand` added stays within capacity."""
        rs = self.running
        ctx = np.array([r.prompt + r.done for r in rs] + [cand.prompt], dtype=np.int64)
        rem = np.array([r.decode - r.done for r in rs] + [cand.decode], dtype=np.int64)
        taus = np.maximum(rem - 1, 0)                 # usage peaks just before each finish
        if self.horizon is not None:
            taus = np.minimum(taus, self.horizon)
        taus = np.unique(taus)
        live = rem[:, None] > taus[None, :]
        use = np.where(live, -(-(ctx[:, None] + taus[None, :] + 1) // BLOCK), 0).sum(axis=0)
        return int(use.max()) <= self.cap


def place_elsewhere(victim, e, engines):
    """Move an evicted or waiting request to another engine instead of recomputing it locally.
    "free": its KV moves with it at no cost (upper bound on migration benefit);
    "recompute": the destination re-prefills prompt + generated tokens (no KV transfer)."""
    need = nblocks(victim.prompt + victim.done) + 2
    dests = [x for x in engines if x is not e and x.cap - x.used() >= need]
    if not dests:
        return False
    x = max(dests, key=lambda x: x.cap - x.used())
    if e.migrate == "recompute" or not victim.decoding():
        victim.target, victim.prefilled = victim.prompt + victim.done, 0
        if victim.done:
            victim.preemptions += 1
    x.running.append(victim)
    e.migrations += 1
    return True


def step(e, aware, engines=()):
    grow = lambda r: nblocks(r.prompt + r.done + 1) - nblocks(r.prompt + r.done)
    need = sum(grow(r) for r in e.running if r.decoding())
    free = e.cap - e.used()
    preempted = False
    while need > free:
        if aware:
            # Prefill chunks can lag the planned trajectory by a step; counted, never silent.
            e.overflows += 1
        if e.victim == "most_remaining":
            victim = max(e.running, key=lambda r: r.decode - r.done)
            e.running.remove(victim)
        else:
            victim = e.running.pop()   # newest running request (vLLM recompute preemption)
        if victim.decoding():
            need -= grow(victim)
        free += victim.held()
        preempted = True
        if e.migrate and place_elsewhere(victim, e, engines):
            continue
        victim.preemptions += 1
        e.preempted += 1
        victim.target = victim.prompt + victim.done
        victim.prefilled = 0
        e.waiting.insert(0, victim)
    free -= need
    decoding = [r for r in e.running if r.decoding()]
    budget = TOKEN_BUDGET - len(decoding)
    prefill = 0
    for r in e.running:  # continue chunked prefills; their blocks were reserved on admission
        if not r.decoding() and budget > 0:
            n = min(budget, r.target - r.prefilled)
            r.prefilled += n
            prefill += n
            budget -= n
            if r.preemptions:
                e.replayed += n
    # Blind FCFS admission: the head needs blocks for its whole (re)prefill; vLLM schedules no
    # waiting request in a step that preempted.
    while e.waiting and budget > 0 and not preempted:
        r = e.waiting[0]
        if nblocks(r.target) > free:
            break
        e.waiting.pop(0)
        n = min(budget, r.target)
        r.prefilled = n
        free -= nblocks(r.target)
        prefill += n
        budget -= n
        if r.preemptions:
            e.replayed += n
        e.running.append(r)
    ctx = sum(r.ctx() for r in e.running)
    c0, c1, c2, c3 = e.model
    dt = c0 + c1 * len(decoding) + c2 * prefill + c3 * ctx
    e.t += dt
    e.busy_weighted_batch += dt * len(decoding)
    finished = []
    for r in list(e.running):
        if r.decoding():   # includes requests whose prefill completed this step
            r.done += 1
            if r.done >= r.decode:
                e.running.remove(r)
                finished.append(r)
    return finished


LPT = lambda q: sorted(q, key=lambda r: -r.decode)
# name: (queue order, admission, victim, TP16 share of output tokens or None)
#   admission "now": admit if the whole prompt fits the free blocks now (vLLM behaviour);
#   "trajectory": admit only if the full future KV trajectory fits (never preempts);
#   ("horizon", W): the trajectory must fit for the next W decode steps only.
#   victim "newest": vLLM; "most_remaining": keep requests close to finishing, which free KV soonest.
#   share s (mixed layouts): the longest requests holding fraction s of all output tokens are
#   TP16-only while any shorter request is left for TP8 (TP8 falls back to them once none is).
POLICIES = {
    "blind": None,  # static hand-off as measured; uses no output length
    "blind_pull": (lambda q: q, "now", "newest", None),
    "aware_fcfs": (lambda q: q, "trajectory", "newest", None),
    "aware_lpt": (LPT, "trajectory", "newest", None),
    "aware_spt": (lambda q: sorted(q, key=lambda r: r.decode), "trajectory", "newest", None),
    "lpt_now": (LPT, "now", "newest", None),
    "lpt_now_victim": (LPT, "now", "most_remaining", None),
    "lpt_h1000": (LPT, ("horizon", 1000), "most_remaining", None),
    "lpt_h3000": (LPT, ("horizon", 3000), "most_remaining", None),
    "split_now": (LPT, "now", "newest", 0.38),
    "split_h3000": (LPT, ("horizon", 3000), "most_remaining", 0.38),
    "split_traj": (LPT, "trajectory", "newest", 0.38),
    # Fifth field: migration. Evicted requests move to an engine with room, and an engine with
    # room and an empty shared queue steals work from a KV-bound engine (see rebalance).
    "blind_mig_free": (lambda q: q, "now", "newest", None, "free"),
    "blind_mig_rc": (lambda q: q, "now", "newest", None, "recompute"),
    "lpt_now_mig_free": (LPT, "now", "most_remaining", None, "free"),
    "lpt_now_mig_rc": (LPT, "now", "most_remaining", None, "recompute"),
    "lpt_h3000_mig_free": (LPT, ("horizon", 3000), "most_remaining", None, "free"),
}


def rebalance(e, engines):
    """Engine `e` has room and nothing left in the shared queue: take a waiting request from a
    KV-bound engine, or failing that the longest-remaining running request of a KV-bound engine."""
    for donor in sorted(engines, key=lambda x: -len(x.waiting)):
        if donor is e:
            continue
        if donor.waiting and nblocks(donor.waiting[0].target) + 2 <= e.cap - e.used():
            e.running.append(donor.waiting.pop(0))
            e.migrations += 1
            return
        if donor.waiting and donor.running:
            r = max(donor.running, key=lambda r: r.decode - r.done)
            if r.decoding() and nblocks(r.prompt + r.done) + 2 <= e.cap - e.used():
                donor.running.remove(r)
                if e.migrate == "recompute":
                    r.target, r.prefilled = r.prompt + r.done, 0
                    r.preemptions += 1
                e.running.append(r)
                e.migrations += 1
                return


def simulate(manifest_reqs, kinds, policy, kv_scale=1.0):
    reqs = [Req(*r) for r in manifest_reqs]
    engines = [Engine(f"{k}_{i}", k, kv_scale) for i, k in enumerate(kinds)]
    spec = POLICIES[policy]
    queue = []
    if spec:
        order, admission, victim, share = spec[:4]
        queue = order(reqs)
        for e in engines:
            e.victim = victim
            e.horizon = admission[1] if isinstance(admission, tuple) else None
            e.migrate = spec[4] if len(spec) > 4 else None
        long_only = set()
        if share and len({e.kind for e in engines}) > 1:
            total, acc = sum(r.decode for r in reqs), 0
            for r in sorted(reqs, key=lambda r: -r.decode):
                if acc >= share * total:
                    break
                long_only.add(r.rid)
                acc += r.decode
    else:
        admission = None
        # As measured: with both kinds (M0, large_fresh_traffic_share 0.5) every other request
        # goes to the TP16 engine and the rest round-robin over TP8; otherwise round robin.
        large = [e for e in engines if e.kind == "tp16"]
        small = [e for e in engines if e.kind == "tp8"]
        for i, r in enumerate(reqs):
            if large and small:
                pool = large if i % 2 == 0 else small
                pool[(i // 2) % len(pool)].waiting.append(r)
            else:
                engines[i % len(engines)].waiting.append(r)
    strict = admission == "trajectory"
    remaining = len(reqs)
    while remaining:
        active = [e for e in engines if e.running or e.waiting or queue]
        e = min(active, key=lambda x: x.t)
        # Engines pull from the shared queue; their own preempted requests go first.
        if queue and not e.waiting and (admission == "now" or e.dirty
                                        or e.steps_since_try >= RETRY_STEPS or not e.running):
            e.dirty, e.steps_since_try = False, 0
            short_left = any(r.rid not in long_only for r in queue)
            for r in list(queue):
                if e.kind == "tp8" and r.rid in long_only and short_left:
                    continue
                ok = (nblocks(r.prompt + 1) <= e.cap - e.used() if admission == "now"
                      else e.fits(r))
                if ok:
                    queue.remove(r)
                    e.running.append(r)
                    if admission == "now":
                        break  # one per step, like the gateway's poll loop
        if e.migrate and not queue and not e.waiting:
            rebalance(e, engines)
        if not e.running and not e.waiting:
            others = [x.t for x in active if x is not e and x.running]
            if not others:
                raise RuntimeError("a request does not fit an empty engine")
            e.t = min(others)  # idle until another engine's clock catches up
            continue
        e.steps_since_try += 1
        for r in step(e, strict, engines):
            remaining -= 1
            e.dirty = True
            e.last_finish = e.t
    makespan = max(e.last_finish for e in engines)
    return {"makespan_s": makespan,
            "preemptions": sum(e.preempted for e in engines),
            "replayed_tokens": sum(e.replayed for e in engines),
            "migrations": sum(e.migrations for e in engines),
            "overflows": sum(e.overflows for e in engines),
            "last": {e.name: round(e.last_finish, 1) for e in engines},
            "mean_batch": {e.name: round(e.busy_weighted_batch / max(e.t, 1e-9), 1) for e in engines}}


def load(path):
    m = json.load(open(path))
    return [(r["request_id"], r["prompt_tokens"], r["max_tokens"]) for r in m["requests"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--layouts", default="H,S,M")
    ap.add_argument("--policies", default="blind,aware_fcfs,aware_backfill,aware_lpt,aware_spt")
    ap.add_argument("--kv-scale", type=float, default=1.0)
    ap.add_argument("--decode-cost-scale", type=float, default=1.0,
                    help="scale c1 (step time per decode request); 0 = decode throughput linear "
                         "in batch at fixed context")
    args = ap.parse_args()
    for kind, (c0, c1, c2, c3) in STEP_MODEL.items():
        STEP_MODEL[kind] = (c0, c1 * args.decode_cost_scale, c2, c3)
    reqs = load(args.manifest)
    for layout in args.layouts.split(","):
        base = None
        for policy in args.policies.split(","):
            res = simulate(reqs, LAYOUTS[layout], policy, args.kv_scale)
            base = base or res["makespan_s"]
            print(f"{layout:2} {policy:15} makespan={res['makespan_s']:7.1f}s "
                  f"({(res['makespan_s'] / base - 1) * 100:+5.1f}%) preempt={res['preemptions']:4} "
                  f"replayed={res['replayed_tokens']:8} migrations={res['migrations']} "
                  f"overflows={res['overflows']} "
                  f"last={res['last']} batch={res['mean_batch']}",
                  flush=True)


if __name__ == "__main__":
    main()
