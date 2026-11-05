#!/usr/bin/env python3
"""AutoscaleBurst in-cluster load generator: router (admission queue + metrics sampler +
reactive scale-up controller) and open-loop trace client.

Single file, stdlib + aiohttp only, Python >= 3.10. Shipped into the vLLM pod by itself.

  router  --port 9000 --out-dir DIR --cap N [--controller JSON] ...
  client  --router URL --trace FILE --out FILE.jsonl [--start-at EPOCH] ...
  smoke   --router URL [--n 4 --in 128 --out 16]

All timestamps are epoch seconds (time.time()) of the process that writes them.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import os
import random
import signal
import sys
import time

import aiohttp
from aiohttp import web


def _raise_nofile():
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = hard if hard != resource.RLIM_INFINITY else 1 << 20
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception:  # noqa: BLE001
        pass


def _pct(vals, q):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


class JsonlWriter:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", buffering=1 << 16)

    def write(self, rec):
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def flush(self):
        try:
            self.f.flush()
        except ValueError:
            pass

    def close(self):
        self.flush()
        self.f.close()


# =============================================================================== router
class Item:
    __slots__ = ("raw", "rid", "prompt_len", "max_tokens", "arrived_at", "enq_at", "queue_s",
                 "attempts", "dispatched_at", "fut", "done", "last_error", "last_upstream_status",
                 "instance")

    def __init__(self, raw, body, now, fut):
        self.raw = raw
        self.rid = body.get("request_id")
        p = body.get("prompt")
        self.prompt_len = len(p) if isinstance(p, (list, str)) else None
        self.max_tokens = body.get("max_tokens")
        self.arrived_at = now
        self.enq_at = now
        self.queue_s = 0.0
        self.attempts = 0
        self.dispatched_at = None
        self.fut = fut
        self.done = False
        self.last_error = None
        self.last_upstream_status = None
        self.instance = None


class Instance:
    def __init__(self, name, url, state, cap, meta, now):
        self.name = name
        self.url = url.rstrip("/")
        self.state = state
        self.cap = cap
        self.meta = meta
        self.outstanding = 0
        self.served = 0
        self.out_tokens = 0
        self.errors = 0
        self.registered_at = now
        self.joined_at = now if state == "active" else None
        self.healthy_at = None
        self.left_at = None
        self.first_dispatch_at = None
        self.first_done_at = None
        self.last_dispatch_at = None
        self.poller = None
        self.metrics = None

    def to_json(self):
        return {k: getattr(self, k) for k in (
            "url", "state", "cap", "outstanding", "served", "out_tokens", "errors", "registered_at",
            "joined_at", "healthy_at", "left_at", "first_dispatch_at", "first_done_at",
            "last_dispatch_at", "meta", "metrics")}


METRIC_KEYS = {
    "vllm:num_requests_running": "num_running",
    "vllm:num_requests_waiting": "num_waiting",
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:gpu_cache_usage_perc": "kv_usage",
    "omp:inflight_requests": "omp_inflight",
}


def parse_prom(text):
    out = {"num_running": None, "num_waiting": None, "kv_usage": None}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = METRIC_KEYS.get(name)
        if key is None:
            continue
        try:
            v = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[key] = v if out.get(key) is None else out[key] + v
    return out


class Router:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out_dir, exist_ok=True)
        self.req_log = JsonlWriter(os.path.join(a.out_dir, "router_requests.jsonl"))
        self.sample_log = JsonlWriter(os.path.join(a.out_dir, "samples.jsonl"))
        self.event_log = JsonlWriter(os.path.join(a.out_dir, "events.jsonl"))
        self.dec_log = JsonlWriter(os.path.join(a.out_dir, "decisions.jsonl"))
        self.instances: dict[str, Instance] = {}
        self.queue: collections.deque[Item] = collections.deque()
        self.wake = asyncio.Event()
        self.c = {"arrived": 0, "dispatched": 0, "completed": 0, "failed": 0, "rejected_429": 0,
                  "retries": 0, "inflight": 0}
        self.ctl = {"mode": "off", "threshold": 0.0, "sustain_s": 10.0, "period_s": 1.0,
                    "max_triggers": 0, "lanes": []}
        if a.controller:
            self.ctl.update(json.loads(a.controller))
        self.ctl_samples: collections.deque = collections.deque()
        self.ctl_since = 0.0  # window restarts after each trigger / config change
        self.high_since = None  # start of the current run of samples > threshold
        self.triggers: list[dict] = []
        self.trig_event = asyncio.Event()
        self.closed_at = None  # set by POST /controller/close: no further triggers
        self.shutting_down = False
        self.stop = asyncio.Event()
        self.session: aiohttp.ClientSession | None = None
        self.admin_session: aiohttp.ClientSession | None = None
        self.tasks: list[asyncio.Task] = []
        self.started_at = time.time()

    # ------------------------------------------------------------------ logging
    def event(self, kind, **kw):
        rec = {"t_router": time.time(), "kind": kind, **kw}
        self.event_log.write(rec)
        return rec

    # ------------------------------------------------------------------ lifecycle
    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
            timeout=aiohttp.ClientTimeout(total=self.a.upstream_timeout_s))
        self.admin_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
            timeout=aiohttp.ClientTimeout(total=1.0))
        self.ctl_since = time.time()
        self.event("router_start", run_id=self.a.run_id, cap=self.a.cap, pid=os.getpid(),
                   args=vars(self.a), controller=dict(self.ctl))
        self.dec_log.write({"kind": "config", "t": time.time(), "config": dict(self.ctl)})
        for fn in (self._dispatcher, self._reaper, self._sampler, self._controller, self._flusher):
            self.tasks.append(asyncio.create_task(fn()))

    async def close(self):
        for t in self.tasks:
            t.cancel()
        for inst in self.instances.values():
            if inst.poller:
                inst.poller.cancel()
        self.event("router_stop", counters=dict(self.c), queue_depth=len(self.queue))
        for w in (self.req_log, self.sample_log, self.event_log, self.dec_log):
            w.close()
        await self.session.close()
        await self.admin_session.close()

    # ------------------------------------------------------------------ dispatch
    def _pick(self):
        best = None
        bkey = None
        for inst in self.instances.values():
            if inst.state != "active" or inst.outstanding >= inst.cap:
                continue
            key = (inst.outstanding, inst.last_dispatch_at or 0.0)
            if bkey is None or key < bkey:
                best, bkey = inst, key
        return best

    async def _dispatcher(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            while self.queue:
                inst = self._pick()
                if inst is None:
                    break
                item = self.queue.popleft()
                if item.done:
                    continue
                now = time.time()
                item.queue_s += now - item.enq_at
                item.attempts += 1
                item.dispatched_at = now
                item.instance = inst.name
                inst.outstanding += 1
                inst.last_dispatch_at = now
                self.c["dispatched"] += 1
                self.c["inflight"] += 1
                if inst.first_dispatch_at is None:
                    inst.first_dispatch_at = now
                    self.event("first_dispatch", name=inst.name, request_id=item.rid)
                asyncio.create_task(self._forward(item, inst))

    async def _forward(self, item, inst):
        status, data, err = None, None, None
        try:
            async with self.session.post(inst.url + "/v1/completions", data=item.raw,
                                         headers={"Content-Type": "application/json"}) as r:
                status = r.status
                data = await r.read()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
        finally:
            inst.outstanding -= 1
            self.c["inflight"] -= 1
            self.wake.set()
        item.last_upstream_status = status
        now = time.time()
        if status == 200:
            ctoks = None
            try:
                ctoks = json.loads(data)["usage"]["completion_tokens"]
            except Exception:  # noqa: BLE001
                pass
            inst.served += 1
            inst.out_tokens += int(ctoks or 0)
            if inst.first_done_at is None:
                inst.first_done_at = now
                self.event("first_completion", name=inst.name, request_id=item.rid)
            self._finish(item, 200, data, completion_tokens=ctoks)
            return
        if status is None or status >= 500:
            inst.errors += 1
            item.last_error = err or f"upstream {status}: {(data or b'')[:200].decode('utf-8', 'replace')}"
            if item.attempts < self.a.max_attempts and not item.done:
                self.c["retries"] += 1
                item.enq_at = now
                self.queue.appendleft(item)
                self.wake.set()
                return
            self._finish(item, 502, json.dumps({"error": "upstream failed", "detail": item.last_error,
                                                "attempts": item.attempts}).encode())
            return
        # 4xx etc: pass through
        item.last_error = (data or b"")[:200].decode("utf-8", "replace")
        self._finish(item, status, data or b"")

    def _finish(self, item, status, body, completion_tokens=None):
        if item.done:
            return
        item.done = True
        now = time.time()
        if status == 200:
            self.c["completed"] += 1
        elif status == 429:
            self.c["rejected_429"] += 1
        else:
            self.c["failed"] += 1
        self.req_log.write({
            "request_id": item.rid, "arrived_at": item.arrived_at, "dispatched_at": item.dispatched_at,
            "done_at": now, "instance": item.instance, "status": status, "attempts": item.attempts,
            "queue_s": item.queue_s, "upstream_status": item.last_upstream_status,
            "error": None if status == 200 else item.last_error, "prompt_len": item.prompt_len,
            "max_tokens": item.max_tokens, "completion_tokens": completion_tokens})
        if not item.fut.done():
            item.fut.set_result((status, body))

    def _headers(self, item):
        return {"x-asb-queue-ms": str(int(round(item.queue_s * 1000))),
                "x-asb-instance": item.instance or "",
                "x-asb-attempts": str(item.attempts),
                "x-asb-arrived-at": repr(item.arrived_at),
                "x-asb-dispatched-at": repr(item.dispatched_at) if item.dispatched_at else ""}

    async def _reaper(self):
        to = self.a.admission_timeout_s
        while True:
            await asyncio.sleep(0.25)
            now = time.time()
            if not self.queue:
                continue
            if all(now - it.arrived_at <= to for it in self.queue):
                continue
            keep = collections.deque()
            for it in self.queue:
                if now - it.arrived_at > to and not it.done:
                    it.queue_s += now - it.enq_at
                    it.last_error = "admission timeout"
                    self._finish(it, 429, json.dumps({"error": "admission timeout"}).encode())
                elif not it.done:
                    keep.append(it)
            self.queue = keep

    # ------------------------------------------------------------------ HTTP: data plane
    async def h_completions(self, req):
        if self.shutting_down:
            return web.json_response({"error": "router shutting down"}, status=503)
        raw = await req.read()
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"bad json: {e}"}, status=400)
        fut = asyncio.get_running_loop().create_future()
        item = Item(raw, body, time.time(), fut)
        self.c["arrived"] += 1
        self.queue.append(item)
        self.wake.set()
        status, data = await fut
        return web.Response(body=data, status=status, content_type="application/json",
                            headers=self._headers(item))

    # ------------------------------------------------------------------ HTTP: admin
    async def h_add_instance(self, req):
        b = await req.json()
        name, url, mode = b.get("name"), b.get("url"), b.get("mode", "join")
        if not name or not url or mode not in ("join", "pending"):
            return web.json_response({"error": "need name, url, mode in join|pending"}, status=400)
        old = self.instances.get(name)
        if old is not None and old.state != "left":
            return web.json_response({"error": f"instance {name} exists ({old.state})"}, status=409)
        cap = int(b.get("cap") or self.a.cap)
        now = time.time()
        inst = Instance(name, url, "active" if mode == "join" else "pending", cap, b.get("meta"), now)
        self.instances[name] = inst
        self.event("registered", name=name, url=url, mode=mode, cap=cap, meta=b.get("meta"))
        if mode == "join":
            self.event("joined", name=name, via="join")
            self.wake.set()
        else:
            inst.poller = asyncio.create_task(self._poll_health(inst))
        return web.json_response({"ok": True, "instance": inst.to_json()})

    async def _poll_health(self, inst):
        n = 0
        while inst.state == "pending":
            n += 1
            try:
                async with self.admin_session.get(inst.url + "/health",
                                                  timeout=aiohttp.ClientTimeout(total=1.0)) as r:
                    ok = r.status == 200
                    await r.read()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                ok = False
            if ok and inst.state == "pending":
                now = time.time()
                inst.healthy_at = now
                inst.joined_at = now
                inst.state = "active"
                self.event("healthy", name=inst.name, polls=n, since_register_s=now - inst.registered_at)
                self.event("joined", name=inst.name, via="pending")
                self.wake.set()
                return
            await asyncio.sleep(0.25)

    async def h_del_instance(self, req):
        name = req.match_info["name"]
        inst = self.instances.get(name)
        if inst is None:
            return web.json_response({"error": "unknown instance"}, status=404)
        prev = inst.state
        inst.state = "left"
        inst.left_at = time.time()
        if inst.poller:
            inst.poller.cancel()
        self.event("left", name=name, prev_state=prev, outstanding=inst.outstanding)
        return web.json_response({"ok": True, "instance": inst.to_json()})

    def _oldest_wait(self, now):
        return max((now - it.arrived_at for it in self.queue if not it.done), default=0.0)

    def state(self):
        now = time.time()
        return {"t": now, "run_id": self.a.run_id, "queue_depth": len(self.queue),
                "oldest_wait_s": self._oldest_wait(now),
                "instances": {n: i.to_json() for n, i in self.instances.items()},
                "counters": dict(self.c), "controller": dict(self.ctl), "controller_closed_at": self.closed_at,
                "triggers": self.triggers}

    async def h_state(self, req):
        return web.json_response(self.state())

    async def h_event(self, req):
        try:
            b = await req.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "bad json"}, status=400)
        rec = {"t_router": time.time(), **b}
        self.event_log.write(rec)
        self.event_log.flush()
        return web.json_response({"ok": True, "t_router": rec["t_router"]})

    async def h_shutdown(self, req):
        now_flag = req.query.get("now") in ("1", "true", "yes")
        self.shutting_down = True
        self.event("shutdown_requested", now=now_flag)
        asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(self._drain(now_flag)))
        return web.json_response({"ok": True, "queue_depth": len(self.queue), "inflight": self.c["inflight"]})

    async def _drain(self, now_flag):
        if not now_flag:
            while self.queue or self.c["inflight"] > 0:
                await asyncio.sleep(0.2)
        await asyncio.sleep(0.2)  # let the shutdown response go out
        self.stop.set()

    async def h_health(self, req):
        return web.Response(text="ok")

    # ------------------------------------------------------------------ HTTP: controller
    def _visible(self, include_dry):
        return [t for t in self.triggers if include_dry or t["actuate"]]

    async def h_triggers(self, req):
        after = int(req.query.get("after", 0))
        wait = float(req.query.get("wait", 0))
        dry = req.query.get("include_dry") in ("1", "true")
        deadline = time.time() + wait
        while True:
            res = [t for t in self._visible(dry) if t["k"] > after]
            rem = deadline - time.time()
            if res or rem <= 0:
                return web.json_response(res)
            if self.closed_at is not None:
                return web.json_response(res)
            ev = self.trig_event
            try:
                await asyncio.wait_for(ev.wait(), rem)
            except asyncio.TimeoutError:
                pass

    def _wake_waiters(self):
        ev, self.trig_event = self.trig_event, asyncio.Event()
        ev.set()

    async def h_close(self, req):
        if self.closed_at is None:
            self.closed_at = time.time()
            self.dec_log.write({"kind": "closed", "t": self.closed_at, "n_triggers": len(self.triggers)})
            self.dec_log.flush()
            self.event("controller_closed", n_triggers=len(self.triggers))
            self._wake_waiters()
        return web.json_response({"ok": True, "closed_at": self.closed_at,
                                  "fired": [t["k"] for t in self.triggers if t["actuate"]]})

    async def h_trigger_k(self, req):
        k = int(req.match_info["k"])
        wait = float(req.query.get("wait", 0))
        deadline = time.time() + wait
        while True:
            for t in self.triggers:
                if t["k"] == k and t["actuate"]:
                    return web.json_response(t)
            if self.closed_at is not None:
                return web.json_response({"error": "controller closed", "k": k,
                                          "closed_at": self.closed_at}, status=410)
            rem = deadline - time.time()
            if rem <= 0:
                return web.Response(status=204)
            ev = self.trig_event
            try:
                await asyncio.wait_for(ev.wait(), rem)
            except asyncio.TimeoutError:
                pass

    async def h_ctl_config(self, req):
        b = await req.json()
        new = {"mode": "off", "threshold": 0.0, "sustain_s": 10.0, "period_s": 1.0,
               "max_triggers": 0, "lanes": []}
        new.update(b)
        if new["mode"] not in ("off", "dry", "live"):
            return web.json_response({"error": "mode must be off|dry|live"}, status=400)
        self.ctl = new
        self.ctl_samples.clear()
        self.ctl_since = time.time()
        self.high_since = None
        self.dec_log.write({"kind": "config", "t": self.ctl_since, "config": dict(new)})
        self.dec_log.flush()
        self.event("controller_config", config=dict(new))
        return web.json_response({"ok": True, "config": new})

    # ------------------------------------------------------------------ loops
    async def _controller(self):
        nxt = time.time()
        while True:
            period = float(self.ctl.get("period_s", 1.0)) or 1.0
            nxt += period
            await asyncio.sleep(max(0.0, nxt - time.time()))
            if time.time() - nxt > period:  # fell behind (should not happen): resync
                nxt = time.time()
            cfg = self.ctl
            if cfg.get("mode", "off") == "off":
                continue
            now = time.time()
            qd = len(self.queue)
            sustain = float(cfg.get("sustain_s", 10.0))
            self.ctl_samples.append((now, qd))
            lo = now - sustain - 0.25 * period
            while self.ctl_samples and self.ctl_samples[0][0] < lo:
                self.ctl_samples.popleft()
            thr = float(cfg.get("threshold", 0.0))
            if qd > thr:
                if self.high_since is None:
                    self.high_since = now
            else:
                self.high_since = None
            k = len(self.triggers) + 1
            if k > int(cfg.get("max_triggers", 0)) or self.closed_at is not None:
                continue
            win = [s for s in self.ctl_samples if s[0] > self.ctl_since]
            need = max(1, int(math.floor(sustain / period + 1e-9)))
            # Rule: the queue must have been observed > threshold continuously for >= sustain_s
            # (run start high_since; reset by any sample <= threshold, by a config change, and
            # moved to the trigger time after a trigger), every sample in the last sustain_s after
            # the previous trigger must be > threshold, and there must be >= floor(sustain/period)
            # such samples. Strict: a late timer wakeup only delays the trigger by one period.
            sustained = self.high_since is not None and now - self.high_since >= sustain
            if sustained and len(win) >= need and all(q > thr for _, q in win):
                lanes = cfg.get("lanes") or []
                live = cfg["mode"] == "live"
                rec = {"kind": "trigger", "k": k, "t": now, "lane": lanes[k - 1] if k <= len(lanes) else None,
                       "queue_depth": qd, "window": [[round(t, 3), q] for t, q in win],
                       "mode": cfg["mode"], "would_trigger": True, "actuate": live,
                       "threshold": thr, "sustain_s": sustain, "period_s": period}
                rec["high_since"] = self.high_since
                self.triggers.append(rec)
                self.ctl_since = now
                self.high_since = now
                self.dec_log.write(rec)
                self.dec_log.flush()
                self.event("trigger", k=k, lane=rec["lane"], mode=cfg["mode"], actuate=live, queue_depth=qd)
                self._wake_waiters()

    async def _scrape(self, inst):
        try:
            async with self.admin_session.get(inst.url + "/metrics",
                                              timeout=aiohttp.ClientTimeout(total=0.8)) as r:
                if r.status != 200:
                    return None
                return parse_prom(await r.text())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return None

    async def _sampler(self):
        period = self.a.sample_period_s
        nxt = time.time()
        while True:
            nxt += period
            await asyncio.sleep(max(0.0, nxt - time.time()))
            if time.time() - nxt > period:
                nxt = time.time()
            now = time.time()
            snap = {"t": now, "queue_depth": len(self.queue), "oldest_wait_s": self._oldest_wait(now),
                    "active": sum(i.state == "active" for i in self.instances.values()),
                    "pending": sum(i.state == "pending" for i in self.instances.values()),
                    "arrived": self.c["arrived"], "dispatched": self.c["dispatched"],
                    "completed": self.c["completed"], "failed": self.c["failed"],
                    "rejected": self.c["rejected_429"], "retries": self.c["retries"],
                    "inflight": self.c["inflight"], "instances": {}}
            insts = list(self.instances.items())
            for n, i in insts:
                snap["instances"][n] = {"state": i.state, "outstanding": i.outstanding, "served": i.served,
                                        "out_tokens": i.out_tokens, "errors": i.errors, "metrics": None}
            scrape = [(n, i) for n, i in insts if i.state in ("active", "pending")]
            if scrape:
                res = await asyncio.gather(*(self._scrape(i) for _, i in scrape))
                for (n, i), m in zip(scrape, res):
                    i.metrics = m
                    snap["instances"][n]["metrics"] = m
            snap["scrape_s"] = time.time() - now
            self.sample_log.write(snap)

    async def _flusher(self):
        while True:
            await asyncio.sleep(1.0)
            for w in (self.req_log, self.sample_log, self.event_log, self.dec_log):
                w.flush()


async def run_router(a):
    _raise_nofile()
    r = Router(a)
    await r.start()
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_post("/v1/completions", r.h_completions)
    app.router.add_get("/health", r.h_health)
    app.router.add_post("/admin/instances", r.h_add_instance)
    app.router.add_delete("/admin/instances/{name}", r.h_del_instance)
    app.router.add_get("/admin/state", r.h_state)
    app.router.add_post("/admin/event", r.h_event)
    app.router.add_post("/admin/shutdown", r.h_shutdown)
    app.router.add_get("/controller/triggers", r.h_triggers)
    app.router.add_get("/controller/trigger/{k}", r.h_trigger_k)
    app.router.add_post("/controller/config", r.h_ctl_config)
    app.router.add_post("/controller/close", r.h_close)
    runner = web.AppRunner(app, access_log=None, handle_signals=False)
    await runner.setup()
    site = web.TCPSite(runner, a.host, a.port, backlog=4096)
    await site.start()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, r.stop.set)
        except NotImplementedError:
            pass
    print(json.dumps({"router": f"{a.host}:{a.port}", "out_dir": a.out_dir, "controller": r.ctl}), flush=True)
    await r.stop.wait()
    await r.close()
    await runner.cleanup()


# =============================================================================== client
def make_prompt(seed, i, n):
    rng = random.Random(seed * 1_000_003 + i)
    return [rng.randrange(1000, 200000) for _ in range(n)]


def _fhdr(h, k):
    v = h.get(k)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


async def one_request(session, url, model, run_id, seed, i, n_in, n_out, sched_t, due, timeout_s):
    body = {"model": model, "prompt": make_prompt(seed, i, n_in), "max_tokens": n_out,
            "ignore_eos": True, "temperature": 0.0, "request_id": f"{run_id}-{i}"}
    data = json.dumps(body, separators=(",", ":")).encode()
    send_at = time.time()
    rec = {"i": i, "run_id": run_id, "sched_t": sched_t, "send_at": send_at, "done_at": None,
           "lateness_s": (send_at - due) if due is not None else None, "status": None, "error": None,
           "in": n_in, "out": n_out, "prompt_tokens": None, "output_tokens": None, "instance": None,
           "attempts": None, "queue_s": None, "arrived_at": None, "dispatched_at": None,
           "engine_ttft_s": None, "engine_started_at_s": None, "engine_finished_at_s": None,
           "engine_elapsed_s": None, "ttft_s": None, "tpot_s": None, "e2e_s": None, "mismatch": None}
    try:
        async with session.post(url, data=data, headers={"Content-Type": "application/json"},
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            raw = await r.read()
            rec["status"] = r.status
            h = r.headers
            q = _fhdr(h, "x-asb-queue-ms")
            rec["queue_s"] = q / 1000.0 if q is not None else None
            rec["instance"] = h.get("x-asb-instance") or None
            a = _fhdr(h, "x-asb-attempts")
            rec["attempts"] = int(a) if a is not None else None
            rec["arrived_at"] = _fhdr(h, "x-asb-arrived-at")
            rec["dispatched_at"] = _fhdr(h, "x-asb-dispatched-at")
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        raw = None
    rec["done_at"] = time.time()
    rec["e2e_s"] = rec["done_at"] - send_at
    if rec["status"] == 200:
        try:
            j = json.loads(raw)
            u = j.get("usage") or {}
            rec["prompt_tokens"] = u.get("prompt_tokens")
            rec["output_tokens"] = u.get("completion_tokens")
            o = j.get("omp") or {}
            rec["engine_ttft_s"] = o.get("ttft_s")
            rec["engine_started_at_s"] = o.get("started_at_s")
            rec["engine_finished_at_s"] = o.get("finished_at_s")
            rec["engine_elapsed_s"] = o.get("elapsed_s")
            if rec["queue_s"] is not None and rec["engine_ttft_s"] is not None:
                rec["ttft_s"] = rec["queue_s"] + rec["engine_ttft_s"]
            ot = rec["output_tokens"]
            if (ot is not None and ot > 1 and rec["engine_finished_at_s"] is not None
                    and rec["engine_started_at_s"] is not None and rec["engine_ttft_s"] is not None):
                rec["tpot_s"] = (rec["engine_finished_at_s"] - rec["engine_started_at_s"]
                                 - rec["engine_ttft_s"]) / (ot - 1)
            mm = []
            if rec["prompt_tokens"] != n_in:
                mm.append(f"prompt_tokens {rec['prompt_tokens']} != in {n_in}")
            if rec["output_tokens"] != n_out:
                mm.append(f"completion_tokens {rec['output_tokens']} != out {n_out}")
            rec["mismatch"] = "; ".join(mm) or None
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"bad response: {type(e).__name__}: {e}"[:300]
    elif raw is not None:
        rec["error"] = raw[:300].decode("utf-8", "replace")
    return rec


def summarize(recs, t_start, t_end, run_id):
    ok = [r for r in recs if r["status"] == 200]
    errs = collections.Counter(str(r["status"]) for r in recs if r["status"] != 200)
    lat = [r["lateness_s"] for r in recs if r["lateness_s"] is not None]

    def pc(key, rs=ok):
        v = [r[key] for r in rs if r[key] is not None]
        return {"p50": _pct(v, 50), "p90": _pct(v, 90), "p99": _pct(v, 99),
                "mean": (sum(v) / len(v)) if v else None}
    return {"run_id": run_id, "n": len(recs), "ok": len(ok), "errors_by_status": dict(errs),
            "mismatch": sum(1 for r in recs if r.get("mismatch")),
            "ttft_s": pc("ttft_s"), "tpot_s": pc("tpot_s"), "queue_s": pc("queue_s"), "e2e_s": pc("e2e_s"),
            "duration_s": t_end - t_start, "start_at": t_start, "end_at": t_end,
            "max_lateness_s": max(lat) if lat else None, "p99_lateness_s": _pct(lat, 99),
            "out_tokens_ok": sum(r["output_tokens"] or 0 for r in ok),
            "by_instance": dict(collections.Counter(r["instance"] for r in ok))}


async def run_client(a):
    _raise_nofile()
    with open(a.trace) as f:
        tr = json.load(f)
    reqs = tr["requests"]
    if a.limit:
        reqs = reqs[:a.limit]
    url = a.router.rstrip("/") + "/v1/completions"
    start_at = a.start_at if a.start_at is not None else time.time() + 2.0
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    w = JsonlWriter(a.out)
    recs = []
    pending = set()

    async def flusher():
        while True:
            await asyncio.sleep(1.0)
            w.flush()

    async def fire(r, due, sched_t):
        rec = await one_request(session, url, a.model, a.run_id, a.seed, r["i"], int(r["in"]),
                                int(r["out"]), sched_t, due, a.timeout_s)
        recs.append(rec)
        w.write(rec)

    print(json.dumps({"client": a.run_id, "n": len(reqs), "start_at": start_at, "sequential": a.sequential,
                      "last_t": reqs[-1]["t"] * a.time_scale if reqs else 0}), flush=True)
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
                                    timeout=aiohttp.ClientTimeout(total=a.timeout_s))
    ft = asyncio.create_task(flusher())
    try:
        if a.sequential:
            # closed loop, concurrency 1: trace times ignored; sched_t = actual offset from start_at
            d = start_at - time.time()
            if d > 0:
                await asyncio.sleep(d)
            for r in reqs:
                await fire(r, None, time.time() - start_at)
        for r in ([] if a.sequential else reqs):
            due = start_at + float(r["t"]) * a.time_scale
            d = due - time.time()
            if d > 0:
                await asyncio.sleep(d)
            t = asyncio.create_task(fire(r, due, r["t"]))
            pending.add(t)
            t.add_done_callback(pending.discard)
        if pending:
            await asyncio.gather(*list(pending))
    finally:
        ft.cancel()
        await session.close()
        w.close()
    s = summarize(recs, start_at, time.time(), a.run_id)
    s["trace"] = os.path.abspath(a.trace)
    s["sequential"] = a.sequential
    s["trace_meta"] = {k: v for k, v in tr.get("meta", {}).items()
                       if k not in ("per_min_counts", "dropped_invalid_per_min")}
    with open(a.out + ".summary.json", "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s), flush=True)


async def run_smoke(a):
    url = a.router.rstrip("/") + "/v1/completions"
    async with aiohttp.ClientSession() as session:
        for i in range(a.n):
            rec = await one_request(session, url, a.model, a.run_id, a.seed, i, a.in_tokens, a.out_tokens,
                                    0.0, None, a.timeout_s)
            print(json.dumps(rec), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    q = sp.add_parser("router")
    q.add_argument("--host", default="0.0.0.0")
    q.add_argument("--port", type=int, default=9000)
    q.add_argument("--out-dir", required=True)
    q.add_argument("--cap", type=int, required=True, help="max outstanding requests per instance")
    q.add_argument("--admission-timeout-s", type=float, default=600.0)
    q.add_argument("--upstream-timeout-s", type=float, default=1800.0)
    q.add_argument("--max-attempts", type=int, default=3)
    q.add_argument("--sample-period-s", type=float, default=1.0)
    q.add_argument("--controller", default=None, help='JSON, e.g. {"mode":"live","threshold":8,...}')
    q.add_argument("--run-id", default="asb")

    q = sp.add_parser("client")
    q.add_argument("--router", required=True)
    q.add_argument("--trace", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--run-id", default="asb")
    q.add_argument("--start-at", type=float, default=None, help="epoch; default now+2")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--timeout-s", type=float, default=2700.0)
    q.add_argument("--model", default="model")
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--time-scale", type=float, default=1.0)
    q.add_argument("--sequential", action="store_true",
                   help="closed loop, concurrency 1 (calibration); ignores trace times, lateness_s=null")

    q = sp.add_parser("smoke")
    q.add_argument("--router", required=True)
    q.add_argument("--n", type=int, default=4)
    q.add_argument("--in", dest="in_tokens", type=int, default=128)
    q.add_argument("--out", dest="out_tokens", type=int, default=16)
    q.add_argument("--model", default="model")
    q.add_argument("--run-id", default="smoke")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--timeout-s", type=float, default=600.0)

    a = p.parse_args(argv)
    if sys.version_info < (3, 10):
        sys.exit("python >= 3.10 required")
    fn = {"router": run_router, "client": run_client, "smoke": run_smoke}[a.cmd]
    try:
        asyncio.run(fn(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
