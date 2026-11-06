#!/usr/bin/env python3
"""Stub engine implementing the AutoscaleBurst engine HTTP API (stdlib + aiohttp).

Capacity model: at most --slots requests run concurrently (others wait inside the
stub, counted as num_requests_waiting). A running request sleeps
in_tokens/--prefill-tps (TTFT, measured from engine start of the request) and then
--tpot-s per additional output token.
/health returns 503 until --start-healthy-after seconds after process start.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web


class Stub:
    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.sem = asyncio.Semaphore(a.slots)
        self.running = 0
        self.waiting = 0
        self.inflight = 0
        self.served = 0
        self.draining = False
        self.generation = a.generation

    def healthy(self):
        return (not self.draining) and time.time() - self.t0 >= self.a.start_healthy_after

    async def health(self, req):
        return web.Response(status=200 if self.healthy() else 503, text="ok" if self.healthy() else "not ready")

    async def metrics(self, req):
        n = self.a.name
        lay = self.a.layout
        lab = f'instance="{n}",layout="{lay}"'
        kv = min(1.0, self.running / max(1, self.a.slots))
        body = (
            f"vllm:num_requests_running{{{lab}}} {self.running}\n"
            f"vllm:num_requests_waiting{{{lab}}} {self.waiting}\n"
            f"vllm:kv_cache_usage_perc{{{lab}}} {kv:.4f}\n"
            f"omp:inflight_requests{{{lab}}} {self.inflight}\n"
            f"omp:served_requests{{{lab}}} {self.served}\n"
        )
        return web.Response(text=body, content_type="text/plain")

    async def completions(self, req):
        if not self.healthy():
            return web.json_response({"error": "not ready"}, status=503)
        try:
            body = await req.json()
            prompt = body["prompt"]
            if not isinstance(prompt, list):
                raise ValueError("prompt must be a token id array")
            max_tokens = int(body["max_tokens"])
            if max_tokens < 1:
                raise ValueError("max_tokens must be >= 1")
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": f"bad request: {e}"}, status=400)
        rid = str(body.get("request_id", ""))
        n_in = len(prompt)
        self.inflight += 1
        self.waiting += 1
        try:
            async with self.sem:
                self.waiting -= 1
                self.running += 1
                try:
                    started = time.time()
                    ttft = n_in / self.a.prefill_tps
                    await asyncio.sleep(ttft)
                    await asyncio.sleep(self.a.tpot_s * (max_tokens - 1))
                    finished = time.time()
                finally:
                    self.running -= 1
        finally:
            self.inflight -= 1
        self.served += 1
        toks = [(i * 7919 + 13) % 100000 + 1000 for i in range(max_tokens)]
        resp = {
            "id": f"cmpl-{rid}", "object": "text_completion", "model": body.get("model", "stub"),
            "choices": [{"index": 0, "text": " x" * max_tokens, "token_ids": toks, "finish_reason": "length"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": max_tokens, "total_tokens": n_in + max_tokens},
            "omp": {"instance_id": self.a.name, "generation": self.generation,
                    "ttft_s": ttft, "elapsed_s": finished - started,
                    "started_at_s": started, "finished_at_s": finished},
        }
        return web.json_response(resp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--name", default="stub")
    p.add_argument("--layout", default="stub")
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--prefill-tps", type=float, default=20000.0)
    p.add_argument("--tpot-s", type=float, default=0.02)
    p.add_argument("--start-healthy-after", type=float, default=0.0)
    a = p.parse_args()

    async def build():
        s = Stub(a)
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/completions", s.completions)
        app.router.add_get("/health", s.health)
        app.router.add_get("/metrics", s.metrics)
        return app

    print(json.dumps({"stub": a.name, "port": a.port, "slots": a.slots, "prefill_tps": a.prefill_tps,
                      "tpot_s": a.tpot_s, "start_healthy_after": a.start_healthy_after}), flush=True)
    web.run_app(build(), host=a.host, port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
