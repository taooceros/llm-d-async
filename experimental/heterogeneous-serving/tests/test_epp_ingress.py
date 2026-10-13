"""llm-d EPP ingress contract: engine surface, EPP arm configs and client accounting."""

import asyncio
import json
import pathlib
import unittest
from types import SimpleNamespace

import yaml
from fastapi.testclient import TestClient

from hetero.experiment import (EPP_CAPACITY_REFUSAL, EppIngress, _epp_placement_reset, _epp_record,
                               _post_epp)
from hetero.qualify_migration import _check_accounting
from hetero.serving_http import _render_metrics, build_app

ROOT = pathlib.Path(__file__).resolve().parents[1]


class Engine:
    instance_id, generation, layout_id = "large", 1, "M"
    settings = SimpleNamespace(model="gemma", max_model_len=36864)

    def __init__(self):
        self.calls = []

    def status(self):
        return {"state": "ready", "total_blocks": 282, "free_blocks": 200, "block_size": 256,
                "num_running": 3, "num_waiting": 1, "inflight": 4, "served": 9}

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"request_id": kwargs["request_id"], "instance_id": "large", "prompt_tokens": 3,
                "output_tokens": 2, "output_token_ids": [7, 8], "text": "ab", "finish_reason": "length",
                "ttft_s": 0.1, "elapsed_s": 0.2, "started_at_s": 10.0, "finished_at_s": 10.2}


class Response:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code, self._body, self.headers = status_code, body, headers or {}

    def json(self):
        return self._body


class EngineSurfaceTests(unittest.TestCase):
    def test_every_metric_the_epp_arms_read_is_exported(self):
        families = {line.split("{")[0] for line in _render_metrics(Engine()).splitlines()
                    if line and not line.startswith("#")}
        for path in sorted((ROOT / "deploy/epp").glob("*.yaml")):
            config = yaml.safe_load(path.read_text())
            extractor = next(p for p in config["plugins"] if p["type"] == "core-metrics-extractor")
            (engine,) = extractor["parameters"]["engineConfigs"]
            specs = [engine[key] for key in ("queuedRequestsSpec", "runningRequestsSpec", "kvUsageSpec",
                                             "cacheBlockSizeSpec", "cacheNumBlocksSpec")]
            specs += [metric["metricSpec"] for metric in engine["customMetrics"]]
            # The core extractor fails a poll whose configured family is absent.
            self.assertLessEqual(set(specs), families, path.name)

    def test_standard_token_array_prompt_and_models(self):
        engine = Engine()
        client = TestClient(build_app(engine))
        ok = client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 2,
                                                  "ignore_eos": True, "request_id": "r1"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual((engine.calls[0]["prompt"], engine.calls[0]["prompt_token_ids"]), (None, [1, 2, 3]))
        self.assertEqual(ok.json()["choices"][0]["token_ids"], [7, 8])
        for batch in ([[1, 2], [3]], ["a", "b"], []):
            self.assertEqual(client.post("/v1/completions", json={"prompt": batch}).status_code, 400)
        self.assertEqual(client.get("/v1/models").json()["data"][0]["max_model_len"], 36864)

    def test_client_disconnect_cancels_the_engine_generation(self):
        class Hanging(Engine):
            async def generate(self, **kwargs):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        async def scenario():
            engine = Hanging()
            engine.started, engine.cancelled = asyncio.Event(), False
            body = json.dumps({"prompt": [1, 2, 3], "max_tokens": 2, "request_id": "r1"}).encode()
            messages = [{"type": "http.request", "body": body, "more_body": False}]
            sent = []

            async def receive():
                if messages:
                    return messages.pop(0)
                await engine.started.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)

            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                     "scheme": "http", "path": "/v1/completions", "raw_path": b"/v1/completions",
                     "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
                     "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8000)}
            await asyncio.wait_for(build_app(engine)(scope, receive, send), timeout=5)
            return engine.cancelled, sent[0]["status"]

        self.assertEqual(asyncio.run(scenario()), (True, 499))



class EppClientTests(unittest.TestCase):
    ingress = EppIngress(url="http://router", model="gemma",
                         endpoints={"10.0.0.1:8000": "small_a", "10.0.0.3:8000": "large"})

    def response(self, *, upstream="10.0.0.3:8000", selected="large"):
        item = self.ingress.payload("r1", [1, 2, 3], 2)
        body = {"id": "r1", "choices": [{"token_ids": [7, 8], "finish_reason": "length"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "omp": {"instance_id": "large", "generation": 1, "ttft_s": 0.1,
                        "started_at_s": 10.0, "finished_at_s": 10.2}}
        headers = {"x-omp-gateway-protocol": "llm-d-epp-ext-proc-v3", "x-omp-upstream-host": upstream,
                   "x-omp-route-trace": json.dumps({"selected": selected, "bound": True}),
                   "x-omp-epp-queue-ms": "1250"}
        return item, Response(200, body, headers)

    def test_engine_response_is_one_exactly_accounted_segment(self):
        item, response = self.response()
        record = _epp_record(item, response, self.ingress)
        self.assertEqual((record["initial_instance"], record["epp_queue_s"]), ("large", 1.25))
        accounting = _check_accounting([record], {"r1": 2})
        self.assertTrue(accounting["all_budgets_exact"] and accounting["no_duplicate_delivery"])

    def test_destination_other_than_the_serving_engine_is_refused(self):
        for mismatch in ({"upstream": "10.0.0.1:8000"}, {"selected": "small_a"}):
            item, response = self.response(**mismatch)
            with self.assertRaises(ValueError):
                _epp_record(item, response, self.ingress)

    def test_only_capacity_refusals_wait_for_room_and_every_resend_is_counted(self):
        refusal = Response(429, headers={"x-omp-route-trace": json.dumps({"refusal": EPP_CAPACITY_REFUSAL})})

        class Client:
            def __init__(self, responses):
                self.responses = list(responses)

            async def post(self, path, content, headers):
                return self.responses.pop(0)

        def post(responses):
            stats = {"capacity_retries": 0, "retry_request_bytes": 0}
            response = asyncio.run(_post_epp(Client(responses), b"0123456789", stats))
            return response, stats["capacity_retries"], stats["retry_request_bytes"]

        response, retries, resent = post([refusal, refusal, Response(200)])
        self.assertEqual((response.status_code, retries, resent), (200, 2, 20))
        other = Response(429, headers={"x-omp-route-trace": json.dumps({"refusal": "flow_control_full"})})
        self.assertEqual(post([other, Response(200)]), (other, 0, 0))

    def test_placement_reset_waits_out_live_reservations_and_never_forces(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        idle = {"fresh_dispatched": 0, "fresh_to_large": 0, "reservations": 0, "leases": 0,
                "projection": {"large": {"blocks": 0, "requests": 0}}}
        seen = []

        class Admin(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                seen.append(self.path)
                live = {**idle, "reservations": 1, "fresh_dispatched": 7}
                if len(seen) == 1:
                    self.reply(409, {"reset": False, "error": "requests still hold reservations", "before": live})
                else:
                    self.reply(200, {"reset": True, "before": {**idle, "fresh_dispatched": 7}})

            def do_GET(self):
                self.reply(200, idle)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Admin)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            result = _epp_placement_reset(url, timeout_s=5.0, poll_s=0.01)
        finally:
            server.shutdown()
        self.assertEqual((result["attempts"], result["before"]["fresh_dispatched"]), (2, 7))
        self.assertEqual(seen, ["/hetero/reset", "/hetero/reset"])


if __name__ == "__main__":
    unittest.main()
