"""One engine boot inside a Ray job: `vllm serve` (tpu-inference on the GPU), its log lines
prefixed with epoch seconds, readiness, then the CUDA campaign's greedy output check.

    python -m hetero_jax_gpu.serve MODEL [--keep] [--pidfile F] [-- extra vllm args]

Prints "MARK <event> <epoch>" (engine_start, ready, check_done | engine_died; engine_stop, engine_exited) and
"CHECK <response json>", like scripts/gpu_autoscale/pod_run.sh; engine settings match it.
Exit 0 only if the check request succeeded. --keep leaves the server running; with --pidfile, SIGTERM to this
process stops the engine (a same-host restart retires the old instance this way).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ENGINE_ARGS = ["--served-model-name", "gemma", "--max-model-len", "8192", "--max-num-seqs", "512",
               "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.92",
               "--limit-mm-per-prompt", '{"image": 0, "audio": 0, "video": 0}', "--seed", "0",
               "--disable-uvicorn-access-log"]
CHECK = {"model": "gemma", "max_tokens": 32, "temperature": 0,
         "messages": [{"role": "user", "content": "Name the three primary colors in one short sentence."}]}


def mark(event: str) -> None:
    print(f"MARK {event} {time.time():.6f}", flush=True)


def pump(stream) -> None:
    for line in stream:
        sys.stdout.write(f"{time.time():.6f} {line}")
        sys.stdout.flush()


def http(url: str, body: dict | None = None, timeout: float = 600) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pidfile", help="write this process's pid here; SIGTERM stops the engine and exits 0")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    cmd = ["vllm", "serve", args.model, "--port", str(args.port), *ENGINE_ARGS, *extra]
    print("CMD " + json.dumps(cmd), flush=True)
    mark("engine_start")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=pump, args=(proc.stdout,), daemon=True).start()

    stopping = threading.Event()

    def stop(signum, frame):  # a same-host restart (warm-*) retires this engine
        # Only signal here: the main thread may sit in proc.wait() holding Popen's waitpid lock, so waiting in the
        # handler deadlocks (seen 2026-10-23). The main thread's wait returns once vllm exits; a timer escalates.
        if not stopping.is_set():
            stopping.set()
            mark("engine_stop")
            proc.terminate()
            timer = threading.Timer(120, proc.kill)
            timer.daemon = True
            timer.start()

    if args.pidfile:
        signal.signal(signal.SIGTERM, stop)
        with open(args.pidfile + ".tmp", "w") as f:
            f.write(str(os.getpid()))
        os.replace(args.pidfile + ".tmp", args.pidfile)

    base = f"http://127.0.0.1:{args.port}"
    while True:
        if proc.poll() is not None:
            mark("engine_died")
            return 1
        try:
            http(base + "/health", timeout=2)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(0.1)
    mark("ready")
    ok = True
    try:
        check = http(base + "/v1/chat/completions", CHECK).decode()
    except Exception as e:  # report and fail the run
        check, ok = json.dumps({"error": repr(e)}), False
    print("CHECK " + check, flush=True)
    mark("check_done")
    if args.keep:
        proc.wait()
        mark("engine_exited")
        return 0 if stopping.is_set() and ok else 1
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
