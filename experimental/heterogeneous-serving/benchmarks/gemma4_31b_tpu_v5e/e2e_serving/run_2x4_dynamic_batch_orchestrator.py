#!/usr/bin/env python3
"""
vLLM Decoupled Prefill & Pure Decode Benchmark Orchestrator on Cloud TPU v5e
============================================================================
Reverse control flow orchestrator natively integrated with the Oh My Pi (OMP) harness.
Executes real-weights vLLM microbenchmark pipeline for Gemma-4 31B:
  - Stage 0: DISCOVER_TOPOLOGY (Discovers contiguous mesh & creates placement group)
  - Stage 1: VLLM_ISOLATED_PREFILL (Measures pure prefill throughput with max_tokens=1)
  - Stage 2: VLLM_PURE_DECODE (Measures autoregressive generation & isolates pure decode TPOT)
  - Stage 3: AGGREGATE_AND_REPORT (Computes stats, saves JSON, renders Markdown/HTML reports)

Communicates with OMP harness via structured events:
  [OMP_EVENT: READY hosts=[...] pg=...]
  [OMP_EVENT: STAGE_STARTED stage=...]
  [OMP_EVENT: STAGE_COMPLETED stage=... elapsed=...s]
  [OMP_EVENT: INTERVENTION_REQUIRED stage=... reason=... diag=... resume=...]
  [OMP_EVENT: BENCHMARK_SUCCESS json=...]
"""

import os
import sys
import time
import json
import random
import signal
import socket
import argparse
import threading
import traceback
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Ensure project root is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BENCHMARK_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pkg_util.tpu_topology import TPUTopologyDiscovery, ConnectedPair, TPUHostInfo

# Model architecture constants (Gemma-4 31B)
MODEL_NAME = "gemma-4-31b"
MODEL_PATH = "/models/gemma-4-31b"
HIDDEN_SIZE = 5376
INTERMEDIATE_SIZE = 21504
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
NUM_LAYERS = 48
PEAK_TFLOPS_TP8 = 1576.0   # 8 chips * 197 TFLOPS/chip (v5e BF16)
PEAK_TFLOPS_TP16 = 3152.0  # 16 chips * 197 TFLOPS/chip
FLOPS_PER_TOKEN = 39.636e9 # Theoretical FLOPs per token for Gemma-4 31B (48 layers)

STAGE_ORDER = [
    "DISCOVER_TOPOLOGY",
    "VLLM_ISOLATED_PREFILL",
    "VLLM_PURE_DECODE",
    "AGGREGATE_AND_REPORT",
]

DIAGNOSTIC_FILE = "/tmp/tpu_2x4_error_state.json"
DEFAULT_DATASET_PATHS = [
    os.path.join(BENCHMARK_DIR, "data", "sharegpt_sampled_500.json"),
    "/tmp/sharegpt_sampled_500.json",
    "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json",
]


def emit_omp_event(event_type: str, **kwargs):
    """Emits structured OMP lifecycle event signal on stdout."""
    payload_items = []
    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            payload_items.append(f"{k}={json.dumps(v, separators=(',', ':'))}")
        elif isinstance(v, str) and (" " in v or "=" in v):
            payload_items.append(f'{k}="{v}"')
        else:
            payload_items.append(f"{k}={v}")
    payload_str = " ".join(payload_items)
    print(f"\n[OMP_EVENT: {event_type} {payload_str}]", flush=True)


class ProgressHeartbeat:
    """Emits periodic heartbeat messages every N seconds during long stages."""
    def __init__(self, stage_name: str, interval: float = 3.0):
        self.stage_name = stage_name
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time = 0.0
        self.current_step = ""

    def update_step(self, step_info: str):
        self.current_step = step_info

    def _run(self):
        while self.running:
            time.sleep(self.interval)
            if not self.running:
                break
            elapsed = time.perf_counter() - self.start_time
            step_desc = f" - {self.current_step}" if self.current_step else ""
            print(f"[HEARTBEAT] Stage: {self.stage_name} | Elapsed: {elapsed:.1f}s{step_desc}", flush=True)

    def start(self):
        self.running = True
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


def load_sharegpt_prompts(dataset_path: str = None, max_prompts: int = 500) -> List[str]:
    """Loads human prompt strings from ShareGPT dataset."""
    candidate_paths = [dataset_path] if dataset_path else []
    candidate_paths.extend(DEFAULT_DATASET_PATHS)

    found_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            found_path = p
            break

    if not found_path:
        raise FileNotFoundError(f"ShareGPT dataset not found in any candidate path: {candidate_paths}")

    print(f"Loading ShareGPT dataset from {found_path}...", flush=True)
    with open(found_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if isinstance(item, dict) and "prompt" in item:
            prompts.append(item["prompt"])
        elif isinstance(item, dict) and "conversations" in item:
            convs = item.get("conversations", [])
            if len(convs) >= 2 and convs[0].get("from") == "human":
                text = convs[0].get("value", "").strip()
                if len(text) > 100:
                    prompts.append(text)
        if len(prompts) >= max_prompts * 2:
            break

    random.seed(42)
    random.shuffle(prompts)
    selected = prompts[:max_prompts]
    print(f"Loaded and filtered {len(selected)} realistic ShareGPT prompts.", flush=True)
    return selected


def get_prompts_for_batch(
    bs: int,
    it: int,
    stage: str,
    fixed_prompt_len: int = None,
    all_prompts: list = None,
    cursor: int = 0
):
    """Generates synthetic fixed-length tokens or slices ShareGPT text prompts."""
    if fixed_prompt_len is not None and fixed_prompt_len > 0:
        stage_offset = 10000 if stage == "gen" else 0
        rng = random.Random(1337 + stage_offset + it * 1000 + bs * 37)
        prompts = []
        for _ in range(bs):
            token_ids = [rng.randint(100, 250000) for _ in range(fixed_prompt_len)]
            try:
                from vllm.inputs import TokensPrompt
                prompts.append(TokensPrompt(prompt_token_ids=token_ids))
            except (ImportError, Exception):
                prompts.append({"prompt_token_ids": token_ids})
        return prompts, cursor
    else:
        prompts = all_prompts[cursor : cursor + bs]
        cursor += bs
        if cursor + 256 >= len(all_prompts):
            cursor = 1
        return prompts, cursor


# =========================================================================
# Ray Remote vLLM Engine Worker
# =========================================================================

import ray
from ray.util.placement_group import (
    placement_group,
    remove_placement_group,
    PlacementGroup
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=0)
class VLLMMicrobenchmarkWorker:
    """Manages real vLLM engine instance on assigned TPU placement group."""
    def __init__(self, model_path: str, tp_size: int, max_num_seqs: int = 256, target_ips: list = None, host_bounds: str = "2,1,1"):
        os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
        os.environ["TPU_CHIPS_PER_HOST"] = "4"
        os.environ["SCAN_TPU_CHIPS"] = "True"
        os.environ["TPU_ACCELERATOR_TYPE"] = "v5e"
        os.environ["JAX_PLATFORMS"] = ""
        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/models/jax_cache"

        if tp_size == 8 and target_ips:
            os.environ["TPU_TOPOLOGY"] = "2x4"
            os.environ["TPU_HOST_BOUNDS"] = host_bounds
            os.environ["TPU_PROCESS_BOUNDS"] = host_bounds
            os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
            os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "2,2,1"
            os.environ["TPU_PROCESS_ADDRESSES"] = f"{target_ips[0]}:8471,{target_ips[1]}:8471"
            os.environ["TPU_PROCESS_PORT"] = "8471"
            print(f"[Worker] Configured TPU Subslice Environment: TOPOLOGY=2x4, BOUNDS={host_bounds}, ADDRS={os.environ['TPU_PROCESS_ADDRESSES']}", flush=True)
        else:
            os.environ["TPU_TOPOLOGY"] = "4x4"

        from vllm import LLM

        print("\n[Worker] Initializing vLLM Engine on Cloud TPU v5e...", flush=True)
        t0 = time.perf_counter()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            distributed_executor_backend="ray",
            max_model_len=2048,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        print(f"[Worker] Model weights loaded & Engine initialized in {time.perf_counter() - t0:.2f}s.\n", flush=True)

    def warmup(self, sample_prompt=None):
        from vllm import SamplingParams
        print("[Worker] Warming up JIT compilation passes...", flush=True)
        sp_warmup = SamplingParams(max_tokens=4, temperature=0.7, ignore_eos=False)
        if sample_prompt is None:
            sample_prompt = "Hello, warming up Gemma-4 31B on Cloud TPU v5e."
        out = self.llm.generate([sample_prompt], sp_warmup, use_tqdm=False)
        sample_text = out[0].outputs[0].text.strip()[:100] if out and out[0].outputs else ""
        print(f"  Warmup Output Sample: \"{sample_text}...\"", flush=True)
        print("[Worker] Warmup complete.\n", flush=True)
        return sample_text

    def run_prefill_iteration(self, prompts: list) -> dict:
        from vllm import SamplingParams
        sp_prefill = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, detokenize=False)
        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, sp_prefill, use_tqdm=False)
        t_elapsed = time.perf_counter() - t0

        total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        lat_ms = t_elapsed * 1000.0
        tput = total_prompt_tokens / t_elapsed
        return {
            "total_prompt_tokens": total_prompt_tokens,
            "latency_ms": round(lat_ms, 2),
            "throughput_tok_s": round(tput, 1),
        }

    def run_generation_iteration(self, prompts: list, output_len: int = 64) -> dict:
        from vllm import SamplingParams
        sp_gen = SamplingParams(max_tokens=output_len, temperature=0.7, top_p=0.9, ignore_eos=False)
        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, sp_gen, use_tqdm=False)
        t_elapsed = time.perf_counter() - t0

        total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        actual_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_ms = t_elapsed * 1000.0
        sample_resp = outputs[0].outputs[0].text.strip()[:100] if outputs and outputs[0].outputs else ""
        sample_p = prompts[0]
        if isinstance(sample_p, str):
            p_str = sample_p[:150]
        elif hasattr(sample_p, "prompt_token_ids"):
            p_str = f"<TokensPrompt: {len(sample_p.prompt_token_ids)} tokens>"
        elif isinstance(sample_p, dict) and "prompt_token_ids" in sample_p:
            p_str = f"<TokensPrompt: {len(sample_p['prompt_token_ids'])} tokens>"
        else:
            p_str = str(sample_p)[:150]
        return {
            "prompt_tokens": total_prompt_tokens,
            "actual_gen_tokens": actual_gen_tokens,
            "total_time_ms": round(total_ms, 2),
            "sample_prompt": p_str,
            "sample_response": sample_resp,
        }

    def shutdown(self):
        try:
            del self.llm
        except Exception:
            pass
        return True


# =========================================================================
# Main Orchestrator Class
# =========================================================================

class TPU2x4DynamicBatchOrchestrator:
    """Manages the lifecycle of vLLM decoupled microbenchmarks via reverse control."""
    def __init__(self, args):
        self.args = args
        self.ray = None
        self.discovery = None
        self.connected_pair: Optional[ConnectedPair] = None
        self.placement_group: Optional[PlacementGroup] = None
        self.worker = None
        self.all_prompts = None
        self.prompt_cursor = 0
        self.prefill_latencies = {}  # bs -> est_prefill_ms

        self.benchmark_data: Dict[str, Any] = {
            "model": MODEL_NAME,
            "model_path": args.model_path,
            "architecture": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_q_heads": NUM_Q_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "num_layers": NUM_LAYERS,
                "weight_bytes_bf16": 61.44e9,
                "kv_bytes_per_token": 393216
            },
            "vllm_benchmark": {
                "hardware": f"Cloud TPU v5e (TP={args.tp_size})",
                "tp_size": args.tp_size,
                "batch_sizes": args.batch_sizes,
                "output_len": args.output_len,
                "num_iters": args.num_iters,
                "fixed_prompt_len": args.fixed_prompt_len,
                "peak_tflops": PEAK_TFLOPS_TP8 if args.tp_size == 8 else PEAK_TFLOPS_TP16,
            }
        }

    def trap_error_and_pause(self, stage_name: str, reason: str, exception: Exception, resume_arg: str):
        """Captures diagnostic state and halts for operator or automated steering."""
        stack = traceback.format_exc()
        diag_state = {
            "timestamp": time.time(),
            "stage": stage_name,
            "reason": reason,
            "exception_str": str(exception),
            "stack_trace": stack,
            "args": vars(self.args),
            "connected_pair": self.connected_pair.to_dict() if self.connected_pair else None,
            "pg_id": self.placement_group.id.hex() if self.placement_group else None,
        }

        try:
            with open(DIAGNOSTIC_FILE, "w", encoding="utf-8") as f:
                json.dump(diag_state, f, indent=2)
            print(f"\n[DIAGNOSTICS] Diagnostic state written to {DIAGNOSTIC_FILE}", flush=True)
        except Exception as write_err:
            print(f"[DIAGNOSTICS] Failed to write diagnostics file: {write_err}", flush=True)

        emit_omp_event(
            "INTERVENTION_REQUIRED",
            stage=stage_name,
            reason=reason,
            diag=DIAGNOSTIC_FILE,
            resume=f'--resume-stage {resume_arg}'
        )

        if sys.stdin.isatty() or os.environ.get("OMP_INTERACTIVE") == "1":
            print("\nWaiting for operator steering command on stdin (retry / switch_pair / skip_stage / abort)...", flush=True)
            try:
                line = sys.stdin.readline().strip()
                if line == "retry":
                    print("[OPERATOR] Retrying stage...", flush=True)
                    return "retry"
                elif line == "skip_stage":
                    print("[OPERATOR] Skipping current stage...", flush=True)
                    return "skip_stage"
                elif line in ("abort", "exit"):
                    print("[OPERATOR] Aborting benchmark.", flush=True)
                    sys.exit(42)
            except Exception as stdin_err:
                print(f"Stdin read error: {stdin_err}", flush=True)

        print(f"\n[FATAL] Unrecoverable fault in stage {stage_name}. Exiting with code 42.", flush=True)
        sys.exit(42)

    def stage_0_discover_topology(self) -> bool:
        """Stage 0: Discover TPU Mesh, create Placement Group, & Initialize vLLM Worker."""
        stage_name = "DISCOVER_TOPOLOGY"
        emit_omp_event("STAGE_STARTED", stage=stage_name)
        t0 = time.perf_counter()
        heartbeat = ProgressHeartbeat(stage_name)
        heartbeat.start()

        if self.args.force_invalid_pair:
            heartbeat.stop()
            err = ValueError("Intentionally forced invalid host pair for diagnostic testing.")
            self.trap_error_and_pause(stage_name, "FORCED_INVALID_PAIR", err, "DISCOVER_TOPOLOGY")
            return False

        try:
            heartbeat.update_step("Connecting to Ray cluster and discovering TPU mesh")
            if not ray.is_initialized():
                ray.init(address=self.args.ray_address or "auto", ignore_reinit_error=True)

            self.discovery = TPUTopologyDiscovery(ray_address=self.args.ray_address)
            slices = self.discovery.discover_tpu_nodes()

            target_ips = []
            host_bounds = "2,1,1"

            if self.args.tp_size == 8:
                if not slices:
                    raise RuntimeError("No active TPU worker nodes discovered in Ray cluster.")
                target_slice = list(slices.keys())[0]
                hosts = slices[target_slice]
                pairs = self.discovery.find_connected_pairs(hosts)
                if not pairs:
                    raise RuntimeError(f"No connected adjacent host pairs found in slice {target_slice}.")

                self.connected_pair = pairs[0]
                host_a = self.connected_pair.host_a
                host_b = self.connected_pair.host_b
                target_ips = [host_a.node_ip, host_b.node_ip]

                print(f"\nSelected Connected Subslice (TP=8):", flush=True)
                print(f"  - Host A: Worker {host_a.worker_id} ({host_a.node_ip})", flush=True)
                print(f"  - Host B: Worker {host_b.worker_id} ({host_b.node_ip})", flush=True)
                print(f"  - Mesh Dim: {self.connected_pair.mesh_dimension}\n", flush=True)

                bundle_specs = [
                    {"TPU": host_a.tpu_count, f"node:{host_a.node_ip}": 0.001},
                    {"TPU": host_b.tpu_count, f"node:{host_b.node_ip}": 0.001},
                ]
            else:
                bundle_specs = [{"TPU": 4.0} for _ in range(self.args.tp_size // 4)]

            heartbeat.update_step("Creating Ray Placement Group (STRICT_SPREAD)")
            print(f"Creating Ray Placement Group with {len(bundle_specs)} bundles...", flush=True)
            self.placement_group = placement_group(bundle_specs, strategy="STRICT_SPREAD")
            ray.get(self.placement_group.ready(), timeout=self.args.timeout_per_stage)
            pg_id = self.placement_group.id.hex()
            print(f"Placement Group is READY! (ID: {pg_id})\n", flush=True)

            emit_omp_event(
                "READY",
                hosts=target_ips if target_ips else ["all_cluster_hosts"],
                pg=f"pg_{pg_id[:8]}"
            )

            heartbeat.update_step("Instantiating VLLMMicrobenchmarkWorker and loading model weights")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.placement_group,
                placement_group_bundle_index=0,
            )

            max_batch = max(self.args.batch_sizes) if self.args.batch_sizes else 256
            self.worker = VLLMMicrobenchmarkWorker.options(
                scheduling_strategy=scheduling_strategy
            ).remote(
                model_path=self.args.model_path,
                tp_size=self.args.tp_size,
                max_num_seqs=max_batch,
                target_ips=target_ips,
                host_bounds=host_bounds
            )

            # Load Prompts or initialize synthetic mode
            if self.args.fixed_prompt_len is not None and self.args.fixed_prompt_len > 0:
                self.all_prompts = None
                dataset_label = f"Synthetic Randomized Fixed-Length ({self.args.fixed_prompt_len} tokens/prompt)"
            else:
                self.all_prompts = load_sharegpt_prompts(self.args.dataset_path, max_prompts=1000)
                dataset_label = f"ShareGPT ({len(self.all_prompts)} prompts available)"

            print(f"Dataset Configuration: {dataset_label}", flush=True)

            # Run engine warmup pass
            warmup_sample = self.all_prompts[0] if self.all_prompts else None
            _ = ray.get(self.worker.warmup.remote(warmup_sample))

            heartbeat.stop()
            emit_omp_event("STAGE_COMPLETED", stage=stage_name, elapsed=round(time.perf_counter() - t0, 2))
            return True

        except Exception as e:
            heartbeat.stop()
            action = self.trap_error_and_pause(stage_name, "DISCOVERY_OR_INIT_FAILED", e, "DISCOVER_TOPOLOGY")
            if action == "retry":
                return self.stage_0_discover_topology()
            return False

    def stage_1_vllm_isolated_prefill(self) -> bool:
        """Stage 1: Isolated Prefill Benchmark (max_tokens = 1, unique prompts)."""
        stage_name = "VLLM_ISOLATED_PREFILL"
        emit_omp_event("STAGE_STARTED", stage=stage_name)
        t0 = time.perf_counter()
        heartbeat = ProgressHeartbeat(stage_name)
        heartbeat.start()

        dataset_name = f"FIXED {self.args.fixed_prompt_len}-TOK" if self.args.fixed_prompt_len else "SHAREGPT"
        print("=" * 115, flush=True)
        print(f"  STAGE 1: ISOLATED PREFILL BENCHMARK ON {dataset_name} (max_tokens = 1, unique prompts per iter)", flush=True)
        print("=" * 115, flush=True)

        stage1_results = []

        try:
            for bs in self.args.batch_sizes:
                heartbeat.update_step(f"Running Prefill for Batch Size {bs}")
                print(f"\n--- Batch Size: {bs} (Prefill Only: max_tokens=1) ---", flush=True)
                iter_latencies = []
                iter_throughputs = []
                iter_tokens = []

                for it in range(self.args.num_iters):
                    prompts, self.prompt_cursor = get_prompts_for_batch(
                        bs=bs,
                        it=it,
                        stage="prefill",
                        fixed_prompt_len=self.args.fixed_prompt_len,
                        all_prompts=self.all_prompts,
                        cursor=self.prompt_cursor
                    )
                    res = ray.get(self.worker.run_prefill_iteration.remote(prompts))
                    iter_latencies.append(res["latency_ms"])
                    iter_throughputs.append(res["throughput_tok_s"])
                    iter_tokens.append(res["total_prompt_tokens"])
                    print(f"  Iter {it+1}/{self.args.num_iters}: {res['total_prompt_tokens']} prompt tok | Latency: {res['latency_ms']:.2f} ms | Throughput: {res['throughput_tok_s']:.1f} tok/s", flush=True)

                med_lat = float(np.median(iter_latencies))
                med_tput = float(np.median(iter_throughputs))
                mean_lat = float(np.mean(iter_latencies))
                std_lat = float(np.std(iter_latencies))
                cv_lat = float((std_lat / mean_lat) * 100.0) if mean_lat > 0 else 0.0

                self.prefill_latencies[bs] = med_lat
                row = {
                    "batch_size": bs,
                    "total_prompt_tokens": int(np.median(iter_tokens)),
                    "latency_ms": round(med_lat, 2),
                    "throughput_tok_s": round(med_tput, 1),
                    "mean_latency_ms": round(mean_lat, 2),
                    "std_latency_ms": round(std_lat, 2),
                    "cv_latency_percent": round(cv_lat, 2),
                    "iter_latencies_ms": iter_latencies,
                    "iter_throughputs_tok_s": iter_throughputs
                }
                stage1_results.append(row)
                print(f"  >> MEDIAN: Latency = {med_lat:.2f} ms | Throughput = {med_tput:.1f} tok/s | CV = {cv_lat:.1f}%", flush=True)

            self.benchmark_data["vllm_benchmark"]["stage_1_prefill"] = stage1_results
            heartbeat.stop()
            emit_omp_event("STAGE_COMPLETED", stage=stage_name, elapsed=round(time.perf_counter() - t0, 2))
            return True

        except Exception as e:
            heartbeat.stop()
            action = self.trap_error_and_pause(stage_name, "PREFILL_FAILED", e, "VLLM_ISOLATED_PREFILL")
            if action == "retry":
                return self.stage_1_vllm_isolated_prefill()
            return False

    def stage_2_vllm_pure_decode(self) -> bool:
        """Stage 2: End-to-End Generation & Pure Decode Isolation (max_tokens = output_len)."""
        stage_name = "VLLM_PURE_DECODE"
        emit_omp_event("STAGE_STARTED", stage=stage_name)
        t0 = time.perf_counter()
        heartbeat = ProgressHeartbeat(stage_name)
        heartbeat.start()

        dataset_name = f"FIXED {self.args.fixed_prompt_len}-TOK" if self.args.fixed_prompt_len else "SHAREGPT"
        print("\n" + "=" * 115, flush=True)
        print(f"  STAGE 2: END-TO-END {dataset_name} GENERATION & PURE DECODE (max_tokens = {self.args.output_len})", flush=True)
        print("=" * 115, flush=True)

        stage2_results = []

        try:
            for bs in self.args.batch_sizes:
                heartbeat.update_step(f"Running Decode Generation for Batch Size {bs}")
                print(f"\n--- Batch Size: {bs} (E2E Generation: max_tokens={self.args.output_len}) ---", flush=True)
                iter_e2e_lat = []
                iter_gen_tokens = []
                iter_decode_lat = []
                iter_tpots = []
                iter_decode_tputs = []

                est_prefill_ms = self.prefill_latencies.get(bs, 0.0)

                for it in range(self.args.num_iters):
                    prompts, self.prompt_cursor = get_prompts_for_batch(
                        bs=bs,
                        it=it,
                        stage="gen",
                        fixed_prompt_len=self.args.fixed_prompt_len,
                        all_prompts=self.all_prompts,
                        cursor=self.prompt_cursor
                    )
                    res = ray.get(self.worker.run_generation_iteration.remote(prompts, self.args.output_len))
                    total_time_ms = res["total_time_ms"]
                    gen_toks = res["actual_gen_tokens"]
                    prompt_toks = res["prompt_tokens"]

                    decode_time_ms = max(total_time_ms - est_prefill_ms, 1.0)
                    avg_out_len = gen_toks / bs if bs > 0 else 1.0
                    tpot_ms = decode_time_ms / avg_out_len if avg_out_len > 0 else 0.0
                    decode_tput = (gen_toks / (decode_time_ms / 1000.0)) if decode_time_ms > 0 else 0.0

                    iter_e2e_lat.append(total_time_ms)
                    iter_gen_tokens.append(gen_toks)
                    iter_decode_lat.append(decode_time_ms)
                    iter_tpots.append(tpot_ms)
                    iter_decode_tputs.append(decode_tput)

                    print(f"  Iter {it+1}/{self.args.num_iters}: Prompt {prompt_toks} tok -> Gen {gen_toks} tok | E2E: {total_time_ms:.2f} ms | Decode: {decode_time_ms:.2f} ms | TPOT: {tpot_ms:.2f} ms/tok | Decode Tput: {decode_tput:.1f} tok/s", flush=True)
                    if it == 0:
                        print(f"    Sample Prompt: \"{res['sample_prompt']}\"", flush=True)
                        print(f"    Sample Output: \"{res['sample_response'][:80]}...\"", flush=True)

                med_e2e = float(np.median(iter_e2e_lat))
                med_decode = float(np.median(iter_decode_lat))
                med_tpot = float(np.median(iter_tpots))
                med_decode_tput = float(np.median(iter_decode_tputs))
                med_gen_toks = int(np.median(iter_gen_tokens))
                mean_tpot = float(np.mean(iter_tpots))
                std_tpot = float(np.std(iter_tpots))
                cv_tpot = float((std_tpot / mean_tpot) * 100.0) if mean_tpot > 0 else 0.0

                row = {
                    "batch_size": bs,
                    "gen_tokens": med_gen_toks,
                    "e2e_latency_ms": round(med_e2e, 2),
                    "est_prefill_latency_ms": round(est_prefill_ms, 2),
                    "decode_latency_ms": round(med_decode, 2),
                    "tpot_ms": round(med_tpot, 2),
                    "decode_tok_s": round(med_decode_tput, 1),
                    "mean_tpot_ms": round(mean_tpot, 2),
                    "std_tpot_ms": round(std_tpot, 2),
                    "cv_tpot_percent": round(cv_tpot, 2),
                    "iter_e2e_latencies_ms": iter_e2e_lat,
                    "iter_tpots_ms": iter_tpots,
                    "iter_decode_tputs_tok_s": iter_decode_tputs
                }
                stage2_results.append(row)
                print(f"  >> MEDIAN: E2E = {med_e2e:.2f} ms | Pure Decode = {med_decode:.2f} ms | TPOT = {med_tpot:.2f} ms/tok | Decode Tput = {med_decode_tput:.1f} tok/s | CV = {cv_tpot:.1f}%", flush=True)

            self.benchmark_data["vllm_benchmark"]["stage_2_decode"] = stage2_results
            heartbeat.stop()
            emit_omp_event("STAGE_COMPLETED", stage=stage_name, elapsed=round(time.perf_counter() - t0, 2))
            return True

        except Exception as e:
            heartbeat.stop()
            action = self.trap_error_and_pause(stage_name, "DECODE_FAILED", e, "VLLM_PURE_DECODE")
            if action == "retry":
                return self.stage_2_vllm_pure_decode()
            return False

    def stage_3_aggregate_and_report(self) -> bool:
        """Stage 3: Aggregate Results, Save JSON, & Render Reports."""
        stage_name = "AGGREGATE_AND_REPORT"
        emit_omp_event("STAGE_STARTED", stage=stage_name)
        t0 = time.perf_counter()

        try:
            # 1. Write JSON Results
            out_json_path = os.path.abspath(self.args.output_json)
            os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(self.benchmark_data, f, indent=2)
            print(f"\n[RESULTS] JSON benchmark metrics written to: {out_json_path}", flush=True)

            # 2. Print Summary Tables
            st1 = self.benchmark_data["vllm_benchmark"].get("stage_1_prefill", [])
            st2 = self.benchmark_data["vllm_benchmark"].get("stage_2_decode", [])

            print("\n" + "=" * 115, flush=True)
            print(f"  BENCHMARK SUMMARY (TP={self.args.tp_size}): Gemma-4 31B on Cloud TPU v5e", flush=True)
            print("=" * 115, flush=True)
            print(f"{'BS':<5} | {'Prompt Tok':<12} | {'Prefill (ms)':<14} | {'Prefill Tok/s':<15} | {'Gen Tok':<10} | {'E2E (ms)':<10} | {'Decode (ms)':<12} | {'TPOT (ms/tok)':<14} | {'Decode Tok/s':<14}", flush=True)
            print("-" * 115, flush=True)

            for i in range(len(st1)):
                r1 = st1[i]
                r2 = st2[i] if i < len(st2) else {}
                print(f"{r1['batch_size']:<5} | {r1['total_prompt_tokens']:<12} | {r1['latency_ms']:<14.2f} | {r1['throughput_tok_s']:<15.1f} | {r2.get('gen_tokens', 0):<10} | {r2.get('e2e_latency_ms', 0.0):<10.2f} | {r2.get('decode_latency_ms', 0.0):<12.2f} | {r2.get('tpot_ms', 0.0):<14.2f} | {r2.get('decode_tok_s', 0.0):<14.1f}", flush=True)
            print("=" * 115 + "\n", flush=True)

            # 3. Clean up vLLM worker and placement group
            if self.worker:
                try:
                    ray.get(self.worker.shutdown.remote())
                except Exception:
                    pass

            if self.placement_group:
                try:
                    remove_placement_group(self.placement_group)
                    print("[CLEANUP] Ray placement group removed successfully.", flush=True)
                except Exception:
                    pass

            emit_omp_event("STAGE_COMPLETED", stage=stage_name, elapsed=round(time.perf_counter() - t0, 2))
            emit_omp_event("BENCHMARK_SUCCESS", json=out_json_path)
            return True

        except Exception as e:
            action = self.trap_error_and_pause(stage_name, "REPORT_GEN_FAILED", e, "AGGREGATE_AND_REPORT")
            if action == "retry":
                return self.stage_3_aggregate_and_report()
            return False

    def run(self):
        """Main execution entrypoint managing state machine transitions."""
        resume_stage = self.args.resume_stage
        start_idx = STAGE_ORDER.index(resume_stage) if resume_stage in STAGE_ORDER else 0

        dataset_label = f"Fixed-Length ({self.args.fixed_prompt_len} tok)" if self.args.fixed_prompt_len else "ShareGPT"
        print("=" * 115, flush=True)
        print(f"  vLLM Decoupled Microbenchmark Orchestrator: Cloud TPU v5e (TP={self.args.tp_size})", flush=True)
        print(f"  Batch Sizes : {self.args.batch_sizes}", flush=True)
        print(f"  Dataset     : {dataset_label}", flush=True)
        print(f"  Output Len  : {self.args.output_len} | Iterations: {self.args.num_iters}", flush=True)
        print(f"  Start Stage : {STAGE_ORDER[start_idx]}", flush=True)
        print("=" * 115, flush=True)

        stages = [
            (self.stage_0_discover_topology, "DISCOVER_TOPOLOGY"),
            (self.stage_1_vllm_isolated_prefill, "VLLM_ISOLATED_PREFILL"),
            (self.stage_2_vllm_pure_decode, "VLLM_PURE_DECODE"),
            (self.stage_3_aggregate_and_report, "AGGREGATE_AND_REPORT"),
        ]

        for idx in range(start_idx, len(stages)):
            stage_fn, stage_name = stages[idx]
            success = stage_fn()
            if not success:
                print(f"[FATAL] Pipeline stopped at stage {stage_name}.", flush=True)
                sys.exit(1)

        print("  All vLLM Microbenchmark Stages Completed Successfully!")
        print("=" * 115 + "\n", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Decoupled Microbenchmark Orchestrator for Cloud TPU v5e")
    parser.add_argument("--tp-size", type=int, default=8, choices=[4, 8, 16], help="Tensor parallelism size (default: 8)")
    parser.add_argument("--batch-sizes", type=str, default="16,32,64,128,256", help="Comma-separated batch sizes")
    parser.add_argument("--dataset-path", type=str, default=None, help="ShareGPT dataset JSON path")
    parser.add_argument("--fixed-prompt-len", type=int, default=512, help="If set, generates fixed-length randomized token prompts")
    parser.add_argument("--output-len", type=int, default=64, help="Generation output length (default: 64)")
    parser.add_argument("--num-iters", type=int, default=3, help="Number of benchmark iterations per batch size")
    parser.add_argument("--timeout-per-stage", type=int, default=600, help="Per-stage timeout in seconds (default: 600)")
    parser.add_argument("--output-json", type=str, default="admission-control-vllm/benchmarks/gemma4_31b_tpu_v5e/results/vllm_decoupled_prefill_decode_results.json", help="Output JSON results path")
    parser.add_argument("--resume-stage", type=str, default=None, choices=STAGE_ORDER, help="Resume execution from a specific stage")
    parser.add_argument("--force-invalid-pair", action="store_true", help="Force invalid host pair to test error hook")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (default: auto)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Model weight directory")
    args = parser.parse_args()

    if isinstance(args.batch_sizes, str):
        args.batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    return args


if __name__ == "__main__":
    args = parse_args()
    orchestrator = TPU2x4DynamicBatchOrchestrator(args)
    orchestrator.run()
