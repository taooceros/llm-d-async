"""
vLLM E2E Serving Benchmark for Gemma-4 31B on Cloud TPU v5e (TP=16) using ShareGPT Dataset
==========================================================================================
Passes the explicit Ray Placement Group to vLLM via native PlacementGroupSchedulingStrategy
without modifying any vLLM or tpu-inference source code.
"""
import os
import sys
import time
import json
import argparse
import random
import numpy as np

MODEL_PATH = "/models/gemma-4-31b"
DEFAULT_DATASET_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sharegpt_sampled_500.json"),
    "/tmp/sharegpt_sampled_500.json",
    "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json",
]


def load_sharegpt_prompts(dataset_path: str = None, max_prompts: int = 500):
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


import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
    PlacementGroup
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=0)
class VLLMShareGPTBenchmarkWorker:
    def __init__(self, model_path: str, tp_size: int, max_num_seqs: int = 32, target_ips: list = None, host_bounds: str = "2,1,1"):
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

        print("\n[Worker] Initializing vLLM Engine inside explicitly assigned placement group...", flush=True)
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
            sample_prompt = "Hello, how are you today?"
        out = self.llm.generate([sample_prompt], sp_warmup, use_tqdm=False)
        sample_text = out[0].outputs[0].text.strip()[:100]
        print(f"  Warmup Output Sample: \"{sample_text}...\"", flush=True)
        print("[Worker] Warmup complete.\n", flush=True)
        return sample_text

    def run_prefill_iteration(self, prompts: list[str]) -> dict:
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

    def run_generation_iteration(self, prompts: list[str], output_len: int = 64) -> dict:
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
        """Gracefully release vLLM engine resources."""
        try:
            del self.llm
        except Exception:
            pass
        return True
def get_prompts_for_batch(
    bs: int,
    it: int,
    stage: str,
    fixed_prompt_len: int = None,
    all_prompts: list = None,
    cursor: int = 0
):
    if fixed_prompt_len is not None and fixed_prompt_len > 0:
        # Generate exact fixed-length token sequences using unique random IDs per sequence
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


def run_vllm_sharegpt_benchmark(
    tp_size: int = 16,
    dataset_path: str = None,
    output_len: int = 64,
    num_iters: int = 5,
    batch_sizes=None,
    fixed_prompt_len: int = None,
):
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 32]
    # 1. Connect to Ray cluster
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    # 2. Step 0: Clean up stale placement groups
    print("\n[Step 0] Cleaning up stale placement groups in Ray cluster...", flush=True)
    table = placement_group_table()
    cleaned = 0
    for pg_id, pg_info in table.items():
        if pg_info.get("state") in ["CREATED", "PENDING", "RESCHEDULING"]:
            try:
                pg_handle = PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id)))
                remove_placement_group(pg_handle)
                cleaned += 1
            except Exception:
                pass
    if cleaned > 0:
        print(f"[Step 0] Cleaned {cleaned} stale placement group(s). Waiting 3s for resource reclamation...", flush=True)
        time.sleep(3.0)
    else:
        print("[Step 0] No stale placement groups found.", flush=True)
    # 3. Discover TPU worker nodes & create explicit node-pinned Placement Group
    tpu_nodes = [
        n for n in ray.nodes()
        if n.get("Alive") and n.get("Resources", {}).get("TPU", 0) > 0
    ]
    tpu_nodes = sorted(tpu_nodes, key=lambda n: n["NodeManagerAddress"])
    host_bounds = "2,1,1"
    if tp_size == 8:
        try:
            from pkg_util.tpu_topology import TPUTopologyDiscovery
            discovery = TPUTopologyDiscovery()
            pair = discovery.select_connected_2host_submesh()
            target_ips = [pair.host_a.node_ip, pair.host_b.node_ip]
            host_bounds = "2,1,1" if "Dimension X" in pair.mesh_dimension else "1,2,1"
            print(f"[Step 0] Topology Discovery selected connected pair: {pair.host_a.node_ip} + {pair.host_b.node_ip} ({pair.mesh_dimension}, bounds={host_bounds})", flush=True)
        except Exception as e:
            print(f"[Step 0] Topology discovery fallback: {e}, using first 2 TPU nodes", flush=True)
            target_ips = [n["NodeManagerAddress"] for n in tpu_nodes[:2]]
    else:
        target_ips = [n["NodeManagerAddress"] for n in tpu_nodes[:4]]

    bundle_specs = [{"TPU": 4.0, f"node:{ip}": 0.001} for ip in target_ips]
    print(f"[Step 0] Creating explicit node-pinned Placement Group across {len(target_ips)} hosts: {target_ips}...", flush=True)
    pg = placement_group(bundle_specs, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=120)
    print("[Step 0] Placement Group created and verified READY!\n", flush=True)

    # 4. Load ShareGPT prompts or initialize synthetic mode
    if fixed_prompt_len is not None and fixed_prompt_len > 0:
        all_prompts = None
        dataset_label = f"Synthetic Randomized Fixed-Length ({fixed_prompt_len} tokens/prompt)"
    else:
        all_prompts = load_sharegpt_prompts(dataset_path, max_prompts=1000)
        dataset_label = f"ShareGPT ({len(all_prompts)} prompts available)"

    print("=" * 115, flush=True)
    print(f"  vLLM Real Weights Benchmark (TP={tp_size}): Gemma-4 31B", flush=True)
    print(f"  Model Path     : {MODEL_PATH}", flush=True)
    print(f"  Dataset        : {dataset_label}", flush=True)
    print(f"  Max Output Len : {output_len} tokens", flush=True)
    print(f"  Batch Sizes    : {batch_sizes}", flush=True)
    print(f"  Iterations     : {num_iters} per batch size", flush=True)
    print("=" * 115, flush=True)

    # 5. Launch Benchmark Worker Actor on bundle 0 of the Placement Group
    print("\n[Step 1] Launching VLLMShareGPTBenchmarkWorker inside Placement Group Bundle 0...", flush=True)
    worker = VLLMShareGPTBenchmarkWorker.options(
        num_cpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0
        )
    ).remote(model_path=MODEL_PATH, tp_size=tp_size, max_num_seqs=max(batch_sizes), target_ips=target_ips, host_bounds=host_bounds)
    # 6. Warmup
    _ = ray.get(worker.warmup.remote(all_prompts[0] if all_prompts else None))

    prompt_cursor = 1
    all_results = {"prefill": [], "generation": [], "samples": []}

    # -------------------------------------------------------------------------
    # STAGE 1: ISOLATED PREFILL BENCHMARK (max_tokens = 1)
    # -------------------------------------------------------------------------
    print("=" * 115, flush=True)
    dataset_name = f"FIXED {fixed_prompt_len}-TOK" if fixed_prompt_len else "SHAREGPT"
    print(f"  STAGE 1: ISOLATED PREFILL BENCHMARK ON {dataset_name} (max_tokens = 1, unique prompts per iter)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'BS':<4} | {'Iter':<6} | {'Total Tok':<10} | {'Latency (ms)':<15} | {'Throughput (tok/s)':<22} | {'Notes':<15}", flush=True)
    print("-" * 115, flush=True)

    prefill_med_times = {}

    for bs in batch_sizes:
        iter_records = []
        for it in range(num_iters):
            prompts, prompt_cursor = get_prompts_for_batch(
                bs=bs,
                it=it,
                stage="prefill",
                fixed_prompt_len=fixed_prompt_len,
                all_prompts=all_prompts,
                cursor=prompt_cursor
            )
            res = ray.get(worker.run_prefill_iteration.remote(prompts))
            record = {
                "batch_size": bs,
                "iteration": it + 1,
                "total_prompt_tokens": res["total_prompt_tokens"],
                "latency_ms": res["latency_ms"],
                "throughput_tok_s": res["throughput_tok_s"],
            }
            iter_records.append(record)
            all_results["prefill"].append(record)

            print(f"{bs:<4} | #{it+1:<5} | {res['total_prompt_tokens']:<10} | {res['latency_ms']:<15.2f} | {res['throughput_tok_s']:<22.1f} | {'Measured':<15}", flush=True)

        med_lat = np.median([r["latency_ms"] for r in iter_records])
        med_tput = np.median([r["throughput_tok_s"] for r in iter_records])
        prefill_med_times[bs] = med_lat / 1000.0
        avg_tokens = int(np.mean([r["total_prompt_tokens"] for r in iter_records]))
        print(f"{bs:<4} | MEDIAN | ~{avg_tokens:<9} | {med_lat:<15.2f} | {med_tput:<22.1f} | {'Summary':<15}", flush=True)
        print("-" * 115, flush=True)

    # -------------------------------------------------------------------------
    # STAGE 2: END-TO-END GENERATION BENCHMARK (max_tokens = output_len)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115, flush=True)
    print(f"  STAGE 2: END-TO-END {dataset_name} GENERATION & DECODE (max_tokens = {output_len})", flush=True)
    print("=" * 115, flush=True)
    print(f"{'BS':<4} | {'Iter':<6} | {'Prompt Tok':<10} | {'Gen Tok':<8} | {'Total (ms)':<12} | {'Est Prefill':<12} | {'Decode (ms)':<12} | {'TPOT (ms)':<10} | {'Decode Tok/s':<12}", flush=True)
    print("-" * 115, flush=True)

    for bs in batch_sizes:
        iter_gen_records = []
        for it in range(num_iters):
            prompts, prompt_cursor = get_prompts_for_batch(
                bs=bs,
                it=it,
                stage="gen",
                fixed_prompt_len=fixed_prompt_len,
                all_prompts=all_prompts,
                cursor=prompt_cursor
            )
            res = ray.get(worker.run_generation_iteration.remote(prompts, output_len=output_len))
            total_ms = res["total_time_ms"]
            actual_gen_tokens = res["actual_gen_tokens"]
            actual_steps = actual_gen_tokens / bs if bs > 0 else 1

            est_prefill_ms = prefill_med_times[bs] * 1000.0
            pure_decode_ms = max(total_ms - est_prefill_ms, 1e-3)
            tpot_ms = pure_decode_ms / actual_steps if actual_steps > 0 else 0.0
            decode_tput = (actual_gen_tokens / (pure_decode_ms / 1000.0)) if pure_decode_ms > 0 else 0.0

            record = {
                "batch_size": bs,
                "iteration": it + 1,
                "prompt_tokens": res["prompt_tokens"],
                "actual_gen_tokens": actual_gen_tokens,
                "total_time_ms": total_ms,
                "est_prefill_ms": round(est_prefill_ms, 2),
                "pure_decode_ms": round(pure_decode_ms, 2),
                "tpot_ms": round(tpot_ms, 2),
                "decode_throughput_tok_s": round(decode_tput, 1),
            }
            iter_gen_records.append(record)
            all_results["generation"].append(record)

            if it == 0 and len(all_results["samples"]) < 4:
                all_results["samples"].append({
                    "batch_size": bs,
                    "prompt": res["sample_prompt"] + "...",
                    "response": res["sample_response"] + "..."
                })

            print(f"{bs:<4} | #{it+1:<5} | {res['prompt_tokens']:<10} | {actual_gen_tokens:<8} | {total_ms:<12.2f} | {est_prefill_ms:<12.2f} | {pure_decode_ms:<12.2f} | {tpot_ms:<10.2f} | {decode_tput:<12.1f}", flush=True)

        med_total = np.median([r["total_time_ms"] for r in iter_gen_records])
        med_decode = np.median([r["pure_decode_ms"] for r in iter_gen_records])
        med_tpot = np.median([r["tpot_ms"] for r in iter_gen_records])
        med_decode_tput = np.median([r["decode_throughput_tok_s"] for r in iter_gen_records])

        print(f"{bs:<4} | MEDIAN | ---------- | -------- | {med_total:<12.2f} | {est_prefill_ms:<12.2f} | {med_decode:<12.2f} | {med_tpot:<10.2f} | {med_decode_tput:<12.1f}", flush=True)
        print("-" * 115, flush=True)

    # -------------------------------------------------------------------------
    # PRINT SAMPLE MODEL GENERATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115, flush=True)
    print("  SAMPLE GENERATED RESPONSES FROM GEMMA-4 31B (VERIFYING SEMANTIC COHERENCE)", flush=True)
    print("=" * 115, flush=True)
    for i, s in enumerate(all_results["samples"]):
        print(f"\n[Sample {i+1} - Batch Size {s['batch_size']}]", flush=True)
        print(f"Prompt   : {s['prompt']}", flush=True)
        print(f"Response : {s['response']}", flush=True)
    print("=" * 115, flush=True)

    # Clean up worker and placement group gracefully
    try:
        ray.get(worker.shutdown.remote())
        ray.kill(worker)
        time.sleep(1.0)
    except Exception:
        pass

    try:
        remove_placement_group(pg)
    except Exception:
        pass
    return all_results


def main():
    parser = argparse.ArgumentParser(description="vLLM Real Weights Benchmark (ShareGPT & Fixed-Length Synthetic)")
    parser.add_argument("--tp-size", type=int, default=16, choices=[4, 8, 16])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--fixed-prompt-len", type=int, default=None, help="If set, generates fixed-length randomized token prompts instead of ShareGPT text")
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--batch-sizes", type=str, default="1,4,16,32")
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    run_vllm_sharegpt_benchmark(
        tp_size=args.tp_size,
        dataset_path=args.dataset_path,
        output_len=args.output_len,
        num_iters=args.num_iters,
        batch_sizes=batch_sizes,
        fixed_prompt_len=args.fixed_prompt_len,
    )
if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
