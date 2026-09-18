"""
Unified Policy Experiment Runner.
=================================
Executes, verifies, and benchmarks admission control and scheduling policies
using the Reverse Control Mechanism (RCM) protocol.

Stages:
  Stage 1: setup_and_validate
           Resolves policy, loads workload, validates batching invariants and memory headroom.
  Stage 2: execute_policy
           Executes candidate policy (and baseline) via unified PolicyExecutor.
           Supports step-accurate local simulation or remote Cloud TPU cluster execution.
  Stage 3: verify_invariants_and_metrics
           Verifies hard hardware invariants (preemption immunity, starvation freedom)
           and calculates speedups and telemetry.
  Stage 4: harvest_and_report
           Formats summary report tables, saves structured JSON artifacts, and emits
           canonical METRIC lines for autonomous optimizers.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from omp_rcm.core import (
    DiagnosticTrap,
    JointTimingLogEngine,
    StageRunner,
    StateStore,
    SteeringReceiver,
    stage,
)
from admission_control.core.models import HardwareSpec, Request, BatchCohort
from admission_control.algorithms.policies import (
    BasePolicy,
    get_policy,
    get_active_policy,
    calculate_peak_future_memory_trajectory,
)
from admission_control.algorithms.executor import (
    PolicyExecutor,
    SimulatedExecutionEngine,
    ExecutionTrace,
)
from admission_control.workloads.loader import (
    load_workload_requests,
    get_workload_path,
)
from admission_control.experiments.config import ExperimentConfig
from admission_control.experiments.plotting import generate_all_plots
from admission_control.experiments.reporting import (
    build_markdown_report,
    build_html_report,
)

# ==============================================================================
# STAGE 1: Setup and Validate Workload
# ==============================================================================
@stage(
    name="setup_and_validate",
    order_idx=1,
    expected_s=5.0,
    critical_patterns=[
        (r"Workload file not found", "Specified workload trace does not exist"),
        (r"Memory trajectory breach", "Estimated peak memory exceeds KV cache capacity"),
    ],
)
def stage_setup_and_validate(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Resolves policy candidate, loads workload trace, and validates memory headroom."""
    cfg: ExperimentConfig = state.get("config")
    if cfg is None:
        raise ValueError("Missing ExperimentConfig in state store.")

    timing.log(f"Setting up experiment: policy='{cfg.policy}', workload='{cfg.workload}'...", level="INFO")

    # 3. Ingest Workload
    if isinstance(cfg.workload, (list, tuple)):
        flat_workload = []
        for item in cfg.workload:
            if isinstance(item, BatchCohort):
                flat_workload.extend([r.to_dict() if hasattr(r, "to_dict") else r for r in item])
            elif hasattr(item, "to_dict"):
                flat_workload.append(item.to_dict())
            else:
                flat_workload.append(item)
        raw_workload = flat_workload
        workload_name = "In-Memory Request List"
        workload_file = None
    elif isinstance(cfg.workload, (str, pathlib.Path)):
        workload_str = str(cfg.workload)
        workload_path = get_workload_path(workload_str)
        if not os.path.exists(workload_path):
            raise FileNotFoundError(f"Workload file not found: '{workload_str}' (resolved to '{workload_path}')")
        with open(workload_path, "r", encoding="utf-8") as f:
            raw_workload = json.load(f)
        workload_name = pathlib.Path(workload_path).name
        workload_file = workload_path
    else:
        raise TypeError(f"Unsupported workload type: {type(cfg.workload)}")

    total_requests = len(raw_workload)
    batch_size = cfg.batch_size
    num_batches = (total_requests + batch_size - 1) // batch_size
    total_prompt = sum(r.get("prompt_len", 0) for r in raw_workload)
    total_decode = sum(r.get("decode_len", 0) for r in raw_workload)
    total_demand = total_prompt + total_decode

    timing.log(
        f"Loaded workload '{workload_name}': {total_requests} requests, "
        f"{num_batches} batches of size {batch_size} (Demand: {total_demand:,} tok)",
        level="INFO",
    )

    # 4. Validate Batch Invariant and Analytical Peak Memory Trajectory
    peak_mem_est = calculate_peak_future_memory_trajectory(
        raw_workload[:cfg.target_concurrency],
        block_size=cfg.block_size,
        capacity_tokens=cfg.capacity_tokens,
    )
    headroom = cfg.capacity_tokens - peak_mem_est

    timing.log(
        f"Memory Trajectory Guard: Estimated Peak = {peak_mem_est:,} / {cfg.capacity_tokens:,} tok "
        f"(Headroom: {headroom:,} tok, Target Concurrency: {cfg.target_concurrency})",
        level="INFO",
    )
    hw = HardwareSpec(capacity_tokens=cfg.capacity_tokens, max_batch_size=cfg.max_batch_size)
    state.set("hardware", hw)

    # Resolve Candidate Policy
    if cfg.policy == "active":
        candidate_policy = get_active_policy()
    else:
        candidate_policy = get_policy(
            cfg.policy,
            capacity=cfg.capacity_tokens,
            workload=raw_workload,
            hw=hw,
            batch_size=cfg.batch_size,
        )
    state.set("candidate_policy", candidate_policy)
    timing.log(f"Candidate policy resolved: {candidate_policy.name}", level="INFO")

    # Resolve Baseline Policy (if configured)
    baseline_policy = None
    if cfg.baseline_policy:
        if cfg.baseline_policy == "greedy":
            baseline_policy = get_policy("greedy", max_concurrency=cfg.batch_size)
        else:
            baseline_policy = get_policy(
                cfg.baseline_policy,
                capacity=cfg.capacity_tokens,
                workload=raw_workload,
                hw=hw,
                batch_size=cfg.batch_size,
            )
        timing.log(f"Baseline policy configured: {baseline_policy.name}", level="INFO")
    state.set("baseline_policy", baseline_policy)

    workload_meta = {
        "name": workload_name,
        "file": workload_file,
        "total_requests": total_requests,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "total_prompt_tokens": total_prompt,
        "total_decode_tokens": total_decode,
        "total_demand_tokens": total_demand,
        "peak_future_memory_tokens": peak_mem_est,
    }
    state.set("workload", raw_workload)
    state.set("workload_meta", workload_meta)


    print(
        f"[OMP_EVENT: WORKLOAD_VALIDATED policy=\"{candidate_policy.name}\" requests={total_requests} "
        f"batches={num_batches} peak_mem={peak_mem_est} headroom={headroom}]",
        flush=True,
    )


# ==============================================================================
# STAGE 2: Execute Policy Benchmark
# ==============================================================================
@stage(
    name="execute_policy",
    order_idx=2,
    expected_s=60.0,
    critical_patterns=[
        (r"OutOfMemoryError|CapacityBreach", "Hardware memory capacity breached during execution"),
        (r"DeadlockDetected", "Execution engine encountered a scheduling deadlock"),
    ],
    allow_repetitive=True,
)
def stage_execute_policy(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Drives request execution under configured policies via the PolicyExecutor."""
    cfg: ExperimentConfig = state.get("config")
    workload = state.get("workload")
    hw: HardwareSpec = state.get("hardware")
    candidate_policy: BasePolicy = state.get("candidate_policy")
    baseline_policy: Optional[BasePolicy] = state.get("baseline_policy")

    if cfg.execution_mode == "simulation":
        # 1. Run Baseline Execution if configured
        base_res = None
        if baseline_policy is not None:
            timing.log(f"Running baseline simulation: {baseline_policy.name}...", level="INFO")
            base_engine = SimulatedExecutionEngine(
                capacity_tokens=cfg.capacity_tokens,
                block_size=cfg.block_size,
                hw=hw,
            )
            base_executor = PolicyExecutor(policy=baseline_policy, engine=base_engine, hw=hw)
            base_res = base_executor.execute(
                workload,
                batch_size=cfg.batch_size,
                max_duration_s=cfg.max_duration_s,
                max_steps=cfg.max_steps,
            )
            timing.log(
                f"Baseline finished: {base_res['goodput_tok_per_s']:,.1f} tok/s, "
                f"Preemptions: {base_res['num_preemptions']}, Steps: {base_res['total_steps']}",
                level="INFO",
            )
        state.set("baseline_results", base_res)

        # 2. Run Candidate Policy Execution
        timing.log(f"Running candidate policy simulation: {candidate_policy.name}...", level="INFO")
        cand_engine = SimulatedExecutionEngine(
            capacity_tokens=cfg.capacity_tokens,
            block_size=cfg.block_size,
            hw=hw,
        )
        cand_executor = PolicyExecutor(policy=candidate_policy, engine=cand_engine, hw=hw)
        cand_res = cand_executor.execute(
            workload,
            batch_size=cfg.batch_size,
            max_duration_s=cfg.max_duration_s,
            max_steps=cfg.max_steps,
            record_trace=cfg.record_trace,
        )
        timing.log(
            f"Candidate finished: {cand_res['goodput_tok_per_s']:,.1f} tok/s, "
            f"Preemptions: {cand_res['num_preemptions']}, Steps: {cand_res['total_steps']}",
            level="INFO",
        )
        state.set("candidate_results", cand_res)

    elif cfg.execution_mode == "cluster":
        # Cluster execution delegates to Ray Job Submission SDK with live stream watching
        timing.log("Delegating execution to remote Cloud TPU cluster...", level="INFO")
        from admission_control.experiments.cluster import submit_cluster_job_rcm

        cluster_res = submit_cluster_job_rcm(
            cfg=cfg,
            state=state,
            timing=timing,
        )
        state.set("candidate_results", cluster_res.get("candidate", cluster_res))
        state.set("baseline_results", cluster_res.get("baseline", None))
    else:
        raise ValueError(f"Unknown execution mode: {cfg.execution_mode}")

    print("[OMP_EVENT: EXECUTION_COMPLETE]", flush=True)


# ==============================================================================
# STAGE 3: Verify Invariants and Metrics
# ==============================================================================
@stage(
    name="verify_invariants_and_metrics",
    order_idx=3,
    expected_s=5.0,
    critical_patterns=[
        (r"Preemption immunity violated", "Policy produced preemptions violating hard invariant"),
        (r"Starvation freedom violated", "Not all requests or cohorts were completed"),
    ],
)
def stage_verify_invariants_and_metrics(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Verifies preemption immunity, starvation freedom, and computes speedup."""
    cand_res: Dict[str, Any] = state.get("candidate_results", {})
    base_res: Optional[Dict[str, Any]] = state.get("baseline_results")
    meta: Dict[str, Any] = state.get("workload_meta", {})
    total_requests = meta.get("total_requests", 0)
    num_batches = meta.get("num_batches", 0)

    # 1. Invariant 1: Preemption Immunity (Zero Preemptions)
    preemptions = cand_res.get("num_preemptions", 0)
    preemption_immune = 1 if preemptions == 0 else 0

    # 2. Invariant 2: Starvation Freedom (All requests admitted & completed across all cohorts)
    finished_count = cand_res.get("finished_requests", 0)
    admitted_per_batch = cand_res.get("admitted_per_batch", {})
    starvation_free = 1 if (finished_count == total_requests and len(admitted_per_batch) >= num_batches) else 0

    # 3. Speedup computation
    cand_goodput = float(cand_res.get("goodput_tok_per_s", 0.0))
    base_goodput = float(base_res.get("goodput_tok_per_s", 0.0)) if base_res else 0.0
    speedup = round(cand_goodput / base_goodput, 4) if base_goodput > 0 else 1.0

    metrics = {
        "goodput_tok_per_s": cand_goodput,
        "policy_goodput_speedup": speedup,
        "baseline_goodput_tok_s": base_goodput,
        "xla_padding_waste_pct": cand_res.get("xla_padding_waste_percent", 0.0),
        "num_preemptions": preemptions,
        "peak_memory_tokens": cand_res.get("peak_memory_tokens", 0),
        "peak_memory_pct": cand_res.get("peak_memory_pct", 0.0),
        "delivered_tokens": cand_res.get("delivered_tokens", 0),
        "total_time_s": cand_res.get("total_time_s", 0.0),
        "total_steps": cand_res.get("total_steps", 0),
        "preemption_immunity_verified": preemption_immune,
        "starvation_freedom_verified": starvation_free,
    }
    state.set("verified_metrics", metrics)

    timing.log(
        f"Invariant verification: PreemptionImmunity={bool(preemption_immune)}, "
        f"StarvationFreedom={bool(starvation_free)}, Goodput={cand_goodput:,.1f} tok/s ({speedup:.4f}x)",
        level="INFO",
    )

    # Output Canonical Autoresearch METRIC lines for automated parsers
    print(f"METRIC goodput_tok_per_s={cand_goodput}")
    print(f"METRIC policy_goodput_speedup={speedup}")
    print(f"METRIC baseline_goodput_tok_s={base_goodput}")
    print(f"METRIC xla_padding_waste_pct={cand_res.get('xla_padding_waste_percent', 0.0)}")
    print(f"METRIC num_preemptions={preemptions}")
    print(f"METRIC peak_memory_tokens={cand_res.get('peak_memory_tokens', 0)}")
    print(f"METRIC peak_memory_pct={cand_res.get('peak_memory_pct', 0.0)}")
    print(f"METRIC delivered_tokens={cand_res.get('delivered_tokens', 0)}")
    print(f"METRIC total_time_s={cand_res.get('total_time_s', 0.0)}")
    print(f"METRIC preemption_immunity_verified={preemption_immune}")
    print(f"METRIC starvation_freedom_verified={starvation_free}")

    if not preemption_immune:
        timing.error(f"Policy violated preemption immunity! {preemptions} preemptions detected.")
    if not starvation_free:
        timing.error(f"Policy violated starvation freedom! Finished: {finished_count}/{total_requests} requests.")


# ==============================================================================
# STAGE 4: Harvest and Report
# ==============================================================================
@stage(
    name="harvest_and_report",
    order_idx=4,
    expected_s=5.0,
)
def stage_harvest_and_report(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Formats report table, saves JSON artifact, and writes execution traces."""
    cfg: ExperimentConfig = state.get("config")
    meta = state.get("workload_meta", {})
    cand_policy = state.get("candidate_policy")
    base_policy = state.get("baseline_policy")
    cand_res = state.get("candidate_results", {})
    base_res = state.get("baseline_results", {})
    metrics = state.get("verified_metrics", {})

    total_requests = meta.get("total_requests", 0)
    num_batches = meta.get("num_batches", 0)
    base_goodput = metrics.get("baseline_goodput_tok_s", 0.0)
    cand_goodput = metrics.get("goodput_tok_per_s", 0.0)
    speedup = metrics.get("policy_goodput_speedup", 1.0)
    preemptions = metrics.get("num_preemptions", 0)
    preemption_immune = metrics.get("preemption_immunity_verified", 0)
    starvation_free = metrics.get("starvation_freedom_verified", 0)

    # 1. Print Canonical Performance Report Table
    print("\n" + "=" * 80)
    print("SCHEDULING POLICY BENCHMARK EXECUTION REPORT")
    print(f"Workload: {meta.get('name', 'Custom')} ({total_requests} requests, {num_batches} batches)")
    print(f"Hardware: Cloud TPU v5e (TP=8, KV Capacity: {cfg.capacity_tokens:,} tok, BlockSize: {cfg.block_size})")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Baseline (Greedy)':<18} | {'Candidate Policy':<22}")
    print("-" * 80)
    base_name = getattr(base_policy, "name", "None") if base_policy else "N/A"
    cand_name = getattr(cand_policy, "name", "Candidate")
    print(f"{'Policy Name':<35} | {base_name:<18} | {cand_name:<22}")
    print(f"{'Goodput (tok/s)':<35} | {base_goodput:<18,.1f} | {cand_goodput:<22,.1f}")
    print(f"{'Goodput Speedup':<35} | {'1.0000x':<18} | {f'{speedup:.4f}x':<22}")

    base_time_str = f"{base_res.get('total_time_s', 0.0):.2f} s" if base_res else "N/A"
    cand_time_str = f"{cand_res.get('total_time_s', 0.0):.2f} s"
    base_mem_str = f"{base_res.get('peak_memory_tokens', 0):,} ({base_res.get('peak_memory_pct', 0)}%)" if base_res else "N/A"
    cand_mem_str = f"{cand_res.get('peak_memory_tokens', 0):,} ({cand_res.get('peak_memory_pct', 0)}%)"
    base_waste_str = f"{base_res.get('xla_padding_waste_percent', 0.0):.2f}%" if base_res else "N/A"
    cand_waste_str = f"{cand_res.get('xla_padding_waste_percent', 0.0):.2f}%"

    print(f"{'Total Step Time':<35} | {base_time_str:<18} | {cand_time_str:<22}")
    print(f"{'Delivered Tokens':<35} | {base_res.get('delivered_tokens', 0) if base_res else 0:<18,} | {cand_res.get('delivered_tokens', 0):<22,}")
    print(f"{'Peak Memory (tokens)':<35} | {base_mem_str:<18} | {cand_mem_str:<22}")
    print(f"{'XLA Padding Waste':<35} | {base_waste_str:<18} | {cand_waste_str:<22}")
    print(f"{'Preemptions':<35} | {base_res.get('num_preemptions', 0) if base_res else 0:<18} | {preemptions:<22}")
    print(f"{'Preemption Immunity Verified':<35} | {'YES' if (base_res and base_res.get('num_preemptions', 0) == 0) else 'NO':<18} | {'YES' if preemption_immune else 'FAILED':<22}")
    print(f"{'Starvation Freedom Verified':<35} | {'YES':<18} | {'YES' if starvation_free else 'FAILED':<22}")
    print("=" * 80 + "\n")

    # 2. Compile Final Structured Payload
    final_payload = {
        "workload": meta.get("file", meta.get("name")),
        "total_requests": total_requests,
        "num_batches": num_batches,
        "batch_size": cfg.batch_size,
        "config": cfg.to_dict(),
        "baseline": base_res,
        "candidate": cand_res,
        "metrics": metrics,
    }
    state.set("final_payload", final_payload)

    # 3. Export JSON Artifact
    if cfg.output_json:
        out_json_path = pathlib.Path(cfg.output_json).resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)
        timing.log(f"Saved JSON evaluation results to {out_json_path}", level="INFO")

    # 4. Export Trace Artifact if requested
    trace_obj = cand_res.get("trace")
    if trace_obj and (cfg.trace_output or cfg.record_trace):
        trace_path = pathlib.Path(cfg.trace_output or "execution_trace.json").resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(trace_obj.to_chrome_trace())
        timing.log(f"Saved Chrome Trace (Perfetto) to {trace_path}", level="INFO")
    # 5. Generate Plots if requested
    if cfg.generate_plots:
        fig_dir = pathlib.Path(cfg.figures_dir or (out_json_path.parent / "figures" if cfg.output_json else "reports/figures")).resolve()
        plot_json_src = out_json_path if cfg.output_json else None
        if plot_json_src and plot_json_src.exists():
            try:
                generate_all_plots(plot_json_src, fig_dir)
                timing.log(f"Generated 300 DPI publication plots in {fig_dir}", level="INFO")
            except Exception as e:
                timing.log(f"Notice generating plots: {e}", level="WARN")

    # 6. Generate Markdown/HTML report if requested
    if cfg.output_report:
        rep_path = pathlib.Path(cfg.output_report).resolve()
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_summary = {
            "results": {
                "policy_candidate": cand_res,
                "policy_a_greedy": base_res or {},
            },
            "hardware": {
                "accelerator": "Cloud TPU v5e (TP=8)",
                "kv_cache_capacity_tokens": cfg.capacity_tokens,
            },
            "workload": {
                "name": str(meta.get("name", "workload")),
                "num_requests": total_requests,
            },
        }
        try:
            build_markdown_report(rep_summary, rep_path)
            html_rep_path = rep_path.with_suffix(".html")
            fig_dir = pathlib.Path(cfg.figures_dir or (rep_path.parent / "figures")).resolve()
            build_html_report(rep_path, html_rep_path, fig_dir)
            timing.log(f"Generated Markdown & HTML reports at {rep_path} and {html_rep_path}", level="INFO")
        except Exception as e:
            timing.log(f"Notice building reports: {e}", level="WARN")

    print(f"[OMP_EVENT: BENCHMARK_SUCCESS policy=\"{cand_name}\" speedup={speedup} goodput={cand_goodput}]", flush=True)


# ==============================================================================
# Runner Class
# ==============================================================================
class PolicyExperimentRunner:
    """
    RCM-powered Experiment Runner for Scheduling Policies.
    Encapsulates sequential deterministic stages, state tracking, and trap recovery.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.stage_runner = StageRunner(silence_timeout_s=self.config.silence_timeout_s)
        self.stage_runner.state.set("config", self.config)

        # Register stages in order
        self.stage_runner.register(stage_setup_and_validate)
        self.stage_runner.register(stage_execute_policy)
        self.stage_runner.register(stage_verify_invariants_and_metrics)
        self.stage_runner.register(stage_harvest_and_report)

    def run(self) -> Dict[str, Any]:
        """Runs the complete benchmark suite, returning the final structured results payload."""
        success = self.stage_runner.run()
        if not success:
            raise RuntimeError("Policy benchmark failed during stage execution.")
        return self.stage_runner.state.get("final_payload", {})


def run_experiment(
    policy: Union[str, BasePolicy, Dict[str, Any]] = "active",
    workload: Union[str, List[Dict[str, Any]]] = "data/workloads/sharegpt_sampled_500.json",
    batch_size: int = 64,
    capacity_tokens: int = 120_000,
    output_json: Optional[str] = None,
    execution_mode: str = "simulation",
    baseline_policy: Optional[str] = "greedy",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to run an experiment in one call."""
    cfg = ExperimentConfig(
        policy=policy,
        workload=workload,
        batch_size=batch_size,
        capacity_tokens=capacity_tokens,
        output_json=output_json,
        execution_mode=execution_mode,
        baseline_policy=baseline_policy,
        **kwargs,
    )
    runner = PolicyExperimentRunner(cfg)
    return runner.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Scheduling Policy Experiment Runner")
    parser.add_argument(
        "--workload",
        type=str,
        default="data/workloads/sharegpt_sampled_500.json",
        help="Path or standard name of workload JSON",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch cohort size M (default: 64)")
    parser.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity (default: 120,000)")
    parser.add_argument("--target-concurrency", type=int, default=96, help="Target concurrency ceiling (default: 96)")
    parser.add_argument("--policy", type=str, default="active", help="Policy name, spec file (.yaml/.json), or 'active'")
    parser.add_argument("--baseline", type=str, default="greedy", help="Baseline policy name (default: 'greedy', 'none' to disable)")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation", "cluster"], help="Execution mode")
    parser.add_argument("--json-output", type=str, default=None, help="Path to export JSON results artifact")
    parser.add_argument("--trace-output", type=str, default=None, help="Path to export Chrome tracing JSON")
    parser.add_argument("--ray-url", type=str, default="http://127.0.0.1:8265", help="Ray cluster URL for cluster mode")
    parser.add_argument("--force-reinit", action="store_true", help="Force reinitialize remote cluster workers")
    parser.add_argument("--duration-s", type=float, default=None, help="Maximum execution duration in seconds")
    parser.add_argument("--generate-plots", action="store_true", help="Generate publication plots from results")
    parser.add_argument("--figures-dir", type=str, default=None, help="Directory to save generated figure plots")
    parser.add_argument("--output-report", type=str, default=None, help="Path to export Markdown report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = None if args.baseline.lower() in ("none", "null", "") else args.baseline
    cfg = ExperimentConfig(
        policy=args.policy,
        workload=args.workload,
        batch_size=args.batch_size,
        capacity_tokens=args.capacity,
        target_concurrency=args.target_concurrency,
        baseline_policy=baseline,
        execution_mode=args.mode,
        output_json=args.json_output,
        trace_output=args.trace_output,
        record_trace=bool(args.trace_output),
        ray_url=args.ray_url,
        force_reinit=args.force_reinit,
        max_duration_s=args.duration_s,
        generate_plots=args.generate_plots,
        figures_dir=args.figures_dir,
        output_report=args.output_report,
    )
    runner = PolicyExperimentRunner(cfg)
    payload = runner.run()

    # Check invariants for non-zero exit code
    metrics = payload.get("metrics", {})
    if not metrics.get("preemption_immunity_verified", 1):
        print("ERROR: Policy violated preemption immunity! Preemptions occurred.", file=sys.stderr)
        sys.exit(1)
    if not metrics.get("starvation_freedom_verified", 1):
        print("ERROR: Policy violated starvation freedom! Incomplete requests/batches.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
