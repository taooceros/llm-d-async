"""
Policy Testing, Comparison & Parameter Sweeping Suite.
======================================================
Makes extending, verifying, and comparing scheduling policies trivial.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from admission_control.core.models import HardwareSpec, Request
from admission_control.algorithms.policies import (
    BasePolicy,
    get_policy,
    get_active_policy,
    verify_policy_invariants,
)
from admission_control.algorithms.executor import (
    PolicyExecutor,
    SimulatedExecutionEngine,
)
from admission_control.workloads.loader import (
    load_workload_requests,
    get_workload_path,
)
from admission_control.experiments.config import ExperimentConfig
from admission_control.experiments.runner import PolicyExperimentRunner, run_experiment


def test_policy(
    policy: Union[str, BasePolicy, Dict[str, Any], Any] = "active",
    workload: Union[str, List[Dict[str, Any]]] = "data/workloads/sharegpt_sampled_500.json",
    batch_size: int = 64,
    capacity_tokens: int = 120_000,
    baseline_policy: Optional[str] = "greedy",
    target_concurrency: int = 96,
    max_duration_s: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tests a scheduling policy in one line, verifying all hardware invariants.

    Returns:
        Dict with keys:
          - 'passed': bool (True if both preemption immunity and starvation freedom pass)
          - 'goodput_tok_per_s': float
          - 'speedup': float (relative to baseline)
          - 'num_preemptions': int
          - 'preemption_immune': bool
          - 'starvation_free': bool
          - 'metrics': Dict[str, Any]
          - 'results': Dict[str, Any] (full execution payload)
    """
    cfg = ExperimentConfig(
        policy=policy,
        workload=workload,
        batch_size=batch_size,
        capacity_tokens=capacity_tokens,
        target_concurrency=target_concurrency,
        baseline_policy=baseline_policy,
        execution_mode="simulation",
        max_duration_s=max_duration_s,
        max_steps=max_steps,
    )
    runner = PolicyExperimentRunner(cfg)
    payload = runner.run()
    metrics = payload.get("metrics", {})

    preempt_ok = bool(metrics.get("preemption_immunity_verified", 0))
    starve_ok = bool(metrics.get("starvation_freedom_verified", 0))
    passed = preempt_ok and starve_ok

    return {
        "passed": passed,
        "policy_name": payload.get("candidate", {}).get("policy_name", str(policy)),
        "goodput_tok_per_s": metrics.get("goodput_tok_per_s", 0.0),
        "speedup": metrics.get("policy_goodput_speedup", 1.0),
        "num_preemptions": metrics.get("num_preemptions", 0),
        "preemption_immune": preempt_ok,
        "starvation_free": starve_ok,
        "total_time_s": metrics.get("total_time_s", 0.0),
        "total_steps": metrics.get("total_steps", 0),
        "peak_memory_tokens": metrics.get("peak_memory_tokens", 0),
        "metrics": metrics,
        "results": payload,
    }


def compare_policies(
    policies: Sequence[Union[str, BasePolicy, Dict[str, Any], Any]],
    workload: Union[str, List[Dict[str, Any]]] = "data/workloads/sharegpt_sampled_500.json",
    batch_size: int = 64,
    capacity_tokens: int = 120_000,
    baseline_policy: Optional[str] = "greedy",
    print_table: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates and compares multiple scheduling policies side by side.

    Returns:
        Dict with keys:
          - 'comparison': List of per-policy metric dicts sorted by goodput descending
          - 'best_policy': Name of top-performing policy passing all invariants
          - 'table': Formatted text comparison table
    """
    results = []
    baseline_goodput = None

    for pol in policies:
        res = test_policy(
            policy=pol,
            workload=workload,
            batch_size=batch_size,
            capacity_tokens=capacity_tokens,
            baseline_policy=baseline_policy if baseline_goodput is None else None,
        )
        if baseline_goodput is None and res["metrics"].get("baseline_goodput_tok_s"):
            baseline_goodput = res["metrics"]["baseline_goodput_tok_s"]
        results.append(res)

    # Recompute speedup relative to common baseline if available
    if baseline_goodput and baseline_goodput > 0:
        for r in results:
            r["speedup"] = round(r["goodput_tok_per_s"] / baseline_goodput, 4)

    # Sort by goodput descending (filtering valid candidates)
    sorted_res = sorted(results, key=lambda x: (x["passed"], x["goodput_tok_per_s"]), reverse=True)
    best_candidate = sorted_res[0]["policy_name"] if sorted_res else "None"

    # Build comparison table
    lines = [
        "=" * 95,
        "SCHEDULING POLICY COMPARATIVE EVALUATION MATRIX",
        "=" * 95,
        f"{'Rank':<5} | {'Policy Name':<32} | {'Goodput (tok/s)':<17} | {'Speedup':<10} | {'Preempt':<8} | {'Status':<8}",
        "-" * 95,
    ]
    for idx, r in enumerate(sorted_res, 1):
        status = "PASSED" if r["passed"] else "VIOLATED"
        lines.append(
            f"{idx:<5} | {r['policy_name'][:32]:<32} | {r['goodput_tok_per_s']:>15,.1f} | "
            f"{r['speedup']:>8.4f}x | {r['num_preemptions']:>8} | {status:<8}"
        )
    lines.append("=" * 95)
    lines.append(f"Optimal Policy Discovered: {best_candidate} ({sorted_res[0]['speedup']:.4f}x goodput)")
    table_str = "\n".join(lines)

    if print_table:
        print("\n" + table_str + "\n")

    return {
        "comparison": sorted_res,
        "best_policy": best_candidate,
        "table": table_str,
    }


def sweep_policy(
    policy_factory: Callable[[Any], Union[str, BasePolicy, Dict[str, Any]]],
    param_name: str,
    values: Sequence[Any],
    workload: Union[str, List[Dict[str, Any]]] = "data/workloads/sharegpt_sampled_500.json",
    batch_size: int = 64,
    capacity_tokens: int = 120_000,
    print_table: bool = True,
) -> Dict[str, Any]:
    """
    Sweeps a single policy hyperparameter (e.g. concurrency cap, priority weight)
    and reports the optimal parameter setting.
    """
    sweep_results = []
    for val in values:
        pol_spec = policy_factory(val)
        res = test_policy(
            policy=pol_spec,
            workload=workload,
            batch_size=batch_size,
            capacity_tokens=capacity_tokens,
        )
        sweep_results.append({
            "param_value": val,
            "policy_name": res["policy_name"],
            "goodput_tok_per_s": res["goodput_tok_per_s"],
            "speedup": res["speedup"],
            "num_preemptions": res["num_preemptions"],
            "passed": res["passed"],
            "peak_memory_tokens": res["peak_memory_tokens"],
        })

    valid = [r for r in sweep_results if r["passed"]]
    best = max(valid, key=lambda x: x["goodput_tok_per_s"]) if valid else (sweep_results[0] if sweep_results else None)

    lines = [
        "=" * 85,
        f"POLICY PARAMETER SWEEP: '{param_name}'",
        "=" * 85,
        f"{param_name:<18} | {'Goodput (tok/s)':<18} | {'Speedup':<10} | {'Preempt':<8} | {'Status':<8}",
        "-" * 85,
    ]
    for r in sweep_results:
        status = "PASSED" if r["passed"] else "VIOLATED"
        lines.append(
            f"{str(r['param_value']):<18} | {r['goodput_tok_per_s']:>16,.1f} | "
            f"{r['speedup']:>8.4f}x | {r['num_preemptions']:>8} | {status:<8}"
        )
    lines.append("=" * 85)
    if best:
        lines.append(f"Optimal '{param_name}' = {best['param_value']} ({best['goodput_tok_per_s']:,.1f} tok/s, {best['speedup']:.4f}x)")
    table_str = "\n".join(lines)

    if print_table:
        print("\n" + table_str + "\n")

    return {
        "param_name": param_name,
        "results": sweep_results,
        "best_param_value": best["param_value"] if best else None,
        "best_result": best,
        "table": table_str,
    }


def quick_eval(
    policy: Union[str, BasePolicy, Dict[str, Any], Any] = "active",
    num_requests: int = 64,
    prompt_len: int = 256,
    decode_len: int = 2048,
    capacity_tokens: int = 120_000,
) -> Dict[str, Any]:
    """Rapid policy evaluator on a synthetic or small request pool for quick development test cycles."""
    synthetic_workload = [
        {"req_id": i, "prompt_len": prompt_len, "decode_len": decode_len}
        for i in range(num_requests)
    ]
    return test_policy(
        policy=policy,
        workload=synthetic_workload,
        batch_size=num_requests,
        capacity_tokens=capacity_tokens,
    )
