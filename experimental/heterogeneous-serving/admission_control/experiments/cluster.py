"""
Cluster Execution Submitter for RCM Distributed Experiments.
============================================================
Handles Ray Job Submission, node-pinned topology locking, log streaming,
and reverse control trap monitoring for remote Cloud TPU cluster execution.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Optional

from omp_rcm.core import (
    DiagnosticTrap,
    JointTimingLogEngine,
    StateStore,
    SteeringReceiver,
)
from omp_rcm.ray_job import RayJobSupervisor
from admission_control.experiments.config import ExperimentConfig


def submit_cluster_job_rcm(
    cfg: ExperimentConfig,
    state: StateStore,
    timing: JointTimingLogEngine,
) -> Dict[str, Any]:
    """Submits and streams a benchmark job on Cloud TPU cluster with RCM monitoring via RayJobSupervisor."""
    ray_url = cfg.ray_url

    # Initialize RCM RayJobSupervisor
    supervisor = RayJobSupervisor(
        ray_url=ray_url,
        timing_engine=timing,
        trap=DiagnosticTrap(),
        steering=SteeringReceiver(),
    )
    state.set("ray_supervisor", supervisor)

    # Resolve project root
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent

    # Prepare runtime environment
    runtime_env = {
        "working_dir": str(project_root),
        "excludes": [
            ".venv*",
            "tpu-inference",
            "vllm",
            ".git",
            ".jj",
            "build",
            "dist",
            "formalization*",
            "*.lake*",
            "skaffold*",
            "*.olean*",
            "*.tar*",
        ],
        "env_vars": {
            "TPU_MULTIHOST_BACKEND": "ray",
            "SCAN_TPU_CHIPS": "True",
            "TPU_ACCELERATOR_TYPE": "v5e",
            "JAX_PLATFORMS": "",
            "JAX_COMPILATION_CACHE_DIR": "/models/jax_cache",
            "PYTHONUNBUFFERED": "1",
        },
    }

    # Build entrypoint CLI arguments
    runner_args = [f"--policy {cfg.policy}"]
    runner_args.append(f"--target-concurrency {cfg.target_concurrency}")
    if cfg.force_reinit:
        runner_args.append("--force-reinit")
    if cfg.max_duration_s and cfg.max_duration_s > 0:
        runner_args.append(f"--duration-s {cfg.max_duration_s}")
    if isinstance(cfg.workload, (str, pathlib.Path)):
        runner_args.append(f"--workload-json {cfg.workload}")

    entrypoint = f"python3 experiments/tp8_preemption_benchmark/cluster_runner.py {' '.join(runner_args)}"

    # Submit and begin native RCM supervision
    submission_id = supervisor.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        metadata={"policy": cfg.policy, "target_concurrency": cfg.target_concurrency},
    )
    if supervisor.active_job:
        state.set_ray_job(supervisor.active_job.to_dict())
    # Supervise until completion with RCM trap interception, event forwarding, and in-flight steering
    sup_result = supervisor.supervise_until_completion(
        poll_interval_s=1.5,
        stdout_stream=True,
        stage_name="execute_policy",
        state_store=state,
    )
    full_output = sup_result.get("full_output", "")
    parsed = sup_result.get("measurements", {})
    if not parsed:
        json_match = re.search(
            r"EMPIRICAL TPU HARDWARE MEASUREMENTS \(JSON OUTPUT\)\n=+\n(\{.*?\})\n\n\[OMP_EVENT",
            full_output,
            re.DOTALL,
        )
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except Exception:
                parsed = {}

    meas = parsed.get("empirical_measurements", {})
    return {
        "candidate": meas.get(f"policy_{cfg.policy}", meas.get("policy_oracle_optimal", {})),
        "baseline": meas.get("policy_a_greedy", None),
        "all_measurements": meas,
        "full_output": full_output,
    }
