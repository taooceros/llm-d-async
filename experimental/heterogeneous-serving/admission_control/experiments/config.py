"""
Experiment Configuration Data Model.
===================================
Defines declarative configuration for policy experiments, benchmarks, and sweeps.
"""

from __future__ import annotations

import os
import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


@dataclass
class ExperimentConfig:
    """Declarative specification for an admission control & scheduling experiment."""

    # Policy specification: registry name, file path (.yaml/.json/.py), dict, or BasePolicy instance
    policy: Union[str, Dict[str, Any], Any] = "active"

    # Workload specification: file path, workload name in data/workloads/, or list of request dicts
    workload: Union[str, List[Dict[str, Any]], Any] = "sharegpt_sampled_500.json"

    # Core hardware & batching parameters
    batch_size: int = 64
    capacity_tokens: int = 120_000
    block_size: int = 16
    max_batch_size: int = 128
    target_concurrency: int = 96

    # Stopping criteria
    max_duration_s: Optional[float] = None
    max_steps: Optional[int] = None

    # Execution environment: "simulation" (local step-accurate) or "cluster" (remote TPU Ray cluster)
    execution_mode: str = "simulation"

    # Baseline policy for relative speedup computation (None to disable baseline run)
    baseline_policy: Optional[str] = "greedy"

    # Output artifact locations
    output_json: Optional[str] = None
    output_report: Optional[str] = None
    generate_plots: bool = False
    figures_dir: Optional[str] = None
    record_trace: bool = False
    trace_output: Optional[str] = None
    # Cluster-specific parameters
    ray_url: str = "http://127.0.0.1:8265"
    force_reinit: bool = False
    silence_timeout_s: float = 600.0

    # Additional custom parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes config to dict (converting non-serializable objects to repr)."""
        d = asdict(self)
        if not isinstance(self.policy, (str, dict, list, int, float, bool, type(None))):
            d["policy"] = getattr(self.policy, "name", str(self.policy))
        if not isinstance(self.workload, (str, dict, list, int, float, bool, type(None))):
            d["workload"] = str(self.workload)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Constructs an ExperimentConfig from dict, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: Union[str, pathlib.Path]) -> ExperimentConfig:
        with open(json_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
