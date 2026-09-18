"""
Performance & Roofline Metrics Calculation Utilities.
Calculates TTFT, TPOT, Throughput, TFLOPS, Model FLOPs Utilization (MFU),
and percentile variance statistics.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from .model_config import ModelArchitectureConfig, HardwareTopologyConfig


@dataclasses.dataclass
class ServingBenchmarkMetrics:
    total_requests: int
    total_tokens: int
    total_duration_s: float
    throughput_tokens_per_sec: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p99_ms: float
    tflops_achieved: float
    mfu_percent: float
    memory_bandwidth_gb_per_sec: float
    hbm_efficiency_percent: float

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_duration_s": round(self.total_duration_s, 3),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
            "ttft_mean_ms": round(self.ttft_mean_ms, 2),
            "ttft_p50_ms": round(self.ttft_p50_ms, 2),
            "ttft_p90_ms": round(self.ttft_p90_ms, 2),
            "ttft_p99_ms": round(self.ttft_p99_ms, 2),
            "tpot_mean_ms": round(self.tpot_mean_ms, 2),
            "tpot_p50_ms": round(self.tpot_p50_ms, 2),
            "tpot_p90_ms": round(self.tpot_p90_ms, 2),
            "tpot_p99_ms": round(self.tpot_p99_ms, 2),
            "tflops_achieved": round(self.tflops_achieved, 2),
            "mfu_percent": round(self.mfu_percent, 2),
            "memory_bandwidth_gb_per_sec": round(self.memory_bandwidth_gb_per_sec, 2),
            "hbm_efficiency_percent": round(self.hbm_efficiency_percent, 2),
        }


def compute_percentiles(latencies_ms: Sequence[float]) -> Tuple[float, float, float, float]:
    """Returns (mean, p50, p90, p99) from latency array in ms."""
    if not latencies_ms:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(latencies_ms, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
        float(np.percentile(arr, 99)),
    )


def compute_serving_metrics(
    ttft_latencies_ms: Sequence[float],
    tpot_latencies_ms: Sequence[float],
    total_tokens: int,
    total_duration_s: float,
    model_config: ModelArchitectureConfig,
    hardware_config: HardwareTopologyConfig,
    batch_size: int = 1,
) -> ServingBenchmarkMetrics:
    """Computes comprehensive serving performance, roofline, and MFU metrics."""
    total_requests = len(ttft_latencies_ms)
    throughput = float(total_tokens / total_duration_s) if total_duration_s > 0 else 0.0

    ttft_mean, ttft_p50, ttft_p90, ttft_p99 = compute_percentiles(ttft_latencies_ms)
    tpot_mean, tpot_p50, tpot_p90, tpot_p99 = compute_percentiles(tpot_latencies_ms)

    # Theoretical TFLOPS = (throughput * FLOPs_per_token) / 1e12
    tflops_achieved = (throughput * model_config.flops_per_token) / 1e12
    peak_tflops = hardware_config.peak_tflops
    mfu_percent = (tflops_achieved / peak_tflops * 100.0) if peak_tflops > 0 else 0.0

    # Memory bandwidth roofline during decode:
    # Model weights (loaded per step) + KV cache per step
    # Decode step bandwidth = (Weight_Bytes + Batch_Size * KV_Bytes_per_token) / (TPOT_s)
    tpot_sec = (tpot_mean / 1000.0) if tpot_mean > 0 else 1e-6
    bytes_per_decode_step = model_config.weight_bytes_bf16 + (batch_size * model_config.kv_cache_bytes_per_token())
    mem_bandwidth_gb_s = (bytes_per_decode_step / tpot_sec) / 1e9
    peak_mem_bandwidth = hardware_config.memory_bandwidth_gb_per_sec
    hbm_efficiency = (mem_bandwidth_gb_s / peak_mem_bandwidth * 100.0) if peak_mem_bandwidth > 0 else 0.0

    return ServingBenchmarkMetrics(
        total_requests=total_requests,
        total_tokens=total_tokens,
        total_duration_s=total_duration_s,
        throughput_tokens_per_sec=throughput,
        ttft_mean_ms=ttft_mean,
        ttft_p50_ms=ttft_p50,
        ttft_p90_ms=ttft_p90,
        ttft_p99_ms=ttft_p99,
        tpot_mean_ms=tpot_mean,
        tpot_p50_ms=tpot_p50,
        tpot_p90_ms=tpot_p90,
        tpot_p99_ms=tpot_p99,
        tflops_achieved=tflops_achieved,
        mfu_percent=mfu_percent,
        memory_bandwidth_gb_per_sec=mem_bandwidth_gb_s,
        hbm_efficiency_percent=hbm_efficiency,
    )
