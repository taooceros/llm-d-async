"""
Shared Model Architecture & Hardware Physical Constants.
Supports Gemma-4 31B, LLaMA-3, and Cloud TPU hardware topologies.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass(frozen=True)
class ModelArchitectureConfig:
    name: str
    hidden_size: int
    intermediate_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int
    rms_norm_eps: float = 1e-6

    @property
    def flops_per_token(self) -> float:
        """
        Theoretical forward FLOPs per token:
        2 * (2 * hidden_size * (num_q_heads * head_dim + 2 * num_kv_heads * head_dim) +
             3 * hidden_size * intermediate_size) * num_layers
        """
        qkv_dense_flops = 2 * self.hidden_size * (self.num_q_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim)
        o_proj_flops = 2 * (self.num_q_heads * self.head_dim) * self.hidden_size
        mlp_flops = 2 * 3 * self.hidden_size * self.intermediate_size  # Gate, Up, Down projections
        return float((qkv_dense_flops + o_proj_flops + mlp_flops) * self.num_layers)

    @property
    def weight_bytes_bf16(self) -> float:
        """Estimated parameter size in bytes (BF16 = 2 bytes/param)."""
        # Linear layer weights + embeddings
        param_count = (
            self.num_layers * (
                self.hidden_size * (self.num_q_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim) +  # QKV
                (self.num_q_heads * self.head_dim) * self.hidden_size +  # O
                3 * self.hidden_size * self.intermediate_size  # MLP
            ) +
            self.vocab_size * self.hidden_size  # Embeddings
        )
        return float(param_count * 2.0)

    def kv_cache_bytes_per_token(self) -> float:
        """KV cache size in bytes per token across all layers: 2 * num_layers * num_kv_heads * head_dim * 2 (BF16)."""
        return float(2 * self.num_layers * self.num_kv_heads * self.head_dim * 2)


@dataclasses.dataclass(frozen=True)
class HardwareTopologyConfig:
    device_type: str  # e.g. "v5e", "v5p", "v4"
    num_chips: int
    tp_size: int
    mesh_shape: tuple[int, ...]
    peak_tflops_per_chip_bf16: float = 197.0  # TPU v5e peak BF16 TFLOPS
    hbm_per_chip_gb: float = 16.0  # TPU v5e has 16GB HBM per chip
    memory_bandwidth_gb_per_sec: float = 819.0  # Peak HBM bandwidth (v5e)
    ici_bandwidth_gb_per_sec: float = 400.0  # Inter-chip interconnect

    @property
    def peak_tflops(self) -> float:
        return self.num_chips * self.peak_tflops_per_chip_bf16

    @property
    def total_hbm_gb(self) -> float:
        return self.num_chips * self.hbm_per_chip_gb


# Predefined Standard Models
GEMMA_4_31B = ModelArchitectureConfig(
    name="gemma-4-31b",
    hidden_size=5376,
    intermediate_size=21504,
    num_q_heads=32,
    num_kv_heads=16,
    head_dim=128,
    num_layers=48,
    vocab_size=256000,
    rms_norm_eps=1e-6,
)

LLAMA_3_70B = ModelArchitectureConfig(
    name="llama-3-70b",
    hidden_size=8192,
    intermediate_size=28672,
    num_q_heads=64,
    num_kv_heads=8,
    head_dim=128,
    num_layers=80,
    vocab_size=128256,
    rms_norm_eps=1e-5,
)

MODEL_REGISTRY: Dict[str, ModelArchitectureConfig] = {
    "gemma-4-31b": GEMMA_4_31B,
    "gemma4-31b": GEMMA_4_31B,
    "llama-3-70b": LLAMA_3_70B,
    "llama3-70b": LLAMA_3_70B,
}


def get_hardware_topology(topology_name: str) -> HardwareTopologyConfig:
    """Returns hardware config for named topologies: '1x4' (TP=4), '2x4' (TP=8), '4x4' (TP=16)."""
    norm = topology_name.lower().replace(" ", "").replace("tpu_", "")
    if norm in ("1x4", "tp4", "single_host", "singlehost"):
        return HardwareTopologyConfig(
            device_type="v5e",
            num_chips=4,
            tp_size=4,
            mesh_shape=(1, 4),
        )
    elif norm in ("2x4", "tp8", "two_host", "twohost", "subslice"):
        return HardwareTopologyConfig(
            device_type="v5e",
            num_chips=8,
            tp_size=8,
            mesh_shape=(2, 4),
        )
    elif norm in ("4x4", "tp16", "multihost", "cluster"):
        return HardwareTopologyConfig(
            device_type="v5e",
            num_chips=16,
            tp_size=16,
            mesh_shape=(4, 4),
        )
    else:
        # Default to 2x4 (TP=8)
        return HardwareTopologyConfig(
            device_type="v5e",
            num_chips=8,
            tp_size=8,
            mesh_shape=(2, 4),
        )
