"""
Unified Benchmark Common Utilities.
Provides shared models, datasets, cluster managers, metrics, and reporting.
"""

from .model_config import (
    GEMMA_4_31B,
    LLAMA_3_70B,
    MODEL_REGISTRY,
    HardwareTopologyConfig,
    ModelArchitectureConfig,
    get_hardware_topology,
)
from .dataset import (
    generate_synthetic_prompts,
    load_sharegpt_prompts,
    sample_prompts_for_bucket,
)
from .cluster import (
    cleanup_stale_placement_groups,
    create_node_pinned_placement_group,
    init_ray_cluster,
)
from .metrics import (
    ServingBenchmarkMetrics,
    compute_percentiles,
    compute_serving_metrics,
)
from .reporting import (
    render_html_report,
    render_markdown_report,
)

__all__ = [
    "GEMMA_4_31B",
    "LLAMA_3_70B",
    "MODEL_REGISTRY",
    "HardwareTopologyConfig",
    "ModelArchitectureConfig",
    "get_hardware_topology",
    "generate_synthetic_prompts",
    "load_sharegpt_prompts",
    "sample_prompts_for_bucket",
    "cleanup_stale_placement_groups",
    "create_node_pinned_placement_group",
    "init_ray_cluster",
    "ServingBenchmarkMetrics",
    "compute_percentiles",
    "compute_serving_metrics",
    "render_html_report",
    "render_markdown_report",
]
