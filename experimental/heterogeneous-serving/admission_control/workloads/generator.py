"""
Workload Synthesizer: Generates realistic LLM batch benchmark request traces.
=============================================================================
Samples from empirical token distributions (p0, p50, p90, p99, p100) or
parametric distributions to synthesize heterogeneous workloads.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from admission_control.core.models import Request


def sample_empirical_distribution(
    percentiles: List[float],
    values: List[float],
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Samples request token lengths using piecewise linear interpolation across quantiles."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(percentiles[0], percentiles[-1], n_samples)
    return np.interp(u, percentiles, values).astype(int)


def generate_synthetic_workload(
    num_requests: int = 64,
    min_prompt: int = 256,
    max_prompt: int = 1536,
    min_decode: int = 256,
    max_decode: int = 1024,
    seed: int = 42,
) -> List[Request]:
    """Generates a uniform or bounded random workload trace."""
    rng = np.random.default_rng(seed)
    prompts = rng.integers(min_prompt, max_prompt + 1, size=num_requests)
    decodes = rng.integers(min_decode, max_decode + 1, size=num_requests)

    return [
        Request(req_id=i, prompt_len=int(p), decode_len=int(d))
        for i, (p, d) in enumerate(zip(prompts, decodes))
    ]


def generate_distribution_workload(
    profile: Dict[str, float],
    num_requests: int = 64,
    min_decode: int = 256,
    max_decode: int = 1024,
    normalize_median: Optional[int] = None,
    seed: int = 42,
) -> List[Request]:
    """Generates requests matching quantile profile distributions."""
    percentiles = [0.0, 50.0, 90.0, 99.0, 100.0]
    raw_values = [profile["p0"], profile["p50"], profile["p90"], profile["p99"], profile["p100"]]

    if normalize_median and profile.get("p50", 0) > 0:
        scale = normalize_median / profile["p50"]
        values = [max(16, int(v * scale)) for v in raw_values]
    else:
        values = raw_values

    prompts = sample_empirical_distribution(percentiles, values, num_requests, seed=seed)
    rng = np.random.default_rng(seed + 1)
    decodes = rng.integers(min_decode, max_decode + 1, size=num_requests)

    return [
        Request(req_id=i, prompt_len=int(p), decode_len=int(d))
        for i, (p, d) in enumerate(zip(prompts, decodes))
    ]
