"""
Shared Dataset & Prompt Sampling Utilities.
Loads ShareGPT JSON data or generates deterministic synthetic token sequences.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
from typing import Dict, List, Optional, Tuple

DEFAULT_DATASET_CANDIDATES = [
    pathlib.Path("benchmarks/gemma4_31b_tpu_v5e/data/sharegpt_sampled_500.json"),
    pathlib.Path("data/sharegpt_sampled_500.json"),
    pathlib.Path("data/ShareGPT_V3_unfiltered_cleaned_split.json"),
]


def load_sharegpt_prompts(
    dataset_path: Optional[str | pathlib.Path] = None,
    max_samples: int = 500,
    seed: int = 42,
) -> List[str]:
    """Loads human conversational prompts from ShareGPT JSON datasets."""
    rng = random.Random(seed)
    candidates = [pathlib.Path(dataset_path)] if dataset_path else DEFAULT_DATASET_CANDIDATES

    target_file: Optional[pathlib.Path] = None
    for cand in candidates:
        if cand and cand.exists():
            target_file = cand
            break

    if target_file is None:
        # Generate clean synthetic conversational prompts if file not found
        return generate_synthetic_prompts(count=max_samples, seed=seed)

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        for item in data:
            if isinstance(item, dict) and "conversations" in item:
                convs = item["conversations"]
                for turn in convs:
                    if turn.get("from") in ("human", "user") and len(turn.get("value", "").strip()) > 20:
                        prompts.append(turn["value"].strip())
            elif isinstance(item, str) and len(item.strip()) > 20:
                prompts.append(item.strip())

        if prompts:
            rng.shuffle(prompts)
            return prompts[:max_samples]

    except Exception:
        pass

    return generate_synthetic_prompts(count=max_samples, seed=seed)


def generate_synthetic_prompts(count: int = 100, seed: int = 42) -> List[str]:
    """Generates synthetic prompts with varying token lengths for benchmarking."""
    rng = random.Random(seed)
    base_topics = [
        "Explain the mathematical roofline model for distributed tensor parallelism on TPU v5e.",
        "Analyze the decoupled prefill and decode serving architecture for large language models.",
        "Write a detailed technical design for Megascale ICI interconnects across multihost TPU meshes.",
        "Compare speculative decoding algorithms with draft tokens vs paged attention KV caching.",
        "Describe XLA ahead-of-time graph compilation strategies for sparse mixture-of-experts models.",
    ]
    prompts = []
    for i in range(count):
        topic = rng.choice(base_topics)
        repeat = rng.randint(1, 15)
        prompts.append(f"Prompt #{i + 1}: " + " ".join([topic] * repeat))
    return prompts


def sample_prompts_for_bucket(
    prompts: List[str],
    target_token_len: int,
    batch_size: int,
    chars_per_token: float = 4.0,
    seed: int = 42,
) -> List[str]:
    """Selects or pads prompts to match the target token length bucket for a batch."""
    rng = random.Random(seed)
    target_chars = int(target_token_len * chars_per_token)
    sampled = []

    for _ in range(batch_size):
        base = rng.choice(prompts) if prompts else "Benchmark evaluation token sequence."
        if len(base) < target_chars:
            # Repeat to reach target length
            multiplier = (target_chars // len(base)) + 1
            extended = (base + " ") * multiplier
            sampled.append(extended[:target_chars])
        else:
            sampled.append(base[:target_chars])

    return sampled
