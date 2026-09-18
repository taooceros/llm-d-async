"""Dynamic Knapsack Admission Policy Module."""

import collections
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from admission_control.core.models import ActiveRequest, BatchCohort, Request, SystemState
from admission_control.algorithms.policies.base import BasePolicy
from admission_control.algorithms.policies.priority import (
    PRIORITY_REGISTRY,
    _extract_request_demands,
    slack_packed_lpt_priority_key,
)


def calculate_peak_future_memory_trajectory(
    active_requests: Sequence[Union[ActiveRequest, Request]],
    candidate_requests: Optional[Union[Request, ActiveRequest, Sequence[Union[Request, ActiveRequest]]]] = None,
    block_size: int = 1,
    capacity_tokens: Optional[int] = None,
) -> int:
    """Calculates exact peak future memory trajectory across future steps tau >= t."""
    def align(l: int) -> int:
        return l if block_size <= 1 else math.ceil(l / block_size) * block_size

    cands = (
        list(candidate_requests)
        if isinstance(candidate_requests, (list, tuple))
        else ([candidate_requests] if candidate_requests is not None else [])
    )
    all_reqs = list(active_requests) + cands
    if not all_reqs:
        return 0

    demands = [_extract_request_demands(r) for r in all_reqs]
    n_act = len(active_requests)

    if capacity_tokens is not None:
        curr_mem = sum(align(p + c) for p, _, c in demands[:n_act]) + sum(align(p) for p, _, _ in demands[n_act:])
        if curr_mem > capacity_tokens:
            return curr_mem
        total_lifetime = sum(align(p + d) for p, d, _ in demands)
        if total_lifetime <= capacity_tokens:
            return total_lifetime

    arr = np.array(demands, dtype=np.int64)
    p_arr, d_arr, c_arr = arr[:, 0], arr[:, 1], arr[:, 2]
    rem_arr = np.maximum(0, d_arr - c_arr)

    vec_align = (lambda a: a) if block_size <= 1 else (lambda a: ((a + block_size - 1) // block_size) * block_size)
    initial_mem = int(np.sum(vec_align(p_arr + c_arr)))

    positive_rems = rem_arr[rem_arr > 0]
    if len(positive_rems) == 0:
        return initial_mem

    horizons = np.unique(positive_rems)
    mask = rem_arr[:, None] >= horizons[None, :]
    aligned_active = vec_align(p_arr[:, None] + c_arr[:, None] + horizons[None, :])
    trajectory = np.sum(np.where(mask, aligned_active, 0), axis=0)
    return int(max(initial_mem, int(np.max(trajectory))))


def dynamic_knapsack_admit_guard(
    active_requests: Sequence[Union[ActiveRequest, Request, Any]],
    candidate: Union[Request, ActiveRequest, Any],
    capacity: int = 120_000,
    target_concurrency: int = 96,
    block_size: int = 1,
) -> bool:
    """Dynamic Knapsack Admission Guard for Aggressive Oversubscription."""
    if len(active_requests) >= target_concurrency:
        return False
    return calculate_peak_future_memory_trajectory(
        active_requests, candidate, block_size=block_size, capacity_tokens=capacity
    ) <= capacity


def schedule_dynamic_knapsack_batches(
    batches: List[List[Any]],
    target_concurrency: int = 96,
    capacity: int = 120_000,
    intra_batch_policy: str = "slack_packed_lpt",
    step_latency_s: float = 0.035,
    block_size: int = 1,
) -> Dict[str, Any]:
    """Continuous Multi-Batch Scheduler with Dynamic Knapsack Admission Control."""
    key_fn = PRIORITY_REGISTRY.get(intra_batch_policy.lower(), slack_packed_lpt_priority_key)
    queues: List[List[Dict[str, Any]]] = []
    total_reqs = 0
    for b_idx, batch in enumerate(batches):
        sorted_b = sorted(batch, key=lambda r: key_fn(r if isinstance(r, Request) else Request.from_dict(r)))
        batch_items = []
        for r_idx, r in enumerate(sorted_b):
            p, d, c = _extract_request_demands(r)
            r_id = r.req_id if isinstance(r, Request) else r.get("req_id", b_idx * 64 + r_idx)
            batch_items.append({"req_id": r_id, "prompt_len": p, "decode_len": d, "tokens_decoded": c, "batch_id": b_idx})
        queues.append(batch_items)
        total_reqs += len(batch_items)

    active, finished = [], []
    step, peak_mem, preemptions, throttled = 0, 0, 0, 0
    start_steps, admitted_per_batch = {}, collections.defaultdict(int)

    while len(finished) < total_reqs:
        while len(active) < target_concurrency:
            cand_batch_idx, candidate = next(((i, q[0]) for i, q in enumerate(queues) if q), (None, None))
            if candidate is None:
                break
            if dynamic_knapsack_admit_guard(active, candidate, capacity, target_concurrency, block_size):
                admitted = queues[cand_batch_idx].pop(0)
                active.append(admitted)
                start_steps[admitted["req_id"]] = step
                admitted_per_batch[cand_batch_idx] += 1
            else:
                throttled += 1
                break

        curr_mem = sum(r["prompt_len"] + r["tokens_decoded"] for r in active)
        peak_mem = max(peak_mem, curr_mem)
        if curr_mem > capacity:
            preemptions += 1

        step += 1
        still_active = []
        for r in active:
            r["tokens_decoded"] += 1
            (finished if r["tokens_decoded"] >= r["decode_len"] else still_active).append(r)
        active = still_active

    delivered_tokens = sum(r["decode_len"] for r in finished)
    total_time_s = step * step_latency_s
    goodput = round(delivered_tokens / total_time_s, 1) if total_time_s > 0 else 0.0

    return {
        "target_concurrency": target_concurrency,
        "capacity_tokens": capacity,
        "total_steps": step,
        "total_time_s": total_time_s,
        "delivered_tokens": delivered_tokens,
        "goodput_tok_per_s": goodput,
        "peak_observed_memory": peak_mem,
        "preemptions": preemptions,
        "throttled_events": throttled,
        "finished_requests": len(finished),
        "start_steps": start_steps,
        "admitted_per_batch": dict(admitted_per_batch),
    }


class DynamicKnapsackPolicy(BasePolicy):
    """Dynamic Knapsack Admission Control with Aggressive Concurrency Oversubscription."""

    def __init__(
        self,
        target_concurrency: int = 96,
        capacity: int = 120_000,
        intra_batch_policy: str = "slack_packed_lpt",
        slack_weight: float = 0.006,
        block_size: int = 16,
        **kwargs: Any,
    ):
        super().__init__(name=f"Dynamic Knapsack Admission (Target={target_concurrency}, Cap={capacity})")
        self.target_concurrency = target_concurrency
        self.capacity = capacity
        self.intra_batch_policy = intra_batch_policy
        self.slack_weight = slack_weight
        self.block_size = block_size

    def order_batch(
        self, batch: Union[BatchCohort, Sequence[Request]], system_state: Optional[SystemState] = None
    ) -> Union[BatchCohort, List[Request]]:
        reqs = batch.requests if isinstance(batch, BatchCohort) else batch
        if self.intra_batch_policy == "slack_packed_lpt":
            key_fn = lambda r: slack_packed_lpt_priority_key(r, weight=self.slack_weight)
        else:
            key_fn = PRIORITY_REGISTRY.get(self.intra_batch_policy.lower(), slack_packed_lpt_priority_key)
        ordered = sorted(reqs, key=lambda r: key_fn(r if isinstance(r, Request) else Request.from_dict(r)))
        return batch.reorder(ordered) if isinstance(batch, BatchCohort) else list(ordered)

    def can_admit(
        self, active_requests: Sequence[ActiveRequest], candidate: Request, system_state: Optional[SystemState] = None
    ) -> bool:
        cap = system_state.capacity_tokens if system_state and system_state.capacity_tokens > 0 else self.capacity
        block_sz = system_state.block_size if system_state and system_state.block_size > 0 else self.block_size
        return dynamic_knapsack_admit_guard(
            active_requests=active_requests,
            candidate=candidate,
            capacity=cap,
            target_concurrency=self.target_concurrency,
            block_size=block_sz,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "target_concurrency": self.target_concurrency,
            "capacity": self.capacity,
            "intra_batch_policy": self.intra_batch_policy,
            "slack_weight": self.slack_weight,
            "block_size": self.block_size,
        }
