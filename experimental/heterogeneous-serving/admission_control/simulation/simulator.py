"""
Step-Accurate vLLM V1 Scheduler and Preemption Dynamics Simulator.
==================================================================
Simulates PagedAttention block allocation, decode expansion, preemption cascades,
and discarded token recomputation with cohort-isolated batch ingestion and
deterministic analytical memory trajectory projection.
"""

import collections
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from admission_control.core.constants import (
    BATCH_COHORT_SIZE,
    DEFAULT_BLOCK_SIZE,
    TP8_DEFAULT_KV_CAPACITY_TOKENS,
)
from admission_control.core.models import (
    ActiveRequest,
    BatchCohort,
    HardwareSpec,
    Request,
    SystemState,
)
from admission_control.simulation.block_manager import BlockManager


@dataclass(frozen=True)
class ScheduleStep:
    """Immutable typed record capturing engine telemetry and request transitions at a discrete step."""
    step: int
    active_count: int
    memory_used: int
    xla_bucket: int
    padding_slots: int
    step_latency_s: float
    admitted_ids: Tuple[Any, ...] = field(default_factory=tuple)
    finished_ids: Tuple[Any, ...] = field(default_factory=tuple)
    preempted_ids: Tuple[Any, ...] = field(default_factory=tuple)
    waiting_count: int = 0
    preemptions: int = 0
    discarded_tokens: int = 0
    used_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step, "running": self.active_count, "active_count": self.active_count,
            "waiting": self.waiting_count, "waiting_count": self.waiting_count,
            "used_tokens": self.memory_used, "memory_used": self.memory_used, "used_pct": self.used_pct,
            "xla_bucket": self.xla_bucket, "padding_slots": self.padding_slots, "step_latency_s": self.step_latency_s,
            "admitted_ids": list(self.admitted_ids), "finished_ids": list(self.finished_ids),
            "preempted_ids": list(self.preempted_ids), "preemptions": self.preemptions,
            "discarded_tokens": self.discarded_tokens,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _extract_demands(req: Any, default_id: Any = 0) -> Tuple[Any, int, int, int, int]:
    """Extracts (req_id, prompt_len, decode_len, tokens_decoded, batch_id) from diverse request models."""
    if isinstance(req, dict):
        return (
            req.get("req_id", default_id),
            req.get("prompt_len", 0),
            req.get("decode_len", 0),
            req.get("tokens_decoded", req.get("generated_tokens", 0)),
            req.get("batch_id", 0),
        )
    return (
        getattr(req, "req_id", default_id),
        getattr(req, "prompt_len", 0),
        getattr(req, "decode_len", 0),
        getattr(req, "tokens_decoded", getattr(req, "generated_tokens", 0)),
        getattr(req, "batch_id", 0),
    )


def project_memory_trajectory(
    active_requests: Sequence[Union[ActiveRequest, Any]],
    pending_requests: Sequence[Union[Request, Any]],
    block_size: int = 16,
) -> List[int]:
    """Decoupled analytical memory trajectory projection equation."""
    parsed_reqs: List[Tuple[int, int, int]] = []
    for r in list(active_requests) + list(pending_requests):
        _, p, d, c, _ = _extract_demands(r)
        parsed_reqs.append((p, c, max(0, d - c)))

    if not parsed_reqs:
        return []

    tau_max = max((rem for _, _, rem in parsed_reqs), default=0)
    bs = max(1, block_size)

    if tau_max == 0:
        prompt_mem = sum(((p + bs - 1) // bs) * bs if bs > 1 else p for p, _, _ in parsed_reqs)
        return [prompt_mem] if prompt_mem > 0 else []

    trajectory = [0] * tau_max
    for p, c, rem in parsed_reqs:
        for tau in range(rem):
            curr_tokens = p + c + tau
            blocks = (curr_tokens + bs - 1) // bs if bs > 1 else curr_tokens
            trajectory[tau] += blocks * bs if bs > 1 else curr_tokens

    return trajectory


class SimRequest:
    """Represents a request inside the simulated vLLM scheduler."""

    def __init__(self, req_id: Any, prompt_len: int, decode_len: int, arrival_step: int = 0, batch_id: int = 0) -> None:
        self.req_id, self.prompt_len, self.decode_len = req_id, prompt_len, decode_len
        self.arrival_step, self.batch_id = arrival_step, batch_id
        self.state = "WAITING"
        self.generated_tokens = self.computed_tokens = self.preemption_count = 0
        self.discarded_decode_tokens = self.recomputed_prompt_tokens = 0
        self.preempted_tokens_saved = self.total_reprefill_tokens = self.paused_steps = 0
        self.admit_step: Optional[int] = None
        self.finish_step: Optional[int] = None

    def blocks_needed_for_len(self, token_len: int, block_size: int) -> int:
        return math.ceil(token_len / block_size)

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.decode_len

    @property
    def is_finished(self) -> bool:
        return self.generated_tokens >= self.decode_len

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id, "prompt_len": self.prompt_len, "decode_len": self.decode_len,
            "arrival_step": self.arrival_step, "batch_id": self.batch_id, "state": self.state,
            "generated_tokens": self.generated_tokens, "preemption_count": self.preemption_count,
            "discarded_decode_tokens": self.discarded_decode_tokens,
        }


class VLLMSchedulerSimulator:
    """
    Step-by-step simulator for vLLM Scheduler.
    Implements exact block allocation, preemption upon capacity breach, and recomputation.
    """

    def __init__(
        self,
        capacity_tokens: int = TP8_DEFAULT_KV_CAPACITY_TOKENS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_num_seqs: int = 64,
        max_num_batched_tokens: int = 4096,
        step_latency_ms: float = 7.5,
        preemption_mode: str = "reprefill",
        hw: Optional[HardwareSpec] = None,
        cohort_size: int = BATCH_COHORT_SIZE,
    ) -> None:
        self.capacity_tokens, self.block_size = capacity_tokens, block_size
        self.max_num_seqs, self.max_num_batched_tokens = max_num_seqs, max_num_batched_tokens
        self.step_latency_ms, self.preemption_mode, self.cohort_size = step_latency_ms, preemption_mode, cohort_size
        self.hw = hw or HardwareSpec(capacity_tokens=capacity_tokens, max_batch_size=max_num_seqs, block_size=block_size)
        self.bm: Optional[BlockManager] = None
        self.cohort_queues: List[collections.deque] = []
        self.running: List[SimRequest] = []
        self.finished: List[SimRequest] = []
        self.step_records: List[ScheduleStep] = []
        self.preemption_events: List[Dict[str, Any]] = []
        self.step_idx = self.total_preemptions = self.total_discarded_tokens = 0
        self.total_reprefill_tokens = self.total_suspensions = 0
        self._admission_guard_fn: Optional[Callable] = None
        self._can_draw_next_batch_fn: Optional[Callable] = None
        self._active_preemption_mode = preemption_mode
        self._enforce_preemption_immunity = True
        self._policy: Optional[Any] = None
    def reset(
        self,
        requests: Sequence[Union[SimRequest, Request, Dict[str, Any], BatchCohort]],
        admission_guard_fn: Optional[Callable] = None,
        can_draw_next_batch_fn: Optional[Callable] = None,
        preemption_mode: Optional[str] = None,
        enforce_preemption_immunity: bool = True,
        policy: Optional[Any] = None,
    ) -> None:
        """Initializes or resets simulator state with the supplied request trace."""
        self.bm = BlockManager(self.capacity_tokens, self.block_size)
        self.running, self.finished, self.step_records, self.preemption_events = [], [], [], []
        self.step_idx = self.total_preemptions = self.total_discarded_tokens = 0
        self.total_reprefill_tokens = self.total_suspensions = 0
        self._admission_guard_fn = admission_guard_fn
        self._can_draw_next_batch_fn = can_draw_next_batch_fn
        self._active_preemption_mode = preemption_mode or self.preemption_mode
        self._enforce_preemption_immunity = enforce_preemption_immunity
        self._policy = policy
        cohort_map: Dict[int, List[SimRequest]] = collections.defaultdict(list)
        for i, r in enumerate(requests):
            if isinstance(r, BatchCohort):
                for req_item in r:
                    cohort_map[r.batch_id].append(SimRequest(
                        req_id=req_item.req_id, prompt_len=req_item.prompt_len, decode_len=req_item.decode_len,
                        arrival_step=getattr(req_item, "arrival_time", getattr(req_item, "arrival_step", 0)),
                        batch_id=r.batch_id,
                    ))
            elif isinstance(r, SimRequest):
                r.batch_id = getattr(r, "batch_id", 0) or (i // self.cohort_size)
                cohort_map[r.batch_id].append(r)
            else:
                r_id, p, d, _, _ = _extract_demands(r, default_id=i)
                b_id = r.get("batch_id") if isinstance(r, dict) else getattr(r, "batch_id", None)
                if b_id is None:
                    b_id = i // self.cohort_size
                arr = r.get("arrival_step", r.get("arrival_time", 0)) if isinstance(r, dict) else getattr(r, "arrival_step", getattr(r, "arrival_time", 0))
                cohort_map[b_id].append(SimRequest(req_id=r_id, prompt_len=p, decode_len=d, arrival_step=arr, batch_id=b_id))

        self.cohort_queues = [collections.deque(cohort_map[b]) for b in range(max(cohort_map.keys(), default=-1) + 1)]

        if self._policy is not None and hasattr(self._policy, "order_batch"):
            for b_idx in range(len(self.cohort_queues)):
                q = self.cohort_queues[b_idx]
                if q:
                    sim_reqs = list(q)
                    batch_models = [Request(req_id=sr.req_id, prompt_len=sr.prompt_len, decode_len=sr.decode_len, arrival_time=sr.arrival_step) for sr in sim_reqs]
                    cohort = BatchCohort.from_requests(batch_models, batch_id=b_idx, arrival_step=sim_reqs[0].arrival_step)
                    ordered = self._policy.order_batch(cohort)
                    order_map = {sr.req_id: sr for sr in sim_reqs}
                    reordered_sim = [order_map[r.req_id] for r in ordered if r.req_id in order_map]
                    self.cohort_queues[b_idx] = collections.deque(reordered_sim)

    def has_unfinished(self) -> bool:
        """Predicate indicating whether requests remain in flight or pending admission."""
        return bool(self.running) or any(len(q) > 0 for q in self.cohort_queues)

    def step(self) -> ScheduleStep:
        """Advances simulation clock by 1 discrete step."""
        if self.bm is None:
            raise RuntimeError("Simulator has not been initialized. Call reset() or run() first.")

        bm, step, mode = self.bm, self.step_idx, self._active_preemption_mode
        step_preemptions = step_discarded = 0
        finished_this_step, preempted_this_step, admitted_this_step = [], [], []

        # 1. Decode step for running requests
        still_running: List[SimRequest] = []
        for req in self.running:
            if req.state == "SUSPENDED":
                req.paused_steps += 1
                still_running.append(req)
                continue

            curr_b = req.blocks_needed_for_len(req.prompt_len + req.generated_tokens, self.block_size)
            next_b = req.blocks_needed_for_len(req.prompt_len + req.generated_tokens + 1, self.block_size)
            needed = next_b - curr_b
            if needed > 0 and bm.can_allocate(needed):
                bm.allocate(req.req_id, needed)

            req.generated_tokens += 1
            req.computed_tokens = req.prompt_len + req.generated_tokens

            if req.generated_tokens >= req.decode_len:
                req.state, req.finish_step = "FINISHED", step
                bm.free(req.req_id)
                self.finished.append(req)
                finished_this_step.append(req.req_id)
            else:
                still_running.append(req)
        self.running = still_running

        # 2. Preemption resolution upon capacity breach
        while bm.free_blocks < 0 or sum(
            r.blocks_needed_for_len(r.prompt_len + r.generated_tokens, self.block_size)
            for r in self.running
        ) > bm.total_blocks:
            if not self.running:
                break

            if mode == "suspend":
                active_now = [r for r in self.running if r.state == "RUNNING"]
                if active_now:
                    active_now.sort(key=lambda r: (r.decode_len - r.generated_tokens), reverse=True)
                    victim = active_now[0]
                    victim.state, victim.preemption_count = "SUSPENDED", victim.preemption_count + 1
                    self.total_suspensions += 1
                    self.preemption_events.append({"step": step, "victim_id": victim.req_id, "action": "SUSPEND", "discarded_tokens": 0})
                    break

            victim = self.running.pop()
            victim.state, victim.preemption_count = "WAITING", victim.preemption_count + 1
            self.total_preemptions += 1
            step_preemptions += 1
            bm.free(victim.req_id)

            discarded = victim.generated_tokens
            victim.discarded_decode_tokens += discarded
            self.total_discarded_tokens += discarded
            step_discarded += discarded

            if mode == "reprefill":
                victim.preempted_tokens_saved += discarded
                cost = victim.prompt_len + discarded
                victim.total_reprefill_tokens += cost
                self.total_reprefill_tokens += cost
            else:
                victim.generated_tokens = victim.computed_tokens = 0

            preempted_this_step.append(victim.req_id)
            self.preemption_events.append({
                "step": step, "victim_id": victim.req_id,
                "action": "REPREFILL" if mode == "reprefill" else "REGENERATE",
                "discarded_tokens": discarded,
            })
            self.cohort_queues[victim.batch_id].appendleft(victim)
            if mode == "suspend":
                break

        # 3. Resume suspended streams if capacity permits
        if mode == "suspend":
            for req in self.running:
                if req.state == "SUSPENDED":
                    needed_b = req.blocks_needed_for_len(req.prompt_len + req.generated_tokens + 1, self.block_size)
                    curr_b = req.blocks_needed_for_len(req.prompt_len + req.generated_tokens, self.block_size)
                    if bm.free_blocks >= (needed_b - curr_b):
                        req.state = "RUNNING"

        # 4. Admission of waiting streams
        num_batched_tokens = 0
        while len(self.running) < self.max_num_seqs:
            cand_b_idx = next((i for i, q in enumerate(self.cohort_queues) if q), None)
            if cand_b_idx is None:
                break
            candidate = self.cohort_queues[cand_b_idx][0]

            active_batches = {r.batch_id for r in self.running}
            earliest_active_batch = min(active_batches) if active_batches else cand_b_idx

            if self._policy is not None:
                state = SystemState(
                    step=step,
                    capacity_tokens=self.capacity_tokens,
                    used_tokens=bm.used_tokens,
                    max_batch_size=self.max_num_seqs,
                    block_size=self.block_size,
                    current_batch_id=earliest_active_batch,
                )
                if cand_b_idx > earliest_active_batch:
                    if not self._policy.can_draw_next_batch(self.running, candidate, state):
                        break
                else:
                    if not self._policy.can_admit(self.running, candidate, state):
                        break
            else:
                if cand_b_idx > earliest_active_batch:
                    if self._can_draw_next_batch_fn is not None:
                        state = SystemState(
                            step=step,
                            capacity_tokens=self.capacity_tokens,
                            used_tokens=bm.used_tokens,
                            max_batch_size=self.max_num_seqs,
                            block_size=self.block_size,
                            current_batch_id=earliest_active_batch,
                        )
                        if not self._can_draw_next_batch_fn(self.running, candidate, state):
                            break
                    elif self._enforce_preemption_immunity:
                        traj = project_memory_trajectory(self.running, [candidate], block_size=self.block_size)
                        if traj and max(traj) > self.capacity_tokens:
                            break

                if self._admission_guard_fn and not self._admission_guard_fn(self.running, bm, candidate):
                    break

            needed_tokens = candidate.prompt_len + (candidate.generated_tokens if mode == "reprefill" else 0)
            if num_batched_tokens + needed_tokens > self.max_num_batched_tokens and admitted_this_step:
                break

            needed_blocks = candidate.blocks_needed_for_len(needed_tokens, self.block_size)
            if not bm.can_allocate(needed_blocks):
                break

            self.cohort_queues[cand_b_idx].popleft()
            bm.allocate(candidate.req_id, needed_blocks)
            candidate.state = "RUNNING"
            if candidate.admit_step is None:
                candidate.admit_step = step
            candidate.computed_tokens = candidate.prompt_len
            num_batched_tokens += needed_tokens
            self.running.append(candidate)
            admitted_this_step.append(candidate.req_id)

        active_cnt = len(self.running)
        sched_step = ScheduleStep(
            step=step,
            active_count=active_cnt,
            memory_used=bm.used_tokens,
            xla_bucket=self.hw.get_bucket(active_cnt),
            padding_slots=self.hw.padding_slots(active_cnt),
            step_latency_s=self.hw.step_latency(active_cnt) if self.hw else (self.step_latency_ms / 1000.0),
            admitted_ids=tuple(admitted_this_step),
            finished_ids=tuple(finished_this_step),
            preempted_ids=tuple(preempted_this_step),
            waiting_count=sum(len(q) for q in self.cohort_queues),
            preemptions=step_preemptions,
            discarded_tokens=step_discarded,
            used_pct=(bm.used_tokens / self.capacity_tokens * 100.0) if self.capacity_tokens > 0 else 0.0,
        )
        self.step_records.append(sched_step)
        self.step_idx += 1
        return sched_step

    def _build_results(self) -> Dict[str, Any]:
        """Constructs standardized summary results payload."""
        delivered = sum(r.decode_len for r in self.finished)
        sim_time = self.step_idx * (self.step_latency_ms / 1000.0)
        peak_tokens = max((s.memory_used for s in self.step_records), default=0)
        tot_disc = self.total_discarded_tokens
        return {
            "total_steps": self.step_idx,
            "total_time_s": sim_time,
            "delivered_tokens": delivered,
            "goodput_tok_per_s": (delivered / sim_time) if sim_time > 0 else 0.0,
            "total_preemptions": self.total_preemptions,
            "total_discarded_tokens": tot_disc,
            "total_reprefill_tokens": self.total_reprefill_tokens,
            "total_suspensions": self.total_suspensions,
            "preemption_mode": self._active_preemption_mode,
            "goodput_efficiency_pct": (delivered / (delivered + tot_disc) * 100.0) if (delivered + tot_disc) > 0 else 100.0,
            "preemption_events": self.preemption_events,
            "step_metrics": [s.to_dict() for s in self.step_records],
            "steps": self.step_records,
            "step_records": self.step_records,
            "peak_memory_tokens": peak_tokens,
            "peak_memory_pct": (peak_tokens / self.capacity_tokens * 100.0) if self.capacity_tokens > 0 else 0.0,
            "finished_requests": len(self.finished),
        }

    def run(
        self,
        requests: Sequence[Union[SimRequest, Request, Dict[str, Any], BatchCohort]],
        admission_guard_fn: Optional[Callable[[List[SimRequest], BlockManager, SimRequest], bool]] = None,
        can_draw_next_batch_fn: Optional[Callable[[List[SimRequest], SimRequest, SystemState], bool]] = None,
        preemption_mode: Optional[str] = None,
        enforce_preemption_immunity: bool = True,
    ) -> Dict[str, Any]:
        """Executes discrete step simulation loop until all requests finish."""
        self.reset(requests, admission_guard_fn, can_draw_next_batch_fn, preemption_mode, enforce_preemption_immunity)
        while self.has_unfinished():
            self.step()
        return self._build_results()

    def _simulate_core(
        self,
        requests: List[SimRequest],
        admission_guard_fn: Optional[Callable[[List[SimRequest], BlockManager, SimRequest], bool]] = None,
        preemption_mode: Optional[str] = None,
        enforce_preemption_immunity: bool = True,
    ) -> Dict[str, Any]:
        """Backward compatibility adapter for legacy _simulate_core callers."""
        return self.run(requests, admission_guard_fn, preemption_mode=preemption_mode, enforce_preemption_immunity=enforce_preemption_immunity)

    def simulate_policy_a_greedy(self, raw_requests: List[Dict[str, int]]) -> Dict[str, Any]:
        """Policy A: Default Greedy Reactive Preemption (watermark=0.0)."""
        res = self.run(raw_requests, enforce_preemption_immunity=False)
        res["policy_name"] = "Policy A: Greedy Reactive Preemption"
        return res

    def simulate_policy_b_predictive(self, raw_requests: List[Dict[str, int]], cap: int = 48) -> Dict[str, Any]:
        """Policy B: Predictive Admission Gating (cap active streams)."""
        res = self.run(raw_requests, admission_guard_fn=lambda running, bm, c: len(running) < cap)
        res["policy_name"] = f"Policy B: Predictive Gating (Cap={cap})"
        return res

    def simulate_policy(
        self,
        policy: Any,
        raw_requests: List[Dict[str, int]],
        batch_size: int = 64,
        max_duration_s: Optional[float] = None,
        max_steps: Optional[int] = None,
        **policy_kwargs: Any,
    ) -> Dict[str, Any]:
        """Simulates an arbitrary BasePolicy using unified PolicyExecutor and SimulatedExecutionEngine."""
        from admission_control.algorithms.executor import PolicyExecutor, SimulatedExecutionEngine
        hw = HardwareSpec(capacity_tokens=self.capacity_tokens, max_batch_size=self.max_num_seqs)
        engine = SimulatedExecutionEngine(
            capacity_tokens=self.capacity_tokens, block_size=self.block_size,
            step_latency_s=self.step_latency_ms / 1000.0, hw=hw,
        )
        executor = PolicyExecutor(policy=policy, engine=engine, hw=hw, block_size=self.block_size, **policy_kwargs)
        res = executor.execute(raw_requests, batch_size=batch_size, max_duration_s=max_duration_s, max_steps=max_steps)
        res["policy_name"] = res.get("policy_name", getattr(executor.policy, "name", "Custom Policy"))
        return res

    def simulate_all_policies(self, raw_requests: List[Dict[str, int]]) -> List[Dict[str, Any]]:
        """Simulates all comparative policies against the provided request trace."""
        return [
            self.simulate_policy_a_greedy(raw_requests),
            self.simulate_policy_b_predictive(raw_requests, cap=48),
            self.simulate_policy("knapsack", raw_requests, target_concurrency=96, capacity=self.capacity_tokens),
            self.simulate_policy("quantized_knapsack", raw_requests, capacity=self.capacity_tokens),
        ]
