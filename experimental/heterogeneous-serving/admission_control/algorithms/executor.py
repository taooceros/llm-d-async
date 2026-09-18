"""
Unified Policy Execution Engine & Admission Controller.
======================================================
Provides a decoupled execution abstraction that drives scheduling policies
identically across discrete simulation and physical Cloud TPU hardware.
"""

import time
import math
import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional, Union, Callable

from admission_control.core.models import (
    Request,
    HardwareSpec,
    ActiveRequest,
    SystemState,
)
from admission_control.core.constants import (
    TP8_DEFAULT_KV_CAPACITY_TOKENS,
    DEFAULT_BLOCK_SIZE,
)
from admission_control.simulation.block_manager import BlockManager
from admission_control.algorithms.policies import (
    BasePolicy,
    DynamicKnapsackPolicy,
    get_policy,
    _extract_request_demands,
)


@dataclass
class StepOutput:
    """Output metrics collected from a single decode step."""
    step: int
    active_count: int
    used_tokens: int
    step_latency_s: float
    admitted_req_ids: List[Any] = field(default_factory=list)
    finished_req_ids: List[Any] = field(default_factory=list)
    preempted_req_ids: List[Any] = field(default_factory=list)
    tokens_generated: int = 0
    discarded_tokens: int = 0
    raw_outputs: Optional[List[Any]] = None

@dataclass
class StepRecord:
    """Detailed telemetry record for a single scheduling step."""
    step: int
    active_count: int
    used_tokens: int
    capacity_tokens: int
    bucket: int
    step_latency_s: float
    admitted_req_ids: List[Any] = field(default_factory=list)
    finished_req_ids: List[Any] = field(default_factory=list)
    throttled_guard: Optional[str] = None
    throttled_reason: Optional[str] = None


@dataclass
class ExecutionTrace:
    """
    Complete telemetry trace of a policy execution session.
    Provides export to Google Chrome Trace Event format (Perfetto / chrome://tracing),
    structured timelines, and step-by-step diagnostic telemetry.
    """
    policy_name: str
    steps: List[StepRecord] = field(default_factory=list)
    requests: List[Any] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_chrome_trace(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Exports execution trace to Google Chrome Trace Event format.
        Compatible directly with https://ui.perfetto.dev and chrome://tracing.
        """
        events = [
            {
                "name": "process_name",
                "ph": "M",
                "pid": 1,
                "args": {"name": f"Policy: {self.policy_name}"},
            }
        ]
        for s in self.steps:
            ts_us = s.step * 1000 # virtual microseconds
            events.append({
                "name": "KV Cache Tokens",
                "ph": "C",
                "ts": ts_us,
                "pid": 1,
                "args": {"used_tokens": s.used_tokens, "capacity": s.capacity_tokens},
            })
            events.append({
                "name": "Active Concurrency",
                "ph": "C",
                "ts": ts_us,
                "pid": 1,
                "args": {"active_streams": s.active_count, "bucket": s.bucket},
            })
            if s.throttled_guard:
                events.append({
                    "name": f"Throttled: {s.throttled_guard}",
                    "cat": "Admission",
                    "ph": "i",
                    "ts": ts_us,
                    "s": "p",
                    "pid": 1,
                    "args": {"reason": s.throttled_reason or ""},
                })
        for req in self.requests:
            admit = getattr(req, "admit_step", None)
            finish = getattr(req, "finish_step", None)
            if admit is not None and finish is not None:
                b_id = getattr(req, "batch_id", 0)
                r_id = getattr(req, "req_id", 0)
                events.append({
                    "name": f"Req {r_id} (B{b_id})",
                    "cat": f"Batch_{b_id}",
                    "ph": "X",
                    "ts": admit * 1000,
                    "dur": max(1000, (finish - admit) * 1000),
                    "pid": 1,
                    "tid": b_id,
                    "args": {
                        "prompt_len": getattr(req, "prompt_len", 0),
                        "decode_len": getattr(req, "decode_len", 0),
                        "tokens_decoded": getattr(req, "tokens_decoded", 0),
                        "preemption_count": getattr(req, "preemption_count", 0),
                    },
                })
        if file_path:
            import json
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2)
        return events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total_steps": len(self.steps),
            "total_requests": len(self.requests),
            "summary": self.summary,
        }


class ExecutionEngine(ABC):
    """Abstract interface for request execution environments (simulation or physical engine)."""

    @abstractmethod
    def add_request(self, request: Any) -> None:
        """Admits and registers a request into the execution environment."""
        pass

    @abstractmethod
    def step(self) -> StepOutput:
        """Executes one decode iteration across all currently active requests."""
        pass

    @abstractmethod
    def get_active_requests(self) -> List[ActiveRequest]:
        """Returns the list of currently active (running) requests."""
        pass

    @abstractmethod
    def has_unfinished_requests(self) -> bool:
        """Returns True if any requests are currently running or waiting inside the engine."""
        pass

    @abstractmethod
    def preempt_request(self, req_id: Any) -> int:
        """Evicts a request from memory upon capacity breach. Returns discarded tokens."""
        pass

    @abstractmethod
    def get_used_tokens(self) -> int:
        """Returns the current number of KV cache tokens consumed."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets engine state and clears all KV cache allocations."""
        pass


class SimulatedExecutionEngine(ExecutionEngine):
    """
    Step-accurate discrete event execution engine for local simulation.
    Uses BlockManager to enforce exact PagedAttention block allocation and preemption.
    """

    def __init__(
        self,
        capacity_tokens: int = TP8_DEFAULT_KV_CAPACITY_TOKENS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        step_latency_s: float = 0.035,
        hw: Optional[HardwareSpec] = None,
    ):
        self.capacity_tokens = capacity_tokens
        self.block_size = block_size
        self.step_latency_s = step_latency_s
        self.hw = hw or HardwareSpec(capacity_tokens=capacity_tokens)
        self.block_manager = BlockManager(capacity_tokens, block_size)

        self.step_num = 0
        self.active_requests: Dict[Any, ActiveRequest] = {}
        self.finished_requests: List[ActiveRequest] = []
        self.pending_admission: collections.deque = collections.deque()
        self.total_preemptions = 0
        self.total_discarded_tokens = 0
        self._victim_selector: Optional[Callable[[List[Any], Optional[SystemState]], Optional[Any]]] = None

    def set_victim_selector(self, selector: Callable[[List[Any], Optional[SystemState]], Optional[Any]]) -> None:
        """Configures the victim selection strategy for capacity breach preemptions."""
        self._victim_selector = selector

    def add_request(self, request: Any) -> None:
        p, d, c = _extract_request_demands(request)
        r_id = getattr(request, "req_id", request.get("req_id", len(self.active_requests)) if isinstance(request, dict) else id(request))
        b_id = getattr(request, "batch_id", request.get("batch_id", 0) if isinstance(request, dict) else 0)
        arr_step = getattr(request, "arrival_step", request.get("arrival_step", getattr(request, "arrival_time", self.step_num)) if isinstance(request, dict) else self.step_num)
        act = ActiveRequest(
            req_id=r_id,
            prompt_len=p,
            decode_len=d,
            tokens_decoded=c,
            batch_id=b_id,
            arrival_step=arr_step,
            admit_step=self.step_num,
        )

        # Allocate initial prompt blocks
        prompt_blocks = math.ceil(p / self.block_size)
        if self.block_manager.can_allocate(prompt_blocks):
            self.block_manager.allocate(r_id, prompt_blocks)
            self.active_requests[r_id] = act
        else:
            # Staged in pending admission
            self.pending_admission.append(act)

    def get_active_requests(self) -> List[ActiveRequest]:
        return list(self.active_requests.values())

    def has_unfinished_requests(self) -> bool:
        return len(self.active_requests) > 0 or len(self.pending_admission) > 0

    def get_used_tokens(self) -> int:
        return self.block_manager.used_tokens

    def preempt_request(self, req_id: Any) -> int:
        if req_id not in self.active_requests:
            return 0
        act = self.active_requests.pop(req_id)
        discarded = act.tokens_decoded
        self.block_manager.free(req_id)
        act.state = "PREEMPTED"
        act.preemption_count += 1
        act.discarded_tokens += discarded
        act.tokens_decoded = 0
        self.total_preemptions += 1
        self.total_discarded_tokens += discarded
        # Re-queue for recomputation
        self.pending_admission.appendleft(act)
        return discarded

    def step(self) -> StepOutput:
        admitted = []
        finished = []
        preempted = []
        tokens_generated = 0
        discarded = 0

        # Try to admit any pending requests that were waiting on memory
        while self.pending_admission:
            cand = self.pending_admission[0]
            needed_blocks = math.ceil((cand.prompt_len + cand.tokens_decoded) / self.block_size)
            if self.block_manager.can_allocate(needed_blocks):
                self.pending_admission.popleft()
                self.block_manager.allocate(cand.req_id, needed_blocks)
                self.active_requests[cand.req_id] = cand
                admitted.append(cand.req_id)
            else:
                break

        # 1. Decode step for active requests
        still_running = {}
        for r_id, req in list(self.active_requests.items()):
            curr_len = req.prompt_len + req.tokens_decoded
            next_len = curr_len + 1
            curr_b = math.ceil(curr_len / self.block_size)
            next_b = math.ceil(next_len / self.block_size)
            needed = next_b - curr_b

            if needed > 0:
                if self.block_manager.can_allocate(needed):
                    self.block_manager.allocate(r_id, needed)
                else:
                    # Cannot allocate -> capacity cliff triggered, handled in preemption below
                    pass

            req.tokens_decoded += 1
            tokens_generated += 1

            if req.tokens_decoded >= req.decode_len:
                req.state = "FINISHED"
                req.finish_step = self.step_num
                self.block_manager.free(r_id)
                self.finished_requests.append(req)
                finished.append(r_id)
            else:
                still_running[r_id] = req

        self.active_requests = still_running

        # 2. Resolve capacity breach via preemption (LIFO victim by default)
        while self.block_manager.free_blocks < 0:
            if not self.active_requests:
                break
            if self._victim_selector is not None:
                victim = self._victim_selector(list(self.active_requests.values()), None)
                victim_id = getattr(victim, "req_id", victim.get("req_id") if isinstance(victim, dict) else list(self.active_requests.keys())[-1])
            else:
                victim_id = list(self.active_requests.keys())[-1]
            disc = self.preempt_request(victim_id)
            preempted.append(victim_id)
            discarded += disc

        # Step latency calculation from hardware bucket
        active_cnt = len(self.active_requests)
        lat = self.hw.step_latency(active_cnt) if self.hw else self.step_latency_s

        out = StepOutput(
            step=self.step_num,
            active_count=active_cnt,
            used_tokens=self.get_used_tokens(),
            step_latency_s=lat,
            admitted_req_ids=admitted,
            finished_req_ids=finished,
            preempted_req_ids=preempted,
            tokens_generated=tokens_generated,
            discarded_tokens=discarded,
        )
        self.step_num += 1
        return out

    def reset(self) -> None:
        self.block_manager.free_all()
        self.active_requests.clear()
        self.finished_requests.clear()
        self.pending_admission.clear()
        self.step_num = 0
        self.total_preemptions = 0
        self.total_discarded_tokens = 0


class VLLMExecutionEngine(ExecutionEngine):
    """
    Physical hardware execution engine wrapping live vLLM `LLMEngine`.
    Drives requests step-by-step using `llm_engine.add_request()` and `llm_engine.step()`.
    """

    def __init__(self, llm_engine: Any, tracker: Optional[Any] = None, hw: Optional[HardwareSpec] = None):
        self.engine = llm_engine
        self.tracker = tracker
        self.hw = hw or HardwareSpec(capacity_tokens=TP8_DEFAULT_KV_CAPACITY_TOKENS)

        self.step_num = 0
        self.active_requests: Dict[Any, ActiveRequest] = {}
        self.finished_requests: List[ActiveRequest] = []
        self._sampling_params_cache: Dict[Any, Any] = {}

    def add_request(self, request: Any) -> None:
        from vllm import SamplingParams

        p, d, c = _extract_request_demands(request)
        r_id = getattr(request, "req_id", request.get("req_id", len(self.active_requests)) if isinstance(request, dict) else id(request))
        b_id = getattr(request, "batch_id", request.get("batch_id", 0) if isinstance(request, dict) else 0)

        # Clamp prompt to prevent context length breach
        max_safe_p = max(16, 4096 - d - 32)
        p = min(p, max_safe_p)

        prompt_token_ids = [100 + (i % 256) for i in range(p)]
        prompt_dict = {"prompt_token_ids": prompt_token_ids}
        sp = SamplingParams(max_tokens=d, ignore_eos=True, temperature=0.0)

        str_id = str(r_id)
        self._sampling_params_cache[str_id] = sp
        self.engine.add_request(str_id, prompt_dict, sp)

        act = ActiveRequest(
            req_id=r_id,
            prompt_len=p,
            decode_len=d,
            tokens_decoded=0,
            batch_id=b_id,
            arrival_step=self.step_num,
            admit_step=self.step_num,
        )
        self.active_requests[str_id] = act

    def get_active_requests(self) -> List[ActiveRequest]:
        return list(self.active_requests.values())

    def has_unfinished_requests(self) -> bool:
        return self.engine.has_unfinished_requests()

    def get_used_tokens(self) -> int:
        # Sum of active request tokens plus cached blocks if available
        tokens = sum(r.prompt_len + r.tokens_decoded for r in self.active_requests.values())
        return tokens

    def preempt_request(self, req_id: Any) -> int:
        str_id = str(req_id)
        if str_id in self.active_requests:
            act = self.active_requests.pop(str_id)
            discarded = act.tokens_decoded
            act.state = "PREEMPTED"
            act.discarded_tokens += discarded
            try:
                self.engine.abort_request(str_id)
            except Exception:
                pass
            return discarded
        return 0

    def step(self) -> StepOutput:
        t0 = time.perf_counter()
        outputs = self.engine.step()
        step_lat = time.perf_counter() - t0

        finished = []
        tokens_generated = 0

        for req_out in outputs:
            r_id_str = str(req_out.request_id)
            if r_id_str in self.active_requests:
                act = self.active_requests[r_id_str]
                num_out = len(req_out.outputs[0].token_ids) if req_out.outputs else 0
                new_tokens = max(0, num_out - act.tokens_decoded)
                act.tokens_decoded = num_out
                tokens_generated += new_tokens

                if req_out.finished:
                    act.state = "FINISHED"
                    act.finish_step = self.step_num
                    self.finished_requests.append(act)
                    finished.append(act.req_id)
                    del self.active_requests[r_id_str]

        # Query tracker for empirical preemption telemetry
        tracker_stats = self.tracker.get_stats(self.engine) if self.tracker else {}
        preempt_cnt = tracker_stats.get("num_preemptions", 0)
        discarded_tok = tracker_stats.get("total_tokens_wasted", 0)

        out = StepOutput(
            step=self.step_num,
            active_count=len(self.active_requests),
            used_tokens=self.get_used_tokens(),
            step_latency_s=step_lat,
            finished_req_ids=finished,
            tokens_generated=tokens_generated,
            discarded_tokens=discarded_tok,
            raw_outputs=outputs,
        )
        self.step_num += 1
        return out

    def reset(self) -> None:
        self.active_requests.clear()
        self.finished_requests.clear()
        self._sampling_params_cache.clear()
        self.step_num = 0
        if self.tracker:
            self.tracker.reset()


class PolicyExecutor:
    """
    Decoupled Policy Executor & Continuous Admission Controller.
    Orchestrates execution of an arbitrary scheduling policy over a workload trace.
    """

    def __init__(
        self,
        policy: Union[str, BasePolicy, Any],
        engine: ExecutionEngine,
        hw: Optional[HardwareSpec] = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        **policy_kwargs: Any,
    ):
        self.policy = get_policy(policy, **policy_kwargs)
        self.engine = engine
        self.hw = hw or HardwareSpec(capacity_tokens=TP8_DEFAULT_KV_CAPACITY_TOKENS)
        self.block_size = block_size
        if hasattr(self.engine, "set_victim_selector") and hasattr(self.policy, "select_preemption_victim"):
            self.engine.set_victim_selector(
                lambda active, state: self.policy.select_preemption_victim(active, state)
            )

    def execute(
        self,
        workload: Union[List[Request], List[List[Request]], List[Dict[str, Any]]],
        batch_size: int = 64,
        max_duration_s: Optional[float] = None,
        max_steps: Optional[int] = None,
        record_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes workload under the configured policy.
        Preserves inter-batch ordering, reordering only within batches of size `batch_size`.
        """
        t0 = time.perf_counter()

        # Partition workload into batches if not already partitioned
        if workload and isinstance(workload[0], list):
            batches = workload
        else:
            batches = [workload[i : i + batch_size] for i in range(0, len(workload), batch_size)]

        system_state = SystemState(
            step=0,
            current_time_s=0.0,
            capacity_tokens=self.hw.capacity_tokens,
            max_batch_size=self.hw.max_batch_size,
            block_size=self.block_size,
        )

        # Apply intra-batch reordering per policy (passing real-time system state)
        queues: List[List[Any]] = []
        total_requests = 0
        for b_idx, batch in enumerate(batches):
            ordered = self.policy.order_batch(batch, system_state)
            batch_items = []
            for r_idx, r in enumerate(ordered):
                p, d, c = _extract_request_demands(r)
                r_id = getattr(r, "req_id", r.get("req_id", b_idx * batch_size + r_idx) if isinstance(r, dict) else b_idx * batch_size + r_idx)
                arr = getattr(r, "arrival_step", getattr(r, "arrival_time", r.get("arrival_time", 0) if isinstance(r, dict) else 0))
                batch_items.append({
                    "req_id": r_id,
                    "prompt_len": p,
                    "decode_len": d,
                    "tokens_decoded": c,
                    "batch_id": b_idx,
                    "arrival_step": arr,
                })
            queues.append(batch_items)
            total_requests += len(batch_items)

        throttling_reasons = collections.Counter()

        step = 0
        total_delivered_tokens = 0
        throttled_events = 0
        admitted_per_batch: Dict[int, int] = collections.defaultdict(int)
        peak_observed_memory = 0
        step_active_profile = []
        step_memory_profile = []
        trace = ExecutionTrace(policy_name=getattr(self.policy, "name", "Policy")) if record_trace else None

        while any(len(q) > 0 for q in queues) or self.engine.has_unfinished_requests():
            elapsed = time.perf_counter() - t0
            if max_duration_s is not None and elapsed >= max_duration_s:
                break
            if max_steps is not None and step >= max_steps:
                break

            active = self.engine.get_active_requests()
            curr_mem = self.engine.get_used_tokens()
            if curr_mem > peak_observed_memory:
                peak_observed_memory = curr_mem

            system_state.step = step
            system_state.current_time_s = elapsed
            system_state.used_tokens = curr_mem

            # ----------------------------------------------------
            # 1. ADMISSION STAGE
            # ----------------------------------------------------
            step_admitted_ids: List[Any] = []
            step_throttled_guard: Optional[str] = None
            step_throttled_reason: Optional[str] = None
            while True:
                candidate = None
                cand_batch_idx = None
                for b_idx in range(len(queues)):
                    if queues[b_idx]:
                        candidate = queues[b_idx][0]
                        cand_batch_idx = b_idx
                        break

                if candidate is None:
                    break

                # Determine if candidate is drawn from next batch
                active_batches = {r.batch_id for r in active}
                earliest_active_batch = min(active_batches) if active_batches else cand_batch_idx
                system_state.current_batch_id = earliest_active_batch

                if cand_batch_idx > earliest_active_batch:
                    if hasattr(self.policy, "evaluate_batch_transition"):
                        dec = self.policy.evaluate_batch_transition(active, candidate, system_state)
                        can_admit = dec.admitted
                        guard_name = dec.guard_name
                    elif hasattr(self.policy, "evaluate_admission"):
                        dec = self.policy.evaluate_admission(active, candidate, system_state)
                        can_admit = dec.admitted
                        guard_name = dec.guard_name
                    else:
                        can_admit = self.policy.can_draw_next_batch(active, candidate, system_state)
                        guard_name = getattr(self.policy, "name", "Policy")
                else:
                    if hasattr(self.policy, "evaluate_admission"):
                        dec = self.policy.evaluate_admission(active, candidate, system_state)
                        can_admit = dec.admitted
                        guard_name = dec.guard_name
                    else:
                        can_admit = self.policy.can_admit(active, candidate, system_state)
                        guard_name = getattr(self.policy, "name", "Policy")
                if can_admit:
                    admitted = queues[cand_batch_idx].pop(0)
                    self.engine.add_request(admitted)
                    admitted_per_batch[cand_batch_idx] += 1
                    step_admitted_ids.append(admitted["req_id"])
                    active = self.engine.get_active_requests()
                    self.policy.on_admit(admitted, system_state)
                else:
                    step_throttled_guard = guard_name
                    step_throttled_reason = getattr(dec, "reason", "") if 'dec' in locals() and dec is not None else ""
                    throttled_events += 1
                    throttling_reasons[guard_name] += 1
                    break
            # ----------------------------------------------------
            # 2. STEP EXECUTION STAGE
            # ----------------------------------------------------
            step_out = self.engine.step()
            total_delivered_tokens += step_out.tokens_generated
            step_active_profile.append(step_out.active_count)
            step_memory_profile.append(step_out.used_tokens)
            for f_id in step_out.finished_req_ids:
                self.policy.on_finish(f_id, system_state)
            self.policy.on_step(step_out, system_state)
            if trace is not None:
                step_bucket = self.hw.get_bucket(step_out.active_count)
                step_lat = self.hw.step_latency(step_out.active_count)
                trace.steps.append(
                    StepRecord(
                        step=step,
                        active_count=step_out.active_count,
                        used_tokens=step_out.used_tokens,
                        capacity_tokens=self.hw.capacity_tokens,
                        bucket=step_bucket,
                        step_latency_s=step_lat,
                        admitted_req_ids=step_admitted_ids,
                        finished_req_ids=list(step_out.finished_req_ids),
                        throttled_guard=step_throttled_guard,
                        throttled_reason=step_throttled_reason,
                    )
                )
            step += 1

        wall_time_s = time.perf_counter() - t0
        if isinstance(self.engine, SimulatedExecutionEngine):
            # Hardware-simulated execution time across TPU static shape compilation buckets
            total_time_s = sum(self.hw.step_latency(n) for n in step_active_profile)
        else:
            total_time_s = wall_time_s

        goodput = (total_delivered_tokens / total_time_s) if total_time_s > 0 else 0.0
        # Calculate hardware padding waste
        total_active_slots = sum(step_active_profile)
        total_bucket_slots = sum(self.hw.get_bucket(n) for n in step_active_profile)
        padding_waste = (
            ((total_bucket_slots - total_active_slots) / total_bucket_slots * 100.0)
            if total_bucket_slots > 0
            else 0.0
        )

        finished_count = len(getattr(self.engine, "finished_requests", []))
        preemptions = getattr(self.engine, "total_preemptions", 0)
        discarded_tok = getattr(self.engine, "total_discarded_tokens", 0)

        finished_reqs = getattr(self.engine, "finished_requests", [])
        delays = [
            (r.admit_step - r.arrival_step)
            for r in finished_reqs
            if r.admit_step is not None and r.arrival_step is not None
        ]
        avg_queuing_delay = round(sum(delays) / len(delays), 2) if delays else 0.0
        max_queuing_delay = max(delays) if delays else 0
        peak_active = max(step_active_profile) if step_active_profile else 0

        res_payload = {
            "policy_name": getattr(self.policy, "name", "Policy"),
            "total_steps": step,
            "goodput_tok_per_s": goodput,
            "total_time_s": total_time_s,
            "wall_time_s": wall_time_s,
            "delivered_tokens": total_delivered_tokens,
            "finished_requests": finished_count,
            "total_requests": total_requests,
            "num_preemptions": preemptions,
            "tokens_discarded_recomputed": discarded_tok,
            "peak_memory_tokens": peak_observed_memory,
            "peak_memory_pct": round(peak_observed_memory / self.hw.capacity_tokens * 100.0, 2),
            "xla_padding_waste_percent": round(padding_waste, 2),
            "throttled_events": throttled_events,
            "throttling_breakdown": dict(throttling_reasons),
            "admitted_per_batch": dict(admitted_per_batch),
            "step_active_profile": step_active_profile,
            "step_memory_profile": step_memory_profile,
            "avg_queuing_delay_steps": avg_queuing_delay,
            "max_queuing_delay_steps": max_queuing_delay,
            "peak_active_concurrency": peak_active,
        }
        if trace is not None:
            trace.requests = list(getattr(self.engine, "finished_requests", []))
            trace.summary = {
                "goodput_tok_per_s": goodput,
                "total_time_s": total_time_s,
                "total_delivered_tokens": total_delivered_tokens,
                "num_preemptions": preemptions,
            }
            res_payload["trace"] = trace
        return res_payload


__all__ = [
    "StepOutput",
    "ExecutionEngine",
    "SimulatedExecutionEngine",
    "VLLMExecutionEngine",
    "PolicyExecutor",
    "StepRecord",
    "ExecutionTrace",
]
