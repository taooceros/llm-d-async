"""
Admission Guards & Victim Selection Module.
===========================================
Defines admission decision records, composable admission guards, `Guard`,
`BatchTransition`, and `Victim` namespaces for stream gating and preemption victim selection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from admission_control.core.models import (
    SystemState,
    AdmissionDecision,
    Request,
    ActiveRequest,
)
from admission_control.algorithms.policies.priority import _extract_request_demands, PriorityRule
from admission_control.algorithms.policies.knapsack import calculate_peak_future_memory_trajectory



class AdmissionGuard(ABC):
    """Abstract composable admission guard for evaluating candidate requests."""
    name: str = "AdmissionGuard"

    def evaluate(
        self,
        active_requests: List[Any],
        candidate: Any,
        system_state: Optional[SystemState] = None,
    ) -> AdmissionDecision:
        """Evaluates admission decision and returns structured AdmissionDecision with diagnostics."""
        admitted = self.can_admit(active_requests, candidate, system_state)
        return AdmissionDecision(
            admitted=admitted,
            guard_name=self.name,
            reason="Admitted" if admitted else f"Rejected by {self.name}",
        )

    @abstractmethod
    def can_admit(
        self,
        active_requests: List[Any],
        candidate: Any,
        system_state: Optional[SystemState] = None,
    ) -> bool:
        pass

    def __and__(self, other: Any) -> Any:
        if isinstance(other, AdmissionGuard):
            return AllOfGuard([self, other])
        return NotImplemented

    def __or__(self, other: Any) -> Any:
        if isinstance(other, AdmissionGuard):
            return AnyOfGuard([self, other])
        return NotImplemented
    def __invert__(self) -> "AdmissionGuard":
        return NotGuard(self)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def to_dict(self) -> Dict[str, Any]:
        return {"guard_class": self.__class__.__name__, "name": self.name}


class FunctionalAdmissionGuard(AdmissionGuard):
    def __init__(self, fn: Callable[[List[Any], Any, Optional[SystemState]], bool], name: str = "custom_guard"):
        self.fn = fn
        self.name = name

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        return self.fn(active_requests, candidate, system_state)


def _guard_complexity_rank(guard: "AdmissionGuard") -> int:
    """Computes computational complexity rank for optimal guard evaluation ordering."""
    cls_name = guard.__class__.__name__
    if cls_name in ("ConcurrencyCapGuard", "AdaptiveConcurrencyGuard", "PromptLengthGuard", "DecodeLengthGuard"):
        return 1
    elif cls_name in ("RateLimitGuard", "PrefillTokenBudgetGuard"):
        return 2
    elif cls_name == "MemoryHeadroomGuard":
        return 3
    elif cls_name in ("PeakMemoryGuard", "DynamicKnapsackGuard"):
        return 10
    elif cls_name in ("AllOfGuard", "AnyOfGuard", "KOfMGuard", "ConditionalGuard"):
        return 20
    return 5


class AllOfGuard(AdmissionGuard):
    def __init__(self, guards: List[AdmissionGuard], auto_optimize_order: bool = True):
        flattened: List[AdmissionGuard] = []
        for g in guards:
            if isinstance(g, AllOfGuard):
                flattened.extend(g.guards)
            else:
                flattened.append(g)
        if auto_optimize_order:
            self.guards = sorted(flattened, key=_guard_complexity_rank)
        else:
            self.guards = flattened
        self.name = " & ".join(g.name for g in self.guards)

    def evaluate(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> AdmissionDecision:
        for g in self.guards:
            dec = g.evaluate(active_requests, candidate, system_state) if hasattr(g, "evaluate") else AdmissionDecision(g.can_admit(active_requests, candidate, system_state), g.name)
            if not dec.admitted:
                return dec
        return AdmissionDecision(True, self.name, "All guards passed")

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        return self.evaluate(active_requests, candidate, system_state).admitted

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "all_of", "guards": [g.to_dict() for g in self.guards], "name": self.name}


class AnyOfGuard(AdmissionGuard):
    def __init__(self, guards: List[AdmissionGuard]):
        self.guards = guards
        self.name = " | ".join(g.name for g in guards)

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        return any(g.can_admit(active_requests, candidate, system_state) for g in self.guards)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "any_of", "guards": [g.to_dict() for g in self.guards], "name": self.name}


class NotGuard(AdmissionGuard):
    def __init__(self, guard: AdmissionGuard):
        self.guard = guard
        self.name = f"~({guard.name})"

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        return not self.guard.can_admit(active_requests, candidate, system_state)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "not", "guard": self.guard.to_dict(), "name": self.name}


class ConcurrencyCapGuard(AdmissionGuard):
    def __init__(self, cap: int):
        self.cap = cap
        self.name = f"ConcurrencyCap({cap})"

    def evaluate(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> AdmissionDecision:
        n = len(active_requests)
        if n >= self.cap:
            return AdmissionDecision(
                admitted=False,
                guard_name=self.name,
                reason=f"Active streams {n} reached concurrency cap {self.cap}",
                details={"active_count": n, "cap": self.cap},
            )
        return AdmissionDecision(
            admitted=True,
            guard_name=self.name,
            reason=f"Active streams {n} below concurrency cap {self.cap}",
            details={"active_count": n, "cap": self.cap},
        )

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        return self.evaluate(active_requests, candidate, system_state).admitted

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "concurrency_cap", "cap": self.cap, "name": self.name}


class PeakMemoryGuard(AdmissionGuard):
    def __init__(self, capacity: int = 120_000, block_size: int = 16):
        self.capacity = capacity
        self.block_size = block_size
        self.name = f"PeakMemoryGuard({capacity})"

    def evaluate(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> AdmissionDecision:
        cap = system_state.capacity_tokens if system_state else self.capacity
        peak = calculate_peak_future_memory_trajectory(active_requests, candidate, block_size=self.block_size, capacity_tokens=cap)
        if peak > cap:
            return AdmissionDecision(
                admitted=False,
                guard_name=self.name,
                reason=f"Future peak memory trajectory {peak:,} tokens exceeds capacity {cap:,} tokens",
                details={"projected_peak_tokens": peak, "capacity_tokens": cap, "excess_tokens": peak - cap},
            )
        return AdmissionDecision(
            admitted=True,
            guard_name=self.name,
            reason=f"Future peak memory trajectory {peak:,} tokens within capacity {cap:,} tokens",
            details={"projected_peak_tokens": peak, "capacity_tokens": cap},
        )

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        return self.evaluate(active_requests, candidate, system_state).admitted

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "peak_memory", "capacity": self.capacity, "block_size": self.block_size, "name": self.name}


class MemoryHeadroomGuard(AdmissionGuard):
    def __init__(self, headroom_tokens: int = 5_000, capacity: int = 120_000):
        self.headroom_tokens = headroom_tokens
        self.capacity = capacity
        self.name = f"MemoryHeadroomGuard({headroom_tokens})"

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        cap = system_state.capacity_tokens if system_state else self.capacity
        p, d, c = _extract_request_demands(candidate)
        curr_mem = (
            system_state.used_tokens
            if system_state and system_state.used_tokens > 0
            else sum(_extract_request_demands(r)[0] + _extract_request_demands(r)[2] for r in active_requests)
        )
        return (curr_mem + p + self.headroom_tokens) <= cap

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "memory_headroom", "headroom_tokens": self.headroom_tokens, "capacity": self.capacity, "name": self.name}


class PrefillTokenBudgetGuard(AdmissionGuard):
    """
    Limits total prompt tokens admitted in a single scheduling step
    to ensure colocated prefill does not breach the hardware single-step token budget.
    """
    def __init__(self, max_tokens_per_step: int = 2048):
        self.max_tokens_per_step = max_tokens_per_step
        self.name = f"PrefillTokenBudget({max_tokens_per_step})"
        self._step_admitted_tokens = 0
        self._last_step = -1

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        curr_step = system_state.step if system_state else 0
        if curr_step != self._last_step:
            self._last_step = curr_step
            self._step_admitted_tokens = 0

        p, d, c = _extract_request_demands(candidate)
        if (self._step_admitted_tokens + p) > self.max_tokens_per_step:
            return False

        self._step_admitted_tokens += p
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "prefill_token_budget", "max_tokens_per_step": self.max_tokens_per_step, "name": self.name}


class RateLimitGuard(AdmissionGuard):
    """Limits the number of stream admissions permitted in a single step."""
    def __init__(self, max_admissions_per_step: int = 16):
        self.max_admissions_per_step = max_admissions_per_step
        self.name = f"RateLimit({max_admissions_per_step})"
        self._step_admitted_count = 0
        self._last_step = -1

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        curr_step = system_state.step if system_state else 0
        if curr_step != self._last_step:
            self._last_step = curr_step
            self._step_admitted_count = 0

        if self._step_admitted_count >= self.max_admissions_per_step:
            return False

        self._step_admitted_count += 1
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "rate_limit", "max_admissions_per_step": self.max_admissions_per_step, "name": self.name}


class KOfMGuard(AdmissionGuard):
    """
    Quorum / K-of-M threshold admission guard.
    Passes if at least k of the supplied m guards pass.
    """
    def __init__(self, guards: List[AdmissionGuard], k: int, name: Optional[str] = None):
        self.guards = guards
        self.k = k
        self.name = name or f"Quorum({k}/{len(guards)}: " + ", ".join(g.name for g in guards) + ")"

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        passed = sum(1 for g in self.guards if g.can_admit(active_requests, candidate, system_state))
        return passed >= self.k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "k_of_m",
            "k": self.k,
            "guards": [g.to_dict() for g in self.guards],
            "name": self.name,
        }


class ConditionalGuard(AdmissionGuard):
    """
    Dynamic conditional admission guard that switches between two guards
    based on a system state condition trigger.
    """
    def __init__(
        self,
        condition: Callable[[Optional[SystemState], Optional[List[Any]]], bool],
        true_guard: AdmissionGuard,
        false_guard: AdmissionGuard,
        name: Optional[str] = None,
    ):
        self.condition = condition
        self.true_guard = true_guard
        self.false_guard = false_guard
        self.name = name or f"ConditionalGuard({true_guard.name} IF cond ELSE {false_guard.name})"

    def can_admit(
        self,
        active_requests: List[Any],
        candidate: Any,
        system_state: Optional[SystemState] = None,
    ) -> bool:
        cond_val = self.condition(system_state, active_requests)
        active_guard = self.true_guard if cond_val else self.false_guard
        return active_guard.can_admit(active_requests, candidate, system_state)

    def evaluate(
        self,
        active_requests: List[Any],
        candidate: Any,
        system_state: Optional[SystemState] = None,
    ) -> AdmissionDecision:
        cond_val = self.condition(system_state, active_requests)
        active_guard = self.true_guard if cond_val else self.false_guard
        dec = active_guard.evaluate(active_requests, candidate, system_state)
        return AdmissionDecision(
            admitted=dec.admitted,
            guard_name=self.name,
            reason=f"[{'TRUE' if cond_val else 'FALSE'} branch: {active_guard.name}] {dec.reason}",
            details={"condition_result": cond_val, "branch_guard": active_guard.name, **dec.details},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "conditional",
            "true_guard": self.true_guard.to_dict(),
            "false_guard": self.false_guard.to_dict(),
            "name": self.name,
        }


class PromptLengthGuard(AdmissionGuard):
    """Rejects requests with prompt length exceeding a maximum budget."""
    def __init__(self, max_prompt_len: int = 4096):
        self.max_prompt_len = max_prompt_len
        self.name = f"PromptLengthGuard({max_prompt_len})"

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        p, _, _ = _extract_request_demands(candidate)
        return p <= self.max_prompt_len

    def evaluate(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> AdmissionDecision:
        p, _, _ = _extract_request_demands(candidate)
        admitted = p <= self.max_prompt_len
        return AdmissionDecision(
            admitted=admitted,
            guard_name=self.name,
            reason=f"Candidate prompt {p} {'within' if admitted else 'exceeds'} prompt limit {self.max_prompt_len}",
            details={"prompt_len": p, "max_prompt_len": self.max_prompt_len},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "prompt_length_limit", "max_prompt_len": self.max_prompt_len, "name": self.name}


class DecodeLengthGuard(AdmissionGuard):
    """Rejects requests with decode length exceeding a maximum budget."""
    def __init__(self, max_decode_len: int = 4096):
        self.max_decode_len = max_decode_len
        self.name = f"DecodeLengthGuard({max_decode_len})"

    def can_admit(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> bool:
        _, d, _ = _extract_request_demands(candidate)
        return d <= self.max_decode_len

    def evaluate(self, active_requests: List[Any], candidate: Any, system_state: Optional[SystemState] = None) -> AdmissionDecision:
        _, d, _ = _extract_request_demands(candidate)
        admitted = d <= self.max_decode_len
        return AdmissionDecision(
            admitted=admitted,
            guard_name=self.name,
            reason=f"Candidate decode {d} {'within' if admitted else 'exceeds'} decode limit {self.max_decode_len}",
            details={"decode_len": d, "max_decode_len": self.max_decode_len},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "decode_length_limit", "max_decode_len": self.max_decode_len, "name": self.name}


class AdaptiveConcurrencyGuard(AdmissionGuard):
    """
    Dynamic state-aware concurrency guard that adapts its active concurrency limit
    based on real-time KV cache memory pressure.
    """
    def __init__(
        self,
        high_cap: int = 128,
        throttled_cap: int = 103,
        memory_threshold: float = 0.85,
        capacity: int = 120_000,
        name: Optional[str] = None,
    ):
        self.high_cap = high_cap
        self.throttled_cap = throttled_cap
        self.memory_threshold = memory_threshold
        self.capacity = capacity
        self.name = name or f"AdaptiveConcurrency(high={high_cap}, throttled={throttled_cap} @ {memory_threshold:.0%})"

    def can_admit(
        self,
        active_requests: List[Any],
        candidate: Any,
        system_state: Optional[SystemState] = None,
    ) -> bool:
        cap = system_state.capacity_tokens if system_state else self.capacity
        used = (
            system_state.used_tokens
            if system_state and system_state.used_tokens > 0
            else sum(_extract_request_demands(r)[0] + _extract_request_demands(r)[2] for r in active_requests)
        )
        util = used / cap if cap > 0 else 0.0
        target = self.throttled_cap if util >= self.memory_threshold else self.high_cap
        return len(active_requests) < target

    def evaluate(
        self,
        active_requests: List[Any],
        candidate: Any,
        system_state: Optional[SystemState] = None,
    ) -> AdmissionDecision:
        cap = system_state.capacity_tokens if system_state else self.capacity
        used = (
            system_state.used_tokens
            if system_state and system_state.used_tokens > 0
            else sum(_extract_request_demands(r)[0] + _extract_request_demands(r)[2] for r in active_requests)
        )
        util = used / cap if cap > 0 else 0.0
        target = self.throttled_cap if util >= self.memory_threshold else self.high_cap
        current = len(active_requests)
        admitted = current < target
        return AdmissionDecision(
            admitted=admitted,
            guard_name=self.name,
            reason=(
                f"Active concurrency {current} within adaptive cap {target} (utilization: {util:.1%})"
                if admitted
                else f"Active concurrency {current} reached adaptive cap {target} (utilization: {util:.1%})"
            ),
            details={
                "current_active": current,
                "target_cap": target,
                "high_cap": self.high_cap,
                "throttled_cap": self.throttled_cap,
                "memory_utilization": util,
                "memory_threshold": self.memory_threshold,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "adaptive_concurrency",
            "high_cap": self.high_cap,
            "throttled_cap": self.throttled_cap,
            "memory_threshold": self.memory_threshold,
            "capacity": self.capacity,
            "name": self.name,
        }


class Guard:
    """Factory namespace for constructing admission guards."""
    @staticmethod
    def concurrency_cap(cap: int) -> AdmissionGuard:
        return ConcurrencyCapGuard(cap)

    @staticmethod
    def peak_future_memory(capacity: int = 120_000, block_size: int = 16) -> AdmissionGuard:
        return PeakMemoryGuard(capacity=capacity, block_size=block_size)

    @staticmethod
    def memory_headroom(headroom_tokens: int = 5_000, capacity: int = 120_000) -> AdmissionGuard:
        return MemoryHeadroomGuard(headroom_tokens=headroom_tokens, capacity=capacity)

    @staticmethod
    def preemption_immune(capacity: int = 120_000, block_size: int = 16) -> AdmissionGuard:
        return PeakMemoryGuard(capacity=capacity, block_size=block_size)

    @staticmethod
    def custom(fn: Callable[[List[Any], Any, Optional[SystemState]], bool], name: str = "custom") -> AdmissionGuard:
        return FunctionalAdmissionGuard(fn, name)

    @staticmethod
    def prefill_token_budget(max_tokens_per_step: int = 2048) -> AdmissionGuard:
        return PrefillTokenBudgetGuard(max_tokens_per_step)

    @staticmethod
    def rate_limit(max_admissions_per_step: int = 16) -> AdmissionGuard:
        return RateLimitGuard(max_admissions_per_step)

    @staticmethod
    def k_of_m(guards: List[AdmissionGuard], k: int, name: Optional[str] = None) -> AdmissionGuard:
        return KOfMGuard(guards, k, name=name)

    @staticmethod
    def conditional(
        condition: Callable[[Optional[SystemState], Optional[List[Any]]], bool],
        true_guard: AdmissionGuard,
        false_guard: AdmissionGuard,
        name: Optional[str] = None,
    ) -> AdmissionGuard:
        return ConditionalGuard(condition, true_guard, false_guard, name=name)

    @staticmethod
    def prompt_length_limit(max_prompt_len: int = 4096) -> AdmissionGuard:
        return PromptLengthGuard(max_prompt_len)

    @staticmethod
    def decode_length_limit(max_decode_len: int = 4096) -> AdmissionGuard:
        return DecodeLengthGuard(max_decode_len)

    @staticmethod
    def adaptive_concurrency(
        high_cap: int = 128,
        throttled_cap: int = 103,
        memory_threshold: float = 0.85,
        capacity: int = 120_000,
        name: Optional[str] = None,
    ) -> AdmissionGuard:
        return AdaptiveConcurrencyGuard(
            high_cap=high_cap,
            throttled_cap=throttled_cap,
            memory_threshold=memory_threshold,
            capacity=capacity,
            name=name,
        )


class BatchTransition:
    """Factory namespace for constructing inter-batch transition rules."""
    @staticmethod
    def strict_barrier() -> AdmissionGuard:
        return FunctionalAdmissionGuard(lambda active, cand, state: len(active) == 0, "StrictBatchBarrier")

    @staticmethod
    def preemption_immune(capacity: int = 120_000, target_concurrency: int = 104, block_size: int = 16) -> AdmissionGuard:
        return ConcurrencyCapGuard(target_concurrency) & PeakMemoryGuard(capacity=capacity, block_size=block_size)

    @staticmethod
    def custom(fn: Callable[[List[Any], Any, Optional[SystemState]], bool], name: str = "custom_transition") -> AdmissionGuard:
        return FunctionalAdmissionGuard(fn, name)


class Victim:
    """Factory namespace for victim selection functions."""
    @staticmethod
    def lifo() -> Callable[[List[Any], Optional[SystemState]], Optional[Any]]:
        return lambda active, state: active[-1] if active else None

    @staticmethod
    def least_progress() -> Callable[[List[Any], Optional[SystemState]], Optional[Any]]:
        def _victim(active, state):
            if not active:
                return None
            return min(active, key=lambda r: _extract_request_demands(r)[2])
        return _victim

    @staticmethod
    def max_yield() -> Callable[[List[Any], Optional[SystemState]], Optional[Any]]:
        """Selects request with most generated tokens (frees maximum contiguous memory)."""
        def _victim(active, state):
            if not active:
                return None
            return max(active, key=lambda r: _extract_request_demands(r)[2])
        return _victim

    @staticmethod
    def srpt() -> Callable[[List[Any], Optional[SystemState]], Optional[Any]]:
        """Selects request with largest remaining decode (protects near-completion streams)."""
        def _victim(active, state):
            if not active:
                return None
            return max(active, key=lambda r: (_extract_request_demands(r)[1] - _extract_request_demands(r)[2]))
        return _victim

    @staticmethod
    def lowest_priority(rule: PriorityRule) -> Callable[[List[Any], Optional[SystemState]], Optional[Any]]:
        def _victim(active, state):
            if not active:
                return None
            return max(active, key=lambda r: rule.key(r, state))
        return _victim
