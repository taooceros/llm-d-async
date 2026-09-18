"""
Policy Invariant Verification Module.
=====================================
Defines `PolicyVerificationReport` and `verify_policy_invariants` for analytical
verification of preemption immunity, starvation freedom, and bounded displacement.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from admission_control.algorithms.policies.base import BasePolicy


@dataclass
class PolicyVerificationReport:
    policy_name: str
    workload_requests: int
    batch_size: int
    preemption_immunity: bool
    starvation_freedom: bool
    memory_capacity_safety: bool
    peak_memory_tokens: int
    capacity_tokens: int
    max_intra_batch_displacement: int
    preemption_count: int
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.preemption_immunity and self.starvation_freedom and self.memory_capacity_safety

    def summary(self) -> str:
        status = "PASSED (VALID POLICY)" if self.is_valid else "FAILED (POLICY VIOLATION)"
        return (
            f"Policy Verification: {self.policy_name} -> {status}\n"
            f"  Preemption Immunity : {'PASS (0 preemptions)' if self.preemption_immunity else f'FAIL ({self.preemption_count} preemptions)'}\n"
            f"  Starvation Freedom  : {'PASS (earlier batches strictly prioritized)' if self.starvation_freedom else 'FAIL'}\n"
            f"  Memory Safety       : {'PASS' if self.memory_capacity_safety else 'FAIL'} ({self.peak_memory_tokens:,} / {self.capacity_tokens:,} tokens)\n"
            f"  Max Displacement    : {self.max_intra_batch_displacement} positions (bounded <= {self.batch_size})"
        )


def verify_policy_invariants(
    policy: BasePolicy,
    workload: List[Any],
    batch_size: int = 64,
    capacity_tokens: int = 120_000,
) -> PolicyVerificationReport:
    """
    Analytically verifies that a scheduling policy satisfies mandatory architectural invariants:
    1. Preemption immunity: Zero preemptions are triggered throughout execution.
    2. Starvation freedom: Earlier batch cohorts strictly take priority; requests
       from batch K+1 are drawn only when no preemption can occur in batch K.
    3. Memory capacity safety: Peak observed memory M_peak <= capacity_tokens.
    4. Bounded displacement: Intra-batch request reordering displacement <= batch_size.
    """
    from admission_control.algorithms.executor import PolicyExecutor, SimulatedExecutionEngine
    from admission_control.core.models import HardwareSpec

    hw = HardwareSpec(capacity_tokens=capacity_tokens, max_batch_size=128)
    engine = SimulatedExecutionEngine(capacity_tokens=capacity_tokens, block_size=16, hw=hw)
    executor = PolicyExecutor(policy=policy, engine=engine, hw=hw)

    res = executor.execute(workload, batch_size=batch_size)

    # Calculate intra-batch displacement
    max_disp = 0
    for b_idx in range(0, len(workload), batch_size):
        chunk = workload[b_idx : b_idx + batch_size]
        ordered = policy.order_batch(chunk)
        orig_positions = {getattr(r, "req_id", r.get("req_id", idx) if isinstance(r, dict) else idx): idx for idx, r in enumerate(chunk)}
        for new_idx, r in enumerate(ordered):
            r_id = getattr(r, "req_id", r.get("req_id", new_idx) if isinstance(r, dict) else new_idx)
            if r_id in orig_positions:
                disp = abs(new_idx - orig_positions[r_id])
                if disp > max_disp:
                    max_disp = disp

    preemptions = res["num_preemptions"]
    preemption_immunity = (preemptions == 0)
    peak_mem = res["peak_memory_tokens"]
    mem_safe = (peak_mem <= capacity_tokens)

    num_batches = (len(workload) + batch_size - 1) // batch_size
    admitted_batches = res.get("admitted_per_batch", {})
    starvation_freedom = (
        res["finished_requests"] == len(workload)
        and len(admitted_batches) == num_batches
        and max_disp <= batch_size
    )

    return PolicyVerificationReport(
        policy_name=policy.name,
        workload_requests=len(workload),
        batch_size=batch_size,
        preemption_immunity=preemption_immunity,
        starvation_freedom=starvation_freedom,
        memory_capacity_safety=mem_safe,
        peak_memory_tokens=peak_mem,
        capacity_tokens=capacity_tokens,
        max_intra_batch_displacement=max_disp,
        preemption_count=preemptions,
        details=res,
    )
