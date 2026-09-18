"""
Admission Control & Scheduling Policies Package.
================================================
Provides the generic, modular scheduling policy architecture, priority rules,
admission guards, builders, dynamic registry, and analytical verification routines.

Policy Modules:
- `base`: Abstract BasePolicy foundation
- `composable`: ComposablePolicy (the generic policy engine: priority + guards + transitions + victims)
- `knapsack`: Memory trajectory lookahead and dynamic knapsack guard
- `priority`: Composable PriorityRule hierarchy and Priority factory namespace
- `guards`: Composable AdmissionGuard hierarchy, Guard, BatchTransition, and Victim namespaces
- `builder`: PolicyBuilder fluent DSL
- `verification`: PolicyVerificationReport and verify_policy_invariants
- `registry`: Global policy registry, dynamic resolution, and active policy state
"""

# 1. Base & Composable Policy
from admission_control.algorithms.policies.base import BasePolicy
from admission_control.algorithms.policies.composable import ComposablePolicy
from admission_control.core.models import SystemState, Request, HardwareSpec, BatchCohort

# 2. Priority Rules & Keys
from admission_control.algorithms.policies.priority import (
    binned_lpt_priority_key,
    pure_lpt_priority_key,
    density_priority_key,
    spt_priority_key,
    slack_packed_lpt_priority_key,
    PRIORITY_REGISTRY,
    order_requests_by_policy,
    slice_into_windowed_batches,
    _extract_request_demands,
    PriorityRule,
    FunctionalPriorityRule,
    ChainedPriorityRule,
    ReversedPriorityRule,
    ScaledPriorityRule,
    WeightedSumPriorityRule,
    CompositePriorityRule,
    PrefixClusteringPriorityRule,
    AdaptiveMemoryPriorityRule,
    AgeBoostedPriorityRule,
    Priority,
)

# 3. Knapsack & Memory Trajectory
from admission_control.algorithms.policies.knapsack import (
    calculate_peak_future_memory_trajectory,
    dynamic_knapsack_admit_guard,
    schedule_dynamic_knapsack_batches,
    DynamicKnapsackPolicy,
)

# 4. Admission Guards, Decisions, Transitions & Victims
from admission_control.algorithms.policies.guards import (
    AdmissionDecision,
    AdmissionGuard,
    FunctionalAdmissionGuard,
    AllOfGuard,
    AnyOfGuard,
    NotGuard,
    ConcurrencyCapGuard,
    PeakMemoryGuard,
    MemoryHeadroomGuard,
    PrefillTokenBudgetGuard,
    RateLimitGuard,
    KOfMGuard,
    ConditionalGuard,
    PromptLengthGuard,
    DecodeLengthGuard,
    AdaptiveConcurrencyGuard,
    Guard,
    BatchTransition,
    Victim,
    _guard_complexity_rank,
)

# 5. Policy Builder
from admission_control.algorithms.policies.builder import PolicyBuilder

# 6. Verification
from admission_control.algorithms.policies.verification import (
    PolicyVerificationReport,
    verify_policy_invariants,
)

# 7. Registry & Dynamic Discovery
from admission_control.algorithms.policies.registry import (
    register_policy,
    list_registered_policies,
    get_policy,
    create_policy,
    get_active_policy,
    set_active_policy,
)

# 8. Continuous Surgical Preempt Policy
from admission_control.algorithms.policies.surgical_preempt import (
    ContinuousSurgicalPreemptPolicy,
    calculate_exact_trajectory,
)

__all__ = [
    # Priority keys & functions
    "binned_lpt_priority_key",
    "slack_packed_lpt_priority_key",
    "pure_lpt_priority_key",
    "density_priority_key",
    "spt_priority_key",
    "PRIORITY_REGISTRY",
    "order_requests_by_policy",
    "_extract_request_demands",
    "slice_into_windowed_batches",
    "calculate_peak_future_memory_trajectory",
    "dynamic_knapsack_admit_guard",
    "schedule_dynamic_knapsack_batches",
    # Policies
    "BasePolicy",
    "ComposablePolicy",
    "DynamicKnapsackPolicy",
    "ContinuousSurgicalPreemptPolicy",
    "calculate_exact_trajectory",
    "SystemState",
    "Request",
    "HardwareSpec",
    "BatchCohort",
    # Priority rules & factory
    "PriorityRule",
    "FunctionalPriorityRule",
    "ChainedPriorityRule",
    "ReversedPriorityRule",
    "ScaledPriorityRule",
    "WeightedSumPriorityRule",
    "CompositePriorityRule",
    "PrefixClusteringPriorityRule",
    "AdaptiveMemoryPriorityRule",
    "AgeBoostedPriorityRule",
    "Priority",
    # Admission guards & factory
    "AdmissionDecision",
    "AdmissionGuard",
    "FunctionalAdmissionGuard",
    "AllOfGuard",
    "AnyOfGuard",
    "NotGuard",
    "ConcurrencyCapGuard",
    "PeakMemoryGuard",
    "MemoryHeadroomGuard",
    "PrefillTokenBudgetGuard",
    "RateLimitGuard",
    "KOfMGuard",
    "ConditionalGuard",
    "PromptLengthGuard",
    "DecodeLengthGuard",
    "AdaptiveConcurrencyGuard",
    "Guard",
    "BatchTransition",
    "Victim",
    # Builder
    "PolicyBuilder",
    # Verification
    "_guard_complexity_rank",
    "PolicyVerificationReport",
    "verify_policy_invariants",
    # Registry & Factories
    "get_policy",
    "create_policy",
    "get_active_policy",
    "set_active_policy",
    "register_policy",
    "list_registered_policies",
]
