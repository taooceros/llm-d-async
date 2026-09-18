"""
Admission Control & Clairvoyant Oracle Scheduling for vLLM on Cloud TPUs.
========================================================================
A production-grade Python package providing:
- Core data models for requests, physical hardware limits, and schedule plans
- Clairvoyant mathematical oracle profile solvers (asymmetric staggering, water-filling, binned LPT)
- Step-accurate vLLM V1 scheduler and memory trajectory simulator
- TPU mesh topology discovery and deterministic node-pinned placement group management
- Persistent hot in-HBM Ray benchmark worker actor with zero-weight reload overhead
- Ray Job Submission SDK client and cluster benchmark orchestrator
- Empirical trace loaders, validators, and distribution-based workload synthesizers
"""

__version__ = "0.2.0"

from admission_control.core.models import (
    Request,
    HardwareSpec,
    SchedulePlan,
    TokenProfile,
    ActiveRequest,
    SystemState,
)
from admission_control.core.constants import (
    MODEL_NAME,
    MODEL_PATH,
    TP8_DEFAULT_KV_CAPACITY_TOKENS,
)
from admission_control.algorithms.solver import (
    simulate_schedule_plan,
    solve_asymmetric_staggering,
    solve_profile_water_filling,
    solve_schedule,
    run_comparative_suite,
    solve_optimal_composable_policy,
    PolicyComparisonMatrix,
    compare_policies,
    CapacityScalingReport,
    analyze_policy_capacity_scaling,
)
from admission_control.algorithms.policies import (
    binned_lpt_priority_key,
    slack_packed_lpt_priority_key,
    pure_lpt_priority_key,
    order_requests_by_policy,
    slice_into_windowed_batches,
    calculate_peak_future_memory_trajectory,
    dynamic_knapsack_admit_guard,
    schedule_dynamic_knapsack_batches,
    BasePolicy,
    DynamicKnapsackPolicy,
    get_policy,
    get_active_policy,
    set_active_policy,
    PriorityRule,
    Priority,
    AdmissionGuard,
    Guard,
    BatchTransition,
    Victim,
    ComposablePolicy,
    PolicyBuilder,
    PolicyVerificationReport,
    verify_policy_invariants,
)
from admission_control.algorithms.executor import (
    StepOutput,
    ExecutionEngine,
    SimulatedExecutionEngine,
    VLLMExecutionEngine,
    PolicyExecutor,
)
from admission_control.simulation.simulator import (
    SimRequest,
    VLLMSchedulerSimulator,
)
from admission_control.simulation.block_manager import (
    BlockManager,
)
from admission_control.tpu.topology import (
    TPUHostInfo,
    ConnectedPair,
    TPUTopologyDiscovery,
)
from admission_control.tpu.placement import (
    cleanup_stale_placement_groups,
    create_node_pinned_placement_group,
    validate_tp8_v5e_topology,
)
from admission_control.tpu.xla_profiler import (
    XLAProfiler,
    XLACompilerPhaseStats,
    HLOInstructionProfile,
)
from admission_control.serving.worker import (
    PreemptionTelemetryTracker,
    TPUPreemptionBenchmarkWorker,
)
from admission_control.serving.orchestrator import (
    run_benchmark_orchestrator,
    submit_orchestrator_job,
)
from admission_control.serving.client import (
    TPUJobSubmitter,
    TPUServingClient,
    StructuredEvent,
    emit_omp_event,
)
from admission_control.workloads.loader import (
    load_workload_requests,
    save_workload_requests,
    get_workload_path,
    get_workload_profile,
)
from admission_control.workloads.generator import (
    generate_synthetic_workload,
    generate_distribution_workload,
)

__all__ = [
    "__version__",
    "Request",
    "HardwareSpec",
    "SchedulePlan",
    "TokenProfile",
    "MODEL_NAME",
    "MODEL_PATH",
    "TP8_DEFAULT_KV_CAPACITY_TOKENS",
    "ActiveRequest",
    "SystemState",
    "solve_asymmetric_staggering",
    "solve_profile_water_filling",
    "solve_schedule",
    "run_comparative_suite",
    "solve_optimal_composable_policy",
    "PolicyComparisonMatrix",
    "compare_policies",
    "CapacityScalingReport",
    "analyze_policy_capacity_scaling",
    "binned_lpt_priority_key",
    "slack_packed_lpt_priority_key",
    "pure_lpt_priority_key",
    "order_requests_by_policy",
    "slice_into_windowed_batches",
    "calculate_peak_future_memory_trajectory",
    "dynamic_knapsack_admit_guard",
    "schedule_dynamic_knapsack_batches",
    "BasePolicy",
    "DynamicKnapsackPolicy",
    "get_policy",
    "get_active_policy",
    "set_active_policy",
    "PriorityRule",
    "Priority",
    "AdmissionGuard",
    "Guard",
    "BatchTransition",
    "Victim",
    "ComposablePolicy",
    "PolicyBuilder",
    "PolicyVerificationReport",
    "verify_policy_invariants",
    "StepOutput",
    "ExecutionEngine",
    "SimulatedExecutionEngine",
    "VLLMExecutionEngine",
    "PolicyExecutor",
    "cleanup_stale_placement_groups",
    "create_node_pinned_placement_group",
    "XLAProfiler",
    "XLACompilerPhaseStats",
    "HLOInstructionProfile",
    "ConnectedPair",
    "TPUTopologyDiscovery",
    "cleanup_stale_placement_groups",
    "create_node_pinned_placement_group",
    "validate_tp8_v5e_topology",
    "PreemptionTelemetryTracker",
    "TPUPreemptionBenchmarkWorker",
    "run_benchmark_orchestrator",
    "submit_orchestrator_job",
    "TPUJobSubmitter",
    "TPUServingClient",
    "StructuredEvent",
    "emit_omp_event",
    "save_workload_requests",
    "get_workload_path",
    "get_workload_profile",
    "generate_synthetic_workload",
    "generate_distribution_workload",
]
