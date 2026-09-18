"""
Admission Control & Scheduling Policy Experimentation Framework.
================================================================
Provides a modular, declarative, and RCM-compliant infrastructure for
extending, testing, sweeping, and benchmarking scheduling policies on
simulation and physical Cloud TPU hardware.
"""

from admission_control.experiments.config import ExperimentConfig
from admission_control.experiments.runner import (
    PolicyExperimentRunner,
    run_experiment,
)
from admission_control.experiments.testing import (
    test_policy,
    compare_policies,
    sweep_policy,
    quick_eval,
)
from admission_control.experiments.plotting import (
    generate_all_plots,
    plot_kv_capacity_cliff,
    plot_policy_goodput_comparison,
    plot_xla_padding_bubbles,
    plot_throughput_breakdown,
)
from admission_control.experiments.reporting import (
    build_markdown_report,
    build_html_report,
)
from admission_control.experiments import oracle_scheduler

__all__ = [
    "ExperimentConfig",
    "PolicyExperimentRunner",
    "run_experiment",
    "test_policy",
    "compare_policies",
    "sweep_policy",
    "quick_eval",
    "generate_all_plots",
    "plot_kv_capacity_cliff",
    "plot_policy_goodput_comparison",
    "plot_xla_padding_bubbles",
    "plot_throughput_breakdown",
    "build_markdown_report",
    "build_html_report",
    "oracle_scheduler",
]
