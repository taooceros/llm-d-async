"""
Scheduling Policy Registry & Dynamic Discovery Module.
======================================================
Implements global registration, dynamic policy resolution from string names,
file specs, JSON/YAML dictionaries, or Python classes, and active policy management.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from admission_control.core.models import Request, SystemState
from admission_control.algorithms.policies.base import BasePolicy
from admission_control.algorithms.policies.composable import ComposablePolicy
from admission_control.algorithms.policies.priority import (
    PRIORITY_REGISTRY,
    Priority,
    PriorityRule,
)
from admission_control.algorithms.policies.guards import (
    AdmissionGuard,
    Guard,
    BatchTransition,
    Victim,
)
from admission_control.algorithms.policies.builder import PolicyBuilder


_POLICY_REGISTRY: Dict[str, Any] = {}


def register_policy(name: str, policy_or_factory: Any = None, aliases: Optional[List[str]] = None) -> Any:
    """
    Registers a named policy class, builder, instance, or factory function into the global registry.
    Can be used as a function or as a class/function decorator.
    """
    if policy_or_factory is None:
        def decorator(cls_or_fn: Any) -> Any:
            return register_policy(name, cls_or_fn, aliases=aliases)
        return decorator

    clean_name = name.lower().strip()
    _POLICY_REGISTRY[clean_name] = policy_or_factory
    if aliases:
        for a in aliases:
            _POLICY_REGISTRY[a.lower().strip()] = policy_or_factory
    return policy_or_factory


def list_registered_policies() -> Dict[str, str]:
    """Returns mapping of all registered policy names to their description or class name."""
    res = {}
    for k, v in _POLICY_REGISTRY.items():
        if isinstance(v, type):
            desc = v.__name__
        elif callable(v):
            desc = f"Factory({getattr(v, '__name__', 'custom')})"
        elif hasattr(v, "name"):
            desc = v.name
        else:
            desc = str(v)
        res[k] = desc
    return res


def get_policy(
    policy: Union[str, BasePolicy, Dict[str, Any], type, Callable[..., Any]],
    **kwargs: Any,
) -> BasePolicy:
    """Factory helper to resolve or construct a BasePolicy instance from name, class, file, or dict."""
    import os
    import importlib.util

    if isinstance(policy, BasePolicy):
        return policy
    if isinstance(policy, PolicyBuilder):
        return policy.build()
    if isinstance(policy, type) and issubclass(policy, BasePolicy):
        return policy(**kwargs)
    if callable(policy) and not isinstance(policy, type):
        res = policy(**kwargs)
        if isinstance(res, (BasePolicy, PolicyBuilder)):
            return res.build() if isinstance(res, PolicyBuilder) else res
        return res
    if isinstance(policy, dict):
        if "priority" in policy or "guards" in policy:
            return PolicyBuilder.from_dict(policy).build()
        pol_type = policy.get("policy_type", policy.get("type", "knapsack"))
        merged_kwargs = {**policy, **kwargs}
        merged_kwargs.pop("policy_type", None)
        merged_kwargs.pop("type", None)
        return get_policy(pol_type, **merged_kwargs)
    if isinstance(policy, str):
        clean_str = policy.strip()
        # 1. Check if string points to a file on disk
        if (clean_str.endswith(".yaml") or clean_str.endswith(".yml")) and os.path.exists(clean_str):
            with open(clean_str, "r", encoding="utf-8") as f:
                return PolicyBuilder.from_yaml(f.read()).build()
        if clean_str.endswith(".json") and os.path.exists(clean_str):
            with open(clean_str, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
                return get_policy(data, **kwargs)
        # Check dynamic python module loading: path/to/script.py:PolicyClass or path/to/script.py
        if ":" in clean_str and clean_str.split(":", 1)[0].endswith(".py"):
            py_file, symbol_name = clean_str.split(":", 1)
            if os.path.exists(py_file):
                spec = importlib.util.spec_from_file_location("custom_policy_mod", py_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    target = getattr(mod, symbol_name.strip())
                    return get_policy(target, **kwargs)
        elif clean_str.endswith(".py") and os.path.exists(clean_str):
            spec = importlib.util.spec_from_file_location("custom_policy_mod", clean_str)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr_val in mod.__dict__.values():
                    if isinstance(attr_val, type) and issubclass(attr_val, BasePolicy) and attr_val is not BasePolicy:
                        return attr_val(**kwargs)

        # 2. Check registry
        key = clean_str.lower()
        if key in _POLICY_REGISTRY:
            entry = _POLICY_REGISTRY[key]
            if isinstance(entry, type):
                return entry(**kwargs)
            elif callable(entry):
                return entry(**kwargs)
            elif isinstance(entry, BasePolicy):
                return entry
            elif isinstance(entry, PolicyBuilder):
                return entry.build()
        if key in PRIORITY_REGISTRY:
            return _make_knapsack(intra_batch_policy=key, **kwargs)
        return _make_knapsack(**kwargs)
    raise TypeError(f"Cannot resolve policy from type {type(policy)}")


def create_policy(
    name: str = "CustomPolicy",
    priority: Optional[Union[str, PriorityRule, Callable[[Any], Any]]] = None,
    guard: Optional[Union[AdmissionGuard, Callable[[List[Any], Any, Optional[SystemState]], bool]]] = None,
    concurrency_cap: Optional[int] = None,
    capacity: int = 120_000,
    block_size: int = 16,
    victim_selector: Optional[Callable[[List[Any], Optional[SystemState]], Optional[Any]]] = None,
    batch_transition: Optional[Union[AdmissionGuard, Callable[[List[Any], Any, Optional[SystemState]], bool]]] = None,
) -> BasePolicy:
    """Convenience factory to construct an arbitrary scheduling policy in one concise call."""
    builder = PolicyBuilder(name)
    if priority is not None:
        builder.with_priority(priority)
    if concurrency_cap is not None:
        builder.with_concurrency_cap(concurrency_cap)
    if guard is not None:
        builder.admit_if(guard)
    if batch_transition is not None:
        builder.draw_next_batch_if(batch_transition)
    if victim_selector is not None:
        builder.with_victim_selector(victim_selector)
    return builder.build()


_ACTIVE_POLICY_INSTANCE: Optional[BasePolicy] = None


def get_active_policy() -> BasePolicy:
    """Returns the current active candidate scheduling policy under evaluation."""
    global _ACTIVE_POLICY_INSTANCE
    if _ACTIVE_POLICY_INSTANCE is not None:
        return _ACTIVE_POLICY_INSTANCE
    return _make_knapsack(target_concurrency=103)


def set_active_policy(policy: Union[BasePolicy, str, Dict[str, Any]]) -> BasePolicy:
    """Sets the active candidate scheduling policy."""
    global _ACTIVE_POLICY_INSTANCE
    _ACTIVE_POLICY_INSTANCE = get_policy(policy)
    return _ACTIVE_POLICY_INSTANCE


def _make_greedy(max_concurrency: int = 256, **kwargs: Any) -> ComposablePolicy:
    return (
        PolicyBuilder(f"Greedy Reactive Preemption (Max={max_concurrency})")
        .admit_if(Guard.concurrency_cap(max_concurrency))
        .build()
    )


def _make_predictive(cap: int = 48, capacity: int = 120_000, block_size: int = 16, **kwargs: Any) -> ComposablePolicy:
    return (
        PolicyBuilder(f"Predictive Gating (Cap={cap})")
        .with_priority(Priority.binned_lpt())
        .admit_if(Guard.concurrency_cap(cap))
        .admit_if(Guard.peak_future_memory(capacity=capacity, block_size=block_size))
        .build()
    )


def _make_knapsack(
    target_concurrency: int = 103,
    capacity: int = 120_000,
    block_size: int = 16,
    slack_weight: float = 0.0010,
    intra_batch_policy: str = "slack_packed_lpt",
    **kwargs: Any,
) -> ComposablePolicy:
    p_rule = (
        Priority.slack_packed_lpt(weight=slack_weight)
        if intra_batch_policy.lower() == "slack_packed_lpt"
        else Priority.from_name(intra_batch_policy)
    )
    return (
        PolicyBuilder(f"QuantizedKnapsack-{target_concurrency}")
        .with_priority(p_rule)
        .admit_if(Guard.concurrency_cap(target_concurrency))
        .admit_if(Guard.preemption_immune(capacity=capacity, block_size=block_size))
        .draw_next_batch_if(
            BatchTransition.preemption_immune(
                capacity=capacity, target_concurrency=target_concurrency, block_size=block_size
            )
        )
        .with_victim_selector(Victim.least_progress())
        .build()
    )


def _make_discrete_wave(wave_size: int = 32, **kwargs: Any) -> ComposablePolicy:
    return (
        PolicyBuilder(f"Discrete Wave (Size={wave_size})")
        .admit_if(Guard.concurrency_cap(wave_size))
        .build()
    )


# Register built-in policies into registry using generic ComposablePolicy factories
register_policy("greedy", _make_greedy, aliases=["fcfs", "greedy_fcfs", "policy_a", "policy_a_greedy", "reactive"])
register_policy("predictive", _make_predictive, aliases=["gating", "predictive_gating", "policy_b", "policy_b_predictive"])
register_policy("discrete_wave", _make_discrete_wave, aliases=["wave", "discrete_waves", "policy_c"])
register_policy("knapsack", _make_knapsack, aliases=["dynamic_knapsack", "slack_packed", "oversubscribed", "policy_f"])
register_policy("quantized_knapsack", lambda **kw: get_active_policy(), aliases=["quantized", "active", "knapsack_103", "quantizedknapsack-103"])
register_policy("composable", ComposablePolicy, aliases=["composable_policy"])

__all__ = [
    "register_policy",
    "list_registered_policies",
    "get_policy",
    "create_policy",
    "get_active_policy",
    "set_active_policy",
    "BasePolicy",
    "ComposablePolicy",
]
