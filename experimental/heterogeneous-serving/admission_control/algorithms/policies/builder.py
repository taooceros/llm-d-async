"""
Fluent Policy Builder Module.
=============================
Implements `PolicyBuilder`, a declarative fluent builder DSL for constructing
composable, conditional, and staged scheduling policies from code, dicts, or YAML.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from admission_control.core.models import (
    ActiveRequest,
    AdmissionDecision,
    BatchCohort,
    Request,
    SystemState,
)
from admission_control.algorithms.policies.base import BasePolicy
from admission_control.algorithms.policies.priority import Priority, PriorityRule
from admission_control.algorithms.policies.guards import (
    AdmissionGuard,
    AllOfGuard,
    AnyOfGuard,
    ConcurrencyCapGuard,
    Guard,
    BatchTransition,
    Victim,
)
from admission_control.algorithms.policies.composable import ComposablePolicy


class PolicyBuilder:
    """
    Declarative fluent builder DSL for expressing scheduling policies.
    """
    def __init__(self, name: str = "CustomComposablePolicy"):
        self._name = name
        self._priority_rule: Optional[PriorityRule] = None
        self._guards: List[AdmissionGuard] = []
        self._batch_guards: List[AdmissionGuard] = []
        self._victim_selector: Optional[Callable[[Sequence[ActiveRequest], Optional[SystemState]], Optional[ActiveRequest]]] = None
        self._on_admit_fn: Optional[Callable[[Union[Request, ActiveRequest], Optional[SystemState]], None]] = None
        self._on_step_fn: Optional[Callable[[Any, Optional[SystemState]], None]] = None
        self._on_finish_fn: Optional[Callable[[Any, Optional[SystemState]], None]] = None

    def with_priority(self, rule: PriorityRule) -> "PolicyBuilder":
        self._priority_rule = rule
        return self

    def admit_if(self, guard: AdmissionGuard) -> "PolicyBuilder":
        self._guards.append(guard)
        return self

    def with_concurrency_cap(self, cap: int) -> "PolicyBuilder":
        """Convenience method to add a ConcurrencyCapGuard."""
        return self.admit_if(ConcurrencyCapGuard(cap))

    def draw_next_batch_if(self, guard: AdmissionGuard) -> "PolicyBuilder":
        self._batch_guards.append(guard)
        return self

    def with_victim_selector(
        self,
        selector: Callable[[Sequence[ActiveRequest], Optional[SystemState]], Optional[ActiveRequest]],
    ) -> "PolicyBuilder":
        self._victim_selector = selector
        return self

    def on_admit(
        self,
        callback: Callable[[Union[Request, ActiveRequest], Optional[SystemState]], None],
    ) -> "PolicyBuilder":
        self._on_admit_fn = callback
        return self

    def on_step(self, callback: Callable[[Any, Optional[SystemState]], None]) -> "PolicyBuilder":
        self._on_step_fn = callback
        return self

    def on_finish(self, callback: Callable[[Any, Optional[SystemState]], None]) -> "PolicyBuilder":
        self._on_finish_fn = callback
        return self

    def build(self) -> ComposablePolicy:
        admit_guard = AllOfGuard(self._guards) if self._guards else None
        batch_guard = AllOfGuard(self._batch_guards) if self._batch_guards else admit_guard
        return ComposablePolicy(
            name=self._name,
            priority_rule=self._priority_rule,
            admission_guard=admit_guard,
            batch_transition_guard=batch_guard,
            victim_selector=self._victim_selector,
            on_admit_fn=self._on_admit_fn,
            on_step_fn=self._on_step_fn,
            on_finish_fn=self._on_finish_fn,
        )

    @classmethod
    def _parse_priority_spec(cls, spec: Union[str, Dict[str, Any], PriorityRule, None]) -> Optional[PriorityRule]:
        if not spec:
            return None
        if isinstance(spec, PriorityRule):
            return spec
        if isinstance(spec, str):
            spec_str = spec
            s_lower = spec_str.lower()
            import re
            if "slack" in s_lower:
                m = re.search(r"([0-9\.]+)", spec_str)
                w = float(m.group(1)) if m else 0.0010
                return Priority.slack_packed_lpt(weight=w)
            elif "binned" in s_lower:
                m = re.search(r"([0-9]+)", spec_str)
                bs = int(m.group(1)) if m else 16
                return Priority.binned_lpt(bin_size=bs)
            elif "shortest_prompt" in s_lower:
                return Priority.shortest_prompt()
            elif "longest_prompt" in s_lower:
                return Priority.longest_prompt()
            elif "spt" in s_lower:
                return Priority.spt()
            elif "density" in s_lower:
                return Priority.density()
            elif "fifo" in s_lower or "arrival" in s_lower:
                return Priority.fifo()
            elif "lpt" in s_lower:
                return Priority.lpt()
            spec = {"type": spec_str}

        p_type = spec.get("type", "lpt").lower()
        if p_type == "lpt":
            p = Priority.lpt()
        elif p_type == "spt":
            p = Priority.spt()
        elif p_type == "binned_lpt":
            p = Priority.binned_lpt(spec.get("bin_size", 4))
        elif p_type in ("slack_packed", "slack_packed_lpt"):
            p = Priority.slack_packed_lpt(spec.get("weight", 0.0010))
        elif p_type == "shortest_prompt":
            p = Priority.shortest_prompt()
        elif p_type == "longest_prompt":
            p = Priority.longest_prompt()
        elif p_type == "density":
            p = Priority.density()
        elif p_type in ("fifo", "arrival"):
            p = Priority.fifo()
        elif p_type == "edf":
            p = Priority.edf(spec.get("deadline_attr", "deadline"))
        elif p_type == "composite":
            p = Priority.composite(spec.get("weights", {"lpt": 1.0}), name=spec.get("name", "Composite"))
        elif p_type == "prefix_cluster":
            sec = cls._parse_priority_spec(spec.get("secondary")) if "secondary" in spec else None
            p = Priority.prefix_cluster(spec.get("prefix_key", "prefix_hash"), secondary=sec)
        elif p_type == "adaptive_memory":
            norm = cls._parse_priority_spec(spec.get("normal_rule", "lpt"))
            press = cls._parse_priority_spec(spec.get("pressure_rule", "spt"))
            p = Priority.adaptive_memory(norm, press, threshold=spec.get("threshold", 0.85))
        elif p_type == "age_boosted":
            base = cls._parse_priority_spec(spec.get("base_rule", "lpt"))
            p = Priority.age_boosted(base, age_weight=spec.get("age_weight", 0.01))
        else:
            p = Priority.lpt()

        if "then_by" in spec and spec["then_by"]:
            p = p.then_by(cls._parse_priority_spec(spec["then_by"]))
        return p

    @classmethod
    def _parse_guard_spec(cls, spec: Union[str, Dict[str, Any], AdmissionGuard, None]) -> Optional[AdmissionGuard]:
        if not spec:
            return None
        if isinstance(spec, AdmissionGuard):
            return spec
        if isinstance(spec, str):
            spec_str = spec
            if "&" in spec_str:
                parts = [cls._parse_guard_spec(p.strip()) for p in spec_str.split("&")]
                valid = [p for p in parts if p is not None]
                return AllOfGuard(valid) if valid else None
            elif "|" in spec_str:
                parts = [cls._parse_guard_spec(p.strip()) for p in spec_str.split("|")]
                valid = [p for p in parts if p is not None]
                return AnyOfGuard(valid) if valid else None
            import re
            s_lower = spec_str.lower()
            if "concurrencycap" in s_lower:
                m = re.search(r"([0-9]+)", spec_str)
                cap = int(m.group(1)) if m else 103
                return Guard.concurrency_cap(cap)
            elif "peakmemory" in s_lower or "preemption" in s_lower:
                m = re.search(r"([0-9]+)", spec_str)
                cap = int(m.group(1)) if m else 120_000
                return Guard.preemption_immune(capacity=cap, block_size=16)
            elif "headroom" in s_lower:
                m = re.search(r"([0-9]+)", spec_str)
                tok = int(m.group(1)) if m else 5000
                return Guard.memory_headroom(headroom_tokens=tok)
            elif "budget" in s_lower:
                m = re.search(r"([0-9]+)", spec_str)
                b = int(m.group(1)) if m else 2048
                return Guard.prefill_token_budget(max_tokens_per_step=b)
            elif "rate" in s_lower:
                m = re.search(r"([0-9]+)", spec_str)
                r = int(m.group(1)) if m else 16
                return Guard.rate_limit(max_admissions_per_step=r)
            spec = {"type": spec_str}

        g_type = spec.get("type", "concurrency_cap").lower()
        if g_type == "concurrency_cap":
            return Guard.concurrency_cap(spec.get("cap", 103))
        elif g_type in ("peak_future_memory", "preemption_immune"):
            return Guard.preemption_immune(spec.get("capacity", 120_000), spec.get("block_size", 16))
        elif g_type == "memory_headroom":
            return Guard.memory_headroom(spec.get("headroom_tokens", 5_000), spec.get("capacity", 120_000))
        elif g_type == "prefill_token_budget":
            return Guard.prefill_token_budget(spec.get("max_tokens_per_step", 2048))
        elif g_type == "rate_limit":
            return Guard.rate_limit(spec.get("max_admissions_per_step", 16))
        elif g_type == "all_of":
            sub = [cls._parse_guard_spec(g) for g in spec.get("guards", [])]
            return AllOfGuard([s for s in sub if s is not None])
        elif g_type == "any_of":
            sub = [cls._parse_guard_spec(g) for g in spec.get("guards", [])]
            return AnyOfGuard([s for s in sub if s is not None])
        elif g_type in ("k_of_m", "quorum"):
            sub = [cls._parse_guard_spec(g) for g in spec.get("guards", [])]
            return Guard.k_of_m([s for s in sub if s is not None], k=spec.get("k", len(sub)))
        elif g_type in ("prompt_length_limit", "prompt_limit"):
            return Guard.prompt_length_limit(spec.get("max_prompt_len", 4096))
        elif g_type in ("decode_length_limit", "decode_limit"):
            return Guard.decode_length_limit(spec.get("max_decode_len", 4096))
        elif g_type == "adaptive_concurrency":
            return Guard.adaptive_concurrency(
                high_cap=spec.get("high_cap", 128),
                throttled_cap=spec.get("throttled_cap", 103),
                memory_threshold=spec.get("memory_threshold", 0.85),
                capacity=spec.get("capacity", 120_000),
            )
        return Guard.concurrency_cap(spec.get("cap", 103))

    @classmethod
    def _parse_batch_transition_spec(cls, spec: Union[str, Dict[str, Any], AdmissionGuard, None]) -> Optional[AdmissionGuard]:
        if not spec:
            return None
        if isinstance(spec, AdmissionGuard):
            return spec
        if isinstance(spec, str):
            s_lower = spec.lower()
            if "strict" in s_lower or "barrier" in s_lower:
                return BatchTransition.strict_barrier()
            elif "preemption" in s_lower or "knapsack" in s_lower or "concurrency" in s_lower:
                import re
                m_cap = re.search(r"([0-9]+)", spec)
                cap = int(m_cap.group(1)) if m_cap else 103
                return BatchTransition.preemption_immune(capacity=120_000, target_concurrency=cap, block_size=16)
            return cls._parse_guard_spec(spec)

        b_type = spec.get("type", "preemption_immune").lower()
        if b_type == "strict_barrier":
            return BatchTransition.strict_barrier()
        elif b_type == "preemption_immune":
            return BatchTransition.preemption_immune(
                capacity=spec.get("capacity", 120_000),
                target_concurrency=spec.get("target_concurrency", 103),
                block_size=spec.get("block_size", 16),
            )
        return cls._parse_guard_spec(spec)

    @classmethod
    def from_dict(cls, spec: Dict[str, Any]) -> "PolicyBuilder":
        """Reconstructs a PolicyBuilder from a declarative specification dict."""
        name = spec.get("name", "DeserializedComposablePolicy")
        builder = cls(name)
        p_spec = spec.get("priority", spec.get("priority_rule"))
        if p_spec:
            p = cls._parse_priority_spec(p_spec)
            if p is not None:
                builder.with_priority(p)
        g_spec = spec.get("guards", spec.get("admission_guard"))
        if g_spec:
            if isinstance(g_spec, list):
                for sub in g_spec:
                    g = cls._parse_guard_spec(sub)
                    if g is not None:
                        builder.admit_if(g)
            else:
                g = cls._parse_guard_spec(g_spec)
                if g is not None:
                    builder.admit_if(g)
        bt_spec = spec.get("batch_transition", spec.get("batch_transition_guard"))
        if bt_spec:
            bt = cls._parse_batch_transition_spec(bt_spec)
            if bt is not None:
                builder.draw_next_batch_if(bt)
        if "victim_selector" in spec:
            v_sel = spec["victim_selector"]
            if v_sel == "least_progress":
                builder.with_victim_selector(Victim.least_progress())
            elif v_sel == "lifo":
                builder.with_victim_selector(Victim.lifo())
        return builder

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PolicyBuilder":
        """Reconstructs a PolicyBuilder from a YAML specification string."""
        import yaml
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError(f"YAML must define a dictionary, got: {type(data)}")
        return cls.from_dict(data)
