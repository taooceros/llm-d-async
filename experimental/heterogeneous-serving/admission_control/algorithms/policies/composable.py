"""
Composable Scheduling Policy Module.
====================================
Implements `ComposablePolicy`, an extensible and modular scheduling policy
assembled cleanly from decoupled PriorityRule, AdmissionGuard,
BatchTransitionRule, and VictimSelector components.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from admission_control.core.models import (
    ActiveRequest,
    AdmissionDecision,
    BatchCohort,
    Request,
    SystemState,
)
from admission_control.algorithms.policies.base import BasePolicy
from admission_control.algorithms.policies.priority import PriorityRule
from admission_control.algorithms.policies.guards import (
    AdmissionGuard,
    AllOfGuard,
    AnyOfGuard,
    ConcurrencyCapGuard,
    PeakMemoryGuard,
    MemoryHeadroomGuard,
    Guard,
)


class ComposablePolicy(BasePolicy):
    """
    Fully composable, modular scheduling policy.
    Assembled cleanly from decoupled PriorityRule, AdmissionGuard,
    BatchTransitionRule, and VictimSelector components.
    """
    def __init__(
        self,
        name: str = "ComposablePolicy",
        priority_rule: Optional[PriorityRule] = None,
        admission_guard: Optional[AdmissionGuard] = None,
        batch_transition_guard: Optional[AdmissionGuard] = None,
        victim_selector: Optional[Callable[[Sequence[ActiveRequest], Optional[SystemState]], Optional[ActiveRequest]]] = None,
        on_admit_fn: Optional[Callable[[Union[Request, ActiveRequest], Optional[SystemState]], None]] = None,
        on_step_fn: Optional[Callable[[Any, Optional[SystemState]], None]] = None,
        on_finish_fn: Optional[Callable[[Any, Optional[SystemState]], None]] = None,
    ):
        self.name = name
        self.priority_rule = priority_rule
        self.admission_guard = admission_guard
        self.batch_transition_guard = batch_transition_guard
        self.victim_selector = victim_selector
        self._on_admit_fn = on_admit_fn
        self._on_step_fn = on_step_fn
        self._on_finish_fn = on_finish_fn

    def order_batch(
        self,
        batch: Union[BatchCohort, Sequence[Request]],
        system_state: Optional[SystemState] = None,
    ) -> Union[BatchCohort, List[Request]]:
        """Reorders requests strictly within a cohort boundary."""
        requests = batch.requests if isinstance(batch, BatchCohort) else batch
        if self.priority_rule is None:
            ordered = list(requests)
        else:
            ordered = sorted(requests, key=lambda r: self.priority_rule.key(r, system_state))
        if isinstance(batch, BatchCohort):
            return batch.reorder(ordered)
        return list(ordered)

    def clone(self, name: Optional[str] = None) -> Any:
        """Clones this policy into a mutable PolicyBuilder for fine-tuning or adaptation."""
        from admission_control.algorithms.policies.builder import PolicyBuilder
        builder = PolicyBuilder(name or f"{self.name}_Clone")
        if self.priority_rule:
            builder.with_priority(self.priority_rule)
        if self.admission_guard:
            if isinstance(self.admission_guard, AllOfGuard):
                for g in self.admission_guard.guards:
                    builder.admit_if(g)
            else:
                builder.admit_if(self.admission_guard)
        if self.batch_transition_guard:
            if isinstance(self.batch_transition_guard, AllOfGuard):
                for g in self.batch_transition_guard.guards:
                    builder.draw_next_batch_if(g)
            else:
                builder.draw_next_batch_if(self.batch_transition_guard)
        if self.victim_selector:
            builder.with_victim_selector(self.victim_selector)
        if self._on_admit_fn:
            builder.on_admit(self._on_admit_fn)
        if self._on_step_fn:
            builder.on_step(self._on_step_fn)
        if self._on_finish_fn:
            builder.on_finish(self._on_finish_fn)
        return builder

    def with_priority(self, rule: PriorityRule, name: Optional[str] = None) -> "ComposablePolicy":
        """Returns a cloned policy with updated request prioritization rule."""
        builder = self.clone(name or f"{self.name}[P={rule.name}]")
        builder.with_priority(rule)
        return builder.build()

    def with_concurrency_cap(self, new_cap: int, name: Optional[str] = None) -> "ComposablePolicy":
        """Returns a cloned policy with all ConcurrencyCapGuard thresholds updated to new_cap."""
        from admission_control.algorithms.policies.builder import PolicyBuilder
        def _update_guard(g: AdmissionGuard) -> AdmissionGuard:
            if isinstance(g, ConcurrencyCapGuard):
                return Guard.concurrency_cap(new_cap)
            elif isinstance(g, AllOfGuard):
                return AllOfGuard([_update_guard(sub) for sub in g.guards])
            elif isinstance(g, AnyOfGuard):
                return AnyOfGuard([_update_guard(sub) for sub in g.guards])
            return g

        builder = PolicyBuilder(name or f"{self.name}[Cap={new_cap}]")
        if self.priority_rule:
            builder.with_priority(self.priority_rule)
        if self.admission_guard:
            builder.admit_if(_update_guard(self.admission_guard))
        if self.batch_transition_guard:
            builder.draw_next_batch_if(_update_guard(self.batch_transition_guard))
        if self.victim_selector:
            builder.with_victim_selector(self.victim_selector)
        if self._on_admit_fn:
            builder.on_admit(self._on_admit_fn)
        if self._on_step_fn:
            builder.on_step(self._on_step_fn)
        if self._on_finish_fn:
            builder.on_finish(self._on_finish_fn)
        return builder.build()

    def with_capacity(self, new_capacity: int, name: Optional[str] = None) -> "ComposablePolicy":
        """Returns a cloned policy with memory capacity guards scaled to new_capacity."""
        from admission_control.algorithms.policies.builder import PolicyBuilder
        def _update_mem(g: AdmissionGuard) -> AdmissionGuard:
            if isinstance(g, PeakMemoryGuard):
                return Guard.preemption_immune(capacity=new_capacity, block_size=g.block_size)
            elif isinstance(g, MemoryHeadroomGuard):
                return Guard.memory_headroom(headroom_tokens=g.headroom_tokens, capacity=new_capacity)
            elif isinstance(g, AllOfGuard):
                return AllOfGuard([_update_mem(sub) for sub in g.guards])
            elif isinstance(g, AnyOfGuard):
                return AnyOfGuard([_update_mem(sub) for sub in g.guards])
            return g

        builder = PolicyBuilder(name or f"{self.name}[Mem={new_capacity}]")
        if self.priority_rule:
            builder.with_priority(self.priority_rule)
        if self.admission_guard:
            builder.admit_if(_update_mem(self.admission_guard))
        if self.batch_transition_guard:
            builder.draw_next_batch_if(_update_mem(self.batch_transition_guard))
        if self.victim_selector:
            builder.with_victim_selector(self.victim_selector)
        if self._on_admit_fn:
            builder.on_admit(self._on_admit_fn)
        if self._on_step_fn:
            builder.on_step(self._on_step_fn)
        if self._on_finish_fn:
            builder.on_finish(self._on_finish_fn)
        return builder.build()

    def to_json(self, indent: int = 2) -> str:
        """Serializes policy configuration to a clean JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serializes policy configuration to a clean YAML string."""
        import yaml
        return yaml.dump(self.to_dict(), sort_keys=False)

    def to_python_code(self) -> str:
        """Generates clean, standalone, executable Python code reconstructing this policy."""
        lines = [
            "# Auto-generated scheduling policy definition",
            "from admission_control.algorithms.policies import (",
            "    PolicyBuilder, Priority, Guard, BatchTransition, Victim,",
            ")",
            "",
            f"policy = (",
            f"    PolicyBuilder({repr(self.name)})",
        ]
        if self.priority_rule:
            p_str = self.priority_rule.name
            import re
            if "slack" in p_str.lower():
                m = re.search(r"([0-9\.]+)", p_str)
                w = float(m.group(1)) if m else 0.0010
                lines.append(f"        .with_priority(Priority.slack_packed_lpt(weight={w}))")
            elif "binned" in p_str.lower():
                m = re.search(r"([0-9]+)", p_str)
                bs = int(m.group(1)) if m else 16
                lines.append(f"        .with_priority(Priority.binned_lpt(bin_size={bs}))")
            elif "shortest_prompt" in p_str.lower():
                lines.append("        .with_priority(Priority.shortest_prompt())")
            elif "lpt" in p_str.lower():
                lines.append("        .with_priority(Priority.lpt())")
            elif "spt" in p_str.lower():
                lines.append("        .with_priority(Priority.spt())")
            elif "density" in p_str.lower():
                lines.append("        .with_priority(Priority.density())")
            else:
                lines.append(f"        .with_priority(Priority.custom({repr(p_str)}, lambda r: r.decode_len))")

        if self.admission_guard:
            g_str = self.admission_guard.name
            parts = [p.strip() for p in g_str.split("&")]
            guard_exprs = []
            for part in parts:
                p_lower = part.lower()
                import re
                if "concurrencycap" in p_lower:
                    m = re.search(r"([0-9]+)", part)
                    cap = int(m.group(1)) if m else 103
                    guard_exprs.append(f"Guard.concurrency_cap({cap})")
                elif "peakmemory" in p_lower or "preemption" in p_lower:
                    m = re.search(r"([0-9]+)", part)
                    cap = int(m.group(1)) if m else 120_000
                    guard_exprs.append(f"Guard.preemption_immune(capacity={cap})")
                elif "headroom" in p_lower:
                    m = re.search(r"([0-9]+)", part)
                    tok = int(m.group(1)) if m else 5000
                    guard_exprs.append(f"Guard.memory_headroom({tok})")
            if guard_exprs:
                combined_expr = " & ".join(guard_exprs)
                lines.append(f"        .admit_if({combined_expr})")

        if self.batch_transition_guard:
            lines.append("        .draw_next_batch_if(BatchTransition.preemption_immune(capacity=120_000, target_concurrency=103))")

        if self.victim_selector:
            v_name = getattr(self.victim_selector, "__name__", "least_progress")
            if "least" in v_name:
                lines.append("        .with_victim_selector(Victim.least_progress())")
            elif "lifo" in v_name:
                lines.append("        .with_victim_selector(Victim.lifo())")

        lines.append("        .build()")
        lines.append(")")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Generates a Mermaid graph diagram representing the policy flow."""
        lines = ["graph TD"]
        lines.append(f"    subgraph Policy[\"{self.name}\"]")
        lines.append("        direction TB")
        if self.priority_rule:
            p_name = self.priority_rule.name.replace('"', "'")
            lines.append(f"        P[\"Priority: {p_name}\"] --> Q[\"Sorted Intra-Batch Queue\"]")
            lines.append("        Q --> G0[\"Evaluate Admission\"]")
        else:
            lines.append("        Q[\"Intra-Batch Queue\"] --> G0[\"Evaluate Admission\"]")

        if self.admission_guard:
            g_name = self.admission_guard.name.replace('"', "'")
            lines.append(f"        G0 --> G[\"Admission Guard: {g_name}\"]")
            lines.append("        G -->|Pass| BT[\"Batch Transition Guard\"]")
            lines.append("        G -->|Fail| T[\"Throttle Step\"]")
        else:
            lines.append("        G0 --> BT[\"Batch Transition Guard\"]")

        if self.batch_transition_guard:
            bt_name = self.batch_transition_guard.name.replace('"', "'")
            lines.append(f"        BT --> BTG[\"{bt_name}\"]")
            lines.append("        BTG -->|Admit| E[\"Accelerator TPU HBM Execution\"]")
            lines.append("        BTG -->|Hold| H[\"Hold for Current Batch Drain\"]")
        else:
            lines.append("        BT --> E[\"Accelerator TPU HBM Execution\"]")

        if self.victim_selector:
            v_name = getattr(self.victim_selector, "__name__", "victim").replace('"', "'")
            lines.append(f"        E -.->|Capacity Breach| V[\"Victim: {v_name}\"]")

        lines.append("    end")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generates a rich markdown documentation summary of the policy specification."""
        lines = [f"# Scheduling Policy: {self.name}", ""]
        lines.append(f"**Policy Class**: `{self.__class__.__name__}`  ")
        lines.append(f"\n```\n{self.describe()}\n```\n")
        lines.append("## Component Specification\n")
        lines.append("| Component | Description |")
        lines.append("| :--- | :--- |")
        if self.priority_rule:
            lines.append(f"| **Priority Rule** | `{self.priority_rule.name}` |")
        if self.admission_guard:
            lines.append(f"| **Admission Guard** | `{self.admission_guard.name}` |")
        if self.batch_transition_guard:
            lines.append(f"| **Batch Transition** | `{self.batch_transition_guard.name}` |")
        if self.victim_selector:
            v_name = getattr(self.victim_selector, "__name__", str(self.victim_selector))
            lines.append(f"| **Victim Selector** | `{v_name}` |")
        lines.append("")
        lines.append("## Architecture Flow\n")
        lines.append(f"```mermaid\n{self.to_mermaid()}\n```\n")
        return "\n".join(lines)

    def can_admit(
        self,
        active_requests: Sequence[ActiveRequest],
        candidate: Request,
        system_state: Optional[SystemState] = None,
    ) -> bool:
        """Determines if candidate request can be admitted at current step."""
        if self.admission_guard is None:
            return True
        return self.admission_guard.can_admit(active_requests, candidate, system_state)

    def evaluate_admission(
        self,
        active_requests: Sequence[ActiveRequest],
        candidate: Request,
        system_state: Optional[SystemState] = None,
    ) -> AdmissionDecision:
        """Evaluates admission decision and returns structured AdmissionDecision with diagnostics."""
        if self.admission_guard is None:
            return AdmissionDecision(admitted=True, guard_name=self.name, reason="No admission guard configured")
        if hasattr(self.admission_guard, "evaluate"):
            return self.admission_guard.evaluate(active_requests, candidate, system_state)
        admitted = self.admission_guard.can_admit(active_requests, candidate, system_state)
        return AdmissionDecision(
            admitted=admitted,
            guard_name=self.admission_guard.name,
            reason="Admitted by guard" if admitted else "Rejected by guard",
        )

    def can_draw_next_batch(
        self,
        active_requests: Sequence[ActiveRequest],
        next_batch_candidate: Request,
        system_state: Optional[SystemState] = None,
    ) -> bool:
        """Enforces preemption immunity for inter-batch transitions."""
        if self.batch_transition_guard is not None:
            return self.batch_transition_guard.can_admit(active_requests, next_batch_candidate, system_state)
        return self.can_admit(active_requests, next_batch_candidate, system_state)

    def evaluate_batch_transition(
        self,
        active_requests: Sequence[ActiveRequest],
        next_batch_candidate: Request,
        system_state: Optional[SystemState] = None,
    ) -> AdmissionDecision:
        """Evaluates batch transition decision and returns structured AdmissionDecision with diagnostics."""
        if self.batch_transition_guard is not None:
            if hasattr(self.batch_transition_guard, "evaluate"):
                return self.batch_transition_guard.evaluate(active_requests, next_batch_candidate, system_state)
            admitted = self.batch_transition_guard.can_admit(active_requests, next_batch_candidate, system_state)
            return AdmissionDecision(
                admitted=admitted,
                guard_name=self.batch_transition_guard.name,
                reason="Admitted by batch transition" if admitted else "Rejected by batch transition",
            )
        return self.evaluate_admission(active_requests, next_batch_candidate, system_state)

    def select_preemption_victim(
        self,
        active_requests: Sequence[ActiveRequest],
        system_state: Optional[SystemState] = None,
    ) -> Optional[ActiveRequest]:
        """Selects victim request to preempt on memory pressure. Default: configured victim selector or LIFO."""
        if self.victim_selector is not None:
            return self.victim_selector(active_requests, system_state)
        return super().select_preemption_victim(active_requests, system_state)

    def on_admit(
        self,
        request: Union[Request, ActiveRequest],
        system_state: Optional[SystemState] = None,
    ) -> None:
        if self._on_admit_fn is not None:
            self._on_admit_fn(request, system_state)

    def on_step(self, step_output: Any, system_state: Optional[SystemState] = None) -> None:
        if self._on_step_fn is not None:
            self._on_step_fn(step_output, system_state)

    def on_finish(self, request_id: Any, system_state: Optional[SystemState] = None) -> None:
        if self._on_finish_fn is not None:
            self._on_finish_fn(request_id, system_state)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_class": self.__class__.__name__,
            "name": self.name,
            "priority_rule": getattr(self.priority_rule, "name", None),
            "admission_guard": getattr(self.admission_guard, "name", None),
            "batch_transition_guard": getattr(self.batch_transition_guard, "name", None),
        }

    def describe(self) -> str:
        """Returns a human-readable structural ASCII tree of the composable policy."""
        lines = [f"ComposablePolicy: {self.name}"]
        p_name = getattr(self.priority_rule, "name", str(self.priority_rule)) if self.priority_rule else "None"
        lines.append(f"  ├── Priority Rule     : {p_name}")
        g_name = getattr(self.admission_guard, "name", str(self.admission_guard)) if self.admission_guard else "None"
        lines.append(f"  ├── Admission Guards  : {g_name}")
        bt_name = getattr(self.batch_transition_guard, "name", str(self.batch_transition_guard)) if self.batch_transition_guard else "None"
        lines.append(f"  ├── Batch Transition  : {bt_name}")
        v_name = getattr(self.victim_selector, "__name__", str(self.victim_selector)) if self.victim_selector else "None"
        lines.append(f"  └── Victim Selector   : {v_name}")
        return "\n".join(lines)
