"""
Base Scheduling Policy Module.
==============================
Defines `BasePolicy`, the abstract foundation for all scheduling and admission
control policies in the admission control system.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from admission_control.core.models import (
    ActiveRequest,
    BatchCohort,
    Request,
    SystemState,
)


class BasePolicy:
    """Abstract base class for all scheduling and admission control policies."""
    name: str = "BasePolicy"

    def __init__(self, name: Optional[str] = None):
        if name is not None:
            self.name = name

    def order_batch(
        self,
        batch: Union[BatchCohort, Sequence[Request], List[Any]],
        system_state: Optional[SystemState] = None,
    ) -> Union[BatchCohort, List[Any]]:
        """Reorders requests strictly within an individual batch cohort."""
        if isinstance(batch, BatchCohort):
            return batch
        return list(batch)

    def can_admit(
        self,
        active_requests: Sequence[Union[ActiveRequest, Any]],
        candidate: Union[Request, Any],
        system_state: Optional[SystemState] = None,
    ) -> bool:
        """Evaluates whether candidate request can be admitted at current step."""
        return True

    def can_draw_next_batch(
        self,
        active_requests: Sequence[Union[ActiveRequest, Any]],
        next_batch_candidate: Union[Request, Any],
        system_state: Optional[SystemState] = None,
    ) -> bool:
        """
        Enforces project contract: Requests may be drawn from a new batch only if
        no request in the current batch can be preempted.
        """
        return self.can_admit(active_requests, next_batch_candidate, system_state)

    def select_preemption_victim(
        self,
        active_requests: Sequence[Union[ActiveRequest, Any]],
        system_state: Optional[SystemState] = None,
    ) -> Optional[Any]:
        """Selects a victim request to preempt if memory headroom is breached. Default: LIFO."""
        return active_requests[-1] if active_requests else None

    def is_paused(self, req_id: Any, step: int) -> bool:
        """Returns True if the request is paused at the given step."""
        return False

    def on_admit(self, request: Any, system_state: Optional[SystemState] = None) -> None:
        """Lifecycle hook invoked immediately after a request is admitted into the engine."""
        pass

    def on_step(self, step_output: Any, system_state: Optional[SystemState] = None) -> None:
        """Lifecycle hook invoked immediately after a step completes."""
        pass

    def on_finish(self, request_id: Any, system_state: Optional[SystemState] = None) -> None:
        """Lifecycle hook invoked immediately after a request finishes generation."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {"policy_class": self.__class__.__name__, "name": self.name}

    def describe(self) -> str:
        """Returns a human-readable structural summary of the policy."""
        return f"{self.__class__.__name__}: {self.name}"

    def to_json(self, indent: int = 2) -> str:
        """Serializes policy configuration to a clean JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serializes policy configuration to a clean YAML string."""
        import yaml
        return yaml.dump(self.to_dict(), sort_keys=False)

    def to_markdown(self) -> str:
        """Generates a markdown documentation summary of the policy specification."""
        lines = [f"# Scheduling Policy: {self.name}", ""]
        lines.append(f"**Policy Class**: `{self.__class__.__name__}`  ")
        lines.append(f"\n```\n{self.describe()}\n```\n")
        return "\n".join(lines)


__all__ = ["BasePolicy"]
