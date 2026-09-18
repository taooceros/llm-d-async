"""
Scheduling Priority Rules & Keys Module.
========================================
Defines priority keys, composable priority rules, and the `Priority` factory namespace
for request sorting and ranking within batch cohorts.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from admission_control.core.models import Request, SystemState


def binned_lpt_priority_key(req: Request, bin_size: int = 16) -> Tuple[int, int, int]:
    """
    Primary priority key: coarse decode binning (-decode_len // bin_size).
    Tiebreaker 1: shorter prompt first (favors quick completions, minimizes prompt KV reservation).
    Tiebreaker 2: longer decode length first within the same bin.
    """
    return (-(req.decode_len // bin_size), req.prompt_len, -req.decode_len)


def pure_lpt_priority_key(req: Request) -> Tuple[int, int]:
    """Pure Longest Processing Time First: (-decode_len, prompt_len)."""
    return (-req.decode_len, req.prompt_len)


def density_priority_key(req: Request) -> Tuple[float, int]:
    """Density key: prioritizes requests with higher decode-to-total token ratios."""
    ratio = req.decode_len / max(1, req.prompt_len + req.decode_len)
    return (-ratio, -req.decode_len)


def spt_priority_key(req: Request) -> Tuple[int, int]:
    """Shortest Processing Time First: (decode_len, prompt_len)."""
    return (req.decode_len, req.prompt_len)


def slack_packed_lpt_priority_key(req: Request, weight: float = 0.006) -> Tuple[float, int, int]:
    """
    Continuous Slack-Packed LPT priority rule.
    Computes score = -decode_len + weight * prompt_len.
    Higher priority (lower tuple value) is given to long decodes with small prompts.
    """
    score = -req.decode_len + weight * req.prompt_len
    return (score, -req.decode_len, req.prompt_len)


PRIORITY_REGISTRY: Dict[str, Callable[[Request], Any]] = {
    "binned_lpt": binned_lpt_priority_key,
    "pure_lpt": pure_lpt_priority_key,
    "density": density_priority_key,
    "spt": spt_priority_key,
    "slack_packed": slack_packed_lpt_priority_key,
}


def order_requests_by_policy(
    requests: List[Request],
    policy: str = "binned_lpt",
) -> List[Request]:
    """Sorts requests according to the named priority policy."""
    key_fn = PRIORITY_REGISTRY.get(policy.lower(), binned_lpt_priority_key)
    return sorted(requests, key=key_fn)


def slice_into_windowed_batches(
    requests: List[Request],
    batch_size: int = 64,
    order_policy: str = "binned_lpt",
    intra_batch_policy: Optional[str] = None,
) -> List[List[Request]]:
    """
    Slices a continuous request stream into disjoint cohorts of size `batch_size`.
    Within each cohort, requests are sorted by `order_policy` / `intra_batch_policy`.
    """
    effective_policy = intra_batch_policy or order_policy
    batches = []
    for i in range(0, len(requests), batch_size):
        cohort = requests[i : i + batch_size]
        ordered_cohort = order_requests_by_policy(cohort, policy=effective_policy)
        batches.append(ordered_cohort)
    return batches


def _extract_request_demands(req: Any) -> Tuple[int, int, int]:
    """Helper to extract (prompt_len, decode_len, tokens_decoded) from various request types."""
    if isinstance(req, dict):
        p = req.get("prompt_len", req.get("prompt_tokens", 0))
        d = req.get("decode_len", req.get("output_tokens", 0))
        c = req.get("tokens_decoded", 0)
    else:
        p = getattr(req, "prompt_len", getattr(req, "prompt_tokens", 0))
        d = getattr(req, "decode_len", getattr(req, "output_tokens", 0))
        c = getattr(req, "tokens_decoded", 0)
    return int(p), int(d), int(c)


class PriorityRule(ABC):
    """Abstract composable priority rule for ranking candidate requests."""
    name: str = "PriorityRule"

    @abstractmethod
    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        pass

    def then_by(self, secondary: "PriorityRule") -> "PriorityRule":
        return ChainedPriorityRule(self, secondary)

    def reversed(self) -> "PriorityRule":
        return ReversedPriorityRule(self)

    def __add__(self, other: "PriorityRule") -> "PriorityRule":
        return WeightedSumPriorityRule([(1.0, self), (1.0, other)])

    def __mul__(self, weight: float) -> "PriorityRule":
        return ScaledPriorityRule(self, weight)

    def __rmul__(self, weight: float) -> "PriorityRule":
        return ScaledPriorityRule(self, weight)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class FunctionalPriorityRule(PriorityRule):
    def __init__(self, fn: Callable[[Any], Any], name: str = "custom_priority"):
        self.fn = fn
        self.name = name

    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        return self.fn(req)


class ChainedPriorityRule(PriorityRule):
    def __init__(self, primary: PriorityRule, secondary: PriorityRule):
        self.primary = primary
        self.secondary = secondary
        self.name = f"{primary.name} -> {secondary.name}"

    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        k1 = self.primary.key(req, state)
        k2 = self.secondary.key(req, state)
        if isinstance(k1, tuple) and isinstance(k2, tuple):
            return k1 + k2
        elif isinstance(k1, tuple):
            return k1 + (k2,)
        elif isinstance(k2, tuple):
            return (k1,) + k2
        return (k1, k2)


class ReversedPriorityRule(PriorityRule):
    def __init__(self, rule: PriorityRule):
        self.rule = rule
        self.name = f"Reversed({rule.name})"

    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        k = self.rule.key(req, state)
        if isinstance(k, (int, float)):
            return -k
        if isinstance(k, tuple):
            return tuple(-x if isinstance(x, (int, float)) else x for x in k)
        return k


class ScaledPriorityRule(PriorityRule):
    def __init__(self, rule: PriorityRule, scale: float):
        self.rule = rule
        self.scale = scale
        self.name = f"{scale:.3f} * {rule.name}"

    def key(self, req: Any, state: Optional[SystemState] = None) -> float:
        val = self.rule.key(req, state)
        return float(val) * self.scale


class WeightedSumPriorityRule(PriorityRule):
    def __init__(self, components: List[Tuple[float, PriorityRule]]):
        self.components = components
        self.name = " + ".join(f"{w:.3f}*{r.name}" for w, r in components)

    def key(self, req: Any, state: Optional[SystemState] = None) -> float:
        total = 0.0
        for w, rule in self.components:
            val = rule.key(req, state)
            total += w * float(val if isinstance(val, (int, float)) else val[0])
        return total

    def __add__(self, other: PriorityRule) -> "WeightedSumPriorityRule":
        return WeightedSumPriorityRule(self.components + [(1.0, other)])


class CompositePriorityRule(PriorityRule):
    """
    Multi-objective composite priority rule.
    Computes a weighted linear combination of feature scores.
    """
    def __init__(self, weights: Dict[str, float], name: str = "CompositePriority"):
        self.weights = weights
        self.name = name

    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        p, d, c = _extract_request_demands(req)
        score = 0.0
        if "lpt" in self.weights:
            score += self.weights["lpt"] * (-d)
        if "spt" in self.weights:
            score += self.weights["spt"] * d
        if "shortest_prompt" in self.weights:
            score += self.weights["shortest_prompt"] * p
        if "longest_prompt" in self.weights:
            score += self.weights["longest_prompt"] * (-p)
        if "density" in self.weights:
            score += self.weights["density"] * (-d / max(1, p + d) * 1000.0)
        return (score, -d, p)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "composite", "weights": self.weights, "name": self.name}


class PrefixClusteringPriorityRule(PriorityRule):
    """
    Clusters requests sharing common prefix or session identifiers
    to maximize KV cache reuse (Radix attention) in serving engines.
    """
    def __init__(self, prefix_key: str = "prefix_hash", secondary_rule: Optional[PriorityRule] = None):
        self.prefix_key = prefix_key
        self.secondary_rule = secondary_rule or Priority.lpt()
        self.name = f"PrefixCluster({prefix_key}) -> {self.secondary_rule.name}"

    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        if isinstance(req, dict):
            val = req.get("metadata", {}).get(self.prefix_key, req.get(self.prefix_key, 0))
        else:
            meta = getattr(req, "metadata", {})
            val = meta.get(self.prefix_key, getattr(req, self.prefix_key, 0))
        sec = self.secondary_rule.key(req, state)
        return (val, sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "prefix_cluster",
            "prefix_key": self.prefix_key,
            "secondary": self.secondary_rule.to_dict() if hasattr(self.secondary_rule, "to_dict") else None,
        }


class AdaptiveMemoryPriorityRule(PriorityRule):
    """
    State-aware priority rule that dynamically switches between two priority rules
    based on real-time KV cache memory utilization.
    """
    def __init__(self, normal_rule: PriorityRule, pressure_rule: PriorityRule, threshold: float = 0.85):
        self.normal_rule = normal_rule
        self.pressure_rule = pressure_rule
        self.threshold = threshold
        self.name = f"AdaptiveMemory({normal_rule.name} -> {pressure_rule.name} @ {threshold:.0%})"

    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        if state is not None and state.capacity_tokens > 0:
            util = state.used_tokens / state.capacity_tokens
            if util >= self.threshold:
                return self.pressure_rule.key(req, state)
        return self.normal_rule.key(req, state)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "adaptive_memory",
            "normal_rule": self.normal_rule.to_dict() if hasattr(self.normal_rule, "to_dict") else self.normal_rule.name,
            "pressure_rule": self.pressure_rule.to_dict() if hasattr(self.pressure_rule, "to_dict") else self.pressure_rule.name,
            "threshold": self.threshold,
            "name": self.name,
        }


class AgeBoostedPriorityRule(PriorityRule):
    """
    State-aware priority rule that boosts priority for older waiting requests
    proportional to waiting duration to prevent starvation.
    """
    def __init__(self, base_rule: PriorityRule, age_weight: float = 0.01):
        self.base_rule = base_rule
        self.age_weight = age_weight
        self.name = f"AgeBoosted({base_rule.name}, weight={age_weight})"

    def key(self, req: Any, state: Optional[SystemState] = None) -> Any:
        base_k = self.base_rule.key(req, state)
        arr = getattr(req, "arrival_step", getattr(req, "arrival_time", req.get("arrival_time", 0) if isinstance(req, dict) else 0))
        cur = state.step if state is not None else 0
        wait = max(0, cur - arr)
        boost = -self.age_weight * wait
        if isinstance(base_k, (int, float)):
            return base_k + boost
        elif isinstance(base_k, tuple):
            return (base_k[0] + boost,) + base_k[1:]
        return base_k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "age_boosted",
            "base_rule": self.base_rule.to_dict() if hasattr(self.base_rule, "to_dict") else self.base_rule.name,
            "age_weight": self.age_weight,
            "name": self.name,
        }


class Priority:
    """Factory namespace for constructing composable priority rules."""
    @staticmethod
    def lpt() -> PriorityRule:
        return FunctionalPriorityRule(lambda r: -getattr(r, "decode_len", r.get("decode_len", 0) if isinstance(r, dict) else 0), "LPT")

    @staticmethod
    def spt() -> PriorityRule:
        return FunctionalPriorityRule(lambda r: getattr(r, "decode_len", r.get("decode_len", 0) if isinstance(r, dict) else 0), "SPT")

    shortest_processing_time = spt
    longest_processing_time = lpt

    @staticmethod
    def binned_lpt(bin_size: int = 4) -> PriorityRule:
        return FunctionalPriorityRule(
            lambda r: (
                -(getattr(r, "decode_len", r.get("decode_len", 0) if isinstance(r, dict) else 0) // bin_size),
                getattr(r, "prompt_len", r.get("prompt_len", 0) if isinstance(r, dict) else 0),
                -getattr(r, "decode_len", r.get("decode_len", 0) if isinstance(r, dict) else 0),
            ),
            f"BinnedLPT({bin_size})",
        )

    @staticmethod
    def shortest_prompt() -> PriorityRule:
        return FunctionalPriorityRule(lambda r: getattr(r, "prompt_len", r.get("prompt_len", 0) if isinstance(r, dict) else 0), "ShortestPrompt")

    @staticmethod
    def longest_prompt() -> PriorityRule:
        return FunctionalPriorityRule(lambda r: -getattr(r, "prompt_len", r.get("prompt_len", 0) if isinstance(r, dict) else 0), "LongestPrompt")

    @staticmethod
    def slack_packed_lpt(weight: float = 0.006) -> PriorityRule:
        return FunctionalPriorityRule(
            lambda r: (
                -getattr(r, "decode_len", r.get("decode_len", 0) if isinstance(r, dict) else 0)
                + weight * getattr(r, "prompt_len", r.get("prompt_len", 0) if isinstance(r, dict) else 0),
                -getattr(r, "decode_len", r.get("decode_len", 0) if isinstance(r, dict) else 0),
                getattr(r, "prompt_len", r.get("prompt_len", 0) if isinstance(r, dict) else 0),
            ),
            f"SlackPackedLPT({weight})",
        )

    @staticmethod
    def density() -> PriorityRule:
        def _density(r):
            p = getattr(r, "prompt_len", r.get("prompt_len", 0) if isinstance(r, dict) else 0)
            d = getattr(r, "decode_len", r.get("decode_len", 0) if isinstance(r, dict) else 0)
            return (-d / max(1, p + d), -d)
        return FunctionalPriorityRule(_density, "Density")

    @staticmethod
    def custom(fn: Callable[[Any], Any], name: str = "custom") -> PriorityRule:
        return FunctionalPriorityRule(fn, name)

    @staticmethod
    def fifo() -> PriorityRule:
        return FunctionalPriorityRule(lambda r: getattr(r, "arrival_time", r.get("arrival_time", 0) if isinstance(r, dict) else 0), "FIFO")

    @staticmethod
    def arrival() -> PriorityRule:
        return Priority.fifo()

    @staticmethod
    def edf(deadline_attr: str = "deadline") -> PriorityRule:
        def _edf(r):
            if isinstance(r, dict):
                return r.get("metadata", {}).get(deadline_attr, r.get("arrival_time", 0) + r.get("decode_len", 0))
            meta = getattr(r, "metadata", {})
            return meta.get(deadline_attr, getattr(r, "arrival_time", 0) + getattr(r, "decode_len", 0))
        return FunctionalPriorityRule(_edf, f"EDF({deadline_attr})")

    @staticmethod
    def composite(weights: Dict[str, float], name: str = "Composite") -> PriorityRule:
        return CompositePriorityRule(weights, name=name)

    @staticmethod
    def prefix_cluster(prefix_key: str = "prefix_hash", secondary: Optional[PriorityRule] = None) -> PriorityRule:
        return PrefixClusteringPriorityRule(prefix_key, secondary)

    @staticmethod
    def adaptive_memory(normal_rule: PriorityRule, pressure_rule: PriorityRule, threshold: float = 0.85) -> PriorityRule:
        return AdaptiveMemoryPriorityRule(normal_rule, pressure_rule, threshold=threshold)

    @staticmethod
    def age_boosted(base_rule: PriorityRule, age_weight: float = 0.01) -> PriorityRule:
        return AgeBoostedPriorityRule(base_rule, age_weight=age_weight)
