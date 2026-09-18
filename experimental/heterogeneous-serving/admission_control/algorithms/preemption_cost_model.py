"""Mathematical Preemption Cost Model on Cloud TPU v5e (TP=8, Gemma-4 31B)."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from admission_control.core.models import ActiveRequest, HardwareSpec


class PreemptionAction(str, Enum):
    """Action taken when an active stream must be preempted."""
    SUSPEND = "suspend"
    SAVE_AND_REPREFILL = "save_and_reprefill"
    SEQUENTIAL_REGENERATE = "sequential_regenerate"
    HYBRID = "hybrid"


class VictimRule(str, Enum):
    """Rule for selecting which active stream to preempt upon capacity breach."""
    LEAST_PROGRESS = "least_progress"
    MAX_YIELD = "max_yield"
    SRPT = "srpt"
    FIFO = "fifo"


@dataclass(frozen=True)
class PreemptionCostEvaluation:
    """Analytical cost, token loss, and memory recovery for a candidate victim under an action."""
    action: PreemptionAction
    victim_req_id: Any
    tokens_demanded: int
    tokens_discarded: int
    recompute_tokens: int
    recompute_latency_s: float
    memory_recovered_tokens: int
    memory_recovered_blocks: int

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class PreemptionTradeoffResult:
    """Quantitative ROI analysis of an oversubscribed planned-preemption schedule."""
    concurrency_target: int
    concurrency_baseline: int
    target_bucket: int
    baseline_bucket: int
    decode_time_saved_s: float
    reprefill_penalty_s: float
    net_time_saved_s: float
    roi_ratio: float
    is_favorable: bool
    optimal_preemption_action: PreemptionAction
    recommended_victim_rule: VictimRule


def _extract_stream_demands(req: Any) -> Tuple[Any, int, int, int]:
    """Extracts (req_id, prompt_len, decode_len, tokens_decoded) from ActiveRequest or dict."""
    if isinstance(req, dict):
        return req.get("req_id", 0), req.get("prompt_len", 0), req.get("decode_len", 0), req.get("tokens_decoded", req.get("generated_tokens", 0))
    return getattr(req, "req_id", id(req)), getattr(req, "prompt_len", 0), getattr(req, "decode_len", 0), getattr(req, "tokens_decoded", getattr(req, "generated_tokens", 0))


class PreemptionCostModel:
    """Evaluates compute, memory, and accelerator latencies for preemption decisions on TPU v5e."""

    def __init__(
        self,
        hw: Optional[HardwareSpec] = None,
        param_count: float = 31e9,
        effective_tflops: float = 630.0,
    ) -> None:
        self.hw: HardwareSpec = hw or HardwareSpec(capacity_tokens=120_000, max_batch_size=128)
        self.param_count: float = param_count
        self.effective_tflops: float = effective_tflops

    def reprefill_latency_s(self, tokens: int) -> float:
        """Parallel prefill latency across TPU MXUs: FLOPs = 2 * Params * Tokens."""
        if tokens <= 0:
            return 0.0
        return max(0.015, (2.0 * self.param_count * tokens) / (self.effective_tflops * 1e12))

    def sequential_decode_latency_s(self, tokens: int, batch_size: int = 64) -> float:
        """Sequential decode latency if tokens had to be regenerated 1-by-1."""
        return (tokens * self.hw.step_latency(batch_size)) if tokens > 0 else 0.0

    def compute_recompute_latency(
        self,
        request: Union[ActiveRequest, Any],
        action: PreemptionAction = PreemptionAction.SAVE_AND_REPREFILL,
        batch_size: int = 64,
    ) -> float:
        """Calculates recomputation latency in seconds for an active request under the given action."""
        _, p, _, c = _extract_stream_demands(request)
        if action == PreemptionAction.SUSPEND:
            return 0.0
        if action == PreemptionAction.SEQUENTIAL_REGENERATE:
            return self.reprefill_latency_s(p) + self.sequential_decode_latency_s(c, batch_size)
        return self.reprefill_latency_s(p + c)

    calculate_recompute_latency_s = compute_recompute_latency

    def compute_memory_recovery(
        self,
        request: Union[ActiveRequest, Any],
        action: PreemptionAction = PreemptionAction.SAVE_AND_REPREFILL,
        block_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Calculates (recovered_tokens, recovered_blocks) achieved by preempting request."""
        if action == PreemptionAction.SUSPEND:
            return 0, 0
        bs = block_size or self.hw.block_size
        _, p, _, c = _extract_stream_demands(request)
        tokens = p + c
        blocks = math.ceil(tokens / bs) if bs > 1 else tokens
        return (blocks * bs if bs > 1 else tokens), blocks

    calculate_memory_recovery = compute_memory_recovery

    def evaluate_preemption_cost(
        self,
        victim: Union[ActiveRequest, Any],
        action: PreemptionAction = PreemptionAction.SAVE_AND_REPREFILL,
        batch_size: int = 64,
        block_size: Optional[int] = None,
    ) -> PreemptionCostEvaluation:
        """Computes full typed PreemptionCostEvaluation for a candidate victim request."""
        r_id, p, _, c = _extract_stream_demands(victim)
        tokens_demanded = p + c
        tokens_discarded = c if action == PreemptionAction.SEQUENTIAL_REGENERATE else 0
        recompute_tokens = 0 if action == PreemptionAction.SUSPEND else (p + c)
        recompute_lat_s = self.compute_recompute_latency(victim, action, batch_size)
        rec_tokens, rec_blocks = self.compute_memory_recovery(victim, action, block_size)

        return PreemptionCostEvaluation(
            action=action,
            victim_req_id=r_id,
            tokens_demanded=tokens_demanded,
            tokens_discarded=tokens_discarded,
            recompute_tokens=recompute_tokens,
            recompute_latency_s=round(recompute_lat_s, 6),
            memory_recovered_tokens=rec_tokens,
            memory_recovered_blocks=rec_blocks,
        )

    evaluate_victim = evaluate_preemption_cost

    def rank_victims(
        self,
        active_requests: Sequence[Union[ActiveRequest, Any]],
        rule: VictimRule = VictimRule.MAX_YIELD,
    ) -> List[Any]:
        """Ranks active streams in priority order for victim preemption according to rule."""
        if not active_requests:
            return []
        reqs = list(active_requests)
        if rule == VictimRule.LEAST_PROGRESS:
            reqs.sort(key=lambda r: _extract_stream_demands(r)[3])
        elif rule == VictimRule.MAX_YIELD:
            reqs.sort(key=lambda r: _extract_stream_demands(r)[3], reverse=True)
        elif rule == VictimRule.SRPT:
            reqs.sort(key=lambda r: max(0, _extract_stream_demands(r)[2] - _extract_stream_demands(r)[3]), reverse=True)
        elif rule == VictimRule.FIFO:
            reqs.sort(key=lambda r: getattr(r, "admit_step", getattr(r, "arrival_step", 0)) or 0)
        return reqs

    def select_victim(
        self,
        active_requests: Sequence[Union[ActiveRequest, Any]],
        rule: VictimRule = VictimRule.MAX_YIELD,
    ) -> Optional[Any]:
        """Selects the single best victim stream to preempt according to rule."""
        ranked = self.rank_victims(active_requests, rule=rule)
        return ranked[0] if ranked else None


_DEFAULT_COST_MODEL = PreemptionCostModel()


def compute_recompute_latency(
    request: Union[ActiveRequest, Any],
    action: PreemptionAction = PreemptionAction.SAVE_AND_REPREFILL,
    batch_size: int = 64,
    model: Optional[PreemptionCostModel] = None,
) -> float:
    return (model or _DEFAULT_COST_MODEL).compute_recompute_latency(request, action, batch_size)


def compute_memory_recovery(
    request: Union[ActiveRequest, Any],
    action: PreemptionAction = PreemptionAction.SAVE_AND_REPREFILL,
    block_size: Optional[int] = None,
    model: Optional[PreemptionCostModel] = None,
) -> Tuple[int, int]:
    return (model or _DEFAULT_COST_MODEL).compute_memory_recovery(request, action, block_size)


def evaluate_preemption_cost(
    victim: Union[ActiveRequest, Any],
    action: PreemptionAction = PreemptionAction.SAVE_AND_REPREFILL,
    batch_size: int = 64,
    block_size: Optional[int] = None,
    model: Optional[PreemptionCostModel] = None,
) -> PreemptionCostEvaluation:
    return (model or _DEFAULT_COST_MODEL).evaluate_preemption_cost(victim, action, batch_size, block_size)
