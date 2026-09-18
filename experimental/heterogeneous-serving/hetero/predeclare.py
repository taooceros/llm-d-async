"""Frozen predeclaration for the heterogeneous-serving evaluation.

Written and committed before any timing run.  Everything an outcome could be
argued into after the fact -- hypotheses, contrasts, the practically-useful
threshold, arm order, repetition count, exclusion rules and the analysis method
-- is fixed here.  The artifact records source and configuration hashes so that
a later run can be checked against what was declared.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any, Dict, List

from .experiment import ADMITTED_CAP, ARMS, BASE_POLICY, WARMUP_MAX_TOKENS, WARMUP_REQUESTS

PREDECLARATION_PATH = "results/hetero/predeclaration.json"

HYPOTHESES = [
    {
        "id": "H1",
        "name": "instance-mix benefit",
        "statement": (
            "With migration disabled, the static heterogeneous layout M (TP8+TP8+TP16) "
            "completes the same useful-token budget in less wall time than the strongest "
            "qualified homogeneous layout at the same 32 chips."
        ),
        "contrast": "M0 versus min(makespan of H, S)",
        "supports": "static heterogeneity plus capacity-aware gateway placement",
        "does_not_support": "migration",
    },
    {
        "id": "H2",
        "name": "incremental migration benefit",
        "statement": (
            "On the identical layout, routing policy and proxy path, enabling "
            "recompute migration reduces makespan relative to M0 after all online "
            "costs (proposal, quiescence, replay, commit, release)."
        ),
        "contrast": "MR versus M0",
        "supports": "the migration mechanism itself",
        "does_not_support": (
            "larger small-instance batches or earlier memory release on their own"
        ),
    },
    {
        "id": "H3",
        "name": "transport benefit",
        "statement": (
            "Host-staged KV transfer reduces makespan relative to recompute migration "
            "on the identical layout and migration policy."
        ),
        "contrast": "MK-H versus MR",
        "note": (
            "For this model a migrated token carries 384 KiB of KV "
            "(16 kv heads x 128 head_dim x 2 for K/V x 2 bytes x 48 layers), so a "
            "host-staged transfer can plausibly cost more than recomputing the same "
            "prefix.  The direction of this contrast is genuinely open."
        ),
    },
    {
        "id": "H4",
        "name": "system benefit",
        "statement": (
            "The frozen migration treatment beats the strongest tested and qualified "
            "equal-resource homogeneous baseline."
        ),
        "contrast": "frozen treatment versus min(makespan of H, S)",
        "note": (
            "Beating M0 but losing to a uniform layout demonstrates a useful mechanism "
            "inside a mixed layout, not an overall system advantage."
        ),
    },
]

FALSIFICATION = [
    "No committed migrations occur under the frozen policy on W0.",
    "The destination (large) becomes the bottleneck and its queue grows monotonically.",
    "Replayed tokens exceed 10% of useful tokens without a makespan reduction.",
    "Gateway control overhead, not backend execution, dominates the difference.",
    "Any arm fails the correctness assertions (missing/duplicate delivery, budget drift).",
    "The paired effect is smaller than the spread across repetitions of the same arm.",
]

MINIMUM_USEFUL_EFFECT = {
    "metric": "makespan reduction 1 - T_X/T_B on W0",
    "threshold_fraction": 0.05,
    "rationale": (
        "The treatment adds a central ownership protocol, a migration coordinator and "
        "a second engine shape to operate.  Below a 5% makespan reduction that "
        "operational cost is not repaid, so smaller effects are reported as measured "
        "but not claimed as practically useful."
    ),
    "declared_before_any_timing_run": True,
}

ANALYSIS = {
    "primary_metrics": [
        "end-to-end makespan T over the whole frozen manifest",
        "useful throughput U/T for the same useful output U",
    ],
    "secondary_metrics": [
        "useful tokens/s/chip",
        "sustained-interval throughput over the 20%-80% cumulative useful-token window",
        "preemptions, replayed tokens, migration counts and handoff latency",
        "per-instance scheduled batch and context distribution",
    ],
    "reporting_rule": (
        "Makespan reduction is reported as 1 - T_X/T_B and throughput increase as "
        "T_B/T_X - 1; the two percentages are never interchanged."
    ),
    "pairing": (
        "Repetition index pairs arms.  Paired differences are reported per repetition "
        "with median and full range; with three repetitions no p-value is computed and "
        "no historical noise range is imported."
    ),
    "initialization": (
        "Engine initialization is measured and reported separately from serving; the "
        "timed interval begins after warmup has drained and ends when the last request "
        "of the manifest is delivered."
    ),
}

EXCLUSION_RULES = [
    "A repetition is excluded only for an infrastructure fault: engine actor death, "
    "placement-group loss, gateway restart, or a Ray job failure unrelated to the arm.",
    "An excluded repetition is reported with its cause and re-run at the same index.",
    "No repetition is excluded for being slow, fast, or inconvenient.",
    "An arm that does not complete every request is reported as incomplete and is "
    "never used as winner evidence.",
]


def _source_hashes(root: pathlib.Path) -> Dict[str, str]:
    files = sorted(root.glob("hetero/**/*.py")) + [root / "scripts" / "hetero_ctl.py"]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.exists()
    }


def build_predeclaration(
    root: pathlib.Path,
    *,
    manifests: List[Dict[str, Any]],
    layouts: Dict[str, Any],
    engine_settings: Dict[str, Any],
    qualification: Dict[str, Any],
    arm_order: List[Dict[str, Any]],
    repetitions: int,
) -> Dict[str, Any]:
    payload = {
        "frozen_at": time.time(),
        "goal_document": "HETEROGENEOUS_LLMD_GATEWAY_GOAL.md",
        "hypotheses": HYPOTHESES,
        "falsification": FALSIFICATION,
        "minimum_useful_effect": MINIMUM_USEFUL_EFFECT,
        "analysis": ANALYSIS,
        "exclusion_rules": EXCLUSION_RULES,
        "arms": ARMS,
        "arm_order": arm_order,
        "repetitions": repetitions,
        "routing_and_migration_policy": BASE_POLICY,
        "admitted_population_cap": ADMITTED_CAP,
        "warmup": {
            "requests": WARMUP_REQUESTS,
            "max_tokens": WARMUP_MAX_TOKENS,
            "policy": "dispatched before the clock starts, drained to idle, ids never "
            "enter the timed manifest",
        },
        "engine_settings": engine_settings,
        "layouts": layouts,
        "manifests": manifests,
        "qualification": qualification,
        "measurement_contract": {
            "enable_prefix_caching": False,
            "sampling": "greedy (temperature=0); unsupported continuation features "
            "block migration rather than reset semantics",
            "ignore_eos": True,
            "reason_for_ignore_eos": (
                "fixed output budgets make useful-token totals identical across arms; "
                "prompt provenance stays real"
            ),
            "all_headline_runs_traverse_the_gateway": True,
            "instrumentation_identical_across_arms": True,
        },
        "source_hashes": _source_hashes(root),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["predeclaration_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return payload
