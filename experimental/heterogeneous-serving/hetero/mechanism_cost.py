"""Prospective matched-history native mechanism diagnostic, not gateway results.

Called by Main's RCM/Ray entrypoint against already resident, idle M engines.
No engine launch, native patch, synthetic history, shortened background budget,
transport fallback, or modeled/subtracted transfer cost is permitted here.
The protocol and prerequisite fidelity protocol are byte-pinned. All attempted
and unattempted trials remain in the report, including failures and cleanup ACKs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import socket
import time
import uuid
from typing import Any, Dict, Optional

import ray

from .manifest import _assign_unique, _tokenise_catalogue, load_manifest, manifest_path
from .qualify_continuation import (
    ContinuationQualificationSession, EXCLUDED_WORKLOADS,
    PROTOCOL_PATH as FIDELITY_PROTOCOL_PATH, PROTOCOL_SHA256 as FIDELITY_PROTOCOL_SHA256,
    SUFFIX_TOKENS, _digest, _save, _suffix_comparison,
)

PROTOCOL_PATH = "results/hetero/mechanism_cost_protocol.json"
PROTOCOL_SHA256 = "7bdc574f1e8c24fd2a560898a4fb8541d5edd825f330a8ff0237cafb06bf9f4a"
INSTANCES = ("small_a", "small_b", "large")


def _clock() -> Dict[str, float]:
    return {"wall_s": time.time(), "monotonic_s": time.monotonic(),
            "control_cpu_s": time.process_time(), "control_thread_cpu_s": time.thread_time()}


def _duration(start: Dict[str, float], end: Dict[str, float]) -> Dict[str, float]:
    return {key: end[key] - start[key] for key in start.keys() & end.keys()}


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _protocol(path: str, digest: str, directory: pathlib.Path, name: str) -> tuple:
    source = pathlib.Path(path)
    if not source.is_absolute():
        source = pathlib.Path(__file__).resolve().parents[1] / source
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    target = directory / name
    target.write_bytes(raw)
    if actual != digest:
        raise ValueError(f"prospective protocol bytes changed: {source}; refusing new criteria")
    return json.loads(raw), {"path": str(target), "sha256": actual, "bytes": len(raw)}


def _select_prompts(protocol: Dict[str, Any], fidelity: Dict[str, Any], catalogue_name: str,
                    manifest_root: Optional[pathlib.Path]) -> Dict[str, Any]:
    excluded, forbidden, manifests = set(), set(), {}
    for name in EXCLUDED_WORKLOADS:
        manifest = load_manifest(name, manifest_root)
        path = manifest_path(name, manifest_root)
        ids = {row["source_id"] for row in manifest["requests"]}
        excluded.update(ids)
        forbidden.update(tuple(row["prompt_token_ids"]) for row in manifest["requests"])
        manifests[name] = {"path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "manifest_sha256": manifest["manifest_sha256"], "source_ids": sorted(ids)}
    tokenised = _tokenise_catalogue(fidelity["model"], catalogue_name)
    background = protocol["background"]
    demands = [row["prompt_tokens"] for row in protocol["checkpoint_cases"]]
    demands += [background["prompt_tokens"]] * (len(INSTANCES) * background["requests_per_instance"])
    # Exclude token-identical manifest prompts even if the catalogue uses another
    # conversation ID. _assign_unique also rejects duplicates among these demands.
    prompt_lengths = set(demands)
    eligible = [row for row in tokenised["rows"] if row["id"] not in excluded
                and all(tuple(row["token_ids"][:size]) not in forbidden for size in prompt_lengths)]
    assignments, stats = _assign_unique(eligible, demands)
    if stats["shortfalls"] or len(assignments) != len(demands):
        raise ValueError("held-out real ShareGPT cannot satisfy exact foreground/background lengths")
    if (len({row["source_id"] for row in assignments}) != len(demands)
            or len({tuple(row["token_ids"]) for row in assignments}) != len(demands)):
        raise ValueError("foreground/background selection is not source- and token-unique")
    cases, backgrounds = [], {name: [] for name in INSTANCES}
    for index, (bound, wanted) in enumerate(zip(assignments, demands)):
        tokens = bound["token_ids"]
        if len(tokens) != wanted or tuple(tokens) in forbidden or bound["source_id"] in excluded:
            raise ValueError("real prompt selection violated token length or exclusion")
        row = {key: value for key, value in bound.items() if key != "token_ids"}
        row.update(prompt_token_ids=tokens, prompt_tokens=wanted, prompt_token_ids_sha256=_digest(tokens))
        if index < len(protocol["checkpoint_cases"]):
            case = protocol["checkpoint_cases"][index]
            row.update(case, case_id=f"p{wanted}-h{case['generated_history_tokens']}",
                       original_max_tokens=case["generated_history_tokens"] + SUFFIX_TOKENS)
            cases.append(row)
        else:
            slot = index - len(cases)
            instance = INSTANCES[slot // background["requests_per_instance"]]
            row.update(slot=slot % background["requests_per_instance"],
                       max_tokens=background["max_tokens"], temperature=background["temperature"],
                       ignore_eos=background["ignore_eos"])
            backgrounds[instance].append(row)
    return {"catalogue": catalogue_name, "catalogue_path": tokenised["path"],
            "catalogue_sha256": tokenised["sha256"], "model": fidelity["model"],
            "tokenizer": {"from_pretrained": fidelity["model"], "add_special_tokens": True,
                          "truncation": "real leading tokens, never padding"},
            "excluded_workloads": manifests, "excluded_source_ids": sorted(excluded),
            "cases": cases, "background": backgrounds,
            "matching_scope": "same 24 distinct background vectors/budgets in all trials; no assumed realized-load equality"}


class _CostSession(ContinuationQualificationSession):
    """Keep real RPC observations without changing native execution or ownership."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.rpc_timings: list[Dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def call(self, instance: str, method: str, **arguments: Any) -> Any:
        row = {"instance": instance, "method": method, "start": _clock(), "acknowledged": False,
               **{key: arguments[key] for key in ("request_id", "transaction_id") if key in arguments}}
        self.rpc_timings.append(row)
        try:
            result = super().call(instance, method, **arguments)
            row["acknowledged"] = True
            return result
        except BaseException as exc:
            row["error"] = _error(exc)
            raise
        finally:
            row["end"] = _clock()
            row["elapsed"] = _duration(row["start"], row["end"])


def _qualification(path: Optional[str], directory: pathlib.Path) -> Dict[str, Any]:
    if path is None:
        raise ValueError("qualification_report_path must identify a passed real native continuation report")
    raw = pathlib.Path(path).read_bytes()
    report = json.loads(raw)
    _save(directory, "prerequisite_qualification.json", report)
    if (report.get("qualification") != "real-native-continuation" or report.get("passed") is not True
            or report.get("protocol_sha256") != FIDELITY_PROTOCOL_SHA256
            or not report.get("passed_flags") or not all(report["passed_flags"].values())
            or not report.get("cleanup", {}).get("passed")):
        raise ValueError("native continuation mechanisms are not qualified under the unchanged fidelity protocol")
    registry_path = pathlib.Path(report["cleanup"]["registry_after"]["path"])
    registry_raw = registry_path.read_bytes()
    registry = json.loads(registry_raw)
    _save(directory, "qualified_registry.json", registry)
    if _digest(registry) != report["cleanup"]["registry_after"]["uncompressed_sha256"]:
        raise ValueError("qualification registry artifact changed")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "registry": registry}


def _qualified_residents(session: _CostSession, qualification: Dict[str, Any]) -> bool:
    prior = {row["instance_id"]: row for row in qualification["registry"]}
    return all(name in prior and all(prior[name][key] == session.descriptors[name][key] for key in
               ("generation", "layout_id", "tp_size", "model", "engine_actor_name", "ray_namespace"))
               for name in INSTANCES)


def _sample(session: _CostSession) -> Dict[str, Any]:
    start = _clock()
    refs = [session.actors[name].migration_candidates.remote(
        min_generated_tokens=0, min_context_tokens=0, min_remaining_tokens=0, limit=100000)
        for name in INSTANCES]
    rows = session.get(refs)
    end = _clock()
    observations = {}
    for name, row in zip(INSTANCES, rows):
        if row["instance_id"] != name or row["generation"] != session.descriptors[name]["generation"]:
            raise RuntimeError("background observation has stale native identity")
        observations[name] = row
    return {"start": start, "end": end, "instances": observations}


def _background_state(sample: Dict[str, Any], requests: list[Dict[str, Any]]) -> Dict[str, Any]:
    observed, missing, invalid = {}, [], []
    for request in requests:
        rid, name = request["request_id"], request["instance_id"]
        rows = sample["instances"][name]["candidates"]
        matching = [row for row in rows if row["request_id"] == rid]
        if len(matching) != 1:
            missing.append(rid)
            continue
        row = matching[0]
        observed[rid] = row
        if (row["prompt_tokens"] != request["prompt_tokens"]
                or not 0 < row["generated_tokens"] < request["max_tokens"]
                or row["remaining_tokens"] != request["max_tokens"] - row["generated_tokens"]
                or row["allocated_blocks"] <= 0 or row["unsupported_continuation"]):
            invalid.append(rid)
    return {"passed": bool(requests) and not missing and not invalid, "observed": observed,
            "missing_running_requests": missing, "invalid_or_exhausted": invalid,
            "scope": "native running, allocated and already generating; waiting requests never count"}


class _Background:
    def __init__(self, session: _CostSession, plan: Dict[str, Any], label: str) -> None:
        self.session, self.plan, self.label = session, plan, label
        self.requests: list[Dict[str, Any]] = []
        self.pending: Dict[Any, Dict[str, Any]] = {}
        self.results: Dict[str, Any] = {}
        self.errors: list[Dict[str, Any]] = []
        self.cancellations: list[Dict[str, Any]] = []
        self.readiness_observations: list[Dict[str, Any]] = []

    def start(self) -> None:
        for name in INSTANCES:
            for prompt in self.plan[name]:
                rid = f"{self.session.prefix}-{self.label}-bg-{name}-{prompt['slot']}"
                arguments = {key: prompt[key] for key in
                             ("prompt_token_ids", "max_tokens", "temperature", "ignore_eos")}
                row = {**prompt, "request_id": rid, "instance_id": name}
                self.session.state.claim(rid, name, self.session.descriptors[name]["generation"])
                self.requests.append(row)
                self.session.model_execution_attempted = True
                try:
                    ref = self.session.actors[name].generate.remote(request_id=rid, **arguments)
                    self.pending[ref] = row
                except BaseException as exc:
                    self.errors.append({"request_id": rid, "error": _error(exc)})
                    self._cancel(row)
                    raise

    def _cancel(self, row: Dict[str, Any]) -> None:
        try:
            receipt = self.session.cancel_owned(row["request_id"], row["instance_id"])
            if receipt.get("cancelled") is not True:
                raise RuntimeError("background cancellation lacks native cancelled ACK")
            self.cancellations.append(receipt)
        except Exception as exc:
            self.session.unresolved_request(row["request_id"], exc)
            self.errors.append({"request_id": row["request_id"], "cleanup_error": _error(exc)})

    def ready(self) -> Dict[str, Any]:
        deadline = time.monotonic() + self.session.timeout_s
        attempts = self.readiness_observations
        while True:
            sample = _sample(self.session)
            attempts.append(sample)
            decision = _background_state(sample, self.requests)
            completed, _ = self.session.ray.wait(list(self.pending), num_returns=len(self.pending), timeout=0)
            if completed:
                raise RuntimeError("background completed before all 24 requests were observed generating")
            if decision["passed"]:
                return {"sample": sample, "decision": decision}
            if time.monotonic() >= deadline:
                raise TimeoutError(f"background readiness failed: {decision}")
            time.sleep(0.05)

    def drain(self) -> Dict[str, Any]:
        """Always attempt every submitted fixed budget; cancellation is failure cleanup only."""
        deadline = time.monotonic() + self.session.timeout_s
        try:
            while self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("fixed-budget background drain timed out; trial remains incomplete")
                done, _ = self.session.ray.wait(list(self.pending), num_returns=1, timeout=remaining)
                if not done:
                    raise TimeoutError("fixed-budget background drain timed out; trial remains incomplete")
                for ref in done:
                    row = self.pending[ref]
                    try:
                        result = self.session.get(ref)
                        self.results[row["request_id"]] = result
                        if (result["request_id"] != row["request_id"] or result["instance_id"] != row["instance_id"]
                                or result["prompt_tokens"] != row["prompt_tokens"]
                                or result["output_tokens"] != row["max_tokens"]
                                or len(result["output_token_ids"]) != row["max_tokens"]
                                or result["finish_reason"] != "length" or result.get("migrated_transaction")):
                            raise RuntimeError("background violated its original finite output contract")
                        self.session.state.finish(row["request_id"], result["output_tokens"])
                    except Exception as exc:
                        self.errors.append({"request_id": row["request_id"], "error": _error(exc)})
                        self._cancel(row)
                    finally:
                        self.pending.pop(ref)
        except Exception as exc:
            self.errors.append({"error": _error(exc)})
        finally:
            for ref, row in list(self.pending.items()):
                self._cancel(row)
                self.pending.pop(ref)
        expected = sum(len(rows) for rows in self.plan.values())
        return {"passed": len(self.requests) == len(self.results) == expected and not self.errors,
                "submitted": len(self.requests), "completed": len(self.results), "expected": expected,
                "errors": self.errors, "cancellation_receipts": self.cancellations,
                "results": self.results}


def _native_idle(session: _CostSession) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"passed": False, "idle": {}, "native_state": {}, "errors": []}
    for name in session.actors:
        try:
            idle = session.call(name, "await_idle", timeout_s=0 if session.unresolved else session.timeout_s)
            state = session.call(name, "migration_state")
            evidence["idle"][name], evidence["native_state"][name] = idle, state
        except Exception as exc:
            evidence["errors"].append({"instance_id": name, "error": _error(exc)})
    evidence["passed"] = (not session.unresolved and not evidence["errors"]
        and set(evidence["idle"]) == set(INSTANCES)
        and all(row["idle"] and not row["draining"] and not row["pending_batches"]
                and not row["scheduler_has_work"]
                and row["free_blocks"] == session.initial_idle[name]["free_blocks"]
                for name, row in evidence["idle"].items())
        and all(not row["quiesced"] and not row["prepared"] and not row["active_continuations"]
                for row in evidence["native_state"].values()))
    return evidence


def _start_capture(session: _CostSession, attempted: list[str]) -> None:
    for name in INSTANCES:
        # Record before RPC: a lost start ACK must still be cleaned up.
        attempted.append(name)
        session.call(name, "start_iteration_capture")


def _capture_audit(captures: Dict[str, Any]) -> bool:
    if set(captures) != set(INSTANCES):
        return False
    for payload in captures.values():
        audit, rows = payload["audit"], payload["rows"]
        if (audit["rows"] != len(rows) or audit["retired_rows"] != len(rows)
                or any(audit[key] for key in ("open_rows", "unmatched_retirements", "inflight_batches"))
                or len({row["iteration"] for row in rows}) != len(rows)
                or any(row.get("completed_at_s") is None for row in rows)):
            return False
    return True


def _stop_capture(session: _CostSession, attempted: list[str], directory: pathlib.Path,
                  label: str, idle: Dict[str, Any]) -> tuple:
    from .experiment import _write_iterations

    captures, errors = {}, []
    for name in attempted:
        if not idle.get("idle", {}).get(name, {}).get("idle"):
            errors.append({"instance_id": name,
                           "error": "capture left resident: native ownership/work unresolved; cannot safely stop"})
            continue
        try:
            captures[name] = session.call(name, "stop_iteration_capture")
        except Exception as exc:
            errors.append({"instance_id": name, "error": _error(exc)})
    raw = _save(directory, f"{label}-native-captures.json.gz", captures)
    evidence: Dict[str, Any] = {"passed": False, "raw_artifact": raw, "errors": errors}
    try:
        evidence["iterations"] = _write_iterations(directory, label, 1, captures)
        evidence["passed"] = not errors and _capture_audit(captures) and evidence["iterations"]["one_to_one"]
    except Exception as exc:
        evidence["errors"].append({"error": _error(exc)})
    return evidence, captures


def _clock_offset(sample: Dict[str, Any], instance: str) -> list[float]:
    remote = sample["instances"][instance]["observed_at_s"]
    return [remote - sample["end"]["wall_s"], remote - sample["start"]["wall_s"]]


def _admission_observation(rows: list[Dict[str, Any]], ready_at_s: float,
                           result: Dict[str, Any]) -> Dict[str, Any]:
    first = result["first_token_at_s"]
    window = [row for row in rows if ready_at_s <= row["scheduled_at_s"] <= first]
    admissions = [row for row in window if row.get("new_requests", 0)]
    identifiable = (len(admissions) == 1 and admissions[0]["new_requests"] == 1
                    and not any(row.get("preempted_request_ids") for row in window))
    stream_created = first - result["ttft_s"]
    return {"identified": identifiable, "native_stream_created_at_s": stream_created,
        "first_native_scheduled_iteration": admissions[0]["iteration"] if identifiable else None,
        "admission_to_first_schedule_return_s": admissions[0]["scheduled_at_s"] - stream_created
                                                  if identifiable else None,
        "scope": "same destination wall clock, stream registration to first schedule-return timestamp; includes frontend admission and scheduler queue/control, not device compute",
        "identification": "all eight backgrounds already running at readiness, exactly one fresh admission before foreground first token, and no native preemption in that interval; otherwise attribution remains unavailable"}


def _metrics(transfer: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any],
             calls: list[Dict[str, Any]], captures: Dict[str, Any],
             requests: list[Dict[str, Any]]) -> Dict[str, Any]:
    marks, result = transfer["timings"], transfer["result"]
    proposal, released = marks["proposal"], marks["source_release_acknowledged"]
    first = result["first_token_at_s"]
    if first is None or not math.isfinite(first):
        raise RuntimeError("native destination did not report an actual first-token timestamp")
    destination = transfer["destination"]
    offsets = {name: [min(_clock_offset(before, name)[0], _clock_offset(after, name)[0]),
                      max(_clock_offset(before, name)[1], _clock_offset(after, name)[1])]
               for name in INSTANCES}
    first_bounds = [first - offsets[destination][1], first - offsets[destination][0]]
    # These are observed clock envelopes, not a point clock correction or a
    # priced transport subtraction. They assume no unobserved clock step outside
    # the before/after envelopes. Raw clocks and their bounds are retained.
    handoff_end_lower = max(released["wall_s"], first_bounds[0])
    handoff_end_upper = max(released["wall_s"], first_bounds[1])
    before_state, after_state = _background_state(before, requests), _background_state(after, requests)
    progress = {}
    for request in requests:
        rid = request["request_id"]
        left, right = before_state["observed"].get(rid), after_state["observed"].get(rid)
        progress[rid] = {"instance_id": request["instance_id"],
                         "before_generated_tokens": left["generated_tokens"] if left else None,
                         "after_generated_tokens": right["generated_tokens"] if right else None,
                         "useful_output_delta": right["generated_tokens"] - left["generated_tokens"]
                                                if left and right else None}
    iteration_windows = {}
    for name, payload in captures.items():
        lo, hi = offsets[name]
        rows = [row for row in payload["rows"]
                if row["scheduled_at_s"] - hi >= proposal["wall_s"]
                and row["completed_at_s"] - lo <= handoff_end_lower]
        iteration_windows[name] = {"fully_inside_handoff_iterations": [row["iteration"] for row in rows],
            "emitted_output_tokens": sum(row.get("emitted_output_tokens", 0) for row in rows),
            "prefill_tokens": sum(row.get("prefill_tokens", 0) for row in rows),
            "replayed_tokens": sum(row.get("replayed_tokens", 0) for row in rows),
            "scope": "small_b is exclusively background; small_a/large rows may contain foreground"}
    prepared = transfer["preparation"]
    native_evidence = prepared["evidence"]
    destination_rows = captures[destination]["rows"]
    replay_rows = [row for row in destination_rows if row.get("replayed_tokens", 0)]
    replayed = sum(row["replayed_tokens"] for row in replay_rows)
    admission = _admission_observation(
        destination_rows, before["instances"][destination]["observed_at_s"], result)
    expected_replay = transfer["snapshot"]["num_computed_tokens"] if transfer["mechanism"] == "recompute" else 0
    release_calls = [row for row in calls if row["method"] == "release_source"]
    phases = {f"{left}_to_{right}": _duration(marks[left], marks[right]) for left, right in (
        ("proposal", "reservation_start"), ("reservation_start", "reservation_acknowledged"),
        ("reservation_acknowledged", "source_quiesced"), ("source_quiesced", "prepare_start"),
        ("prepare_start", "prepared"), ("prepared", "ownership_committed"),
        ("ownership_committed", "source_release_acknowledged"),
        ("source_release_acknowledged", "continuation_completed"))}
    return {"proposal_to_first_destination_token_wall_s": first - proposal["wall_s"],
        "first_token_wall_scope": "destination time.time minus coordinator time.time; raw, uncorrected cross-host clocks",
        "proposal_to_first_destination_token_clock_envelope_s": [x - proposal["wall_s"] for x in first_bounds],
        "proposal_to_source_release_ack_s": released["monotonic_s"] - proposal["monotonic_s"],
        "native_release_observation": {"receipt": transfer["transaction"]["release_receipt"],
            "rpc_intervals": release_calls,
            "scope": "native release acknowledged inside RPC interval, not an invented native event timestamp"},
        "local_phase_durations": phases, "rpc_observations": calls,
        "rpc_timing_scope": "client-observed wall round trips include queue/network/native work; no unobserved pure-RPC residual is manufactured",
        "control_cpu_scope": "process_time is this coordinator process including Ray threads; thread_time is its orchestration thread. Native actor/worker CPU is not exposed.",
        "native_quiesce_s": transfer["snapshot"]["quiesce_s"],
        "native_prepare_s": prepared["prepare_s"],
        "native_export": native_evidence.get("export"), "native_import_workers": native_evidence.get("import", []),
        "transport_observability": "export_s includes queue drain, device gather and host assembly; prepare RPC also includes nested Ray object transfer; per-worker scatter_s is real import. Separate D2H/assembly/network and remote CPU are not exposed; no residual is labeled transfer cost.",
        "logical_payload_bytes": prepared["transferred_bytes"],
        "total_wire_traffic_bytes": None, "source_worker_export_bytes": None,
        "byte_scope": "logical exported host-array payload nbytes, not aggregate network traffic. Source collective export and destination collective import can replicate payloads across workers. Native import receipts retain observed per-worker array-write bytes and head conversions; source worker byte breakdown and total wire traffic are not exposed. Full wall cost remains charged.",
        "cache_coverage": transfer["cache_coverage"],
        "actual_native_replayed_positions": replayed, "expected_computed_frontier_replay": expected_replay,
        "replay_excludes_uncomputed_tail": replayed == expected_replay == prepared["rebuild_tokens"],
        "native_replay_iterations": [{key: row.get(key) for key in
            ("iteration", "scheduled_at_s", "completed_at_s", "scheduled_to_output_s", "replayed_tokens",
             "prefill_tokens", "decode_tokens", "scheduled_requests")} for row in replay_rows],
        "native_replay_schedule_to_last_retirement_s": (
            max(row["completed_at_s"] for row in replay_rows) - min(row["scheduled_at_s"] for row in replay_rows)
            if replay_rows else None),
        "replay_timing_scope": "actual native iterations containing replay, including mixed background work and queueing; not pure replay device or remote CPU cost",
        "destination_queue_admission": admission,
        "destination_queue_and_replay_scope": "ordinary native stream creation-to-first-token ttft_s includes queue/admission, replay and first useful decode; admission attribution is reported only when unambiguous",
        "destination_activation_to_first_token_s": result["ttft_s"],
        "native_first_token_at_s": first, "clock_offset_envelopes_s": offsets,
        "handoff_window": {"start_coordinator_wall_s": proposal["wall_s"],
            "end_coordinator_wall_bounds_s": [handoff_end_lower, handoff_end_upper],
            "definition": "proposal to later of destination first token and source release ACK; conservative fully enclosed native iterations",
            "clock_assumption": "native wall offset remains within observed before/after RPC envelopes; no point correction"},
        "background_before": before_state, "background_after": after_state, "background_progress": progress,
        "background_progress_scope": "before proposal to after full 32-token suffix completion, not mislabeled as handoff-only",
        "not_exhausted_during_handoff_proven": before_state["passed"] and after_state["passed"],
        "not_exhausted_proof": "all fixed-budget requests still native-running with output remaining after suffix completion; absence/preemption is unproven, never assumed live",
        "native_iterations_inside_handoff": iteration_windows,
        "unrelated_small_b_progress_during_handoff": iteration_windows.get("small_b", {}).get("emitted_output_tokens", 0) > 0,
        "nonnegative_observed_progress": all(row["useful_output_delta"] is not None and row["useful_output_delta"] >= 0
                                             for row in progress.values())}


def _run_trial(session: _CostSession, case: Dict[str, Any], reference: Dict[str, Any],
               plan: Dict[str, Any], row: Dict[str, Any], directory: pathlib.Path) -> bool:
    label = row["trial_id"]
    background = _Background(session, plan, label)
    attempted, checkpoint, captures = [], None, {}
    rpc_start = len(session.rpc_timings)
    row.update(status="running", started=_clock(), errors=[], passed=False)
    row["background_input_sha256"] = _digest(plan)
    _save(directory, f"{label}-trial.json.gz", row)
    try:
        _start_capture(session, attempted)
        checkpoint = session.start_checkpoint(case, label)
        background.start()
        row["background_readiness"] = background.ready()
        row["transfer"] = session.transfer_checkpoint(checkpoint, destination="large", mechanism=row["mechanism"],
            reference_token_ids=reference["output_token_ids"], reference_text=reference["text"], diagnostic=False)
        row["background_after_transfer"] = _sample(session)
    except Exception as exc:
        row["errors"].append(_error(exc))
    finally:
        if checkpoint is not None and not checkpoint["resolved"]:
            row["checkpoint_cleanup"] = session.abort_checkpoint(checkpoint)
        # Never shorten healthy background just because the foreground failed.
        drained = background.drain()
        row["background_drain"] = {key: value for key, value in drained.items() if key != "results"}
        row["native_idle"] = _native_idle(session)
        row["capture"], captures = _stop_capture(session, attempted, directory, label, row["native_idle"])
        row["rpc_timings"] = session.rpc_timings[rpc_start:]
        # Native cleanup precedes artifact writes: a disk error must not strand
        # an otherwise resolvable request or leave healthy capture enabled.
        row["background_results_artifact"] = _save(directory, f"{label}-background-results.json.gz", drained)
        row["background_requests_artifact"] = _save(directory, f"{label}-background-inputs.json.gz", background.requests)
        row["readiness_observations_artifact"] = _save(
            directory, f"{label}-readiness.json.gz", background.readiness_observations)
    transfer = row.get("transfer", {})
    try:
        if transfer.get("result") and row.get("background_after_transfer"):
            row["metrics"] = _metrics(transfer, row["background_readiness"]["sample"],
                row["background_after_transfer"], row["rpc_timings"], captures, background.requests)
    except Exception as exc:
        row["errors"].append(_error(exc))
    metrics = row.get("metrics", {})
    expected_emitted = {name: sum(prompt["max_tokens"] for prompt in plan[name]) for name in INSTANCES}
    expected_emitted["small_a"] += case["generated_history_tokens"]
    expected_emitted["large"] += SUFFIX_TOKENS
    observed_emitted = {name: sum(native.get("emitted_output_tokens", 0) for native in capture["rows"])
                        for name, capture in captures.items()}
    row["useful_token_audit"] = {"expected": expected_emitted, "observed": observed_emitted,
                                  "passed": observed_emitted == expected_emitted}
    row["passed_flags"] = {
        "native_transfer_and_original_contract": bool(transfer.get("passed")),
        "ordinary_path_without_top32": transfer.get("diagnostic") is False and bool(
            transfer.get("output_contract", {}).get("ordinary_path_without_top32")),
        "all_background_admitted_and_generating": bool(row.get("background_readiness", {}).get("decision", {}).get("passed")),
        "all_background_fixed_budgets_completed": drained["passed"],
        "background_not_exhausted_during_handoff": bool(metrics.get("not_exhausted_during_handoff_proven")),
        "unrelated_small_b_useful_progress_during_handoff": bool(metrics.get("unrelated_small_b_progress_during_handoff")),
        "observed_background_progress_consistent": bool(metrics.get("nonnegative_observed_progress")),
        "native_replay_frontier_audited": bool(metrics.get("replay_excludes_uncomputed_tail")),
        "all_native_iterations_strictly_retired": row["capture"]["passed"],
        "useful_tokens_match_all_captured_iterations": row["useful_token_audit"]["passed"],
        "native_pages_idle_and_ownership_resolved": row["native_idle"]["passed"],
    }
    row["passed"] = not row["errors"] and all(row["passed_flags"].values())
    row.update(status="completed" if row["passed"] else "failed", finished=_clock())
    row["artifact"] = _save(directory, f"{label}-trial.json.gz", row)
    return row["native_idle"]["passed"] and row["capture"]["passed"]


def _pair(trials: list[Dict[str, Any]]) -> Dict[str, Any]:
    snapshots = [row.get("transfer", {}).get("snapshot") for row in trials]
    keys = ("prompt_token_ids", "generated_token_ids", "generated_text", "sampling", "num_computed_tokens", "remaining_tokens")
    matched = (len(snapshots) == 2 and all(snapshots)
               and all(key in snapshots[0] and key in snapshots[1] and snapshots[0][key] == snapshots[1][key]
                       for key in keys))
    backgrounds = len(trials) == 2 and all(row.get("background_input_sha256") for row in trials) and (
        trials[0]["background_input_sha256"] == trials[1]["background_input_sha256"])
    result = {"trial_ids": [row["trial_id"] for row in trials], "matched_real_prompt_history_and_budget": bool(matched),
              "byte_identical_background_vectors_and_budgets": bool(backgrounds), "passed": False}
    outputs = [row.get("transfer", {}).get("result", {}).get("new_output_token_ids") for row in trials]
    if len(outputs) == 2 and all(value is not None for value in outputs):
        result["same_destination_suffix_comparison"] = _suffix_comparison(outputs[0], outputs[1])
    result["passed"] = (matched and backgrounds and all(row["passed"] for row in trials)
                        and result.get("same_destination_suffix_comparison", {}).get("passed", False))
    return result


def run_mechanism_cost(protocol_path: str = PROTOCOL_PATH, *, qualification_report_path: Optional[str] = None,
                       evaluation_protocol_path: Optional[str] = None,
                       manifest_root: Optional[str] = None, artifact_root: Optional[str] = None,
                       catalogue_name: str = "large", timeout_s: float = 900.0) -> Dict[str, Any]:
    """Run all 18 trials only inside Main's already initialized RCM/Ray job.

    A returned report is not a pass: inspect passed_flags and cleanup.unresolved.
    The required qualification report must cover these still-resident engine
    generations. No runtime is started here and no failed trial is discarded.
    """
    root = pathlib.Path(artifact_root or os.environ.get("OMP_HETERO_ARTIFACTS", "/tmp/omp_hetero/artifacts"))
    directory = root / f"mechanism-cost-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    report: Dict[str, Any] = {"diagnostic": "real-native-matched-history-mechanism-cost", "passed": False,
        "artifact_dir": str(directory), "protocol_path": str(protocol_path), "errors": [], "trials": [], "pairs": [],
        "references": [], "hardware_model_execution_started": False, "hardware_model_execution_attempted": False,
        "evidence_scope": "prospective direct-engine mechanism diagnostic under fixed real background; not a gateway headline/system comparison",
        "clock_origins": {"coordinator": {"hostname": socket.gethostname(), "pid": os.getpid(), "start": _clock()},
            "local_marks": "time.time Unix wall, time.monotonic same-process elapsed, process_time local control CPU, thread_time local orchestration CPU",
            "native_timestamps": "actor/worker host time.time; native durations use actor monotonic or worker perf_counter; raw cross-host wall clocks are not presumed synchronized",
            "observer_overhead": "native iteration capture, ownership journaling and local RPC evidence are included, never subtracted; no top32 collection; background samples occur outside measured handoff"}}
    session, selection = None, None
    try:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive; timeout retains a failed trial")
        protocol, report["protocol_artifact"] = _protocol(protocol_path, PROTOCOL_SHA256, directory, "prospective_protocol.json")
        report["protocol_sha256"] = PROTOCOL_SHA256
        fidelity, report["fidelity_protocol_artifact"] = _protocol(
            FIDELITY_PROTOCOL_PATH, FIDELITY_PROTOCOL_SHA256, directory, "fidelity_protocol.json")
        for case in protocol["checkpoint_cases"]:
            case_id = f"p{case['prompt_tokens']}-h{case['generated_history_tokens']}"
            for repetition, order in enumerate(protocol["paired_mechanism_order"], 1):
                for index, mechanism in enumerate(order):
                    report["trials"].append({"trial_id": f"{case_id}-r{repetition}-{index + 1}-{mechanism}",
                        "case_id": case_id, "repetition": repetition, "order_index": index,
                        "mechanism": mechanism, "status": "not_started", "passed": False})
        from .experiment import _frozen_cohort, _loaded_runtime

        report["evaluation_cohort"] = _frozen_cohort(evaluation_protocol_path, "MR")
        _frozen_cohort(evaluation_protocol_path, "MK-H")
        from .gateway.lifecycle import find_gateway
        gateway = find_gateway()
        if gateway is None:
            raise RuntimeError("qualified mechanism cost requires the resident M gateway")
        status = ray.get(gateway.status.remote(), timeout=120)
        if status["layout_id"] != "M":
            raise RuntimeError("mechanism cost requires the qualified M layout")
        report["loaded_runtime"] = _loaded_runtime(
            status, report["evaluation_cohort"]["qualified_runtime"]
        )
        qualification = _qualification(qualification_report_path, directory)
        report["qualification"] = qualification
        selection = _select_prompts(protocol, fidelity, catalogue_name, pathlib.Path(manifest_root) if manifest_root else None)
        report["prompt_selection_artifact"] = _save(directory, "selected_real_prompts.json.gz", selection)
        session = _CostSession(fidelity, directory, timeout_s)
        session.attach()
        report["qualified_resident_generations"] = _qualified_residents(session, qualification)
        if not report["qualified_resident_generations"]:
            raise RuntimeError("resident descriptor generation/build ownership differs from fidelity qualification")
        report["clock_origins"]["native_descriptors"] = session.descriptors
        report["clock_origins"]["native_identity_artifacts"] = {
            name: str(directory / f"identity-{name}.json") for name in INSTANCES}
        report["clock_origins"]["worker_layout_artifacts"] = {
            name: str(directory / f"kv-layout-{name}.json.gz") for name in INSTANCES}
        for case in selection["cases"]:
            label, attempted = f"{case['case_id']}-uninterrupted-reference", []
            reference_row: Dict[str, Any] = {"case_id": case["case_id"], "passed": False}
            report["references"].append(reference_row)
            try:
                _start_capture(session, attempted)
                reference = session.owned_generation(session.source, "generate", label,
                    prompt_token_ids=case["prompt_token_ids"], max_tokens=case["original_max_tokens"],
                    temperature=0.0, ignore_eos=True)
                reference_row["result_artifact"] = _save(directory, f"{label}.json.gz", reference)
                reference_row["passed"] = (reference["output_tokens"] == len(reference["output_token_ids"]) == case["original_max_tokens"]
                    and reference["prompt_tokens"] == case["prompt_tokens"] and reference["finish_reason"] == "length")
            finally:
                reference_row["native_idle"] = _native_idle(session)
                reference_row["capture"], _ = _stop_capture(session, attempted, directory, label, reference_row["native_idle"])
            if not (reference_row["passed"] and reference_row["native_idle"]["passed"] and reference_row["capture"]["passed"]):
                raise RuntimeError("authentic uninterrupted reference or its strict native capture failed")
            for row in (item for item in report["trials"] if item["case_id"] == case["case_id"]):
                safe = _run_trial(session, case, reference, selection["background"], row, directory)
                _save(directory, "report.json", report)
                if not safe:
                    raise RuntimeError("fail-closed: native cleanup/capture unresolved; remaining trials retained unattempted")
        report["exclusion_manifests_unchanged"] = all(
            hashlib.sha256(pathlib.Path(row["path"]).read_bytes()).hexdigest() == row["file_sha256"]
            for row in selection["excluded_workloads"].values())
    except Exception as exc:
        report["errors"].append(_error(exc))
    finally:
        if session is not None:
            report["hardware_model_execution_started"] = session.model_execution_started
            report["hardware_model_execution_attempted"] = session.model_execution_attempted
            try:
                report["cleanup"] = session.close()
            except Exception as exc:
                report["cleanup"] = {"passed": False, "unresolved": session.unresolved, "error": _error(exc)}
                report["errors"].append(_error(exc))
        else:
            report["cleanup"] = {"passed": False, "not_started": True, "unresolved": []}
    for row in report["trials"]:
        if row["status"] == "not_started":
            row["not_started_reason"] = report["errors"][-1] if report["errors"] else "run did not reach this declared trial"
        elif row["status"] == "running":
            row["status"] = "incomplete"
            row.setdefault("errors", []).extend(report["errors"])
    for case_id, repetition in dict.fromkeys((row["case_id"], row["repetition"]) for row in report["trials"]):
        paired = [row for row in report["trials"] if row["case_id"] == case_id and row["repetition"] == repetition]
        report["pairs"].append({"case_id": case_id, "repetition": repetition, **_pair(paired)})
    report["passed_flags"] = {
        "prospective_protocol_unchanged": report.get("protocol_sha256") == PROTOCOL_SHA256,
        "qualified_still_resident_native_paths": report.get("qualified_resident_generations", False),
        "held_out_evaluation_manifests_unchanged": report.get("exclusion_manifests_unchanged", False),
        "all_three_authentic_references": len(report["references"]) == 3 and all(row["passed"] for row in report["references"]),
        "all_18_declared_trials_passed": len(report["trials"]) == 18 and all(row["passed"] for row in report["trials"]),
        "all_nine_ab_ba_ab_pairs_matched": len(report["pairs"]) == 9 and all(row["passed"] for row in report["pairs"]),
        "acknowledged_cleanup_and_resident_preservation": report["cleanup"]["passed"],
    }
    report["passed"] = not report["errors"] and all(report["passed_flags"].values())
    report["report_artifact"] = _save(directory, "report.json", report)
    return report
