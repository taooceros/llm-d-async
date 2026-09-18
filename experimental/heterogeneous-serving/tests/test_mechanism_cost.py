"""Control/error/cleanup contracts only; no model or performance simulation."""

import copy
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hetero.gateway.state import GatewayState
from hetero.mechanism_cost import (
    INSTANCES, PROTOCOL_PATH, PROTOCOL_SHA256, _Background, _background_state,
    _admission_observation, _capture_audit, _pair, _protocol, _qualification, _select_prompts,
)
from hetero.qualify_continuation import ContinuationQualificationSession


class TransferContractSession(ContinuationQualificationSession):
    """Native receipt fixture for ownership decisions, never a timing workload."""

    def __init__(self, directory, include_diagnostics=False):
        self.directory = pathlib.Path(directory)
        self.state = GatewayState(str(self.directory / "ledger.jsonl"))
        self.source, self.same, self.cross = INSTANCES
        self.descriptors = {name: {"generation": 1} for name in INSTANCES}
        self.layouts = {self.source: [{"groups": [{"layers": [{"block_size": 2, "spec_type": "Full"}]}]}]}
        self.timeout_s = 1
        self.unresolved = []
        self.include_diagnostics = include_diagnostics
        self.reference = list(range(34))
        self.case = {"case_id": "contract", "prompt_token_ids": [90, 91], "prompt_tokens": 2,
                     "generated_history_tokens": 2, "original_max_tokens": 34}
        self.checkpoint = {"request_id": "contract-rid", "case": self.case, "resolved": False,
                           "transaction_id": None, "native_checkpoint": {}, "source_reference": "source"}
        self.state.claim("contract-rid", self.source, 1)

    def get(self, reference):
        return {"output_tokens": 2, "output_token_ids": self.reference[:2], "prompt_tokens": 2,
                "finish_reason": "migrated", "text": "prefix"}

    def event(self, *args, **kwargs):
        pass

    def call(self, instance, method, **kwargs):
        if method == "kv_capacity":
            return {"total_blocks": 100, "free_blocks": 99, "block_size": 2, "num_running": 0, "num_waiting": 0}
        if method == "quiesce":
            return {**kwargs, "prompt_token_ids": self.case["prompt_token_ids"],
                    "generated_token_ids": self.reference[:2], "generated_text": "prefix",
                    "prompt_len": 2, "generated_len": 2, "num_computed_tokens": 3,
                    "sampling": {"max_tokens": 34, "temperature": 0, "ignore_eos": True},
                    "remaining_tokens": 32, "unsupported_continuation": [], "snapshot_at_s": 0,
                    "kv_addressing": {"num_computed_tokens": 3, "groups": [{"group_id": 0,
                        "logical_block_indices": [0, 1], "valid_token_ranges": [[0, 2], [2, 3]]}]}}
        transaction_id = kwargs.get("transaction_id") or kwargs["payload"]["transaction_id"]
        receipt = {"instance_id": instance, "generation": 1, "transaction_id": transaction_id}
        if method == "reserve_continuation":
            return {**receipt, "accepted": True, "native_reserved_blocks": 18}
        if method == "prepare_continuation":
            return {**receipt, "accepted": True, "transferred_bytes": 0, "num_computed_tokens": 0,
                    "rebuild_tokens": 3, "evidence": {}}
        if method == "commit_continuation":
            return {**receipt, "activated": True}
        if method == "release_source":
            return {**receipt, "released": True}
        if method == "await_continuation":
            result = {**receipt, "request_id": "contract-rid", "new_output_tokens": 32,
                      "new_output_token_ids": self.reference[2:], "finish_reason": "length",
                      "full_output_text": "prefixsuffix", "text": "suffix"}
            if self.include_diagnostics:
                result.update(first_token_id=2, first_token_logprobs={2: -1.0},
                              logprobs_origin="prepared_native_continuation")
            return result
        raise AssertionError(method)


class MechanismCostContracts(unittest.TestCase):
    def test_ordinary_continuation_without_top32_preserves_ownership_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            session = TransferContractSession(directory)
            result = session.transfer_checkpoint(session.checkpoint, destination="large", mechanism="recompute",
                                                 reference_token_ids=session.reference, diagnostic=False)
            self.assertTrue(result["passed"], result.get("error"))
            self.assertTrue(session.checkpoint["resolved"])
            self.assertEqual(session.state.owner("contract-rid").committed_offset, 34)
            session.state.close()

    def test_fidelity_default_still_rejects_missing_prepared_diagnostics_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            session = TransferContractSession(directory)
            result = session.transfer_checkpoint(session.checkpoint, destination="large", mechanism="recompute",
                                                 reference_token_ids=session.reference)
            self.assertFalse(result["passed"])
            self.assertTrue(result["cleanup"]["acknowledged"])
            self.assertTrue(session.checkpoint["resolved"])
            session.state.close()

    def test_ordinary_cost_result_contaminated_by_top32_is_ineligible(self):
        with tempfile.TemporaryDirectory() as directory:
            session = TransferContractSession(directory, include_diagnostics=True)
            result = session.transfer_checkpoint(session.checkpoint, destination="large", mechanism="recompute",
                                                 reference_token_ids=session.reference, diagnostic=False)
            self.assertFalse(result["passed"])
            self.assertTrue(session.checkpoint["resolved"])
            session.state.close()

    def test_nonboolean_diagnostic_rejected_before_consuming_checkpoint(self):
        session = ContinuationQualificationSession.__new__(ContinuationQualificationSession)
        with self.assertRaises(ValueError):
            session.transfer_checkpoint({}, destination="large", mechanism="recompute",
                                        reference_token_ids=[], diagnostic="false")

    def test_changed_prospective_protocol_is_rejected_without_discarding_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            raw = pathlib.Path(PROTOCOL_PATH).read_bytes() + b"\n"
            source = root / "changed.json"
            source.write_bytes(raw)
            with self.assertRaises(ValueError):
                _protocol(str(source), PROTOCOL_SHA256, root, "evidence.json")
            self.assertEqual((root / "evidence.json").read_bytes(), raw)

    def test_failed_qualification_cannot_be_used_for_cost_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "failed.json"
            source.write_text(json.dumps({"passed": False, "qualification": "real-native-continuation"}))
            with self.assertRaises(ValueError):
                _qualification(str(source), root)
            self.assertFalse(json.loads((root / "prerequisite_qualification.json").read_text())["passed"])

    def test_waiting_or_exhausted_background_is_not_readiness(self):
        requests = [{"request_id": "a", "instance_id": "small_b", "prompt_tokens": 256, "max_tokens": 4096}]
        native = {"request_id": "a", "prompt_tokens": 256, "generated_tokens": 1, "remaining_tokens": 4095,
                  "allocated_blocks": 2, "unsupported_continuation": []}
        sample = {"instances": {"small_b": {"candidates": [native]}}}
        self.assertTrue(_background_state(sample, requests)["passed"])
        sample["instances"]["small_b"]["candidates"] = []
        self.assertFalse(_background_state(sample, requests)["passed"])
        sample["instances"]["small_b"]["candidates"] = [{**native, "generated_tokens": 4096, "remaining_tokens": 0}]
        self.assertFalse(_background_state(sample, requests)["passed"])

    def test_native_preemption_makes_unlabeled_admission_attribution_ambiguous(self):
        rows = [{"iteration": 1, "scheduled_at_s": 2, "new_requests": 1, "preempted_request_ids": []}]
        result = {"first_token_at_s": 3, "ttft_s": 2}
        self.assertTrue(_admission_observation(rows, 1, result)["identified"])
        rows[0]["preempted_request_ids"] = ["resident"]
        observed = _admission_observation(rows, 1, result)
        self.assertFalse(observed["identified"])
        self.assertIsNone(observed["first_native_scheduled_iteration"])
        self.assertIsNone(observed["admission_to_first_schedule_return_s"])

    def test_open_native_tail_invalidates_capture_despite_equal_row_counts(self):
        audit = {"rows": 1, "retired_rows": 1, "open_rows": 0, "unmatched_retirements": 0, "inflight_batches": 0}
        captures = {name: {"audit": dict(audit), "rows": [{"iteration": 1, "completed_at_s": 1}]} for name in INSTANCES}
        self.assertTrue(_capture_audit(captures))
        captures["small_b"]["audit"]["inflight_batches"] = 1
        self.assertFalse(_capture_audit(captures))

    def test_changed_real_history_fails_pair_even_when_both_trial_flags_pass(self):
        snapshot = {"prompt_token_ids": [1], "generated_token_ids": [2], "generated_text": "a", "sampling": {},
                    "num_computed_tokens": 1, "remaining_tokens": 32}
        trial = {"trial_id": "a", "passed": True, "background_input_sha256": "same",
                 "transfer": {"snapshot": snapshot, "result": {"new_output_token_ids": [3] * 32}}}
        other = copy.deepcopy(trial)
        self.assertTrue(_pair([trial, other])["passed"])
        other["transfer"]["snapshot"]["generated_token_ids"] = [7]
        self.assertFalse(_pair([trial, other])["passed"])

    def test_incomplete_native_snapshots_remain_failed_pairs_instead_of_breaking_report(self):
        trial = {"trial_id": "failed", "passed": False,
                 "transfer": {"snapshot": {"prompt_token_ids": [1]}}}
        result = _pair([trial, copy.deepcopy(trial)])
        self.assertFalse(result["passed"])
        self.assertFalse(result["matched_real_prompt_history_and_budget"])

    def test_token_alias_of_evaluation_prompt_is_excluded_even_with_new_source_id(self):
        protocol = {"checkpoint_cases": [{"prompt_tokens": 3, "generated_history_tokens": 2}],
                    "background": {"prompt_tokens": 2, "requests_per_instance": 1, "max_tokens": 9,
                                   "temperature": 0, "ignore_eos": True}}
        manifest = {"manifest_sha256": "fixture", "requests": [{"source_id": "evaluation", "prompt_token_ids": [1, 2]}]}
        catalogue = {"path": "fixture", "sha256": "fixture", "rows": [
            {"id": "alias", "index": 0, "num_tokens": 3, "token_ids": [1, 2, 3]},
            *[{"id": f"held-{n}", "index": n, "num_tokens": 3, "token_ids": [n, n, n]} for n in range(3, 7)]]}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manifest.json"
            path.write_text("{}")
            with patch("hetero.mechanism_cost.load_manifest", return_value=manifest), \
                 patch("hetero.mechanism_cost.manifest_path", return_value=path), \
                 patch("hetero.mechanism_cost._tokenise_catalogue", return_value=catalogue):
                selected = _select_prompts(protocol, {"model": "/mounted"}, "large", None)
            rows = selected["cases"] + [row for group in selected["background"].values() for row in group]
            self.assertNotIn("alias", [row["source_id"] for row in rows])

    def test_one_background_error_does_not_drop_other_fixed_budgets(self):
        finished, cancelled = [], []
        good = {"request_id": "good", "instance_id": "small_b", "prompt_tokens": 2, "max_tokens": 3}
        bad = {**good, "request_id": "bad"}
        result = {**good, "output_tokens": 3, "output_token_ids": [1, 2, 3], "finish_reason": "length"}
        def get(ref):
            if ref == "bad-ref":
                raise RuntimeError("real request error boundary")
            return result
        def cancel(rid, instance):
            cancelled.append(rid)
            return {"request_id": rid, "cancelled": True}
        session = SimpleNamespace(timeout_s=1, ray=SimpleNamespace(wait=lambda refs, **kw: ([refs[0]], refs[1:])),
            get=get, state=SimpleNamespace(finish=lambda rid, count: finished.append(rid)), cancel_owned=cancel,
            unresolved_request=lambda rid, exc: self.fail("cleanup should be acknowledged"))
        background = _Background(session, {"small_b": [bad, good]}, "contract")
        background.requests = [bad, good]
        background.pending = {"bad-ref": bad, "good-ref": good}
        drained = background.drain()
        self.assertFalse(drained["passed"])
        self.assertEqual(finished, ["good"])
        self.assertEqual(cancelled, ["bad"])
        self.assertEqual(drained["results"]["good"]["output_tokens"], 3)

    def test_drain_timeout_keeps_failed_cleanup_identity_instead_of_claiming_idle(self):
        unresolved = []
        row = {"request_id": "pending", "instance_id": "small_b"}
        def fail_cancel(*args):
            raise RuntimeError("lost native cancellation ACK")
        session = SimpleNamespace(timeout_s=1, ray=SimpleNamespace(wait=lambda refs, **kw: ([], refs)),
            cancel_owned=fail_cancel, unresolved_request=lambda rid, exc: unresolved.append(rid))
        background = _Background(session, {"small_b": [row]}, "contract")
        background.requests, background.pending = [row], {"pending-ref": row}
        drained = background.drain()
        self.assertFalse(drained["passed"])
        self.assertEqual(unresolved, ["pending"])
        self.assertFalse(drained["cancellation_receipts"])


if __name__ == "__main__":
    unittest.main()
