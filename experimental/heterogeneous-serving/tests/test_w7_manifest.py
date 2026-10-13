"""W7 mid-length band: routing on S and H, W6 repeat composites, and frozen composite shas."""

import hashlib
import json
import pathlib
import tempfile
import unittest

from hetero import manifest as mf
from hetero.gateway.policy import Candidate, CentralScheduler, RoutingPolicy
from hetero.manifest import COMPOSITES, W7_SPEC

ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_W7 = ROOT / "data/manifests/W7.json"
CAMPAIGN_POLICY = {"decode_length_blind": True, "large_fresh_traffic_share": 0.0,
                   "reserve_large_headroom_blocks": 0}


def w7_contexts():
    """(prompt, decode) pairs: the real W7 manifest when frozen locally, else band-edge rows."""
    if REAL_W7.exists():
        return [(r["prompt_tokens"], r["max_tokens"]) for r in json.loads(REAL_W7.read_text())["requests"]]
    lo, hi, cap = (W7_SPEC[k] for k in ("min_prompt_tokens", "max_prompt_tokens", "max_context_tokens"))
    return [(lo, 64), (lo, 2048), (30000, 1000), (hi, 2048), (hi, cap - hi)]


def part_rows(name, count):
    return [{"request_id": f"{name}-{i:05d}", "arrival_index": i, "prompt_token_ids": [i, i + 1, i + 2],
             "prompt_tokens": 3, "max_tokens": 10 + i, "source_id": f"govreport:{i:016x}"}
            for i in range(count)]


def write_part(root, name, requests):
    body = mf._manifest_body(requests)
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    mf.manifest_path(name, root).write_text(json.dumps({
        "workload": name, "model": "m", "catalogue": "c", "catalogue_sha256": "x", "requests": requests,
        "manifest_sha256": hashlib.sha256(body.encode()).hexdigest()}))


class Spec(unittest.TestCase):
    def test_spec_and_window(self):
        self.assertEqual(W7_SPEC, {"requests": 120, "catalogue": "midctx", "seed": 20261007,
                                   "min_prompt_tokens": 16384, "max_prompt_tokens": 49152,
                                   "max_context_tokens": 51200})
        self.assertEqual(mf.CATALOGUE_WINDOWS["W7"], (0, 240))

    def test_contexts_in_band(self):
        for prompt, decode in w7_contexts():
            self.assertTrue(16384 <= prompt <= 49152)
            self.assertLessEqual(prompt + decode, 51200)


class Routing(unittest.TestCase):
    def test_every_context_eligible_on_s_tp8(self):
        s = [Candidate(f"small_{c}", 8, 51200, 256, 206, 206, 206, 0, 0, 0) for c in "abcd"]
        policy = RoutingPolicy.from_dict(dict(CAMPAIGN_POLICY, small_instances=tuple(c.instance_id for c in s),
                                              large_instances=()))
        for prompt, decode in w7_contexts():
            chosen, trace = CentralScheduler(policy).select_initial(s, prompt_tokens=prompt, max_tokens=decode)
            self.assertIsNotNone(chosen, (prompt, decode))
            self.assertEqual(trace["eligible"], [c.instance_id for c in s])

    def test_every_context_eligible_on_h_with_campaign_policy(self):
        h = [Candidate(n, 16, 131072, 512, 283, 283, 283, 0, 0, 0) for n in ("large_a", "large_b")]
        policy = RoutingPolicy.from_dict(dict(CAMPAIGN_POLICY, small_instances=(),
                                              large_instances=("large_a", "large_b")))
        for prompt, decode in w7_contexts():
            chosen, trace = CentralScheduler(policy).select_initial(h, prompt_tokens=prompt, max_tokens=decode)
            self.assertIn(chosen, ("large_a", "large_b"), (prompt, decode))
            self.assertEqual(trace["eligible"], ["large_a", "large_b"])

    def test_band_campaign_policy_matches_test_policy(self):
        import re
        text = (ROOT / "scripts/band_campaign.sh").read_text()
        self.assertEqual(json.loads(re.search(r"\[plain\]='([^']*)'", text).group(1)), CAMPAIGN_POLICY)


class RepeatComposites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        write_part(self.root, "W6", part_rows("W6", 40))

    def test_w6x2_has_each_source_twice(self):
        self.assertEqual(COMPOSITES["W6x2"], (("W6", 2),))
        self.assertEqual(COMPOSITES["W6x4"], (("W6", 4),))
        mf.write_composite("W6x2", self.root)
        rows = mf.load_manifest("W6x2", self.root)["requests"]
        self.assertEqual(len(rows), 80)
        self.assertEqual(len({r["request_id"] for r in rows}), 80)
        self.assertEqual({r["repeat"] for r in rows}, {0, 1})
        counts = {}
        for r in rows:
            counts.setdefault(r["source_id"], []).append(r["repeat"])
        self.assertEqual(len(counts), 40)
        for repeats in counts.values():
            self.assertEqual(sorted(repeats), [0, 1])

    def test_w6x4_repeat_labels(self):
        mf.write_composite("W6x4", self.root)
        rows = mf.load_manifest("W6x4", self.root)["requests"]
        self.assertEqual(len(rows), 160)
        self.assertEqual({r["repeat"] for r in rows}, {0, 1, 2, 3})

    def test_repeat_field_does_not_change_the_sha(self):
        base = mf.load_manifest("W6", self.root)
        expanded = mf._expand_composite("W6x2", COMPOSITES["W6x2"], self.root)
        body = json.dumps([[f"W6x2-p0r{k}-{r['request_id']}", r["prompt_token_ids"], r["max_tokens"]]
                           for k in range(2) for r in base["requests"]], separators=(",", ":"))
        self.assertEqual(expanded["manifest_sha256"], hashlib.sha256(body.encode()).hexdigest())

    @unittest.skipUnless((ROOT / "data/manifests/W0x4.json").exists(), "frozen composites absent")
    def test_existing_composites_round_trip_with_frozen_sha(self):
        for name in ("W0x4", "MIX_W4_W0x4", "MIX_W4_W5", "MIX_W6_W0"):
            path = mf.manifest_path(name)
            if path.exists():
                frozen = json.loads(path.read_text())["manifest_sha256"]
                self.assertEqual(mf.load_manifest(name)["manifest_sha256"], frozen, name)


if __name__ == "__main__":
    unittest.main()
