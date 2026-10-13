"""W6 long-context workload: catalogue bands, manifest freeze, and routing on M and H."""

import json
import pathlib
import unittest

from hetero.gateway.policy import Candidate, CentralScheduler, RoutingPolicy
from hetero.manifest import COMPOSITES, W6_SPEC

ROOT = pathlib.Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data/workloads/longctx_govreport.json"
BLOCK = 256


def cand(name, tp, max_len, total):
    return Candidate(name, tp, max_len, BLOCK, total, total, total, 0, 0, 0)


def scheduler(**overrides):
    base = dict(small_instances=("small_a", "small_b"), large_instances=("large",),
                large_fresh_traffic_share=0.0, decode_length_blind=True)
    base.update(overrides)
    return CentralScheduler(RoutingPolicy.from_dict(base))


class Catalogue(unittest.TestCase):
    @unittest.skipUnless(CATALOGUE.exists(), "catalogue not built")
    def test_bands(self):
        records = json.loads(CATALOGUE.read_text())
        ids = [r["id"] for r in records]
        self.assertEqual(ids, sorted(set(ids)))
        for r in records:
            self.assertGreaterEqual(r["prompt_tokens"], W6_SPEC["min_prompt_tokens"])
            self.assertLessEqual(r["prompt_tokens"] + r["reference_decode_tokens"],
                                 W6_SPEC["max_context_tokens"])
            self.assertTrue(64 <= r["reference_decode_tokens"] <= 2048)
            self.assertEqual(r["source"], "govreport")

    def test_composite_registered(self):
        self.assertEqual(COMPOSITES["MIX_W6_W0"], (("W6", 1), ("W0", 1)))


class Routing(unittest.TestCase):
    def m_candidates(self):
        return [cand("small_a", 8, 16384, 206), cand("small_b", 8, 16384, 206),
                cand("large", 16, 131072, 566)]

    def test_w0_sized_prompt_goes_to_tp8(self):
        chosen, _ = scheduler().select_initial(self.m_candidates(), prompt_tokens=500, max_tokens=300)
        self.assertIn(chosen, ("small_a", "small_b"))

    def test_60k_prompt_goes_to_tp16(self):
        chosen, trace = scheduler().select_initial(self.m_candidates(), prompt_tokens=60000, max_tokens=700)
        self.assertEqual(chosen, "large")
        self.assertEqual(trace["eligible"], ["large"])

    def test_h_accepts_catalogue_max_context(self):
        # H = large_a + large_b, max_model_len 131072, ~566 blocks of 256 (144,896 tokens).
        # Decode-blind placement charges prompt + one block; the default 64-block large
        # headroom still leaves room for the largest catalogue prompt (127,370 tokens).
        h = [cand("large_a", 16, 131072, 566), cand("large_b", 16, 131072, 566)]
        policy = RoutingPolicy.from_dict(dict(small_instances=(), large_instances=("large_a", "large_b"),
                                              large_fresh_traffic_share=0.0, decode_length_blind=True))
        chosen, trace = CentralScheduler(policy).select_initial(h, prompt_tokens=127370, max_tokens=1534)
        self.assertIn(chosen, ("large_a", "large_b"))
        self.assertEqual(trace["eligible"], ["large_a", "large_b"])
        # A context of exactly 131072 passes max_model_len eligibility on H.
        _, full = CentralScheduler(policy).select_initial(h, prompt_tokens=129572, max_tokens=1500)
        self.assertEqual(full["eligible"], ["large_a", "large_b"])

    def test_campaign_policy_admits_longest_prompt_at_any_block_size(self):
        # The w6_campaign.sh policy drops the 64-block large headroom; with 512-token blocks
        # (283 blocks = 144,896 tokens) the default headroom would refuse a 127,370-token prompt.
        import re
        text = (ROOT / "scripts/w6_campaign.sh").read_text()
        campaign = json.loads(re.search(r"\[plain\]='([^']*)'", text).group(1))
        for block, total in ((256, 566), (512, 283), (1024, 141)):
            h = [Candidate(n, 16, 131072, block, total, total, total, 0, 0, 0)
                 for n in ("large_a", "large_b")]
            policy = RoutingPolicy.from_dict(dict(campaign, small_instances=(),
                                                  large_instances=("large_a", "large_b")))
            chosen, _ = CentralScheduler(policy).select_initial(h, prompt_tokens=127370, max_tokens=1534)
            self.assertIn(chosen, ("large_a", "large_b"), block)
            default = RoutingPolicy.from_dict(dict(campaign, reserve_large_headroom_blocks=64,
                                                   small_instances=(), large_instances=("large_a", "large_b")))
            if block >= 512:
                self.assertIsNone(CentralScheduler(default).select_initial(
                    h, prompt_tokens=127370, max_tokens=1534)[0])


if __name__ == "__main__":
    unittest.main()
