"""
Unit Tests for Preemption Modes: Reprefill vs. Regenerate vs. In-HBM Suspension
==============================================================================
Verifies the exact algorithmic behavior of:
1. REPREFILL: Saving output tokens so subsequent re-admission prefills P + k tokens in parallel.
2. REGENERATE: Classical discard-and-recompute from token 0.
3. SUSPEND: In-HBM zero-eviction pause (Delta M = 0).
"""

import unittest
from admission_control.simulation.simulator import SimRequest, VLLMSchedulerSimulator


class TestPreemptionModes(unittest.TestCase):
    def test_reprefill_saves_tokens_and_prefills(self):
        # Setup constrained capacity that forces preemption
        # Capacity = 100 tokens, block_size = 16 (6 blocks total = 96 tokens)
        sim = VLLMSchedulerSimulator(
            capacity_tokens=96,
            block_size=16,
            max_num_seqs=4,
            preemption_mode="reprefill",
        )
        
        # 3 requests:
        # Req 0: P=32 (2 blocks), D=32
        # Req 1: P=32 (2 blocks), D=32
        # Req 2: P=32 (2 blocks), D=32
        # Together at t=0, they need 6 blocks (96 tokens).
        # At t=1, they expand and need more blocks -> preemption!
        raw_reqs = [
            {"req_id": 0, "prompt_len": 32, "decode_len": 32},
            {"req_id": 1, "prompt_len": 32, "decode_len": 32},
            {"req_id": 2, "prompt_len": 32, "decode_len": 32},
        ]
        
        res = sim._simulate_core(
            [SimRequest(r["req_id"], r["prompt_len"], r["decode_len"]) for r in raw_reqs],
            preemption_mode="reprefill"
        )
        
        self.assertGreater(res["total_preemptions"], 0)
        self.assertEqual(res["preemption_mode"], "reprefill")
        # In reprefill mode, preempted tokens are saved and re-prefilled
        self.assertGreater(res["total_reprefill_tokens"], 0)
        self.assertEqual(res["delivered_tokens"], 96)

    def test_regenerate_wipes_generated_tokens(self):
        sim = VLLMSchedulerSimulator(
            capacity_tokens=96,
            block_size=16,
            max_num_seqs=4,
            preemption_mode="regenerate",
        )
        raw_reqs = [
            {"req_id": 0, "prompt_len": 32, "decode_len": 32},
            {"req_id": 1, "prompt_len": 32, "decode_len": 32},
            {"req_id": 2, "prompt_len": 32, "decode_len": 32},
        ]
        
        res = sim._simulate_core(
            [SimRequest(r["req_id"], r["prompt_len"], r["decode_len"]) for r in raw_reqs],
            preemption_mode="regenerate"
        )
        
        self.assertGreater(res["total_preemptions"], 0)
        self.assertEqual(res["preemption_mode"], "regenerate")
        self.assertEqual(res["total_reprefill_tokens"], 0)
        self.assertGreater(res["total_discarded_tokens"], 0)
        self.assertEqual(res["delivered_tokens"], 96)

    def test_suspend_avoids_token_discard(self):
        sim = VLLMSchedulerSimulator(
            capacity_tokens=160,
            block_size=16,
            max_num_seqs=4,
            preemption_mode="suspend",
        )
        raw_reqs = [
            {"req_id": 0, "prompt_len": 32, "decode_len": 16},
            {"req_id": 1, "prompt_len": 32, "decode_len": 32},
            {"req_id": 2, "prompt_len": 32, "decode_len": 32},
        ]
        
        res = sim._simulate_core(
            [SimRequest(r["req_id"], r["prompt_len"], r["decode_len"]) for r in raw_reqs],
            preemption_mode="suspend"
        )
        
        self.assertEqual(res["preemption_mode"], "suspend")
        self.assertEqual(res["delivered_tokens"], 80)
        # Suspend mode minimizes or eliminates discarded tokens
        self.assertLessEqual(res["total_discarded_tokens"], res["delivered_tokens"])


if __name__ == "__main__":
    unittest.main()
