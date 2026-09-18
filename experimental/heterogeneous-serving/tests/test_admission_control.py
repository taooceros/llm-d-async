"""
Comprehensive Unit Test Suite for `admission_control`.
======================================================
Tests:
- HardwareSpec bucket matching, latencies, and padding slots
- Request models and TokenProfile aggregation
- Priority sorting rules (binned LPT, pure LPT, density, SPT)
- Windowed batch slicing and intra-batch reordering bounds
- Schedule simulation and memory trajectory calculations
- Analytical asymmetric cohort staggering optimizer
- Dynamic 2D profile water-filling solver
- vLLM Discrete-Event Scheduler Simulator & Paged Block Allocator
- Workload trace ingestion and synthetic generation
- TPU topology coordinate parsing and submesh selection
"""

import unittest
import os
import tempfile
import json

from admission_control.core.models import (
    Request,
    HardwareSpec,
    SchedulePlan,
    TokenProfile,
    BatchCohort,
)
from admission_control.algorithms.policies import (
    binned_lpt_priority_key,
    slack_packed_lpt_priority_key,
    pure_lpt_priority_key,
    density_priority_key,
    spt_priority_key,
    order_requests_by_policy,
    slice_into_windowed_batches,
    calculate_peak_future_memory_trajectory,
    dynamic_knapsack_admit_guard,
    schedule_dynamic_knapsack_batches,
)
from admission_control.algorithms.solver import (
    simulate_schedule_plan,
    solve_asymmetric_staggering,
    solve_profile_water_filling,
    solve_schedule,
    run_comparative_suite,
)
from admission_control.simulation.block_manager import BlockManager
from admission_control.simulation.simulator import (
    SimRequest,
    VLLMSchedulerSimulator,
)
from admission_control.workloads.loader import (
    load_workload_requests,
    save_workload_requests,
    get_workload_profile,
)
from admission_control.workloads.generator import (
    generate_synthetic_workload,
)
from admission_control.tpu.topology import (
    TPUTopologyDiscovery,
    TPUHostInfo,
    ConnectedPair,
)


class TestAdmissionControlCore(unittest.TestCase):
    """Tests core data models, hardware specifications, and static XLA bucket matching."""

    def setUp(self):
        self.hw = HardwareSpec(capacity_tokens=120_000, max_batch_size=128)

    def test_xla_bucket_selection(self):
        self.assertEqual(self.hw.get_bucket(0), 0)
        self.assertEqual(self.hw.get_bucket(1), 16)
        self.assertEqual(self.hw.get_bucket(16), 16)
        self.assertEqual(self.hw.get_bucket(17), 32)
        self.assertEqual(self.hw.get_bucket(32), 32)
        self.assertEqual(self.hw.get_bucket(33), 64)
        self.assertEqual(self.hw.get_bucket(64), 64)
        self.assertEqual(self.hw.get_bucket(65), 128)
        self.assertEqual(self.hw.get_bucket(128), 128)
        self.assertEqual(self.hw.get_bucket(200), 256)

    def test_step_latency_and_padding(self):
        # Latency must be non-decreasing with bucket size
        lat_16 = self.hw.step_latency(16)
        lat_32 = self.hw.step_latency(32)
        lat_64 = self.hw.step_latency(64)
        self.assertLess(lat_16, lat_32)
        self.assertLess(lat_32, lat_64)

        # Padding slots
        self.assertEqual(self.hw.padding_slots(16), 0)
        self.assertEqual(self.hw.padding_slots(15), 1)
        self.assertEqual(self.hw.padding_slots(48), 16)  # Bucket 64 - 48 = 16

    def test_request_and_profile_aggregation(self):
        reqs = [
            Request(req_id=0, prompt_len=100, decode_len=500),
            Request(req_id=1, prompt_len=200, decode_len=400),
            Request(req_id=2, prompt_len=300, decode_len=300),
        ]
        profile = TokenProfile.from_requests(reqs)
        self.assertEqual(profile.num_requests, 3)
        self.assertEqual(profile.total_prompt_tokens, 600)
        self.assertEqual(profile.total_decode_tokens, 1200)
        self.assertEqual(profile.total_demand_tokens, 1800)
        self.assertEqual(profile.min_prompt, 100)
        self.assertEqual(profile.max_prompt, 300)
        self.assertEqual(profile.avg_prompt, 200.0)


class TestAdmissionControlAlgorithms(unittest.TestCase):
    """Tests priority sorting rules, solvers, and schedule simulation."""

    def test_binned_lpt_priority_tiebreaker(self):
        # Two requests with decode length in same 16-token bin:
        # req_a: D=500, P=200 -> bin = -(500//16) = -31
        # req_b: D=498, P=100 -> bin = -(498//16) = -31
        # req_b has shorter prompt (P=100 < 200), so req_b comes first!
        req_a = Request(req_id=0, prompt_len=200, decode_len=500)
        req_b = Request(req_id=1, prompt_len=100, decode_len=498)

        key_a = binned_lpt_priority_key(req_a, bin_size=16)
        key_b = binned_lpt_priority_key(req_b, bin_size=16)
        self.assertLess(key_b, key_a)

    def test_windowed_batch_slicing_bounds(self):
        reqs = [Request(req_id=i, prompt_len=100, decode_len=i * 10) for i in range(128)]
        batches = slice_into_windowed_batches(reqs, batch_size=64, intra_batch_policy="binned_lpt")
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 64)
        self.assertEqual(len(batches[1]), 64)

        # Invariant: batch 0 contains only req_ids in [0, 63]
        batch0_ids = {r.req_id for r in batches[0]}
        self.assertEqual(batch0_ids, set(range(64)))

    def test_schedule_simulation_metrics(self):
        hw = HardwareSpec(capacity_tokens=120_000)
        reqs = [Request(req_id=i, prompt_len=256, decode_len=512) for i in range(32)]
        start_times = {r.req_id: 0 for r in reqs}

        plan = simulate_schedule_plan(reqs, start_times, hw, policy_name="Test Full Concurrency")
        self.assertEqual(plan.makespan_steps, 512)
        self.assertGreater(plan.goodput_tok_s, 0.0)
        self.assertEqual(plan.preemptions, 0)
        self.assertFalse(plan.metadata.get("feasible") is False)

    def test_asymmetric_staggering_solver(self):
        # Stress workload where 64 * (256 + 2048) = 147,456 > 120,000 capacity
        hw = HardwareSpec(capacity_tokens=120_000)
        reqs = [Request(req_id=i, prompt_len=256, decode_len=2048) for i in range(64)]

        plan = solve_asymmetric_staggering(reqs, hw)
        self.assertTrue(plan.metadata.get("feasible"))
        self.assertLessEqual(plan.peak_mem_tokens, 120_000)
        self.assertEqual(plan.preemptions, 0)

    def test_water_filling_solver(self):
        hw = HardwareSpec(capacity_tokens=120_000, max_batch_size=64)
        reqs = [Request(req_id=i, prompt_len=300 + i * 5, decode_len=400 + i * 2) for i in range(64)]

        plan = solve_profile_water_filling(reqs, hw, priority="lpt")
        self.assertGreater(plan.makespan_steps, 0)
        self.assertGreater(plan.goodput_tok_s, 0.0)
        self.assertLessEqual(plan.peak_mem_tokens, 120_000)


    def test_calculate_peak_future_memory_trajectory_exact(self):
        # Verify exact piecewise linear calculation:
        # req1: P=200, D=800, C=0 (rem=800, total=1000)
        # req2: P=300, D=500, C=0 (rem=500, total=800)
        # cand: P=250, D=600, C=0 (rem=600, total=850)
        # At delta=0: mem = 200 + 300 + 250 = 750
        # At delta=500 (req2 finishes): mem before release = 750 + 3 * 500 = 2250 (PEAK!)
        # After release of req2: mem = 2250 - 800 = 1450
        # At delta=600 (cand finishes): mem before release = 1450 + 2 * 100 = 1650
        # After release of cand: mem = 1650 - 850 = 800
        # At delta=800 (req1 finishes): mem before release = 800 + 1 * 200 = 1000
        # After release of req1: mem = 0
        req1 = Request(req_id=1, prompt_len=200, decode_len=800)
        req2 = Request(req_id=2, prompt_len=300, decode_len=500)
        cand = Request(req_id=3, prompt_len=250, decode_len=600)

        peak = calculate_peak_future_memory_trajectory([req1, req2], cand)
        self.assertEqual(peak, 2250)

    def test_dynamic_knapsack_admission_guard_concurrency_and_headroom(self):
        req1 = Request(req_id=1, prompt_len=200, decode_len=800)
        req2 = Request(req_id=2, prompt_len=300, decode_len=500)
        cand = Request(req_id=3, prompt_len=250, decode_len=600)

        # Cap=3000 >= 2250 -> True
        self.assertTrue(dynamic_knapsack_admit_guard([req1, req2], cand, capacity=3000, target_concurrency=96))
        # Cap=2000 < 2250 -> False (throttled)
        self.assertFalse(dynamic_knapsack_admit_guard([req1, req2], cand, capacity=2000, target_concurrency=96))
        # Concurrency limit reached -> False
        self.assertFalse(dynamic_knapsack_admit_guard([req1, req2], cand, capacity=3000, target_concurrency=2))

    def test_dynamic_knapsack_throttles_on_tight_headroom(self):
        # Heavy cohort B0 of 64 requests with P=800, D=1000 (total demand 1,800 tokens each)
        # Total B0 demand = 64 * 1800 = 115,200 tokens
        # Headroom in 120,000 token pool = 120,000 - 115,200 = 4,800 tokens
        b0 = [Request(req_id=i, prompt_len=800, decode_len=1000) for i in range(64)]
        b1 = [Request(req_id=64 + i, prompt_len=800, decode_len=1000) for i in range(64)]

        # Candidate from B1 has demand 1,800 tokens
        # At step 0: candidate 1 admitted (115,200 + 1,800 = 117,000 <= 120,000)
        # candidate 2 admitted (117,000 + 1,800 = 118,800 <= 120,000)
        # candidate 3 throttled! (118,800 + 1,800 = 120,600 > 120,000)
        active = list(b0)
        admitted = []
        throttled = 0
        for cand in b1:
            if dynamic_knapsack_admit_guard(active + admitted, cand, capacity=120_000, target_concurrency=96):
                admitted.append(cand)
            else:
                throttled += 1

        self.assertEqual(len(admitted), 2)
        self.assertEqual(throttled, 62)
        peak_admitted = calculate_peak_future_memory_trajectory(active + admitted)
        self.assertEqual(peak_admitted, 118_800)
        self.assertLessEqual(peak_admitted, 120_000)

    def test_schedule_dynamic_knapsack_batches_multi_cohort(self):
        # 2 cohorts of 64 requests (128 total)
        b0 = [Request(req_id=i, prompt_len=400, decode_len=600 + (i % 50)) for i in range(64)]
        b1 = [Request(req_id=64 + i, prompt_len=400, decode_len=600 + (i % 50)) for i in range(64)]

        for cap in [96, 104, 112]:
            res = schedule_dynamic_knapsack_batches([b0, b1], target_concurrency=cap, capacity=120_000)
            self.assertEqual(res["finished_requests"], 128)
            self.assertEqual(res["preemptions"], 0)
            self.assertLessEqual(res["peak_observed_memory"], 120_000)
            self.assertGreater(res["goodput_tok_per_s"], 0.0)
            self.assertEqual(res["target_concurrency"], cap)

class TestVLLMSimulator(unittest.TestCase):
    """Tests the discrete-event vLLM scheduler simulator and block manager."""

    def test_block_manager_allocations(self):
        bm = BlockManager(capacity_tokens=160, block_size=16)  # 10 blocks
        self.assertEqual(bm.total_blocks, 10)
        self.assertEqual(bm.free_blocks, 10)

        bm.allocate(req_id=0, num_blocks=4)
        self.assertEqual(bm.free_blocks, 6)
        self.assertEqual(bm.used_tokens, 64)

        freed = bm.free(req_id=0)
        self.assertEqual(freed, 4)
        self.assertEqual(bm.free_blocks, 10)

    def test_simulator_preemption_and_recovery(self):
        # Tiny capacity: 1000 tokens pool (62 blocks of 16 tokens)
        sim = VLLMSchedulerSimulator(capacity_tokens=1000, block_size=16, max_num_seqs=16)
        raw_reqs = [{"req_id": i, "prompt_len": 50, "decode_len": 100} for i in range(16)]

        # Policy A will exhaust 1000 tokens during decode, triggering preemptions
        res_a = sim.simulate_policy_a_greedy(raw_reqs)
        self.assertGreater(res_a["total_preemptions"], 0)
        self.assertGreater(res_a["total_discarded_tokens"], 0)

        # Policy B with tight concurrency cap avoids preemption
        res_b = sim.simulate_policy_b_predictive(raw_reqs, cap=4)
        self.assertEqual(res_b["total_preemptions"], 0)
        self.assertEqual(res_b["total_discarded_tokens"], 0)


class TestWorkloadsAndTopology(unittest.TestCase):
    """Tests trace loading, synthetic workload creation, and TPU topology helpers."""

    def test_synthetic_workload_generation(self):
        reqs = generate_synthetic_workload(num_requests=32, min_prompt=100, max_prompt=500, min_decode=200, max_decode=600)
        self.assertEqual(len(reqs), 32)
        for r in reqs:
            self.assertTrue(100 <= r.prompt_len <= 500)
            self.assertTrue(200 <= r.decode_len <= 600)

    def test_workload_serialization_roundtrip(self):
        reqs = [Request(req_id=i, prompt_len=100 + i, decode_len=200 + i) for i in range(10)]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_path = tf.name

        try:
            save_workload_requests(reqs, tmp_path)
            loaded = load_workload_requests(tmp_path)
            self.assertEqual(len(loaded), 10)
            for orig, rec in zip(reqs, loaded):
                self.assertEqual(orig.req_id, rec.req_id)
                self.assertEqual(orig.prompt_len, rec.prompt_len)
                self.assertEqual(orig.decode_len, rec.decode_len)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def test_workload_heterogeneous_key_loading(self):
        sample_data = [
            {"id": 10, "input_tokens": 512, "output_tokens": 128, "timestamp": 100},
            {"req_id": 20, "prompt_tokens": 1024, "output_len": 256},
            {"req_id": 30, "prompt_len": 256, "decode_len": 512},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
            json.dump(sample_data, tf)
            tmp_path = tf.name

        try:
            reqs = load_workload_requests(tmp_path)
            self.assertEqual(len(reqs), 3)
            self.assertEqual(reqs[0].req_id, 10)
            self.assertEqual(reqs[0].prompt_len, 512)
            self.assertEqual(reqs[0].decode_len, 128)
            self.assertEqual(reqs[0].arrival_time, 100)
            self.assertEqual(reqs[1].prompt_len, 1024)
            self.assertEqual(reqs[1].decode_len, 256)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    def test_tpu_host_coordinate_derivation(self):
        # 2x4 mesh: 2 hosts along X, 1 host along Y (or 1x2)
        coords, x_range, y_range = TPUTopologyDiscovery.compute_host_and_chip_coords(
            worker_id=0,
            host_bounds=(1, 2),
            chips_per_host=(2, 2),
        )
        self.assertEqual(coords, (0, 0))
        self.assertEqual(x_range, (0, 1))
        self.assertEqual(y_range, (0, 1))

class TestXLAProfiler(unittest.TestCase):
    """Tests XLA profiling analysis and report generation."""

    def test_xla_profiler_nonexistent_directory(self):
        from admission_control.tpu.xla_profiler import XLAProfiler
        profiler = XLAProfiler(jax_cache_dir="/tmp/nonexistent_cache_dir_12345")
        profile = profiler.analyze_cache()
        self.assertEqual(profile.total_modules, 0)
        report = profiler.generate_report()
        self.assertIn("Cloud TPU v5e XLA Compiler", report)

class TestDecoupledPolicyExecution(unittest.TestCase):
    """Tests the decoupled policy interface, execution engines, and unified PolicyExecutor."""

    def setUp(self):
        from admission_control.algorithms.policies import (
            BasePolicy,
            ComposablePolicy,
            DynamicKnapsackPolicy,
            get_policy,
        )
        from admission_control.algorithms.executor import (
            SimulatedExecutionEngine,
            PolicyExecutor,
        )
        self.BasePolicy = BasePolicy
        self.ComposablePolicy = ComposablePolicy
        self.DynamicKnapsackPolicy = DynamicKnapsackPolicy
        self.get_policy = get_policy
        self.SimulatedExecutionEngine = SimulatedExecutionEngine
        self.PolicyExecutor = PolicyExecutor

    def test_policy_factory_resolution(self):
        p_greedy = self.get_policy("greedy", max_concurrency=64)
        self.assertIsInstance(p_greedy, self.BasePolicy)
        self.assertIn("Greedy", p_greedy.name)

        p_predictive = self.get_policy("predictive", cap=48)
        self.assertIsInstance(p_predictive, self.BasePolicy)
        self.assertIn("Predictive", p_predictive.name)

        p_wave = self.get_policy("discrete_waves", wave_size=32)
        self.assertIsInstance(p_wave, self.BasePolicy)
        self.assertIn("Discrete Wave", p_wave.name)

        p_knapsack = self.get_policy("knapsack", target_concurrency=96, capacity=120_000)
        self.assertIsInstance(p_knapsack, self.BasePolicy)
        self.assertIn("Knapsack", p_knapsack.name)

    def test_simulated_execution_engine_step_and_completion(self):
        engine = self.SimulatedExecutionEngine(capacity_tokens=10_000, block_size=16)
        engine.add_request({"req_id": 1, "prompt_len": 32, "decode_len": 5})
        engine.add_request({"req_id": 2, "prompt_len": 16, "decode_len": 3})

        self.assertEqual(len(engine.get_active_requests()), 2)
        self.assertTrue(engine.has_unfinished_requests())

        # Step 3 times: req 2 should finish
        for _ in range(3):
            step_out = engine.step()

        self.assertIn(2, step_out.finished_req_ids)
        self.assertEqual(len(engine.get_active_requests()), 1)
        self.assertEqual(engine.get_active_requests()[0].req_id, 1)

        # Step 2 more times: req 1 should finish
        for _ in range(2):
            step_out = engine.step()

        self.assertIn(1, step_out.finished_req_ids)
        self.assertEqual(len(engine.get_active_requests()), 0)
        self.assertFalse(engine.has_unfinished_requests())

    def test_policy_executor_preserves_batch_boundaries(self):
        # Two cohorts of 4 requests each
        batch_0 = [{"req_id": i, "prompt_len": 32, "decode_len": 10} for i in range(4)]
        batch_1 = [{"req_id": i + 4, "prompt_len": 32, "decode_len": 10} for i in range(4)]
        workload = [batch_0, batch_1]

        # Policy with discrete waves: cannot admit batch 1 until batch 0 drains
        policy = self.get_policy("discrete_waves", wave_size=4)
        engine = self.SimulatedExecutionEngine(capacity_tokens=20_000, block_size=16)
        executor = self.PolicyExecutor(policy=policy, engine=engine)

        result = executor.execute(workload, batch_size=4)
        self.assertEqual(result["finished_requests"], 8)
        self.assertEqual(result["num_preemptions"], 0)
        self.assertEqual(result["admitted_per_batch"][0], 4)
        self.assertEqual(result["admitted_per_batch"][1], 4)
        self.assertGreater(result["goodput_tok_per_s"], 0.0)

    def test_dynamic_knapsack_policy_oversubscription_and_immunity(self):
        # Generate 3 batches of 8 requests each with long decodes
        batches = [
            [{"req_id": b * 8 + i, "prompt_len": 100, "decode_len": 500} for i in range(8)]
            for b in range(3)
        ]
        # Tight capacity: only enough memory for ~12 concurrent requests (12 * 600 = 7,200)
        cap = 8_000
        engine = self.SimulatedExecutionEngine(capacity_tokens=cap, block_size=16)
        policy = self.DynamicKnapsackPolicy(target_concurrency=16, capacity=cap)
        executor = self.PolicyExecutor(policy=policy, engine=engine)

        res = executor.execute(batches, batch_size=8)
        self.assertEqual(res["finished_requests"], 24)
        self.assertEqual(res["num_preemptions"], 0)
        self.assertLessEqual(res["peak_memory_tokens"], cap)
        self.assertGreater(res["throttled_events"], 0)

    def test_simulator_simulate_policy_integration(self):
        sim = VLLMSchedulerSimulator(capacity_tokens=50_000, max_num_seqs=64)
        raw_reqs = [{"req_id": i, "prompt_len": 128, "decode_len": 256} for i in range(16)]
        res = sim.simulate_policy("knapsack", raw_reqs, target_concurrency=16)
        self.assertEqual(res["finished_requests"], 16)
        self.assertEqual(res["num_preemptions"], 0)
        self.assertIn("Knapsack", res["policy_name"])



class TestComposableSchedulingPolicyInterface(unittest.TestCase):
    """Tests the new composable priority rules, admission guards, policy builder, and invariant verifier."""

    def setUp(self):
        from admission_control.algorithms.policies import (
            Priority,
            Guard,
            BatchTransition,
            Victim,
            ComposablePolicy,
            PolicyBuilder,
            verify_policy_invariants,
        )
        from admission_control.algorithms.executor import PolicyExecutor, SimulatedExecutionEngine
        self.Priority = Priority
        self.Guard = Guard
        self.BatchTransition = BatchTransition
        self.Victim = Victim
        self.ComposablePolicy = ComposablePolicy
        self.PolicyBuilder = PolicyBuilder
        self.verify_policy_invariants = verify_policy_invariants
        self.PolicyExecutor = PolicyExecutor
        self.SimulatedExecutionEngine = SimulatedExecutionEngine

    def test_priority_rule_composition_and_chaining(self):
        req_a = Request(req_id=0, prompt_len=200, decode_len=500)
        req_b = Request(req_id=1, prompt_len=100, decode_len=500)
        req_c = Request(req_id=2, prompt_len=50, decode_len=400)

        # Chained LPT -> Shortest prompt
        rule = self.Priority.lpt().then_by(self.Priority.shortest_prompt())
        key_a = rule.key(req_a)
        key_b = rule.key(req_b)
        key_c = rule.key(req_c)

        # req_b has same decode (500) as req_a, but shorter prompt (100 < 200) -> key_b < key_a
        self.assertLess(key_b, key_a)
        # req_a has longer decode (500 > 400) than req_c -> key_a < key_c
        self.assertLess(key_a, key_c)

        # Reversed rule
        rev = rule.reversed()
        self.assertGreater(rev.key(req_b), rev.key(req_a))

        # Weighted sum
        w_rule = 0.5 * self.Priority.spt() + 0.5 * self.Priority.shortest_prompt()
        self.assertIsInstance(w_rule.key(req_a), float)

    def test_admission_guard_boolean_algebra(self):
        g_cap = self.Guard.concurrency_cap(10)
        g_mem = self.Guard.peak_future_memory(capacity=50_000)
        g_and = g_cap & g_mem
        g_or = g_cap | g_mem
        g_not = ~g_cap

        cand = {"prompt_len": 100, "decode_len": 500}
        empty_active = []
        self.assertTrue(g_and.can_admit(empty_active, cand))
        self.assertTrue(g_or.can_admit(empty_active, cand))
        self.assertFalse(g_not.can_admit(empty_active, cand))

        # Full active (10 requests)
        full_active = [{"prompt_len": 100, "decode_len": 500, "tokens_decoded": 0} for _ in range(10)]
        self.assertFalse(g_and.can_admit(full_active, cand))
        self.assertTrue(g_not.can_admit(full_active, cand))

    def test_policy_builder_execution_and_invariants(self):
        policy = (
            self.PolicyBuilder("TestComposable")
            .with_priority(self.Priority.binned_lpt(bin_size=4).then_by(self.Priority.shortest_prompt()))
            .admit_if(self.Guard.concurrency_cap(8))
            .admit_if(self.Guard.preemption_immune(capacity=10_000, block_size=16))
            .draw_next_batch_if(self.BatchTransition.preemption_immune(capacity=10_000, target_concurrency=8))
            .with_victim_selector(self.Victim.least_progress())
            .build()
        )

        batch_0 = [{"req_id": i, "prompt_len": 50, "decode_len": 100} for i in range(8)]
        batch_1 = [{"req_id": i + 8, "prompt_len": 50, "decode_len": 100} for i in range(8)]
        workload = batch_0 + batch_1

        report = self.verify_policy_invariants(policy, workload, batch_size=8, capacity_tokens=10_000)
        self.assertTrue(report.is_valid)
        self.assertTrue(report.preemption_immunity)
        self.assertTrue(report.starvation_freedom)
        self.assertTrue(report.memory_capacity_safety)
        self.assertEqual(report.preemption_count, 0)

    def test_solve_optimal_composable_policy(self):
        from admission_control.algorithms.solver import solve_optimal_composable_policy
        from admission_control.core.models import HardwareSpec

        workload = [
            {"req_id": i, "prompt_len": 64 + (i % 32), "decode_len": 128 + (i % 64)}
            for i in range(16)
        ]
        hw = HardwareSpec(capacity_tokens=10_000, max_batch_size=32)
        best_pol, meta = solve_optimal_composable_policy(workload, hw, batch_size=8)
        self.assertIsNotNone(best_pol)
        self.assertGreater(meta["goodput_tok_per_s"], 0.0)
        self.assertEqual(meta["num_preemptions"], 0)
        self.assertEqual(meta["finished_requests"], 16)

    def test_policy_builder_from_dict_and_get_policy(self):
        from admission_control.algorithms.policies import get_policy

        spec = {
            "name": "DeclarativeTestPolicy",
            "priority": {
                "type": "binned_lpt",
                "bin_size": 4,
                "then_by": {"type": "shortest_prompt"},
            },
            "guards": [
                {"type": "concurrency_cap", "cap": 16},
                {"type": "preemption_immune", "capacity": 20_000, "block_size": 16},
            ],
            "batch_transition": {
                "type": "preemption_immune",
                "capacity": 20_000,
                "target_concurrency": 16,
            },
            "victim_selector": "least_progress",
        }

        pol = get_policy(spec)
        self.assertIsInstance(pol, self.ComposablePolicy)
        self.assertEqual(pol.name, "DeclarativeTestPolicy")
        self.assertIsNotNone(pol.priority_rule)
        self.assertIsNotNone(pol.admission_guard)
        self.assertIsNotNone(pol.batch_transition_guard)

    def test_preemption_victim_selector_integration(self):
        from admission_control.algorithms.policies import Victim
        engine = self.SimulatedExecutionEngine(capacity_tokens=512, block_size=16)
        engine.set_victim_selector(Victim.least_progress())

        engine.add_request({"req_id": 1, "prompt_len": 64, "decode_len": 20})
        for _ in range(10):
            engine.step()

        engine.add_request({"req_id": 2, "prompt_len": 64, "decode_len": 20})
        for _ in range(5):
            engine.step()

        engine.add_request({"req_id": 3, "prompt_len": 64, "decode_len": 20})
        for _ in range(2):
            engine.step()

        # Force capacity breach by artificially shrinking free blocks
        engine.block_manager.free_blocks = -1
        step_out = engine.step()

        # Least progress victim selector should choose req 3 (least progress)
        self.assertGreater(len(step_out.preempted_req_ids), 0)
        self.assertEqual(step_out.preempted_req_ids[0], 3)

    def test_prefill_budget_rate_limit_and_describe(self):
        from admission_control.algorithms.policies import (
            PrefillTokenBudgetGuard,
            RateLimitGuard,
            SystemState,
        )

        # Test PrefillTokenBudgetGuard
        bg = PrefillTokenBudgetGuard(max_tokens_per_step=1000)
        state = SystemState(step=1)
        cand1 = {"prompt_len": 600, "decode_len": 100}
        cand2 = {"prompt_len": 500, "decode_len": 100}

        self.assertTrue(bg.can_admit([], cand1, state))
        self.assertFalse(bg.can_admit([], cand2, state))  # 600 + 500 = 1100 > 1000

        # New step resets budget
        state2 = SystemState(step=2)
        self.assertTrue(bg.can_admit([], cand2, state2))

        # Test RateLimitGuard
        rg = RateLimitGuard(max_admissions_per_step=2)
        self.assertTrue(rg.can_admit([], cand1, state))
        self.assertTrue(rg.can_admit([], cand1, state))
        self.assertFalse(rg.can_admit([], cand1, state))  # 3rd admission rejected

        # Test FIFO and EDF priority
        r_old = Request(req_id=1, prompt_len=100, decode_len=100, arrival_time=10)
        r_new = Request(req_id=2, prompt_len=100, decode_len=100, arrival_time=20)
        p_fifo = self.Priority.fifo()
        self.assertLess(p_fifo.key(r_old), p_fifo.key(r_new))

        # Test describe()
        policy = (
            self.PolicyBuilder("DescribedPolicy")
            .with_priority(self.Priority.lpt())
            .admit_if(self.Guard.concurrency_cap(64))
            .build()
        )
        desc = policy.describe()
        self.assertIn("DescribedPolicy", desc)
        self.assertIn("Priority Rule", desc)
        self.assertIn("Admission Guards", desc)

    def test_policy_lifecycle_hooks(self):
        admitted_log = []
        steps_log = []
        finished_log = []

        policy = (
            self.PolicyBuilder("HookPolicy")
            .with_priority(self.Priority.lpt())
            .admit_if(self.Guard.concurrency_cap(16))
            .on_admit(lambda req, state: admitted_log.append(req["req_id"]))
            .on_step(lambda step_out, state: steps_log.append(step_out.step))
            .on_finish(lambda f_id, state: finished_log.append(f_id))
            .build()
        )

        workload = [
            {"req_id": i, "prompt_len": 32, "decode_len": 5}
            for i in range(4)
        ]
        engine = self.SimulatedExecutionEngine(capacity_tokens=5000, block_size=16)
        executor = self.PolicyExecutor(policy=policy, engine=engine)
        res = executor.execute(workload, batch_size=4)

        self.assertEqual(len(admitted_log), 4)
        self.assertEqual(admitted_log, [0, 1, 2, 3])
        self.assertGreater(len(steps_log), 0)
        self.assertEqual(len(finished_log), 4)
        self.assertEqual(set(finished_log), {0, 1, 2, 3})


    def test_composite_and_prefix_clustering_priority_rules(self):
        from admission_control.algorithms.policies import (
            CompositePriorityRule,
            PrefixClusteringPriorityRule,
        )

        # Composite rule
        comp = self.Priority.composite({"lpt": 1.0, "shortest_prompt": 0.0010})
        self.assertIsInstance(comp, CompositePriorityRule)
        r1 = Request(req_id=1, prompt_len=200, decode_len=500)
        r2 = Request(req_id=2, prompt_len=100, decode_len=500)
        self.assertLess(comp.key(r2), comp.key(r1))

        # Prefix clustering rule
        pref = self.Priority.prefix_cluster("session_id")
        self.assertIsInstance(pref, PrefixClusteringPriorityRule)
        r_a = Request(req_id=1, prompt_len=100, decode_len=500, metadata={"session_id": "sess_1"})
        r_b = Request(req_id=2, prompt_len=100, decode_len=500, metadata={"session_id": "sess_2"})
        key_a = pref.key(r_a)
        key_b = pref.key(r_b)
        self.assertEqual(key_a[0], "sess_1")
        self.assertEqual(key_b[0], "sess_2")
        self.assertLess(key_a, key_b)

    def test_cli_policy_subcommand(self):
        from admission_control.cli import cmd_policy
        import io
        from contextlib import redirect_stdout

        # Test CLI describe
        class DescribeArgs:
            name = "active"
            describe = True
            verify = False
            optimize = False
            compare = False
            workload = None
            capacity = 120_000
            batch_size = 64
            num_requests = None

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_policy(DescribeArgs())
        out = f.getvalue()
        self.assertIn("ComposablePolicy", out)
        self.assertIn("Priority Rule", out)

    def test_compare_policies_and_comparison_matrix(self):
        from admission_control.algorithms.solver import compare_policies, PolicyComparisonMatrix
        from admission_control.core.models import HardwareSpec

        workload = [
            {"req_id": i, "prompt_len": 32, "decode_len": 64}
            for i in range(16)
        ]
        hw = HardwareSpec(capacity_tokens=10_000, max_batch_size=16)
        policies = [
            self.PolicyBuilder("PolA").with_priority(self.Priority.lpt()).admit_if(self.Guard.concurrency_cap(8)).build(),
            self.PolicyBuilder("PolB").with_priority(self.Priority.spt()).admit_if(self.Guard.concurrency_cap(8)).build(),
        ]
        matrix = compare_policies(policies, workload, hw=hw, batch_size=8)
        self.assertIsInstance(matrix, PolicyComparisonMatrix)
        self.assertEqual(len(matrix.records), 2)

        summary = matrix.summary()
        self.assertIn("PolA", summary)
        self.assertIn("PolB", summary)

        md = matrix.to_markdown()
        self.assertIn("| Policy Strategy |", md)
        self.assertIn("PolA", md)

    def test_policy_registry_and_cli_list(self):
        from admission_control.algorithms.policies import (
            register_policy,
            list_registered_policies,
            get_policy,
            BasePolicy,
        )
        from admission_control.cli import cmd_policy
        import io
        from contextlib import redirect_stdout

        # Register custom policy
        class CustomRegistryPolicy(BasePolicy):
            name = "CustomRegistryPolicy"

        register_policy("test_custom_reg", CustomRegistryPolicy, aliases=["custom_reg_alias"])

        # Verify retrieval via get_policy
        pol1 = get_policy("test_custom_reg")
        self.assertIsInstance(pol1, CustomRegistryPolicy)
        pol2 = get_policy("custom_reg_alias")
        self.assertIsInstance(pol2, CustomRegistryPolicy)

        # Verify list_registered_policies
        reg_list = list_registered_policies()
        self.assertIn("test_custom_reg", reg_list)
        self.assertIn("custom_reg_alias", reg_list)

        # Verify CLI --list
        class ListArgs:
            name = "active"
            list = True
            describe = False
            verify = False
            optimize = False
            compare = False
            workload = None
            capacity = 120_000
            batch_size = 64
            num_requests = None

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_policy(ListArgs())
        out = f.getvalue()
        self.assertIn("REGISTERED SCHEDULING POLICIES", out)
        self.assertIn("test_custom_reg", out)

    def test_admission_decision_and_diagnostics(self):
        from admission_control.algorithms.policies import (
            AdmissionDecision,
            ConcurrencyCapGuard,
            PeakMemoryGuard,
            SystemState,
        )

        # Test AdmissionDecision boolean behavior and diagnostics
        d_pass = AdmissionDecision(True, "CapGuard", "Streams within cap")
        d_fail = AdmissionDecision(False, "CapGuard", "Streams exceeded cap", {"cap": 10})
        self.assertTrue(d_pass)
        self.assertFalse(d_fail)
        self.assertEqual(d_fail.reason, "Streams exceeded cap")
        self.assertEqual(d_fail.details["cap"], 10)

        # Test ConcurrencyCapGuard.evaluate()
        cg = ConcurrencyCapGuard(5)
        cand = {"prompt_len": 50, "decode_len": 50}
        active_4 = [{"prompt_len": 50, "decode_len": 50, "tokens_decoded": 0} for _ in range(4)]
        active_5 = [{"prompt_len": 50, "decode_len": 50, "tokens_decoded": 0} for _ in range(5)]

        dec_pass = cg.evaluate(active_4, cand)
        self.assertTrue(dec_pass.admitted)
        self.assertIn("below concurrency cap", dec_pass.reason)

        dec_fail = cg.evaluate(active_5, cand)
        self.assertFalse(dec_fail.admitted)
        self.assertIn("reached concurrency cap", dec_fail.reason)
        self.assertEqual(dec_fail.details["active_count"], 5)

        # Test ComposablePolicy.evaluate_admission()
        pol = self.PolicyBuilder("DiagPolicy").with_priority(self.Priority.lpt()).admit_if(cg).build()
        p_dec = pol.evaluate_admission(active_5, cand)
        self.assertFalse(p_dec.admitted)
        self.assertIn("reached concurrency cap", p_dec.reason)

    def test_executor_throttling_breakdown(self):
        from admission_control.core.models import HardwareSpec

        # Policy with strict cap of 2
        pol = self.PolicyBuilder("ThrottledPolicy").with_priority(self.Priority.lpt()).admit_if(self.Guard.concurrency_cap(2)).build()
        engine = self.SimulatedExecutionEngine(capacity_tokens=5000, block_size=16)
        executor = self.PolicyExecutor(policy=pol, engine=engine)
        workload = [
            {"req_id": i, "prompt_len": 32, "decode_len": 5}
            for i in range(4)
        ]
        res = executor.execute(workload, batch_size=4)
        self.assertIn("throttling_breakdown", res)
        self.assertGreater(res["throttled_events"], 0)
        self.assertIn("ConcurrencyCap(2)", res["throttling_breakdown"])
        self.assertGreater(res["throttling_breakdown"]["ConcurrencyCap(2)"], 0)

    def test_analyze_policy_capacity_scaling(self):
        from admission_control.algorithms.solver import analyze_policy_capacity_scaling, CapacityScalingReport
        from admission_control.core.models import HardwareSpec

        workload = [
            {"req_id": i, "prompt_len": 32, "decode_len": 64}
            for i in range(16)
        ]
        policy = self.PolicyBuilder("ScalingTest").with_priority(self.Priority.lpt()).admit_if(self.Guard.concurrency_cap(8)).build()
        report = analyze_policy_capacity_scaling(policy, workload, capacities=[5000, 10000], batch_size=8)
        self.assertIsInstance(report, CapacityScalingReport)
        self.assertEqual(len(report.records), 2)
        self.assertEqual(report.records[0]["capacity"], 5000)
        self.assertEqual(report.records[1]["capacity"], 10000)

        summary = report.summary()
        self.assertIn("Capacity Scaling Analysis", summary)
        self.assertIn("ScalingTest", summary)

        md = report.to_markdown()
        self.assertIn("| Capacity Pool (tokens) |", md)
        self.assertIn("YES (0 preemptions)", md)


    def test_adaptive_and_age_boosted_priority_rules(self):
        from admission_control.core.models import SystemState, Request
        from admission_control.algorithms.policies import Priority, PolicyBuilder

        # Test AdaptiveMemoryPriorityRule
        rule_mem = Priority.adaptive_memory(Priority.lpt(), Priority.spt(), threshold=0.80)
        req_short = Request(req_id=1, prompt_len=100, decode_len=50, arrival_time=100)
        req_long = Request(req_id=2, prompt_len=100, decode_len=1000, arrival_time=0)

        state_low = SystemState(step=10, used_tokens=50_000, capacity_tokens=100_000)
        state_high = SystemState(step=10, used_tokens=90_000, capacity_tokens=100_000)

        # Under low memory: LPT ranks req_long first
        order_low = sorted([req_short, req_long], key=lambda r: rule_mem.key(r, state_low))
        self.assertEqual(order_low[0].req_id, 2)

        # Under high memory: SPT ranks req_short first
        order_high = sorted([req_short, req_long], key=lambda r: rule_mem.key(r, state_high))
        self.assertEqual(order_high[0].req_id, 1)

        # Test AgeBoostedPriorityRule
        rule_age = Priority.age_boosted(Priority.spt(), age_weight=10.0)
        state_cur = SystemState(step=100)
        order_age = sorted([req_short, req_long], key=lambda r: rule_age.key(r, state_cur))
        # req_long waited 100 steps (score = 1000 - 1000 = 0), req_short waited 0 steps (score = 50)
        self.assertEqual(order_age[0].req_id, 2)

        # Test building policy with adaptive memory priority
        pol = PolicyBuilder("AdaptivePolicy").with_priority(rule_mem).build()
        self.assertIn("AdaptiveMemory", pol.describe())
        d = pol.to_dict()
        self.assertIn("AdaptiveMemory", d["priority_rule"])
    def test_conditional_guard_and_length_guards(self):
        from admission_control.core.models import SystemState
        from admission_control.algorithms.policies import Guard

        # Test ConditionalGuard
        cg = Guard.conditional(
            condition=lambda s, a: (s.used_tokens / s.capacity_tokens) > 0.80 if s else False,
            true_guard=Guard.concurrency_cap(4),
            false_guard=Guard.concurrency_cap(10),
            name="AdaptiveCap",
        )

        active_5 = [{"prompt_len": 50, "decode_len": 50, "tokens_decoded": 0} for _ in range(5)]
        cand = {"prompt_len": 50, "decode_len": 50}

        state_low = SystemState(used_tokens=50_000, capacity_tokens=100_000)
        dec_low = cg.evaluate(active_5, cand, state_low)
        self.assertTrue(dec_low.admitted)
        self.assertIn("FALSE branch", dec_low.reason)

        state_high = SystemState(used_tokens=90_000, capacity_tokens=100_000)
        dec_high = cg.evaluate(active_5, cand, state_high)
        self.assertFalse(dec_high.admitted)
        self.assertIn("TRUE branch", dec_high.reason)

        # Test PromptLengthGuard
        plg = Guard.prompt_length_limit(1024)
        self.assertTrue(plg.can_admit([], {"prompt_len": 512, "decode_len": 128}))
        self.assertFalse(plg.can_admit([], {"prompt_len": 2048, "decode_len": 128}))

        # Test DecodeLengthGuard
        dlg = Guard.decode_length_limit(2048)
        self.assertTrue(dlg.can_admit([], {"prompt_len": 512, "decode_len": 1024}))
        self.assertFalse(dlg.can_admit([], {"prompt_len": 512, "decode_len": 4096}))

    def test_policy_mutation_algebra(self):
        from admission_control.algorithms.policies import get_active_policy, Priority

        pol = get_active_policy()

        # Test clone into builder
        builder = pol.clone("ClonedKnapsack")
        cloned = builder.build()
        self.assertEqual(cloned.name, "ClonedKnapsack")
        self.assertEqual(cloned.priority_rule.name, pol.priority_rule.name)

        # Test with_priority mutation
        p_spt = pol.with_priority(Priority.spt())
        self.assertEqual(p_spt.priority_rule.name, "SPT")
        self.assertIn("P=SPT", p_spt.name)

        # Test with_concurrency_cap mutation
        p_cap = pol.with_concurrency_cap(112)
        self.assertIn("ConcurrencyCap(112)", p_cap.admission_guard.name)
        self.assertIn("Cap=112", p_cap.name)

        # Test with_capacity mutation
        p_mem = pol.with_capacity(80_000)
        self.assertIn("PeakMemoryGuard(80000)", p_mem.admission_guard.name)
        self.assertIn("Mem=80000", p_mem.name)


    def test_block_manager_radix_prefix_caching(self):
        from admission_control.simulation.block_manager import BlockManager

        # 1. Default non-caching behavior
        bm_std = BlockManager(capacity_tokens=1000, block_size=16, enable_prefix_caching=False)
        bm_std.allocate(req_id=1, num_blocks=10)
        bm_std.allocate(req_id=2, num_blocks=10)
        self.assertEqual(bm_std.used_blocks, 20)
        bm_std.free(1)
        self.assertEqual(bm_std.used_blocks, 10)
        bm_std.free(2)
        self.assertEqual(bm_std.used_blocks, 0)

        # 2. Reference-counted prefix caching
        bm_radix = BlockManager(capacity_tokens=1000, block_size=16, enable_prefix_caching=True)
        # Req 1: total 10 blocks, 6 prefix blocks with 'sys_prompt'
        bm_radix.allocate(req_id=1, num_blocks=10, prefix_hash="sys_prompt", prefix_tokens=6 * 16)
        self.assertEqual(bm_radix.used_blocks, 10) # 6 shared + 4 unique = 10

        # Req 2: total 12 blocks, shares same 'sys_prompt' (6 blocks) -> only allocates 6 unique blocks!
        bm_radix.allocate(req_id=2, num_blocks=12, prefix_hash="sys_prompt", prefix_tokens=6 * 16)
        self.assertEqual(bm_radix.used_blocks, 16) # 6 shared + 4 (req 1) + 6 (req 2) = 16 (instead of 22)
        self.assertEqual(bm_radix.prefix_hits, 1)
        self.assertEqual(bm_radix.saved_tokens, 6 * 16)

        # Free Req 1: unique 4 freed. 6 shared blocks stay allocated for Req 2
        freed_1 = bm_radix.free(1)
        self.assertEqual(freed_1, 4)
        self.assertEqual(bm_radix.used_blocks, 12) # 6 shared + 6 (req 2) = 12

        # Free Req 2: unique 6 freed. Prefix ref_count drops to 0, so 6 shared blocks freed
        freed_2 = bm_radix.free(2)
        self.assertEqual(freed_2, 12) # 6 unique + 6 shared = 12
        self.assertEqual(bm_radix.used_blocks, 0)
        self.assertEqual(bm_radix.free_blocks, bm_radix.total_blocks)

    def test_executor_queuing_delay_and_concurrency_analytics(self):
        from admission_control.core.models import HardwareSpec
        from admission_control.algorithms.policies import PolicyBuilder, Priority, Guard
        from admission_control.algorithms.executor import PolicyExecutor, SimulatedExecutionEngine

        # Workload with staggered arrivals: req 0 at step 0, req 1 at step 10
        workload = [
            {"req_id": 0, "prompt_len": 32, "decode_len": 5, "arrival_step": 0},
            {"req_id": 1, "prompt_len": 32, "decode_len": 5, "arrival_step": 10},
        ]
        policy = PolicyBuilder("DelayTest").with_priority(Priority.lpt()).admit_if(Guard.concurrency_cap(1)).build()
        engine = SimulatedExecutionEngine(capacity_tokens=5000, block_size=16)
        executor = PolicyExecutor(policy=policy, engine=engine)
        res = executor.execute(workload, batch_size=2)

        self.assertIn("avg_queuing_delay_steps", res)
        self.assertIn("max_queuing_delay_steps", res)
        self.assertIn("peak_active_concurrency", res)
        self.assertEqual(res["peak_active_concurrency"], 1)
        self.assertIsInstance(res["avg_queuing_delay_steps"], float)


    def test_cli_policy_spec_file_loading(self):
        import io
        import tempfile
        import yaml
        from contextlib import redirect_stdout
        from admission_control.cli import cmd_policy

        spec_content = {
            "name": "YAMLTestPolicy",
            "priority": {"type": "lpt"},
            "guards": [{"type": "concurrency_cap", "cap": 96}],
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tf:
            yaml.dump(spec_content, tf)
            tmp_path = tf.name

        try:
            class SpecArgs:
                name = "active"
                spec = tmp_path
                export = None
                list = False
                describe = True
                verify = False
                optimize = False
                compare = False

            f = io.StringIO()
            with redirect_stdout(f):
                cmd_policy(SpecArgs())
            cli_out = f.getvalue()
            self.assertIn("YAMLTestPolicy", cli_out)
            self.assertIn("ConcurrencyCap(96)", cli_out)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_adaptive_concurrency_guard(self):
        from admission_control.core.models import SystemState
        from admission_control.algorithms.policies import (
            Guard,
            AdaptiveConcurrencyGuard,
            PolicyBuilder,
            _guard_complexity_rank,
        )

        guard = Guard.adaptive_concurrency(high_cap=128, throttled_cap=103, memory_threshold=0.85, capacity=100_000)
        self.assertIsInstance(guard, AdaptiveConcurrencyGuard)
        self.assertEqual(_guard_complexity_rank(guard), 1)

        # Low memory state (50% utilization): cap should be 128
        state_low = SystemState(step=10, used_tokens=50_000, capacity_tokens=100_000)
        dummy_active_110 = [{"prompt_len": 100, "tokens_decoded": 100}] * 110
        dec_low = guard.evaluate(dummy_active_110, {"prompt_len": 100, "decode_len": 100}, state_low)
        self.assertTrue(dec_low.admitted)
        self.assertEqual(dec_low.details["target_cap"], 128)

        # High memory state (90% utilization): cap should be 103
        state_high = SystemState(step=20, used_tokens=90_000, capacity_tokens=100_000)
        dec_high = guard.evaluate(dummy_active_110, {"prompt_len": 100, "decode_len": 100}, state_high)
        self.assertFalse(dec_high.admitted)
        self.assertEqual(dec_high.details["target_cap"], 103)

        # Serialization and deserialization
        g_dict = guard.to_dict()
        self.assertEqual(g_dict["type"], "adaptive_concurrency")
        self.assertEqual(g_dict["high_cap"], 128)
        self.assertEqual(g_dict["throttled_cap"], 103)

        pol = PolicyBuilder("AdaptiveTest").admit_if(guard).build()
        self.assertIn("AdaptiveConcurrency", pol.describe())

        parsed = PolicyBuilder._parse_guard_spec(g_dict)
        self.assertIsInstance(parsed, AdaptiveConcurrencyGuard)
        self.assertEqual(parsed.high_cap, 128)
        self.assertEqual(parsed.throttled_cap, 103)


    def test_batch_cohort_policy_ordering(self):
        """Verifies policy ordering integrates seamlessly with immutable BatchCohort instances."""
        policy = self.PolicyBuilder("Knapsack-Cohort-Test").with_priority(self.Priority.lpt()).build()
        reqs = [
            Request(req_id=0, prompt_len=100, decode_len=200),
            Request(req_id=1, prompt_len=100, decode_len=800),
            Request(req_id=2, prompt_len=100, decode_len=500),
        ]
        cohort = BatchCohort.from_requests(reqs, batch_id=7, arrival_step=42)
        ordered_cohort = policy.order_batch(cohort)
        self.assertIsInstance(ordered_cohort, BatchCohort)
        self.assertEqual(ordered_cohort.batch_id, 7)
        self.assertEqual(ordered_cohort.arrival_step, 42)
        self.assertEqual([r.req_id for r in ordered_cohort], [1, 2, 0])


class TestModularExperimentationAndPolicyTesting(unittest.TestCase):
    """Tests the modular experiment configuration, RCM runner, testing and sweep utilities."""

    def setUp(self):
        self.small_workload = [
            {"req_id": 0, "prompt_len": 128, "decode_len": 256},
            {"req_id": 1, "prompt_len": 256, "decode_len": 512},
            {"req_id": 2, "prompt_len": 64, "decode_len": 128},
            {"req_id": 3, "prompt_len": 128, "decode_len": 256},
        ]

    def test_experiment_config_serialization(self):
        from admission_control.experiments.config import ExperimentConfig
        cfg = ExperimentConfig(
            policy="knapsack",
            workload=self.small_workload,
            batch_size=2,
            capacity_tokens=50_000,
            target_concurrency=4,
        )
        d = cfg.to_dict()
        self.assertEqual(d["policy"], "knapsack")
        self.assertEqual(d["batch_size"], 2)
        self.assertEqual(d["capacity_tokens"], 50_000)
        self.assertEqual(d["target_concurrency"], 4)

        reconstructed = ExperimentConfig.from_dict(d)
        self.assertEqual(reconstructed.policy, "knapsack")
        self.assertEqual(reconstructed.batch_size, 2)

    def test_policy_experiment_runner_rcm_execution(self):
        from admission_control.experiments.config import ExperimentConfig
        from admission_control.experiments.runner import PolicyExperimentRunner
        cfg = ExperimentConfig(
            policy="active",
            workload=self.small_workload,
            batch_size=2,
            capacity_tokens=60_000,
            baseline_policy="greedy",
        )
        runner = PolicyExperimentRunner(cfg)
        payload = runner.run()

        self.assertIn("candidate", payload)
        self.assertIn("baseline", payload)
        self.assertIn("metrics", payload)
        metrics = payload["metrics"]
        self.assertEqual(metrics["preemption_immunity_verified"], 1)
        self.assertEqual(metrics["starvation_freedom_verified"], 1)
        self.assertGreater(metrics["goodput_tok_per_s"], 0)

    def test_policy_convenience_creation_and_resolution(self):
        from admission_control.algorithms.policies import (
            create_policy,
            get_policy,
            Priority,
            Guard,
            BasePolicy,
        )
        # 1. create_policy
        p = create_policy(
            name="MyCustomPolicy",
            priority=Priority.shortest_processing_time(),
            guard=Guard.concurrency_cap(16),
            concurrency_cap=16,
        )
        self.assertIsInstance(p, BasePolicy)
        self.assertEqual(p.name, "MyCustomPolicy")

        # 2. get_policy with class
        class SubPolicy(BasePolicy):
            def __init__(self, **kw):
                super().__init__("SubPolicy")
        resolved = get_policy(SubPolicy)
        self.assertIsInstance(resolved, SubPolicy)

        # 3. get_policy with YAML string file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tf:
            tf.write("""
name: YamlFilePolicy
priority:
  type: functional
  name: lpt
concurrency_cap: 32
""")
            tmp_yaml_path = tf.name
        try:
            p_yaml = get_policy(tmp_yaml_path)
            self.assertEqual(p_yaml.name, "YamlFilePolicy")
        finally:
            if os.path.exists(tmp_yaml_path):
                os.remove(tmp_yaml_path)

    def test_policy_testing_utilities(self):
        from admission_control.experiments.testing import (
            test_policy,
            compare_policies,
            sweep_policy,
            quick_eval,
        )
        # 1. quick_eval
        q_res = quick_eval("active", num_requests=8, prompt_len=64, decode_len=128)
        self.assertTrue(q_res["passed"])
        self.assertEqual(q_res["num_preemptions"], 0)

        # 2. test_policy
        t_res = test_policy(
            policy="knapsack",
            workload=self.small_workload,
            batch_size=2,
            capacity_tokens=50_000,
        )
        self.assertTrue(t_res["passed"])
        self.assertGreater(t_res["goodput_tok_per_s"], 0)

        # 3. compare_policies
        c_res = compare_policies(
            policies=["greedy", "knapsack", "predictive"],
            workload=self.small_workload,
            batch_size=2,
            capacity_tokens=50_000,
            print_table=False,
        )
        self.assertEqual(len(c_res["comparison"]), 3)
        self.assertIn("best_policy", c_res)
        self.assertIn("table", c_res)

        # 4. sweep_policy
        s_res = sweep_policy(
            policy_factory=lambda c: f"knapsack",
            param_name="cap",
            values=[32, 64],
            workload=self.small_workload,
            batch_size=2,
            capacity_tokens=50_000,
            print_table=False,
        )
        self.assertEqual(len(s_res["results"]), 2)
        self.assertEqual(s_res["param_name"], "cap")
if __name__ == "__main__":
    unittest.main()
