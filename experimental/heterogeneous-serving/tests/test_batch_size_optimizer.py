"""
Unit Tests for Batch Size Policy Optimizer (admission_control/algorithms/batch_size_optimizer.py).
=================================================================================================
Tests:
- Theoretical upper bounds on average batch size & lower bounds on makespan
- Analytical equivalence between average batch size maximization and makespan minimization
- Exact Branch-and-Bound permutation search with lower bound pruning
- Exact Integer Linear Programming (CP-SAT) solver with linearized trapezoidal memory
- Combinatorial Policy Search over M=64 batch windows
- Preemption immunity and memory capacity enforcement
- CLI entrypoint for optimize-batch-size
"""

import os
import sys
import json
import tempfile
import unittest

from admission_control.core.models import Request, HardwareSpec
from admission_control.algorithms.batch_size_optimizer import (
    TheoreticalBounds,
    ScheduleMetrics,
    compute_theoretical_bounds,
    simulate_schedule_metrics,
    ILPBatchSizeOptimizer,
    BranchAndBoundBatchSizeOptimizer,
    PolicyBatchSizeOptimizer,
    BatchSizeOptimizationSuite,
)


class TestBatchSizeOptimizer(unittest.TestCase):
    """Test suite covering the mathematical foundations, solvers, and policies for batch size optimization."""

    def setUp(self):
        self.hw = HardwareSpec(capacity_tokens=120_000, max_batch_size=64)
        self.sample_requests = [
            Request(req_id=0, prompt_len=20, decode_len=10),
            Request(req_id=1, prompt_len=50, decode_len=30),
            Request(req_id=2, prompt_len=30, decode_len=25),
            Request(req_id=3, prompt_len=40, decode_len=40),
            Request(req_id=4, prompt_len=15, decode_len=15),
            Request(req_id=5, prompt_len=25, decode_len=20),
        ]

    def test_theoretical_bounds_derivation(self):
        """Validates critical path, volume, and memory deficit lower bounds on makespan and upper bounds on bar{B}."""
        bounds = compute_theoretical_bounds(self.sample_requests, self.hw)

        tot_d = sum(r.decode_len for r in self.sample_requests)
        max_d = max(r.decode_len for r in self.sample_requests)

        self.assertEqual(bounds.total_decode_tokens, tot_d)
        self.assertEqual(bounds.critical_path_makespan, max_d)
        self.assertGreaterEqual(bounds.theoretical_min_makespan, max_d)
        self.assertLessEqual(bounds.theoretical_max_avg_batch, self.hw.max_batch_size)
        self.assertAlmostEqual(
            bounds.critical_path_avg_batch,
            tot_d / max_d,
            places=2,
        )

        # Empty requests list test
        empty_bounds = compute_theoretical_bounds([], self.hw)
        self.assertEqual(empty_bounds.num_requests, 0)
        self.assertEqual(empty_bounds.theoretical_max_avg_batch, 0.0)

    def test_equivalence_of_average_batch_size_and_makespan(self):
        """
        Validates the fundamental mathematical theorem:
        sum_{t=0}^{T-1} B(t) == D_total
        bar{B} == D_total / T
        """
        # Test across two different start-time schedules
        starts_a = {r.req_id: 0 for r in self.sample_requests}
        metrics_a = simulate_schedule_metrics(self.sample_requests, starts_a, self.hw, "Concurrent")

        # In Schedule A: all start at 0, makespan = max_d
        tot_d = sum(r.decode_len for r in self.sample_requests)
        self.assertEqual(sum(metrics_a.active_profile), tot_d)
        self.assertAlmostEqual(metrics_a.avg_batch_size, tot_d / metrics_a.makespan_steps, places=2)

        # Schedule B: sequential execution
        starts_b = {}
        cur_t = 0
        for r in self.sample_requests:
            starts_b[r.req_id] = cur_t
            cur_t += r.decode_len
        metrics_b = simulate_schedule_metrics(self.sample_requests, starts_b, self.hw, "Sequential")

        self.assertEqual(sum(metrics_b.active_profile), tot_d)
        self.assertAlmostEqual(metrics_b.avg_batch_size, 1.0, places=2)
        # Sequential makespan is much larger, so avg batch size is strictly smaller
        self.assertLess(metrics_b.avg_batch_size, metrics_a.avg_batch_size)

    def test_branch_and_bound_exact_solver(self):
        """Validates exact Branch-and-Bound solver on small instances with pruning."""
        # Using 6 requests
        bnb_solver = BranchAndBoundBatchSizeOptimizer(self.hw, time_limit_s=5.0)
        metrics = bnb_solver.solve(self.sample_requests)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.preemptions, 0)
        self.assertLessEqual(metrics.peak_mem_tokens, self.hw.capacity_tokens)

        # Makespan must be at least the critical path lower bound
        max_d = max(r.decode_len for r in self.sample_requests)
        self.assertGreaterEqual(metrics.makespan_steps, max_d)
        self.assertGreater(metrics.avg_batch_size, 0.0)

    def test_ilp_cpsat_solver(self):
        """Validates OR-Tools CP-SAT exact formulation on small instances."""
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            self.skipTest("OR-Tools not available")

        ilp_solver = ILPBatchSizeOptimizer(self.hw, time_limit_s=10.0, step_delta=1)
        metrics = ilp_solver.solve(self.sample_requests)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.preemptions, 0)
        self.assertLessEqual(metrics.peak_mem_tokens, self.hw.capacity_tokens)

        # Makespan must achieve or approach theoretical minimum
        max_d = max(r.decode_len for r in self.sample_requests)
        self.assertGreaterEqual(metrics.makespan_steps, max_d)

        # Verify exact mathematical equivalence
        tot_d = sum(r.decode_len for r in self.sample_requests)
        self.assertAlmostEqual(metrics.avg_batch_size, tot_d / metrics.makespan_steps, places=2)

    def test_policy_batch_size_optimizer(self):
        """Validates combinatorial policy search over priority rules and concurrency caps."""
        # Create a trace of 20 requests
        requests = [
            Request(req_id=i, prompt_len=200 + (i * 37) % 500, decode_len=100 + (i * 73) % 400)
            for i in range(20)
        ]
        pol_optimizer = PolicyBatchSizeOptimizer(self.hw, batch_window_m=64)
        best_metrics, all_metrics = pol_optimizer.optimize(
            requests,
            candidate_concurrencies=[16, 32, 64],
        )

        self.assertIsNotNone(best_metrics)
        self.assertEqual(best_metrics.preemptions, 0)
        self.assertGreater(len(all_metrics), 1)

        # All evaluated policies must maintain zero preemptions if feasible
        for m in all_metrics:
            self.assertLessEqual(m.peak_mem_tokens, self.hw.capacity_tokens)

        # Best policy must have maximal average batch size among zero-preemption policies
        for m in all_metrics:
            if m.preemptions == 0:
                self.assertGreaterEqual(best_metrics.avg_batch_size, m.avg_batch_size)

    def test_batch_size_optimization_suite(self):
        """Validates running the complete multi-method comparative suite."""
        suite = BatchSizeOptimizationSuite(hw=self.hw, batch_window_m=64)
        results = suite.run_suite(
            requests=self.sample_requests,
            run_ilp=True,
            run_bnb=True,
            run_policy_search=True,
            ilp_time_limit_s=5.0,
            bnb_time_limit_s=5.0,
        )

        self.assertIn("bounds", results)
        self.assertIn("evaluations", results)
        self.assertIn("best_schedule", results)
        self.assertGreater(len(results["evaluations"]), 0)

        # Table formatting
        table = suite.format_comparison_table(results)
        self.assertIn("Theoretical Upper Bound", table)
        self.assertIn("Avg Batch", table)
        self.assertIn("Makespan", table)

    def test_cli_command_invocation(self):
        """Validates CLI entrypoint integration for optimize-batch-size."""
        from admission_control.cli import main
        # Save sample requests to a temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            raw = [r.to_dict() for r in self.sample_requests]
            json.dump(raw, f)
            temp_path = f.name

        try:
            # Test running through CLI parser
            old_argv = sys.argv
            sys.argv = [
                "admission-control",
                "optimize-batch-size",
                "--workload",
                temp_path,
                "--method",
                "policy-search",
                "--max-batch-size",
                "32",
            ]
            # Must run without raising exception
            main()
        finally:
            sys.argv = old_argv
            if os.path.exists(temp_path):
                os.remove(temp_path)
    def test_makespan_objectives_step_vs_wall_time(self):
        """Validates that step-makespan and wall-time objectives run and minimize their respective metrics."""
        suite = BatchSizeOptimizationSuite(hw=self.hw, batch_window_m=64)
        res_step = suite.run_suite(
            requests=self.sample_requests,
            run_ilp=True,
            run_bnb=False,
            run_policy_search=True,
            objective="step-makespan",
        )
        res_wall = suite.run_suite(
            requests=self.sample_requests,
            run_ilp=True,
            run_bnb=False,
            run_policy_search=True,
            objective="wall-time",
        )
        self.assertIsNotNone(res_step["best_schedule"])
        self.assertIsNotNone(res_wall["best_schedule"])
        self.assertEqual(res_step["best_schedule"].preemptions, 0)
        self.assertEqual(res_wall["best_schedule"].preemptions, 0)

    def test_cli_optimize_makespan_invocation(self):
        """Validates CLI entrypoint integration for optimize-makespan."""
        from admission_control.cli import main
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            raw = [r.to_dict() for r in self.sample_requests]
            json.dump(raw, f)
            temp_path = f.name

        try:
            old_argv = sys.argv
            sys.argv = [
                "admission-control",
                "optimize-makespan",
                "--workload",
                temp_path,
                "--method",
                "policy-search",
                "--objective",
                "wall-time",
            ]
            main()
        finally:
            sys.argv = old_argv
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
