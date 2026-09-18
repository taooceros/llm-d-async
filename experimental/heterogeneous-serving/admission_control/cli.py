"""
Unified Command-Line Interface for Admission Control & TPU Serving.
===================================================================
Commands:
  solve     : Solves deterministic clairvoyant schedule for a workload trace
  simulate  : Runs discrete-event vLLM scheduler simulation with preemption metrics
  topology  : Discovers and displays connected Cloud TPU worker hosts on Ray
  profile   : Displays token distribution statistics for a workload trace
  policy    : Inspect, analytically verify, and optimize composable scheduling policies
"""

import sys
import os
import json
import argparse

from admission_control.core.models import Request, HardwareSpec
from admission_control.algorithms.solver import (
    solve_schedule,
    run_comparative_suite,
    solve_optimal_composable_policy,
)
from admission_control.algorithms.policies import (
    get_policy,
    get_active_policy,
    verify_policy_invariants,
)
from admission_control.workloads.loader import load_workload_requests, get_workload_profile
from admission_control.tpu.topology import TPUTopologyDiscovery
from admission_control.algorithms.batch_size_optimizer import BatchSizeOptimizationSuite


def cmd_solve(args):
    hw = HardwareSpec(capacity_tokens=args.capacity, max_batch_size=args.max_batch_size)
    requests = load_workload_requests(args.workload, max_requests=args.num_requests)
    print(f"[Solver] Loaded {len(requests)} requests from '{args.workload}' (Capacity: {hw.capacity_tokens:,} tokens)")

    if args.compare:
        results = run_comparative_suite(requests, hw)
        print("\n" + "=" * 90)
        print(f"{'Policy Name':<45} | {'Makespan':<10} | {'Time (s)':<10} | {'Goodput (tok/s)':<16}")
        print("-" * 90)
        for name, plan in results.items():
            print(f"{plan.policy_name:<45} | {plan.makespan_steps:<10,} | {plan.total_time_s:<10.2f} | {plan.goodput_tok_s:<16,.1f}")
        print("=" * 90)
    else:
        plan = solve_schedule(requests, hw, priority=args.priority)
        print(f"\nOptimal Plan: {plan.policy_name}")
        print(f"  Makespan Steps : {plan.makespan_steps:,}")
        print(f"  Total Time     : {plan.total_time_s:.2f} s")
        print(f"  Goodput        : {plan.goodput_tok_s:,.1f} tok/s")
        print(f"  Peak Memory    : {plan.peak_mem_tokens:,} tokens ({plan.peak_mem_pct:.1f}%)")
        print(f"  Padding Waste  : {plan.padding_waste_pct:.1f}%")
        print(f"  Preemptions    : {plan.preemptions}")

        if args.output:
            plan.save(args.output)
            print(f"\nSaved schedule plan to {args.output}")


def cmd_simulate(args):
    sim = VLLMSchedulerSimulator(capacity_tokens=args.capacity, block_size=args.block_size)
    requests = load_workload_requests(args.workload, max_requests=args.num_requests)
    raw_reqs = [r.to_dict() for r in requests]
    print(f"[Simulator] Simulating {len(raw_reqs)} requests on {args.capacity:,}-token KV capacity pool...")

    results = sim.simulate_all_policies(raw_reqs)
    print("\n" + "=" * 90)
    print(f"{'Policy':<40} | {'Steps':<8} | {'Preempts':<9} | {'Discarded':<10} | {'Goodput':<14}")
    print("-" * 90)
    for res in results:
        print(
            f"{res.get('policy_name', 'Policy'):<40} | "
            f"{res['total_steps']:<8,} | "
            f"{res['total_preemptions']:<9} | "
            f"{res['total_discarded_tokens']:<10,} | "
            f"{res['goodput_tok_per_s']:<14.1f} tok/s"
        )
    print("=" * 90)


def cmd_topology(args):
    print("[Topology] Querying Ray cluster for connected TPU worker nodes...")
    discovery = TPUTopologyDiscovery()
    slices = discovery.discover_tpu_nodes()

    if not slices:
        print("No TPU nodes discovered on Ray cluster.")
        return

    print(f"\nDiscovered {len(slices)} TPU Slice(s):")
    for s_name, hosts in slices.items():
        print(f"\n  Slice: {s_name} ({len(hosts)} hosts):")
        for h in hosts:
            print(f"    - Worker {h.worker_id:2d}: IP={h.node_ip} | Coords={h.host_coords} | TPUs={h.tpu_count}")

    pair = discovery.select_connected_2host_submesh()
    print(f"\nGuaranteed Connected 2-Host Submesh (TP=8, Mesh: {pair.mesh_dimension}):")
    print(f"  Host A: Worker ID {pair.host_a.worker_id} ({pair.host_a.node_ip})")
    print(f"  Host B: Worker ID {pair.host_b.worker_id} ({pair.host_b.node_ip})")


def cmd_profile(args):
    profile = get_workload_profile(args.workload)
    print(f"\nToken Profile: {os.path.basename(args.workload)}")
    print("=" * 50)
    print(f"  Total Requests : {profile.num_requests:,}")
    print(f"  Prompt Tokens  : {profile.total_prompt_tokens:,} (min={profile.min_prompt}, avg={profile.avg_prompt:.1f}, max={profile.max_prompt})")
    print(f"  Decode Tokens  : {profile.total_decode_tokens:,} (min={profile.min_decode}, avg={profile.avg_decode:.1f}, max={profile.max_decode})")
    print(f"  Total Demand   : {profile.total_demand_tokens:,}")
    print("=" * 50)


def cmd_profile_xla(args):
    from admission_control.tpu.xla_profiler import XLAProfiler
    profiler = XLAProfiler(jax_cache_dir=args.cache_dir)
    report = profiler.generate_report()
    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved XLA profiling report to {args.output}")


def cmd_policy(args):
    if getattr(args, "list", False):
        from admission_control.algorithms.policies import list_registered_policies
        print("=" * 60)
        print("REGISTERED SCHEDULING POLICIES")
        print("=" * 60)
        for name, desc in sorted(list_registered_policies().items()):
            print(f"  {name:<22s} -> {desc}")
        print("=" * 60)
        return
    if getattr(args, "spec", None):
        from admission_control.algorithms.policies import PolicyBuilder
        spec_path = args.spec
        with open(spec_path, "r", encoding="utf-8") as f:
            content = f.read()
        if spec_path.endswith(".yaml") or spec_path.endswith(".yml"):
            pol = PolicyBuilder.from_yaml(content).build()
        else:
            import json
            pol = PolicyBuilder.from_dict(json.loads(content)).build()
    else:
        pol = get_policy(args.name) if args.name != "active" else get_active_policy()

    export_fmt = getattr(args, "export", None)
    if export_fmt:
        if export_fmt == "python":
            if hasattr(pol, "to_python_code"):
                print(pol.to_python_code())
            else:
                print(f"# Policy {pol.name} ({pol.__class__.__name__}) does not support Python code generation.")
        elif export_fmt == "yaml":
            if hasattr(pol, "to_yaml"):
                print(pol.to_yaml())
            else:
                import yaml
                print(yaml.dump(pol.to_dict() if hasattr(pol, "to_dict") else {"name": pol.name}))
        elif export_fmt == "json":
            if hasattr(pol, "to_json"):
                print(pol.to_json())
            else:
                import json
                print(json.dumps(pol.to_dict() if hasattr(pol, "to_dict") else {"name": pol.name}, indent=2))
        elif export_fmt in ("markdown", "md"):
            if hasattr(pol, "to_markdown"):
                print(pol.to_markdown())
            else:
                print(f"# Scheduling Policy: {pol.name}\n\n`{pol.__class__.__name__}`")
        elif export_fmt in ("mermaid", "mmd"):
            if hasattr(pol, "to_mermaid"):
                print(pol.to_mermaid())
            else:
                print(f"graph TD\n    Node[\"{pol.name}\"]")
        return
    do_describe = getattr(args, "describe", False)
    do_verify = getattr(args, "verify", False)
    do_optimize = getattr(args, "optimize", False)
    do_compare = getattr(args, "compare", False)
    if do_describe or (not do_verify and not do_optimize and not do_compare):
        if hasattr(pol, "describe"):
            print(pol.describe())
        else:
            print(f"Policy: {pol.name} ({pol.__class__.__name__})")

    if do_verify:
        if not args.workload:
            print("ERROR: --verify requires --workload <path>", file=sys.stderr)
            sys.exit(1)
        requests = load_workload_requests(args.workload, max_requests=args.num_requests)
        raw = [r.to_dict() for r in requests]
        report = verify_policy_invariants(pol, raw, batch_size=args.batch_size, capacity_tokens=args.capacity)
        print("\n" + report.summary())
        if not report.is_valid:
            sys.exit(1)

    if do_optimize:
        if not args.workload:
            print("ERROR: --optimize requires --workload <path>", file=sys.stderr)
            sys.exit(1)
        requests = load_workload_requests(args.workload, max_requests=args.num_requests)
        raw = [r.to_dict() for r in requests]
        hw = HardwareSpec(capacity_tokens=args.capacity, max_batch_size=128)
        best_pol, meta = solve_optimal_composable_policy(raw, hw, batch_size=args.batch_size)
        print(f"\nOptimal Composable Policy Discovered: {best_pol.name}")
        print(f"  Goodput: {meta['goodput_tok_per_s']:.1f} tok/s")
        print(f"  Preemptions: {meta['num_preemptions']}")
        print(f"  Total Time: {meta['total_time_s']:.2f} s")
        print(f"  Padding Waste: {meta['xla_padding_waste_percent']:.2f}%")

    if do_compare:
        if not args.workload:
            print("ERROR: --compare requires --workload <path>", file=sys.stderr)
            sys.exit(1)
        from admission_control.algorithms.policies import get_policy
        from admission_control.algorithms.solver import compare_policies
        requests = load_workload_requests(args.workload, max_requests=args.num_requests)
        raw = [r.to_dict() for r in requests]
        hw = HardwareSpec(capacity_tokens=args.capacity, max_batch_size=128)
        suite = [
            get_policy("greedy", max_concurrency=args.batch_size),
            get_policy("predictive", cap=48),
            get_policy("discrete_wave", wave_size=32),
            get_policy("knapsack", target_concurrency=96, capacity=args.capacity),
            pol,
        ]
        matrix = compare_policies(suite, raw, hw=hw, batch_size=args.batch_size)
        print("\n" + matrix.summary())

    do_trace = getattr(args, "trace", None)
    if do_trace:
        if not args.workload:
            print("ERROR: --trace requires --workload <path>", file=sys.stderr)
            sys.exit(1)
        from admission_control.algorithms.executor import PolicyExecutor, SimulatedExecutionEngine
        requests = load_workload_requests(args.workload, max_requests=args.num_requests)
        raw = [r.to_dict() for r in requests]
        hw = HardwareSpec(capacity_tokens=args.capacity, max_batch_size=128)
        engine = SimulatedExecutionEngine(capacity_tokens=args.capacity, block_size=16, hw=hw)
        executor = PolicyExecutor(policy=pol, engine=engine, hw=hw)
        res = executor.execute(raw, batch_size=args.batch_size, record_trace=True)
        trace = res.get("trace")
        if trace:
            trace.to_chrome_trace(do_trace)
            print(f"\nExported execution trace to {do_trace} ({len(trace.steps)} steps, {len(trace.requests)} requests)")
            print("Open https://ui.perfetto.dev or chrome://tracing to view interactive timeline visualization.")
def cmd_optimize_batch_size(args):
    hw = HardwareSpec(capacity_tokens=args.capacity, max_batch_size=args.max_batch_size)
    requests = load_workload_requests(args.workload, max_requests=args.num_requests)
    print(f"[Optimizer] Loaded {len(requests)} requests from '{args.workload}'")
    suite = BatchSizeOptimizationSuite(hw=hw, batch_window_m=args.batch_size)
    results = suite.run_suite(
        requests=requests,
        run_ilp=args.method in ["all", "ilp"],
        run_bnb=args.method in ["all", "bnb"],
        run_policy_search=args.method in ["all", "policy-search"],
        ilp_time_limit_s=args.time_limit,
        bnb_time_limit_s=args.time_limit,
        step_delta=args.step_delta,
        objective=getattr(args, "objective", "avg-batch-size"),
    )
    print(suite.format_comparison_table(results))
    if args.output:
        from dataclasses import asdict
        out_data = {
            "bounds": asdict(results["bounds"]),
            "evaluations": [m.to_dict() for m in results["evaluations"]],
            "best_policy": results["best_schedule"].to_dict() if results["best_schedule"] else None,
            "total_runtime_s": results["total_runtime_s"],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"[Export] Saved results to {args.output}")

def cmd_optimize_makespan(args):
    hw = HardwareSpec(capacity_tokens=args.capacity, max_batch_size=args.max_batch_size)
    requests = load_workload_requests(args.workload, max_requests=args.num_requests)
    print(f"[Makespan Optimizer] Loaded {len(requests)} requests from '{args.workload}'")
    suite = BatchSizeOptimizationSuite(hw=hw, batch_window_m=args.batch_size)
    results = suite.run_suite(
        requests=requests,
        run_ilp=args.method in ["all", "ilp"],
        run_bnb=args.method in ["all", "bnb"],
        run_policy_search=args.method in ["all", "policy-search"],
        ilp_time_limit_s=args.time_limit,
        bnb_time_limit_s=args.time_limit,
        step_delta=args.step_delta,
        objective=args.objective,
    )
    print(suite.format_comparison_table(results))
    if args.output:
        from dataclasses import asdict
        out_data = {
            "objective": args.objective,
            "bounds": asdict(results["bounds"]),
            "evaluations": [m.to_dict() for m in results["evaluations"]],
            "best_policy": results["best_schedule"].to_dict() if results["best_schedule"] else None,
            "total_runtime_s": results["total_runtime_s"],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"[Export] Saved results to {args.output}")

def cmd_experiment(args):
    from admission_control.experiments.config import ExperimentConfig
    from admission_control.experiments.runner import PolicyExperimentRunner

    baseline = None if getattr(args, "baseline", "greedy").lower() in ("none", "null", "") else args.baseline
    cfg = ExperimentConfig(
        policy=args.policy,
        workload=args.workload,
        batch_size=args.batch_size,
        capacity_tokens=args.capacity,
        target_concurrency=args.target_concurrency,
        baseline_policy=baseline,
        execution_mode=args.mode,
        output_json=args.output,
        trace_output=args.trace,
        record_trace=bool(args.trace),
        ray_url=args.ray_url,
        force_reinit=args.force_reinit,
    )
    runner = PolicyExperimentRunner(cfg)
    payload = runner.run()
    metrics = payload.get("metrics", {})
    if not metrics.get("preemption_immunity_verified", 1):
        print("ERROR: Policy violated preemption immunity!", file=sys.stderr)
        sys.exit(1)
    if not metrics.get("starvation_freedom_verified", 1):
        print("ERROR: Policy violated starvation freedom!", file=sys.stderr)
        sys.exit(1)


def cmd_test_policy(args):
    from admission_control.experiments.testing import (
        test_policy,
        compare_policies,
        sweep_policy,
        quick_eval,
    )
    if args.quick:
        res = quick_eval(args.policy, capacity_tokens=args.capacity)
        sys.exit(0 if res["passed"] else 1)
    if args.compare:
        compare_policies(args.compare, workload=args.workload, batch_size=args.batch_size, capacity_tokens=args.capacity)
        sys.exit(0)
    if args.sweep:
        from admission_control.algorithms.policies import PolicyBuilder, Priority, Guard, BatchTransition
        if not args.values:
            print("ERROR: --values required for --sweep", file=sys.stderr)
            sys.exit(1)
        if args.sweep == "cap":
            caps = [int(v) for v in args.values]
            def factory(c):
                return (
                    PolicyBuilder(f"Knapsack-Cap-{c}")
                    .with_priority(Priority.slack_packed_lpt(weight=0.0010))
                    .admit_if(Guard.concurrency_cap(c))
                    .admit_if(Guard.preemption_immune(capacity=args.capacity, block_size=16))
                    .draw_next_batch_if(BatchTransition.preemption_immune(capacity=args.capacity, target_concurrency=c, block_size=16))
                    .build()
                )
            sweep_policy(factory, "concurrency_cap", caps, workload=args.workload, capacity_tokens=args.capacity)
        elif args.sweep == "weight":
            weights = [float(v) for v in args.values]
            def factory(w):
                return (
                    PolicyBuilder(f"Knapsack-W-{w}")
                    .with_priority(Priority.slack_packed_lpt(weight=w))
                    .admit_if(Guard.concurrency_cap(103))
                    .admit_if(Guard.preemption_immune(capacity=args.capacity, block_size=16))
                    .draw_next_batch_if(BatchTransition.preemption_immune(capacity=args.capacity, target_concurrency=103, block_size=16))
                    .build()
                )
            sweep_policy(factory, "priority_weight", weights, workload=args.workload, capacity_tokens=args.capacity)
        sys.exit(0)
    res = test_policy(args.policy, workload=args.workload, batch_size=args.batch_size, capacity_tokens=args.capacity)
    sys.exit(0 if res["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(
        prog="admission-control",
        description="Admission Control & Clairvoyant Oracle Scheduling Suite for Cloud TPU v5e",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Solve
    p_solve = subparsers.add_parser("solve", help="Compute deterministic clairvoyant schedule plan")
    p_solve.add_argument("--workload", type=str, required=True, help="Workload JSON file or name")
    p_solve.add_argument("--capacity", type=int, default=120_000, help="KV cache token pool capacity")
    p_solve.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch concurrency")
    p_solve.add_argument("--priority", type=str, default="binned_lpt", choices=["binned_lpt", "pure_lpt", "spt", "density"])
    p_solve.add_argument("--num-requests", type=int, default=None, help="Cap number of requests")
    p_solve.add_argument("--compare", action="store_true", help="Compare with Policies A, B, D")
    p_solve.add_argument("--output", type=str, default=None, help="Path to save SchedulePlan JSON")
    p_solve.set_defaults(func=cmd_solve)

    # Simulate
    p_sim = subparsers.add_parser("simulate", help="Run discrete-event vLLM scheduler preemption simulation")
    p_sim.add_argument("--workload", type=str, required=True, help="Workload JSON file or name")
    p_sim.add_argument("--capacity", type=int, default=120_000, help="KV cache token pool capacity")
    p_sim.add_argument("--block-size", type=int, default=16, help="PagedAttention block size")
    p_sim.add_argument("--num-requests", type=int, default=None, help="Cap number of requests")
    p_sim.set_defaults(func=cmd_simulate)

    # Topology
    p_topo = subparsers.add_parser("topology", help="Discover TPU host placement and connected submeshes")
    p_topo.set_defaults(func=cmd_topology)

    # Profile
    p_prof = subparsers.add_parser("profile", help="Display token length distribution for a workload")
    p_prof.add_argument("--workload", type=str, required=True, help="Workload JSON file or name")
    p_prof.set_defaults(func=cmd_profile)

    # Profile XLA
    p_pxla = subparsers.add_parser("profile-xla", help="Analyze compiled XLA HLO modules and compiler metrics")
    p_pxla.add_argument("--cache-dir", type=str, default="/models/jax_cache", help="JAX HLO cache directory")
    p_pxla.add_argument("--output", type=str, default=None, help="Save report to file")
    p_pxla.set_defaults(func=cmd_profile_xla)

    # Policy
    p_pol = subparsers.add_parser("policy", help="Inspect, analytically verify, and optimize scheduling policies")
    p_pol.add_argument("--name", type=str, default="active", help="Policy name or 'active' (default: 'active')")
    p_pol.add_argument("--spec", type=str, default=None, help="Path to declarative policy YAML or JSON specification file")
    p_pol.add_argument("--describe", action="store_true", help="Print ASCII tree visualization of the policy")
    p_pol.add_argument("--verify", action="store_true", help="Analytically verify preemption immunity and starvation freedom")
    p_pol.add_argument("--optimize", action="store_true", help="Discover the optimal composable policy for the workload")
    p_pol.add_argument("--compare", action="store_true", help="Compare a suite of baseline and knapsack policies")
    p_pol.add_argument("--list", action="store_true", help="List all registered scheduling policies")
    p_pol.add_argument("--trace", type=str, default=None, help="Export step-by-step execution trace to Google Chrome Trace JSON format (compatible with ui.perfetto.dev)")
    p_pol.add_argument("--export", type=str, choices=["python", "yaml", "json", "markdown", "mermaid"], default=None, help="Export policy in python, yaml, json, markdown, or mermaid format")
    p_pol.add_argument("--workload", type=str, default=None, help="Path to workload JSON file")
    p_pol.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity")
    p_pol.add_argument("--batch-size", type=int, default=64, help="Batch cohort size M")
    p_pol.add_argument("--num-requests", type=int, default=None, help="Cap number of requests")
    p_pol.set_defaults(func=cmd_policy)
    # Optimize Batch Size
    p_obs = subparsers.add_parser("optimize-batch-size", help="Find optimal scheduling policy maximizing average batch size")
    p_obs.add_argument("--workload", type=str, required=True, help="Workload JSON file or name")
    p_obs.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity")
    p_obs.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch concurrency")
    p_obs.add_argument("--batch-size", type=int, default=64, help="Batch cohort window M")
    p_obs.add_argument("--num-requests", type=int, default=None, help="Cap number of requests")
    p_obs.add_argument("--method", type=str, default="all", choices=["all", "ilp", "bnb", "policy-search"])
    p_obs.add_argument("--time-limit", type=float, default=30.0, help="Time limit for exact solvers")
    p_obs.add_argument("--objective", type=str, default="avg-batch-size", choices=["avg-batch-size", "step-makespan", "wall-time"])
    p_obs.add_argument("--step-delta", type=int, default=1, help="Step delta for ILP")
    p_obs.add_argument("--output", type=str, default=None, help="Output JSON path")
    p_obs.set_defaults(func=cmd_optimize_batch_size)
    # Optimize Makespan
    p_oms = subparsers.add_parser("optimize-makespan", help="Find optimal scheduling policy minimizing makespan")
    p_oms.add_argument("--workload", type=str, required=True, help="Workload JSON file or name")
    p_oms.add_argument("--objective", type=str, default="wall-time", choices=["wall-time", "step-makespan", "avg-batch-size"], help="Makespan target")
    p_oms.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity")
    p_oms.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch concurrency")
    p_oms.add_argument("--batch-size", type=int, default=64, help="Batch cohort window M")
    p_oms.add_argument("--num-requests", type=int, default=None, help="Cap number of requests")
    p_oms.add_argument("--method", type=str, default="all", choices=["all", "ilp", "bnb", "policy-search"])
    p_oms.add_argument("--time-limit", type=float, default=30.0, help="Time limit for exact solvers")
    p_oms.add_argument("--step-delta", type=int, default=1, help="Step delta for ILP")
    p_oms.add_argument("--output", type=str, default=None, help="Output JSON path")
    p_oms.set_defaults(func=cmd_optimize_makespan)
    # Experiment
    p_exp = subparsers.add_parser("experiment", help="Run RCM-powered scheduling policy experiment (simulation or cluster)")
    p_exp.add_argument("--policy", type=str, default="active", help="Policy name, spec file (.yaml/.json), or 'active'")
    p_exp.add_argument("--workload", type=str, default="sharegpt_sampled_500.json", help="Workload file or standard name")
    p_exp.add_argument("--batch-size", type=int, default=64, help="Batch cohort window M")
    p_exp.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity")
    p_exp.add_argument("--target-concurrency", type=int, default=96, help="Target concurrency ceiling")
    p_exp.add_argument("--baseline", type=str, default="greedy", help="Baseline policy name ('none' to disable)")
    p_exp.add_argument("--mode", type=str, default="simulation", choices=["simulation", "cluster"], help="Execution mode")
    p_exp.add_argument("--output", type=str, default=None, help="Output JSON results artifact path")
    p_exp.add_argument("--trace", type=str, default=None, help="Output Chrome trace path")
    p_exp.add_argument("--ray-url", type=str, default="http://127.0.0.1:8265", help="Ray cluster URL")
    p_exp.add_argument("--force-reinit", action="store_true", help="Force reinitialize persistent cluster worker")
    p_exp.set_defaults(func=cmd_experiment)

    # Test Policy
    p_tp = subparsers.add_parser("test-policy", help="Quickly test, compare, and sweep scheduling policies")
    p_tp.add_argument("--policy", type=str, default="active", help="Policy to test")
    p_tp.add_argument("--workload", type=str, default="sharegpt_sampled_500.json", help="Workload file or standard name")
    p_tp.add_argument("--batch-size", type=int, default=64, help="Batch cohort window M")
    p_tp.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity")
    p_tp.add_argument("--compare", nargs="+", default=None, help="Compare multiple policies side by side")
    p_tp.add_argument("--sweep", type=str, default=None, choices=["cap", "weight"], help="Parameter to sweep")
    p_tp.add_argument("--values", nargs="+", default=None, help="Values for sweep")
    p_tp.add_argument("--quick", action="store_true", help="Run quick 64-request evaluation")
    p_tp.set_defaults(func=cmd_test_policy)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
