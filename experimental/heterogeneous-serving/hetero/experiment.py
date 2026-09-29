"""Whole-workload arm execution: the replication unit of the evaluation.

One call runs one arm once: reset, warm up outside the timed manifest, replay
the frozen manifest through the llm-d gateway, and stop the clock only when the
last request has been delivered.  Everything the gateway does -- scheduling,
reservations, quiescence, replay, commit and output collection -- is inside the
measured interval by construction, because the interval is the dispatch call.

Bulk per-iteration rows are written to the head node's artifact directory
instead of the structured stdout payload; only aggregates travel in the result.
"""

from __future__ import annotations

import asyncio
import csv
from collections import Counter
import hashlib
import gzip
import json
import os
import pathlib
import statistics
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import ray

from .evidence import emit_event, emit_log
from .manifest import load_manifest
from .native_backend import BACKEND_REVISION, RUNTIME_REVISION

ARTIFACT_ROOT = pathlib.Path(os.environ.get("OMP_HETERO_ARTIFACTS", "/tmp/omp_hetero/artifacts"))

# Frozen before any timing run.  Identical for every arm; only the migration
# fields differ between M0/MR/MK-* and they are declared per arm below.
BASE_POLICY: Dict[str, Any] = {
    "large_fresh_traffic_share": 0.5,
    "min_generated_tokens": 128,
    "min_context_tokens": 1024,
    "min_remaining_tokens": 256,
    "max_migrations_per_request": 1,
    "max_inflight_migrations_per_destination": 2,
    "max_inflight_migrations_total": 3,
    "source_pressure_free_block_fraction": 0.12,
    "reserve_large_headroom_blocks": 16,
}

ARMS: Dict[str, Dict[str, Any]] = {
    "H": {"layout": "H", "migration_enabled": False, "migration_mechanism": "recompute"},
    "S": {"layout": "S", "migration_enabled": False, "migration_mechanism": "recompute"},
    "M0": {"layout": "M", "migration_enabled": False, "migration_mechanism": "recompute"},
    "MR": {"layout": "M", "migration_enabled": True, "migration_mechanism": "recompute"},
    "MK-H": {"layout": "M", "migration_enabled": True, "migration_mechanism": "kv_host"},
}

# Admitted-population cap: the external feeder never has more than this many
# unfinished requests at the gateway.  Frozen across arms.
ADMITTED_CAP = 256
WARMUP_REQUESTS = 8
WARMUP_MAX_TOKENS = 32
INGRESS_URL = "http://omp-router.llm-d-optimized-baseline.svc.cluster.local"
INGRESS_PROTOCOL = "envoy-ext-proc-v3"


async def _dispatch_http(
    requests: List[Dict[str, Any]], *, concurrency: int, timeout_s: float
) -> List[Dict[str, Any]]:
    """Collect real client responses through Envoy, with no automatic retries."""
    import httpx

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=INGRESS_URL, limits=limits, timeout=timeout_s, trust_env=False
    ) as client:
        async def dispatch(item: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                started = time.time()
                try:
                    response = await client.post("/v1/completions", json=item)
                    response.raise_for_status()
                    protocol = response.headers.get("x-omp-gateway-protocol")
                    if protocol != INGRESS_PROTOCOL:
                        raise ValueError(f"unqualified ingress protocol: {protocol!r}")
                    body = response.json()
                    if body["id"] != item["request_id"]:
                        raise ValueError("response request identity mismatch")
                    tokens = body["choices"][0]["token_ids"]
                    if body["usage"]["completion_tokens"] != len(tokens):
                        raise ValueError("response usage disagrees with delivered tokens")
                    return {
                        "request_id": body["id"],
                        "ingress_protocol": protocol,
                        "output_token_ids": tokens,
                        "useful_output_tokens": len(tokens),
                        "prompt_tokens": body["usage"]["prompt_tokens"],
                        "segments": [
                            dict(segment, instance_id=segment["instance"])
                            for segment in body["omp"]["segments"]
                        ],
                        "initial_instance": body["omp"]["initial_instance"],
                        "client_started_at_s": started,
                        "client_finished_at_s": time.time(),
                        "response": body,
                    }
                except Exception as exc:
                    return {"request_id": item["request_id"],
                            "client_started_at_s": started, "client_finished_at_s": time.time(),
                            "error": f"{type(exc).__name__}: {exc}"}

        tasks = [asyncio.create_task(dispatch(item)) for item in requests]
        _, pending = await asyncio.wait(tasks, timeout=timeout_s)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return [
            {"request_id": item["request_id"], "client_finished_at_s": time.time(),
             "error": "whole-batch deadline exceeded before final client delivery"}
            if task in pending else task.result()
            for item, task in zip(requests, tasks)
        ]

def _instance_roles(layout_id: str, instances: Iterable[str]) -> Dict[str, List[str]]:
    names = sorted(instances)
    small = [n for n in names if n.startswith("small")]
    large = [n for n in names if n.startswith("large")]
    return {"small_instances": small, "large_instances": large}


def _frozen_cohort(protocol_path: Optional[str], arm: str) -> Dict[str, Any]:
    """Bind this timing run to one immutable amendment and its exact source.

    Refuses an arm the frozen artifact does not authorize, and refuses to run at
    all when the source actually loaded on this cluster differs from the frozen
    tree.  A missing protocol is a refusal, never an unrecorded timing run.
    """
    if not protocol_path:
        raise ValueError("timing runs require the frozen evaluation amendment (--protocol)")
    raw = pathlib.Path(protocol_path).read_bytes()
    protocol = json.loads(raw)
    body = {key: value for key, value in protocol.items() if key != "predeclaration_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != protocol.get("predeclaration_sha256"):
        raise ValueError("frozen amendment body does not match its recorded hash")
    freeze = protocol.get("source_freeze", {})
    if freeze.get("status") != "qualified":
        raise ValueError(f"source freeze is {freeze.get('status')!r}; no timing is authorized")
    authorization = freeze["timing_authorization"]
    if arm not in authorization["authorized_arms"]:
        raise ValueError(
            f"arm {arm} is not authorized by the frozen amendment: {authorization['unauthorized_reason']}"
        )
    root = pathlib.Path(__file__).resolve().parents[1]
    mismatched = sorted(
        name for name, expected in freeze["source_sha256"].items()
        if not (root / name).exists()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected
    )
    if mismatched:
        raise ValueError(f"loaded source differs from the frozen tree: {mismatched}")
    source_files = {str(path.relative_to(root)) for path in (root / "hetero").rglob("*")
                    if path.is_file() and path.suffix in (".py", ".json")}
    source_files.update(("scripts/hetero_ctl.py", "deploy/k8s/omp-gateway.yaml",
                         "deploy/k8s/omp-router.yaml"))
    if set(freeze["source_sha256"]) != source_files:
        raise ValueError("source freeze does not cover the complete serving package/deployment")
    tree_digest = hashlib.sha256(json.dumps(
        freeze["source_sha256"], sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if tree_digest != freeze.get("tree_sha256"):
        raise ValueError("source freeze tree digest does not match its files")

    package_digest = hashlib.sha256(json.dumps(
        {name.removeprefix("hetero/"): digest for name, digest in freeze["source_sha256"].items()
         if name.startswith("hetero/")}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if package_digest != RUNTIME_REVISION:
        raise ValueError("imported serving package differs from the frozen on-disk source")

    from .engine_actor import EngineSettings

    qualified = protocol.get("qualified_runtime")
    if not isinstance(qualified, dict):
        raise ValueError("timing requires qualified_runtime for this native FP32 package")
    for key, expected in (("backend_revision", BACKEND_REVISION),
                          ("runtime_revision", RUNTIME_REVISION)):
        if qualified.get(key) != expected:
            raise ValueError(f"qualified_runtime {key} differs from the imported package")
    settings = qualified.get("engine_settings")
    if (not isinstance(settings, dict)
            or EngineSettings.from_dict(settings).to_dict() != settings
            or protocol.get("engine_settings") != settings):
        raise ValueError("qualified_runtime requires the complete frozen EngineSettings")
    evidence = qualified.get("evidence", {})
    if set(evidence) != {"gate_H", "gate_S", "gate_M", "lifecycle",
                         "continuation", "recompute", "kv_host"}:
        raise ValueError("qualified_runtime is missing required qualification evidence")
    for gate, receipt in evidence.items():
        _verify_frozen_file(root, receipt, f"qualification {gate}")
    if (protocol.get("routing_and_migration_policy") != BASE_POLICY
            or protocol.get("arms") != ARMS
            or protocol.get("admitted_population_cap") != ADMITTED_CAP
            or protocol.get("warmup", {}).get("requests") != WARMUP_REQUESTS
            or protocol.get("warmup", {}).get("max_tokens") != WARMUP_MAX_TOKENS):
        raise ValueError("frozen routing, arm, admission or warmup semantics differ")
    manifests = protocol.get("manifests", [])
    expected_paths = {f"data/manifests/{name}.json"
                      for name in ("W0", "W1", "W2", "E_SHORT", "E_LONG")}
    if len(manifests) != len(expected_paths) or {m.get("path") for m in manifests} != expected_paths:
        raise ValueError("frozen amendment must bind every evaluation manifest")
    for manifest in manifests:
        _verify_frozen_file(root, manifest, "workload manifest")
        current = load_manifest(pathlib.Path(manifest["path"]).stem, root)
        if ({key: value for key, value in current.items() if key != "requests"}
                != {key: value for key, value in manifest.items()
                    if key not in ("path", "file_sha256")}):
            raise ValueError("frozen workload metadata differs from the loaded manifest")
    return {"predeclaration_sha256": protocol["predeclaration_sha256"],
            "protocol_path": str(protocol_path),
            "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
            "source_freeze_tree_sha256": freeze["tree_sha256"],
            "source_freeze_commit_id": freeze["commit_id"],
            "verified_source_files": len(freeze["source_sha256"]),
            "qualified_runtime": qualified, "frozen_manifests": manifests}


def _verify_frozen_file(root: pathlib.Path, receipt: Dict[str, Any], label: str) -> None:
    path = receipt.get("path")
    if not path or not isinstance(receipt.get("file_sha256"), str):
        raise ValueError(f"{label} requires a path and file_sha256")
    source = root / path
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != receipt["file_sha256"]:
        raise ValueError(f"{label} differs from its frozen file: {path}")


def _loaded_runtime(status: Dict[str, Any], qualified: Dict[str, Any]) -> Dict[str, Any]:
    """Verify actual actor bindings and cheap worker receipts, never model probes."""
    from .layout import BUILTIN_LAYOUTS, layout_digest
    from .registry import REGISTRY_NAMESPACE, find_registry

    layout_id = status["layout_id"]
    expected = dict(BUILTIN_LAYOUTS[layout_id])
    bindings = status.get("bindings", {})
    if (set(status["instances"]) != set(expected) or set(bindings) != set(expected)
            or status.get("maintenance") or status.get("capturing")
            or not status.get("discovery", {}).get("ready")):
        raise RuntimeError("gateway does not expose the complete idle static arm layout")
    registry = find_registry()
    if registry is None:
        raise RuntimeError("serving-instance registry actor is not present")
    layout, rows = ray.get([registry.get_layout.remote(layout_id),
                            registry.snapshot.remote(layout_id)], timeout=120)
    if (not layout or layout.get("layout_id") != layout_id
            or layout.get("layout_digest") != layout_digest(layout)
            or layout.get("total_chips_allocated") != sum(expected.values())
            or len(layout["instances"]) != len(expected)
            or {i["instance_id"]: i["tp_size"] for i in layout["instances"]} != expected
            or len(rows) != len(expected) or {r["instance_id"] for r in rows} != set(expected)):
        raise RuntimeError("registry differs from the declared static arm layout")
    instances = {i["instance_id"]: i for i in layout["instances"]}
    hosts = [host for i in instances.values() for host in i["host_ips"]]
    if len(hosts) != len(set(hosts)):
        raise RuntimeError("static arm instances overlap physical hosts")
    patches = json.loads(pathlib.Path(__file__).with_name("native_backend_patch.json").read_text())["files"]
    native_hashes = {p["name"]: p["patched_sha256"] for p in patches}
    settings = qualified["engine_settings"]
    revisions = {key: qualified[key] for key in ("backend_revision", "runtime_revision")}
    loaded = {}
    for row in rows:
        name = row["instance_id"]
        instance = instances[name]
        binding = bindings[name]
        binding_keys = ("instance_id", "layout_id", "generation", "config_digest", "ray_namespace",
                        "engine_actor_name", "endpoint_host", "endpoint_port", "model_build", "tp_size")
        if (any(binding.get(k) != row.get(k) for k in binding_keys)
                or row.get("layout_id") != layout_id
                or row.get("config_digest") != layout["layout_digest"]
                or row.get("ray_namespace") != REGISTRY_NAMESPACE
                or row.get("tp_size") != expected[name]
                or row.get("state") != "ready" or not row.get("routable")):
            raise RuntimeError(f"gateway/registry binding is not ready and identical: {name}")
        actor = ray.get_actor(row["engine_actor_name"], namespace=REGISTRY_NAMESPACE)
        identity, engine_status, workers, caches = ray.get(
            [actor.identity.remote(), actor.status.remote(), actor.worker_evidence.remote(),
             actor.worker_kv_evidence.remote()], timeout=120)
        wanted = {"instance_id": name, "layout_id": layout_id,
                  "layout_digest": layout["layout_digest"], "generation": row["generation"],
                  "node_ip": row["endpoint_host"], "tp_size": expected[name],
                  "host_ips": instance["host_ips"], "engine_settings": settings,
                  "backend_revision": revisions["backend_revision"],
                  "model_build": row["model_build"], "state": "ready"}
        if (any(identity.get(k) != value for k, value in wanted.items())
                or engine_status.get("generation") != row["generation"]
                or engine_status.get("state") != "ready"):
            raise RuntimeError(f"loaded engine identity/settings differ from qualification: {name}")
        try:
            build = json.loads(identity["model_build"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"loaded engine has no qualified model build: {name}") from exc
        parameter_dtype = str(build.get("parameter_dtype", "")).removeprefix("torch.")
        requested_dtype = {"half": "float16", "float": "float32"}.get(settings["dtype"], settings["dtype"])
        wanted_build = {"native_backend": revisions["backend_revision"],
                        "runtime_revision": revisions["runtime_revision"],
                        **{k: settings[k] for k in ("activation_dtype", "kv_cache_dtype", "matmul_precision")}}
        if (any(build.get(k) != value for k, value in wanted_build.items()) or not build.get("vllm")
                or parameter_dtype not in ("bfloat16", "float16", "float32")
                or (requested_dtype != "auto" and parameter_dtype != requested_dtype)):
            raise RuntimeError(f"loaded model build/precision differs from qualification: {name}")
        host_ips = instance["host_ips"]
        if (len(workers) != len(host_ips) or {w.get("node_ip") for w in workers} != set(host_ips)
                or len(caches) != len(host_ips) or {c.get("node_ip") for c in caches} != set(host_ips)):
            raise RuntimeError(f"worker receipts do not cover the complete instance: {name}")
        for worker in workers:
            native = worker.get("native_backend", {})
            files = native.get("files", [])
            precision = worker.get("precision", {})
            devices = worker.get("devices", {})
            rank = host_ips.index(worker["node_ip"])
            wanted_precision = {**revisions, "verified_before_cache_allocation": True,
                                "model_dtype": parameter_dtype,
                                "activation_dtype": settings["activation_dtype"],
                                "configured_cache_dtype": settings["kv_cache_dtype"],
                                "runner_cache_dtype": settings["kv_cache_dtype"],
                                "matmul_precision": settings["matmul_precision"]}
            if (worker.get("instance_id") != name or worker.get("instance_process_rank") != rank
                    or devices.get("global_device_count") != expected[name]
                    or devices.get("jax_process_count") != len(host_ips)
                    or devices.get("relative_position_match") is not True
                    or any(native.get(k) != value for k, value in revisions.items())
                    or native.get("ready") is not True or len(files) != len(native_hashes)
                    or {f.get("name"): f.get("actual_sha256") for f in files} != native_hashes
                    or any(f.get("state") != "patched" for f in files)
                    or any(precision.get(k) != value for k, value in wanted_precision.items())):
                raise RuntimeError(f"worker loaded backend/precision differs from qualification: {name}")
            for key, dtype in (("parameter_dtypes", parameter_dtype),
                               ("decoder_norm_dtypes", settings["activation_dtype"]),
                               ("attention_cache_dtypes", settings["kv_cache_dtype"])):
                counts = precision.get(key, {})
                if set(counts) != {dtype} or not isinstance(counts[dtype], int) or counts[dtype] <= 0:
                    raise RuntimeError(f"worker actual {key} differs from qualification: {name}")
        for cache in caches:
            if (cache.get("backend_revision") != revisions["backend_revision"]
                    or cache.get("configured_cache_dtype") != settings["kv_cache_dtype"]
                    or str(cache.get("runner_cache_dtype")).removeprefix("torch.") != settings["kv_cache_dtype"]
                    or cache.get("cache_dtypes") != [settings["kv_cache_dtype"]]
                    or cache.get("num_cache_arrays", 0) <= 0 or cache.get("total_cache_bytes", 0) <= 0):
                raise RuntimeError(f"worker allocated cache differs from qualification: {name}")
        loaded[name] = {"identity": identity, "worker_evidence": workers, "worker_kv_evidence": caches}
    return {"layout": layout, "instances": loaded}



def run_arm(
    *,
    arm: str,
    workload: str,
    repetition: int,
    run_id: str,
    timeout_s: float = 7200.0,
    policy_overrides: Optional[Dict[str, Any]] = None,
    protocol_path: Optional[str] = None,
    pilot_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Timed arm. pilot_policy selects an UNFROZEN pilot: no cohort binding, an
    explicit policy amendment over BASE_POLICY, and output never eligible as
    cohort or headline evidence."""
    from .gateway.lifecycle import find_gateway

    pilot = pilot_policy is not None
    if pilot:
        unknown = set(pilot_policy) - set(BASE_POLICY)
        if unknown:
            raise ValueError(f"pilot policy may only amend BASE_POLICY fields: {sorted(unknown)}")
        cohort = {"unfrozen_pilot": True, "pilot_policy": dict(pilot_policy),
                  "evidence_scope": "UNFROZEN pilot; never cohort or headline evidence"}
    else:
        cohort = _frozen_cohort(protocol_path, arm)

    def runtime(observed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None if pilot else _loaded_runtime(observed, cohort["qualified_runtime"])

    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm}; known: {sorted(ARMS)}")
    spec = ARMS[arm]
    gateway = find_gateway()
    if gateway is None:
        raise RuntimeError("gateway actor is not running; start it with `gateway-up`")

    ray.get(gateway.reconcile.remote(bind_labels=False), timeout=300)
    status = ray.get(gateway.status.remote(), timeout=120)
    deployed_layout = status["layout_id"]
    if deployed_layout != spec["layout"]:
        raise RuntimeError(
            f"arm {arm} requires layout {spec['layout']} but layout {deployed_layout} is deployed"
        )
    instances = sorted(status["instances"])
    roles = _instance_roles(deployed_layout, instances)

    policy = dict(BASE_POLICY)
    policy.update(pilot_policy or {})
    policy.update(roles)
    policy["migration_enabled"] = spec["migration_enabled"]
    policy["migration_mechanism"] = spec["migration_mechanism"]
    if not roles["large_instances"] or not roles["small_instances"]:
        # Homogeneous layouts have no small/large split; fresh traffic is then
        # balanced purely by the capacity-aware scorer.
        policy["large_fresh_traffic_share"] = 0.0
        policy["migration_enabled"] = False
    from .gateway.policy import RoutingPolicy

    declared_policy = RoutingPolicy.from_dict(policy).to_dict()
    candidate_policy = RoutingPolicy.from_dict({**policy, **(policy_overrides or {})}).to_dict()
    if candidate_policy != declared_policy:
        raise ValueError("policy_overrides cannot change the frozen BASE_POLICY/arm semantics")
    manifest = load_manifest(workload)
    if not pilot:
        frozen_manifest = next((m for m in cohort["frozen_manifests"]
                                if m["path"] == f"data/manifests/{workload}.json"), None)
        if (frozen_manifest is None
                or manifest["manifest_sha256"] != frozen_manifest["manifest_sha256"]
                or manifest["model"] != cohort["qualified_runtime"]["engine_settings"]["model"]):
            raise ValueError("selected workload is not part of the qualified frozen configuration")
    loaded_runtime = runtime(status)
    applied = ray.get(gateway.set_policy.remote(declared_policy), timeout=60)
    if RoutingPolicy.from_dict(applied).to_dict() != declared_policy:
        raise RuntimeError("gateway applied policy differs from the frozen arm")
    requests = manifest["requests"]
    request_namespace = uuid.uuid4().hex

    # Warmup conversations are disjoint from every frozen timed workload.
    from transformers import AutoTokenizer
    from .workload import load_catalogue

    catalogue = load_catalogue("large")
    if catalogue["sha256"] != manifest["catalogue_sha256"] and not pilot:
        raise ValueError("warmup catalogue differs from frozen workload provenance")
    excluded = {row["source_id"] for name in ("W0", "W1", "W2", "E_SHORT", "E_LONG")
                for row in load_manifest(name)["requests"]}
    warmup_prompts = []
    for row in catalogue["records"][27000:]:
        if row["id"] not in excluded:
            warmup_prompts.append(row)
            excluded.add(row["id"])
        if len(warmup_prompts) == WARMUP_REQUESTS:
            break
    if len(warmup_prompts) != WARMUP_REQUESTS:
        raise ValueError("insufficient disjoint real warmup conversations")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"])
    warmup_ids = tokenizer([row["prompt"] for row in warmup_prompts], add_special_tokens=True)["input_ids"]
    warmup_payload = [
        {"request_id": f"warm-{request_namespace}-{index:03d}",
         "prompt_token_ids": token_ids[:256],
         "max_tokens": WARMUP_MAX_TOKENS, "ignore_eos": True}
        for index, token_ids in enumerate(warmup_ids)
    ]

    emit_event("ARM_RESET", arm=arm, workload=workload, repetition=repetition)
    ray.get(gateway.reset_records.remote(), timeout=120)
    ray.get(gateway.resume_admission.remote(), timeout=120)
    idle_before = ray.get(gateway.await_idle_instances.remote(300.0), timeout=360)

    warmup_status = ray.get(gateway.status.remote(), timeout=120)
    if (runtime(warmup_status) != loaded_runtime
            or RoutingPolicy.from_dict(warmup_status["core"]["config"]["policy"]).to_dict() != declared_policy):
        raise RuntimeError("qualified runtime or frozen policy changed before warmup")
    emit_event("ARM_WARMUP", arm=arm, requests=len(warmup_payload))
    warmup = asyncio.run(_dispatch_http(
        warmup_payload, concurrency=len(warmup_payload), timeout_s=1800.0
    ))
    warmup_errors = [w for w in warmup if w.get("error")]
    if warmup_errors:
        # A refused warmup means the fabric is not actually accepting work; a
        # timed arm launched now would measure an empty cluster.
        raise RuntimeError(
            f"arm {arm} warmup failed on {len(warmup_errors)}/{len(warmup)} requests: "
            f"{warmup_errors[0]['error']}"
        )
    # Reserve the artifact identity before touching the fabric: a name clash must
    # never arm capture, consume the warmup, or abandon an armed capture.
    artifact_dir = ARTIFACT_ROOT / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    label = f"{workload}_{arm}_r{repetition}"
    summary_path = artifact_dir / f"{label}_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite retained evidence for {label}: {summary_path}")

    ray.get(gateway.await_idle_instances.remote(300.0), timeout=360)
    ray.get(gateway.reset_records.remote(), timeout=120)
    capture_status = ray.get(gateway.status.remote(), timeout=120)
    if (runtime(capture_status) != loaded_runtime
            or RoutingPolicy.from_dict(capture_status["core"]["config"]["policy"]).to_dict() != declared_policy):
        raise RuntimeError("qualified runtime or frozen policy changed before capture")
    ray.get(gateway.start_capture.remote(), timeout=300)

    payload = [
        {
            "request_id": f"{request_namespace}:{row['request_id']}",
            "prompt_token_ids": row["prompt_token_ids"],
            "max_tokens": row["max_tokens"],
            "ignore_eos": True,
        }
        for row in requests
    ]

    emit_event(
        "ARM_BEGIN",
        arm=arm,
        workload=workload,
        repetition=repetition,
        requests=len(payload),
        useful_output_tokens=manifest["useful_output_tokens"],
        admitted_cap=ADMITTED_CAP,
    )
    started_wall = time.time()
    started_mono = time.perf_counter()
    try:
        records = asyncio.run(_dispatch_http(
            payload, concurrency=ADMITTED_CAP, timeout_s=timeout_s
        ))
    except BaseException:
        # Never leave the fabric armed for a workload that will not be recorded.
        ray.get(gateway.stop_capture.remote(), timeout=600)
        raise
    makespan_s = time.perf_counter() - started_mono
    finished_wall = time.time()
    for row in records:
        row["canonical_request_id"] = row["request_id"].removeprefix(f"{request_namespace}:")
    emit_event("ARM_END", arm=arm, makespan_s=round(makespan_s, 3))

    _write_gz(artifact_dir / f"{label}_records.json.gz", records)
    iterations = ray.get(gateway.stop_capture.remote(), timeout=600)
    final_status = ray.get(gateway.status.remote(), timeout=120)
    traces = ray.get(gateway.traces.remote(200000), timeout=300)

    authoritative = ray.get(gateway.records.remote(), timeout=300)
    _write_gz(artifact_dir / f"{label}_gateway_records.json.gz", authoritative)
    from .qualify_migration import _check_accounting
    accounting = _check_accounting(
        authoritative, {f"{request_namespace}:{r['request_id']}": r["max_tokens"] for r in requests}
    )
    by_id = {r["request_id"]: r for r in authoritative}
    delivery_matches = (
        len(authoritative) == len(by_id) == len(records) == len(requests)
        and all(not r.get("error") and r["output_token_ids"] ==
                by_id.get(r["request_id"], {}).get("output_token_ids") for r in records)
    )
    _write_gz(artifact_dir / f"{label}_accounting.json.gz", accounting)
    progress = _progress_summary(records, iterations, started_wall, finished_wall)
    _write_gz(artifact_dir / f"{label}_progress.json.gz", progress)
    iteration_stats = _write_iterations(artifact_dir, f"{workload}_{arm}", repetition, iterations)
    _write_gz(artifact_dir / f"{label}_traces.json.gz", traces)
    _write_gz(
        artifact_dir / f"{label}_transactions.json.gz",
        final_status["core"]["state"]["transactions"],
    )

    summary = _summarise(
        arm=arm,
        workload=workload,
        repetition=repetition,
        manifest=manifest,
        records=records,
        makespan_s=makespan_s,
        status=final_status,
        iteration_stats=iteration_stats,
        instances=instances,
        request_id_namespace=request_namespace,
    )
    summary["client_delivery_matches"] = delivery_matches
    summary["committed_prefix_preserved"] = accounting["no_duplicate_delivery"]
    summary["progress"] = {key: value for key, value in progress.items() if key != "native_emissions"}
    summary["native_output_budget_exact"] = progress["native_output_tokens"] == summary["useful_output_tokens"]
    summary["all_requests_complete"] = bool(
        summary["all_requests_complete"] and delivery_matches
        and accounting["no_duplicate_delivery"] and accounting["all_budgets_exact"]
    )
    summary.update(
        {
            **cohort,
            "request_id_namespace": request_namespace,
            "run_id": run_id,
            "ingress_url": INGRESS_URL,
            "request_path": "Envoy HTTP /v1/completions -> ExtProc gateway authority -> engine",
            "ingress_protocol": INGRESS_PROTOCOL if all(
                r.get("ingress_protocol") == INGRESS_PROTOCOL for r in records
            ) else None,
            "warmup": {"source_ids": [row["id"] for row in warmup_prompts],
                       "prompt_token_ids": [row["prompt_token_ids"] for row in warmup_payload],
                       "max_tokens": WARMUP_MAX_TOKENS,
                       "catalogue_sha256": catalogue["sha256"]},
            "policy": applied,
            "layout_id": deployed_layout,
            "artifact_dir": str(artifact_dir),
            "started_wall": started_wall,
            "finished_wall": finished_wall,
            "idle_before": idle_before,
            "loaded_runtime": loaded_runtime,
        }
    )
    emit_event(
        "ARM_RESULT",
        arm=arm,
        workload=workload,
        repetition=repetition,
        makespan_s=round(makespan_s, 3),
        useful_tokens=summary["useful_output_tokens"],
        throughput=summary["useful_tokens_per_s"],
        migrations=summary["migrations_committed"],
        complete=summary["all_requests_complete"],
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if (not summary["all_requests_complete"] or not summary["iteration_capture_one_to_one"]
            or not summary["native_output_budget_exact"]):
        raise RuntimeError(
            f"arm {arm} rep {repetition} failed complete exact-token delivery/accounting; "
            f"evidence retained in {artifact_dir}"
        )
    return summary


def _write_gz(path: pathlib.Path, payload: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


ITERATION_FIELDS = [
    "instance_id",
    "iteration",
    "scheduled_at_s",
    "completed_at_s",
    "scheduled_to_output_s",
    "phase",
    "prefill_tokens",
    "decode_tokens",
    "prefill_requests",
    "decode_requests",
    "replayed_tokens",
    "native_preemption_replayed_tokens",
    "schedule_duration_s",
    "host_output_tokens_per_s",
    "scheduled_tokens",
    "scheduled_requests",
    "new_requests",
    "running_requests",
    "waiting_requests",
    "fenced_requests",
    "free_kv_blocks_after_schedule",
    "context_tokens",
    "sampled_tokens",
    "emitted_output_tokens",
    "num_preempted",
    "num_finished",
]

def _progress_summary(
    records: List[Dict[str, Any]], captures: Dict[str, Dict[str, Any]],
    started_wall: float, finished_wall: float,
) -> Dict[str, Any]:
    """Observed native progress in fixed client-completion backlog boundaries."""
    events = sorted(
        (float(row["completed_at_s"]), instance_id, int(row["emitted_output_tokens"]))
        for instance_id, capture in captures.items() for row in capture["rows"]
        if row.get("completed_at_s") is not None and row.get("emitted_output_tokens", 0) > 0
    )
    result = {"native_emissions": events,
              "native_output_tokens": sum(tokens for _, _, tokens in events),
              "admitted_cap": ADMITTED_CAP,
              "boundary_definition": "start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm",
              "time_basis": "host Unix timestamps, not device kernel clocks"}
    if not records or any(row.get("error") for row in records):
        return dict(result, available=False, reason="incomplete_client_delivery")
    completions = sorted(float(row["client_finished_at_s"]) for row in records)
    n = len(completions)
    begin_count = (n + 9) // 10
    end_count = n - ADMITTED_CAP
    drain_start = completions[end_count - 1] if end_count > 0 else started_wall
    result["terminal_drain_s"] = max(0.0, finished_wall - drain_start)
    result["terminal_drain_fraction"] = result["terminal_drain_s"] / (finished_wall - started_wall)
    result["backlog_boundaries_completed_requests"] = [begin_count, end_count]
    if end_count <= begin_count:
        return dict(result, available=False, reason="no_common_deep_backlog_interval")
    begin, end = completions[begin_count - 1], completions[end_count - 1]
    if end <= begin:
        return dict(result, available=False, reason="nonpositive_observed_interval")
    per_instance = {instance_id: 0 for instance_id in captures}
    for completed_at, instance_id, tokens in events:
        if begin < completed_at <= end:
            per_instance[instance_id] += tokens
    return dict(result, available=True, deep_backlog_s=end - begin,
                deep_backlog_begin_s=begin - started_wall, deep_backlog_end_s=end - started_wall,
                deep_backlog_native_output_tokens=sum(per_instance.values()),
                deep_backlog_native_tokens_per_s=sum(per_instance.values()) / (end - begin),
                deep_backlog_instance_tokens=per_instance)


def _write_iterations(
    artifact_dir: pathlib.Path,
    arm: str,
    repetition: int,
    iterations: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    path = artifact_dir / f"{arm}_r{repetition}_iterations.csv.gz"
    stats: Dict[str, Any] = {}
    total_rows = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITERATION_FIELDS)
        writer.writeheader()
        for instance_id, payload in sorted(iterations.items()):
            rows = payload["rows"]
            audit = payload["audit"]
            preemptions = 0
            prefill_rows = 0
            decode_rows = 0
            control_rows = 0
            emitted = 0
            latencies: List[float] = []
            batch_sizes: List[int] = []
            for row in rows:
                preempted = len(row.get("preempted_request_ids") or [])
                preemptions += preempted
                phase = row["phase"]
                if phase in ("prefill", "mixed"):
                    prefill_rows += 1
                elif phase == "decode":
                    decode_rows += 1
                else:
                    control_rows += 1
                emitted += int(row.get("emitted_output_tokens") or 0)
                latency = row.get("scheduled_to_output_s")
                if phase != "control":
                    if latency is not None:
                        latencies.append(float(latency))
                    batch_sizes.append(int(row["scheduled_requests"]))
                csv_row = {field: row.get(field) for field in ITERATION_FIELDS}
                csv_row.update(
                    instance_id=instance_id,
                    num_preempted=preempted,
                    num_finished=len(row.get("finished_request_ids") or []),
                    host_output_tokens_per_s=(
                        int(row.get("emitted_output_tokens") or 0) / latency
                        if latency and phase != "control" else None
                    ),
                )
                writer.writerow(csv_row)
                total_rows += 1
            stats[instance_id] = {
                "capture_audit": audit,
                "iterations": len(rows),
                "mixed_or_prefill_iterations": prefill_rows,
                "decode_only_iterations": decode_rows,
                "control_iterations": control_rows,
                "preemptions": preemptions,
                "emitted_output_tokens": emitted,
                "iteration_latency_p50_s": round(statistics.median(latencies), 6)
                if latencies
                else None,
                "iteration_latency_mean_s": round(statistics.fmean(latencies), 6)
                if latencies
                else None,
                "scheduled_requests_p50": statistics.median(batch_sizes)
                if batch_sizes
                else None,
                "scheduled_requests_max": max(batch_sizes) if batch_sizes else None,
            }
    unmatched = sum(
        int(v["capture_audit"]["unmatched_retirements"]) for v in stats.values()
    )
    unretired = sum(
        int(v["capture_audit"]["rows"]) - int(v["capture_audit"]["retired_rows"])
        for v in stats.values()
    )
    return {
        "path": str(path),
        "rows": total_rows,
        "per_instance": stats,
        "unmatched_retirements": unmatched,
        "rows_without_retirement": unretired,
        "one_to_one": unmatched == 0 and unretired == 0,
    }


def _summarise(
    *,
    arm: str,
    workload: str,
    repetition: int,
    manifest: Dict[str, Any],
    records: List[Dict[str, Any]],
    makespan_s: float,
    status: Dict[str, Any],
    iteration_stats: Dict[str, Any],
    instances: List[str],
    request_id_namespace: str,
) -> Dict[str, Any]:
    expected = {f"{request_id_namespace}:{r['request_id']}": r["max_tokens"]
                for r in manifest["requests"]}
    failed = [r for r in records if r.get("error")]
    counts = Counter(r["request_id"] for r in records)
    duplicates = sorted(rid for rid, count in counts.items() if count != 1)
    missing = sorted(set(expected) - set(counts))
    unexpected = sorted(set(counts) - set(expected))
    delivered = {
        r["request_id"]: len(r.get("output_token_ids") or [])
        for r in records
        if not r.get("error")
    }
    short = {
        rid: (delivered[rid], expected[rid])
        for rid in delivered.keys() & expected.keys()
        if delivered[rid] != expected[rid]
    }
    useful = sum(delivered.values())
    transactions = status["core"]["state"]["transactions"]
    committed = [t for t in transactions if t["state"] in ("committed", "released")]
    declined = [t for t in transactions if t["state"] == "aborted"]
    per_instance_first: Dict[str, int] = {}
    for record in records:
        if record.get("error"):
            continue
        first = record["segments"][0] if record.get("segments") else None
        if first is None:
            continue
        name = first.get("segment_instance") or first.get("instance_id")
        per_instance_first[name] = per_instance_first.get(name, 0) + 1
    replayed = sum(int(t.get("replay_tokens") or 0) for t in committed)
    handoffs = [
        (t["committed_at"] - t["created_at"])
        for t in committed
        if t.get("committed_at")
    ]
    chips = status.get("total_chips") or 32
    return {
        "arm": arm,
        "workload": workload,
        "repetition": repetition,
        "manifest_sha256": manifest["manifest_sha256"],
        "num_requests": len(manifest["requests"]),
        "expected_output_tokens": manifest["useful_output_tokens"],
        "useful_output_tokens": useful,
        "makespan_s": round(makespan_s, 4),
        "useful_tokens_per_s": round(useful / makespan_s, 3) if makespan_s else None,
        "useful_tokens_per_s_per_chip": round(useful / makespan_s / chips, 4)
        if makespan_s
        else None,
        "all_requests_complete": (
            not failed and not short and not missing and not unexpected and not duplicates
            and useful == manifest["useful_output_tokens"]
        ),
        "missing_requests": missing,
        "unexpected_requests": unexpected,
        "duplicate_requests": duplicates,
        "failed_requests": [
            {"request_id": r.get("request_id"), "error": r["error"]} for r in failed
        ],
        "short_requests": short,
        "replayed_tokens": replayed,
        "transferred_bytes": sum(int(t.get("transferred_bytes", 0)) for t in committed),
        "replayed_token_fraction": round(replayed / useful, 6) if useful else None,
        "migrations_committed": len(committed),
        "migrations_declined": len(declined),
        "migration_handoff_s_p50": round(statistics.median(handoffs), 4)
        if handoffs
        else None,
        "migration_handoff_s_max": round(max(handoffs), 4) if handoffs else None,
        "counters": status["core"]["state"]["counters"],
        "initial_placement_counts": per_instance_first,
        "instances": instances,
        "iteration_capture_one_to_one": iteration_stats["one_to_one"],
        "total_preemptions": sum(
            v["preemptions"] for v in iteration_stats["per_instance"].values()
        ),
        "iteration_stats": iteration_stats,
    }


def run_sequence(
    *,
    arms: List[str],
    workload: str,
    repetitions: int,
    run_id: str,
    timeout_s: float = 7200.0,
    protocol_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs several arms for several repetitions inside one job.

    Arm order is reversed on even repetitions so that any drift in cluster
    state is balanced across the compared arms (AB / BA), which is what makes
    the paired differences interpretable.
    """
    results: List[Dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        order = list(arms) if repetition % 2 == 1 else list(reversed(arms))
        emit_event("SEQUENCE_REPETITION", repetition=repetition, order=order)
        for arm in order:
            results.append(
                run_arm(
                    arm=arm,
                    workload=workload,
                    repetition=repetition,
                    run_id=run_id,
                    timeout_s=timeout_s,
                    protocol_path=protocol_path,
                )
            )
    return {
        "run_id": run_id,
        "workload": workload,
        "arms": arms,
        "repetitions": repetitions,
        "results": results,
        "artifact_dir": str(ARTIFACT_ROOT / run_id),
        "paired": _paired_effects(results),
    }


def _paired_effects(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Use the same qualification and pairing rules as the final report."""
    from .analysis import paired_effect

    arms = sorted({row["arm"] for row in results})
    effects = {}
    for index, baseline in enumerate(arms):
        for treatment in arms[index + 1:]:
            effect = paired_effect(results, baseline, treatment)
            if effect is not None:
                effects[f"{treatment}_vs_{baseline}"] = effect
    return effects
