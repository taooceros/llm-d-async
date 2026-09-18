#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper script to pre-allocate Ray Placement Group for vLLM TPU inference.

This wrapper intercepts the CLI arguments to calculate the required TPU hosts,
creates a Ray Placement Group matching the required hosts, and then launches
the vLLM OpenAI API server inside a Ray Actor scheduled on that placement group.
"""

import os
import sys
import math

# Force TPU Ray executor configuration
os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "1"

# Disable Ray usage stats collection
if os.environ.get("RAY_USAGE_STATS_ENABLED", "0") != "1":
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import init_logger

logger = init_logger("vllm.tpu_wrapper")


@ray.remote
class ApiServerActor:
    """Ray Actor wrapping the vLLM OpenAI API Server.
    
    This actor runs inside the pre-created placement group, allowing vLLM
    to automatically discover and reuse the placement group.
    """
    def __init__(self, args):
        self.args = args

    def run(self) -> None:
        import asyncio
        import uvloop
        from vllm.entrypoints.openai.api_server import run_server
        
        # Install uvloop for faster network operations (standard in vLLM)
        uvloop.install()
        
        logger.info(f"Starting vLLM OpenAI API Server with args: {self.args}")
        try:
            asyncio.run(run_server(self.args))
        except Exception as e:
            logger.exception(f"API Server failed: {e}")
            raise


def main():
    # 1. Parse CLI Arguments using vLLM's standard parser
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible REST API Server (TPU Ray Wrapper)"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    
    # Run vLLM's standard CLI validations
    validate_parsed_serve_args(args)

    # 2. Determine JAX TP and PP parameters to calculate required hosts
    tp_size = args.tensor_parallel_size
    pp_size = args.pipeline_parallel_size
    
    # Each physical TPU host typically has 4 chips (e.g. v5e-8, v4-8, v6e)
    # Adjust this if running on configurations with different chip counts per host.
    chips_per_host = 4
    
    # Calculate hosts needed per stage, then multiply by pipeline stages
    hosts_per_stage = math.ceil(tp_size / chips_per_host)
    total_hosts_needed = hosts_per_stage * pp_size
    
    logger.info(
        f"TPU Ray Wrapper | Requested TP={tp_size}, PP={pp_size}. "
        f"Requiring {total_hosts_needed} TPU hosts (assuming {chips_per_host} chips/host)."
    )

    # 3. Initialize Ray
    if not ray.is_initialized():
        logger.info("Initializing Ray client...")
        ray.init(address="auto")

    # 4. Pre-create the placement group with explicit topology-aware node pinning
    from pkg_util.tpu_topology import TPUTopologyDiscovery
    discovery = TPUTopologyDiscovery()
    slices = discovery.discover_tpu_nodes()
    target_slice = list(slices.keys())[0]
    hosts = slices[target_slice]
    
    if total_hosts_needed < len(hosts) and total_hosts_needed == 2:
        pair = discovery.select_connected_2host_submesh(slice_name=target_slice)
        logger.info(f"Selected Connected 2-Host Subslice: Worker {pair.host_a.worker_id} ({pair.host_a.node_ip}) + Worker {pair.host_b.worker_id} ({pair.host_b.node_ip}) -> {pair.mesh_dimension}")
        bundle_specs = [
            {"TPU": float(chips_per_host), f"node:{pair.host_a.node_ip}": 1},
            {"TPU": float(chips_per_host), f"node:{pair.host_b.node_ip}": 1},
        ]
    else:
        # Use all available hosts or generic pack
        bundle_specs = [
            {"TPU": float(chips_per_host), f"node:{h.node_ip}": 1}
            for h in hosts[:total_hosts_needed]
        ]
    logger.info(f"Creating Ray Placement Group with specs: {bundle_specs}")
    pg = ray.util.placement_group(bundle_specs, strategy="STRICT_SPREAD")
    
    logger.info("Waiting for placement group to be ready...")
    ray.get(pg.ready(), timeout=1800)
    logger.info(f"Placement group is ready: {pg.id.hex()}")

    # 5. Spawn the API Server Actor on the first bundle (pins driver to Node 1)
    logger.info("Spawning API Server Actor on the placement group...")
    api_server = ApiServerActor.options(
        name="vllm_api_server_driver",
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0
        )
    ).remote(args)

    # 6. Keep wrapper running while server executes
    try:
        ray.get(api_server.run.remote())
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Tearing down...")
    finally:
        logger.info("Removing placement group...")
        ray.util.remove_placement_group(pg)
        logger.info("Teardown complete.")


if __name__ == "__main__":
    main()
