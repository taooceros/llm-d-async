"""
Ray Cluster & TPU Placement Group Management.
Enforces explicit node-pinned placement groups per AGENTS.md guidelines.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("rcm.cluster")


def init_ray_cluster(address: str = "auto", ignore_reinit: bool = True) -> None:
    """Initializes Ray connection if not already connected."""
    try:
        import ray
        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=ignore_reinit)
    except Exception as e:
        logger.warning(f"Could not initialize Ray client: {e}")


def cleanup_stale_placement_groups() -> None:
    """Removes all stale placement groups to prevent scheduling deadlocks."""
    try:
        import ray
        from ray.util.placement_group import placement_group_table, remove_placement_group
        table = placement_group_table()
        for pg_id, pg_info in table.items():
            if pg_info.get("state") in ("CREATED", "PENDING"):
                try:
                    remove_placement_group(pg_id)
                except Exception:
                    pass
    except Exception:
        pass


def create_node_pinned_placement_group(
    target_node_ips: List[str],
    tpus_per_node: float = 4.0,
    strategy: str = "STRICT_SPREAD",
    timeout_s: float = 60.0,
) -> Optional[Any]:
    """
    Creates an explicit node-pinned Ray placement group across specified TPU worker hosts.
    Mandatory per AGENTS.md Section 1 to prevent generic request hangs and ensure contiguous ICI mesh.
    """
    try:
        import ray
        from ray.util.placement_group import placement_group

        # Define explicit node-pinned bundles
        bundles = [
            {"TPU": tpus_per_node, f"node:{host_ip}": 0.001}
            for host_ip in target_node_ips
        ]

        logger.info(f"Creating node-pinned placement group across hosts: {target_node_ips}")
        pg = placement_group(bundles, strategy=strategy)
        ray.get(pg.ready(), timeout=timeout_s)
        logger.info("Placement group ready and locked.")
        return pg
    except Exception as e:
        logger.error(f"Failed to create node-pinned placement group: {e}")
        return None
