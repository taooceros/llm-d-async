"""Real conversational workload construction from the ShareGPT catalogues.

Prompts are real first-human turns; no random token sequences.  Prompt
selection is unique within a run so that disabling prefix caching actually
measures compute rather than cache hits.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, Iterator, List, Optional, Sequence

CATALOGUES = {
    "small": "data/workloads/sharegpt_sampled_500.json",
    "large": "data/workloads/sharegpt_large.json",
}


def catalogue_path(name: str, root: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Resolves a catalogue from the repo or from the cluster data staging dir.

    The 55 MB ShareGPT catalogue exceeds the Ray job-submission body limit, so
    on the cluster it is staged once at ``$OMP_HETERO_DATA_DIR`` instead of
    riding along in every job's ``working_dir``.
    """
    import os

    root = root or pathlib.Path(__file__).resolve().parents[1]
    if name not in CATALOGUES:
        raise KeyError(f"unknown catalogue {name}; known: {sorted(CATALOGUES)}")
    staged_root = os.environ.get("OMP_HETERO_DATA_DIR")
    if staged_root:
        staged = pathlib.Path(staged_root) / CATALOGUES[name]
        if staged.exists():
            return staged
    path = root / CATALOGUES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"ShareGPT catalogue {name} is not present at {path}; resolve real data before "
            "running a workload"
        )
    return path


def load_catalogue(name: str, root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    path = catalogue_path(name, root)
    raw = path.read_bytes()
    records = json.loads(raw)
    return {
        "name": name,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(records),
        "records": records,
    }


def unique_prompts(
    catalogue: Dict[str, Any],
    count: int,
    *,
    offset: int = 0,
    min_chars: int = 200,
) -> List[Dict[str, str]]:
    """Returns ``count`` distinct real prompts starting at ``offset``."""
    eligible = [r for r in catalogue["records"] if len(r["prompt"]) >= min_chars]
    if offset + count > len(eligible):
        raise ValueError(
            f"catalogue {catalogue['name']} has {len(eligible)} eligible prompts; "
            f"cannot take {count} unique prompts at offset {offset}"
        )
    return [dict(r) for r in eligible[offset : offset + count]]


def prompt_payload_digest(prompts: Sequence[Dict[str, str]]) -> str:
    material = json.dumps(
        [[p["id"], p["prompt"]] for p in prompts], separators=(",", ":"), sort_keys=False
    )
    return hashlib.sha256(material.encode()).hexdigest()
