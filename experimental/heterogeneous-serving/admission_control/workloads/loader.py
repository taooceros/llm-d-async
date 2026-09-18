"""
Workload Trace Loader & Workload Registry.
==========================================
Provides robust ingestion, validation, and profiling for production traces and synthetic benchmarks.
"""

import json
import os
import pathlib
from typing import List, Dict, Any, Optional

from admission_control.core.models import Request, TokenProfile

# Root directory of the repository
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def get_workload_path(filename_or_name: str) -> str:
    """Resolves standard workload filenames to absolute paths."""
    candidates = [
        pathlib.Path(filename_or_name),
        _REPO_ROOT / "data" / "workloads" / filename_or_name,
        _REPO_ROOT / "workload_retrieval" / filename_or_name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c.resolve())
    return filename_or_name


def load_workload_requests(
    workload_path_or_name: str,
    max_requests: Optional[int] = None,
) -> List[Request]:
    """
    Loads request definitions from a JSON file.
    Expects a list of dicts with 'prompt_len' and 'decode_len' keys.
    """
    resolved_path = get_workload_path(workload_path_or_name)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Workload file not found: {workload_path_or_name} (resolved to {resolved_path})")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Workload file must contain a JSON list, got: {type(data)}")

    requests = []
    for i, item in enumerate(data):
        if max_requests is not None and len(requests) >= max_requests:
            break
        p = int(item.get("prompt_len", item.get("input_tokens", item.get("prompt_tokens", item.get("input_len", 256)))))
        d = int(item.get("decode_len", item.get("output_tokens", item.get("max_tokens", item.get("output_len", item.get("generated_tokens", 64))))))
        raw_id = item.get("req_id", item.get("id", i))
        try:
            r_id = int(raw_id)
        except (ValueError, TypeError):
            r_id = i
        arr = int(item.get("arrival_time", item.get("timestamp", 0)))
        req = Request(
            req_id=r_id,
            prompt_len=p,
            decode_len=d,
            arrival_time=arr,
            metadata=item.get("metadata", {}),
        )
        requests.append(req)

    return requests


def save_workload_requests(requests: List[Request], file_path: str):
    """Saves a list of Request objects to JSON."""
    data = [r.to_dict() for r in requests]
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_workload_profile(workload_path_or_name: str) -> TokenProfile:
    """Computes aggregate token statistics for a workload trace."""
    reqs = load_workload_requests(workload_path_or_name)
    return TokenProfile.from_requests(reqs)
