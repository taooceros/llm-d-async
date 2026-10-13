"""Composite multi-phase manifests: expansion order, unique ids, and drift detection."""

import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from hetero import manifest as mf


def write_part(root, name, rows):
    requests = [{"request_id": f"{name}-{i}", "prompt_token_ids": ids, "prompt_tokens": len(ids),
                 "max_tokens": out} for i, (ids, out) in enumerate(rows)]
    body = json.dumps([[r["request_id"], r["prompt_token_ids"], r["max_tokens"]] for r in requests],
                      separators=(",", ":"))
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    mf.manifest_path(name, root).write_text(json.dumps({
        "workload": name, "model": "m", "catalogue": "c", "catalogue_sha256": "x",
        "requests": requests, "manifest_sha256": hashlib.sha256(body.encode()).hexdigest()}))


class Composite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        write_part(self.root, "A", [([1, 2], 3), ([4], 5)])
        write_part(self.root, "B", [([6, 7, 8], 9)])
        patcher = mock.patch.dict(mf.COMPOSITES, {"JOB": (("A", 2), ("B", 1))})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_phases_expand_in_order_with_unique_ids(self):
        mf.write_composite("JOB", self.root)
        job = mf.load_manifest("JOB", self.root)
        self.assertEqual([r["phase_workload"] for r in job["requests"]], ["A"] * 4 + ["B"])
        self.assertEqual(len({r["request_id"] for r in job["requests"]}), 5)
        self.assertEqual(job["useful_output_tokens"], 2 * (3 + 5) + 9)

    def test_changed_part_invalidates_the_composite(self):
        mf.write_composite("JOB", self.root)
        write_part(self.root, "B", [([6, 7, 8], 10)])
        with self.assertRaises(ValueError):
            mf.load_manifest("JOB", self.root)


if __name__ == "__main__":
    unittest.main()
