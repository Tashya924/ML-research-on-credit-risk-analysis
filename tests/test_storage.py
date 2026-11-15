"""Lossless compaction and recovery after interruption between rename and commit."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from creditrisk.artifacts import write_json
from creditrisk.compact import compact_job
from creditrisk.data import file_hash


class StorageTests(unittest.TestCase):
    def fixture(self, root):
        path = root / "model.joblib"
        original = np.arange(200000, dtype=np.float64)
        joblib.dump({"values": original}, path)
        write_json(
            root / "complete.json",
            dict(status="complete", artifacts=[dict(path=path.name, sha256=file_hash(path))]),
        )
        return path, original

    def check_contents(self, root, original):
        np.testing.assert_array_equal(joblib.load(root / "model.joblib")["values"], original)
        manifest = json.loads((root / "complete.json").read_text())
        for item in manifest["artifacts"]:
            self.assertEqual(file_hash(root / item["path"]), item["sha256"])
        self.assertFalse((root / ".repacking_pending.json").exists())

    def test_compaction_is_lossless_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = self.fixture(root)
            size = path.stat().st_size
            compact_job(root)
            self.check_contents(root, original)
            self.assertLess(path.stat().st_size, size)
            digest = file_hash(path)
            compact_job(root)
            self.assertEqual(file_hash(path), digest)

    def test_recovery_after_model_rename_before_manifest_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = self.fixture(root)
            temporary = root / "model.joblib.repacking"
            before_hash, before_bytes = file_hash(path), path.stat().st_size
            joblib.dump(joblib.load(path), temporary, compress=3)
            change = dict(
                path=path.name,
                temporary=temporary.name,
                before_sha256=before_hash,
                after_sha256=file_hash(temporary),
                before_bytes=before_bytes,
                after_bytes=temporary.stat().st_size,
            )
            write_json(
                root / ".repacking_pending.json",
                dict(previous_manifest_sha256=file_hash(root / "complete.json"), changes=[change]),
            )
            temporary.replace(path)
            compact_job(root)
            self.check_contents(root, original)
