"""An audited report must not quietly accept an edited result file."""

import json
import tempfile
import unittest
from pathlib import Path

from creditrisk.data import file_hash
from creditrisk.experiment import METHODS
from creditrisk.report import build


class EvidenceIntegrityTests(unittest.TestCase):
    def test_changed_metrics_are_rejected_before_report_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text(
                json.dumps({"status": "complete", "dataset": {}, "methods": list(METHODS)})
            )
            (root / "metrics.csv").write_text("original result bytes\n")
            digest = file_hash(root / "metrics.csv")
            (root / "evidence.json").write_text(
                json.dumps({"input_hashes": {"metrics.csv": digest}, "derived_hashes": {}})
            )
            (root / "metrics.csv").write_text("edited result bytes\n")
            with self.assertRaisesRegex(ValueError, "Evidence input or derivative changed"):
                build(root, root / "REPORT.md")
