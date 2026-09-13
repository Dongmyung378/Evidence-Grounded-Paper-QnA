"""Dependency-light tests for the final evaluation freeze contract."""

from __future__ import annotations

import csv
import io
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_evaluation_freeze import build_outputs, source_paths


class EvaluationFreezeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "config" / "evaluation_freeze.json").read_text(encoding="utf-8")
        )

    def test_freeze_uses_current_revision_pinned_retrieval(self):
        self.assertEqual(self.config["seed"], 378)
        self.assertEqual(self.config["status"], "frozen")
        self.assertEqual(
            self.config["metric_policy"]["current_retrieval_source"],
            "data/evaluation/frozen_retrieval_metrics.json",
        )
        self.assertFalse(
            self.config["metric_policy"]["historical_mrr_is_current_claim"]
        )

    def test_every_frozen_source_exists_and_is_unique(self):
        paths = source_paths(self.config)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertGreaterEqual(len(paths), 35)

    def test_builder_is_deterministic_and_covers_required_areas(self):
        first_csv, first_json = build_outputs()
        second_csv, second_json = build_outputs()
        self.assertEqual(first_csv, second_csv)
        self.assertEqual(first_json, second_json)
        rows = list(csv.DictReader(io.StringIO(first_csv)))
        self.assertEqual(
            {row["area"] for row in rows},
            {"corpus", "retrieval", "answer_quality", "abstention", "container"},
        )
        self.assertTrue(all(first_json["checks"].values()))


if __name__ == "__main__":
    unittest.main()
