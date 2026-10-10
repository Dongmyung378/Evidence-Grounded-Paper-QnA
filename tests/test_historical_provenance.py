"""Historical results stay verifiable while new code remains unmeasured."""

import hashlib
import sys
import unittest
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation_provenance import resolve_source


class HistoricalProvenanceTests(unittest.TestCase):
    def test_builder_cannot_rebind_historical_scores_to_updated_code(self):
        manifest = ROOT / "data/evaluation/final_evaluation.json"
        table = ROOT / "data/evaluation/final_performance.csv"
        before = (manifest.read_bytes(), table.read_bytes())
        result = subprocess.run(
            [sys.executable, "-B", "scripts/build_evaluation_freeze.py"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("historical evaluation freeze is preserved", result.stderr)
        self.assertEqual(before, (manifest.read_bytes(), table.read_bytes()))

    def test_modified_runtime_resolves_only_to_byte_exact_snapshot(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "scripts/runtime.py"
            source.parent.mkdir()
            source.write_bytes(b"new implementation\n")
            snapshot = root / "data/baselines/v1/scripts/runtime.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(b"measured implementation\n")
            expected = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            paths = {"scripts/runtime.py": snapshot.relative_to(root).as_posix()}
            self.assertEqual(resolve_source("scripts/runtime.py", expected, paths, root), snapshot)
            snapshot.write_bytes(b"tampered snapshot\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_source("scripts/runtime.py", expected, paths, root)

    def test_unrecorded_source_change_cannot_pass(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "artifact.json"
            source.write_bytes(b"modified artifact")
            expected = hashlib.sha256(b"original artifact").hexdigest()
            with self.assertRaisesRegex(ValueError, "without a historical snapshot"):
                resolve_source("artifact.json", expected, {}, root)

    def test_snapshot_path_cannot_escape_baseline_directory(self):
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "inside data/baselines"):
                resolve_source("runtime.py", "0" * 64, {"runtime.py": "outside.py"}, Path(temp))


if __name__ == "__main__":
    unittest.main()
