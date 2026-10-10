"""Exercise byte-exact evidence through real Git checkouts and ZIP export."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from evaluation_provenance import verify_frozen_source


class ArtifactPortabilityTests(unittest.TestCase):
    def test_saved_source_bytes_match_the_original_freeze(self):
        frozen = json.loads(
            (ROOT / "data/evaluation/final_evaluation.json").read_text(encoding="utf-8")
        )
        for relative, expected in frozen["provenance"]["source_hashes"].items():
            with self.subTest(path=relative):
                verified = verify_frozen_source(relative, expected)
                actual = hashlib.sha256(verified.read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    @unittest.skipUnless(shutil.which("git"), "Git is required for checkout integration")
    def test_windows_and_linux_checkout_and_zip_keep_evidence_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            origin = scratch / "origin"
            origin.mkdir()

            def git(cwd, *arguments):
                return subprocess.check_output(
                    ["git", "-c", f"safe.directory={cwd.as_posix()}", *arguments],
                    cwd=cwd, stderr=subprocess.PIPE,
                )

            git(origin, "init")
            (origin / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
            files = {
                "scripts/source.py": (ROOT / "scripts/grounded_generation.py").read_bytes(),
                "data/evaluation/questions.jsonl": (ROOT / "data/evaluation/questions.jsonl").read_bytes(),
                "data/evaluation/final_evaluation.json": (ROOT / "data/evaluation/final_evaluation.json").read_bytes(),
                "data/baselines/2026-09-14/scripts/grounded_generation.py": (
                    ROOT / "data/baselines/2026-09-14/scripts/grounded_generation.py"
                ).read_bytes(),
                "data/quality_benchmark/questions.jsonl": (
                    ROOT / "data/quality_benchmark/questions.jsonl"
                ).read_bytes(),
                "data/quality_benchmark/runs/2026-10-10/answerability-dev-calibration.json": (
                    ROOT / "data/quality_benchmark/runs/2026-10-10/answerability-dev-calibration.json"
                ).read_bytes(),
            }
            for relative, content in files.items():
                path = origin / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            git(origin, "-c", "core.autocrlf=true", "add", ".")
            git(origin, "-c", "user.name=Artifact test", "-c", "user.email=test@example.invalid",
                "commit", "-m", "Record evidence")
            for autocrlf in ("true", "false"):
                checkout = scratch / f"checkout-{autocrlf}"
                git(scratch, "clone", "--no-local", "--config", f"core.autocrlf={autocrlf}",
                    str(origin), str(checkout))
                for relative, expected in files.items():
                    with self.subTest(autocrlf=autocrlf, path=relative):
                        self.assertEqual((checkout / relative).read_bytes(), expected)
            archive = scratch / "source.zip"
            git(origin, "archive", "--format=zip", f"--output={archive}", "HEAD")
            with zipfile.ZipFile(archive) as exported:
                for relative, expected in files.items():
                    with self.subTest(export=relative):
                        self.assertEqual(exported.read(relative), expected)


if __name__ == "__main__":
    unittest.main()
