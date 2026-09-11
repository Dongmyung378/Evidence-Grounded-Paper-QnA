"""Container packaging and environment configuration regression tests."""

import unittest
from pathlib import Path

from app.config import settings_from_environment


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentSettings(unittest.TestCase):
    def test_runtime_values_are_loaded_from_environment(self):
        settings = settings_from_environment(
            {
                "PAPER_QNA_DATA_DIR": "/tmp/paper-qna",
                "PAPER_QNA_SEED": "378",
                "PAPER_QNA_MAX_PENDING_JOBS": "7",
                "PAPER_QNA_MODEL_LOCAL_FILES_ONLY": "yes",
                "PAPER_QNA_PREPARE_MODELS": "off",
            }
        )
        self.assertEqual(settings.data_dir, Path("/tmp/paper-qna"))
        self.assertEqual(settings.seed, 378)
        self.assertEqual(settings.max_pending_jobs, 7)
        self.assertTrue(settings.model_local_files_only)
        self.assertFalse(settings.prepare_question_runtime)

    def test_invalid_runtime_values_fail_fast(self):
        invalid = [
            {"PAPER_QNA_SEED": "not-a-number"},
            {"PAPER_QNA_MAX_PENDING_JOBS": "0"},
            {"PAPER_QNA_PREPARE_MODELS": "sometimes"},
        ]
        for environment in invalid:
            with self.subTest(environment=environment):
                with self.assertRaises(ValueError):
                    settings_from_environment(environment)


class ContainerContract(unittest.TestCase):
    def test_backend_and_ui_are_separate_compose_services(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        self.assertIn("  backend:\n", compose)
        self.assertIn("  ui:\n", compose)
        self.assertIn("target: backend", compose)
        self.assertIn("target: ui", compose)
        self.assertIn("PAPER_QNA_API_URL: http://backend:8000", compose)
        self.assertIn("condition: service_healthy", compose)
        self.assertIn("paper-qna-runtime:/var/lib/paper-qna", compose)
        self.assertIn("paper-qna-model-cache:/var/cache/huggingface", compose)

    def test_images_use_non_root_user_and_split_dependencies(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        backend = (ROOT / "requirements-backend.txt").read_text(encoding="utf-8")
        ui = (ROOT / "requirements-ui.txt").read_text(encoding="utf-8")
        self.assertIn("python:3.11-slim@sha256:", dockerfile)
        self.assertGreaterEqual(dockerfile.count("USER paperqna"), 2)
        self.assertIn("sentence-transformers", backend)
        self.assertNotIn("streamlit", backend)
        self.assertIn("streamlit", ui)
        self.assertNotIn("torch", ui)
        self.assertNotIn("transformers", ui)

    def test_private_and_generated_files_are_outside_build_context(self):
        ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for entry in (".env", "local_notes/", "data/", "tests/"):
            self.assertIn(entry, ignored)
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("hf_", example.replace("hf_example", ""))


if __name__ == "__main__":
    unittest.main()
