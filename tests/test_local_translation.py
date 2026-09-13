"""Dependency-light checks for the local NLLB translation configuration."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local_translation import load_translation_config, translation_config_fingerprint


class LocalTranslationTests(unittest.TestCase):
    def test_translation_model_is_revision_pinned_and_safe(self):
        config = load_translation_config()
        self.assertEqual(config["model"], "facebook/nllb-200-distilled-600M")
        self.assertEqual(config["source_language"], "eng_Latn")
        self.assertEqual(config["target_language"], "kor_Hang")
        self.assertEqual(len(config["revision"]), 40)
        self.assertTrue(config["use_safetensors"])
        self.assertFalse(config["trust_remote_code"])
        self.assertFalse(config["do_sample"])
        self.assertEqual(len(translation_config_fingerprint(config)), 64)


if __name__ == "__main__":
    unittest.main()
