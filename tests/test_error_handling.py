"""Upload validation and localized recoverable-error tests."""

import unittest

from app.service import Settings
from ui.error_messages import (
    ERROR_MESSAGES,
    MAX_UPLOAD_BYTES,
    localized_error_message,
    validate_upload,
)


class UploadPreflight(unittest.TestCase):
    def test_ui_and_api_use_the_same_product_limit(self):
        self.assertEqual(MAX_UPLOAD_BYTES, Settings().max_upload_bytes)
        self.assertIsNone(
            validate_upload(
                "paper.pdf",
                "application/pdf",
                MAX_UPLOAD_BYTES,
                b"%PDF-",
            )
        )
        self.assertEqual(
            validate_upload(
                "paper.pdf",
                "application/pdf",
                MAX_UPLOAD_BYTES + 1,
                b"%PDF-",
            ),
            "file_too_large",
        )

    def test_invalid_uploads_are_rejected_before_the_api_call(self):
        cases = [
            ("empty.pdf", "application/pdf", 0, b"", "empty_file"),
            ("paper.txt", "text/plain", 10, b"%PDF-", "invalid_file_type"),
            ("paper.pdf", "text/plain", 10, b"%PDF-", "invalid_file_type"),
            ("paper.pdf", "application/pdf", 10, b"wrong", "invalid_pdf"),
        ]
        for filename, content_type, size, header, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    validate_upload(filename, content_type, size, header),
                    expected,
                )


class LocalizedErrors(unittest.TestCase):
    def test_required_error_paths_have_english_and_korean_messages(self):
        required = {
            "file_too_large",
            "invalid_file_type",
            "invalid_pdf",
            "analysis_failed",
            "partial_extraction_failed",
            "analysis_timeout",
        }
        for language in ("en", "ko"):
            self.assertTrue(required.issubset(ERROR_MESSAGES[language]))
            self.assertTrue(
                all(ERROR_MESSAGES[language][code].strip() for code in required)
            )

    def test_unknown_backend_error_uses_a_safe_generic_message(self):
        for language in ("en", "ko"):
            self.assertEqual(
                localized_error_message(language, "private_backend_exception"),
                ERROR_MESSAGES[language]["api_error"],
            )


if __name__ == "__main__":
    unittest.main()
