"""Validate the saved browser error-handling and recovery record."""

import json

from evaluate_error_recovery_ui import (
    FROZEN_FILES,
    OUTPUT,
    PARSE_PAPER_ID,
    VALID_PAPER_ID,
    ROOT,
    digest,
    implementation_hashes,
)


def main() -> None:
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["seed"] == 378
    assert result["requirement"] == (
        "file size, invalid PDF, parsing failure, and successful retry"
    )
    assert result["tests_run"] >= 39
    assert result["test_failures"] == result["test_errors"] == 0
    assert result["implementation_sha256"] == implementation_hashes(), (
        "Rerun evaluate_error_recovery_ui after implementation changes"
    )
    assert result["frozen_inputs_unchanged"] is True
    for name in FROZEN_FILES:
        assert result["frozen_input_sha256"][name] == digest(ROOT / name)

    required = {
        "browser_loaded",
        "oversized_message_visible",
        "oversized_upload_disabled",
        "invalid_pdf_message_visible",
        "invalid_upload_disabled",
        "parse_button_initially_ready",
        "parsing_failure_message_visible",
        "parsing_status_closed_as_error",
        "upload_control_ready_after_failure",
        "retry_button_ready_after_failure",
        "question_locked_after_failure",
        "no_spinner_after_failure",
        "successful_retry_completed",
        "question_enabled_after_retry",
    }
    assert required.issubset(result["checks"])
    assert all(result["checks"][name] is True for name in required)
    assert result["client_rejected_before_api"] == [
        "oversized.pdf",
        "invalid.pdf",
    ]
    assert result["request_log"]["uploads"] == [
        "parse-failure.pdf",
        "valid.pdf",
    ]
    assert result["request_log"]["analyses"] == [
        PARSE_PAPER_ID,
        VALID_PAPER_ID,
    ]
    assert result["request_log"]["papers"] == [VALID_PAPER_ID]
    assert result["temporary_files_removed"] is True
    assert result["quality_claim"] == "UI failure handling and recovery only"

    print("Error recovery browser gate passed")
    print("flow=oversized->invalid-pdf->parsing-failure->valid-retry")
    print(
        f"tests={result['tests_run']} client_rejections=2 "
        f"api_uploads={len(result['request_log']['uploads'])} seed={result['seed']}"
    )


if __name__ == "__main__":
    main()
