"""Validate the saved browser evidence-presentation acceptance record."""

import json

from evaluate_evidence_ui import (
    FROZEN_FILES,
    OUTPUT,
    ROOT,
    digest,
    implementation_hashes,
)


def main() -> None:
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["seed"] == 378
    assert result["requirement"] == (
        "answer beside cited evidence text, page, and section"
    )
    assert result["tests_run"] >= 35
    assert result["test_failures"] == result["test_errors"] == 0
    assert result["implementation_sha256"] == implementation_hashes(), (
        "Rerun evaluate_evidence_ui after implementation changes"
    )
    assert result["frozen_inputs_unchanged"] is True
    for name in FROZEN_FILES:
        assert result["frozen_input_sha256"][name] == digest(ROOT / name)

    checks = result["checks"]
    required = {
        "browser_loaded",
        "question_submitted",
        "answer_visible",
        "evidence_panel_visible",
        "answer_and_evidence_side_by_side",
        "page_visible",
        "section_visible",
        "original_text_visible",
        "chunk_id_visible",
        "cited_evidence_count_visible",
        "no_insufficient_warning",
    }
    assert required.issubset(checks)
    assert all(checks[name] is True for name in required)
    assert checks["evidence_count"] >= 1
    assert result["runtime"]["jobs"] == {"completed": 1}
    assert result["api_health_after"]["question_engine"] == "ready"
    assert result["temporary_runtime_removed"] is True
    assert result["quality_claim"] == "evidence presentation and traceability only"

    print("Evidence UI browser gate passed")
    print("flow=upload->analysis->question->answer+source-evidence")
    print(
        f"tests={result['tests_run']} evidence={checks['evidence_count']} "
        f"side_by_side={checks['answer_and_evidence_side_by_side']} seed=378"
    )


if __name__ == "__main__":
    main()
