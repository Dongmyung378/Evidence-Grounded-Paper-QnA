"""Validate the saved Day 36 browser acceptance evidence."""

import json

from evaluate_day36 import (
    FROZEN_FILES,
    OUTPUT,
    ROOT,
    digest,
    implementation_hashes,
)


def main() -> None:
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["roadmap_day"] == 36
    assert result["seed"] == 378
    assert result["roadmap_gate"] == (
        "browser PDF upload -> analysis status -> overview -> question input"
    )
    assert result["tests_run"] >= 28
    assert result["test_failures"] == result["test_errors"] == 0
    assert result["transport"].startswith("headless Chromium against Streamlit")
    assert result["implementation_sha256"] == implementation_hashes(), (
        "Rerun evaluate_day36 after code changes"
    )
    assert result["frozen_inputs_unchanged"] is True
    for name in FROZEN_FILES:
        assert result["frozen_input_sha256"][name] == digest(ROOT / name)

    checks = result["checks"]
    required = {
        "streamlit_server_ready",
        "browser_loaded",
        "file_uploader_visible",
        "selected_filename_visible",
        "analysis_completed_visible",
        "overview_visible",
        "question_input_visible",
        "question_input_enabled",
        "one_pdf_registered",
        "analysis_job_completed",
        "overview_data_created",
        "question_runtime_prepared",
    }
    assert required.issubset(checks)
    assert all(checks[name] is True for name in required)
    assert len(checks["metric_values"]) == 4

    runtime = result["runtime"]
    assert runtime["papers"] == 1
    assert runtime["jobs"] == {"completed": 1}
    assert runtime["filename"] == "paper-003.pdf"
    assert runtime["seed"] == 378
    assert runtime["uploaded_page_count"] == runtime["analyzed_page_count"] > 0
    assert runtime["text_page_count"] > 0
    assert runtime["chunk_count"] > 0
    preparation = runtime["question_runtime_preparation"]
    assert preparation["status"] == "ready"
    assert preparation["embedding_device"] == "cpu"
    assert preparation["reranker_device"] == "cpu"
    assert preparation["generator"]["device"] in {"cpu", "cuda"}
    assert result["api_health_after"]["papers"] == 1
    assert result["api_health_after"]["jobs"]["completed"] == 1
    assert result["temporary_runtime_removed"] is True
    assert result["quality_claim"] == (
        "functional UI integration only; answer quality is unchanged"
    )

    print("Day 36 Streamlit browser gate passed")
    print("flow=browser upload->analysis status->overview->question input")
    print(
        f"tests={result['tests_run']} pages={runtime['analyzed_page_count']} "
        f"chunks={runtime['chunk_count']} seed={result['seed']}"
    )


if __name__ == "__main__":
    main()
