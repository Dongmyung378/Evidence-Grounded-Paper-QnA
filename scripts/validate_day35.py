"""Validate the saved Day 35 complete-flow acceptance evidence."""

import json

from evaluate_day35 import (
    EXPECTED_FLOW,
    OUTPUT,
    ROOT,
    SOURCE_PDF,
    digest,
    implementation_hashes,
)


def main():
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["roadmap_day"] == 35
    assert result["seed"] == 378
    assert result["roadmap_gate"] == "PDF -> parse -> index -> question -> answer"
    assert result["tests_run"] >= 24
    assert result["test_failures"] == result["test_errors"] == 0
    assert result["implementation_sha256"] == implementation_hashes(), (
        "Rerun evaluate_day35 after code changes"
    )
    assert result["frozen_inputs_unchanged"] is True
    for name, fingerprint in result["frozen_input_sha256"].items():
        assert digest(ROOT / name) == fingerprint, f"Frozen input changed: {name}"

    flow = result["complete_flow"]
    assert flow["flow_steps"] == EXPECTED_FLOW
    assert flow["source_sha256"] == digest(SOURCE_PDF)
    assert flow["transport"].startswith("curl over real TCP HTTP")
    assert flow["unknown_paper_question_status_code"] == 404
    assert flow["question_before_analysis_status_code"] == 409
    assert flow["upload_status_code"] == 201
    assert flow["analyze_status_code"] == 202
    assert flow["paper_result_status_code"] == 200
    assert flow["question_status_code"] == 200
    assert flow["job"]["status"] == "completed"
    assert flow["job"]["seed"] == 378
    assert flow["paper_result"]["status"] == "ready"
    assert flow["paper_result"]["overview"]["abstract"]
    assert flow["paper_result"]["overview"]["sections"]
    assert flow["health_before"]["question_engine"] == "lazy"
    assert flow["health_after"]["question_engine"] == "ready"

    trace = flow["traceability"]
    assert trace["pages"] == trace["unique_page_ids"] == 10
    assert trace["chunks"] == trace["unique_chunk_ids"] == 40
    assert trace["all_chunks_link_to_source_page"] is True
    assert trace["all_citations_link_to_runtime_chunk"] is True
    assert flow["runtime_embedding_cache_created"] is True

    answer = flow["question_result"]
    assert answer["question_language"] == "en"
    assert answer["sufficiency"] == "sufficient"
    assert answer["answer"].strip()
    assert answer["evidence"]
    assert all(item["page"] >= 1 and item["text"].strip() for item in answer["evidence"])
    assert result["temporary_runtime_removed"] is True
    assert result["quality_claim"] == "functional integration only"

    print("Day 35 complete API integration gate passed")
    print("flow=PDF->parse->index->question->answer")
    print(
        f"tests={result['tests_run']} pages={trace['pages']} "
        f"chunks={trace['chunks']} question_http={flow['question_status_code']}"
    )


if __name__ == "__main__":
    main()
