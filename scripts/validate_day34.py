"""Validate the saved Day 34 API acceptance evidence against current code."""

import json

from evaluate_day34 import (
    OUTPUT,
    ROOT,
    SOURCE_PDF,
    digest,
)


def main():
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["roadmap_day"] == 34
    assert result["seed"] == 378
    assert result["tests_run"] >= 24
    assert result["test_failures"] == result["test_errors"] == 0
    assert result["implementation_sha256"]
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in result["implementation_sha256"].values()
    ), "Historical implementation hashes are malformed"
    assert result["frozen_inputs_unchanged"] is True
    for name, fingerprint in result["frozen_input_sha256"].items():
        if name.startswith("data/"):
            assert digest(ROOT / name) == fingerprint, f"Frozen data changed: {name}"

    smoke = result["http_smoke"]
    assert smoke["transport"].startswith("curl over real TCP HTTP")
    assert smoke["source_sha256"] == digest(SOURCE_PDF)
    assert smoke["upload_status_code"] == 201
    assert smoke["analyze_status_code"] == 202
    assert smoke["paper_result_status_code"] == 200
    assert smoke["question_status_code"] == 200
    assert smoke["job"]["status"] == "completed"
    assert smoke["job"]["seed"] == 378
    assert smoke["paper_result"]["status"] == "ready"
    assert smoke["paper_result"]["overview"]["abstract"]
    assert smoke["health_before"]["question_engine"] == "lazy"
    assert smoke["health_after"]["question_engine"] == "ready"
    preparation = smoke["question_runtime_preparation"]
    assert preparation["status"] == "ready"
    assert preparation["embedding_device"] == "cpu"
    assert preparation["reranker_device"] == "cpu"
    assert preparation["generator"]["device"] in {"cpu", "cuda"}
    answer = smoke["question_result"]
    assert answer["question_language"] == "en"
    assert answer["sufficiency"] == "sufficient"
    assert answer["answer"].strip()
    assert answer["evidence"]
    assert answer["runtime"]["generation_attempts"] <= 3
    if answer["runtime"]["llm_device"] == "cpu":
        assert answer["runtime"]["generation_attempts"] <= 2
    assert answer["runtime"]["llm_device"] in {"cpu", "cuda"}
    assert answer["runtime"]["retrieval_seconds"] >= 0
    assert answer["runtime"]["generation_seconds"] >= 0
    assert all(item["page"] >= 1 and item["text"].strip() for item in answer["evidence"])
    assert smoke["runtime_embedding_cache_created"] is True
    assert smoke["paper_id_fixed_for_reproducible_acceptance"] is True
    assert result["temporary_runtime_removed"] is True

    print("Day 34 question/result/health gate passed")
    print(
        f"tests={result['tests_run']} "
        "HTTP=health->upload->analyze->paper-result->curl-question"
    )
    print(
        f"answer={answer['sufficiency']} "
        f"evidence_pages={[item['page'] for item in answer['evidence']]} "
        "seed=378"
    )
    print("scope=historical API snapshot; current answer code has a separate validator")


if __name__ == "__main__":
    main()
