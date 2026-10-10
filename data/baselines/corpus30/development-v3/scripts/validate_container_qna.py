"""Validate the fixed Docker Compose Q&A benchmark and semantic review."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


EVALUATION = PROJECT_ROOT / "data" / "evaluation"
CONFIG = PROJECT_ROOT / "config" / "container_qna_benchmark.json"
RESULTS = EVALUATION / "container_qna_results.json"
LABELS = EVALUATION / "container_qna_review_labels.jsonl"
REVIEW = EVALUATION / "container_qna_review.json"
GOLD = EVALUATION / "gold_evidence.jsonl"
HANGUL = re.compile(r"[가-힣]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    configure_utf8_stdout()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    labels = load_jsonl(LABELS)
    selected = [
        (paper["paper_id"], question)
        for paper in config["papers"]
        for question in paper["questions"]
    ]
    expected = {question["question_id"]: (paper_id, question) for paper_id, question in selected}

    assert config["schema_version"] == 1
    assert config["seed"] == 378
    assert len(config["papers"]) == 3
    assert len(selected) == 10
    assert Counter(question["language"] for _, question in selected) == {
        "en": 5,
        "ko": 5,
    }
    assert len(expected) == 10

    provenance = results["provenance"]
    assert results["schema_version"] == 1
    assert results["seed"] == 378
    assert results["gold_loaded_during_execution"] is False
    assert provenance["benchmark_config_sha256"] == sha256(CONFIG)
    assert provenance["implementation_sha256"] == sha256(
        PROJECT_ROOT / "scripts" / "run_container_qna_benchmark.py"
    )
    # Runtime configuration digests identify the measured container image. They
    # are retained as historical provenance and need not match later path-only
    # configuration cleanup.

    assert results["health_before"]["status"] == "ok"
    assert results["health_before"]["version"] == "0.41.0"
    assert results["health_before"]["seed"] == 378
    assert results["health_before"]["papers"] == 0
    assert results["health_after"]["status"] == "ok"
    assert results["health_after"]["papers"] == 3
    assert results["health_after"]["jobs"]["completed"] == 3
    assert results["health_after"]["jobs"]["failed"] == 0
    assert results["ui_http_status"] == 200
    assert results["build_seconds"] > 0

    assert len(results["papers"]) == 3
    config_papers = {paper["paper_id"]: paper for paper in config["papers"]}
    for paper in results["papers"]:
        source = PROJECT_ROOT / config_papers[paper["benchmark_paper_id"]]["pdf"]
        assert len(paper["source_sha256"]) == 64
        assert paper["source_size_bytes"] > 0
        if source.is_file():
            assert paper["source_sha256"] == sha256(source)
            assert paper["source_size_bytes"] == source.stat().st_size
        assert paper["page_count"] > 0
        assert paper["text_page_count"] > 0
        assert paper["chunk_count"] > 0
        assert paper["observed_job_states"][-1] == "completed"
        preparation = paper["question_runtime_preparation"]
        assert preparation["status"] == "ready"
        assert preparation["embedding_device"] == "cpu"
        assert preparation["reranker_device"] == "cpu"
        assert preparation["generator"]["provider"] == "local_nllb"
        assert preparation["generator"]["device"] == "cpu"

    assert len(results["results"]) == 10
    assert {row["question_id"] for row in results["results"]} == set(expected)
    for row in results["results"]:
        paper_id, question = expected[row["question_id"]]
        assert row["benchmark_paper_id"] == paper_id
        assert row["language"] == question["language"]
        assert row["question"] == question["question"]
        assert row["sufficiency"] == "sufficient"
        assert row["answer"].strip()
        assert bool(HANGUL.search(row["answer"])) == (row["language"] == "ko")
        assert row["evidence"]
        assert row["api_runtime"]["fallback_used"] is False
        assert row["api_runtime"]["total_seconds"] >= 0
        for evidence in row["evidence"]:
            assert evidence["page"] >= 1
            assert evidence["text"].strip()
            assert evidence["locator"]["page_label"]
            assert evidence["locator"]["chunk_id"]

    assert results["summary"]["papers"] == 3
    assert results["summary"]["questions"] == 10
    assert results["summary"]["languages"] == {"en": 5, "ko": 5}
    assert results["summary"]["sufficient"] == 10
    assert results["summary"]["fallbacks"] == 0
    assert results["summary"]["evidence_responses"] == 10
    assert results["summary"]["by_language"]["en"]["questions"] == 5
    assert results["summary"]["by_language"]["ko"]["questions"] == 5

    restart = results["offline_restart"]
    assert restart["local_files_only"] is True
    assert restart["response_matches_initial_run"] is True
    assert restart["runtime_volume_preserved"] is True
    assert restart["cache_size_unchanged"] is True
    assert restart["cache_before_restart"] == restart["cache_after_restart"]
    assert restart["health_before_question"]["papers"] == 3
    assert results["stack_removed_after_evaluation"] is True
    assert results["temporary_runtime_volume_removed"] is True
    assert results["model_cache_volume_preserved"] is True

    assert len(labels) == 10
    assert {row["question_id"] for row in labels} == set(expected)
    assert Counter(row["verdict"] for row in labels) == {
        "pass": 4,
        "partial": 6,
    }
    assert all(row["citation_support"] == "pass" for row in labels)
    assert all(row["language_match"] is True for row in labels)

    review_provenance = review["provenance"]
    assert review_provenance["benchmark_config_sha256"] == sha256(CONFIG)
    assert review_provenance["container_results_sha256"] == sha256(RESULTS)
    assert review_provenance["manual_labels_sha256"] == sha256(LABELS)
    assert review_provenance["verified_gold_sha256"] == sha256(GOLD)
    assert review["gold_loaded_during_container_execution"] is False
    assert review["summary"]["verdicts"] == {
        "pass": 4,
        "partial": 6,
        "fail": 0,
    }
    assert review["summary"]["pass_or_partial_rate"] == 1.0
    assert review["summary"]["citation_support"] == {"pass": 10, "fail": 0}
    assert review["summary"]["language_match"] == 10
    assert review["summary"]["exact_gold_page_hits"] == 4
    assert review["summary"]["by_language"]["en"]["verdicts"] == {
        "pass": 3,
        "partial": 2,
        "fail": 0,
    }
    assert review["summary"]["by_language"]["ko"]["verdicts"] == {
        "pass": 1,
        "partial": 4,
        "fail": 0,
    }
    assert review["language_decision"]["decision"] == (
        "retain_korean_with_documented_limits"
    )
    assert all(value == "pass" for value in review["acceptance"].values())
    print("Container Q&A benchmark validation passed")
    print("papers=3 questions=10 en=5 ko=5 contract=10/10")
    print("review=pass:4 partial:6 fail:0 citation_support=10/10")
    print("offline_restart=passed cleanup=passed korean=retained_with_limits")


if __name__ == "__main__":
    main()
