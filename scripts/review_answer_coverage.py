"""모델 재실행 없이 저장된 답변 확장 실험과 사후 판정을 검증하고 집계한다."""

import argparse
import hashlib
import json
from collections import Counter
from statistics import mean

from grounded_generation import _normalize_text
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


OUTPUT = PROJECT_ROOT / "data/evaluation/answer_coverage_review.json"


def read(path):
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()


def build_review():
    outputs_path = "data/evaluation/answer_coverage_outputs.json"
    labels_path = "data/evaluation/answer_coverage_review_labels.jsonl"
    outputs = read(outputs_path)
    assert outputs["seed"] == 378 and outputs["gold_loaded_during_generation"] is False
    for path, expected in outputs["source_hashes"].items():
        assert digest(path) == expected, f"실험 출처가 변경됨: {path}"
    rows = outputs["results"]
    by_id = {r["question"]["question_id"]: r for r in rows}
    answerable = load_jsonl(PROJECT_ROOT / "data/evaluation/questions.jsonl")
    unsupported = load_jsonl(PROJECT_ROOT / "data/evaluation/abstention_questions.jsonl")
    expected_ids = {q["question_id"] for q in answerable} | {q["case_id"] for q in unsupported}
    assert len(rows) == len(by_id) == 60 and set(by_id) == expected_ids
    for question in answerable + unsupported:
        actual = by_id[question.get("question_id", question.get("case_id"))]["question"]
        assert all(actual[key] == question[key] for key in ("paper_id", "question", "question_language"))
    labels = load_jsonl(PROJECT_ROOT / labels_path)
    manifest_path = "data/evaluation/day32_review_manifest.json"
    fixed_ids = {q["question_id"] for q in read(manifest_path)["questions"]}
    assert len(labels) == 20 and {r["question_id"] for r in labels} == fixed_ids
    for label in labels:
        assert label["baseline"] in {"pass", "partial", "fail"}
        assert label["candidate"] in {"pass", "partial", "fail"}

    chunks = {c["chunk_id"]: c for c in load_jsonl(PROJECT_ROOT / "data/processed/chunks.jsonl")}
    checked = 0
    for row in rows:
        candidate = row["candidate"]
        response = candidate["response"]
        if response["sufficiency"] == "insufficient":
            assert not response["evidence_ids"]
            continue
        evidence = {e["evidence_id"]: e for e in candidate["answer_evidence"]}
        cited_ids = response["evidence_ids"]
        assert 1 <= len(cited_ids) <= 5 and set(cited_ids) <= set(evidence)
        for item in evidence.values():
            chunk = chunks[item["chunk_id"]]
            assert item["paper_id"] == chunk["paper_id"] == row["question"]["paper_id"]
            assert item["page"] == chunk["page"]
            assert _normalize_text(item["text"]) == _normalize_text(chunk["text"])
        selected_ids = set()
        for sentence in candidate["selected_sentences"]:
            parts = []
            for span in sentence["source_spans"]:
                source = _normalize_text(evidence[span["evidence_id"]]["text"])
                assert 0 <= span["start"] < span["end"] <= len(source)
                parts.append(source[span["start"]:span["end"]])
                selected_ids.add(span["evidence_id"])
            assert "".join(parts) == sentence["text"]
        assert selected_ids == set(cited_ids)
        checked += 1

    metrics = {}
    for variant in ("baseline", "candidate"):
        valid = [r for r in rows if r["question"]["question_id"].startswith("q-")]
        invalid = [r for r in rows if r not in valid]
        refusal = lambda r: r[variant]["response"]["sufficiency"] == "insufficient"
        metrics[variant] = {
            "fixed20_verdicts": dict(Counter(label[variant] for label in labels)),
            "answerable_answered": sum(not refusal(r) for r in valid),
            "unsupported_refused": sum(refusal(r) for r in invalid),
            "unsupported_former_holdout_refused": sum(refusal(r) for r in invalid if r["question"]["question_id"].startswith("abs-test-")),
            "mean_answer_stage_seconds": round(mean(r[variant]["answer_seconds"] +
                (r["sentence_ranking_seconds"] if variant == "candidate" else 0) for r in valid), 6),
        }
    hashes = {path: digest(path) for path in (outputs_path, labels_path, manifest_path,
        "data/evaluation/gold_evidence.jsonl", "scripts/review_answer_coverage.py")}
    return {
        "schema_version": 1, "seed": 378, "decision": "opt_in_only_default_unchanged",
        "reviewer": "assistant_led_not_independent_human_review",
        "evaluation_scope": "재사용한 개발 질문 40개, 근거 없는 질문 20개. 사후 의미 검토는 고정 20개에 한정한다.",
        "metric_scope": "답변 생성 여부는 정답률이 아니다. 과거 holdout도 이번 오류 분석에 사용했으므로 더 이상 독립 테스트가 아니다.",
        "timing_scope": "호스트 답변 단계 평균. 후보의 문장 재정렬 시간을 포함하며 검색, 모델 초기화, HTTP 및 Docker 시간은 제외한다. 실행 순서와 번역 캐시 영향이 있어 속도 벤치마크로 일반화하지 않는다.",
        "metrics": metrics, "candidate_source_traceability_checked": checked,
        "detail_regression_ids": [r["question_id"] for r in labels if r.get("detail_regression")],
        "translation_semantics_automatically_verified": False,
        "source_hashes": hashes,
    }


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    summary = build_review()
    if args.write:
        OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        assert summary == json.loads(OUTPUT.read_text(encoding="utf-8"))
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print("저장된 60문항 및 원문 연결 검증 통과. 기본 경로는 기존 방식 유지.")


if __name__ == "__main__":
    main()
