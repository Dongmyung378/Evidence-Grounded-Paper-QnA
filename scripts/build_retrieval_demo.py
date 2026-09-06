"""Build the Day 21 baseline table and explainable retrieval demo."""

import argparse
import json
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


DEFAULT_COMPARISON = PROJECT_ROOT / "data" / "evaluation" / "retrieval_comparison.json"
DEFAULT_BASELINE = PROJECT_ROOT / "data" / "evaluation" / "baseline_metrics.json"
DEFAULT_FAILURES = PROJECT_ROOT / "data" / "evaluation" / "failure_analysis.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "retrieval_demo.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "data" / "evaluation" / "retrieval_demo.md"
DEMO_CASES = (
    ("q-001-ko", "multilingual_success", "한국어 일반 질문을 lexical·semantic 신호가 함께 보완한 성공 사례"),
    ("q-042-en", "lexical_rescue", "숫자 중심 질문에서 BM25가 dense의 낮은 순위를 보완한 사례"),
    ("q-049-ko", "known_failure", "Top-5 실패와 원인 및 다음 개선 방향을 설명하는 사례"),
)


def metric_row(values):
    return {
        "recall_at_1": values["recall_at_1"],
        "recall_at_5": values["recall_at_5"],
        "recall_at_10": values["recall_at_10"],
        "mrr": values["mrr"],
    }


def rank_text(rank):
    return "not in Top-10" if rank is None else str(rank)


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    questions = {
        row["question_id"]: row
        for row in load_jsonl(PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl")
    }
    failure_map = {row["question_id"]: row for row in failures["cases"]}
    result_maps = {
        method: {row["question_id"]: row for row in rows}
        for method, rows in comparison["results"].items()
    }

    examples = []
    for question_id, demo_type, summary in DEMO_CASES:
        question = questions[question_id]
        method_results = {}
        for method in ("bm25", "dense", "hybrid_rrf"):
            result = result_maps[method][question_id]
            method_results[method] = {
                "gold_rank": result["gold_rank"],
                "hit_at_5": result["gold_rank"] is not None and result["gold_rank"] <= 5,
                "predicted_pages_at_5": result["predicted_pages"][:5],
            }
        review = failure_map.get(question_id)
        examples.append({
            "question_id": question_id,
            "demo_type": demo_type,
            "summary": summary,
            "paper_id": question["paper_id"],
            "language": question["question_language"],
            "question": question["question"],
            "gold_page": result_maps["hybrid_rrf"][question_id]["gold_page"],
            "method_results": method_results,
            "diagnosis": review["rationale"] if review else "세 검색기의 순위와 Top-5 페이지를 비교해 결합 효과를 확인한다.",
            "next_action": review["recommended_action"] if review else "동일 질문으로 검색 점수와 근거 텍스트를 확인한다.",
        })

    artifact = {
        "schema_version": 1,
        "roadmap_day": 21,
        "evaluation_set": baseline["evaluation_set"],
        "baseline_table": {
            method: metric_row(values)
            for method, values in baseline["metrics"].items()
        },
        "selected_baseline": baseline["selected_baseline"],
        "demo_definition": "success and failure examples from the same fixed page-level evaluation",
        "examples": examples,
        "interactive_command": (
            "python -B scripts/search_hybrid.py --paper-id paper-001 "
            "--query \"이 논문에서 다루는 주요 문제는 무엇인가?\" --top-k 5"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Day 21 Retrieval Baseline and Demo",
        "",
        "## Baseline table",
        "",
        "| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    display_names = {"bm25": "BM25", "dense": "Dense", "hybrid_rrf": "Hybrid RRF"}
    for method, metrics in artifact["baseline_table"].items():
        lines.append(
            f"| {display_names[method]} | {metrics['recall_at_1']:.3f} | "
            f"{metrics['recall_at_5']:.3f} | {metrics['recall_at_10']:.3f} | {metrics['mrr']:.4f} |"
        )
    lines.extend(["", "Hybrid RRF is retained as the selected baseline.", "", "## Retrieval demo", ""])
    for example in examples:
        lines.extend([
            f"### {example['question_id']} — {example['demo_type']}",
            "",
            f"- Question: {example['question']}",
            f"- Paper / gold page: {example['paper_id']} / {example['gold_page']}",
            f"- Summary: {example['summary']}",
        ])
        for method in ("bm25", "dense", "hybrid_rrf"):
            result = example["method_results"][method]
            lines.append(
                f"- {display_names[method]}: gold rank {rank_text(result['gold_rank'])}; "
                f"Top-5 pages {result['predicted_pages_at_5']}"
            )
        lines.extend([
            f"- Diagnosis: {example['diagnosis']}",
            f"- Next action: {example['next_action']}",
            "",
        ])
    lines.extend([
        "## Interactive reproduction",
        "",
        "```bash",
        artifact["interactive_command"],
        "```",
        "",
    ])
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"examples": len(examples), "selected_baseline": artifact["selected_baseline"]}, ensure_ascii=False))
    print(f"Saved: {args.output}")
    print(f"Saved: {args.markdown}")


if __name__ == "__main__":
    main()
