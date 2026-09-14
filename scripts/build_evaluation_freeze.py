"""Build the final evaluation manifest and portfolio performance table."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from model_lock import load_model_lock
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


CONFIG_PATH = PROJECT_ROOT / "config" / "evaluation_freeze.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "final_evaluation.json"
TABLE_PATH = PROJECT_ROOT / "data" / "evaluation" / "final_performance.csv"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(config: dict) -> list[str]:
    paths = [path for group in config["source_groups"].values() for path in group]
    if len(paths) != len(set(paths)):
        raise ValueError("Evaluation freeze source paths must be unique")
    for path in paths:
        if not (PROJECT_ROOT / path).is_file():
            raise FileNotFoundError(f"Evaluation freeze source is missing: {path}")
    return sorted(paths)


def performance_rows(retrieval, expansion, abstention, answer, container):
    rows = []

    def add(area, metric, scope, value, unit, source, qualification):
        rows.append(
            {
                "area": area,
                "metric": metric,
                "scope": scope,
                "value": value,
                "unit": unit,
                "source": source,
                "qualification": qualification,
            }
        )

    corpus = retrieval["corpus"]
    metrics = retrieval["metrics"]
    add("corpus", "papers", "robustness", corpus["papers"], "count", "frozen_retrieval_metrics.json", "Only the first 10 papers have verified accuracy labels.")
    add("corpus", "pages", "robustness", corpus["pages"], "count", "frozen_retrieval_metrics.json", "Text-extractable PDF pages.")
    add("corpus", "chunks", "robustness", corpus["chunks"], "count", "frozen_retrieval_metrics.json", "Section-aware searchable chunks.")
    add("corpus", "expansion_pass_rate", "papers-011..030", expansion["summary"]["passed"] / expansion["summary"]["papers"], "ratio", "corpus_expansion_smoke.json", "Runtime smoke only, not an accuracy claim.")
    add("retrieval", "questions", "verified-40", metrics["questions"], "count", "frozen_retrieval_metrics.json", "20 English and 20 Korean questions.")
    for name in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr"):
        add("retrieval", name, "verified-40-current", metrics[name], "ratio", "frozen_retrieval_metrics.json", "Current code, CPU retrieval devices, and revision-pinned models.")
    add("retrieval", "mrr", "verified-40-en", metrics["by_language"]["en"]["mrr"], "ratio", "frozen_retrieval_metrics.json", "English subset, 20 questions.")
    add("retrieval", "mrr", "verified-40-ko", metrics["by_language"]["ko"]["mrr"], "ratio", "frozen_retrieval_metrics.json", "Korean subset, 20 questions.")

    answer_summary = answer["summary"]
    add("answer_quality", "questions_reviewed", "fixed-20", answer_summary["questions_reviewed"], "count", "grounded_generation_review.json", "13 independent meanings because some questions are bilingual pairs.")
    add("answer_quality", "strict_pass_rate", "fixed-20", answer_summary["strict_pass_rate"], "ratio", "grounded_generation_review.json", "Assistant-led semantic review.")
    add("answer_quality", "pass_or_partial_rate", "fixed-20", answer_summary["pass_or_partial_rate"], "ratio", "grounded_generation_review.json", "Assistant-led semantic review.")
    add("answer_quality", "citation_failures", "fixed-20", answer_summary["citation_support"]["fail"], "count", "grounded_generation_review.json", "Refused questions have no citation and are not applicable.")
    add("answer_quality", "false_abstentions", "fixed-20", answer_summary["false_abstentions"], "count", "grounded_generation_review.json", "Verified answerable questions refused by the relevance gate.")
    add("answer_quality", "answer_stage_time", "fixed-20", answer_summary["fixed_evidence_runtime_seconds"], "seconds_total", "grounded_generation_review.json", "Saved evidence input; not full HTTP latency.")

    abstention_summary = abstention["summary"]
    add("abstention", "answerable_retention", "verified-40", abstention_summary["answerable_retention"], "ratio", "abstention_evaluation.json", "Verified answerable questions retained.")
    add("abstention", "unsupported_refusal", "calibration-10", abstention_summary["calibration_refusal_rate"], "ratio", "abstention_evaluation.json", "Unsupported calibration questions.")
    add("abstention", "unsupported_refusal", "holdout-10", abstention_summary["holdout_refusal_rate"], "ratio", "abstention_evaluation.json", "Unsupported holdout questions.")

    container_summary = container["summary"]
    container_runtime = container_summary["runtime"]
    add("container", "questions_completed", "3-papers-10-questions", container_summary["questions"], "count", "container_qna_review.json", "Five bilingual question meanings.")
    add("container", "pass_or_partial_rate", "3-papers-10-questions", container_summary["pass_or_partial_rate"], "ratio", "container_qna_review.json", "Assistant-led semantic review.")
    add("container", "citation_support", "3-papers-10-questions", container_summary["citation_support"]["pass"] / container_summary["questions"], "ratio", "container_qna_review.json", "Live returned evidence inspected after generation.")
    add("container", "exact_gold_page_hit_rate", "3-papers-10-questions", container_summary["exact_gold_page_hit_rate"], "ratio", "container_qna_review.json", "Disclosed page-alignment limitation.")
    add("container", "english_api_mean", "cpu-5-questions", container_runtime["english_api_mean_seconds"], "seconds", "container_qna_review.json", "Hardware-specific local Docker measurement.")
    add("container", "korean_api_mean", "cpu-5-questions", container_runtime["korean_api_mean_seconds"], "seconds", "container_qna_review.json", "Hardware-specific local Docker measurement.")
    add("container", "first_analysis", "empty-model-cache", container_runtime["first_analysis_seconds"], "seconds", "container_qna_review.json", "Includes model download and initialization.")
    add("container", "later_analysis_mean", "warm-model-cache-2-papers", container_runtime["later_analysis_mean_seconds"], "seconds", "container_qna_review.json", "Hardware-specific local Docker measurement.")
    return rows


def render_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("area", "metric", "scope", "value", "unit", "source", "qualification"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_outputs():
    config = load_json(CONFIG_PATH)
    if config.get("status") != "frozen" or config.get("seed") != 378:
        raise ValueError("Evaluation freeze must be frozen with seed 378")
    evaluation = PROJECT_ROOT / "data" / "evaluation"
    retrieval = load_json(PROJECT_ROOT / config["metric_policy"]["current_retrieval_source"])
    expansion = load_json(evaluation / "corpus_expansion_smoke.json")
    abstention = load_json(evaluation / "abstention_evaluation.json")
    answer = load_json(evaluation / "grounded_generation_review.json")
    container = load_json(evaluation / "container_qna_review.json")
    rows = performance_rows(retrieval, expansion, abstention, answer, container)
    csv_text = render_csv(rows)
    gates = config["quality_gates"]
    checks = {
        "retrieval": retrieval["status"] == "passed" and retrieval["metrics"]["recall_at_5"] >= gates["retrieval"]["minimum_recall_at_5"] and retrieval["metrics"]["recall_at_10"] >= gates["retrieval"]["minimum_recall_at_10"] and retrieval["metrics"]["mrr"] >= gates["retrieval"]["minimum_mrr"] and len(retrieval["lost_top5_hits"]) <= gates["retrieval"]["maximum_lost_top5_hits"],
        "corpus_expansion": expansion["summary"]["passed"] >= gates["corpus_expansion"]["required_passed_papers"] and expansion["summary"]["failed"] <= gates["corpus_expansion"]["maximum_failed_papers"],
        "abstention": abstention["summary"]["answerable_retention"] >= gates["abstention"]["minimum_answerable_retention"] and abstention["summary"]["calibration_refusal_rate"] >= gates["abstention"]["minimum_calibration_refusal"] and abstention["summary"]["holdout_refusal_rate"] >= gates["abstention"]["minimum_holdout_refusal"],
        "answer_quality": answer["summary"]["pass_or_partial_rate"] >= gates["answer_quality"]["minimum_pass_or_partial_rate"] and answer["summary"]["citation_support"]["fail"] <= gates["answer_quality"]["maximum_citation_failures"] and answer["summary"]["false_abstentions"] <= gates["answer_quality"]["maximum_false_abstentions"],
        "container": container["summary"]["questions"] == gates["container"]["required_questions"] and container["summary"]["language_match"] == gates["container"]["required_language_matches"] and container["summary"]["citation_support"]["pass"] == gates["container"]["required_citation_support"] and container["summary"]["container_contract"]["fallbacks"] <= gates["container"]["maximum_fallbacks"] and all(value == "pass" for value in container["acceptance"].values()),
    }
    paths = source_paths(config)
    hashes = {path: sha256(PROJECT_ROOT / path) for path in paths}
    bundle = hashlib.sha256()
    for path, digest in hashes.items():
        bundle.update(path.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
    payload = {
        "schema_version": 1,
        "evaluation": "final-portfolio-evaluation-freeze",
        "status": "frozen" if all(checks.values()) else "failed",
        "freeze_date": config["freeze_date"],
        "seed": config["seed"],
        "delivery": config["delivery"],
        "evaluation_boundary": config["evaluation_boundary"],
        "models": load_model_lock()["models"],
        "metrics": {
            "corpus": {**retrieval["corpus"], "expansion_smoke_passed": expansion["summary"]["passed"], "expansion_smoke_failed": expansion["summary"]["failed"]},
            "retrieval": retrieval["metrics"],
            "abstention": abstention["summary"],
            "answer_quality": answer["summary"],
            "container": container["summary"],
        },
        "checks": checks,
        "checks_scope": "Saved reviewed artifacts and current deterministic validators; long model benchmarks are not repeated by the lightweight verifier.",
        "performance_table": {
            "path": str(TABLE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "rows": len(rows),
            "sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        },
        "provenance": {
            "freeze_config": str(CONFIG_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_hashes": hashes,
            "source_bundle_sha256": bundle.hexdigest(),
        },
        "known_limits": [
            "Only 10 of the 30 papers have manually verified questions and Gold evidence.",
            "Answer quality labels were assistant-led rather than independently human-reviewed.",
            "The fixed 20-question answer review represents 13 independent meanings.",
            "The container benchmark represents five bilingual meanings from three papers.",
            "Container latency is specific to the measured laptop and CPU-only images.",
            "Korean answers are slower and can be less fluent or less complete than English answers.",
            "The local translation model license limits this portfolio use to noncommercial scope.",
        ],
        "change_policy": config["change_policy"],
    }
    return csv_text, payload


def main():
    configure_utf8_stdout()
    csv_text, payload = build_outputs()
    TABLE_PATH.write_text(csv_text, encoding="utf-8")
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    current = payload["metrics"]["retrieval"]
    print("Final evaluation freeze built")
    print(f"status={payload['status']} sources={len(payload['provenance']['source_hashes'])} performance_rows={payload['performance_table']['rows']} seed={payload['seed']}")
    print(f"retrieval=Recall@5:{current['recall_at_5']:.3f} Recall@10:{current['recall_at_10']:.3f} MRR:{current['mrr']:.4f}")
    if payload["status"] != "frozen":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
