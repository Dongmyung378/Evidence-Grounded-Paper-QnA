"""Validate original-roadmap completion gates for Days 25 through 27."""

import csv
import json

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


def load(name):
    path = PROJECT_ROOT / "data" / "evaluation" / name
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    configure_utf8_stdout()
    training = load("day25_training_data.json")
    smoke = load("day25_finetuning_smoke.json")
    fix = load("day27_error_fix.json")
    rejected = load("day27_pool_widening_rejected.json")
    ablation_path = PROJECT_ROOT / "data" / "evaluation" / "ablation.csv"
    report_path = PROJECT_ROOT / "data" / "evaluation" / "day25_day27_report.md"

    assert training["roadmap_day"] == 25
    assert training["leakage_policy"] == {
        "verified_40_role": "test_only",
        "verified_40_used_for_training": False,
        "qasper_test_used": False,
    }
    assert training["splits"]["train"]["questions"] == 1000
    assert training["splits"]["train"]["pairs"] == 4000
    assert training["splits"]["validation"]["questions"] == 200
    assert training["splits"]["validation"]["pairs"] == 800
    train_pairs = load_jsonl(PROJECT_ROOT / training["splits"]["train"]["path"])
    validation_pairs = load_jsonl(PROJECT_ROOT / training["splits"]["validation"]["path"])
    assert len(train_pairs) == 4000 and len(validation_pairs) == 800
    assert {row["label"] for row in train_pairs + validation_pairs} == {0, 1}
    assert all(row["source_dataset"] == "QASPER" for row in train_pairs + validation_pairs)
    assert all(row["split"] == "train" for row in train_pairs)
    assert all(row["split"] == "validation" for row in validation_pairs)
    assert smoke["roadmap_day"] == 25
    assert smoke["status"] == "completed_non_production_smoke"
    assert smoke["optimizer_steps"] == 1
    assert smoke["verified_40_used"] is False
    assert smoke["checkpoint_saved"] is False
    assert smoke["selected_runtime_model"] == "pretrained"

    with ablation_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    assert {row["system"] for row in rows} == {
        "bm25", "dense", "hybrid_rrf", "hybrid_reranker"
    }
    assert {row["scope"] for row in rows} == {"all", "en", "ko"}
    assert all(int(row["questions"]) in {20, 40} for row in rows)

    assert fix["roadmap_day"] == 27
    assert fix["evaluation_set"]["questions"] == 40
    assert fix["controlled_fix"]["question_specific_rules"] is False
    assert fix["after"]["metrics"]["recall_at_5"] == 1.0
    assert fix["after"]["metrics"]["recall_at_10"] == 1.0
    assert fix["cases"]["recovered_at_5"] == ["q-005-ko"]
    assert fix["cases"]["lost_at_5"] == []
    assert any(row["question_id"] == "q-005-ko" for row in fix["cases"]["changed_ranks"])
    assert fix["decision"]["adopt_for_day28_gate"] is True
    assert rejected["decision"]["adopt_for_day28_gate"] is False
    assert report_path.exists() and report_path.stat().st_size > 0

    print("Day 25-27 gate validation passed")
    print("day25=qasper_pairs+cpu_optimizer_step+pretrained_fallback")
    print("day26=ablation.csv systems=4 scopes=3")
    print("day27=q-005-ko_recovered lost_at_5=0 recall_at_5=1.000")


if __name__ == "__main__":
    main()
