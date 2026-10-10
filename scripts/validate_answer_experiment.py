"""Validate published trial provenance and the predeclared stopping decision."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data/quality_benchmark/runs/answer-builder-pilot"

def read(name):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary = read("summary.json")
    for name, expected in summary["artifact_sha256"].items():
        assert digest(ARTIFACTS / name) == expected, f"Changed experiment evidence: {name}"
    protocol, receipt, review = read("protocol.json"), read("prompt-receipt.json"), read("review.json")
    assert digest(ARTIFACTS / "inputs.json") == protocol["input_sha256"]
    assert digest(ARTIFACTS / "review-rubric.json") == protocol["review_rubric_sha256"]
    assert digest(ARTIFACTS / "measured_runner.py") == receipt["runner_sha256"]
    assert digest(ARTIFACTS / "candidate-config.json") == receipt["config_sha256"]
    snapshot = ROOT / "data/quality_benchmark/runs/2026-10-10-final/real-paper-runtime-freeze-sources"
    for relative, expected in {**protocol["runtime_config_sha256"], **protocol["implementation_sha256"]}.items():
        assert digest(snapshot / relative) == expected, f"Measured runtime provenance changed: {relative}"
    for relative, expected in protocol["source_sha256"].items():
        assert digest(ROOT / relative) == expected
    assert not review["independent_human_review"] and receipt["frozen_before_execution"]
    assert receipt["attempts_per_case"] == 1
    cases = read("inputs.json")["cases"]
    ids = [case["case_id"] for case in cases]
    assert len(ids) == len(set(ids)) == protocol["pilot_outputs"] == 8
    assert {case["meaning_id"] for case in cases} == set(protocol["pilot_groups"])
    assert all("reference_answer" not in case for case in cases)
    assert [case["case_id"] for case in review["cases"]] == ids
    tokens = read("token-counts.json")
    assert tokens["prompt_receipt_sha256"] == digest(ARTIFACTS / "prompt-receipt.json")
    assert set(tokens["input_tokens"]) == set(ids)
    assert all(2 < count <= protocol["candidate"]["max_input_tokens"] for count in tokens["input_tokens"].values())
    for mode in ["baseline", "candidate"]:
        raw = read(mode + "-outputs.json")
        assert raw["runtime_unchanged"] and [case["case_id"] for case in raw["outputs"]] == ids
        assert raw["runner_sha256"] == receipt["runner_sha256"]
        assert raw["prompt_sha256"] == digest(ARTIFACTS / "prompt-receipt.json")
        assert raw["input_sha256"] == protocol["input_sha256"]
        assert summary[mode]["valid_contract"] == sum(bool(row["valid_contract"]) for row in raw["outputs"])
        assert abs(summary[mode]["mean_seconds"] - statistics.mean(row["seconds"] for row in raw["outputs"])) < 1e-9
        for label in ["complete", "partial", "fail"]:
            assert summary[mode][label] == sum(row[mode]["label"] == label for row in review["cases"])
        assert summary[mode]["material_unsupported_outputs"] == sum(row[mode]["material_unsupported_claim"] for row in review["cases"])
    gate = protocol["continuation_gate"]
    additional_complete = summary["candidate"]["complete"] - summary["baseline"]["complete"]
    additional_wrong = summary["candidate"]["material_unsupported_outputs"] - summary["baseline"]["material_unsupported_outputs"]
    passed = (additional_complete >= gate["minimum_additional_complete_outputs"]
              and additional_wrong <= gate["maximum_additional_material_unsupported_outputs"]
              and summary["candidate"]["mean_seconds"] <= gate["maximum_mean_candidate_seconds"])
    assert summary["gate"]["additional_complete"] == additional_complete
    assert summary["gate"]["additional_material_unsupported"] == additional_wrong
    assert summary["predeclared_gate_passed"] == passed is False
    assert summary["production_runtime_changed"] is False
    assert summary["decision"] == "stop_model_expansion_keep_default_runtime"
    print("Answer-model experiment validated: 8 DEV outputs, measured artifacts intact, continuation gate failed")
    print("Content and contract reviewed separately; no independent human accuracy claim")

if __name__ == "__main__":
    main()
