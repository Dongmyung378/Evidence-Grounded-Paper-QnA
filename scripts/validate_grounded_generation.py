"""Validate the evidence-first answer path and reviewed artifacts."""

import hashlib
import json
from pathlib import Path

from grounded_generation import load_answer_generation_config
from local_translation import load_translation_config
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


EVALUATION = PROJECT_ROOT / "data" / "evaluation"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    configure_utf8_stdout()
    outputs_path = EVALUATION / "grounded_generation_outputs.json"
    labels_path = EVALUATION / "grounded_generation_review_labels.jsonl"
    review_path = EVALUATION / "grounded_generation_review.json"
    outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    profile = json.loads((PROJECT_ROOT / "config" / "runtime_qna.json").read_text(encoding="utf-8"))
    answer_config = load_answer_generation_config(PROJECT_ROOT / profile["answer_generation_config"])
    translation_config = load_translation_config(PROJECT_ROOT / profile["translation_config"])

    provenance = outputs["provenance"]
    for path_key, hash_key in (
        ("baseline_outputs", "baseline_sha256"),
        ("answer_config", "answer_config_sha256"),
        ("translation_config", "translation_config_sha256"),
    ):
        assert sha256(PROJECT_ROOT / provenance[path_key]) == provenance[hash_key]
    assert sha256(PROJECT_ROOT / "scripts" / "grounded_generation.py") == provenance["implementation_sha256"]
    assert outputs["seed"] == 378
    assert outputs["gold_loaded_during_generation"] is False
    assert outputs["summary"]["questions"] == 20
    assert outputs["summary"]["sufficient"] == 18
    assert outputs["summary"]["insufficient"] == 2
    assert outputs["summary"]["pre_generation_refusals"] == 2
    assert outputs["summary"]["validation_fallbacks"] == 0
    assert len({row["question_id"] for row in outputs["results"]}) == 20
    assert all(row["fallback_used"] is False for row in outputs["results"])

    provenance = review["provenance"]
    assert provenance["outputs_sha256"] == sha256(outputs_path)
    assert provenance["manual_labels_sha256"] == sha256(labels_path)
    assert provenance["question_manifest_sha256"] == sha256(EVALUATION / "answer_review_manifest.json")
    assert provenance["verified_gold_sha256"] == sha256(EVALUATION / "gold_evidence.jsonl")
    assert review["gold_loaded_during_generation"] is False
    assert review["summary"]["verdicts"] == {"pass": 3, "partial": 14, "fail": 3}
    assert review["summary"]["strict_pass_rate"] == 0.15
    assert review["summary"]["pass_or_partial_rate"] == 0.85
    assert review["summary"]["citation_support"] == {"pass": 18, "fail": 0, "not_applicable": 2}
    assert review["summary"]["false_abstentions"] == 2

    assert profile["answer_strategy"] == "evidence_first"
    assert answer_config["answer"]["mode_by_language"] == {"en": "extractive", "ko": "faithful_translation_model"}
    assert translation_config["model"] == "facebook/nllb-200-distilled-600M"
    assert translation_config["license"] == "cc-by-nc-4.0"
    assert translation_config["use_safetensors"] is True
    print("Evidence-first answer validation passed")
    print("fixed20=pass:3 partial:14 fail:3 pass_or_partial=85%")
    print("false_abstentions=2 validation_fallbacks=0")


if __name__ == "__main__":
    main()
