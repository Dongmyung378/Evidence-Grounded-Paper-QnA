"""Calibrated Day 31 policy for refusing unsupported paper answers."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from grounded_answer_contract import canonical_abstention_payload
from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config" / "abstention.json"
REASON_TEXT = {
    "en": {
        "no_candidates": "No searchable text candidate was retrieved from the selected paper.",
        "no_evidence": "No usable text evidence was retrieved from the selected paper.",
        "low_relevance": "The retrieved text is not relevant enough to support an answer.",
    },
    "ko": {
        "no_candidates": "선택한 논문에서 검색 가능한 텍스트 후보를 찾지 못했습니다.",
        "no_evidence": "선택한 논문에서 사용할 수 있는 텍스트 근거를 찾지 못했습니다.",
        "low_relevance": "검색된 텍스트의 관련도가 낮아 답변을 뒷받침할 수 없습니다.",
    },
}


def load_abstention_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_abstention_config(config)


def validate_abstention_config(config):
    assert config["schema_version"] == 1
    assert config["roadmap_day"] == 31
    assert config["status"] == "calibrated"
    policy = config["policy"]
    assert policy["reject_empty_candidates"] is True
    assert policy["reject_empty_evidence"] is True
    assert isinstance(policy["minimum_top_reranker_score"], (int, float))
    assert policy["comparison"] == "accept_if_greater_than_or_equal"
    assert policy["normalize_model_abstention"] is True
    assert policy["skip_llm_on_pre_generation_rejection"] is True
    calibration = config["calibration"]
    assert calibration["answerable_cases"] == 40
    assert calibration["unsupported_cases"] == 10
    assert 0.0 <= calibration["minimum_answerable_retention"] <= 1.0
    assert calibration["minimum_unsupported_refusal_recall"] == 1.0
    return config


def abstention_config_fingerprint(config):
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def retrieval_signals(retrieval_result):
    candidates = retrieval_result.get("candidates") or []
    evidence = retrieval_result.get("evidence") or []
    scores = [float(item["reranker_score"]) for item in candidates]
    return {
        "candidate_count": len(candidates),
        "evidence_count": len(evidence),
        "top_reranker_score": max(scores) if scores else None,
    }


class AbstentionPolicy:
    """Apply deterministic pre-generation and post-generation refusal rules."""

    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = load_abstention_config(self.config_path)
        self.fingerprint = abstention_config_fingerprint(self.config)

    def decide_from_signals(self, signals):
        policy = self.config["policy"]
        candidate_count = int(signals.get("candidate_count", 0))
        evidence_count = int(signals.get("evidence_count", 0))
        top_score = signals.get("top_reranker_score")

        reason_code = None
        if policy["reject_empty_candidates"] and candidate_count == 0:
            reason_code = "no_candidates"
        elif policy["reject_empty_evidence"] and evidence_count == 0:
            reason_code = "no_evidence"
        elif top_score is None or not math.isfinite(float(top_score)):
            reason_code = "no_candidates"
        elif float(top_score) < policy["minimum_top_reranker_score"]:
            reason_code = "low_relevance"

        return {
            "abstain": reason_code is not None,
            "reason_code": reason_code,
            "signals": {
                "candidate_count": candidate_count,
                "evidence_count": evidence_count,
                "top_reranker_score": (
                    float(top_score) if top_score is not None and math.isfinite(float(top_score)) else None
                ),
                "minimum_top_reranker_score": policy[
                    "minimum_top_reranker_score"
                ],
            },
        }

    def evaluate_retrieval(self, retrieval_result):
        return self.decide_from_signals(retrieval_signals(retrieval_result))

    def refusal_payload(self, language, reason_code):
        if language not in REASON_TEXT:
            raise ValueError("language must be 'en' or 'ko'")
        if reason_code not in REASON_TEXT[language]:
            raise ValueError(f"unsupported abstention reason code: {reason_code}")
        return canonical_abstention_payload(
            language,
            REASON_TEXT[language][reason_code],
        )

    def enforce_model_response(self, response, language):
        if response["sufficiency"] == "sufficient":
            return response
        if not self.config["policy"]["normalize_model_abstention"]:
            return response
        return canonical_abstention_payload(
            language,
            response["abstention_reason"],
        )

    def metadata(self):
        return {
            "status": self.config["status"],
            "config_path": str(self.config_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "config_fingerprint": self.fingerprint,
            "policy": deepcopy(self.config["policy"]),
        }
