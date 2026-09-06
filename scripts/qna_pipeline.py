"""Day 30 Q&A pipeline with the calibrated Day 31 abstention policy."""

import time
import json
from pathlib import Path

from abstention_policy import CONFIG_PATH as ABSTENTION_CONFIG_PATH
from abstention_policy import AbstentionPolicy
from answer_quality import validate_informative_answer
from grounded_answer_contract import (
    ABSTENTION_ANSWERS,
    AnswerValidationError,
    build_grounded_answer_request,
    canonical_abstention_payload,
    detect_question_language,
    parse_and_validate_answer,
    resolve_answer_evidence,
)
from local_llm import CONFIG_PATH, LocalTransformersLLM
from local_llm import generation_config_fingerprint, load_generation_config
from production_retrieval import ProductionRetrieval
from retrieval_common import PROJECT_ROOT


def _fallback_reason(language):
    if language == "ko":
        return "로컬 답변 모델이 검증 가능한 JSON 응답을 생성하지 못했습니다."
    return "The local answer model did not produce a valid grounded JSON response."


class GroundedQAPipeline:
    """Run retrieval, refusal policy, local LLM, and grounded-output validation."""

    def __init__(
        self,
        retrieval=None,
        llm=None,
        generation_config_path=CONFIG_PATH,
        abstention_config_path=None,
        enable_abstention=True,
        local_files_only=False,
    ):
        self.retrieval = retrieval or ProductionRetrieval()
        self.generation_config_path = Path(generation_config_path)
        self.generation_config = load_generation_config(self.generation_config_path)
        self.llm = llm
        self.local_files_only = local_files_only
        self.enable_abstention = bool(enable_abstention)
        if abstention_config_path is None:
            runtime_path = PROJECT_ROOT / "config/runtime_qna.json"
            if runtime_path.exists():
                profile = json.loads(runtime_path.read_text(encoding="utf-8"))
                abstention_config_path = PROJECT_ROOT / profile["abstention_config"]
            else:
                abstention_config_path = ABSTENTION_CONFIG_PATH
        self.abstention_policy = (
            AbstentionPolicy(abstention_config_path)
            if self.enable_abstention
            else None
        )

    def _get_llm(self):
        if self.llm is None:
            self.llm = LocalTransformersLLM(
                self.generation_config_path,
                local_files_only=self.local_files_only,
            )
        return self.llm

    def _llm_metadata(self, loaded):
        if self.llm is not None and callable(getattr(self.llm, "metadata", None)):
            return {**self.llm.metadata(), "loaded": loaded}
        if self.llm is not None:
            return {
                "provider": getattr(self.llm, "provider_name", "injected_test_double"),
                "model": getattr(self.llm, "model_name", "unknown"),
                "loaded": loaded,
            }
        return {
            "provider": self.generation_config["provider"],
            "model": self.generation_config["model"],
            "revision": self.generation_config["revision"],
            "config_path": str(
                self.generation_config_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "config_fingerprint": generation_config_fingerprint(
                self.generation_config
            ),
            "loaded": False,
        }

    def ask(self, query, paper_id, question_language=None, include_debug=False):
        started = time.perf_counter()
        detected_language = detect_question_language(query)
        language = question_language or detected_language
        if language not in {"en", "ko"} or language != detected_language:
            raise ValueError("question_language must match the question text (en or ko)")
        retrieval_result = self.retrieval.run(query, paper_id)
        evidence = retrieval_result["evidence"]

        policy_decision = {
            "abstain": False,
            "reason_code": None,
            "signals": None,
        }
        if self.enable_abstention:
            policy_decision = self.abstention_policy.evaluate_retrieval(
                retrieval_result
            )

        pre_generation_abstention = policy_decision["abstain"]
        attempts = []
        response = None
        fallback_used = False
        abstention_source = None
        llm_loaded = False

        if pre_generation_abstention:
            response = self.abstention_policy.refusal_payload(
                language,
                policy_decision["reason_code"],
            )
            abstention_source = "pre_generation_policy"
        else:
            request = build_grounded_answer_request(
                query,
                evidence,
                question_language=language,
            )
            base_messages = request["messages"]
            messages = list(base_messages)
            max_attempts = self.generation_config["validation_attempts"]
            llm = self._get_llm()
            llm_loaded = True
            for attempt_number in range(1, max_attempts + 1):
                raw_response = ""
                try:
                    raw_response = llm.generate(
                        messages,
                        response_schema=request["response_schema"],
                    )
                    response = parse_and_validate_answer(
                        raw_response,
                        evidence,
                        language,
                    )
                    if self.enable_abstention:
                        validate_informative_answer(response, query)
                    attempts.append(
                        {
                            "attempt": attempt_number,
                            "status": "valid",
                            "raw_response": raw_response if include_debug else None,
                        }
                    )
                    break
                except (AnswerValidationError, ValueError, RuntimeError) as exc:
                    response = None
                    attempts.append(
                        {
                            "attempt": attempt_number,
                            "status": "invalid",
                            "error": str(exc),
                            "raw_response": raw_response if include_debug else None,
                        }
                    )
                    messages = base_messages + [
                        {"role": "assistant", "content": raw_response},
                        {
                            "role": "user",
                            "content": (
                                "Your response failed validation. Correct it instead of "
                                "repeating it. Return exactly one JSON object with answer, "
                                "evidence_ids, sufficiency, and abstention_reason. "
                                f"The answer language must be {language}. "
                                "For sufficient, copy one or more IDs exactly from this list: "
                                f"{request['available_evidence_ids']}. "
                                "For insufficient, evidence_ids must be [] and answer must "
                                f"be exactly {ABSTENTION_ANSWERS[language]!r}. "
                                "Never output the placeholder word 'string'. "
                                f"Validation error: {exc}"
                            ),
                        },
                    ]

            fallback_used = response is None
            if fallback_used:
                if not self.generation_config["safe_fallback_on_validation_error"]:
                    raise AnswerValidationError(
                        "all generation attempts failed validation"
                    )
                response = canonical_abstention_payload(
                    language,
                    _fallback_reason(language),
                )
                abstention_source = "validation_fallback"
            elif self.enable_abstention:
                response = self.abstention_policy.enforce_model_response(
                    response,
                    language,
                )
                if response["sufficiency"] == "insufficient":
                    abstention_source = "model"

        cited_evidence = resolve_answer_evidence(response, evidence)
        stages = [
            "query",
            "production_retrieval",
            "reranker",
            "evidence_selection",
        ]
        if self.enable_abstention:
            stages.append("abstention_policy")
        if not pre_generation_abstention:
            stages.extend(["local_llm", "answer_validation"])
        if self.enable_abstention:
            stages.append("abstention_enforcement")

        result = {
            "schema_version": 1,
            "roadmap_day": 31 if self.enable_abstention else 30,
            "roadmap_days": [30, 31] if self.enable_abstention else [30],
            "paper_id": paper_id,
            "query": query,
            "question_language": language,
            "response": response,
            "cited_evidence": cited_evidence,
            "retrieved_evidence": evidence,
            "pipeline": {
                "stages": stages,
                "retrieval_config_fingerprint": retrieval_result["production"][
                    "config_fingerprint"
                ],
                "llm": self._llm_metadata(llm_loaded),
                "llm_skipped": pre_generation_abstention,
                "generation_attempts": len(attempts),
                "fallback_used": fallback_used,
                "abstention_source": abstention_source,
                "abstention": {
                    "enabled": self.enable_abstention,
                    "decision": policy_decision,
                    "policy": (
                        self.abstention_policy.metadata()
                        if self.enable_abstention
                        else None
                    ),
                },
                "runtime_seconds": round(time.perf_counter() - started, 3),
            },
        }
        if include_debug:
            result["debug"] = {"attempts": attempts}
        return result
