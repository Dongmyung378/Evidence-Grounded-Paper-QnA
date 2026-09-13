"""Runtime pipeline for retrieval, abstention, and grounded answers."""

import time
import json
from pathlib import Path

from abstention_policy import CONFIG_PATH as ABSTENTION_CONFIG_PATH
from abstention_policy import AbstentionPolicy
from answer_quality import validate_informative_answer
from grounded_generation import (
    CONFIG_PATH as ANSWER_GENERATION_CONFIG_PATH,
    generate_evidence_first_answer,
    load_answer_generation_config,
)
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
from local_translation import (
    CONFIG_PATH as TRANSLATION_CONFIG_PATH,
    LocalEnglishKoreanTranslator,
    load_translation_config,
    translation_config_fingerprint,
)
from production_retrieval import ProductionRetrieval
from reranker import load_reranker
from retrieval_common import PROJECT_ROOT


def _fallback_reason(language):
    if language == "ko":
        return "로컬 답변 모델이 검증 가능한 JSON 응답을 생성하지 못했습니다."
    return "The local answer model did not produce a valid grounded JSON response."


class GroundedQAPipeline:
    """Run retrieval, refusal policy, answer construction, and validation."""

    def __init__(
        self,
        retrieval=None,
        llm=None,
        generation_config_path=CONFIG_PATH,
        translation_config_path=None,
        answer_generation_config_path=None,
        abstention_config_path=None,
        enable_abstention=True,
        local_files_only=False,
        translator=None,
        sentence_selector=None,
        answer_strategy=None,
    ):
        self.retrieval = retrieval or ProductionRetrieval()
        self.local_files_only = local_files_only
        self.enable_abstention = bool(enable_abstention)
        runtime_path = PROJECT_ROOT / "config/runtime_qna.json"
        profile = (
            json.loads(runtime_path.read_text(encoding="utf-8"))
            if runtime_path.exists()
            else {}
        )
        if answer_strategy is None:
            answer_strategy = (
                "legacy_json"
                if llm is not None and translator is None
                else profile.get("answer_strategy", "legacy_json")
            )
        if answer_strategy not in {"legacy_json", "evidence_first"}:
            raise ValueError("Unsupported answer strategy")
        self.answer_strategy = answer_strategy
        self.sentence_selector = sentence_selector
        if self.answer_strategy == "evidence_first":
            self.answer_generation_config_path = Path(
                answer_generation_config_path
                or PROJECT_ROOT
                / profile.get(
                    "answer_generation_config",
                    str(ANSWER_GENERATION_CONFIG_PATH.relative_to(PROJECT_ROOT)),
                )
            )
            self.answer_generation_config = load_answer_generation_config(
                self.answer_generation_config_path
            )
            self.generation_config_path = Path(
                translation_config_path
                or PROJECT_ROOT
                / profile.get(
                    "translation_config",
                    str(TRANSLATION_CONFIG_PATH.relative_to(PROJECT_ROOT)),
                )
            )
            self.generation_config = load_translation_config(
                self.generation_config_path
            )
            self.llm = translator or llm
        else:
            self.answer_generation_config_path = None
            self.answer_generation_config = None
            self.generation_config_path = Path(generation_config_path)
            self.generation_config = load_generation_config(
                self.generation_config_path
            )
            self.llm = llm
        if abstention_config_path is None:
            if runtime_path.exists():
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
            llm_class = (
                LocalEnglishKoreanTranslator
                if self.answer_strategy == "evidence_first"
                else LocalTransformersLLM
            )
            self.llm = llm_class(
                self.generation_config_path, local_files_only=self.local_files_only
            )
        return self.llm

    def _get_sentence_selector(self):
        if self.sentence_selector is not None:
            return self.sentence_selector
        pipeline = getattr(self.retrieval, "pipeline", None)
        reranker = getattr(pipeline, "reranker", None)
        if reranker is not None:
            self.sentence_selector = reranker
        else:
            config = self.answer_generation_config["sentence_selector"]
            self.sentence_selector = load_reranker(
                config["model"],
                max_length=config["max_length"],
                local_files_only=self.local_files_only,
                device="cpu",
            )
        return self.sentence_selector

    def prepare_generator(self):
        """Load the configured local generator without running an answer request."""
        return self._get_llm().metadata()

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
            "config_fingerprint": (
                translation_config_fingerprint(self.generation_config)
                if self.answer_strategy == "evidence_first"
                else generation_config_fingerprint(self.generation_config)
            ),
            "loaded": False,
        }

    def ask(self, query, paper_id, question_language=None, include_debug=False):
        started = time.perf_counter()
        detected_language = detect_question_language(query)
        language = question_language or detected_language
        if language not in {"en", "ko"} or language != detected_language:
            raise ValueError("question_language must match the question text (en or ko)")
        retrieval_started = time.perf_counter()
        retrieval_result = self.retrieval.run(query, paper_id)
        retrieval_seconds = time.perf_counter() - retrieval_started
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
        generation_seconds = 0.0

        if pre_generation_abstention:
            response = self.abstention_policy.refusal_payload(
                language,
                policy_decision["reason_code"],
            )
            abstention_source = "pre_generation_policy"
        elif self.answer_strategy == "evidence_first":
            generation_started = time.perf_counter()
            answer_model = self._get_llm() if language == "ko" else None
            llm_loaded = language == "ko"
            generated = generate_evidence_first_answer(
                answer_model,
                self._get_sentence_selector(),
                query,
                evidence,
                question_language=language,
                config=self.answer_generation_config,
            )
            generation_seconds = time.perf_counter() - generation_started
            response = generated["response"]
            attempts = [
                {
                    **attempt,
                    "raw_response": (
                        attempt.get("raw_response") if include_debug else None
                    ),
                }
                for attempt in generated["attempts"]
            ]
            fallback_used = generated["fallback_used"]
            if fallback_used:
                abstention_source = "validation_fallback"
            elif response["sufficiency"] == "insufficient":
                abstention_source = "model"
            if self.enable_abstention:
                response = self.abstention_policy.enforce_model_response(
                    response, language
                )
        else:
            request = build_grounded_answer_request(
                query,
                evidence,
                question_language=language,
            )
            base_messages = request["messages"]
            messages = list(base_messages)
            max_attempts = self.generation_config["validation_attempts"]
            llm_load_started = time.perf_counter()
            llm = self._get_llm()
            generation_seconds += time.perf_counter() - llm_load_started
            if getattr(llm, "device", None) == "cpu":
                max_attempts = min(
                    max_attempts,
                    self.generation_config.get(
                        "cpu_validation_attempts", max_attempts
                    ),
                )
            llm_loaded = True
            for attempt_number in range(1, max_attempts + 1):
                raw_response = ""
                attempt_started = time.perf_counter()
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
                finally:
                    generation_seconds += time.perf_counter() - attempt_started

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
            if self.answer_strategy == "evidence_first":
                stages.extend(["sentence_selection"])
                stages.append(
                    "local_translation" if language == "ko" else "extractive_answer"
                )
                stages.append("answer_validation")
            else:
                stages.extend(["local_llm", "answer_validation"])
        if self.enable_abstention:
            stages.append("abstention_enforcement")

        result = {
            "schema_version": 1,
            "roadmap_day": (
                41
                if self.answer_strategy == "evidence_first"
                else (31 if self.enable_abstention else 30)
            ),
            "roadmap_days": (
                [30, 31, 41]
                if self.answer_strategy == "evidence_first"
                else ([30, 31] if self.enable_abstention else [30])
            ),
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
                "answer_strategy": self.answer_strategy,
                "answer_generation_config": (
                    str(
                        self.answer_generation_config_path.relative_to(PROJECT_ROOT)
                    ).replace("\\", "/")
                    if self.answer_generation_config_path
                    else None
                ),
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
                "retrieval_seconds": round(retrieval_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "runtime_seconds": round(time.perf_counter() - started, 3),
            },
        }
        if include_debug:
            result["debug"] = {"attempts": attempts}
        return result
