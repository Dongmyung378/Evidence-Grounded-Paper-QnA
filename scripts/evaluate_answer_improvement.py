"""Compare a compact prompt on the exact saved Day 32 evidence, without Gold."""

import json
import argparse
import os
import time

from answer_quality import build_quality_request, validate_informative_answer, PROMPT_VERSION
from grounded_answer_contract import parse_and_validate_answer, canonical_abstention_payload
from local_llm import LocalTransformersLLM
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import set_seed
    set_seed(378)
    baseline = json.loads((PROJECT_ROOT / "data/evaluation/day32_qna_outputs.json").read_text(encoding="utf-8"))
    llm = (LocalTransformersLLM(PROJECT_ROOT / 'config/generation_candidate.json', local_files_only=True)
           if args.candidate else LocalTransformersLLM(local_files_only=True))
    rows = []
    runtime = None
    if args.runtime:
        from qna_pipeline import GroundedQAPipeline
        class SavedRetrieval:
            def run(self, query, paper_id):
                item = next(x for x in baseline['results'] if x['query'] == query and x['paper_id'] == paper_id)
                signals = item['pipeline']['abstention']['decision']['signals']
                return {'evidence': item['retrieved_evidence'],
                        'candidates': [{'reranker_score': signals['top_reranker_score']}] * signals['candidate_count'],
                        'production': {'config_fingerprint': item['pipeline']['retrieval_config_fingerprint']}}
        runtime = GroundedQAPipeline(retrieval=SavedRetrieval(), llm=llm,
                                    abstention_config_path=PROJECT_ROOT / 'config/abstention.json')
    for item in baseline["results"]:
        started = time.perf_counter()
        attempts = []
        if runtime is not None:
            run = runtime.ask(item['query'], item['paper_id'], item['question_language'], include_debug=True)
            response = run['response']
            attempts = run['debug']['attempts']
        elif item["pipeline"]["llm_skipped"]:
            response = item["response"]
        else:
            request = build_quality_request(item["query"], item["retrieved_evidence"], item["question_language"])
            messages = list(request["messages"])
            response = None
            for _ in range(3):
                raw = llm.generate(messages, request["response_schema"])
                try:
                    response = validate_informative_answer(parse_and_validate_answer(raw, item["retrieved_evidence"], item["question_language"]), item["query"])
                    attempts.append({"raw": raw, "valid": True})
                    break
                except ValueError as exc:
                    attempts.append({"raw": raw, "valid": False, "error": str(exc)})
                    messages = request["messages"] + [{"role": "assistant", "content": raw},
                        {"role": "user", "content": "Correct your response and return the required JSON. " + str(exc)}]
            if response is None:
                response = canonical_abstention_payload(item["question_language"], "Output validation failed.")
        rows.append({"question_id": item["question_id"], "response": response,
                     "attempts": attempts, "seconds": round(time.perf_counter()-started, 3)})
        print(item["question_id"], response["answer"], flush=True)
    output = {"seed": 378, "prompt_version": 'legacy-with-nonanswer-guard' if args.runtime else PROMPT_VERSION,
              "method": "same model, saved retrieved evidence, unchanged pre-generation refusal decisions",
              "model": llm.metadata(), "results": rows}
    name = 'answer_runtime_outputs.json' if args.runtime else ('answer_candidate_outputs.json' if args.candidate else 'answer_improvement_outputs.json')
    (PROJECT_ROOT / 'data/evaluation' / name).write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
