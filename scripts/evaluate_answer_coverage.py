"""현재 검색 결과로 기준선과 정보 범위 확장 답변을 비교한다. Gold는 읽지 않는다."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from abstention_policy import AbstentionPolicy
from answer_coverage import (expand_answer_evidence, generate_coverage_answer, rank_answer_sentences,
                             coverage_decision, load_coverage_config, select_problem_overview)
from grounded_generation import generate_evidence_first_answer
from local_translation import LocalEnglishKoreanTranslator
from production_retrieval import ProductionRetrieval
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout, load_jsonl


CACHE = PROJECT_ROOT / ".tmp_answer_coverage" / "retrieval_sentences.json"
OUTPUT = PROJECT_ROOT / "data/evaluation/answer_coverage_outputs.json"
SOURCES = ["scripts/answer_coverage.py", "scripts/production_retrieval.py",
           "config/answer_coverage.json",
           "scripts/candidate_evidence_pipeline.py", "config/model_lock.json",
           "config/production_retrieval.json", "data/processed/chunks.jsonl",
           "data/evaluation/questions.jsonl", "data/evaluation/abstention_questions.jsonl"]


def hashes(paths):
    return {p: hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in paths}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import set_seed
    import torch
    torch.set_num_threads(8)
    set_seed(378)
    questions = load_jsonl(PROJECT_ROOT/"data/evaluation/questions.jsonl")
    probes = load_jsonl(PROJECT_ROOT/"data/evaluation/abstention_questions.jsonl")
    for p in probes:
        p["question_id"] = p["case_id"]
    policy = AbstentionPolicy(PROJECT_ROOT/"config/abstention_candidate.json")
    coverage_config = load_coverage_config()
    if args.rescore:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        unchanged = [p for p in SOURCES if p != "scripts/answer_coverage.py"]
        if {p: cache["source_hashes"][p] for p in unchanged} != hashes(unchanged):
            raise ValueError("Retrieval cache inputs changed")
        from model_lock import locked_model
        from reranker import load_reranker
        model = locked_model("reranker")
        scorer = load_reranker(model["name"], local_files_only=True, device="cpu", revision=model["revision"])
        chunks = load_jsonl(PROJECT_ROOT/"data/processed/chunks.jsonl")
        for row in cache["rows"]:
            row["expanded_evidence"] = expand_answer_evidence(row["retrieval"], chunks)
            row["sentences"] = rank_answer_sentences(row["question"]["question"], row["expanded_evidence"], scorer)
            print("rescored", row["question"]["question_id"], flush=True)
        cache["source_hashes"] = hashes(SOURCES)
        write_json(CACHE, cache)
    elif args.reuse:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        if cache["source_hashes"] != hashes(SOURCES):
            raise ValueError("Cached sentence scores do not match the implementation and inputs")
    else:
        production = ProductionRetrieval(local_files_only=True,
            embedding_cache_path=PROJECT_ROOT/"data/processed/frozen_dense_embeddings.npz",
            embedding_device="cpu", reranker_device="cpu")
        rows = []
        for question in questions + probes:
            start = time.perf_counter()
            result = production.run(question["question"], question["paper_id"])
            decision = policy.evaluate_retrieval(result)
            candidate_decision, overview = coverage_decision(result, decision, production.pipeline.chunks, coverage_config)
            evidence = expand_answer_evidence(result, production.pipeline.chunks)
            selection_start = time.perf_counter()
            ranked = [] if candidate_decision["abstain"] or overview else rank_answer_sentences(question["question"], evidence, production.pipeline.reranker)
            rows.append({"question": question, "retrieval": result, "decision": decision,
                         "candidate_decision": candidate_decision, "overview": overview,
                         "sentence_ranking_seconds": time.perf_counter()-selection_start,
                         "expanded_evidence": evidence, "sentences": ranked,
                         "preparation_seconds": time.perf_counter()-start})
            print(question["question_id"], "refused="+str(decision["abstain"]),
                  "sentences="+str(len(ranked)), flush=True)
        cache = {"source_hashes": hashes(SOURCES), "rows": rows}
        write_json(CACHE, cache)
    if args.prepare_only:
        return
    from model_lock import locked_model
    from reranker import load_reranker
    model = locked_model("reranker")
    selector = load_reranker(model["name"], local_files_only=True, device="cpu", revision=model["revision"])
    translator = LocalEnglishKoreanTranslator(local_files_only=True)
    manifest = json.loads((PROJECT_ROOT/"data/evaluation/day32_review_manifest.json").read_text(encoding="utf-8"))
    selected_ids = {q["question_id"] for q in manifest["questions"]}
    results = []
    for row in cache["rows"]:
        question = row["question"]
        if not args.all_questions and question["question_id"] not in selected_ids:
            continue
        outputs = {}
        for mode in ("baseline", "candidate"):
            start = time.perf_counter()
            decision = row["decision"] if mode == "baseline" else row["candidate_decision"]
            if decision["abstain"]:
                generated = {"response": policy.refusal_payload(question["question_language"], decision["reason_code"]),
                             "selected_sentences": [], "fallback_used": False, "attempts": []}
            elif mode == "baseline":
                generated = generate_evidence_first_answer(translator, selector, question["question"],
                    row["retrieval"]["evidence"], question["question_language"])
            else:
                generated = generate_coverage_answer(translator, selector, question["question"],
                    row["overview"] or row["expanded_evidence"], records=row["sentences"],
                    selected=select_problem_overview(row["overview"]) if row["overview"] else None,
                    coverage_config=coverage_config)
            generated["answer_seconds"] = time.perf_counter()-start
            outputs[mode] = generated
        results.append({"question": question, "decision": row["decision"],
                        "candidate_decision": row["candidate_decision"],
                        "sentence_ranking_seconds": row["sentence_ranking_seconds"],
                        "retrieval_pages": list(dict.fromkeys(c["page"] for c in row["retrieval"]["candidates"])),
                        "preparation_seconds": row["preparation_seconds"], **outputs})
        print(question["question_id"], outputs["candidate"]["response"]["sufficiency"],
              outputs["candidate"]["response"]["answer"], flush=True)
    write_json(args.output, {"schema_version": 1, "seed": 378, "gold_loaded_during_generation": False,
        "evaluation_scope": "Reused development questions; paired current-retrieval regression, not an independent test.",
        "source_hashes": hashes(SOURCES + ["scripts/evaluate_answer_coverage.py", "scripts/grounded_generation.py",
                                "config/answer_generation.json", "config/translation.json"]),
        "timing_scope": "Host execution; candidate answer time excludes cached sentence ranking, reported separately. This is not Docker HTTP latency.",
        "translator": translator.metadata(), "results": results})


if __name__ == "__main__":
    main()
