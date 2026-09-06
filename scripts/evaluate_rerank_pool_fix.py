"""Day 27 controlled fix: guarantee front-matter coverage in Hybrid Top-20."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from bm25_retrieval import build_index, rank_chunks as rank_bm25_chunks
from dense_retrieval import (
    DEFAULT_MODEL,
    encode_query,
    load_model,
    load_or_create_embeddings,
    rank_chunks as rank_dense_chunks,
)
from hybrid_retrieval import build_rerank_pool, ensure_page_candidate
from reranker import DEFAULT_RERANKER_MODEL, load_reranker, rerank_chunks
from retrieval_common import (
    PROJECT_ROOT,
    chunks_by_paper,
    configure_utf8_stdout,
    evaluate_page_retriever,
    load_chunks,
    load_evaluation_data,
    metrics_from_results,
    rank_pages,
)


DEFAULT_BASELINE = PROJECT_ROOT / "data" / "evaluation" / "reranker_metrics.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "day27_error_fix.json"
METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr")


def delta(after, before):
    return {name: round(after[name] - before[name], 12) for name in METRICS}


def hit(rank, k=5):
    return rank is not None and rank <= k


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--source-k", type=int, default=20)
    parser.add_argument("--final-candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    started = perf_counter()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    before_metrics = baseline["metrics"]["hybrid_reranker"]
    before_rows = {
        row["question_id"]: row for row in baseline["results"]["hybrid_reranker"]
    }
    chunks = load_chunks()
    grouped_chunks = chunks_by_paper(chunks)
    indices_by_paper = {
        paper_id: [index for index, chunk in enumerate(chunks) if chunk["paper_id"] == paper_id]
        for paper_id in grouped_chunks
    }
    bm25_indexes = {
        paper_id: build_index(paper_chunks)
        for paper_id, paper_chunks in grouped_chunks.items()
    }
    embedding_model = load_model(args.embedding_model)
    embeddings, cache_hit = load_or_create_embeddings(
        embedding_model, chunks, args.embedding_model, args.embedding_batch_size
    )
    grouped_embeddings = {
        paper_id: embeddings[indices] for paper_id, indices in indices_by_paper.items()
    }
    reranker = load_reranker(args.reranker_model, max_length=args.max_length)
    questions, gold_by_question = load_evaluation_data()
    reranked_cache = {}
    coverage_injected = {}

    def retrieve(question, page_k):
        question_id = question["question_id"]
        if question_id not in reranked_cache:
            paper_id = question["paper_id"]
            bm25 = rank_bm25_chunks(
                bm25_indexes[paper_id], grouped_chunks[paper_id], question["question"], args.source_k
            )
            dense = rank_dense_chunks(
                grouped_chunks[paper_id],
                grouped_embeddings[paper_id],
                encode_query(embedding_model, question["question"]),
                args.source_k,
            )
            universe = build_rerank_pool(
                {"bm25": bm25, "dense": dense},
                rrf_k=args.rrf_k,
                candidate_k=args.source_k,
                mode="source_union",
            )
            pool, injected = ensure_page_candidate(
                universe[:args.final_candidate_k],
                universe,
                page=1,
                top_k=args.final_candidate_k,
            )
            coverage_injected[question_id] = injected
            reranked_cache[question_id] = rerank_chunks(
                reranker,
                question["question"],
                pool,
                top_k=args.final_candidate_k,
                batch_size=args.reranker_batch_size,
            )
        return rank_pages(reranked_cache[question_id], page_k)

    after_results = evaluate_page_retriever(
        questions, gold_by_question, retrieve, top_k=10
    )
    after_metrics = metrics_from_results(after_results)
    after_rows = {row["question_id"]: row for row in after_results}
    recovered = []
    lost = []
    changed = []
    for question_id in sorted(before_rows):
        before = before_rows[question_id]
        after = after_rows[question_id]
        if before["gold_rank"] != after["gold_rank"]:
            changed.append({
                "question_id": question_id,
                "paper_id": before["paper_id"],
                "language": before["language"],
                "gold_page": before["gold_page"],
                "before_gold_rank": before["gold_rank"],
                "after_gold_rank": after["gold_rank"],
                "before_pages": before["predicted_pages"],
                "after_pages": after["predicted_pages"],
                "front_matter_candidate_injected": coverage_injected[question_id],
            })
        if not hit(before["gold_rank"]) and hit(after["gold_rank"]):
            recovered.append(question_id)
        elif hit(before["gold_rank"]) and not hit(after["gold_rank"]):
            lost.append(question_id)

    metric_delta = delta(after_metrics, before_metrics)
    artifact = {
        "schema_version": 1,
        "roadmap_day": 27,
        "error": {
            "category": "candidate starvation before reranking",
            "largest_case": "q-005-ko",
            "diagnosis": "The verified page was present in dense page Top-10 but disappeared when chunk-level RRF was truncated to 20 before Cross-Encoder scoring.",
        },
        "controlled_fix": {
            "before": "RRF Top-20 -> rerank 20 -> final Top-20",
            "after": "RRF Top-20 with one page-1 candidate guaranteed when absent -> rerank 20 -> final Top-20",
            "unchanged": [
                "verified-40 questions and gold pages",
                "906 chunks",
                "BM25 and dense models",
                "equal RRF weights and rrf_k=60",
                "pretrained Cross-Encoder",
                "reranker input count=20",
                "final candidate count=20",
            ],
            "question_specific_rules": False,
        },
        "evaluation_set": {"name": "verified-40", "questions": 40, "languages": {"en": 20, "ko": 20}},
        "coverage_guard": {
            "page": 1,
            "reason": "English papers normally place the title, abstract, or contribution summary in front matter.",
            "questions_injected": sum(coverage_injected.values()),
            "questions_unchanged": len(coverage_injected) - sum(coverage_injected.values()),
        },
        "before": {"name": "hybrid_top20_reranker", "metrics": before_metrics},
        "after": {"name": "front_matter_guard_reranker_top20", "metrics": after_metrics},
        "delta_after_minus_before": {
            "all": metric_delta,
            "by_language": {
                language: delta(after_metrics["by_language"][language], before_metrics["by_language"][language])
                for language in ("en", "ko")
            },
        },
        "cases": {
            "recovered_at_5": recovered,
            "lost_at_5": lost,
            "changed_ranks": changed,
        },
        "decision": {
            "adopt_for_day28_gate": metric_delta["recall_at_5"] > 0 and not lost,
            "reason": "Adopt only if Recall@5 improves without losing an existing Top-5 hit; Day 28 still freezes the production combination.",
        },
        "runtime": {
            "seconds": round(perf_counter() - started, 3),
            "embedding_cache": "hit" if cache_hit else "created",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "before": {name: before_metrics[name] for name in METRICS},
        "after": {name: after_metrics[name] for name in METRICS},
        "delta": metric_delta,
        "recovered_at_5": recovered,
        "lost_at_5": lost,
        "adopt_for_day28_gate": artifact["decision"]["adopt_for_day28_gate"],
        "runtime_seconds": artifact["runtime"]["seconds"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
