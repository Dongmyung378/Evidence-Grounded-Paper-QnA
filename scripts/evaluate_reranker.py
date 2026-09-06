"""Evaluate pretrained Cross-Encoder reranking on Hybrid Top-20 chunks."""

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
from hybrid_retrieval import reciprocal_rank_fusion
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


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "reranker_metrics.json"
METRIC_NAMES = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr")


def metric_delta(after, before):
    return {
        name: round(after[name] - before[name], 12)
        for name in METRIC_NAMES
    }


def evaluate(
    embedding_model_name=DEFAULT_MODEL,
    reranker_model_name=DEFAULT_RERANKER_MODEL,
    embedding_batch_size=32,
    reranker_batch_size=8,
    candidate_k=20,
    rrf_k=60,
    top_k=10,
    max_length=512,
):
    started = perf_counter()
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

    embedding_model = load_model(embedding_model_name)
    embeddings, cache_hit = load_or_create_embeddings(
        embedding_model,
        chunks,
        embedding_model_name,
        embedding_batch_size,
    )
    grouped_embeddings = {
        paper_id: embeddings[indices]
        for paper_id, indices in indices_by_paper.items()
    }
    cross_encoder = load_reranker(reranker_model_name, max_length=max_length)
    questions, gold_by_question = load_evaluation_data()
    query_embeddings = {}
    candidate_cache = {}
    reranked_cache = {}

    def hybrid_candidates(question):
        question_id = question["question_id"]
        if question_id in candidate_cache:
            return candidate_cache[question_id]
        if question_id not in query_embeddings:
            query_embeddings[question_id] = encode_query(embedding_model, question["question"])
        paper_id = question["paper_id"]
        candidates = reciprocal_rank_fusion({
            "bm25": rank_bm25_chunks(
                bm25_indexes[paper_id],
                grouped_chunks[paper_id],
                question["question"],
                candidate_k,
            ),
            "dense": rank_dense_chunks(
                grouped_chunks[paper_id],
                grouped_embeddings[paper_id],
                query_embeddings[question_id],
                candidate_k,
            ),
        }, rrf_k=rrf_k, top_k=candidate_k, weights={"bm25": 1.0, "dense": 1.0})
        candidate_cache[question_id] = candidates
        return candidates

    def hybrid_pages(question, page_k):
        return rank_pages(hybrid_candidates(question), page_k)

    def reranked_pages(question, page_k):
        question_id = question["question_id"]
        if question_id not in reranked_cache:
            reranked_cache[question_id] = rerank_chunks(
                cross_encoder,
                question["question"],
                hybrid_candidates(question),
                top_k=candidate_k,
                batch_size=reranker_batch_size,
            )
        return rank_pages(reranked_cache[question_id], page_k)

    results = {
        "hybrid_top20": evaluate_page_retriever(
            questions, gold_by_question, hybrid_pages, top_k
        ),
        "hybrid_reranker": evaluate_page_retriever(
            questions, gold_by_question, reranked_pages, top_k
        ),
    }
    metrics = {
        name: metrics_from_results(rows)
        for name, rows in results.items()
    }
    return {
        "schema_version": 1,
        "roadmap_day": 22,
        "config": {
            "embedding_model": embedding_model_name,
            "reranker_model": reranker_model_name,
            "candidate_source": "equal-weight RRF over BM25 Top-20 and dense Top-20 chunks",
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "fusion_weights": {"bm25": 1.0, "dense": 1.0},
            "evaluation_top_k": top_k,
            "max_length": max_length,
            "embedding_batch_size": embedding_batch_size,
            "reranker_batch_size": reranker_batch_size,
            "embedding_cache": "hit" if cache_hit else "created",
            "ranking_unit": "unique PDF page using the highest-scoring chunk per page",
        },
        "evaluation_set": {
            "name": "verified-40",
            "questions": len(questions),
            "languages": {"en": 20, "ko": 20},
            "papers": len(grouped_chunks),
            "chunks": len(chunks),
        },
        "metrics": metrics,
        "delta_reranker_minus_hybrid_top20": {
            "all": metric_delta(metrics["hybrid_reranker"], metrics["hybrid_top20"]),
            "by_language": {
                language: metric_delta(
                    metrics["hybrid_reranker"]["by_language"][language],
                    metrics["hybrid_top20"]["by_language"][language],
                )
                for language in ("en", "ko")
            },
        },
        "results": results,
        "runtime_seconds": round(perf_counter() - started, 3),
    }


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = evaluate(
        args.embedding_model,
        args.reranker_model,
        args.embedding_batch_size,
        args.reranker_batch_size,
        args.candidate_k,
        args.rrf_k,
        args.top_k,
        args.max_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "config": artifact["config"],
        "metrics": artifact["metrics"],
        "delta": artifact["delta_reranker_minus_hybrid_top20"],
        "runtime_seconds": artifact["runtime_seconds"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
