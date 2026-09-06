"""Compare BM25, dense, and RRF hybrid retrieval on the fixed evaluation set."""

import argparse
import csv
import json
from pathlib import Path

from bm25_retrieval import build_index, rank_chunks as rank_bm25_chunks
from dense_retrieval import (
    DEFAULT_MODEL,
    encode_query,
    load_model,
    load_or_create_embeddings,
    rank_chunks as rank_dense_chunks,
)
from hybrid_retrieval import reciprocal_rank_fusion
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


DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "retrieval_comparison.json"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "retrieval_comparison.csv"


def write_comparison_csv(path, metrics_by_retriever):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["retriever", "scope", "questions", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for retriever, metrics in metrics_by_retriever.items():
            for scope, values in [("all", metrics), *metrics["by_language"].items()]:
                writer.writerow({
                    "retriever": retriever,
                    "scope": scope,
                    "questions": values["questions"],
                    "recall_at_1": f"{values['recall_at_1']:.3f}",
                    "recall_at_3": f"{values['recall_at_3']:.3f}",
                    "recall_at_5": f"{values['recall_at_5']:.3f}",
                    "recall_at_10": f"{values['recall_at_10']:.3f}",
                    "mrr": f"{values['mrr']:.6f}",
                })


def evaluate_all(
    model_name=DEFAULT_MODEL,
    batch_size=32,
    candidate_k=20,
    rrf_k=60,
    top_k=10,
    bm25_weight=1.0,
    dense_weight=1.0,
):
    chunks = load_chunks()
    grouped_chunks = chunks_by_paper(chunks)
    bm25_indexes = {paper_id: build_index(paper_chunks) for paper_id, paper_chunks in grouped_chunks.items()}

    model = load_model(model_name)
    embeddings, cache_hit = load_or_create_embeddings(model, chunks, model_name, batch_size)
    grouped_embeddings = {
        paper_id: embeddings[[
            index for index, chunk in enumerate(chunks)
            if chunk["paper_id"] == paper_id
        ]]
        for paper_id in grouped_chunks
    }
    questions, gold_by_question = load_evaluation_data()
    query_embeddings = {}

    def dense_results(question, result_k=None):
        question_id = question["question_id"]
        if question_id not in query_embeddings:
            query_embeddings[question_id] = encode_query(model, question["question"])
        paper_id = question["paper_id"]
        return rank_dense_chunks(
            grouped_chunks[paper_id],
            grouped_embeddings[paper_id],
            query_embeddings[question_id],
            result_k,
        )

    def bm25_pages(question, page_k):
        paper_id = question["paper_id"]
        return rank_pages(
            rank_bm25_chunks(bm25_indexes[paper_id], grouped_chunks[paper_id], question["question"]),
            page_k,
        )

    def dense_pages(question, page_k):
        return rank_pages(dense_results(question), page_k)

    def hybrid_pages(question, page_k):
        paper_id = question["paper_id"]
        fused = reciprocal_rank_fusion({
            "bm25": rank_bm25_chunks(
                bm25_indexes[paper_id], grouped_chunks[paper_id], question["question"], candidate_k
            ),
            "dense": dense_results(question, candidate_k),
        }, rrf_k, candidate_k * 2, {"bm25": bm25_weight, "dense": dense_weight})
        return rank_pages(fused, page_k)

    results_by_retriever = {
        "bm25": evaluate_page_retriever(questions, gold_by_question, bm25_pages, top_k),
        "dense": evaluate_page_retriever(questions, gold_by_question, dense_pages, top_k),
        "hybrid_rrf": evaluate_page_retriever(questions, gold_by_question, hybrid_pages, top_k),
    }
    metrics_by_retriever = {
        name: metrics_from_results(results)
        for name, results in results_by_retriever.items()
    }
    return {
        "config": {
            "model": model_name,
            "candidate_k": candidate_k,
            "rrf_k": rrf_k,
            "fusion_weights": {
                "bm25": bm25_weight,
                "dense": dense_weight,
            },
            "evaluation_top_k": top_k,
            "embedding_cache": "hit" if cache_hit else "created",
        },
        "metrics": metrics_by_retriever,
        "results": results_by_retriever,
    }


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    args = parser.parse_args()

    comparison = evaluate_all(
        args.model,
        args.batch_size,
        args.candidate_k,
        args.rrf_k,
        args.top_k,
        args.bm25_weight,
        args.dense_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_comparison_csv(args.csv_output, comparison["metrics"])
    print(json.dumps({"config": comparison["config"], "metrics": comparison["metrics"]}, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")
    print(f"Saved: {args.csv_output}")


if __name__ == "__main__":
    main()
