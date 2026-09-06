"""Evaluate cached multilingual dense retrieval with page-level metrics."""

import argparse
import json
from pathlib import Path

from dense_retrieval import (
    DEFAULT_MODEL,
    encode_query,
    load_model,
    load_or_create_embeddings,
    rank_chunks,
)
from retrieval_common import (
    chunks_by_paper,
    configure_utf8_stdout,
    evaluate_page_retriever,
    load_chunks,
    load_evaluation_data,
    metrics_from_results,
    rank_pages,
)


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks()
    grouped_chunks = chunks_by_paper(chunks)
    model = load_model(args.model)
    if args.no_cache:
        embeddings, cache_hit = load_or_create_embeddings(
            model, chunks, args.model, args.batch_size, cache_path=None
        )
    else:
        embeddings, cache_hit = load_or_create_embeddings(model, chunks, args.model, args.batch_size)
    grouped_embeddings = {
        paper_id: embeddings[[
            index for index, chunk in enumerate(chunks)
            if chunk["paper_id"] == paper_id
        ]]
        for paper_id in grouped_chunks
    }
    questions, gold_by_question = load_evaluation_data()

    def retrieve_pages(question, top_k):
        paper_id = question["paper_id"]
        ranked_chunks = rank_chunks(
            grouped_chunks[paper_id],
            grouped_embeddings[paper_id],
            encode_query(model, question["question"]),
            top_k=None,
        )
        return rank_pages(ranked_chunks, top_k)

    results = evaluate_page_retriever(questions, gold_by_question, retrieve_pages, args.top_k)
    metrics = {
        "retriever": "dense",
        "model": args.model,
        "embedding_cache": "hit" if cache_hit else "created",
        **metrics_from_results(results),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
