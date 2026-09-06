"""Evaluate BM25 with the same page-level metrics used by dense retrieval."""

import argparse
import json
from pathlib import Path

from bm25_retrieval import build_index, rank_chunks
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
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    grouped_chunks = chunks_by_paper(load_chunks())
    indexes = {
        paper_id: build_index(paper_chunks)
        for paper_id, paper_chunks in grouped_chunks.items()
    }
    questions, gold_by_question = load_evaluation_data()

    def retrieve_pages(question, top_k):
        paper_id = question["paper_id"]
        ranked_chunks = rank_chunks(indexes[paper_id], grouped_chunks[paper_id], question["question"])
        return rank_pages(ranked_chunks, top_k)

    results = evaluate_page_retriever(questions, gold_by_question, retrieve_pages, args.top_k)
    metrics = {"retriever": "bm25", **metrics_from_results(results)}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
