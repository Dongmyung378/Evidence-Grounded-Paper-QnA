"""Search one paper with BM25 + multilingual dense RRF fusion."""

import argparse
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
from retrieval_common import configure_utf8_stdout, load_chunks, save_search_results


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="Optional JSON path for scored Top-K evidence.")
    args = parser.parse_args()

    all_chunks = load_chunks()
    indices = [index for index, chunk in enumerate(all_chunks) if chunk["paper_id"] == args.paper_id]
    chunks = [all_chunks[index] for index in indices]
    if not chunks:
        raise ValueError("검색할 청크가 없습니다.")

    model = load_model(args.model)
    all_embeddings, cache_hit = load_or_create_embeddings(model, all_chunks, args.model, args.batch_size)
    embeddings = all_embeddings[indices]
    bm25 = build_index(chunks)
    fused = reciprocal_rank_fusion({
        "bm25": rank_bm25_chunks(bm25, chunks, args.query, args.candidate_k),
        "dense": rank_dense_chunks(chunks, embeddings, encode_query(model, args.query), args.candidate_k),
    }, args.rrf_k, args.top_k, {"bm25": args.bm25_weight, "dense": args.dense_weight})

    print(f"Embedding cache: {'hit' if cache_hit else 'created'}")
    for result in fused:
        chunk = result["chunk"]
        print(f"\n[{result['rank']}] rrf_score={result['score']:.6f} sources={result['source_ranks']}")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"page: {chunk['page']} | section: {chunk.get('section')}")
        print(f"text: {chunk['text'][:800]}")
    if args.output:
        save_search_results(
            args.output,
            "hybrid_rrf",
            args.query,
            fused,
            {
                "paper_id": args.paper_id,
                "model": args.model,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "fusion_weights": {
                    "bm25": args.bm25_weight,
                    "dense": args.dense_weight,
                },
                "embedding_cache": "hit" if cache_hit else "created",
            },
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
