"""Search one paper with Hybrid retrieval followed by Cross-Encoder reranking."""

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
from reranker import DEFAULT_RERANKER_MODEL, load_reranker, rerank_chunks
from retrieval_common import configure_utf8_stdout, load_chunks, save_search_results


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    all_chunks = load_chunks()
    indices = [index for index, chunk in enumerate(all_chunks) if chunk["paper_id"] == args.paper_id]
    chunks = [all_chunks[index] for index in indices]
    if not chunks:
        raise ValueError(f"Unknown paper_id or no chunks: {args.paper_id}")

    embedding_model = load_model(args.embedding_model)
    all_embeddings, cache_hit = load_or_create_embeddings(
        embedding_model,
        all_chunks,
        args.embedding_model,
        args.embedding_batch_size,
    )
    embeddings = all_embeddings[indices]
    bm25 = build_index(chunks)
    hybrid = reciprocal_rank_fusion({
        "bm25": rank_bm25_chunks(bm25, chunks, args.query, args.candidate_k),
        "dense": rank_dense_chunks(
            chunks,
            embeddings,
            encode_query(embedding_model, args.query),
            args.candidate_k,
        ),
    }, rrf_k=args.rrf_k, top_k=args.candidate_k, weights={"bm25": 1.0, "dense": 1.0})
    cross_encoder = load_reranker(args.reranker_model, max_length=args.max_length)
    reranked = rerank_chunks(
        cross_encoder,
        args.query,
        hybrid,
        top_k=args.top_k,
        batch_size=args.reranker_batch_size,
    )

    print(f"Embedding cache: {'hit' if cache_hit else 'created'}")
    for result in reranked:
        chunk = result["chunk"]
        print(
            f"\n[{result['rank']}] reranker={result['reranker_score']:.6f} "
            f"hybrid_rank={result['pre_rerank_rank']} hybrid_score={result['retrieval_score']:.6f}"
        )
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"page: {chunk['page']} | section: {chunk.get('section')}")
        print(f"text: {chunk['text'][:800]}")
    if args.output:
        save_search_results(
            args.output,
            "hybrid_cross_encoder",
            args.query,
            reranked,
            {
                "paper_id": args.paper_id,
                "embedding_model": args.embedding_model,
                "reranker_model": args.reranker_model,
                "candidate_k": args.candidate_k,
                "top_k": args.top_k,
                "rrf_k": args.rrf_k,
                "fusion_weights": {"bm25": 1.0, "dense": 1.0},
                "max_length": args.max_length,
                "embedding_cache": "hit" if cache_hit else "created",
            },
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
