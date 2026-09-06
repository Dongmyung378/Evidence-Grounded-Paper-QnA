"""Search processed chunks with a multilingual embedding model."""

import argparse
from pathlib import Path

from dense_retrieval import (
    DEFAULT_MODEL,
    encode_query,
    load_chunks,
    load_or_create_embeddings,
    load_model,
    rank_chunks,
)
from retrieval_common import configure_utf8_stdout, save_search_results


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON path for scored Top-K evidence.")
    args = parser.parse_args()

    all_chunks = load_chunks()
    chunks = [chunk for chunk in all_chunks if not args.paper_id or chunk["paper_id"] == args.paper_id]
    if not chunks:
        raise ValueError("검색할 청크가 없습니다.")
    model = load_model(args.model)
    if args.no_cache:
        all_embeddings, cache_hit = load_or_create_embeddings(
            model, all_chunks, args.model, args.batch_size, cache_path=None
        )
    else:
        all_embeddings, cache_hit = load_or_create_embeddings(
            model, all_chunks, args.model, args.batch_size
        )
    indices = [
        index for index, chunk in enumerate(all_chunks)
        if not args.paper_id or chunk["paper_id"] == args.paper_id
    ]
    embeddings = all_embeddings[indices]
    print(f"Embedding cache: {'hit' if cache_hit else 'created'}")
    ranked = rank_chunks(
        chunks,
        embeddings,
        encode_query(model, args.query),
        args.top_k,
    )
    for result in ranked:
        chunk = result["chunk"]
        print(f"\n[{result['rank']}] score={result['score']:.4f}")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"paper_id: {chunk['paper_id']}")
        print(f"page: {chunk['page']}")
        print(f"section: {chunk.get('section')}")
        print(f"text: {chunk['text'][:800]}")
    if args.output:
        save_search_results(
            args.output,
            "dense",
            args.query,
            ranked,
            {
                "paper_id": args.paper_id,
                "model": args.model,
                "top_k": args.top_k,
                "embedding_cache": "hit" if cache_hit else "created",
            },
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
