"""Search one paper with a reusable BM25 index and shared preprocessing."""

import argparse
from pathlib import Path

from bm25_retrieval import build_index, rank_chunks
from retrieval_common import configure_utf8_stdout, load_chunks, save_search_results


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON path for scored Top-K evidence.")
    args = parser.parse_args()

    chunks = load_chunks(args.paper_id)
    if not chunks:
        raise ValueError("검색할 청크가 없습니다.")
    results = rank_chunks(build_index(chunks), chunks, args.query, args.top_k)
    for item in results:
        chunk = item["chunk"]
        print(f"\n[{item['rank']}] score={item['score']:.4f}")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"paper_id: {chunk['paper_id']}")
        print(f"page: {chunk['page']} | section: {chunk.get('section')}")
        print(f"text: {chunk['text'][:800]}")
    if args.output:
        save_search_results(
            args.output,
            "bm25",
            args.query,
            results,
            {"paper_id": args.paper_id, "top_k": args.top_k},
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
