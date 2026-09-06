"""Search one paper and print the Day 24 UI-ready Top-5 evidence."""

import argparse
import json
from pathlib import Path

from production_retrieval import ProductionRetrieval
from retrieval_common import configure_utf8_stdout


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    production = ProductionRetrieval()
    result = production.run(args.query, args.paper_id)

    print(f"Production config: {result['production']['config_fingerprint'][:12]}")
    print(f"Embedding cache: {'hit' if production.embedding_cache_hit else 'created'}")
    print(
        f"Candidates: {len(result['candidates'])} | "
        f"Evidence: {len(result['evidence'])}"
    )
    for item in result["evidence"]:
        print(
            f"\n[{item['rank']}] {item['locator']['page_label']} | "
            f"section: {item['section']} | chunk_id: {item['chunk_id']}"
        )
        print(
            f"reranker={item['scores']['reranker']:.6f} | "
            f"hybrid_rank={item['scores']['pre_rerank_rank']}"
        )
        print(item["text"][:800])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
