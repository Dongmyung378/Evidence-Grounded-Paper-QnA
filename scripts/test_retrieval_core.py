"""Dependency-free checks for retrieval ranking, RRF, and query preprocessing."""

import json
from tempfile import TemporaryDirectory

from bm25_retrieval import tokenize
from hybrid_retrieval import reciprocal_rank_fusion
from retrieval_common import metrics_from_results, rank_pages, save_search_results


def item(chunk_id, page, score, rank):
    return {
        "rank": rank,
        "score": score,
        "chunk": {
            "chunk_id": chunk_id,
            "paper_id": "paper-test",
            "page": page,
            "section": "Test",
            "text": chunk_id,
        },
    }


def main():
    tokens = tokenize("BERT-base-uncased와 GPT-4o는 2024년에 비교되었다.")
    for required in ("bert-base-uncased", "bert", "base", "uncased", "gpt-4o", "gpt", "4o", "2024"):
        assert required in tokens, f"Missing token: {required}"

    pages = rank_pages([item("c-2", 2, 0.8, 2), item("c-1", 1, 0.8, 1), item("c-3", 1, 0.5, 3)])
    assert [page["chunk"]["page"] for page in pages] == [1, 2]

    fused = reciprocal_rank_fusion({
        "bm25": [item("c-1", 1, 12.0, 1), item("c-2", 2, 9.0, 2)],
        "dense": [item("c-2", 2, 0.9, 1), item("c-1", 1, 0.8, 2)],
    })
    assert [row["chunk"]["chunk_id"] for row in fused] == ["c-1", "c-2"]
    assert fused[0]["source_ranks"] == {"bm25": 1, "dense": 2}
    assert set(fused[0]["source_contributions"]) == {"bm25", "dense"}

    dense_weighted = reciprocal_rank_fusion({
        "bm25": [item("c-1", 1, 12.0, 1), item("c-2", 2, 9.0, 2)],
        "dense": [item("c-2", 2, 0.9, 1), item("c-1", 1, 0.8, 2)],
    }, weights={"bm25": 0.8, "dense": 1.2})
    assert dense_weighted[0]["chunk"]["chunk_id"] == "c-2"

    metrics = metrics_from_results([
        {"language": "en", "gold_page": 1, "predicted_pages": [1, 2], "gold_rank": 1},
        {"language": "ko", "gold_page": 2, "predicted_pages": [1, 2], "gold_rank": 2},
    ])
    assert metrics["recall_at_1"] == 0.5
    assert metrics["mrr"] == 0.75

    with TemporaryDirectory() as directory:
        output = f"{directory}/search.json"
        save_search_results(output, "hybrid_rrf", "test query", fused[:1], {"rrf_k": 60})
        saved = json.loads(open(output, encoding="utf-8").read())
        assert saved["results"][0]["score"] == fused[0]["score"]
        assert saved["results"][0]["source_ranks"] == {"bm25": 1, "dense": 2}
    print("Retrieval core validation passed")


if __name__ == "__main__":
    main()
