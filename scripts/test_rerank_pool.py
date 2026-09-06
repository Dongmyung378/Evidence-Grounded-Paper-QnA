"""Dependency-light checks for the Day 27 source-union candidate fix."""

from hybrid_retrieval import build_rerank_pool, ensure_page_candidate


def item(rank, chunk_id, page=2):
    return {
        "rank": rank,
        "score": 1.0 / rank,
        "chunk": {"chunk_id": chunk_id, "text": chunk_id, "page": page},
    }


def main():
    result_sets = {
        "bm25": [item(1, "shared"), item(2, "bm25-only")],
        "dense": [item(1, "shared"), item(2, "dense-only")],
    }
    baseline = build_rerank_pool(result_sets, candidate_k=2, mode="top20")
    expanded = build_rerank_pool(result_sets, candidate_k=2, mode="source_union")
    assert len(baseline) == 2
    assert len(expanded) == 3
    assert {row["chunk"]["chunk_id"] for row in expanded} == {
        "shared", "bm25-only", "dense-only"
    }
    assert expanded[0]["source_ranks"] == {"bm25": 1, "dense": 1}
    guarded, injected = ensure_page_candidate(
        [item(1, "p2-a"), item(2, "p2-b")],
        [item(1, "p2-a"), item(2, "p2-b"), item(3, "front", page=1)],
        page=1,
        top_k=2,
    )
    assert injected
    assert [row["chunk"]["chunk_id"] for row in guarded] == ["p2-a", "front"]
    unchanged, injected = ensure_page_candidate(guarded, guarded, page=1, top_k=2)
    assert not injected and unchanged == guarded
    print("Rerank pool tests passed")
    print("top20=2 source_union=3 page1_guard=passed")


if __name__ == "__main__":
    main()
