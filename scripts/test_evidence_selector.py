"""Dependency-light checks for Day 23 evidence selection."""

from evidence_selector import select_evidence


def candidate(rank, chunk_id, page, text):
    return {
        "rank": rank,
        "score": 10.0 - rank,
        "reranker_score": 10.0 - rank,
        "pre_rerank_rank": rank + 1,
        "retrieval_score": 0.03 - rank / 1000,
        "source_ranks": {"bm25": rank, "dense": rank + 2},
        "chunk": {
            "chunk_id": chunk_id,
            "paper_id": "paper-test",
            "page": page,
            "source_page_id": f"paper-test-p{page:03d}",
            "section": "Methods",
            "section_index": 2,
            "chunk_index": rank,
            "text": text,
            "char_count": len(text),
        },
    }


def main():
    base = "alpha beta gamma delta epsilon zeta eta theta"
    candidates = [
        candidate(1, "c1", 1, base),
        candidate(2, "c1", 1, "same chunk id but another payload"),
        candidate(3, "c2", 1, base),
        candidate(4, "c3", 1, base + " iota"),
        candidate(5, "c4", 1, "unique page one evidence"),
        candidate(6, "c5", 1, "third unique page one evidence"),
        candidate(7, "c6", 2, "page two evidence"),
        candidate(8, "c7", 3, "page three evidence"),
        candidate(9, "c8", 4, "page four evidence"),
    ]
    evidence, summary = select_evidence(candidates, top_k=5, max_per_page=2)

    assert [item["chunk_id"] for item in evidence] == ["c1", "c4", "c6", "c7", "c8"]
    assert summary["input_candidates"] == 9
    assert summary["selected"] == 5
    assert summary["skipped"] == {
        "duplicate_chunk_id": 1,
        "duplicate_text": 1,
        "near_duplicate": 1,
        "page_cap": 1,
    }
    first = evidence[0]
    assert first["evidence_id"] == "ev-01-c1"
    assert first["locator"] == {
        "page_label": "p. 1",
        "section": "Methods",
        "chunk_id": "c1",
    }
    assert first["scores"]["source_ranks"] == {"bm25": 1, "dense": 3}
    print("Evidence selector tests passed")
    print("rules=chunk_id+exact_text+near_duplicate+max_2_per_page")
    print("selected=5")


if __name__ == "__main__":
    main()
