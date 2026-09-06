"""Dependency-light checks for Cross-Encoder reranking metadata and order."""

import numpy as np

from reranker import rerank_chunks


class StubReranker:
    def predict(self, pairs, **kwargs):
        assert len(pairs) == 3
        assert kwargs["show_progress_bar"] is False
        return np.array([0.2, 1.4, 1.4], dtype=np.float32)


def candidate(chunk_id, rank, score):
    return {
        "rank": rank,
        "score": score,
        "source_ranks": {"bm25": rank},
        "chunk": {
            "chunk_id": chunk_id,
            "paper_id": "paper-test",
            "page": rank,
            "section": "Test",
            "text": f"passage {chunk_id}",
        },
    }


def main():
    candidates = [
        candidate("c-3", 1, 0.9),
        candidate("c-2", 2, 0.8),
        candidate("c-1", 3, 0.7),
    ]
    ranked = rerank_chunks(StubReranker(), "query", candidates, top_k=2, batch_size=2)
    assert [row["chunk"]["chunk_id"] for row in ranked] == ["c-1", "c-2"]
    assert ranked[0]["pre_rerank_rank"] == 3
    assert ranked[0]["retrieval_score"] == 0.7
    assert ranked[0]["source_ranks"] == {"bm25": 3}
    print("Reranker core validation passed")


if __name__ == "__main__":
    main()
