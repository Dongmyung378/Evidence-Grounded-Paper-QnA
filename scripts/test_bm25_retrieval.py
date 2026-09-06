"""Verify that BM25 retrieves both numeric and hyphenated model-name queries."""

from bm25_retrieval import build_index, rank_chunks


CHUNKS = [
    {
        "chunk_id": "paper-test-page-001-chunk-001",
        "paper_id": "paper-test",
        "page": 1,
        "section": "Method",
        "text": "The 2024 experiment uses BERT-base-uncased as the encoder.",
    },
    {
        "chunk_id": "paper-test-page-002-chunk-001",
        "paper_id": "paper-test",
        "page": 2,
        "section": "Results",
        "text": "The 2019 baseline uses a convolutional encoder.",
    },
]


def assert_top_chunk(query, expected_chunk_id):
    results = rank_chunks(build_index(CHUNKS), CHUNKS, query, top_k=1)
    assert results[0]["chunk"]["chunk_id"] == expected_chunk_id, results


def main():
    expected = "paper-test-page-001-chunk-001"
    assert_top_chunk("Which model is BERT-base-uncased?", expected)
    assert_top_chunk("What happened in 2024?", expected)
    print("BM25 numeric and model-name retrieval validation passed")


if __name__ == "__main__":
    main()
