"""Pretrained Cross-Encoder loading and chunk reranking helpers."""


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def load_reranker(model_name=DEFAULT_RERANKER_MODEL, max_length=512):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Reranking requires sentence-transformers. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return CrossEncoder(model_name, max_length=max_length)


def rerank_chunks(model, query, candidates, top_k=None, batch_size=8):
    """Score query/chunk pairs and return a deterministic descending ranking."""
    if not candidates:
        return []
    scores = model.predict(
        [(query, item["chunk"]["text"]) for item in candidates],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    scored = []
    for score, item in zip(scores.reshape(-1).tolist(), candidates):
        reranker_score = float(score)
        record = {
            "score": reranker_score,
            "reranker_score": reranker_score,
            "pre_rerank_rank": item["rank"],
            "retrieval_score": float(item["score"]),
            "chunk": item["chunk"],
        }
        if "source_ranks" in item:
            record["source_ranks"] = item["source_ranks"]
        scored.append(record)
    scored.sort(key=lambda item: (-item["score"], item["chunk"]["chunk_id"]))
    if top_k is not None:
        scored = scored[:top_k]
    return [
        {**item, "rank": rank}
        for rank, item in enumerate(scored, start=1)
    ]
