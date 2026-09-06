"""Reciprocal Rank Fusion for combining BM25 and dense candidates."""


def reciprocal_rank_fusion(result_sets, rrf_k=60, top_k=None, weights=None):
    """Fuse ranked chunk lists, optionally weighting each retrieval source."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")
    weights = weights or {}
    source_weights = {
        source: float(weights.get(source, 1.0))
        for source in result_sets
    }
    if any(weight <= 0 for weight in source_weights.values()):
        raise ValueError("All RRF source weights must be greater than zero")

    fused = {}
    for source, results in result_sets.items():
        source_weight = source_weights[source]
        for item in results:
            chunk_id = item["chunk"]["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk": item["chunk"],
                    "score": 0.0,
                    "source_ranks": {},
                    "source_contributions": {},
                }
            contribution = source_weight / (rrf_k + item["rank"])
            fused[chunk_id]["score"] += contribution
            fused[chunk_id]["source_ranks"][source] = item["rank"]
            fused[chunk_id]["source_contributions"][source] = contribution

    ranked = sorted(
        fused.values(),
        key=lambda item: (-item["score"], item["chunk"]["chunk_id"]),
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        {**item, "rank": rank}
        for rank, item in enumerate(ranked, start=1)
    ]


def build_rerank_pool(result_sets, rrf_k=60, candidate_k=20, mode="top20"):
    """Build the Cross-Encoder input pool while preserving RRF metadata.

    ``top20`` reproduces the Day 22 baseline. ``source_union`` retains the
    unique union of each source's Top-K before reranking; the Cross-Encoder then
    returns the final Top-K. The latter fixes candidate starvation without
    changing retrieval models or using question-specific rules.
    """
    if mode not in {"top20", "source_union"}:
        raise ValueError("mode must be 'top20' or 'source_union'")
    return reciprocal_rank_fusion(
        result_sets,
        rrf_k=rrf_k,
        top_k=candidate_k if mode == "top20" else None,
        weights={source: 1.0 for source in result_sets},
    )


def ensure_page_candidate(pool, candidate_universe, page=1, top_k=20):
    """Reserve one candidate for a page only when it is absent from the pool."""
    selected = list(pool[:top_k])
    if any(item["chunk"]["page"] == page for item in selected):
        return selected, False
    fallback = next(
        (item for item in candidate_universe if item["chunk"]["page"] == page),
        None,
    )
    if fallback is None:
        return selected, False
    if len(selected) >= top_k:
        selected = selected[:top_k - 1]
    selected.append({**fallback, "coverage_reason": f"ensure_page_{page}"})
    return [
        {**item, "rank": rank}
        for rank, item in enumerate(selected, start=1)
    ], True
