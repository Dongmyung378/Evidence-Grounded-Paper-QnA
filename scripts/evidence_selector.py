"""Select UI-ready evidence from reranked retrieval candidates."""

import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[가-힣]+|[A-Za-z0-9]+")


def normalize_text(text):
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def text_shingles(text, size=3):
    tokens = normalize_text(text).split()
    if len(tokens) < size:
        return set(tokens)
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def jaccard_similarity(left, right):
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def evidence_object(item, rank):
    """Convert one scored chunk into the stable Day 24 UI contract."""
    chunk = item["chunk"]
    section = chunk.get("section")
    source_ranks = {
        name: int(value)
        for name, value in item.get("source_ranks", {}).items()
    }
    return {
        "evidence_id": f"ev-{rank:02d}-{chunk['chunk_id']}",
        "rank": rank,
        "paper_id": chunk["paper_id"],
        "page": int(chunk["page"]),
        "section": section,
        "chunk_id": chunk["chunk_id"],
        "source_page_id": chunk.get(
            "source_page_id", f"{chunk['paper_id']}-p{int(chunk['page']):03d}"
        ),
        "text": chunk["text"],
        "char_count": int(chunk.get("char_count", len(chunk["text"]))),
        "locator": {
            "page_label": f"p. {int(chunk['page'])}",
            "section": section,
            "chunk_id": chunk["chunk_id"],
        },
        "scores": {
            "reranker": float(item.get("reranker_score", item["score"])),
            "hybrid": float(item.get("retrieval_score", item["score"])),
            "pre_rerank_rank": int(item.get("pre_rerank_rank", item["rank"])),
            "source_ranks": source_ranks,
        },
        "source": {
            "section_index": int(chunk.get("section_index", 0)),
            "chunk_index": int(chunk.get("chunk_index", 0)),
        },
    }


def select_evidence(
    candidates,
    top_k=5,
    max_per_page=2,
    near_duplicate_threshold=0.85,
):
    """Select diverse evidence in reranker order using deterministic rules.

    Duplicate chunk IDs and normalized text are always removed. Highly similar
    chunks are removed using token 3-shingle Jaccard similarity. At most two
    chunks are selected from one PDF page so a single page cannot dominate the
    answer context.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if max_per_page <= 0:
        raise ValueError("max_per_page must be greater than zero")
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between zero and one")

    selected = []
    seen_chunk_ids = set()
    seen_texts = set()
    selected_shingles = []
    page_counts = Counter()
    skipped = Counter()

    for item in candidates:
        chunk = item["chunk"]
        chunk_id = chunk["chunk_id"]
        normalized = normalize_text(chunk["text"])
        shingles = text_shingles(chunk["text"])
        page_key = (chunk["paper_id"], int(chunk["page"]))

        if chunk_id in seen_chunk_ids:
            skipped["duplicate_chunk_id"] += 1
            continue
        seen_chunk_ids.add(chunk_id)
        if normalized in seen_texts:
            skipped["duplicate_text"] += 1
            continue
        if any(
            jaccard_similarity(shingles, prior) >= near_duplicate_threshold
            for prior in selected_shingles
        ):
            skipped["near_duplicate"] += 1
            continue
        if page_counts[page_key] >= max_per_page:
            skipped["page_cap"] += 1
            continue

        seen_texts.add(normalized)
        selected_shingles.append(shingles)
        page_counts[page_key] += 1
        selected.append(evidence_object(item, len(selected) + 1))
        if len(selected) == top_k:
            break

    return selected, {
        "input_candidates": len(candidates),
        "selected": len(selected),
        "skipped": {
            "duplicate_chunk_id": skipped["duplicate_chunk_id"],
            "duplicate_text": skipped["duplicate_text"],
            "near_duplicate": skipped["near_duplicate"],
            "page_cap": skipped["page_cap"],
        },
        "policy": {
            "top_k": top_k,
            "max_per_page": max_per_page,
            "near_duplicate_metric": "token_3_shingle_jaccard",
            "near_duplicate_threshold": near_duplicate_threshold,
        },
    }
