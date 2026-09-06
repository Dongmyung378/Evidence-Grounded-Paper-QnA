"""Shared BM25 indexing and ranking utilities."""

import re


def tokenize(text):
    """Tokenize Korean, words, numbers, and model names consistently.

    Hyphenated or dotted model identifiers are retained as one token and also
    split into their components, so ``bert-base-uncased`` matches both an
    exact model-name query and a query such as ``BERT base``.
    """
    tokens = []
    for token in re.findall(r"[가-힣]+|[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*", text.lower()):
        tokens.append(token)
        if re.search(r"[-_./]", token):
            tokens.extend(re.findall(r"[A-Za-z0-9]+", token))
    return tokens


def build_index(chunks):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "BM25 retrieval requires rank-bm25. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


def rank_chunks(index, chunks, query, top_k=None):
    scores = index.get_scores(tokenize(query))
    ranked = sorted(
        zip(scores.tolist(), chunks),
        key=lambda item: (-item[0], item[1]["chunk_id"]),
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        {"rank": rank, "score": float(score), "chunk": chunk}
        for rank, (score, chunk) in enumerate(ranked, start=1)
    ]
