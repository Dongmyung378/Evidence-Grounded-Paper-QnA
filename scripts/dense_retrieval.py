"""Shared utilities for multilingual dense retrieval and embedding caching."""

import hashlib
from pathlib import Path

from retrieval_common import load_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "dense_embeddings.npz"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def load_model(model_name=DEFAULT_MODEL, local_files_only=False):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name, local_files_only=local_files_only)


def encode_chunks(model, chunks, batch_size=32):
    return model.encode(
        [f"passage: {chunk['text']}" for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def chunks_fingerprint(chunks):
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_create_embeddings(model, chunks, model_name=DEFAULT_MODEL, batch_size=32, cache_path=CACHE_PATH):
    """Load matching passage embeddings or create a local reusable cache."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Dense retrieval requires numpy.") from exc

    fingerprint = chunks_fingerprint(chunks)
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                cache_model = str(cache["model"].item())
                cache_fingerprint = str(cache["fingerprint"].item())
                embeddings = cache["embeddings"]
                if cache_model == model_name and cache_fingerprint == fingerprint and len(embeddings) == len(chunks):
                    return embeddings, True
        except (KeyError, OSError, ValueError):
            # A stale or interrupted cache is safely replaced below.
            pass

    embeddings = encode_chunks(model, chunks, batch_size)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            model=np.array(model_name),
            fingerprint=np.array(fingerprint),
            embeddings=embeddings,
        )
    return embeddings, False


def encode_query(model, query):
    return model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]


def rank_chunks(chunks, embeddings, query_embedding, top_k):
    scores = embeddings @ query_embedding
    ranked = sorted(
        zip(scores.tolist(), chunks),
        key=lambda item: (-item[0], item[1]["chunk_id"]),
    )
    return [
        {"rank": rank, "score": float(score), "chunk": chunk}
        for rank, (score, chunk) in enumerate(ranked[:top_k], start=1)
    ]


def rank_pages(chunks, embeddings, query_embedding, top_k):
    """Rank unique pages by the highest-scoring chunk on each page."""
    scores = embeddings @ query_embedding
    page_scores = {}
    for score, chunk in zip(scores.tolist(), chunks):
        page = chunk["page"]
        if page not in page_scores or score > page_scores[page][0]:
            page_scores[page] = (score, chunk)

    ranked = sorted(
        page_scores.values(),
        key=lambda item: (-item[0], item[1]["page"]),
    )
    return [
        {"rank": rank, "score": float(score), "chunk": chunk}
        for rank, (score, chunk) in enumerate(ranked[:top_k], start=1)
    ]
