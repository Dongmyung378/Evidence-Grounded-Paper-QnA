"""Hybrid retrieval (BM25 + optional BGE-M3 dense), RRF, Dual-Score reranking, and Parent Window expansion."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[가-힣]+")


def tokenize(text: str) -> list[str]:
    return [t.casefold() for t in TOKEN_RE.findall(str(text or ""))]


class SimpleBM25:
    """Dependency-free BM25 index (compatible with rank_bm25 scoring)."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c.get("text", "")) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        df: Counter[str] = Counter()
        for cnt in self.tf:
            for term in cnt:
                df[term] += 1
        n_docs = len(chunks)
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def rank(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        scores: list[tuple[float, int]] = []
        for idx, cnt in enumerate(self.tf):
            dl = self.doc_lens[idx]
            s = 0.0
            for term in q_tokens:
                freq = cnt.get(term, 0)
                if freq <= 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avg_len, 1.0)))
                s += idf * (freq * (self.k1 + 1.0) / denom)
            scores.append((s, idx))
        scores.sort(key=lambda pair: (-pair[0], self.chunks[pair[1]]["chunk_id"]))
        return [
            {"rank": r, "score": float(sc), "chunk": self.chunks[idx]}
            for r, (sc, idx) in enumerate(scores[:top_k], start=1)
        ]


def rrf_fuse(rank_lists: list[list[dict[str, Any]]], rrf_k: int = 60, top_k: int = 25) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, dict[str, Any]] = {}
    for rlist in rank_lists:
        for rank, item in enumerate(rlist, start=1):
            cid = item["chunk"]["chunk_id"]
            scores[cid] += 1.0 / (rrf_k + rank)
            by_id[cid] = item["chunk"]
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [
        {"rank": i, "score": float(sc), "chunk": by_id[cid]}
        for i, (cid, sc) in enumerate(ranked, start=1)
    ]


def expand_parent_window(
    selected_chunks: list[dict[str, Any]],
    paper_chunks: list[dict[str, Any]],
    max_chars: int = 1800,
) -> list[dict[str, Any]]:
    pos_map = {c["chunk_id"]: idx for idx, c in enumerate(paper_chunks)}
    expanded: list[dict[str, Any]] = []
    for rank, item in enumerate(selected_chunks, start=1):
        ch = item["chunk"]
        idx = pos_map[ch["chunk_id"]]
        parts = [ch["text"]]
        if (
            idx - 1 >= 0
            and paper_chunks[idx - 1]["page"] == ch["page"]
            and len(paper_chunks[idx - 1]["text"]) + len(parts[0]) <= max_chars
        ):
            parts.insert(0, paper_chunks[idx - 1]["text"])
        if (
            idx + 1 < len(paper_chunks)
            and paper_chunks[idx + 1]["page"] == ch["page"]
            and sum(len(p) for p in parts) + len(paper_chunks[idx + 1]["text"]) <= max_chars
        ):
            parts.append(paper_chunks[idx + 1]["text"])
        expanded.append(
            {
                "evidence_id": f"ev-{rank:02d}-{ch['chunk_id']}",
                "paper_id": ch["paper_id"],
                "chunk_id": ch["chunk_id"],
                "page": ch["page"],
                "section": ch.get("section") or "Body",
                "text": " ".join(parts),
                "core_text": ch["text"],
                "reranker_score": round(float(item["reranker_score"]), 4),
                "retrieval_score": round(float(item["retrieval_score"]), 4),
            }
        )
    return expanded


class HybridRetriever:
    """Supports full BGE-M3 + bge-reranker-v2-m3 neural stack or lightweight lexical BM25 mode."""

    def __init__(self, use_neural: bool = False, device: str = "cpu"):
        self.use_neural = False
        self.device = device
        self.embedder = None
        self.reranker = None
        self._emb_cache: dict[str, np.ndarray] = {}
        self._bm25_cache: dict[str, SimpleBM25] = {}

        if use_neural:
            try:
                import torch
                from sentence_transformers import CrossEncoder, SentenceTransformer

                dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
                kwargs = {"torch_dtype": torch.float16} if dev == "cuda" else {}
                self.embedder = SentenceTransformer("BAAI/bge-m3", device=dev, model_kwargs=kwargs)
                self.reranker = CrossEncoder(
                    "BAAI/bge-reranker-v2-m3",
                    max_length=1024,
                    device=dev,
                    automodel_args=kwargs,
                )
                self.device = dev
                self.use_neural = True
            except Exception:
                self.use_neural = False

    @property
    def mode(self) -> str:
        return "bge-m3+bge-reranker-v2-m3" if self.use_neural else "bm25+parent-1800"

    def _get_bm25(self, paper_id: str, paper_chunks: list[dict[str, Any]]) -> SimpleBM25:
        if paper_id not in self._bm25_cache:
            self._bm25_cache[paper_id] = SimpleBM25(paper_chunks)
        return self._bm25_cache[paper_id]

    def _get_embeddings(self, paper_id: str, paper_chunks: list[dict[str, Any]]) -> np.ndarray:
        if paper_id not in self._emb_cache:
            embs = self.embedder.encode(
                [c["text"] for c in paper_chunks],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._emb_cache[paper_id] = np.asarray(embs, dtype=np.float32)
        return self._emb_cache[paper_id]

    def retrieve(
        self,
        query: str,
        paper_id: str,
        paper_chunks: list[dict[str, Any]],
        en_query: str | None = None,
        candidate_k: int = 25,
        top_k: int = 5,
        max_per_page: int = 2,
        parent_max_chars: int = 1800,
        en_rerank_weight: float = 0.65,
    ) -> dict[str, Any]:
        bm25 = self._get_bm25(paper_id, paper_chunks)
        queries = [query] + ([en_query] if en_query and en_query != query else [])
        rank_lists = [bm25.rank(q_text, top_k=20) for q_text in queries]

        if self.use_neural and self.embedder is not None and self.reranker is not None:
            paper_embs = self._get_embeddings(paper_id, paper_chunks)
            q_embs = np.asarray(
                self.embedder.encode(queries, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            for q_emb in q_embs:
                sims = (paper_embs @ q_emb).tolist()
                dense_ranked = sorted(
                    zip(sims, paper_chunks), key=lambda x: (-x[0], x[1]["chunk_id"])
                )[:20]
                rank_lists.append(
                    [{"rank": r, "score": float(s), "chunk": c} for r, (s, c) in enumerate(dense_ranked, start=1)]
                )
            fused = rrf_fuse(rank_lists, rrf_k=60, top_k=candidate_k)
            if en_query and en_query != query:
                s_en = self.reranker.predict([(en_query, it["chunk"]["text"]) for it in fused], batch_size=16, show_progress_bar=False)
                s_ko = self.reranker.predict([(query, it["chunk"]["text"]) for it in fused], batch_size=16, show_progress_bar=False)
                scores = (en_rerank_weight * np.asarray(s_en) + (1.0 - en_rerank_weight) * np.asarray(s_ko)).tolist()
            else:
                scores = self.reranker.predict([(query, it["chunk"]["text"]) for it in fused], batch_size=16, show_progress_bar=False).tolist()
            scored = [
                {"chunk": it["chunk"], "reranker_score": float(sc), "retrieval_score": float(it["score"])}
                for sc, it in zip(scores, fused)
            ]
        else:
            fused = rrf_fuse(rank_lists, rrf_k=60, top_k=candidate_k)
            bm25_top = {it["chunk"]["chunk_id"]: it["score"] for it in rank_lists[-1]}
            max_bm = max(bm25_top.values(), default=1.0) or 1.0
            scored = [
                {
                    "chunk": it["chunk"],
                    "reranker_score": round(bm25_top.get(it["chunk"]["chunk_id"], 0.0) / max_bm, 4),
                    "retrieval_score": float(it["score"]),
                }
                for it in fused
            ]

        scored.sort(key=lambda x: (-x["reranker_score"], x["chunk"]["chunk_id"]))
        selected: list[dict[str, Any]] = []
        page_counts: Counter[int] = Counter()
        broad_re = re.compile(
            r"(?:main topic|overview|summary|abstract|contribution|what is this paper|주제|요약|개요|핵심\s*내용|무엇에\s*관한)",
            flags=re.I,
        )
        if paper_chunks and (broad_re.search(query) or (en_query and broad_re.search(en_query))):
            p1_first = next((c for c in paper_chunks if int(c.get("page", 0)) == 1), paper_chunks[0])
            selected.append({"chunk": p1_first, "reranker_score": max((scored[0]["reranker_score"] if scored else 1.0), 0.99), "retrieval_score": 1.0})
            page_counts[int(p1_first["page"])] += 1
        seen_cids = {it["chunk"]["chunk_id"] for it in selected}
        for item in scored:
            cid = item["chunk"]["chunk_id"]
            if cid in seen_cids:
                continue
            pg = int(item["chunk"]["page"])
            if page_counts[pg] >= max_per_page:
                continue
            selected.append(item)
            seen_cids.add(cid)
            page_counts[pg] += 1
            if len(selected) >= top_k:
                break

        evidence = expand_parent_window(selected, paper_chunks, max_chars=parent_max_chars)
        top_score = scored[0]["reranker_score"] if scored else -99.0
        fifth_score = scored[min(4, len(scored) - 1)]["reranker_score"] if scored else 0.0
        return {
            "evidence": evidence,
            "top_reranker_score": round(float(top_score), 4),
            "score_margin_1_5": round(float(top_score - fifth_score), 4),
            "retrieval_mode": self.mode,
        }
