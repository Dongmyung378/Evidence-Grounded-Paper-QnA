"""Day 23-24 Hybrid candidate and UI evidence pipeline."""

from bm25_retrieval import build_index, rank_chunks as rank_bm25_chunks
from dense_retrieval import (
    CACHE_PATH,
    DEFAULT_MODEL,
    encode_query,
    load_model,
    load_or_create_embeddings,
    rank_chunks as rank_dense_chunks,
)
from evidence_selector import select_evidence
from hybrid_retrieval import build_rerank_pool, ensure_page_candidate
from reranker import DEFAULT_RERANKER_MODEL, load_reranker, rerank_chunks
from retrieval_common import chunks_by_paper, load_chunks


class CandidateEvidencePipeline:
    """Load retrieval resources once and serve repeated one-paper searches."""

    def __init__(
        self,
        embedding_model_name=DEFAULT_MODEL,
        reranker_model_name=DEFAULT_RERANKER_MODEL,
        embedding_batch_size=32,
        reranker_batch_size=8,
        max_length=512,
        chunks=None,
        embedding_cache_path=CACHE_PATH,
        embedding_model=None,
        reranker=None,
        local_files_only=False,
        embedding_device=None,
        reranker_device=None,
    ):
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.embedding_batch_size = embedding_batch_size
        self.reranker_batch_size = reranker_batch_size
        self.max_length = max_length
        self.embedding_device = embedding_device
        self.reranker_device = reranker_device
        self.chunks = list(chunks) if chunks is not None else load_chunks()
        if not self.chunks:
            raise ValueError("Candidate evidence requires at least one chunk")
        self.grouped_chunks = chunks_by_paper(self.chunks)
        self.indices_by_paper = {
            paper_id: [
                index
                for index, chunk in enumerate(self.chunks)
                if chunk["paper_id"] == paper_id
            ]
            for paper_id in self.grouped_chunks
        }
        self.bm25_indexes = {
            paper_id: build_index(paper_chunks)
            for paper_id, paper_chunks in self.grouped_chunks.items()
        }
        self.embedding_model = embedding_model or load_model(
            embedding_model_name,
            local_files_only=local_files_only,
            device=embedding_device,
        )
        all_embeddings, self.embedding_cache_hit = load_or_create_embeddings(
            self.embedding_model,
            self.chunks,
            embedding_model_name,
            embedding_batch_size,
            cache_path=embedding_cache_path,
        )
        self.grouped_embeddings = {
            paper_id: all_embeddings[indices]
            for paper_id, indices in self.indices_by_paper.items()
        }
        self.reranker = reranker or load_reranker(
            reranker_model_name,
            max_length=max_length,
            local_files_only=local_files_only,
            device=reranker_device,
        )

    @staticmethod
    def _candidate_object(item):
        chunk = item["chunk"]
        return {
            "candidate_rank": int(item["rank"]),
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "page": int(chunk["page"]),
            "section": chunk.get("section"),
            "source_page_id": chunk.get("source_page_id"),
            "text_excerpt": chunk["text"][:300],
            "reranker_score": float(item["reranker_score"]),
            "hybrid_score": float(item["retrieval_score"]),
            "pre_rerank_rank": int(item["pre_rerank_rank"]),
            "source_ranks": {
                name: int(value)
                for name, value in item.get("source_ranks", {}).items()
            },
        }

    def run(
        self,
        query,
        paper_id,
        source_k=20,
        candidate_k=20,
        evidence_k=5,
        rrf_k=60,
        max_per_page=2,
        near_duplicate_threshold=0.85,
        front_matter_page=1,
    ):
        if paper_id not in self.grouped_chunks:
            raise ValueError(f"Unknown paper_id or no chunks: {paper_id}")
        if candidate_k < evidence_k:
            raise ValueError("candidate_k must be greater than or equal to evidence_k")
        if source_k <= 0:
            raise ValueError("source_k must be greater than zero")

        paper_chunks = self.grouped_chunks[paper_id]
        bm25 = rank_bm25_chunks(
            self.bm25_indexes[paper_id], paper_chunks, query, source_k
        )
        dense = rank_dense_chunks(
            paper_chunks,
            self.grouped_embeddings[paper_id],
            encode_query(self.embedding_model, query),
            source_k,
        )
        universe = build_rerank_pool(
            {"bm25": bm25, "dense": dense},
            rrf_k=rrf_k,
            candidate_k=source_k,
            mode="source_union",
        )
        hybrid, front_matter_injected = ensure_page_candidate(
            universe[:candidate_k],
            universe,
            page=front_matter_page,
            top_k=candidate_k,
        )
        reranked = rerank_chunks(
            self.reranker,
            query,
            hybrid,
            top_k=candidate_k,
            batch_size=self.reranker_batch_size,
        )
        evidence, selection = select_evidence(
            reranked,
            top_k=evidence_k,
            max_per_page=max_per_page,
            near_duplicate_threshold=near_duplicate_threshold,
        )
        return {
            "schema_version": 1,
            "roadmap_days": [23, 24],
            "query": query,
            "paper_id": paper_id,
            "candidate_policy": {
                "candidate_k": candidate_k,
                "sources": {"bm25": source_k, "dense": source_k},
                "fusion": "equal_weight_reciprocal_rank_fusion",
                "fusion_weights": {"bm25": 1.0, "dense": 1.0},
                "rrf_k": rrf_k,
                "front_matter_guard": {
                    "page": front_matter_page,
                    "enabled": True,
                    "candidate_injected": front_matter_injected,
                },
                "reranker_model": self.reranker_model_name,
                "embedding_model": self.embedding_model_name,
            },
            "candidates": [self._candidate_object(item) for item in reranked],
            "evidence_selection": selection,
            "evidence": evidence,
        }
