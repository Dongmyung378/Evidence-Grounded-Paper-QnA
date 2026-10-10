"""Shared data loading, page ranking, and metric helpers for retrieval."""

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
GOLD_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_evidence.jsonl"


def configure_utf8_stdout():
    """Avoid Windows legacy-console failures on symbols found in paper text."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_chunks(paper_id=None):
    chunks = load_jsonl(CHUNKS_PATH)
    return [chunk for chunk in chunks if paper_id is None or chunk["paper_id"] == paper_id]


def chunks_by_paper(chunks):
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["paper_id"]].append(chunk)
    return dict(grouped)


def load_evaluation_data():
    questions = load_jsonl(QUESTIONS_PATH)
    gold_by_question = {
        record["question_id"]: record
        for record in load_jsonl(GOLD_PATH)
    }
    return questions, gold_by_question


def rank_pages(scored_chunks, top_k=None):
    """Aggregate chunk scores to a unique-page ranking using the max score."""
    page_scores = {}
    for item in scored_chunks:
        chunk = item["chunk"]
        page = chunk["page"]
        if page not in page_scores or item["score"] > page_scores[page]["score"]:
            page_scores[page] = item

    ranked = sorted(
        page_scores.values(),
        key=lambda item: (-item["score"], item["chunk"]["page"], item["chunk"]["chunk_id"]),
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        {**item, "rank": rank}
        for rank, item in enumerate(ranked, start=1)
    ]


def retrieval_record(item):
    """Return a JSON-safe evidence record for a saved Top-K search result."""
    chunk = item["chunk"]
    record = {
        "rank": item["rank"],
        "score": item["score"],
        "chunk_id": chunk["chunk_id"],
        "paper_id": chunk["paper_id"],
        "page": chunk["page"],
        "section": chunk.get("section"),
        "text": chunk["text"],
    }
    if "source_ranks" in item:
        record["source_ranks"] = item["source_ranks"]
    for name in ("pre_rerank_rank", "retrieval_score", "reranker_score"):
        if name in item:
            record[name] = item[name]
    return record


def save_search_results(path, retriever, query, results, config):
    """Persist a reproducible Top-K retrieval run, including each score."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "retriever": retriever,
        "query": query,
        "config": config,
        "results": [retrieval_record(item) for item in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_page_retriever(questions, gold_by_question, retrieve_pages, top_k=10):
    """Evaluate a callback that returns ranked, unique pages for one question."""
    results = []
    for question in questions:
        ranked_pages = retrieve_pages(question, top_k)
        pages = [item["chunk"]["page"] for item in ranked_pages]
        gold_page = gold_by_question[question["question_id"]]["gold_page"]
        rank = pages.index(gold_page) + 1 if gold_page in pages else None
        results.append({
            "question_id": question["question_id"],
            "paper_id": question["paper_id"],
            "language": question["question_language"],
            "gold_page": gold_page,
            "predicted_pages": pages,
            "gold_rank": rank,
        })
    return results


def metrics_from_results(results, ks=(1, 3, 5, 10)):
    def recall_at(k, rows):
        return sum(row["gold_page"] in row["predicted_pages"][:k] for row in rows) / len(rows)

    def mrr(rows):
        return sum(1 / row["gold_rank"] if row["gold_rank"] else 0 for row in rows) / len(rows)

    metrics = {f"recall_at_{k}": recall_at(k, results) for k in ks}
    metrics["mrr"] = mrr(results)
    metrics["questions"] = len(results)
    metrics["by_language"] = {}
    for language in ("en", "ko"):
        rows = [row for row in results if row["language"] == language]
        metrics["by_language"][language] = {
            "questions": len(rows),
            **{f"recall_at_{k}": recall_at(k, rows) for k in ks},
            "mrr": mrr(rows),
        }
    return metrics
