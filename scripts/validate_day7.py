"""Validate the Day 7 data/evaluation gate."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    questions = jsonl(ROOT / "data/evaluation/questions.jsonl")
    gold = jsonl(ROOT / "data/evaluation/gold_evidence.jsonl")
    chunks = jsonl(ROOT / "data/processed/chunks.jsonl")
    manifest = list(csv.DictReader((ROOT / "data/metadata/paper_manifest.csv").open(encoding="utf-8")))
    qids = {item["question_id"] for item in questions}
    gids = {item["question_id"] for item in gold}
    paper_ids = {item["paper_id"] for item in manifest}
    chunk_papers = {item["paper_id"] for item in chunks}
    question_papers = {item["paper_id"] for item in questions}
    assert len(questions) == 40
    assert len(qids) == 40 and qids == gids
    assert len(gold) == 40 and all(item["review_status"] == "verified" for item in gold)
    assert paper_ids == chunk_papers
    assert question_papers == {f"paper-{number:03d}" for number in range(1, 11)}
    assert question_papers <= paper_ids
    assert {item["question_language"] for item in questions} == {"en", "ko"}
    assert all(isinstance(item.get("section"), str) and item["section"].strip() for item in chunks)
    assert (ROOT / ".gitignore").read_text(encoding="utf-8").find("data/raw/qasper/") >= 0
    assert (ROOT / ".gitignore").read_text(encoding="utf-8").find("data/raw/papers/*.pdf") >= 0
    print("Day 7 gate validation passed")
    print(
        f"questions=40 benchmark_papers={len(question_papers)} "
        f"corpus_papers={len(paper_ids)} chunks={len(chunks)}"
    )
    print("raw_data_policy=local_only_until_license_verification")


if __name__ == "__main__":
    main()
