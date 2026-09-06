"""Build leakage-safe text-only reranker pairs from QASPER train/validation."""

import argparse
import json
import re
from pathlib import Path

from bm25_retrieval import build_index, rank_chunks
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


QASPER_PATH = PROJECT_ROOT / "data" / "raw" / "qasper"
TRAIN_OUTPUT = PROJECT_ROOT / "data" / "processed" / "reranker_pairs_train.jsonl"
VALIDATION_OUTPUT = PROJECT_ROOT / "data" / "processed" / "reranker_pairs_validation.jsonl"
SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "day25_training_data.json"


def normalize(text):
    return " ".join(re.findall(r"[A-Za-z0-9]+", text.lower()))


def paper_paragraphs(record):
    paragraphs = []
    for section, section_paragraphs in zip(
        record["full_text"]["section_name"], record["full_text"]["paragraphs"]
    ):
        for paragraph_index, text in enumerate(section_paragraphs, start=1):
            text = " ".join(text.split())
            if len(text) < 40 or text.startswith("FLOAT SELECTED:"):
                continue
            paragraphs.append({
                "chunk_id": f"{record['id']}:{len(paragraphs) + 1}",
                "paper_id": record["id"],
                "section": section,
                "paragraph_index": paragraph_index,
                "text": text,
            })
    return paragraphs


def textual_evidence(qas, question_index):
    evidence = []
    for annotation in qas["answers"][question_index]["answer"]:
        if annotation["unanswerable"]:
            continue
        for text in annotation["evidence"]:
            text = " ".join(text.split())
            if len(text) < 40 or text.startswith("FLOAT SELECTED:"):
                continue
            if normalize(text) and normalize(text) not in {normalize(item) for item in evidence}:
                evidence.append(text)
    return evidence


def build_split(dataset, split, max_questions, negatives_per_positive):
    pairs = []
    questions_used = 0
    skipped = {"unanswerable_or_nontext": 0, "no_negative": 0}
    for paper in dataset:
        paragraphs = paper_paragraphs(paper)
        if not paragraphs:
            continue
        index = build_index(paragraphs)
        qas = paper["qas"]
        for question_index, question in enumerate(qas["question"]):
            positives = textual_evidence(qas, question_index)
            if not positives:
                skipped["unanswerable_or_nontext"] += 1
                continue
            positive = positives[0]
            positive_norms = {normalize(text) for text in positives}
            ranked = rank_chunks(index, paragraphs, question)
            negatives = [
                item["chunk"]
                for item in ranked
                if normalize(item["chunk"]["text"]) not in positive_norms
                and normalize(item["chunk"]["text"]) != normalize(positive)
            ][:negatives_per_positive]
            if len(negatives) < negatives_per_positive:
                skipped["no_negative"] += 1
                continue

            question_id = qas["question_id"][question_index]
            base = {
                "split": split,
                "paper_id": paper["id"],
                "question_id": question_id,
                "query": question,
                "source_dataset": "QASPER",
            }
            pairs.append({
                **base,
                "pair_id": f"{split}:{question_id}:pos",
                "passage": positive,
                "label": 1,
                "pair_type": "gold_text_evidence",
            })
            for negative_index, negative in enumerate(negatives, start=1):
                pairs.append({
                    **base,
                    "pair_id": f"{split}:{question_id}:neg{negative_index}",
                    "passage": negative["text"],
                    "label": 0,
                    "pair_type": "bm25_hard_negative",
                })
            questions_used += 1
            if questions_used >= max_questions:
                return pairs, questions_used, skipped
    return pairs, questions_used, skipped


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasper-path", type=Path, default=QASPER_PATH)
    parser.add_argument("--train-questions", type=int, default=1000)
    parser.add_argument("--validation-questions", type=int, default=200)
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--train-output", type=Path, default=TRAIN_OUTPUT)
    parser.add_argument("--validation-output", type=Path, default=VALIDATION_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("QASPER pair generation requires datasets.") from exc

    dataset = load_from_disk(str(args.qasper_path))
    train_pairs, train_questions, train_skipped = build_split(
        dataset["train"], "train", args.train_questions, args.negatives_per_positive
    )
    validation_pairs, validation_questions, validation_skipped = build_split(
        dataset["validation"],
        "validation",
        args.validation_questions,
        args.negatives_per_positive,
    )
    write_jsonl(args.train_output, train_pairs)
    write_jsonl(args.validation_output, validation_pairs)
    summary = {
        "schema_version": 1,
        "roadmap_day": 25,
        "source": "QASPER train/validation only",
        "leakage_policy": {
            "verified_40_role": "test_only",
            "verified_40_used_for_training": False,
            "qasper_test_used": False,
        },
        "evidence_policy": "text evidence only; FLOAT SELECTED table/figure placeholders excluded",
        "negative_policy": "top BM25 non-evidence paragraphs from the same paper",
        "negatives_per_positive": args.negatives_per_positive,
        "splits": {
            "train": {
                "questions": train_questions,
                "pairs": len(train_pairs),
                "positives": sum(row["label"] == 1 for row in train_pairs),
                "negatives": sum(row["label"] == 0 for row in train_pairs),
                "skipped": train_skipped,
                "path": str(args.train_output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            "validation": {
                "questions": validation_questions,
                "pairs": len(validation_pairs),
                "positives": sum(row["label"] == 1 for row in validation_pairs),
                "negatives": sum(row["label"] == 0 for row in validation_pairs),
                "skipped": validation_skipped,
                "path": str(args.validation_output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {args.summary_output}")


if __name__ == "__main__":
    main()
