import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"

REQUIRED_FIELDS = {
    "question_id",
    "paper_id",
    "question",
    "question_language",
    "question_type",
    "answerable",
}

VALID_PAPER_IDS = {
    f"paper-{i:03d}"
    for i in range(1, 11)
}

VALID_LANGUAGES = {"en", "ko"}

VALID_QUESTION_TYPES = {
    "objective",
    "method",
    "result",
    "limitation",
    "experiment",
}


def main():
    if not QUESTIONS_PATH.exists():
        print(f"[ERROR] File not found: {QUESTIONS_PATH}")
        raise SystemExit(1)

    errors = []
    warnings = []
    records = []
    question_ids = []

    raw_lines = QUESTIONS_PATH.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("```"):
            errors.append(
                f"Line {line_number}: Markdown code fence found"
            )
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                f"Line {line_number}: Invalid JSON - {exc.msg}"
            )
            continue

        if not isinstance(record, dict):
            errors.append(
                f"Line {line_number}: JSON value must be an object"
            )
            continue

        records.append(record)

        missing_fields = REQUIRED_FIELDS - set(record.keys())

        for field in sorted(missing_fields):
            errors.append(
                f"Line {line_number}: Missing field '{field}'"
            )

        if "question_id" in record:
            question_id = record["question_id"]
            question_ids.append(question_id)

            if not isinstance(question_id, str):
                errors.append(
                    f"Line {line_number}: question_id must be a string"
                )

        if "paper_id" in record:
            paper_id = record["paper_id"]

            if paper_id not in VALID_PAPER_IDS:
                errors.append(
                    f"Line {line_number}: Invalid paper_id '{paper_id}'"
                )

        if "question" in record:
            if not isinstance(record["question"], str):
                errors.append(
                    f"Line {line_number}: question must be a string"
                )
            elif not record["question"].strip():
                errors.append(
                    f"Line {line_number}: question is empty"
                )

        if "question_language" in record:
            language = record["question_language"]

            if language not in VALID_LANGUAGES:
                errors.append(
                    f"Line {line_number}: Invalid language '{language}'"
                )

        if "question_type" in record:
            question_type = record["question_type"]

            if question_type not in VALID_QUESTION_TYPES:
                errors.append(
                    f"Line {line_number}: Invalid question_type "
                    f"'{question_type}'"
                )

        if "answerable" in record:
            if not isinstance(record["answerable"], bool):
                errors.append(
                    f"Line {line_number}: answerable must be true or false"
                )

    duplicate_ids = [
        question_id
        for question_id, count in Counter(question_ids).items()
        if count > 1
    ]

    for question_id in sorted(duplicate_ids):
        errors.append(
            f"Duplicate question_id: {question_id}"
        )

    paper_counts = Counter(
        record.get("paper_id")
        for record in records
        if record.get("paper_id") in VALID_PAPER_IDS
    )

    language_counts = Counter(
        record.get("question_language")
        for record in records
        if record.get("question_language") in VALID_LANGUAGES
    )

    print("Question Validation Summary")
    print("===========================")
    print(f"Total valid JSON records: {len(records)}")
    print(f"Total question IDs: {len(question_ids)}")
    print(f"Paper counts: {dict(sorted(paper_counts.items()))}")
    print(f"Language counts: {dict(sorted(language_counts.items()))}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")

        print("\nValidation failed.")
        raise SystemExit(1)

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")

    print("\nValidation passed.")
    

if __name__ == "__main__":
    main()