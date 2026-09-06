import json
from pathlib import Path
from collections import Counter

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / 'data' / 'evaluation' / 'questions.jsonl'
EVIDENCE_PATH = PROJECT_ROOT / 'data' / 'evaluation' / 'gold_evidence.jsonl'
PAPERS_DIR = PROJECT_ROOT / 'data' / 'raw' / 'papers'

REQUIRED_FIELDS = {
    'question_id',
    'paper_id',
    'gold_answer',
    'gold_evidence_text',
    'gold_page',
    'gold_section',
    'evidence_type',
    'review_status',
}


def read_jsonl(path):
    records = []
    errors = []

    with path.open('r', encoding='utf-8-sig') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                errors.append(f'{path.name} line {line_number}: {exc.msg}')

    return records, errors


def main():
    question_rows, errors = read_jsonl(QUESTIONS_PATH)
    evidence_rows, evidence_errors = read_jsonl(EVIDENCE_PATH)
    errors.extend(evidence_errors)

    question_ids = {item['question_id'] for _, item in question_rows}
    evidence_ids = []
    paper_page_counts = {}

    for line_number, item in evidence_rows:
        missing = REQUIRED_FIELDS - set(item.keys())
        for field in sorted(missing):
            errors.append(f'evidence line {line_number}: missing {field}')

        question_id = item.get('question_id')
        paper_id = item.get('paper_id')
        evidence_ids.append(question_id)

        if question_id not in question_ids:
            errors.append(f'evidence line {line_number}: unknown question_id {question_id}')

        pdf_path = PAPERS_DIR / f'{paper_id}.pdf'
        if not pdf_path.exists():
            errors.append(f'evidence line {line_number}: missing PDF for {paper_id}')
            continue

        if paper_id not in paper_page_counts:
            paper_page_counts[paper_id] = len(PdfReader(str(pdf_path)).pages)

        page = item.get('gold_page')
        if not isinstance(page, int) or not 1 <= page <= paper_page_counts[paper_id]:
            errors.append(
                f'evidence line {line_number}: invalid page {page} for {paper_id}'
            )

        if not isinstance(item.get('gold_answer'), str) or not item.get('gold_answer').strip():
            errors.append(f'evidence line {line_number}: empty gold_answer')

        if not isinstance(item.get('gold_evidence_text'), str) or not item.get('gold_evidence_text').strip():
            errors.append(f'evidence line {line_number}: empty gold_evidence_text')

        if item.get('evidence_type') != 'text':
            errors.append(f'evidence line {line_number}: evidence_type must be text')

    duplicate_ids = [qid for qid, count in Counter(evidence_ids).items() if count > 1]
    for qid in duplicate_ids:
        errors.append(f'duplicate evidence question_id: {qid}')

    missing_ids = question_ids - set(evidence_ids)
    for qid in sorted(missing_ids):
        errors.append(f'missing evidence for question_id: {qid}')

    print('Gold Evidence Validation Summary')
    print(f'Questions: {len(question_ids)}')
    print(f'Evidence records: {len(evidence_rows)}')
    print(f'PDFs checked: {len(paper_page_counts)}')
    print(f'Duplicate IDs: {len(duplicate_ids)}')
    print(f'Missing IDs: {len(missing_ids)}')

    review_counts = Counter(item.get('review_status') for _, item in evidence_rows)
    print(f'Review status: {dict(review_counts)}')

    if errors:
        print('\nErrors:')
        for error in errors:
            print(f'- {error}')
        print('\nValidation failed.')
        raise SystemExit(1)

    print('\nValidation passed structurally.')
    print('Note: review_status=needs_review records still require manual evidence verification.')


if __name__ == '__main__':
    main()
