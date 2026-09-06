import csv
import os
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "paper_manifest.csv"
PAPERS_DIR = PROJECT_ROOT / "data" / "raw" / "papers"
MAX_FILE_SIZE_MB = 20


def validate_paper(row):
    paper_id = row["paper_id"]
    filename = row["pdf_filename"]
    pdf_path = PAPERS_DIR / filename

    errors = []
    warnings = []

    if not pdf_path.exists():
        errors.append(f"File not found: {pdf_path}")
        return paper_id, errors, warnings

    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File size is {file_size_mb:.2f}MB, exceeding the 20MB limit"
        )

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)

        if page_count == 0:
            errors.append("PDF has no pages")

        extracted_text = "\n".join(
            (page.extract_text() or "")
            for page in reader.pages[:3]
        ).strip()

        if not extracted_text:
            errors.append("No extractable text found")

        manifest_page_count = row.get("page_count", "").strip()

        if manifest_page_count:
            if int(manifest_page_count) != page_count:
                warnings.append(
                    f"Manifest page count is {manifest_page_count}, "
                    f"actual page count is {page_count}"
                )

        license_status = row.get("license_status", "").strip()

        if license_status in {"check", "to-be-verified", "pending"}:
            warnings.append("License status still requires verification")
        elif license_status == "arXiv non-exclusive distribution":
            warnings.append(
                "Raw PDF must remain local-only; the arXiv license grants "
                "distribution rights to arXiv, not this repository"
            )
        elif "NC" in license_status:
            warnings.append(
                f"Restricted reuse license ({license_status}); follow its "
                "attribution and non-commercial terms"
            )

        print(
            f"[PASS] {paper_id} | "
            f"{file_size_mb:.2f}MB | "
            f"{page_count} pages | "
            f"text extracted"
        )

    except Exception as exc:
        errors.append(f"PDF processing failed: {exc}")

    return paper_id, errors, warnings


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    all_errors = []
    all_warnings = []

    for row in rows:
        paper_id, errors, warnings = validate_paper(row)

        for error in errors:
            all_errors.append(f"{paper_id}: {error}")

        for warning in warnings:
            all_warnings.append(f"{paper_id}: {warning}")

    print("\nValidation Summary")
    print("==================")

    if all_errors:
        print("\nErrors:")
        for error in all_errors:
            print(f"- {error}")
    else:
        print("- No errors found")

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"- {warning}")
    else:
        print("- No warnings found")

    if all_errors:
        raise SystemExit(1)

    print("\nAll papers passed basic validation.")


if __name__ == "__main__":
    main()
