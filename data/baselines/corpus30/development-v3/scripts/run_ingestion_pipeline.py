"""Portfolio-friendly single command for the full ingestion pipeline."""

import argparse
import json
from pathlib import Path

from ingest_papers import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    OUTPUT_PATH,
    PAGES_PATH,
    PAPERS_DIR,
    REPORT_PATH,
    PROJECT_ROOT,
    run_pipeline,
)


def main():
    parser = argparse.ArgumentParser(description="Run PDF → pages JSONL → chunks JSONL")
    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    parser.add_argument("--pages-output", type=Path, default=PAGES_PATH)
    parser.add_argument("--chunks-output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        parser.error("chunk-size must be positive and overlap must be smaller than chunk-size")

    report = run_pipeline(
        args.papers_dir,
        args.chunks_output,
        args.pages_output,
        args.report,
        args.chunk_size,
        args.overlap,
    )
    manifest_path = args.report.parent / "pipeline_manifest.json"
    manifest = {
        "command": "python scripts/run_ingestion_pipeline.py",
        "pipeline": "PDF → pages.jsonl → chunks.jsonl",
        "config": report["config"],
        "outputs": report["outputs"],
        "summary": report["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Pipeline completed")
    print(f"Manifest: {manifest_path}")
    print(f"Papers: {report['summary']['papers_succeeded']}/{report['summary']['papers_found']} succeeded")
    print(f"Pages: {report['summary']['pages_extracted']}")
    print(f"Chunks: {report['summary']['chunks_created']}")


if __name__ == "__main__":
    main()
