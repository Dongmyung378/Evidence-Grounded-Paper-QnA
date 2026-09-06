"""Ask one question through the local Q&A pipeline with Day 31 abstention."""

import argparse
import json
import os
from pathlib import Path

from qna_pipeline import GroundedQAPipeline
from retrieval_common import configure_utf8_stdout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load all Hugging Face models from the local cache only.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    configure_utf8_stdout()
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    pipeline = GroundedQAPipeline(local_files_only=args.offline)
    result = pipeline.ask(
        args.query,
        args.paper_id,
        include_debug=args.debug,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
