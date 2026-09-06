"""Build the original-roadmap Day 26 four-system ablation table."""

import argparse
import csv
import json
from pathlib import Path

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


DEFAULT_BASELINE = PROJECT_ROOT / "data" / "evaluation" / "retrieval_comparison.json"
DEFAULT_RERANKER = PROJECT_ROOT / "data" / "evaluation" / "reranker_metrics.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "ablation.csv"
METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr")


def rows_for(system, metrics, stage):
    rows = []
    for scope, values in [("all", metrics), *metrics["by_language"].items()]:
        rows.append({
            "system": system,
            "scope": scope,
            "questions": values["questions"],
            **{name: values[name] for name in METRICS},
            "candidate_stage": stage,
        })
    return rows


def main():
    configure_utf8_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--reranker", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    reranker = json.loads(args.reranker.read_text(encoding="utf-8"))
    if any(values["questions"] != 40 for values in baseline["metrics"].values()):
        raise ValueError("Ablation requires the fixed verified-40 set")
    if reranker["evaluation_set"]["questions"] != 40:
        raise ValueError("Reranker artifact must use the same 40 questions")

    rows = []
    rows.extend(rows_for("bm25", baseline["metrics"]["bm25"], "sparse retrieval"))
    rows.extend(rows_for("dense", baseline["metrics"]["dense"], "multilingual dense retrieval"))
    rows.extend(rows_for("hybrid_rrf", baseline["metrics"]["hybrid_rrf"], "equal-weight RRF"))
    rows.extend(rows_for("hybrid_reranker", reranker["metrics"]["hybrid_reranker"], "Hybrid Top-20 + Cross-Encoder"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Day 26 ablation saved")
    for row in rows:
        if row["scope"] == "all":
            print(
                f"{row['system']}: R@1={row['recall_at_1']:.3f} "
                f"R@5={row['recall_at_5']:.3f} R@10={row['recall_at_10']:.3f} "
                f"MRR={row['mrr']:.3f}"
            )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
