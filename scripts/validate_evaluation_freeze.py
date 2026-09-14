"""Validate the final evaluation freeze against the current repository state."""

import hashlib
import json

from build_evaluation_freeze import CONFIG_PATH, OUTPUT_PATH, TABLE_PATH, build_outputs, sha256
from model_lock import load_model_lock
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


def main():
    configure_utf8_stdout()
    saved = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    saved_csv = TABLE_PATH.read_text(encoding="utf-8")
    expected_csv, expected = build_outputs()
    assert saved_csv == expected_csv
    assert saved == expected
    assert saved["status"] == "frozen"
    assert saved["seed"] == 378
    assert all(saved["checks"].values())
    assert saved["models"] == load_model_lock()["models"]

    retrieval = saved["metrics"]["retrieval"]
    assert retrieval["questions"] == 40
    assert retrieval["recall_at_5"] == 1.0
    assert retrieval["recall_at_10"] == 1.0
    assert retrieval["mrr"] == 0.58375
    abstention = saved["metrics"]["abstention"]
    assert abstention["answerable_retention"] >= 0.95
    assert abstention["calibration_refusal_rate"] == 1.0
    assert abstention["holdout_refusal_rate"] >= 0.9
    assert saved["metrics"]["answer_quality"]["pass_or_partial_rate"] == 0.85
    assert saved["metrics"]["answer_quality"]["citation_support"]["fail"] == 0
    assert saved["metrics"]["container"]["pass_or_partial_rate"] == 1.0
    assert saved["metrics"]["container"]["citation_support"]["pass"] == 10

    assert saved["performance_table"]["sha256"] == hashlib.sha256(
        saved_csv.encode("utf-8")
    ).hexdigest()
    assert saved["performance_table"]["rows"] == 29
    for path, digest in saved["provenance"]["source_hashes"].items():
        assert sha256(PROJECT_ROOT / path) == digest
    assert sha256(CONFIG_PATH) == saved["provenance"]["source_hashes"][
        "config/evaluation_freeze.json"
    ]

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_ko = (PROJECT_ROOT / "README_KO.md").read_text(encoding="utf-8")
    assert "0.5838 / 0.6308 / 0.5367" in readme
    assert "0.5838 / 0.6308 / 0.5367" in readme_ko
    for path in (
        PROJECT_ROOT / "docs" / "evaluation.md",
        PROJECT_ROOT / "docs" / "evaluation_KO.md",
    ):
        assert path.is_file()

    print("Final evaluation freeze validation passed")
    print(
        f"sources={len(saved['provenance']['source_hashes'])} "
        f"performance_rows={saved['performance_table']['rows']} seed=378"
    )
    print("current_retrieval=Recall@5:1.000 Recall@10:1.000 MRR:0.5838")
    print("gates=retrieval+corpus+abstention+answer+container passed")


if __name__ == "__main__":
    main()
