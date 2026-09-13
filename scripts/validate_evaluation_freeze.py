"""Validate the final evaluation freeze against the current repository state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_evaluation_freeze import (
    CONFIG_PATH,
    OUTPUT_PATH,
    TABLE_PATH,
    build_outputs,
    sha256,
)
from model_lock import load_model_lock
from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


def main() -> None:
    configure_utf8_stdout()
    saved = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    saved_csv = TABLE_PATH.read_text(encoding="utf-8")
    expected_csv, expected = build_outputs()
    assert saved_csv == expected_csv
    assert saved == expected
    assert saved["status"] == "frozen"
    assert saved["seed"] == 378
    assert all(saved["checks"].values())

    lock = load_model_lock()
    assert saved["models"] == lock["models"]
    assert lock["models"]["embedding"]["revision"] == (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    )
    assert lock["models"]["reranker"]["revision"] == (
        "1427fd652930e4ba29e8149678df786c240d8825"
    )
    assert lock["models"]["translation"]["revision"] == (
        "38bd1c6a0f58097b26ef7c7cd73a0dcac034350a"
    )
    assert lock["models"]["legacy_generator"]["revision"] == (
        "7ae557604adf67be50417f59c2c2f167def9a775"
    )

    current = saved["metrics"]["retrieval"]["current"]
    assert current["questions"] == 40
    assert current["recall_at_5"] == 1.0
    assert current["recall_at_10"] == 1.0
    assert current["mrr"] == 0.58375
    assert saved["metric_correction"]["headline_metric"] == (
        "current_retrieval_mrr"
    )
    assert saved["metrics"]["abstention"] == {
        "answerable_retention": 0.95,
        "unsupported_holdout_refusal": 1.0,
        "holdout_fallbacks": 0,
    }
    assert saved["metrics"]["answer_quality"]["pass_or_partial_rate"] == 0.85
    assert saved["metrics"]["answer_quality"]["citation_support"]["fail"] == 0
    assert saved["metrics"]["container"]["pass_or_partial_rate"] == 1.0
    assert saved["metrics"]["container"]["citation_support"]["pass"] == 10

    assert saved["performance_table"]["sha256"] == hashlib.sha256(
        saved_csv.encode("utf-8")
    ).hexdigest()
    assert saved["performance_table"]["rows"] == 30
    for path, digest in saved["provenance"]["source_hashes"].items():
        assert sha256(PROJECT_ROOT / path) == digest
    assert sha256(CONFIG_PATH) == saved["provenance"]["source_hashes"][
        "config/evaluation_freeze.json"
    ]

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_ko = (PROJECT_ROOT / "README_KO.md").read_text(encoding="utf-8")
    assert "Production retrieval MRR | 0.5838" in readme
    assert "운영 검색 MRR | 0.5838" in readme_ko
    assert "Production retrieval MRR | 0.6317" not in readme
    assert "운영 검색 MRR | 0.6317" not in readme_ko
    for path in (
        PROJECT_ROOT / "docs" / "reviews" / "final_evaluation.md",
        PROJECT_ROOT / "docs" / "reviews" / "final_evaluation_KO.md",
    ):
        assert path.is_file()

    print("Final evaluation freeze validation passed")
    print(
        "sources="
        f"{len(saved['provenance']['source_hashes'])} "
        f"performance_rows={saved['performance_table']['rows']} seed=378"
    )
    print("current_retrieval=Recall@5:1.000 Recall@10:1.000 MRR:0.5838")
    print("gates=retrieval+corpus+abstention+answer+container passed")


if __name__ == "__main__":
    main()
