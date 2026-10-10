"""Validate versioned synthetic and real-paper inputs without model downloads."""

from __future__ import annotations

import json
import sys

from quality_benchmark import ROOT, load_dataset, validate_dataset


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config, manifest, questions, rubric = load_dataset()
    print(json.dumps(validate_dataset(config, manifest, questions, rubric), ensure_ascii=False, indent=2))
    real_config = ROOT / "data/quality_benchmark/real_paper_config.json"
    if real_config.is_file():
        config, manifest, questions, rubric = load_dataset(real_config)
        print(json.dumps(validate_dataset(config, manifest, questions, rubric), ensure_ascii=False, indent=2))
    print("PASS: frozen inputs, paper isolation, translation groups, question types and final-output rubric")
    print("Scope: input integrity only. Actual model executions are separate records; independent human review and external verification are not attested.")


if __name__ == "__main__":
    main()
