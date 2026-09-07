"""Check current implementation against the saved real HTTP Day 33 run."""

import json

from evaluate_day33 import OUTPUT, ROOT, SOURCE_PDF, digest


def main():
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert result["roadmap_day"] == 33
    assert result["seed"] == 378
    assert result["tests_run"] >= 10
    assert result["test_failures"] == result["test_errors"] == 0
    assert result["implementation_sha256"]
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in result["implementation_sha256"].values()
    ), "Historical implementation hashes are malformed"
    assert result["frozen_inputs_unchanged"] is True
    for name, fingerprint in result["frozen_input_sha256"].items():
        assert digest(ROOT / name) == fingerprint, f"Frozen input changed: {name}"
    smoke = result["http_smoke"]
    assert smoke["source_sha256"] == digest(SOURCE_PDF)
    assert smoke["upload_status_code"] == 201
    assert smoke["analyze_status_code"] == 202
    assert smoke["job"]["status"] == "completed"
    assert smoke["job"]["paper_id"] == smoke["upload"]["paper_id"]
    assert smoke["job"]["seed"] == 378
    assert smoke["job"]["page_count"] == 10
    assert smoke["job"]["chunk_count"] > 0
    assert smoke["traceability_passed"] and smoke["duplicate_analyze_reused_job"]
    assert result["temporary_runtime_removed"]
    assert "data/runtime/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    print("Day 33 upload/analyze gate passed")
    print(f"tests={result['tests_run']} HTTP=201->202->completed seed=378")
    print(f"pages={smoke['job']['page_count']} chunks={smoke['job']['chunk_count']} traceability=passed")
    print("scope=historical parsing_and_chunking snapshot; current code may include later roadmap work")


if __name__ == "__main__":
    main()
