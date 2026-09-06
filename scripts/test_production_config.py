"""Dependency-light checks for the frozen Day 28 config contract."""

from production_retrieval import config_fingerprint, load_production_config


def main():
    first = load_production_config()
    second = load_production_config()
    assert config_fingerprint(first) == config_fingerprint(second)
    assert first["retrieval"]["bm25_top_k"] == 20
    assert first["retrieval"]["dense_top_k"] == 20
    assert first["retrieval"]["candidate_k"] == 20
    assert first["evidence"]["top_k"] == 5
    assert first["status"] == "frozen"
    print("Production config tests passed")
    print(f"fingerprint={config_fingerprint(first)[:12]}")


if __name__ == "__main__":
    main()
