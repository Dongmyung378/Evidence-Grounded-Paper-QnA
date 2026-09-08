"""Validate original-roadmap completion gates for Days 23 and 24."""

import json
from collections import Counter

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout


REQUIRED_EVIDENCE_FIELDS = {
    "evidence_id",
    "rank",
    "paper_id",
    "page",
    "section",
    "chunk_id",
    "source_page_id",
    "text",
    "char_count",
    "locator",
    "scores",
    "source",
}


def load(name):
    path = PROJECT_ROOT / "data" / "evaluation" / name
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    configure_utf8_stdout()
    demo = load("candidate_evidence_demo.json")
    schema = load("evidence_object_schema.json")
    progress = PROJECT_ROOT / "docs" / "roadmap" / "progress.md"

    assert demo["roadmap_days"] == [23, 24]
    assert demo["corpus"] == {"papers": 10, "chunks": 906}
    assert len(demo["examples"]) == 3
    assert set(schema["required"]) == REQUIRED_EVIDENCE_FIELDS
    hits = []
    for case in demo["examples"]:
        assert case["candidate_count"] == 20
        assert len(case["candidates"]) == 20
        assert [item["candidate_rank"] for item in case["candidates"]] == list(range(1, 21))
        assert len({item["chunk_id"] for item in case["candidates"]}) == 20
        assert case["evidence_count"] == 5
        assert len(case["evidence"]) == 5
        assert case["evidence_selection"]["policy"] == {
            "top_k": 5,
            "max_per_page": 2,
            "near_duplicate_metric": "token_3_shingle_jaccard",
            "near_duplicate_threshold": 0.85,
        }
        evidence_ids = set()
        chunk_ids = set()
        pages = Counter()
        for expected_rank, evidence in enumerate(case["evidence"], start=1):
            assert set(evidence) == REQUIRED_EVIDENCE_FIELDS
            assert evidence["rank"] == expected_rank
            assert evidence["paper_id"] == case["paper_id"]
            assert evidence["page"] >= 1
            assert evidence["text"].strip()
            assert evidence["char_count"] == len(evidence["text"])
            assert evidence["locator"]["page_label"] == f"p. {evidence['page']}"
            assert evidence["locator"]["section"] == evidence["section"]
            assert evidence["locator"]["chunk_id"] == evidence["chunk_id"]
            assert set(evidence["scores"]) == {
                "reranker", "hybrid", "pre_rerank_rank", "source_ranks"
            }
            assert 1 <= evidence["scores"]["pre_rerank_rank"] <= 20
            evidence_ids.add(evidence["evidence_id"])
            chunk_ids.add(evidence["chunk_id"])
            pages[evidence["page"]] += 1
        assert len(evidence_ids) == 5
        assert len(chunk_ids) == 5
        assert max(pages.values()) <= 2
        hits.append(case["gold_page_in_evidence"])

    assert any(hits) and not all(hits), "Demo must support both answerable and inspectable failure cases"
    assert "## 21일차부터 24일차" in progress.read_text(encoding="utf-8")
    print("Day 23-24 gate validation passed")
    print("day23=reranked_top20->deduplicated_top5")
    print("day24=evidence_object(page+section+chunk_id+text+scores)")
    print(f"demo_cases={len(demo['examples'])} gold_page_hits={sum(hits)}/{len(hits)}")


if __name__ == "__main__":
    main()
