# Days 23-24 Candidate Selection and Evidence Metadata Report

## Original roadmap contract

- Day 23 task: define Top-20 candidates, Top-5 evidence, and deduplication rules.
- Day 23 completion gate: a person can read the search result and decide whether
  the question can be answered from it.
- Day 24 task: connect `page`, `section`, and `chunk_id` to every evidence item.
- Day 24 completion gate: the evidence object required by the UI is complete.

No answer-generation, score-threshold, or automatic abstention behavior is
included here. Those belong to later roadmap days.

## Day 23 - fixed candidate and evidence policy

The production candidate policy reuses the selected Day 22 retriever without
changing its measured configuration:

1. Retrieve BM25 Top-20 and multilingual dense Top-20 chunks from the uploaded
   paper.
2. Fuse them with equal-weight reciprocal rank fusion (`rrf_k=60`) and retain 20
   unique chunk IDs.
3. Score all 20 query/chunk pairs with the multilingual Cross-Encoder.
4. Scan the reranked list and retain five evidence chunks after deterministic
   deduplication.

The evidence selector applies the following rules in order:

1. remove repeated `chunk_id` values;
2. remove exact duplicates after case-folding and token normalization;
3. remove near duplicates at token 3-shingle Jaccard similarity >= `0.85`;
4. allow at most two selected chunks from the same PDF page;
5. stop after five evidence chunks.

The dependency-light selector test exercises all four rejection paths and
confirms that the output order remains the reranker order.

## Day 24 - UI evidence object

Each selected item now has a stable JSON object with:

- identity: `evidence_id`, `paper_id`, `chunk_id`, `source_page_id`;
- citation locator: numeric `page`, `section`, and a display-ready `locator`;
- evidence content: original `text` and `char_count`;
- explainability: final rank, reranker score, Hybrid score, pre-rerank position,
  and BM25/dense source ranks;
- traceability: `section_index` and `chunk_index`.

`evidence_object_schema.json` records this UI contract. The object is text-only,
matching the MVP exclusion of table, graph, image, and equation interpretation.

## Actual three-case inspection

| Question | Case | Gold page | Gold rank in candidates | Final evidence pages | Gold in evidence |
|---|---|---:|---:|---|---|
| `q-001-ko` | Korean-to-English success | 2 | 2 | 1, 2, 6, 1, 2 | yes |
| `q-043-ko` | reranker recovery | 19 | 1 | 19, 18, 2, 2, 5 | yes |
| `q-005-ko` | known terminology/candidate miss | 1 | not in Top-20 | 10, 8, 3, 3, 4 | no |

All three runs produced exactly 20 unique candidates and five evidence objects.
The first two provide directly inspectable supporting text. The third makes the
insufficiency visible: the verified page is absent even before final evidence
selection, so a reviewer can avoid treating plausible but unsupported text as an
answer. The demo uses gold pages only for evaluation; gold data is never passed
to retrieval or selection.

## Completion and quality decision

- Day 23: complete. Top-20, Top-5, and duplicate rules are explicit,
  deterministic, tested, and represented by readable success/failure examples.
- Day 24: complete. Every selected evidence item carries the UI-required
  `page`, `section`, and `chunk_id`, plus source text and score traceability.
- Quality: appropriate for the original roadmap gate. The selector preserves
  Day 22 ranking behavior and does not hide the known `q-005-ko` candidate miss.
- Remaining limitation: automatic evidence sufficiency and abstention are not
  inferred from Cross-Encoder logits. They must be calibrated on their planned
  later days, and the final retrieval gate remains Day 28.

## Reproduction

```bash
python -B scripts/test_evidence_selector.py
python -B scripts/build_candidate_evidence_demo.py
python -B scripts/search_evidence.py --paper-id paper-001 --query "이 논문에서 다루는 주요 문제는 무엇인가?"
python -B scripts/validate_day23_day24.py
```
