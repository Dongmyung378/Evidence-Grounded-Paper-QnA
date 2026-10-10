# Question facets and source sentence alignment

[English](evidence_planning.md) | [한국어](evidence_planning_KO.md)

The implemented experimental path tracks explicit requests and restores source context across production chunks. Accepted answers expose correspondence between each selected English source sentence and its answer text. These mechanisms make missing information and source use inspectable. Relevance scores remain heuristic signals rather than proof of scientific entailment.

## Request planning

`scripts/answer_planner.py` decomposes explicit compound questions into at most three facets using the question text. The existing pinned cross-encoder ranks source candidates for each facet. Candidate prioritization and larger compound-question budgets are configurable experiments. None of the three core DEV variants met the adoption criteria. The released default disables facet selection, context expansion, larger answer budgets and retrieval retries, while retaining source alignment. Diagnostics record selected sentence IDs and unresolved facets without reading evaluation facts.

For a compound question with an unresolved facet, the pipeline can run one additional search using the missing request together with the question subject. It accepts at most two new source objects from the same paper and keeps the five-citation limit. The retry replaces the first answer only when the new response passes validation and covers more facets. Otherwise the original response is retained. This experimental recovery path is disabled in the released default. Relevance does not establish entailment. This is a bounded recovery mechanism, not an open-ended agent loop.

## Source context

`scripts/context_windows.py` restores at most one preceding and one following chunk from the same page and section. Exact overlap is removed; no source text is silently truncated. The window records original chunk IDs, source hashes and the merged-text hash. A citation to a restored passage includes the source chunk IDs in its locator.

An optional, disabled setting can consider unclassified body text after page one when a document parser labels mathematical prose as front matter. Reference filtering remains active. Reported scalar measurements in prose, such as a correlation coefficient, are distinguished from standalone equations; this does not implement formula reasoning.

## Bilingual alignment

Accepted outputs include `answer_alignment`. Each record pairs a selected source sentence with its answer text, page and evidence ID. A translator can render one source sentence as multiple Korean sentences; alignment is by source translation unit. Korean translations retain the existing numerical, unit, identifier and qualifier checks. `structural_checks_passed` describes those checks and does not certify full semantic equivalence.

The public API exposes typed alignment records, and the UI offers a sentence-by-sentence source comparison. Insufficient answers do not expose alignment as if it were an accepted answer. Aligned source text uses the existing PDF ligature, whitespace and heading normalization. It is not a byte-identical raw chunk substring; public validation rebuilds the same extraction to verify the correspondence. Scientific attribution can be reviewed against the source using the [AIS annotation framework](https://aclanthology.org/2023.cl-4.2/).

## Evaluation

The [30-paper benchmark](../data/benchmarks/corpus30/README.md) separates ten development papers from twenty evaluation papers and preserves paired English/Korean groups. Final adoption depends on measured development gains, refusal behavior and latency. See [results](results.md) for the selected configuration and retained limitations.

## Experimental configuration

The shipped configuration disables `enabled`, `restore_context`, `require_all_facets`, `prioritize_facets` and `expanded_answer_budget`, and sets `maximum_retrieval_retries` to zero. To inspect the implemented recovery path in an isolated DEV experiment, set `enabled` and `restore_context` to true and `maximum_retrieval_retries` to one in a copied runtime configuration. Prioritization, answer expansion and forced coverage refusal are separate options. These settings were not accepted as production defaults; inspect their recorded regressions before adopting them.

The source-alignment API and UI are active independently of these experimental switches. Scalar measurement prose is retained under the existing evidence, numerical and output checks. Standalone equation reasoning remains unsupported.
