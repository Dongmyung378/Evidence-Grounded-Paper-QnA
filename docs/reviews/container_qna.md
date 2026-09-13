# Docker Compose Q&A Benchmark

[English](container_qna.md) | [한국어](container_qna_KO.md)

## Purpose

This benchmark checks the portfolio delivery path, not just saved retrieval output. A fresh Docker Compose project built the backend and UI, uploaded three PDFs through HTTP, analyzed them, asked ten fixed questions, restarted with network model loading disabled, and then removed its temporary runtime resources.

The question set contains five English and Korean pairs from three verified benchmark papers. Gold answers were not loaded during container execution. They were used only after output generation for semantic review.

## Environment and contract results

| Check | Result |
|---|---:|
| Docker Engine | 28.5.2 |
| Docker Compose | 2.40.3-desktop.1 |
| Backend and UI health | Passed |
| UI HTTP response | 200 |
| Papers uploaded and analyzed | 3/3 |
| Questions completed | 10/10 |
| Correct response language | 10/10 |
| Supported answers with traceable evidence | 10/10 |
| Validation fallbacks | 0 |
| Offline cached restart | Passed |
| Deterministic restart response | Passed |
| Temporary stack and runtime cleanup | Passed |

The benchmark used seed 378. All embedding, reranking, and translation inference ran on CPU inside the backend container.

The three analyses recorded 91 non-fatal parser diagnostics: 31 two-column suspicions, 48 long-segment splits, and 12 retained footnote-like lines. All three jobs still completed with 49/49 text pages and 195 traceable chunks. These counters describe extraction decisions and are not failed requests.

## Runtime

| Measurement | Result |
|---|---:|
| Backend and UI build | 111.999 s |
| First paper analysis with empty model cache | 106.471 s |
| Later paper analysis mean | 6.027 s |
| English answer API mean | 1.470 s |
| English answer API maximum | 1.650 s |
| Korean answer API mean | 11.950 s |
| Korean answer API maximum | 23.895 s |
| Korean to English API mean ratio | 8.13x |
| Offline restart stack start | 12.650 s |
| Repeated Korean answer after restart | 11.912 s API, 21.282 s wall |
| Persistent model cache | 6.46 GiB |

The first analysis includes downloading and initializing the pinned models in an empty persistent cache. The next two papers completed analysis in 4.156 and 7.898 seconds. The restart used local model files only, preserved all three processed papers, produced the same answer and evidence pages, and did not change the cache inventory.

## Semantic answer review

| Language | Pass | Partial | Fail | Pass or partial |
|---|---:|---:|---:|---:|
| English | 3 | 2 | 0 | 5/5 |
| Korean | 1 | 4 | 0 | 5/5 |
| Total | 4 | 6 | 0 | 10/10 |

Manual inspection found that all ten returned evidence sets support their answers. Partial results mainly came from omitted details, noisy extracted headers, or awkward Korean translation. The review was assistant-led and was not an independent human evaluation.

Exact overlap with the single verified Gold page was 4/10. This live benchmark therefore does not use exact Gold page equality as its citation-support criterion. The returned passages are traceable and semantically supportive, but the page mismatch must remain visible before making live page-recall claims.

## Language decision

Korean remains in the portfolio MVP. All five Korean requests returned Korean text and supporting English source evidence, and none was semantically judged a complete failure. However, Korean CPU generation averaged 11.950 seconds and four answers were partial, often because wording was awkward or a detail was omitted. This is a documented portfolio limitation, not a production latency or translation-quality guarantee.

This run contains only answerable questions. Abstention quality remains supported by the separate unsupported holdout result of 10/10 refusals.

## Reproduce and validate

The full run downloads models when the persistent cache is empty and can take several minutes.

```bash
python -B scripts/run_container_qna_benchmark.py
python -B scripts/build_container_qna_review.py
python -B scripts/validate_container_qna.py
```

The benchmark uses a dedicated Compose project. It removes only that project's temporary runtime volume and preserves its model-cache volume for repeat runs.
