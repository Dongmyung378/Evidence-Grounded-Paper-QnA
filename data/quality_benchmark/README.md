# Quality evaluation artifacts

[English](README.md) | [한국어](README_KO.md)

This directory preserves the public inputs and measured outputs needed to reproduce and inspect answer behavior.

- `papers/` and the root fixture manifest/questions are authored integration fixtures.
- `real_paper_sources/`, `real_paper_manifest.json` and `real_paper_questions.jsonl` contain publisher-text sources and bilingual development/evaluation questions.
- `deployment/` contains a licensed original PDF and HTTP cases.
- `runs/2026-10-10-final/` contains the frozen measured runtime, final answers, reports and latency evidence.
- `runs/answer-builder-pilot/` contains the predeclared candidate model comparison and its stopping decision.
- `runs/2026-10-10/` and `runs/2026-10-10-v2/` retain only calibration provenance required by public verification and the runtime policy.
- `reproduction/` contains environment and answer comparisons from agent-operated Linux reproduction.

Other intermediate runs and host-specific logs are excluded from Git. Dates identify measured artifacts, not work journals. Preserve frozen files byte-for-byte; `.gitattributes` disables line-ending conversion. See [current results](../../docs/results.md), [experiments](../../docs/experiments.md) and [benchmark design](../../docs/quality_benchmark.md) for scope and limitations.
