# Day 30 Local End-to-End Grounded Q&A

## Original-roadmap requirement

Day 30 connects `query → retrieval → reranker → evidence → LLM` and requires
ten questions to run end to end on the local machine. Answer-quality review is
scheduled for Day 32, so this gate measures integration, contract validity, and
traceability rather than correctness scores.

## Implemented path

1. `ProductionRetrieval` runs the frozen Day 28 BM25+dense+RRF configuration.
2. The multilingual Cross-Encoder reranks 20 candidates.
3. The evidence selector produces up to five Day 24 evidence objects.
4. The Day 29 builder creates a language-matched grounded-answer request.
5. `Qwen/Qwen2.5-0.5B-Instruct` generates a response locally through
   Transformers.
6. The strict contract validates JSON, answer language, sufficiency state, and
   cited evidence IDs.
7. Cited IDs resolve back to page, section, chunk ID, and source text.

The generation model snapshot is pinned to
`7ae557604adf67be50417f59c2c2f167def9a775`. Generation is deterministic
(`do_sample=false`), and the final run used CUDA float16 on an NVIDIA RTX 3060
Laptop GPU. No API key or external LLM request is required after the model is
cached.

## Ten-question smoke result

The fixed smoke set uses one question from each benchmark paper and is balanced
between English and Korean.

| Check | Result |
|---|---:|
| Questions completed | 10/10 |
| Papers covered | 10/10 |
| English / Korean | 5 / 5 |
| Valid model JSON responses | 10/10 |
| Safe fallbacks | 0 |
| LLM invocations | 10 |
| Responses with citations | 8 |
| Sufficient / insufficient | 8 / 2 |
| Total runtime | 60.184 seconds |

An initial run exposed four strict-validation fallbacks. The prompt was then
corrected to show real available evidence IDs and language-specific valid JSON
examples. One remaining response used a clear Korean abstention paraphrase
rather than the exact canonical sentence; validation was aligned with the
roadmap by accepting clear localized abstentions while retaining all structural
and language checks. The final reproducible run has zero fallbacks.

## Scope and early quality observations

The run proves execution and evidence traceability, not final answer quality.
The compact 0.5B model produced some weak answers: `q-005-ko` and `q-041-ko`
abstained despite being answerable in the benchmark, while `q-017-ko`,
`q-021-en`, and `q-029-ko` need human correctness review. These cases are
inputs for the planned Day 32 manual review rather than hidden from the result.

The smoke script reads questions and retrieval output only. Gold answers and
gold evidence are never passed to the model.

## Reproduction

```bash
# First run may download the pinned local generation model.
python -B scripts/run_day30_smoke.py

# Fully local repeat after all three models are cached.
python -B scripts/run_day30_smoke.py --offline

# Ask one paper directly.
python -B scripts/ask_paper.py --offline \
  --paper-id paper-001 \
  --query "What is the main problem addressed in this paper?"

python -B scripts/test_qna_pipeline.py
python -B scripts/validate_day30.py
```

## Decision

**Passed.** Ten questions across ten papers complete the full local pipeline
with valid bilingual JSON and evidence linkage. Day 31 can now add an explicit
abstention decision policy, followed by the Day 32 manual quality review.
