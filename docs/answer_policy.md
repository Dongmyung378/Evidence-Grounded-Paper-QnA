# Evidence-first answer policy

[English](answer_policy.md) | [한국어](answer_policy_KO.md)

This document describes the default evidence-first path. English answers quote
selected source sentences; Korean answers translate those sentences with the
local translation adapter. The policy uses retrieved text rather than an answer
key. Its checks reduce observable errors but do not establish scientific truth.
See the [final-answer quality benchmark](quality_benchmark.md) for evaluation
inputs, counting rules and the remaining independent review requirements.

## Processing order

After the retrieval score gate, answer generation checks explicit question
premises, tests the requested aspect against the retrieved evidence, selects
source sentences, constructs or translates the answer, and validates the final
response. A policy refusal returns the canonical `insufficient` response with
no citations. Translation inference is skipped for these early refusals; this
does not imply that a translation model was never initialized.

An intentional premise or aspect refusal sets `fallback_used` to `false`.
Scoring errors, unusable source context and failed answer or translation
validation are failures handled by a safe refusal with `fallback_used: true`.
Inspect the reason code and final response together when counting outcomes.

## Preserve the requested evidence

Generic question rules recognize lists, comparisons, numeric results,
limitations and evaluation questions in English and Korean. Evaluation wording
such as "how was it validated" or "what experiments" uses a larger source budget
to retain tested objects, baselines, metrics and recognized conditions when
available. An ordinary single-method question is not
automatically treated as a request for an exhaustive list.

The evidence selector retains its default two-chunk page cap for ordinary
questions. Multi-part question types can use the configured evidence budget on
one page. Near-duplicate removal preserves differences in quantities, technical
identifiers and observable comparison direction instead of discarding a second
result just because much of its wording overlaps.

Primary retrieval retains at most five evidence objects with their ranks and
scores. If an explicit prediction-task facet needs plain prose absent from those
objects, answer context can add at most one full source chunk from the same
paper's already reranked 20-candidate pool. It does not replace or rerank the
primary objects, fetch a new chunk or invent the missing task. If the required
task support is unavailable, the answer is refused.

The selector records `evidence_selection.answer_context_supplements` and
`evidence_selection.answer_context_supplement_origins`, including the original
candidate rank and source-text hash. The pipeline records
`pipeline.evidence_context` with `primary_retrieval_count`,
`primary_retrieval_evidence_ids`, `supplement_count`, `answer_context_count` and
`supplement_origins`. Its `retrieved_evidence` contains the complete answer
context, while `cited_evidence` contains only the validated citations actually
used. The HTTP `/question` response exposes those citations as `evidence`.

Sentence selection groups numbered list items with their heading and keeps
adjacent qualifications or negative evaluation conditions where recognized.
Those neighbors must already exist within the supplied evidence record. The
selector does not fetch missing chunks, reconstruct a missing list, or change
evidence IDs. A grouped condition is not silently removed to fit a smaller
budget. Character and sentence limits still bound the available context.

Current starting source limits are defined in
[answer_generation.json](../config/answer_generation.json):

| Question shape | English source sentences | Korean source sentences |
| --- | ---: | ---: |
| Default | 3 | 2 |
| List | 8 | 8 |
| Comparison | 4 | 4 |
| Numeric | 3 | 3 |
| Limitations | 4 | 4 |
| Evaluation | 4 | 4 |

An adjacent-context group can extend the initial source selection up to the
configured answer limit. These budgets do not guarantee complete answers. The
primary retrieval limit is five evidence records, the answer-context limit is
six with the bounded supplement, and the citation limit remains five. Sentence
selection still uses at most eight sentences. Retrieval can miss required facts,
and extraction can discard unusable fragments.
For evaluation questions, available anchors with recognized evaluation cues
are selected with their adjacent conditions. Once such anchors exist, unrelated
method or hyperparameter sentences do not fill unused slots. A four-sentence
budget can therefore produce an answer from two relevant experiment sentences.
Cue matching is lexical and does not prove that a sentence answers the question.
Extracted section headings can also remain attached to otherwise usable prose;
the current heading cleanup does not remove every form of header noise.

## Translation and final-output checks

The default local translation path requires one non-empty output per source
input. Each output is checked against its own source, which detects exchanging
numbers between two sentences even when the combined number set is unchanged.

- Decimal values are canonicalized, including `1.0 = 1` and `1e-3 = 0.001`.
  Supported magnitude expressions and percent or multiplier notation are kept.
- Common measurement units retain their identity. Recognized aliases such as
  `second` and `초` are permitted. Arbitrary unit conversion is not implemented.
- Recognized Latin technical identifiers must remain visible. Acronyms,
  CamelCase names and model identifiers are checked after the local adapter
  restores their protected source spans. Missing protected spans are refused,
  not guessed from the question or an answer key.
- Explicit negation and scope markers are checked conservatively. Source
  `not`, `cannot` or `only` must retain a recognizable Korean negative or
  restriction. Examples include `않습니다`, `없습니다`, `오직`, `에만`,
  `에서만`, `뿐`, `불과`, `유일` and `단지`. A standalone `단` is distinguished from a word
  such as `단계`. Upper bounds, lower bounds and exceptions have separate
  marker checks. The idiom `not only` is not treated as ordinary negation.

These are observable marker checks, not a translation entailment model. A wrong
sentence can retain a negative word or all its numbers and still pass. A correct
paraphrase without a recognized marker can be refused. Lexical checks do not
resolve every negation attachment, implied condition, quantity association or
technical-name transliteration.

The [source-only terminology glossary](../config/translation_glossary.json)
addresses observed development errors such as `robot gripper` translated as
`로봇 지갑` and `ground truth` translated as `토지 진실`. Its explicit mappings
include `robot gripper` to `로봇 그리퍼` and `ground truth` to `참값`. Longer
registered phrases, such as `ground-truth labels` to `정답 레이블`, take
precedence. Matching ignores English case and requires whole word boundaries.
A mapping is used only if its source phrase occurs in the selected sentence.
The glossary restoration step does not insert a term, model name, explanation
or quantity that was absent from that source.
General learning terminology also includes `weight decay` to `가중치 감쇠`,
`temperature scaling` to `온도 보정`, `confidence calibration` to `신뢰도 보정`,
and registered self-supervised, contrastive, augmentation and distillation
phrases. These mappings address terminology observed in development sources;
they do not establish translation quality on unseen papers.
The glossary also distinguishes contrastive prediction tasks, Gaussian blur,
random cropping, color distortion and pretext tasks. Calibration noun, verb and
adjective forms are mapped separately, with longer phrases such as
`not calibrated` taking precedence. Restoring a technical term does not prove
that the model attached it correctly or conjugated the resulting Korean sentence
fluently.

Before NLLB inference, the [local adapter](../scripts/local_translation.py)
protects matched terms, recognized technical identifiers and surface-form
English names with at least two title-case words. `Shadow Hand` is protected
because it occurs in the source, not because a paper-specific name list contains
it. Single title-case names immediately preceding `dataset`, `benchmark` or
`corpus` are also protected by their explicit data-resource context. This is a
surface heuristic and can protect a capitalized common noun incorrectly.
When another sentence in the same supplied translation batch explicitly
declares such a name, its exact spelling is protected wherever it occurs in
that batch. This uses only the supplied source text, not a paper-name list or
external entity lookup. Missing comma or semicolon spaces between Latin words
are restored before masking; numeric punctuation and recognized URLs remain
unchanged. This formatting does not discard or rewrite a source clause.
Each occurrence gets a separate alphabetic marker such as `ZXQAAA`; these
markers introduce no numerical quantities. The output must preserve exactly
one occurrence of every marker and contain no extra or damaged marker. The
adapter then restores the registered Korean term or the exact original Latin
name. For an all-caps glossary phrase also recognized as an acronym, it retains
the source spelling in parentheses. A limited Korean particle agreement rule
uses the restored Korean noun's ending; Latin-name particles are unchanged.
Ordinary numbers remain visible to the model. For explicitly written percentage
bounds, only the relation words are masked and the numeric literal and percent
label remain visible. `below 1%` restores as `1% 미만`, whereas `at most 1%`
restores as `1% 이하`. Restoration removes an immediately adjacent equivalent
percent literal to avoid duplicating the source value. A changed quantity or
an intervening relation is not removed; quantities on both sides of the same
marker are refused as ambiguous. Strictness, direction and the written numeric
literal come from that source; `percent` and `%` are equivalent labels.
This protection does not infer unwritten values, calculate quantities or parse
arbitrary units. All restored values must still pass the shared decimal, unit
and numeric-bound checks. The adapter also refuses untranslated CJK output
and English predicate copies before caching, including copies whose glossary
restoration alone added Korean text. Latin-name or quantity-only fragments
remain permitted; the complete answer still undergoes the final language
check. A failed batch adds no new cache entries.

Input tokenization does not silently truncate. An input above the configured
token budget is refused, and a generated sequence without a terminal end token
is refused. A terminal end token does not prove that every source clause was
translated; an early model ending or an omitted unrecognized condition can
still escape these lexical checks. Unregistered terms, single-word names outside
the recognized data-resource context, and
names outside the surface-form recognizer can still be mistranslated. Protected
marker copying can also fail and cause a refusal. These mechanisms do not
guarantee complete terminology coverage, fluent translation or semantic
equivalence.

The primary deterministic translation uses one beam. With
`retry_on_validation_failure: true` and `retry_num_beams: 2` in
[translation.json](../config/translation.json), only a sentence that failed
marker restoration, language or source-fidelity validation is retried once with
the same protected input and two beams. The original checks remain mandatory.
Token-budget, model-batch shape and other structural failures do not trigger
this retry. Historical configs without the flag retain the single-attempt
behavior. A retry can still omit a term or restriction and is then refused.
`metadata()['validation_retry']` reports cumulative attempts, validated
recoveries, failures and the most recent request's attempted beam settings.
Recovery counts refer to individual sentences, even if another sentence makes
the entire batch fail and prevents all new cache entries. Retry time is part of
translation latency; this mechanism is not a general speed improvement.

`TranslationFidelityError.translation_diagnostics` carries bounded source,
protected input, invalid model output, restored output and failure-stage details
when available. These are for explicitly requested debug output; the ordinary
error message and refusal response do not publish invalid translations. Source
snapshots and raw development outputs must be retained separately when measuring
changes to preprocessing, terminology or validation.

The adapter keeps a bounded in-process LRU cache, with a default capacity of
128 validated sentences and a maximum configurable capacity of 256. Capacity
zero disables it. Keys include the exact input, model revision, translation
configuration fingerprint, glossary fingerprint, protection policy version and
the batch-declared names actually present in that input. The latter prevents
reuse of a translation produced without the current name protection context.
Duplicate inputs in a batch share one model call while retaining output order.
`clear_cache()` removes cached translations, and `metadata()['cache']` reports
hits, misses, size, capacity, evictions and batch duplicates. A repeated
sentence served from cache measures reuse rather than new translation
inference. Compare new inference with the cache disabled or explicitly cleared
while keeping model initialization outside the measured interval. The
[translation fidelity tests](../tests/test_translation_fidelity.py) exercise
restoration failures, numeric and scope rejection, token budgets and bounded
cache behavior without downloading a model.

Final validation also checks numeric support, answer length, language, output
shape and citation IDs. A valid ID establishes a source locator, not that every
claim is fully supported. Failed validation returns a canonical refusal rather
than publishing the invalid translation.

## Check explicit question premises

Question premises are compared with all retrieved evidence before source
sentence selection. An explicit unsupported percentage, multiplier, measurement
or asserted count is refused. Asking what a quantity is, or requesting a list
of three inputs, does not itself assert an observed measurement.

Latin subject identifiers are checked when question wording makes a model,
method or dataset explicit. Ordinary Korean transliterations are not required
to exactly match English spelling. Worldwide comparison claims require both an
explicit worldwide expression and a universal comparison; `global average
pooling`, an `any` yes/no determiner, and every baseline in a finite `global
benchmark` do not trigger that scope rule.

An unsupported question such as “Is accuracy 99%?” can be conservatively refused
instead of producing a corrective explanation. These checks establish absence
of observable premise support in retrieved text. They do not prove that the
premise is false in the full paper or verify a causal explanation.

## Requested-aspect relevance gate

[answerability_policy.py](../scripts/answerability_policy.py) derives topical
anchors from repeated names with title or model context in the supplied source.
The `source_derived_topic_debias_v2` transform preserves question grammar,
replaces recognized named title phrases with their generic subject and masks
the same anchors in a scoring-only copy of the evidence. Scientific uppercase
acronyms are preserved. A focused question can remove an instrumental clause
while retaining the requested result. The original question and evidence
remain intact.

The pinned cross-encoder scores both the full transformed question and the
focused request against each evidence item. It takes the highest evidence score
for each question view, then the lower of those maxima:

```text
aspect_score = min(max(score(full_question, evidence_i)),
                   max(score(focused_request, evidence_i)))
```

Identical question views are scored once. The minimum prevents a relevant
instrument or method name from replacing support for the requested result.
A question without a specific remaining aspect is not refused by this score
alone. Missing or non-finite scores cannot establish a supported answer.

The enabled threshold in
[answerability_policy.json](../config/answerability_policy.json) is
`-3.405527949333191`, accepted when the score is greater than or equal to the
threshold. The [v2 calibration report](../data/quality_benchmark/runs/2026-10-10-v2/answerability-dev-calibration.json)
uses **20 synthetic DEV language cases: five answerable meaning groups and five
unsupported or false-premise groups**, plus **18 previously declared real-paper
DEV cases as answer-retention constraints**. The latter are development inputs,
not an independent evaluation. The original 20-case real-paper subset excludes
`q-001-en` and `q-001-ko` because the unchanged global retrieval threshold
`-3.35` already rejects them. This additional gate does not recover those two
false refusals.

At calibration time, the selected threshold retained all five synthetic
answerable groups and the 18 retention cases, while refusing four of five
negative groups. Bilingual variants are grouped rather than counted as twice
as many independent cases. Neither the relevance threshold nor the combined
calibration-time policy fully separates these DEV groups.

The [exact calibration source](../data/quality_benchmark/runs/2026-10-10-v2/calibration-sources/scripts/answerability_policy.py)
is preserved with SHA-256
`e8d552fe78f40d435789b57772f0dd5f9fff8b54e28aa24efc28983730b55a6b`.
Current runtime code differs from that source. Post-calibration refinements
normalize PDF ligatures and line-break noise in policy views, recognize
open-vocabulary clinical outcomes and research populations in context, check
bilingual privacy requests and distinguish requested measurement dimensions
from unrelated counts. Explicit positive component-use or improvement premises
are refused only when a matching source relation denies that use or preserves
the property; neutral and negative questions are protected.

These are deterministic rule refinements developed on disclosed development
evidence, including the initial real-paper DEV outputs. The score-query
construction, score formula and threshold did not change, and no model
recalibration was performed for these additions. Their unit tests and saved
feature checks do not establish final model accuracy. Earlier disclosed
synthetic HOLDOUT failures are now development material; the new real-paper
HOLDOUT must remain separate and use a frozen runtime before execution.

This calibration verifies limited integration and development retention.
A relevance score is neither a probability of correctness nor a semantic
support proof. Presence and premise rules are bounded heuristics and can still
miss paraphrases or necessary context. False refusals and unsupported retained
answers require final-output review. See the
[completion evaluation](completion_evaluation.md) for separately recorded
real-paper evaluation status; these calibration results are not independent
real-paper semantic accuracy.

## Inspect and verify

The internal result records the aspect decision, score, threshold and policy
fingerprint. Stage timings distinguish sentence selection, answer construction,
translation and validation. Aspect scoring is included in sentence selection
time; premise checks are included in validation time.

With test dependencies installed, run from the repository root:

```bash
python -m unittest discover -s tests -p test_answer_completeness.py -v
python -m unittest discover -s tests -p test_grounded_generation.py -v
python -m unittest discover -s tests -p test_answerability_policy.py -v
```

Unit fixtures and synthetic calibration do not demonstrate independent review,
third-party reproducibility or completion of the project improvement plan.
Follow the [quality benchmark procedure](quality_benchmark.md) to freeze new
real-paper inputs, evaluate final outputs and collect attributable human
judgments of factual correctness, completeness and citation support. Historical
frozen results describe their original implementation and must not be relabeled
as measurements of this policy.
