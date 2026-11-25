# Methodology & Design Decisions · Evidence-Grounded Paper Q&A v2

[한국어](methodology-and-decisions.ko.md) · [README](../README.md) · [Architecture](architecture.md) · [Results](results.md) · [Reproduction & Demo](reproduction.md)

Every modeling, retrieval, verification, training, and threshold decision in v2 was made from controlled diagnostics on the **DEV split (`10` papers, `120` bilingual outputs)** before running a single locked evaluation on the **unseen Holdout split (`20` papers, `240` bilingual outputs)**.

---

## Decision Progression Matrix

| Stage | Hypothesis / Core Question | Measured DEV Evidence (`120` outputs) | Decision |
| :--- | :--- | :--- | :--- |
| **1. Retrieval & Context (`Exp 1B`)** | Does upgrading from `E5-small` + `mMiniLM` + single-chunk to `BGE-M3` + `bge-reranker-v2-m3` + Dual-Score + Parent(1800) fix recall? | Stack A (`BGE-M3`): EN `38/40`, KO `33/40` gold-page Top-5; Stack B (`Qwen3-0.6B`): EN `40/40`, KO `36/40`; combined with Parent(1800) + KO→EN translation raised DEV source page hit to **`82.5%`** (`+26.2%p`, `p < 0.0001`). | **Adopt Stack A (`BGE-M3` + `bge-reranker-v2-m3`) + Dual-Score + Parent(1800)** (single-encoder efficiency + Apache-2.0) |
| **2. Guided JSON + Base 4B (`Exp 1C`)** | Can a local 4B model replace extractive+NLLB without breaking the strict JSON contract? | Free-form JSON had `5+` parse failures (`pair_ans 27/40`, `pair_facts 17/40`). Adding vLLM **Guided JSON (`json_object`) + `max_tokens=1024`** yielded **`120/120` valid**, **`40/40` `pair_ans`**, **`25/40` `pair_facts`** (`3.1×` v1 `8/40`), **`quote/num = 0.96/0.98`**, `2.06 s/q`. | **Adopt `Qwen3-4B-Instruct-2507` + Guided JSON** |
| **3. 14B Teacher Ceiling (`Exp 1C-T`)** | How much headroom does a larger `Qwen3-14B-AWQ` teacher provide over `Qwen3-4B-Instruct-2507`? | `14B-AWQ` (`3.73 s/q`): `pair_ans 39/40`, `pair_facts 24/40`, `en/ko 35/26`, `fp 6/4/4`, `quote 0.93` — **weaker on every metric than newer-gen BF16 `4B-2507`** (`40/40`, `25/40`, `35/27`, `fp 8/7/7`, `quote 0.96`) at `1.8×` slower speed. | **Key diagnostic finding**: newer-gen `4B-2507` outperforms older-gen 4-bit `14B-AWQ` |
| **4. Scientific NLI FP32 Fix (`Exp 2A`)** | Why did initial BF16 fine-tuning of `mDeBERTa-v3-base` collapse `Exp 2B` to `0/40`? | `mDeBERTa-v3` disentangled attention under `bf16=True` / `transformers 5.x` FP16 default suffered **NaN logit collapse** (`NaN >= 0.25 -> False`). Forcing **FP32 (`torch_dtype=float32, bf16=False`)** + `warmup_steps=100` restored clean convergence (`Val Loss = 0.1676`) and pushed DEV false-premise refusal from `7/10 → 10/10` (**100%**). | **Adopt FP32 Scientific NLI training + automated post-train sanity gate** |
| **5. Post-Hoc NLI Ablation (`Exp 2B`)** | Does adding 2-stage focused Post-NLI (`entail >= 0.25`) improve faithfulness beyond Pre-NLI? | False-premise refusal stayed `10/10/10` (zero gain), while Korean answerable retention dropped **`36/40 → 32/40`** and `ko_facts` dropped **`25 → 22`**. | **Reject Post-NLI (`nli_post_entail_min = None`)** |
| **6. 5-Chunk LoRA SFT Ablation (`Exp 3A`)** | Does distilling `547` 4-filter-verified 5-chunk EN/KO examples from `Qwen3-14B-AWQ` improve `4B-2507`? | Completion-only LoRA SFT (`r=32`, 2 epochs, `Val Loss 0.1190`, `95.88%` token acc) **regressed** vs Base 4B + Pre-NLI (`Exp 2A`): `pair_ans 35→32`, `ko_facts 25→22`, `quote 0.96→0.93`. Cause: distilling a **weaker quantized 14B teacher** (`quote 0.93`, 49.4% refusal mix, EN-only `atomic_facts` filter) pulled the stronger `4B-2507` student downward on Korean recall and quote accuracy (while raising EN `32→34`). | **Reject SFT & skip DPO** (avoid spending compute refining a regressed student; preserve pure Apache-2.0 `4B-2507`) |
| **7. NLI Threshold & Conformal Gate (`Exp 2C`)** | Can raising Pre-NLI threshold eliminate the `5`-pair over-refusal in `Exp 2A` (`0.70`) without losing `10/10` false-premise defense? | FP32 NLI achieved near-perfect separation on DEV (**Answerable median = `0.001`**, **False-Premise min = `0.999`**, `20/20 >= 0.999`). Sweeping `0.70 → 0.99` revealed a wide flat optimum across `0.90–0.99`: **`pair_ans 37/40` (`+2`), `pair_facts 24/40` (`+3`), `en/ko 35/26`, `fp 10/10/10`, `quote/num 0.99/0.97`**. Paper-grouped 5-fold OOF Conformal Gate reached **AURC = `0.0669`**. | **Adopt `Base 4B + FP32 Scientific NLI (thr=0.95)` + Conformal Gate** as Final v2 |

---

## Rejected Hypotheses Summary

| Rejected Component | Observed Failure Mode on DEV | Replacement Adopted |
| :--- | :--- | :--- |
| Free-form JSON generation (no `response_format`) | `5+` malformed JSON fallbacks dropped `pair_ans` to `27/40` and `pair_facts` to `17/40` | Guided JSON (`response_format: {"type": "json_object"}`) + `max_tokens=1024` (`0` parse errors) |
| `mDeBERTa-v3` BF16 / FP16 mixed-precision fine-tuning | Disentangled attention NaN collapse (`Exp 2B` fell to `0/40` answers) | Strict **FP32** training (`bf16=False, fp16=False, warmup_steps=100, max_grad_norm=1.0`) + ONNX INT8 export |
| 2-Stage Focused Post-NLI (`entail >= 0.25`) | Lost `4` Korean answerable cases (`36 → 32`) with `+0` false-premise gain | Pre-Gate Scientific NLI only (`nli_post_entail_min = None`) |
| Teacher-14B-AWQ Reverse-Synthesis SFT (`Exp 3A`) | Weaker quantized teacher (`24/40`, `quote 0.93`) + 49.4% refusal balance degraded `4B-2507` Korean recall (`25 → 22`) and `pair_ans` (`35 → 32`) | Zero-shot Base `Qwen3-4B-Instruct-2507` + Fine-tuned Scientific NLI (`thr=0.95`) |
| Direct `arxiv.org` PDF scraping on Colab cloud IPs | Silent rate-limit/blocking returned `0` external training papers | Zero-overlap Hugging Face Parquet streaming (`ccdv/arxiv-summarization`: `150` train papers, `30` frozen `corpus-v2-holdout` validation papers) |

---

## Leakage & Evaluation Hygiene
1. **Zero Training-Eval Overlap**: All `150` external papers used to train the Scientific NLI model (`7,729` examples) and SFT ablation (`547` examples) were filtered against all `30` `corpus30` titles and arXiv IDs.
2. **DEV-Only Model & Threshold Selection**: Every architecture change, ablation rejection (`Exp 2B`, `Exp 3A`), NLI threshold (`0.95`), and 5-fold paper-grouped Conformal Gate was fixed strictly on DEV (`10` papers, `120` outputs).
3. **Single Locked Holdout Pass (`Exp 4`)**: The `20` Holdout papers (`240` outputs) were evaluated once after freezing the v2 winner configuration, verified with Exact Two-Sided McNemar tests and `2,000`-iteration Paper-Cluster Bootstrap 95% CIs.
4. **Fresh Future Holdout (`corpus-v2-holdout`)**: Thirty unseen papers (`677` chunks) from `ccdv/arxiv-summarization:validation` were cryptographically frozen with SHA-256 checksums (`data/corpus_v2_holdout/freeze_v2.json`) for future regression testing.
