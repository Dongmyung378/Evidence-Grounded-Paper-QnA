# Evaluation Results & Local Hardware Verification

> **Language**: **English** | [한국어 (Korean)](results.ko.md)

All results below correspond to the frozen winning configuration (`Exp 2C`: BM25 + `BAAI/bge-m3` + Dual Score `bge-reranker-v2-m3` + Parent `1,800` chars + FP32 `mDeBERTa-v3` Scientific NLI at `0.95` + zero shot `Qwen/Qwen3-4B-Instruct-2507` Guided JSON + Paper Grouped 5 Fold OOF Conformal Gate).

---

## 1. Locked Unseen `corpus30` Holdout (`20` Papers, `240` Outputs)

![Unseen Holdout Bilingual Performance and Semantic Verification](figures/en/holdout-performance.png)

Exact slice breakdown from [`reports/holdout_breakdown.csv`](../reports/holdout_breakdown.csv) and [`reports/final-summary.json`](../reports/final-summary.json):

| Metric (Unseen Holdout, 20 Papers) | English (`120`) | Korean (`120`) | Paired Strict (`120` pairs) | All Outputs (`240`) |
| --- | ---: | ---: | ---: | ---: |
| Contract Valid | `120/120` (`100.0%`) | `120/120` (`100.0%`) | `120/120` (`100.0%`) | **`240/240` (`100.0%`)** |
| Answerable Retained (`80` pairs) | `74/80` (`92.5%`) | `68/80` (`85.0%`) | **`68/80` (`85.0%`)** | **`142/160` (`88.8%`)** |
| All Fact Aliases Matched (`80` pairs) | `54/80` (`67.5%`) | `41/80` (`51.3%`) | **`33/80` (`41.3%`)** | **`95/160` (`59.4%`)** |
| Acceptable Source Page Hit (`80` pairs) | `66/80` (`82.5%`) | `57/80` (`71.3%`) | **`56/80` (`70.0%`)** | **`123/160` (`76.9%`)** |
| Unsupported Refused (`20` pairs) | **`20/20` (`100.0%`)** | **`20/20` (`100.0%`)** | **`20/20` (`100.0%`)** | **`40/40` (`100.0%`)** |
| False Premise Refused (`20` pairs) | **`20/20` (`100.0%`)** | **`19/20` (`95.0%`)** | **`19/20` (`95.0%`)** | **`39/40` (`97.5%`)** |
| Verbatim Quote Verified Rate | `0.878` | `0.884` | **`0.881`** | **`0.881`** |
| Canonical Numeric Fidelity Rate | `0.946` | `0.957` | **`0.951`** | **`0.951`** |
| Mean Wall Latency (`vLLM`, NVIDIA L4) | `1.81 s/q` | `1.74 s/q` | **`1.77 s/q`** | `545.3s` total |

### Semantic Judge Rubric Evaluation (`Qwen3-14B-AWQ`)

To complement strict lexical alias matching, `Qwen3-14B-AWQ` scored semantic completeness and faithfulness against the gold rubric and retrieved evidence passages:

| Split & Run | EN Completeness | KO Completeness | Paired Completeness | Mean Faithfulness | Audit Sheet |
| --- | ---: | ---: | ---: | ---: | --- |
| DEV `Exp 1C` (`80` answerable outputs) | `0.882` | `0.842` | `0.775` | `0.867` | [`exp1c_qwen3_4b_2507_human_audit.csv`](../reports/exp1c_qwen3_4b_2507_human_audit.csv) |
| **Holdout Final (`160` answerable outputs)** | **`0.849`** | **`0.711`** | **`0.690`** | **`0.888`** | [`exp4_v2_final_holdout_human_audit.csv`](../reports/exp4_v2_final_holdout_human_audit.csv) |

---

## 2. Paper Grouped 5 Fold OOF Conformal Gate Calibration

Calibrated exclusively on DEV (`10` papers, `120` outputs, `GroupKFold(5)` by `paper_id`) in [`reports/calibrated_gate.json`](../reports/calibrated_gate.json):

- **OOF AURC**: **`0.0669`**
- **Scientific NLI Pre Gate Threshold**: **`0.95`**

| Risk Tolerance `alpha` | OOF Probability Threshold | Answerable Coverage | Negative Leak Rate | Operating Mode |
| ---: | ---: | ---: | ---: | --- |
| `0.02` | `0.8993` | `86.3%` | `2.5%` | Strict Scientific Audit |
| **`0.05`** | **`0.8727`** | **`88.7%`** | **`5.0%`** | **Recommended Default** |
| `0.10` | `0.4196` | `98.8%` | `10.0%` | High Coverage Exploration |

---

## 3. Local Windows RTX 3060 & ONNX Parity Gate Audit

Executed locally on **Windows 11 + NVIDIA GeForce RTX 3060 Laptop GPU (`6,144 MiB`)** via [`scripts/verify_local_runtime.py`](../scripts/verify_local_runtime.py) and logged in [`reports/local_verification.json`](../reports/local_verification.json):

| Verification Target | Metric / Outcome | Status |
| --- | --- | :---: |
| `corpus_v2_holdout` SHA-256 Integrity | `chunks.jsonl` & `manifest_v2.json` (`30` papers, `677` chunks) `2/2` exact match | **PASS** |
| Local RTX 3060 FP32 PyTorch NLI (`mDeBERTa-v3`) | `1.048 GiB` VRAM (`1.610 GiB` peak / `6.0 GiB`), `26.47 ms/window` (`529.39 ms` / 20 wins), padding invariance **`abs_diff = 0.000000`** (`0.948831` vs `0.948831`), sample FP `0.9986` | **PASS** |
| Legacy TorchScript ONNX INT8 (`322.67 MiB`, CPU 4T) | Padding drift **`abs_diff = 0.037309 > 0.02`** (`0.061635` vs `0.098943`), caught only `2/20` Holdout FPs at `0.95`, `275.78 ms/window` (`2,165.64 ms/q`) | **AUTO-BLOCKED** |
| Automated Test Suite (`pytest -v`) | `9/9` passing (PDF repair, verifier, NLI windows, BM25/RRF/Parent, Conformal Gate, SHA-256 freeze, FastAPI routes, live borderline gates, parity gate) | **PASS** |

---

## 4. Live Local Interactive Demo Verification (English UI)

![Local FastAPI Interactive Web Demo in English UI](figures/en/demo_ui_live.png)

Verified in a live desktop browser against `http://127.0.0.1:18080/?lang=en`:
- Zero console errors and zero failed network requests (`5/5` HTTP 200).
- Preset `paper-030-baselines-en` renders `Verified · Evidence Grounded Answer`, NLI contradiction `0.0044`, Conformal `P(Ans) = 0.993` (`alpha=0.05`, `thr=0.873`), `100% / 100%` quote and numeric check, `3` aligned claims, and `<mark>` highlighted Parent `1,800` char source passages.
