# 방법론 및 설계 의사결정 · Evidence-Grounded Paper Q&A v2

[English](methodology-and-decisions.md) · [README](../README.ko.md) · [아키텍처](architecture.ko.md) · [결과](results.ko.md) · [재현 및 데모](reproduction.ko.md)

v2의 모든 검색, 검증, 학습, 임계값 결정은 **DEV 집합(논문 10편, 120출력)**에서의 통제 실험을 통해서만 확정했으며, **미공개 Holdout 집합(논문 20편, 240출력)**은 우승 구성이 완전히 동결된 뒤 단 1회만 평가했습니다.

---

## 단계별 실험 진행 및 의사결정 매트릭스

| 단계 | 핵심 검증 질문 | DEV 실측 근거 (`120` 출력) | 채택 / 기각 판단 |
| :--- | :--- | :--- | :--- |
| **1. 검색·문맥 업그레이드 (`Exp 1B`)** | `E5-small` + 단일 청크를 `BGE-M3` + `bge-reranker-v2-m3` + Dual-Score + Parent(1800)로 바꾸면 검색 병목이 해소되는가? | Stack A(`BGE-M3`) Top-5 정답 페이지: EN `38/40`, KO `33/40` · Stack B(`Qwen3-0.6B`): EN `40/40`, KO `36/40`. Parent(1800) + 한국어 영문 검색 번역 결합 시 DEV 정답 페이지 인용률 **`82.5%`**(`+26.2%p`, `p < 0.0001`). | **Stack A(`BGE-M3` + `bge-reranker-v2-m3`) + Dual-Score + Parent(1800) 채택** |
| **2. Guided JSON + Base 4B (`Exp 1C`)** | 4B 로컬 생성 모델이 엄격한 응답 계약을 깨지 않고 추출+NLLB를 대체할 수 있는가? | 자유 생성 JSON은 `5+`건 파싱 실패(`pair_ans 27/40`, `pair_facts 17/40`) 발생. vLLM **Guided JSON(`json_object`) + `max_tokens=1024`** 적용 시 **계약 `120/120`**, **`pair_ans 40/40`**, **`pair_facts 25/40`**(v1 `8/40`의 `3.1배`), **`quote/num 0.96/0.98`**, `2.06 s/q`. | **`Qwen3-4B-Instruct-2507` + Guided JSON 채택** |
| **3. 14B 교사 상한선 (`Exp 1C-T`)** | `Qwen3-14B-AWQ` 교사가 `4B-2507`보다 얼마나 높은 상한선을 보이는가? | `14B-AWQ`(`3.73 s/q`): `pair_ans 39/40`, `pair_facts 24/40`, `en/ko 35/26`, `fp 6/4/4`, `quote 0.93` — **신형 BF16 `4B-2507`(`40/40`, `25/40`, `35/27`, `fp 8/7/7`, `quote 0.96`)보다 모든 지표에서 열세**이면서 `1.8배` 느림. | **핵심 실증 발견**: 신형 `4B-2507`이 구형 4-bit `14B-AWQ`를 능가 |
| **4. Scientific NLI FP32 복구 (`Exp 2A`)** | 초기 `mDeBERTa-v3-base` BF16 미세조정 시 왜 `Exp 2B`가 `0/40`으로 붕괴했는가? | `mDeBERTa-v3` Disentangled Attention이 `bf16=True` / `transformers 5.x` FP16 기본 로드에서 **NaN 로짓 붕괴** 유발. **FP32 명시(`bf16=False, fp16=False`)** + `warmup_steps=100`으로 수렴 복구(`Val Loss = 0.1676`), DEV 거짓전제 거절 `7/10 → 10/10`(**100%**). | **FP32 Scientific NLI 학습 + 학습 후 자동 건전성 검증 채택** |
| **5. 사후 Post-NLI 소거 (`Exp 2B`)** | 2단계 집중 Post-NLI(`entail >= 0.25`) 추가 시 충실도가 더 오르는가? | 거짓전제 방어는 `10/10/10` 그대로(추가 이득 0건)인 반면, 한국어 답변 유지가 **`36/40 → 32/40`**, 한국어 핵심사실이 **`25 → 22`**로 하락. | **Post-NLI 기각 (`nli_post_entail_min = None`)** |
| **6. 5-Chunk LoRA SFT 소거 (`Exp 3A`)** | `14B-AWQ` 교사로 역합성·4중 필터 검증한 `547`건 SFT가 `4B-2507`을 개선하는가? | Completion-Only LoRA SFT(`r=32`, 2 epoch, `Val Loss 0.1190`, `95.88%` 정확도)가 `Exp 2A` 대비 **`pair_ans 35→32`**, **`ko_facts 25→22`**, **`quote 0.96→0.93`**로 오히려 하락. 원인: 학생보다 약한 구형 4-bit 교사(`quote 0.93`) + 49.4% 거절 비중 + 영어 전용 `atomic_facts` 필터(영어는 `32→34` 상승). | **SFT 기각 및 DPO 생략** (성능이 하락한 학생 위에 추가 학습하지 않고 순정 `4B-2507` 유지) |
| **7. NLI 임계값 최적화 · Conformal Gate (`Exp 2C`)** | Pre-NLI 임계값을 올리면 `10/10` 거짓전제 방어를 유지하면서 `Exp 2A(0.70)`의 `5`쌍 과거절을 회수할 수 있는가? | FP32 NLI 모순 점수 분포: **정답가능 중앙값 `0.001`** vs **거짓전제 최소값 `0.999`** (`20/20 전원 >= 0.999`). `0.90~0.99` 넓은 안정 구간에서 **`pair_ans 37/40`(+2), `pair_facts 24/40`(+3), `en/ko 35/26`, `fp 10/10/10`, `quote/num 0.99/0.97`** 달성. 논문 그룹 5-Fold OOF Conformal **AURC = `0.0669`**. | **`Base 4B + FP32 Scientific NLI(0.95)` + Conformal Gate 최종 채택** |

---

## 기각된 가설 요약

| 기각 항목 | DEV에서 관측된 실패 원인 | 대체 채택안 |
| :--- | :--- | :--- |
| 자유 형식 JSON 출력 | JSON 파싱 오류 `5+`건으로 `pair_ans 27/40`, `pair_facts 17/40` 하락 | Guided JSON(`response_format: {"type": "json_object"}`) + `max_tokens=1024` (파싱 오류 `0`건) |
| `mDeBERTa-v3` BF16/FP16 혼합 정밀도 학습 | Disentangled Attention NaN 붕괴 (`Exp 2B` `0/40`) | 엄격한 **FP32** 학습(`bf16=False, fp16=False, warmup_steps=100, max_grad_norm=1.0`) + ONNX INT8 양자화 |
| 사후 2단계 집중 Post-NLI (`0.25`) | 거짓전제 추가 이득 없이 한국어 답변 유지 `-4`건(`36→32`) 손실 | 사전 Pre-Gate NLI 단독 적용 (`nli_post_entail_min = None`) |
| 14B-AWQ 교사 역합성 LoRA SFT (`Exp 3A`) | 구형 4-bit 교사(`24/40`, `quote 0.93`)와 높은 거절 비중(49.4%)이 신형 `4B-2507` 한국어 성능을 하향 견인 | 순정 `Qwen3-4B-Instruct-2507` + 파인튜닝 Scientific NLI(`0.95`) |
| Colab 클라우드 IP 직접 arXiv 크롤링 | 속도 제한·차단으로 외부 논문 `0`편 수집 | Hugging Face Parquet 스트리밍(`ccdv/arxiv-summarization`: 학습 150편, `corpus-v2-holdout` 30편 동결) |

---

## 데이터 누수 방지 및 평가 규약
1. **학습-평가 0% 중첩**: Scientific NLI(`7,729`건)와 SFT 소거(`547`건)에 사용한 외부 논문 150편은 `corpus30` 30편의 제목·arXiv ID와 완전히 분리했습니다.
2. **DEV 전용 선택**: 모든 아키텍처·소거 기각(`Exp 2B`, `Exp 3A`)·임계값(`0.95`)·5-Fold 논문 그룹 Conformal Gate는 DEV(`120`출력)에서만 결정했습니다.
3. **Holdout 1회 동결 평가 (`Exp 4`)**: 우승 구성을 고정한 뒤 미공개 Holdout 20편(`240`출력)을 단 1회 평가하고, Exact Two-Sided McNemar 검정 및 `2,000`회 Paper-Cluster Bootstrap 95% 신뢰구간으로 유의성을 검증했습니다.
4. **신규 미공개 평가셋 동결 (`corpus-v2-holdout`)**: 향후 회귀 검증을 위해 `ccdv/arxiv-summarization:validation`에서 30편(`677`청크)을 SHA-256 해시와 함께 별도 동결했습니다(`data/corpus_v2_holdout/freeze_v2.json`).
