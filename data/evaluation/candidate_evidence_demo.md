# 23일차부터 24일차 후보와 근거 사례

## 고정 정책

- 후보군: BM25 Top-20 + Dense Top-20, 동일 가중치 RRF, 고유 청크 20개
- 재정렬: 후보 20개 전체에 다국어 Cross-Encoder 적용
- 최종 근거: 중복이 아닌 상위 5개, PDF 페이지마다 최대 2개
- 유사 중복: token 3-shingle Jaccard 유사도 0.85 이상
- 범위: 텍스트 근거만 사용하며 답변 생성과 자동 거절 판정은 제외

## 사람이 확인하는 답변 가능성 점검표

각 사례의 근거 5개를 읽고 요청한 내용을 직접 뒷받침하는지, 중요한 한정 표현이
있는지, 페이지와 절 위치만으로 확인 가능한지 살핀다. 낮은 점수만으로 답변을
거절하지 않으며 자동 충분성 판정은 이후 로드맵 단계에서 다룬다.

## q-001-ko - multilingual_success

- 질문: 이 논문에서 다루는 주요 문제는 무엇인가?
- 논문: `paper-001`
- Gold 페이지(평가 전용): 2
- 후보 수: 20
- 근거 수: 5
- Top-5 근거에 Gold 페이지 포함: 예
- 목적: 한국어 질문에서 영어 근거를 찾는 대표 성공 사례

### 근거 1: p. 1 - ABSTRACT

- chunk_id: `paper-001-p001-c002`
- 재정렬 점수: -4.340182
- Hybrid 순위: 9
- 원문: ABSTRACT Low-rank decomposition (singular value decomposition / principal component analysis based) is widely used as a first-line tool for mitigating radio-frequency interference (RFI) in low-frequency radio astronomy, both in operational pipelines and methodological studies. In thesingle-epochregime— when only one time–frequency dynamic spectrumD(t,ν) is available—there is no structural guarantee that science and interference can be separated by rank selection alone. We model the observation a

### 근거 2: p. 2 - 2.PROBLEM SETUP

- chunk_id: `paper-001-p002-c003`
- 재정렬 점수: -4.451130
- Hybrid 순위: 19
- 원문: 2.PROBLEM SETUP 2.1.Single-epoch observation matrix We consider a single-epoch time–frequency snapshot obtained by fixing an interferometric baseline and polarization. The observation is represented as a matrix D∈R T×F withTtime samples andFfrequency channels, D(t,ν) =S(t,ν) +I(t,ν) +N(t,ν),(1) whereS(t,ν) is the scientific component,I(t,ν) is structured RFI, andN(t,ν) is noise. In the single-epoch regime we assume no additional diversity from repeated epochs or external information (e.g., satel

### 근거 3: p. 6 - 5.RESULTS ON SYNTHETIC EXPERIMENTS

- chunk_id: `paper-001-p006-c003`
- 재정렬 점수: -4.644390
- Hybrid 순위: 15
- 원문: he two components. Figure 3.FWSVD sensitivity to (w core,w prot). Median relative bias in the science band as a function ofw prot for several fixed values ofw core. Stronger down-weighting of the science core (smallerw core) reduces the median bias, but gains saturate oncew core ≲0.03–0.05.

### 근거 4: p. 1 - 1.INTRODUCTION

- chunk_id: `paper-001-p001-c005`
- 재정렬 점수: -4.912340
- Hybrid 순위: 2
- 원문: perform RFI detection/mitigation on time–frequency data (e.g., Akeret et al. 2017; Wilensky et al. 2019; Kerrigan et al. 2019; Yang et al. 2020; Vafaei Sadr et al. 2020; Connor & van Leeuwen 2018), but typically rely on large labeled data sets and diversity across epochs or instruments. This work instead targets thesingle-epoch, fixed-baseline configuration. 1.1.Relation to prior work A substantial literature uses low-rank and lowrank+sparse decompositions for mitigating foregrounds and RFI (Les

### 근거 5: p. 2 - 1.INTRODUCTION

- chunk_id: `paper-001-p002-c002`
- 재정렬 점수: -4.969682
- Hybrid 순위: 17
- 원문: bility via mixed-mode inspection, rank sweeps, and Pareto diagnostics, and we propose a minimal, reproducible quality-assurance (QA) framework for reporting the operational risk of any chosen singleepoch cleaning configuration. Figure 1 provides a synthetic illustration of the single-epoch failure mechanism considered in this paper: when foreground-like and RFIlike structure co-occupy the leading singular mode, lowrank subtraction cannot cleanly separate contamination removal from scientific pre

## q-043-ko - reranker_recovery

- 질문: 다중 소스 distributionally robust graph learning을 평가하기 위해 어떤 실험이 사용되는가?
- 논문: `paper-010`
- Gold 페이지(평가 전용): 19
- 후보 수: 20
- 근거 수: 5
- Top-5 근거에 Gold 페이지 포함: 예
- 목적: Day 22 reranker가 Top-5로 복구한 사례

### 근거 1: p. 19 - VI. E XPERIMENTAL RESULTS

- chunk_id: `paper-010-p019-c001`
- 재정렬 점수: 4.334221
- Hybrid 순위: 15
- 원문: 19 seven representative baselines spanning three methodological families: smooth-signal optimization methods, deep-learningbased structure-learning methods, and distributionally robust graph learning methods. Section VI-A reports controlled experiments on synthetic multi-source networks, designed to isolate the effect of target-domain sample scarcity, source-target heterogeneity, and the number of available source domains under known ground truth. Section VI-B validates the framework on a real m

### 근거 2: p. 18 - VI. E XPERIMENTAL RESULTS

- chunk_id: `paper-010-p018-c007`
- 재정렬 점수: 4.136217
- Hybrid 순위: 2
- 원문: VI. E XPERIMENTAL RESULTS We evaluate the proposed multi-source Wasserstein distributionally robust graph learning framework (MS-WDRO) against

### 근거 3: p. 2 - I. I NTRODUCTION

- chunk_id: `paper-010-p002-c002`
- 재정렬 점수: 3.829852
- Hybrid 순위: 12
- 원문: at the empirical distribution certifies that the learned estimator performs well for any distribution that lies within a controlled optimal transport distance from the training data, a guarantee that purely moment-based ambiguity sets [22], [23] cannot provide in the absence of shape assumptions. The foundations of Wasserstein DRO, duality theory, tractable reformulations, and finite-sample performance guarantees, have been developed in considerable depth [24], and the framework has been applied

### 근거 4: p. 2 - I. I NTRODUCTION

- chunk_id: `paper-010-p002-c006`
- 재정렬 점수: 3.820397
- Hybrid 순위: 10
- 원문: sing source distributions into a structurally sound nominal distribution for robust graph estimation. Despite clear practical motivation, the intersection of multisource learning, Wasserstein distributional robustness, and smooth-signal graph topology inference remains largely unexplored. A further challenge is the joint calibration of multiple interacting hyperparameters inherent in any WDRO-based graph learning framework: the ambiguity set radius determines the degree of distributional robustn

### 근거 5: p. 5 - III. W ASSERSTEIN DISTRIBUTIONALLY ROBUST GRAPH

- chunk_id: `paper-010-p005-c004`
- 재정렬 점수: 3.493623
- Hybrid 순위: 8
- 원문: III. W ASSERSTEIN DISTRIBUTIONALLY ROBUST GRAPH LEARNING This section develops the proposed distributionally robust graph learning framework in four steps. We begin by revisiting the baseline estimator of Section II-B through the lens of empirical risk minimization and robustifying it against a single source’s sampling uncertainty via a Wasserstein ambiguity set (Section III-A). We then extend this single-source construction to the heterogeneous multi-source setting motivating this paper, replac

## q-005-ko - known_limitation

- 질문: 제안된 2단계 하이퍼파라미터 전이 프레임워크는 어떻게 작동하는가?
- 논문: `paper-002`
- Gold 페이지(평가 전용): 1
- 후보 수: 20
- 근거 수: 5
- Top-5 근거에 Gold 페이지 포함: 아니요
- 목적: 후보 생성 또는 용어 불일치를 사람이 판별하는 사례

### 근거 1: p. 10 - 5 Conclusion

- chunk_id: `paper-002-p010-c003`
- 재정렬 점수: 6.455389
- Hybrid 순위: 17
- 원문: 5 Conclusion We propose a compute-efficient two-step framework for hyperparameter transfer in largescale Mixture-of-Experts (MoE) pretraining, enabling accurate optimal learning rate estimation without exhaustive sweeps. By combining µP-based width transferability with a linear scaling law across token budgets, we show that optimal learning rates for trilliontoken large-scale MoE pretraining can be reliably predicted from small proxy experiments, thereby reducing unnecessary computation. We vali

### 근거 2: p. 8 - 3.3.3 Applying to Our Foundation Model

- chunk_id: `paper-002-p008-c004`
- 재정렬 점수: 5.181528
- Hybrid 순위: 14
- 원문: 3.3.3 Applying to Our Foundation Model To validate the effectiveness of our approach, we apply the proposed two-step hyperparameter transfer framework to predict the optimal learning rate for pretraining a large-scale MoE foundation model (155B total, 17B active parameters) over a 10T-token horizon. Detailed model configurations are provided in Table 3. As shown in Figure 2b, the full-scale pretraining of this MoE model requires approximately 98× the total compute of the proxy runs used for opti

### 근거 3: p. 3 - 1 Introduction

- chunk_id: `paper-002-p003-c002`
- 재정렬 점수: 4.162503
- Hybrid 순위: 5
- 원문: MoE Architectures:Building on prior work discussing the adaptation of µP to MoE architectures, we study zero-shot hyperparameter transfer when scaling both model width and the total number of experts (i.e., sparsity expansion). We further explore the use of the Muon optimizer in this setting, extending the empirical scope ofµP-based MoE scaling. • Token-Scale Extrapolation within a Two-Step Framework:We introduce a twostep predictive framework to extrapolate optimal learning rates to unseen, lon

### 근거 4: p. 3 - 2 Methods

- chunk_id: `paper-002-p003-c003`
- 재정렬 점수: 3.954626
- Hybrid 순위: 1
- 원문: 2 Methods Our two-step hyperparameter transfer framework requires two key components: (1) formulating µP specifically for MoE architectures, and (2) establishing a scaling law that predicts the optimal learning rate as the token budget scales. Section 2.1 describes our adaptation of µP to MoE architectures, and the results in Section 3.2 validate that this formulation ensures that the optimal learning rate can be reliably transferred across model widths. Section 2.2 then addresses extrapolation

### 근거 5: p. 4 - 2.2 Extrapolating Optimal Learning Rates to Long Token Horizons

- chunk_id: `paper-002-p004-c004`
- 재정렬 점수: 0.521606
- Hybrid 순위: 2
- 원문: 2.2 Extrapolating Optimal Learning Rates to Long Token Horizons Although µP-based hyperparameter transfer significantly reduces search costs, performing a direct search for the optimal learning rate over trillions of training tokens (e.g., 10T tokens) remains computationally prohibitive even when using small proxy models. To address this challenge, we aim to identify the optimal learning rate within a constrained token budget and extrapolate the results to the full target token horizon. Building
