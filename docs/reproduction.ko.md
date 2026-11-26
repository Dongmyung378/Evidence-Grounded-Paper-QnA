# 재현 및 실행 가이드

> **언어**: [English](reproduction.md) | **한국어 (Korean)**

## 1. 로컬 환경 구성 및 자동 테스트

지원 환경: Windows, macOS, Linux의 Python `3.10` 이상.

```powershell
# 저장소 복제
git clone https://github.com/Dongmyung378/Evidence-Grounded-Paper-QnA.git
cd Evidence-Grounded-Paper-QnA

# 가상환경 생성 및 의존성 설치
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 단위, 응답 계약, Conformal Gate, FastAPI, 경계 게이트, NLI 패리티 테스트 9종 실행
.\.venv\Scripts\python.exe -m pytest -v
```

Linux 및 macOS 환경:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -v
```

## 2. 로컬 인터랙티브 웹 데모 실행

```powershell
# Windows 기본 실행 (http://127.0.0.1:8000 자동 열기)
.\launch-paper-qna.bat demo

# 사용자 지정 포트 실행
.\launch-paper-qna.bat demo --port 18080 --no-browser
```

```bash
# Linux / macOS 실행
sh ./launch-paper-qna.sh demo
```

별도 LLM 서버를 연결하지 않아도 웹 데모에서 다음 기능을 즉시 검증할 수 있습니다.
- `corpus30` 논문 `30`편과 미공개 Holdout 검증 프리셋 `14`건(정답가능, NLI 경계, Conformal 경계, 거짓전제, 근거부족 한·영 쌍)
- 실시간 Scientific NLI 모순 임계값 슬라이더(`0.50` ~ `0.99`) 및 논문 그룹 Conformal Gate `alpha`(`0.02`, `0.05`, `0.10`) 선택
- 신규 영문 논문 PDF(`20 MiB` 이하) 업로드, 페이지·섹션 기반 청킹 및 근거 발췌 응답
- 한국어 / English UI 즉시 전환(`?lang=ko`, `?lang=en` 쿼리 지원)

## 3. 선택: 로컬 RTX 3060 6 GB Qwen3-4B 서버 연동

일반 6 GB 소비자용 GPU(RTX 3060 Laptop GPU 실측)에서 실시간 이중언어 생성 답변을 연결하려면 다음 순서로 실행합니다.

1. `llama-server` 또는 Ollama로 `Qwen3-4B-Instruct-2507`(`Q4_K_M`, 약 `2.7 GiB` VRAM) 서버를 실행합니다.
   ```powershell
   llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M `
     --host 127.0.0.1 --port 8080 -ngl 99 -np 1 -c 8192
   ```
2. 데모를 로컬 서버 주소와 연결해 실행합니다.
   ```powershell
   .\launch-paper-qna.bat demo --llm-url http://127.0.0.1:8080
   ```
3. `BAAI/bge-m3` 및 `bge-reranker-v2-m3` 신경망 검색을 로컬에서 직접 활성화하려면 `--neural-retrieval` 플래그를 추가합니다.

## 4. 로컬 하드웨어 패리티 게이트 및 동결 코퍼스 무결성 검증

아래 스크립트로 `corpus_v2_holdout` SHA-256 무결성(`2/2` 파일), 로컬 GPU/CPU에서의 FP32 PyTorch `mDeBERTa-v3` VRAM·지연시간·패딩 불변성, 그리고 `models/scientific_nli_int8/` 존재 시 ONNX INT8 배포 게이트를 재실측할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_local_runtime.py --include-pytorch
```

Windows 11 + NVIDIA GeForce RTX 3060 Laptop GPU(`6,144 MiB`) 환경의 실측 감사 로그는 [`reports/local_verification.json`](../reports/local_verification.json)에 저장되어 있습니다.

## 5. Colab Enterprise NVIDIA L4 전체 실험 재현

외부 무누설 arXiv 150편 수집, 검색 A/B, FP32 Scientific NLI 파인튜닝, DEV 소거 실험(`Exp 1C` ~ `Exp 3A`), 논문 그룹 5폴드 OOF Conformal Gate 보정, 미공개 Holdout `240`문항 평가, `Qwen3-14B-AWQ` Semantic Judge 채점까지 전체 과정을 재현하려면:

1. Google Colab Enterprise의 **NVIDIA L4 (24 GB)** 런타임에서 [`notebooks/paper_qna_v2_workbench.ipynb`](../notebooks/paper_qna_v2_workbench.ipynb)를 엽니다.
2. 셀을 순서대로 실행하면 모든 체크포인트와 리포트가 `/content/PaperQnA_v2/`에 저장됩니다.
3. 체크포인트로부터 표준 요약 JSON·CSV와 한·영 분리 차트를 로컬에서 재생성할 수 있습니다.
   ```powershell
   .\.venv\Scripts\python.exe scripts/build_reports_and_figures.py
   ```
