# 재현 및 로컬 데모 실행 안내 · Evidence-Grounded Paper Q&A v2

[English](reproduction.md) · [README](../README.ko.md) · [아키텍처](architecture.ko.md) · [방법론 및 설계 의사결정](methodology-and-decisions.ko.md) · [결과](results.ko.md)

## 1. 로컬 인터랙티브 웹 데모 실행

```powershell
# 1. 가상환경 생성 및 의존성 설치
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 로컬 인터랙티브 웹 UI 실행 (기본 주소 http://127.0.0.1:8000)
.\launch-paper-qna.bat demo
# Linux / macOS:
sh ./launch-paper-qna.sh demo
```

별도 설정 없이 바로 동작하는 기능:
- **벤치마크 논문 30편 + 미공개 Holdout 검증 프리셋 10건**: 영어·한국어 답변가능·거짓전제·근거부족 사례를 클릭 한 번으로 실행하며, `v1 vs v2` 비교 배지, Scientific NLI 모순 게이지, Conformal Gate 확률, 원문 인용 하이라이트를 즉시 확인합니다.
- **실시간 PDF 업로드**: 최대 20 MiB 텍스트형 영문 PDF를 업로드해 즉시 청킹·검색·검증할 수 있습니다.
- **선택 연동 — 로컬 `Qwen3-4B-Instruct-2507` LLM 서버 (RTX 3060 6 GB)**:
  ```powershell
  # llama.cpp 또는 Ollama로 Qwen3-4B-Instruct-2507 Q4_K_M(~2.7 GiB VRAM) 실행 후 연결:
  llama-server.exe -m qwen3-4b-instruct-2507-q4_k_m.gguf -ngl 99 -fa -ctk q8_0 -ctv q8_0 -c 6144 --port 8080
  .\launch-paper-qna.bat demo --llm-url http://127.0.0.1:8080
  ```

## 2. CLI 실행 예시

```powershell
# 로컬 모델, Conformal Gate, NLI 패리티 게이트 및 LLM 서버 연결 상태 확인
.\launch-paper-qna.bat status

# 벤치마크 논문에 한국어/영어 질의 실행
.\launch-paper-qna.bat ask --paper-id paper-030 --question "다변량 예측기와 비교한 세 스크리닝 휴리스틱 또는 순위 기준선은 무엇인가요?"

# 로컬 영문 PDF에 직접 질의
.\launch-paper-qna.bat ask --pdf path\to\paper.pdf --question "What dataset and metrics are evaluated?" --llm-url http://127.0.0.1:8080
```

## 3. 자동 테스트 및 로컬 하드웨어 검증

```powershell
# 단위·계약·Conformal Gate·FastAPI API·NLI 패리티 게이트 전체 테스트
.\.venv\Scripts\python.exe -m pytest -v

# 로컬 RTX 3060 FP32 NLI 실측 + ONNX INT8 배포 게이트 감사 + SHA-256 무결성 검증
.\.venv\Scripts\python.exe scripts\verify_local_runtime.py
```

## 4. Colab L4 전체 실험 재현

[`notebooks/paper_qna_v2_workbench.ipynb`](../notebooks/paper_qna_v2_workbench.ipynb)을 Google Colab / Colab Enterprise(NVIDIA L4 24 GB)에서 열면 다음 전체 실험을 원샷으로 재현할 수 있습니다.
1. 검색 스택 A/B (`Exp 1B`: `BGE-M3` vs `Qwen3-0.6B`)
2. Base 4B (`Exp 1C`) · Teacher 14B (`Exp 1C-T`) DEV 120문항 평가 + Semantic Judge
3. 외부 arXiv 150편(누수 0%) FP32 `mDeBERTa-v3` Scientific NLI 학습 및 `Exp 2A / 2B / 2C`
4. 역합성 5-Chunk LoRA SFT 소거 실험 (`Exp 3A`)
5. 논문 그룹 5-Fold OOF Conformal Gate(`calibrated_gate.json`) + 미공개 Holdout 240문항 회귀(`Exp 4`) + `corpus-v2-holdout`(30편) 동결
