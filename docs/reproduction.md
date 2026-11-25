# Reproduction & Local Demo Guide · Evidence-Grounded Paper Q&A v2

[한국어](reproduction.ko.md) · [README](../README.md) · [Architecture](architecture.md) · [Methodology & Decisions](methodology-and-decisions.md) · [Results](results.md)

## 1. Launch the Interactive Local Web Demo

```powershell
# 1. Create virtual environment & install runtime dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Launch the interactive Web UI (http://127.0.0.1:8000)
.\launch-paper-qna.bat demo
# Linux / macOS:
sh ./launch-paper-qna.sh demo
```

What works out of the box:
- **30 Benchmark Papers + 10 Curated Unseen Holdout Presets** (EN/KO answerable, false-premise, and unsupported traps with `v1 vs v2` comparison strip, inline verbatim quote highlights, Scientific NLI contradiction gauge, and Conformal Gate probability).
- **Live PDF Upload**: Upload any text-based English PDF (up to 20 MiB) to chunk, index, and query immediately.
- **Optional Live Local `Qwen3-4B-Instruct-2507` LLM Server (RTX 3060 6 GB)**:
  ```powershell
  # Serve Qwen3-4B-Instruct-2507 Q4_K_M (~2.7 GiB VRAM) with llama.cpp or Ollama, then connect:
  llama-server.exe -m qwen3-4b-instruct-2507-q4_k_m.gguf -ngl 99 -fa -ctk q8_0 -ctv q8_0 -c 6144 --port 8080
  .\launch-paper-qna.bat demo --llm-url http://127.0.0.1:8080
  ```

## 2. CLI Usage

```powershell
# Check loaded papers, Conformal Gate, NLI parity gate, and LLM server connection
.\launch-paper-qna.bat status

# Query a benchmark paper directly in English or Korean
.\launch-paper-qna.bat ask --paper-id paper-030 --question "다변량 예측기와 비교한 세 스크리닝 휴리스틱 또는 순위 기준선은 무엇인가요?"

# Query a newly uploaded local PDF
.\launch-paper-qna.bat ask --pdf path\to\paper.pdf --question "What dataset and metrics are evaluated?" --llm-url http://127.0.0.1:8080
```

## 3. Software Verification & Local Hardware Audit

```powershell
# Run the complete unit, verifier, Conformal Gate, FastAPI API, and NLI parity-gate test suite
.\.venv\Scripts\python.exe -m pytest -v

# Re-run local RTX 3060 FP32 NLI + ONNX INT8 deployment gate + SHA-256 verification
.\.venv\Scripts\python.exe scripts\verify_local_runtime.py
```

## 4. Full Colab L4 Experiment Reproduction

Open [`notebooks/paper_qna_v2_workbench.ipynb`](../notebooks/paper_qna_v2_workbench.ipynb) in Google Colab / Colab Enterprise (NVIDIA L4 24 GB) to reproduce:
1. Retrieval A/B (`Exp 1B`: `BGE-M3` vs `Qwen3-0.6B`)
2. Base 4B (`Exp 1C`) & Teacher 14B (`Exp 1C-T`) DEV 120-output benchmarks + Semantic Judge
3. Zero-overlap external arXiv (`150` papers) FP32 `mDeBERTa-v3` Scientific NLI training + `Exp 2A / 2B / 2C`
4. Reverse 5-chunk LoRA SFT ablation (`Exp 3A`)
5. Paper-grouped 5-fold OOF Conformal Gate (`calibrated_gate.json`) + unseen Holdout `240`-output regression (`Exp 4`) + `corpus-v2-holdout` (`30` papers) freeze
