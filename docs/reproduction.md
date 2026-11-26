# Reproduction & Deployment Guide

> **Language**: **English** | [한국어 (Korean)](reproduction.ko.md)

## 1. Local Environment Setup & Automated Test Suite

Requirements: Python `3.10+` on Windows, macOS, or Linux.

```powershell
# Clone repository
git clone https://github.com/Dongmyung378/Evidence-Grounded-Paper-QnA.git
cd Evidence-Grounded-Paper-QnA

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run all 9 unit, contract, Conformal Gate, FastAPI, borderline gate, and parity tests
.\.venv\Scripts\python.exe -m pytest -v
```

On Linux and macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -v
```

## 2. Launching the Interactive Local Web Demo

```powershell
# Windows (opens http://127.0.0.1:8000)
.\launch-paper-qna.bat demo

# Custom port (useful if 8000 is occupied)
.\launch-paper-qna.bat demo --port 18080 --no-browser
```

```bash
# Linux / macOS
sh ./launch-paper-qna.sh demo
```

Without a live LLM server, the demo immediately supports:
- All `30` benchmark papers from `corpus30` and `14` verified unseen Holdout presets (English & Korean across Answerable, NLI Borderline, Conformal Borderline, False Premise, and Unsupported categories)
- Live interactive Scientific NLI threshold slider (`0.50` to `0.99`) and Paper Grouped Conformal Gate `alpha` selector (`0.02`, `0.05`, `0.10`)
- Live English PDF upload (`<= 20 MiB`) with section/page aware chunking and grounded extractive fallback
- Instant Korean / English UI toggle (`?lang=en` or `?lang=ko`)

## 3. Optional: Attaching a Local RTX 3060 6 GB Qwen3-4B Server

To enable live generative bilingual synthesis on a consumer 6 GB GPU (tested on RTX 3060 Laptop GPU):

1. Start `Qwen3-4B-Instruct-2507` (`Q4_K_M`, ~2.7 GiB VRAM) using `llama-server` or Ollama:
   ```powershell
   llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M `
     --host 127.0.0.1 --port 8080 -ngl 99 -np 1 -c 8192
   ```
2. Launch the demo connected to the local server:
   ```powershell
   .\launch-paper-qna.bat demo --llm-url http://127.0.0.1:8080
   ```
3. Optional neural retrieval (`BAAI/bge-m3` + `bge-reranker-v2-m3`) can be enabled by adding `--neural-retrieval`.

## 4. Verifying Local Hardware Parity & Frozen Corpus Integrity

Run the local verification audit script to check `corpus_v2_holdout` SHA-256 hashes (`2/2` files), measure FP32 PyTorch `mDeBERTa-v3` latency/VRAM/padding invariance on your local GPU or CPU, and run the ONNX INT8 deployment gate if an exported binary is present under `models/scientific_nli_int8/`:

```powershell
.\.venv\Scripts\python.exe scripts/verify_local_runtime.py --include-pytorch
```

Verified reference output on Windows 11 + NVIDIA GeForce RTX 3060 Laptop GPU (`6,144 MiB`) is recorded in [`reports/local_verification.json`](../reports/local_verification.json).

## 5. Full Experimental Reproduction (Colab Enterprise NVIDIA L4)

To reproduce the complete experimental lifecycle from scratch (external `150` arXiv paper crawling with `0%` overlap against `corpus30`, retrieval A/B, FP32 Scientific NLI training, DEV ablations `Exp 1C` through `Exp 3A`, Paper Grouped 5 Fold OOF Conformal Gate calibration, locked Holdout `240` output regression, and `Qwen3-14B-AWQ` Semantic Judge evaluation):

1. Open [`notebooks/paper_qna_v2_workbench.ipynb`](../notebooks/paper_qna_v2_workbench.ipynb) in Google Colab Enterprise with an **NVIDIA L4 (24 GB)** runtime.
2. Execute the cells sequentially; all intermediate checkpoints, experiment logs, and JSON/CSV reports persist cleanly to `/content/PaperQnA_v2/`.
3. Regenerate canonical summary JSON/CSV tables and localized EN/KO figures locally from checkpoints:
   ```powershell
   .\.venv\Scripts\python.exe scripts/build_reports_and_figures.py
   ```
