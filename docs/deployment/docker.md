# Docker Compose Local Run

[English](docker.md) | [한국어](docker_KO.md)

## Architecture

Docker Compose builds two targets from the root `Dockerfile`.

- `backend` runs one Uvicorn worker on port 8000. It owns PDF parsing, retrieval, reranking, answer generation, and persistent runtime state.
- `ui` runs Streamlit on port 8501. It calls `http://backend:8000` through the private Compose network.

Both images run as the unprivileged `paperqna` user. Compose drops Linux capabilities, enables the no-new-privileges restriction, waits for backend health before starting the UI, and monitors both services with HTTP health checks.

The UI image only contains Streamlit and HTTP client dependencies. The backend image contains a CPU-only PyTorch runtime and the retrieval and generation dependencies. This split keeps the UI image small and avoids installing model libraries twice.

## Start and stop

Build and start the local demo:

```bash
docker compose up --build
```

Open the following addresses:

- UI: `http://127.0.0.1:8501`
- API documentation: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/health`

Stop the services while retaining uploaded-paper state and downloaded models:

```bash
docker compose down
```

Delete the two project volumes only when their contents are no longer needed:

```bash
docker compose down --volumes
```

## Environment configuration

Compose supplies runtime settings through environment variables. Defaults work without a local `.env` file. To customize them, copy `.env.example` to `.env` and edit the copy.

| Variable | Default | Purpose |
|---|---:|---|
| `PAPER_QNA_API_PORT` | `8000` | Backend port exposed on the host |
| `PAPER_QNA_UI_PORT` | `8501` | Streamlit port exposed on the host |
| `PAPER_QNA_SEED` | `378` | Reproducible application seed |
| `PAPER_QNA_MAX_PENDING_JOBS` | `16` | Maximum queued analysis jobs |
| `PAPER_QNA_PREPARE_MODELS` | `true` | Prepare search and generation models during analysis |
| `PAPER_QNA_MODEL_LOCAL_FILES_ONLY` | `false` | Prevent model downloads when all files are already cached |
| `PAPER_QNA_LOG_LEVEL` | `info` | Uvicorn log level |
| `HF_TOKEN` | empty | Optional Hugging Face token for model downloads |

The API data path and UI-to-backend address are fixed inside Compose so host-specific values do not leak into application code. Invalid integer or Boolean application settings stop the backend immediately with a clear configuration error.

Do not commit `.env`. It is ignored by Git and excluded from the Docker build context. The example contains no credential.

## Persistent data

Compose creates two named volumes:

- `paper-qna-runtime` stores private uploaded PDFs, SQLite metadata, extracted pages, chunks, and per-paper dense indexes.
- `paper-qna-model-cache` stores Hugging Face model downloads so subsequent runs can start without downloading them again.

The repository corpus, evaluation data, local implementation journal, and host `.env` are excluded from the image build context. User uploads are not copied into an image.

## Verification

The acceptance evaluator selects unused host ports, disables model preparation only for the packaging gate, builds both images, and runs the same Compose stack used by the quick start. It verifies both health checks, backend and UI HTTP responses, internal service discovery, seed 378, and non-root users. It then removes the temporary containers, network, and test volumes.

```bash
python -B scripts/evaluate_container.py
python -B scripts/validate_container.py
```

The saved machine-readable result is `data/evaluation/container_results.json`. This gate proves local container packaging and service connectivity. It does not prove answer quality or public server deployment.

## Operational notes

- Keep one backend worker because the runtime directory uses a single-owner analysis worker and the loaded models are shared in process.
- The default image is CPU-only and works without an NVIDIA runtime. Local Python execution may be faster when the host GPU is available.
- The first model-enabled analysis can take longer while pinned model files are downloaded and loaded.
- If ports 8000 or 8501 are already used, set different host ports in `.env`.
- If offline mode is enabled before the model cache is populated, the first question cannot load the models.
