# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/paperqna

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system paperqna \
    && useradd --system --gid paperqna --create-home paperqna

WORKDIR /workspace


FROM runtime-base AS backend

ARG TORCH_VERSION=2.5.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements-backend.txt ./requirements-backend.txt
RUN python -m pip install --index-url "${PYTORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && python -m pip install -r requirements-backend.txt

COPY app ./app
COPY config ./config
COPY scripts ./scripts

RUN mkdir -p /var/lib/paper-qna /var/cache/huggingface \
    && chown -R paperqna:paperqna /workspace /var/lib/paper-qna /var/cache/huggingface

USER paperqna

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


FROM runtime-base AS ui

COPY requirements-ui.txt ./requirements-ui.txt
RUN python -m pip install -r requirements-ui.txt

COPY .streamlit ./.streamlit
COPY ui ./ui

RUN chown -R paperqna:paperqna /workspace

USER paperqna

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=12 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()"]

CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
