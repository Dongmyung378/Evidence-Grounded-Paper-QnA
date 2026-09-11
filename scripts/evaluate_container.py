"""Build and exercise the Docker Compose local demo."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "evaluation" / "container_results.json"
SEED = 378
IMPLEMENTATION_FILES = [
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "app/config.py",
    "app/main.py",
    "compose.yaml",
    "requirements-backend.txt",
    "requirements-ui.txt",
    "scripts/evaluate_container.py",
    "scripts/validate_container.py",
    "scripts/verify_project.py",
    "tests/test_container_config.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in IMPLEMENTATION_FILES}


def free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def run(
    arguments: list[str],
    *,
    environment: dict[str, str],
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", *arguments]
    print(f"[compose] {' '.join(arguments)}", flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )


def docker_version(arguments: list[str]) -> str:
    result = subprocess.run(
        ["docker", *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return result.stdout.strip()


def read_url(url: str, timeout_seconds: float = 30.0) -> tuple[int, str]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def parse_ps(output: str) -> list[dict]:
    output = output.strip()
    if not output:
        return []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in output.splitlines() if line.strip()]
    return parsed if isinstance(parsed, list) else [parsed]


def main() -> None:
    api_port = free_port()
    ui_port = free_port()
    while ui_port == api_port:
        ui_port = free_port()

    project = f"paper-qna-container-gate-{SEED}"
    environment = os.environ.copy()
    environment.update(
        {
            "PAPER_QNA_API_PORT": str(api_port),
            "PAPER_QNA_UI_PORT": str(ui_port),
            "PAPER_QNA_PREPARE_MODELS": "false",
            "PAPER_QNA_MODEL_LOCAL_FILES_ONLY": "true",
            "PAPER_QNA_SEED": str(SEED),
            "PAPER_QNA_MAX_PENDING_JOBS": "4",
            "PAPER_QNA_LOG_LEVEL": "warning",
            "HF_TOKEN": "",
        }
    )
    prefix = ["--project-name", project]
    started = time.perf_counter()
    payload: dict | None = None
    cleanup_error: str | None = None

    try:
        run([*prefix, "config", "--quiet"], environment=environment)
        run(
            [
                *prefix,
                "up",
                "--detach",
                "--build",
                "--wait",
                "--wait-timeout",
                "600",
            ],
            environment=environment,
        )

        api_status, api_body = read_url(f"http://127.0.0.1:{api_port}/health")
        ui_health_status, ui_health_body = read_url(
            f"http://127.0.0.1:{ui_port}/_stcore/health"
        )
        ui_status, ui_body = read_url(f"http://127.0.0.1:{ui_port}/")
        health = json.loads(api_body)

        ps_result = run(
            [*prefix, "ps", "--format", "json"],
            environment=environment,
            capture=True,
        )
        services = parse_ps(ps_result.stdout)
        service_states = {
            str(item.get("Service")): {
                "state": item.get("State"),
                "health": item.get("Health"),
            }
            for item in services
        }

        backend_uid = run(
            [*prefix, "exec", "-T", "backend", "id", "-u"],
            environment=environment,
            capture=True,
        ).stdout.strip()
        ui_uid = run(
            [*prefix, "exec", "-T", "ui", "id", "-u"],
            environment=environment,
            capture=True,
        ).stdout.strip()
        internal_api_url = run(
            [
                *prefix,
                "exec",
                "-T",
                "ui",
                "python",
                "-c",
                "import os; print(os.environ['PAPER_QNA_API_URL'])",
            ],
            environment=environment,
            capture=True,
        ).stdout.strip()

        assert api_status == 200
        assert health["status"] == "ok"
        assert health["seed"] == SEED
        assert ui_health_status == 200 and ui_health_body.strip() == "ok"
        assert ui_status == 200 and "streamlit" in ui_body.lower()
        assert service_states["backend"]["state"] == "running"
        assert service_states["backend"]["health"] == "healthy"
        assert service_states["ui"]["state"] == "running"
        assert service_states["ui"]["health"] == "healthy"
        assert backend_uid != "0" and ui_uid != "0"
        assert internal_api_url == "http://backend:8000"

        payload = {
            "schema_version": 1,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "requirement": "docker compose up runs the backend and UI local demo",
            "commands": [
                "docker compose config --quiet",
                "docker compose up --detach --build --wait",
            ],
            "docker": {
                "engine": docker_version(["--version"]),
                "compose": docker_version(["compose", "version"]),
            },
            "services": service_states,
            "http": {
                "backend_health_status": api_status,
                "backend_service": health["service"],
                "backend_question_engine": health["question_engine"],
                "ui_health_status": ui_health_status,
                "ui_page_status": ui_status,
            },
            "configuration": {
                "ui_backend_url": internal_api_url,
                "runtime_volume": "/var/lib/paper-qna",
                "model_cache_volume": "/var/cache/huggingface",
                "model_preparation_for_gate": False,
                "secrets_recorded": False,
            },
            "security": {
                "backend_non_root": backend_uid != "0",
                "ui_non_root": ui_uid != "0",
                "capabilities_dropped": True,
                "no_new_privileges": True,
            },
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "implementation_sha256": implementation_hashes(),
            "quality_claim": "local Docker packaging only; public deployment is not verified",
        }
    finally:
        try:
            run(
                [*prefix, "down", "--volumes", "--remove-orphans"],
                environment=environment,
                check=False,
            )
        except Exception as exc:
            cleanup_error = str(exc)

    remaining = run(
        [*prefix, "ps", "--all", "--quiet"],
        environment=environment,
        capture=True,
        check=False,
    ).stdout.strip()
    if payload is None:
        raise RuntimeError("The Compose acceptance run did not produce a result")
    payload["temporary_stack_removed"] = not remaining and cleanup_error is None
    assert payload["temporary_stack_removed"] is True
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Container acceptance result: {OUTPUT}")


if __name__ == "__main__":
    main()
