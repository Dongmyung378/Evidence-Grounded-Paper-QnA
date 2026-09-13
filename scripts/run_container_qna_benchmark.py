"""Run the fixed three-paper Q&A benchmark against Docker Compose."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "container_qna_benchmark.json"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "container_qna_results.json"
PROJECT_NAME = "paper-qna-container-benchmark"
PAPER_ID = re.compile(r"^paper-[0-9a-f]{32}$")
HANGUL = re.compile(r"[가-힣]")


def configure_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_manifest(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported container benchmark schema")
    if config.get("seed") != 378:
        raise ValueError("The container benchmark seed must remain 378")
    papers = config.get("papers") or []
    questions = [question for paper in papers for question in paper["questions"]]
    language_counts = Counter(question["language"] for question in questions)
    if len(papers) != 3 or len(questions) != 10:
        raise ValueError("The benchmark requires exactly three papers and ten questions")
    if language_counts != {"en": 5, "ko": 5}:
        raise ValueError("The benchmark requires five English and five Korean questions")
    if len({question["question_id"] for question in questions}) != len(questions):
        raise ValueError("Container benchmark question IDs must be unique")
    if len({paper["paper_id"] for paper in papers}) != len(papers):
        raise ValueError("Container benchmark paper IDs must be unique")

    canonical_questions = {
        row["question_id"]: row
        for row in load_jsonl(ROOT / "data" / "evaluation" / "questions.jsonl")
    }
    for paper in papers:
        pdf_path = ROOT / paper["pdf"]
        if not pdf_path.is_file() or pdf_path.read_bytes()[:5] != b"%PDF-":
            raise ValueError(f"Benchmark PDF is missing or invalid: {paper['pdf']}")
        for question in paper["questions"]:
            canonical = canonical_questions.get(question["question_id"])
            if canonical is None:
                raise ValueError(f"Unknown question: {question['question_id']}")
            if canonical["paper_id"] != paper["paper_id"]:
                raise ValueError(f"Question paper mismatch: {question['question_id']}")
            if canonical["question_language"] != question["language"]:
                raise ValueError(f"Question language mismatch: {question['question_id']}")
            if canonical["question"] != question["question"]:
                raise ValueError(f"Question text mismatch: {question['question_id']}")


def free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def run_command(
    arguments: list[str],
    *,
    environment: dict[str, str],
    capture: bool = False,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"[실행] {' '.join(arguments)}", flush=True)
    return subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        check=check,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def compose(
    arguments: list[str],
    *,
    environment: dict[str, str],
    capture: bool = False,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["docker", "compose", "-p", PROJECT_NAME, *arguments],
        environment=environment,
        capture=capture,
        check=check,
        timeout=timeout,
    )


def wait_for_health(base_url: str, timeout_seconds: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5, trust_env=False) as client:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return response.json()
        except (httpx.HTTPError, ValueError) as error:
            last_error = error
        time.sleep(0.5)
    raise RuntimeError(f"Backend health check timed out: {last_error}")


def wait_for_job(
    client: httpx.Client,
    location: str,
    timeout_seconds: float = 1800.0,
) -> tuple[dict, list[str]]:
    deadline = time.monotonic() + timeout_seconds
    states: list[str] = []
    while time.monotonic() < deadline:
        response = client.get(location)
        response.raise_for_status()
        job = response.json()
        if not states or states[-1] != job["status"]:
            states.append(job["status"])
        if job["status"] in {"completed", "failed"}:
            return job, states
        time.sleep(0.5)
    raise RuntimeError(f"Analysis job timed out after {timeout_seconds:.0f} seconds")


def read_preparation(
    runtime_paper_id: str,
    *,
    environment: dict[str, str],
) -> dict:
    if not PAPER_ID.fullmatch(runtime_paper_id):
        raise ValueError("Unexpected runtime paper ID")
    script = (
        "import json,pathlib;"
        f"p=pathlib.Path('/var/lib/paper-qna/papers/{runtime_paper_id}/ingestion_report.json');"
        "print(json.dumps(json.loads(p.read_text(encoding='utf-8')).get('question_runtime_preparation')))"
    )
    result = compose(
        ["exec", "-T", "backend", "python", "-c", script],
        environment=environment,
        capture=True,
        timeout=30,
    )
    return json.loads(result.stdout.strip())


def cache_inventory(*, environment: dict[str, str]) -> dict:
    script = (
        "import json,pathlib;"
        "p=pathlib.Path('/var/cache/huggingface');"
        "f=[x for x in p.rglob('*') if x.is_file() and '.locks' not in x.parts];"
        "print(json.dumps({'files':len(f),'bytes':sum(x.stat().st_size for x in f)}))"
    )
    result = compose(
        ["exec", "-T", "backend", "python", "-c", script],
        environment=environment,
        capture=True,
        timeout=60,
    )
    return json.loads(result.stdout.strip())


def remove_benchmark_runtime_volume(*, environment: dict[str, str]) -> bool:
    volume_name = f"{PROJECT_NAME}_paper-qna-runtime"
    inspected = run_command(
        ["docker", "volume", "inspect", volume_name],
        environment=environment,
        capture=True,
        check=False,
        timeout=30,
    )
    if inspected.returncode != 0:
        return True
    metadata = json.loads(inspected.stdout)[0]
    labels = metadata.get("Labels") or {}
    if labels.get("com.docker.compose.project") != PROJECT_NAME:
        raise RuntimeError("Refusing to remove a runtime volume owned by another project")
    if labels.get("com.docker.compose.volume") != "paper-qna-runtime":
        raise RuntimeError("Refusing to remove an unexpected Docker volume")
    removed = run_command(
        ["docker", "volume", "rm", volume_name],
        environment=environment,
        capture=True,
        check=False,
        timeout=60,
    )
    return removed.returncode == 0


def validate_answer(response: dict, expected_language: str) -> None:
    if response.get("question_language") != expected_language:
        raise AssertionError("API response language does not match the question")
    answer = response.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise AssertionError("API returned an empty answer")
    if expected_language == "ko" and not HANGUL.search(answer):
        raise AssertionError("Korean question did not receive a Korean answer")
    if expected_language == "en" and HANGUL.search(answer):
        raise AssertionError("English question received Korean answer text")
    sufficiency = response.get("sufficiency")
    evidence = response.get("evidence")
    if sufficiency == "sufficient":
        if not isinstance(evidence, list) or not evidence:
            raise AssertionError("Sufficient answer has no evidence")
        evidence_ids = set()
        for item in evidence:
            if item["evidence_id"] in evidence_ids:
                raise AssertionError("Answer contains duplicate evidence IDs")
            evidence_ids.add(item["evidence_id"])
            if item["page"] < 1 or not item["text"].strip():
                raise AssertionError("Evidence page or text is invalid")
            locator = item["locator"]
            if not locator["chunk_id"].strip() or not locator["page_label"].strip():
                raise AssertionError("Evidence locator is incomplete")
    elif sufficiency == "insufficient":
        if evidence:
            raise AssertionError("Insufficient answer must not include evidence")
        if not response.get("abstention_reason"):
            raise AssertionError("Insufficient answer must explain the refusal")
    else:
        raise AssertionError("Unexpected answer sufficiency value")
    runtime = response.get("runtime") or {}
    for key in ("total_seconds", "retrieval_seconds", "generation_seconds"):
        if not isinstance(runtime.get(key), (int, float)) or runtime[key] < 0:
            raise AssertionError(f"Invalid runtime field: {key}")
    if runtime.get("fallback_used"):
        raise AssertionError("Container benchmark does not allow validation fallbacks")


def upload_and_analyze(
    client: httpx.Client,
    paper: dict,
    *,
    environment: dict[str, str],
) -> dict:
    source_path = ROOT / paper["pdf"]
    upload_started = time.perf_counter()
    with source_path.open("rb") as source:
        upload_response = client.post(
            "/upload",
            files={"file": (source_path.name, source, "application/pdf")},
        )
    upload_seconds = time.perf_counter() - upload_started
    upload_response.raise_for_status()
    upload = upload_response.json()
    runtime_paper_id = upload["paper_id"]
    if not PAPER_ID.fullmatch(runtime_paper_id):
        raise AssertionError("Upload returned an invalid paper ID")

    analysis_started = time.perf_counter()
    analyze_response = client.post("/analyze", json={"paper_id": runtime_paper_id})
    analyze_response.raise_for_status()
    job, states = wait_for_job(client, analyze_response.headers["location"])
    analysis_seconds = time.perf_counter() - analysis_started
    if job["status"] != "completed":
        raise AssertionError(f"Analysis failed: {job}")
    paper_response = client.get(f"/papers/{runtime_paper_id}")
    paper_response.raise_for_status()
    paper_result = paper_response.json()
    if paper_result["status"] != "ready" or not paper_result["overview"]:
        raise AssertionError("Analyzed paper is not ready")
    preparation = read_preparation(runtime_paper_id, environment=environment)
    if not preparation or preparation.get("status") != "ready":
        raise AssertionError("Question runtime was not prepared during analysis")
    return {
        "benchmark_paper_id": paper["paper_id"],
        "runtime_paper_id": runtime_paper_id,
        "source_pdf": paper["pdf"],
        "source_sha256": digest(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "upload_seconds": round(upload_seconds, 3),
        "analysis_seconds": round(analysis_seconds, 3),
        "observed_job_states": states,
        "page_count": job["page_count"],
        "text_page_count": job["text_page_count"],
        "chunk_count": job["chunk_count"],
        "warning_count": job["warning_count"],
        "question_runtime_preparation": preparation,
    }


def ask_question(
    client: httpx.Client,
    runtime_paper_id: str,
    question: dict,
) -> dict:
    started = time.perf_counter()
    response = client.post(
        "/question",
        json={
            "paper_id": runtime_paper_id,
            "question": question["question"],
        },
        timeout=900,
    )
    wall_seconds = time.perf_counter() - started
    response.raise_for_status()
    answer = response.json()
    validate_answer(answer, question["language"])
    return {
        "question_id": question["question_id"],
        "benchmark_paper_id": None,
        "runtime_paper_id": runtime_paper_id,
        "question": question["question"],
        "language": question["language"],
        "answer": answer["answer"],
        "sufficiency": answer["sufficiency"],
        "abstention_reason": answer["abstention_reason"],
        "evidence": answer["evidence"],
        "wall_seconds": round(wall_seconds, 3),
        "api_runtime": answer["runtime"],
    }


def timing_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(ordered[p95_index], 3),
        "maximum": round(max(values), 3),
    }


def language_summary(results: list[dict], language: str) -> dict:
    rows = [row for row in results if row["language"] == language]
    return {
        "questions": len(rows),
        "sufficient": sum(row["sufficiency"] == "sufficient" for row in rows),
        "insufficient": sum(row["sufficiency"] == "insufficient" for row in rows),
        "fallbacks": sum(row["api_runtime"]["fallback_used"] for row in rows),
        "wall_seconds": timing_summary([row["wall_seconds"] for row in rows]),
        "api_total_seconds": timing_summary(
            [row["api_runtime"]["total_seconds"] for row in rows]
        ),
        "retrieval_seconds": timing_summary(
            [row["api_runtime"]["retrieval_seconds"] for row in rows]
        ),
        "generation_seconds": timing_summary(
            [row["api_runtime"]["generation_seconds"] for row in rows]
        ),
    }


def runtime_versions(environment: dict[str, str]) -> dict:
    docker = run_command(
        ["docker", "version", "--format", "{{json .}}"],
        environment=environment,
        capture=True,
        timeout=30,
    )
    compose_version = run_command(
        ["docker", "compose", "version", "--short"],
        environment=environment,
        capture=True,
        timeout=30,
    )
    version_data = json.loads(docker.stdout.strip())
    return {
        "docker_client": version_data["Client"]["Version"],
        "docker_server": version_data["Server"]["Version"],
        "compose": compose_version.stdout.strip(),
    }


def main() -> None:
    configure_output()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_manifest(config)
    output_path.unlink(missing_ok=True)

    api_port = free_port()
    ui_port = free_port()
    while ui_port == api_port:
        ui_port = free_port()
    online_environment = os.environ.copy()
    online_environment.update(
        {
            "PAPER_QNA_API_PORT": str(api_port),
            "PAPER_QNA_UI_PORT": str(ui_port),
            "PAPER_QNA_SEED": "378",
            "PAPER_QNA_PREPARE_MODELS": "true",
            "PAPER_QNA_MODEL_LOCAL_FILES_ONLY": "false",
            "PAPER_QNA_LOG_LEVEL": "warning",
        }
    )
    offline_environment = dict(online_environment)
    offline_environment["PAPER_QNA_MODEL_LOCAL_FILES_ONLY"] = "true"
    base_url = f"http://127.0.0.1:{api_port}"
    ui_url = f"http://127.0.0.1:{ui_port}"

    build_seconds = None
    stack_removed = False
    papers: list[dict] = []
    results: list[dict] = []
    restart: dict | None = None
    versions = runtime_versions(online_environment)
    try:
        compose(["down", "--remove-orphans"], environment=online_environment, check=False)
        if not args.skip_build:
            build_started = time.perf_counter()
            compose(
                ["build", "backend", "ui"],
                environment=online_environment,
                timeout=3600,
            )
            build_seconds = round(time.perf_counter() - build_started, 3)

        compose(
            ["up", "-d", "--wait", "backend", "ui"],
            environment=online_environment,
            timeout=1800,
        )
        health_before = wait_for_health(base_url)
        if health_before["seed"] != 378:
            raise AssertionError("Container runtime seed is not 378")
        if health_before["papers"] != 0:
            raise RuntimeError(
                "The benchmark runtime volume is not empty. Use a fresh Compose project."
            )
        with httpx.Client(base_url=base_url, timeout=120, trust_env=False) as client:
            for paper in config["papers"]:
                paper_result = upload_and_analyze(
                    client,
                    paper,
                    environment=online_environment,
                )
                papers.append(paper_result)
                for question in paper["questions"]:
                    result = ask_question(
                        client,
                        paper_result["runtime_paper_id"],
                        question,
                    )
                    result["benchmark_paper_id"] = paper["paper_id"]
                    results.append(result)
                    print(
                        f"{result['question_id']} | {result['sufficiency']} | "
                        f"wall={result['wall_seconds']:.3f}s | "
                        f"api={result['api_runtime']['total_seconds']:.3f}s",
                        flush=True,
                    )
        health_after = wait_for_health(base_url)
        with httpx.Client(timeout=30, trust_env=False) as client:
            ui_response = client.get(ui_url)
            ui_response.raise_for_status()
        cache_before = cache_inventory(environment=online_environment)
        if cache_before["files"] <= 0 or cache_before["bytes"] <= 0:
            raise AssertionError("The persistent model cache is empty")

        restart_target_id = config["gates"]["offline_restart_question"]
        restart_target = next(
            row for row in results if row["question_id"] == restart_target_id
        )
        compose(["down", "--remove-orphans"], environment=online_environment)
        restart_started = time.perf_counter()
        compose(
            ["up", "-d", "--wait", "backend", "ui"],
            environment=offline_environment,
            timeout=900,
        )
        restart_stack_seconds = time.perf_counter() - restart_started
        restart_health_before = wait_for_health(base_url)
        if restart_health_before["question_engine"] != "lazy":
            raise AssertionError("Question engine should be lazy after container restart")
        if restart_health_before["papers"] != 3:
            raise AssertionError("Runtime volume did not preserve the three uploaded papers")
        with httpx.Client(base_url=base_url, timeout=120, trust_env=False) as client:
            repeated = ask_question(
                client,
                restart_target["runtime_paper_id"],
                {
                    "question_id": restart_target["question_id"],
                    "language": restart_target["language"],
                    "question": restart_target["question"],
                },
            )
        restart_health_after = wait_for_health(base_url)
        cache_after = cache_inventory(environment=offline_environment)
        restart = {
            "local_files_only": True,
            "stack_start_seconds": round(restart_stack_seconds, 3),
            "health_before_question": restart_health_before,
            "health_after_question": restart_health_after,
            "question_id": repeated["question_id"],
            "question_wall_seconds": repeated["wall_seconds"],
            "question_api_runtime": repeated["api_runtime"],
            "response_matches_initial_run": (
                repeated["answer"] == restart_target["answer"]
                and repeated["sufficiency"] == restart_target["sufficiency"]
                and [item["page"] for item in repeated["evidence"]]
                == [item["page"] for item in restart_target["evidence"]]
            ),
            "runtime_volume_preserved": restart_health_before["papers"] == 3,
            "cache_before_restart": cache_before,
            "cache_after_restart": cache_after,
            "cache_size_unchanged": cache_before == cache_after,
        }
        if not restart["response_matches_initial_run"]:
            raise AssertionError("Offline restart changed the deterministic response")
        if not restart["cache_size_unchanged"]:
            raise AssertionError("Offline restart changed the model cache inventory")

        output = {
            "schema_version": 1,
            "evaluation": config["name"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "seed": 378,
            "gold_loaded_during_execution": False,
            "delivery": "Docker Compose local portfolio demo",
            "compose_project": PROJECT_NAME,
            "versions": versions,
            "ports": {"backend": api_port, "ui": ui_port},
            "build_seconds": build_seconds,
            "health_before": health_before,
            "health_after": health_after,
            "ui_http_status": ui_response.status_code,
            "provenance": {
                "benchmark_config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                "benchmark_config_sha256": digest(config_path),
                "runtime_config_sha256": digest(ROOT / "config" / "runtime_qna.json"),
                "answer_config_sha256": digest(ROOT / "config" / "answer_generation.json"),
                "translation_config_sha256": digest(ROOT / "config" / "translation.json"),
                "implementation_sha256": digest(Path(__file__)),
            },
            "papers": papers,
            "results": results,
            "summary": {
                "papers": len(papers),
                "questions": len(results),
                "languages": Counter(row["language"] for row in results),
                "sufficient": sum(row["sufficiency"] == "sufficient" for row in results),
                "insufficient": sum(row["sufficiency"] == "insufficient" for row in results),
                "fallbacks": sum(row["api_runtime"]["fallback_used"] for row in results),
                "evidence_responses": sum(bool(row["evidence"]) for row in results),
                "analysis_seconds": timing_summary(
                    [paper["analysis_seconds"] for paper in papers]
                ),
                "first_analysis_seconds": papers[0]["analysis_seconds"],
                "later_analysis_seconds": timing_summary(
                    [paper["analysis_seconds"] for paper in papers[1:]]
                ),
                "by_language": {
                    language: language_summary(results, language)
                    for language in ("en", "ko")
                },
            },
            "offline_restart": restart,
            "stack_removed_after_evaluation": False,
            "temporary_runtime_volume_removed": False,
            "model_cache_volume_preserved": True,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
        print(f"Saved: {output_path}")
    finally:
        cleanup = compose(
            ["down", "--remove-orphans"],
            environment=offline_environment,
            check=False,
            timeout=180,
        )
        stack_removed = cleanup.returncode == 0
        runtime_removed = (
            remove_benchmark_runtime_volume(environment=offline_environment)
            if stack_removed
            else False
        )
        if output_path.is_file():
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            saved["stack_removed_after_evaluation"] = stack_removed
            saved["temporary_runtime_volume_removed"] = runtime_removed
            output_path.write_text(
                json.dumps(saved, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
