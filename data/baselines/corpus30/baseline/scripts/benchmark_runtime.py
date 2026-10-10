"""Measure real cold and warm single-paper Q&A latency without changing baselines.

This CLI uses the pinned runtime models. Unit tests may inject an engine through
run_benchmark, but test-double reports must be explicitly labelled as such.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.question_engine import RuntimeQuestionEngine, load_jsonl
from quality_benchmark import IMPLEMENTATIONS


def process_memory():
    """Report native process memory, including model tensors when supported."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"source":"Windows GetProcessMemoryInfo", "rss_bytes":counters.WorkingSetSize,
                    "peak_rss_bytes":counters.PeakWorkingSetSize, "scope":"entire_process"}
        return {"source":"unavailable", "rss_bytes":None, "peak_rss_bytes":None}
    try:
        import resource
        factor = 1 if sys.platform == "darwin" else 1024
        return {"source":"resource.getrusage", "rss_bytes":None,
                "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor,
                "scope":"entire_process"}
    except (ImportError, OSError):
        return {"source":"unavailable", "rss_bytes":None, "peak_rss_bytes":None}


def percentile(values, quantile):
    """Linearly interpolated percentile, including explicit small-sample behavior."""
    values = sorted(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency samples must be finite non-negative numbers")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(values):
    return {
        "count": len(values),
        "p50_seconds": round(percentile(values, 0.5), 6),
        "p95_seconds": round(percentile(values, 0.95), 6),
        "min_seconds": round(min(values), 6),
        "max_seconds": round(max(values), 6),
    }


def stage_values(result):
    stages = result["pipeline"].get("stage_timings", {})
    return {
        f"{group}.{name}": float(value)
        for group, timings in stages.items()
        for name, value in timings.items()
        if name.endswith("_seconds") and isinstance(value, (int, float))
    }


def response_signature(result):
    encoded = json.dumps(result["response"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_benchmark(
    engine,
    question,
    paper_id,
    directory,
    *,
    repetitions=10,
    warmup=1,
    execution_mode="real_models",
    translation_cache_mode="enabled",
):
    """Keep the first call and warm-up calls out of the warmed percentile samples."""
    if repetitions < 2 or warmup < 0:
        raise ValueError("Use at least two repetitions and a non-negative warmup")
    if execution_mode not in {"real_models", "test_double"}:
        raise ValueError("Unsupported benchmark execution mode")
    if execution_mode == "real_models" and not isinstance(engine, RuntimeQuestionEngine):
        raise ValueError("Injected engines must be labelled execution_mode=test_double")
    if translation_cache_mode not in {"enabled", "disabled"}:
        raise ValueError("Unsupported translation cache measurement mode")

    def invoke():
        if translation_cache_mode == "disabled":
            engine.clear_translation_cache()
        started = time.perf_counter()
        result = engine.ask(question, paper_id, directory)
        return result, time.perf_counter() - started

    cold_result, cold_wall = invoke()
    cold_memory = process_memory()
    warmup_requests = []
    for _ in range(warmup):
        result, wall = invoke()
        warmup_requests.append({
            "wall_seconds": round(wall, 6),
            "request_mode": result["pipeline"].get("request_mode"),
        })

    samples = []
    for index in range(repetitions):
        result, wall = invoke()
        pipeline = result["pipeline"]
        mode = pipeline.get("request_mode")
        if mode not in {None, "warm"}:
            raise RuntimeError("A warmed sample unexpectedly activated runtime resources")
        samples.append({
            "iteration": index + 1,
            "wall_seconds": round(wall, 6),
            "request_mode": mode,
            "runtime_seconds": pipeline["runtime_seconds"],
            "pipeline_seconds": pipeline.get("pipeline_runtime_seconds", pipeline["runtime_seconds"]),
            "stage_timings": pipeline.get("stage_timings", {}),
            "stage_values": stage_values(result),
            "sufficiency": result["response"]["sufficiency"],
            "fallback_used": pipeline.get("fallback_used", False),
            "response_sha256": response_signature(result),
            "process_memory": process_memory(),
            "translation_cache": pipeline.get("llm", {}).get("cache"),
            "translation_validation_retry": pipeline.get("llm", {}).get("validation_retry"),
        })
    stage_names = sorted(set().union(*(row["stage_values"] for row in samples)))
    complete_stages = {
        name: summarize([row["stage_values"][name] for row in samples])
        for name in stage_names
        if all(name in row["stage_values"] for row in samples)
    }
    return {
        "schema_version": 1,
        "execution_mode": execution_mode,
        "translation_cache_mode": translation_cache_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": paper_id,
        "question": question,
        "question_language": cold_result["question_language"],
        "cold_start": {
            "wall_seconds": round(cold_wall, 6),
            "runtime_seconds": cold_result["pipeline"]["runtime_seconds"],
            "request_mode": cold_result["pipeline"].get("request_mode"),
            "activation": cold_result["pipeline"].get("activation", {}),
            "stage_timings": cold_result["pipeline"].get("stage_timings", {}),
            "process_memory": cold_memory,
            "translation_cache": cold_result["pipeline"].get("llm", {}).get("cache"),
            "translation_validation_retry": cold_result["pipeline"].get("llm", {}).get("validation_retry"),
        },
        "warmup_requests": warmup_requests,
        "warm": {
            "wall_latency": summarize([row["wall_seconds"] for row in samples]),
            "request_runtime_latency": summarize([row["runtime_seconds"] for row in samples]),
            "pipeline_latency": summarize([row["pipeline_seconds"] for row in samples]),
            "stages": complete_stages,
            "samples": samples,
            "sufficient_count": sum(row["sufficiency"] == "sufficient" for row in samples),
            "fallback_count": sum(row["fallback_used"] for row in samples),
            "responses_identical": len({row["response_sha256"] for row in samples}) == 1,
            "peak_process_rss_bytes": max(
                (row["process_memory"]["peak_rss_bytes"] for row in samples
                 if row["process_memory"].get("peak_rss_bytes") is not None), default=None),
        },
        "models": cold_result["pipeline"].get("llm", {}),
        "retrieval_config_fingerprint": cold_result["pipeline"].get(
            "retrieval_config_fingerprint"
        ),
        "measurement_notes": [
            "The first request includes activation and first inference; it is not a warm sample.",
            "Fresh process means cold in-process models, not an emptied operating-system or model-file cache.",
            "Warm-up requests are recorded but excluded from p50 and p95.",
            "A request timer includes runtime lock waiting and activation. Stage timers are nested and must not be added twice.",
            "p50 and p95 use linear interpolation. Small repeated samples do not establish a production service-level guarantee.",
            "This report measures latency and output stability, not independent answer or translation correctness.",
            "Memory readings cover the entire process. Native peak RSS includes model allocations and preceding initialization.",
            "Enabled translation caching may reuse exact source sentences. Disabled mode clears sentence outputs before every request while keeping models loaded; it still repeats the same question.",
        ],
    }


def environment_metadata():
    versions = {}
    for package in ("numpy", "torch", "transformers", "sentence-transformers", "rank-bm25"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": versions,
    }
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
        result["torch_cpu_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["cuda_available"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True, help="Corpus or one-paper chunks.jsonl")
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--translation-cache-mode", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_chunks = load_jsonl(args.chunks)
    chunks = [row for row in source_chunks if row.get("paper_id") == args.paper_id]
    if not chunks:
        parser.error("--chunks must contain non-empty chunks for --paper-id")
    if args.repetitions < 2 or args.warmup < 0 or not args.question.strip():
        parser.error("Require a non-empty question, repetitions >= 2 and warmup >= 0")
    protected = (ROOT / "data/evaluation").resolve()
    output = args.output.resolve()
    if output == args.chunks.resolve() or protected in output.parents:
        parser.error("Write runtime reports outside the frozen data/evaluation directory and input chunks")
    config_names = (
        "runtime_qna.json", "production_retrieval.json", "abstention.json", "answer_generation.json",
        "answerability_policy.json", "translation.json", "translation_glossary.json", "model_lock.json",
        "answer_output_schema.json", "generation_runtime.json",
    )
    implementation_names = tuple(dict.fromkeys([
        "scripts/benchmark_runtime.py", "app/question_engine.py", *IMPLEMENTATIONS,
    ]))
    config_hashes = {name: hashlib.sha256((ROOT / "config" / name).read_bytes()).hexdigest()
                     for name in config_names}
    implementation_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in implementation_names}

    construction_started = time.perf_counter()
    engine = RuntimeQuestionEngine(local_files_only=args.local_files_only)
    construction_seconds = time.perf_counter() - construction_started
    with tempfile.TemporaryDirectory(prefix="paper-qna-runtime-benchmark-") as name:
        directory = Path(name)
        selected_input = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks)
        (directory / "chunks.jsonl").write_bytes(selected_input.encode("utf-8"))
        report = run_benchmark(
            engine,
            args.question.strip(),
            args.paper_id,
            directory,
            repetitions=args.repetitions,
            warmup=args.warmup,
            translation_cache_mode=args.translation_cache_mode,
        )
    report["engine_construction_seconds"] = round(construction_seconds, 6)
    report["input_chunks"] = {
        "sha256": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
        "selected_chunks_sha256": hashlib.sha256(selected_input.encode("utf-8")).hexdigest(),
        "source_chunk_count": len(source_chunks),
        "chunk_count": len(chunks),
        "embedding_disk_cache": "fresh_temporary_directory",
    }
    report["environment"] = environment_metadata()
    report["seed"] = engine.seed
    report["model_lock"] = json.loads((ROOT / "config/model_lock.json").read_text(encoding="utf-8"))
    if (any(hashlib.sha256((ROOT / "config" / name).read_bytes()).hexdigest() != value
            for name, value in config_hashes.items())
            or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
                   for name, value in implementation_hashes.items())):
        raise RuntimeError("Runtime sources or settings changed during the benchmark; repeat on a stable revision")
    report["config_sha256"] = config_hashes
    report["implementation_sha256"] = implementation_hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "execution_mode": report["execution_mode"],
                      "cold_wall_seconds": report["cold_start"]["wall_seconds"],
                      "warm_wall_latency": report["warm"]["wall_latency"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
