"""Verify a running HTTP deployment through PDF upload and real-model answers."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from check_local_demo import synthetic_pdf
from grounded_answer_contract import ABSTENTION_ANSWERS
from quality_benchmark import assert_writable_output


def request(base, route, body=None, content_type="application/json", timeout=240):
    headers = {"Content-Type": content_type} if body is not None else {}
    with urllib.request.urlopen(urllib.request.Request(
        base + route, data=body, headers=headers,
    ), timeout=timeout) as response:
        return json.loads(response.read())


def upload(base, pdf):
    boundary = "paper-qna-reproduction-boundary"
    prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
              "filename=\"integration.pdf\"\r\nContent-Type: application/pdf\r\n\r\n").encode()
    return request(base, "/upload", prefix + pdf + f"\r\n--{boundary}--\r\n".encode(),
                   f"multipart/form-data; boundary={boundary}")


def analyze(base, pdf):
    paper_id = upload(base, pdf)["paper_id"]
    job = request(base, "/analyze", json.dumps({"paper_id": paper_id}).encode())
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        job = request(base, f"/jobs/{job['job_id']}")
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.2)
    if job["status"] != "completed":
        raise RuntimeError(f"PDF analysis did not complete: {job}")
    return paper_id, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ui-url")
    parser.add_argument("--real-cases", type=Path, help="JSON with a PDF path and question/marker cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_writable_output(args.output)
    base = args.base_url.rstrip("/")
    source_path = ROOT / "data/quality_benchmark/papers/fixture-alder.json"
    client_hashes = {"scripts/check_deployment.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "scripts/check_local_demo.py": hashlib.sha256((ROOT / "scripts/check_local_demo.py").read_bytes()).hexdigest(),
                     "scripts/quality_benchmark.py": hashlib.sha256((ROOT / "scripts/quality_benchmark.py").read_bytes()).hexdigest()}
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    real_cases_hash = hashlib.sha256(args.real_cases.read_bytes()).hexdigest() if args.real_cases else None
    pdf = synthetic_pdf(json.loads(source_path.read_text(encoding="utf-8")))
    health = request(base, "/health")
    paper_id, job = analyze(base, pdf)
    cases = [
        ("numeric-en", "What classification accuracy did the proposed Alder method report?", ["85%"]),
        ("list-en", "Which three inputs are used by the Alder model?", ["temperature", "pressure", "vibration"]),
        ("numeric-ko", "제안된 Alder 방법이 보고한 분류 정확도는 얼마인가요?", ["85%"]),
        ("unsupported-en", "Which university funded the Alder study?", []),
        ("unsupported-ko", "Alder 연구에 연구비를 지원한 대학은 어디인가요?", []),
    ]
    results = []
    for case_id, question, terms in cases:
        started = time.perf_counter()
        payload = request(base, "/question", json.dumps(
            {"paper_id": paper_id, "question": question}, ensure_ascii=False).encode("utf-8"))
        if terms:
            passed = (payload["sufficiency"] == "sufficient" and bool(payload["evidence"])
                      and all(term.casefold() in payload["answer"].casefold() for term in terms))
        else:
            passed = (payload["sufficiency"] == "insufficient" and not payload["evidence"]
                      and bool(payload.get("abstention_reason"))
                      and payload["answer"] == ABSTENTION_ANSWERS[payload["question_language"]])
        passed = passed and all(e["locator"]["chunk_id"].startswith(paper_id + "-")
                                for e in payload["evidence"])
        if not passed:
            raise RuntimeError(f"Deployment verification failed: {case_id}: {payload}")
        results.append({"case_id": case_id, "passed": True,
                        "http_wall_seconds": round(time.perf_counter() - started, 6),
                        "response": payload})
        print(f"Passed {case_id}", flush=True)
    ui_healthy = None
    if args.ui_url:
        with urllib.request.urlopen(args.ui_url.rstrip("/") + "/_stcore/health", timeout=20) as response:
            ui_healthy = response.status == 200
        if not ui_healthy:
            raise RuntimeError("UI health endpoint failed")
    real_check = None
    if args.real_cases:
        inputs = json.loads(args.real_cases.read_text(encoding="utf-8"))
        real_pdf_path = Path(inputs["pdf_path"])
        if not real_pdf_path.is_absolute():
            real_pdf_path = ROOT / real_pdf_path
        real_pdf = real_pdf_path.read_bytes()
        if inputs.get("pdf_sha256") and hashlib.sha256(real_pdf).hexdigest() != inputs["pdf_sha256"]:
            raise ValueError("Real deployment fixture PDF differs from its declared byte hash")
        real_id, real_job = analyze(base, real_pdf)
        real_results = []
        for case in inputs["cases"]:
            payload = request(base, "/question", json.dumps(
                {"paper_id": real_id, "question": case["question"]}, ensure_ascii=False).encode("utf-8"))
            if payload["sufficiency"] != "sufficient" or not payload["evidence"]:
                raise RuntimeError(f"Real PDF supported question was refused: {case['question']}")
            if any(not e["locator"]["chunk_id"].startswith(real_id + "-") for e in payload["evidence"]):
                raise RuntimeError("Real PDF citation belongs to another uploaded paper")
            from quality_benchmark import marker_present
            if any(not any(marker_present(payload["answer"], term) for term in group)
                   for group in case.get("required_term_groups", [])):
                raise RuntimeError(f"Real PDF answer omitted a required lexical marker: {case['question']}")
            real_results.append({"question": case["question"], "response": payload, "passed": True})
            print("Passed real PDF question", flush=True)
        real_check = {"pdf_sha256": hashlib.sha256(real_pdf).hexdigest(), "analysis": real_job,
                      "results": real_results, "marker_coverage_is_not_semantic_accuracy": True}
    report = {"schema_version": 1, "scope": "authored_synthetic_pdf_network_integration",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "passed": True, "execution_kind": "network_api", "health": health,
              "analysis": job, "results": results, "ui_healthy": ui_healthy,
              "real_pdf_check": real_check,
              "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
              "source_sha256": source_hash, "client_implementation_sha256": client_hashes,
              "real_cases_sha256": real_cases_hash,
              "independent_human_review": False, "third_party_verification": "not_attested"}
    if (hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash
            or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
                   for name, value in client_hashes.items())
            or (args.real_cases and hashlib.sha256(args.real_cases.read_bytes()).hexdigest() != real_cases_hash)):
        raise RuntimeError("Client sources or test inputs changed during deployment verification")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"Saved deployment check: {args.output}")


if __name__ == "__main__":
    main()
