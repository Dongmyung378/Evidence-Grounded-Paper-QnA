"""Verify local Scientific NLI (ONNX INT8 Deployment Gate + RTX 3060 FP32), Conformal Gate, and FastAPI Web Demo."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_qna.nli_gate import ConformalGate, ScientificNLIOnnx, ScientificNLIPyTorch
from paper_qna.pipeline import PaperQnAEngine
from paper_qna.web_demo import create_app


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_holdout_onnx_parity(
    nli: ScientificNLIOnnx,
    ckpt_path: Path,
    questions_path: Path,
    n_answerable: int = 20,
    n_false_premise: int = 20,
    threshold: float = 0.95,
) -> dict[str, Any]:
    q_map = {
        json.loads(line)["case_id"]: json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    recs = sorted(
        [json.loads(line) for line in ckpt_path.read_text(encoding="utf-8").splitlines() if line.strip()],
        key=lambda r: r["case_id"],
    )
    ans_sample = [r for r in recs if q_map[r["case_id"]]["answerability"] == "answerable"][:n_answerable]
    fp_sample = [r for r in recs if q_map[r["case_id"]]["answerability"] == "false_premise"][:n_false_premise]
    subset = ans_sample + fp_sample

    latencies_ms: list[float] = []
    window_counts: list[int] = []
    contra_diffs: list[float] = []
    decision_matches = 0
    fp_caught_onnx = 0
    ans_passed_onnx = 0

    for rec in subset:
        q = q_map[rec["case_id"]]
        out = rec["output"]
        fp32_contra = float(out["pipeline"].get("nli_max_contradiction", 0.0))
        fp32_ent = float(out["pipeline"].get("nli_max_entailment", 0.0))
        fp32_trig = bool(fp32_contra >= threshold and fp32_contra > fp32_ent)

        res = nli.score_question_against_evidence(q["question"], out["retrieved_evidence"])
        onnx_contra = float(res["max_contra"])
        onnx_ent = float(res["max_entail"])
        onnx_trig = bool(onnx_contra >= threshold and onnx_contra > onnx_ent)

        latencies_ms.append(float(res["latency_ms"]))
        window_counts.append(int(res["window_count"]))
        contra_diffs.append(abs(onnx_contra - fp32_contra))
        if onnx_trig == fp32_trig:
            decision_matches += 1
        if q["answerability"] == "false_premise" and onnx_trig:
            fp_caught_onnx += 1
        if q["answerability"] == "answerable" and not onnx_trig:
            ans_passed_onnx += 1

    per_win_ms = [lat / max(wc, 1) for lat, wc in zip(latencies_ms, window_counts)]
    return {
        "n_cases": len(subset),
        "n_answerable": len(ans_sample),
        "n_false_premise": len(fp_sample),
        "threshold": threshold,
        "decision_parity_count": decision_matches,
        "decision_parity_rate": round(decision_matches / max(len(subset), 1), 4),
        "mean_abs_contra_diff": round(statistics.mean(contra_diffs), 6),
        "max_abs_contra_diff": round(max(contra_diffs), 6),
        "false_premise_caught_onnx": fp_caught_onnx,
        "answerable_passed_onnx": ans_passed_onnx,
        "latency_per_question_p50_ms": round(statistics.median(latencies_ms), 2),
        "latency_per_question_mean_ms": round(statistics.mean(latencies_ms), 2),
        "latency_per_window_p50_ms": round(statistics.median(per_win_ms), 2),
        "mean_windows_per_question": round(statistics.mean(window_counts), 2),
    }


def benchmark_rtx3060_fp32_nli() -> dict[str, Any]:
    """Measure real local RTX 3060 6GB CUDA FP32 mDeBERTa-v3 VRAM, padding invariance, and latency."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"tested": False, "reason": "cuda_unavailable"}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fp32_local = ROOT / "models" / "scientific_nli_mdeberta"
        model_ref = str(fp32_local) if (fp32_local / "config.json").is_file() else "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        nli = ScientificNLIPyTorch(model_ref, device="cuda")
        pad_check = nli.check_padding_invariance()

        ev = [
            {
                "evidence_id": f"ev-{i:02d}",
                "page": i,
                "text": " ".join(
                    [
                        "The proposed estimator guarantees asymptotic coverage under spatial correlation,",
                        "however finite-sample spatial failures remain possible near boundary discontinuities and the bias floor is NOT eliminated.",
                    ]
                    * 10
                ),
            }
            for i in range(1, 6)
        ]
        q_fp = "Since the paper establishes that the estimator completely eliminates the bias floor and permits no spatial failures, what is its variance?"
        # Warmup + 5 timed runs
        _ = nli.score_question_against_evidence(q_fp, ev)
        torch.cuda.synchronize()
        lats = []
        last_res = None
        for _ in range(5):
            torch.cuda.synchronize()
            last_res = nli.score_question_against_evidence(q_fp, ev)
            torch.cuda.synchronize()
            lats.append(float(last_res["latency_ms"]))

        alloc_gib = round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
        peak_gib = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        total_gib = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        del nli
        torch.cuda.empty_cache()
        return {
            "tested": True,
            "model_ref": model_ref,
            "device": torch.cuda.get_device_name(0),
            "vram_allocated_gib": alloc_gib,
            "vram_peak_gib": peak_gib,
            "vram_total_gib": total_gib,
            "padding_invariance": pad_check,
            "sample_false_premise_contra": last_res["max_contra"] if last_res else None,
            "windows_per_question": last_res["window_count"] if last_res else 0,
            "latency_per_question_p50_ms": round(statistics.median(lats), 2),
            "latency_per_window_p50_ms": round(statistics.median(lats) / max(last_res["window_count"], 1), 2) if last_res else 0.0,
            "gate_status": "PASS" if pad_check["passed"] else "FAIL",
        }
    except Exception as exc:
        return {"tested": False, "reason": str(exc)}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.perf_counter()
    onnx_path = ROOT / "models" / "scientific_nli_int8" / "model_quantized.onnx"
    gate_path = ROOT / "reports" / "calibrated_gate.json"
    freeze_v2_path = ROOT / "data" / "corpus_v2_holdout" / "freeze_v2.json"

    freeze_v2 = json.loads(freeze_v2_path.read_text(encoding="utf-8"))
    v2_checks = {}
    for fname, expected in freeze_v2["sha256"].items():
        actual = sha256_file(ROOT / "data" / "corpus_v2_holdout" / fname)
        assert actual == expected, f"corpus_v2_holdout/{fname} SHA-256 mismatch!"
        v2_checks[fname] = {"sha256": actual, "verified": True}

    gpu_info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_info = {
                "cuda_available": True,
                "device_name": props.name,
                "total_vram_mib": round(props.total_memory / (1024 ** 2), 1),
                "torch_version": torch.__version__,
            }
    except Exception:
        pass

    # 1. Local ONNX INT8 Deployment Gate Audit
    onnx_report: dict[str, Any] = {"available": False}
    if onnx_path.is_file():
        nli = ScientificNLIOnnx(onnx_path.parent, num_threads=4)
        pad_gate = nli.check_padding_invariance()
        ckpt_path = ROOT / "outputs" / "v2_checkpoints" / "exp4_v2_final_regression_holdout.checkpoint.jsonl"
        q_path = ROOT / "data" / "benchmarks" / "corpus30" / "questions.jsonl"
        parity = (
            verify_holdout_onnx_parity(nli, ckpt_path, q_path, n_answerable=20, n_false_premise=20)
            if ckpt_path.is_file()
            else None
        )
        passed = bool(pad_gate["passed"] and (parity is None or parity["decision_parity_rate"] >= 0.95))
        onnx_report = {
            "available": True,
            "file_size_mib": nli.model_size_mib,
            "sha256": sha256_file(onnx_path),
            "threads": 4,
            "padding_invariance": pad_gate,
            "holdout_fp32_vs_int8_parity": parity,
            "gate_status": "PASS" if passed else "BLOCKED",
            "diagnosis": (
                "Legacy TorchScript ONNX export + dynamic INT8 breaks DeBERTa-v2 disentangled relative-position padding invariance; deploy Scientific NLI in FP32 PyTorch."
                if not passed
                else "Verified ONNX INT8 parity."
            ),
        }
        print(f"[ONNX GATE: {onnx_report['gate_status']}] pad_diff={pad_gate['abs_diff']} | parity={parity}")

    # 2. Local RTX 3060 6GB CUDA FP32 NLI Benchmark
    rtx_report = benchmark_rtx3060_fp32_nli()
    print("[RTX 3060 FP32 NLI]", json.dumps(rtx_report, ensure_ascii=False))

    # 3. Conformal Gate & Live FastAPI Web Demo End-to-End Verification
    gate = ConformalGate.from_file(gate_path)
    engine = PaperQnAEngine(root=ROOT)
    app = create_app(engine=engine)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200 and r_health.json()["status"] == "ok"
    r_papers = client.get("/api/papers")
    assert r_papers.status_code == 200 and len(r_papers.json()["papers"]) == 30
    r_presets = client.get("/api/presets")
    presets = r_presets.json()["presets"]
    assert r_presets.status_code == 200 and len(presets) >= 6

    demo_checks = []
    for cat in ("ans", "fp", "unsup"):
        sample = next(p for p in presets if p["category"] == cat and p["language"] == "ko")
        resp = client.post(
            "/api/ask",
            json={"paper_id": sample["paper_id"], "question": sample["question"], "nli_threshold": 0.95},
        )
        assert resp.status_code == 200, f"Preset {sample['id']} failed: {resp.text}"
        body = resp.json()
        expected_suff = "sufficient" if cat == "ans" else "insufficient"
        assert body["response"]["sufficiency"] == expected_suff, f"Unexpected sufficiency for {sample['id']}"
        demo_checks.append(
            {
                "case_id": sample["id"],
                "category": cat,
                "sufficiency": body["response"]["sufficiency"],
                "nli_contra": body["nli"]["max_contra"],
                "nli_engine": body["nli"]["engine"],
                "wall_ms": body["pipeline"]["wall_ms"],
                "conformal_prob": body["conformal"]["probability"] if body.get("conformal") else None,
            }
        )

    summary = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "gpu": gpu_info,
        },
        "corpus_v2_holdout_sha256": v2_checks,
        "conformal_gate": {
            "oof_aurc": gate.oof_aurc,
            "nli_contra_threshold": gate.nli_contra_threshold,
            "thresholds": gate.thresholds,
        },
        "scientific_nli_onnx_int8_gate": onnx_report,
        "rtx3060_fp32_pytorch_nli": rtx_report,
        "web_demo_api_checks": {
            "health_ok": True,
            "papers_loaded": len(r_papers.json()["papers"]),
            "presets_loaded": len(presets),
            "nli_runtime_status": engine.nli_backend_status,
            "preset_live_checks": demo_checks,
        },
        "total_verification_seconds": round(time.perf_counter() - t0, 2),
    }

    out_path = ROOT / "reports" / "local_verification.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVED] {out_path} ({summary['total_verification_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
