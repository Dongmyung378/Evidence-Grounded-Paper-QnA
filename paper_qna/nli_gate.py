"""Sliding-Window Scientific NLI (FP32 PyTorch + ONNX INT8 with Parity Gate) and Paper-Grouped OOF Conformal Gate."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def sliding_windows(text: str, win_words: int = 200, stride: int = 120) -> list[str]:
    words = str(text or "").split()
    if len(words) <= win_words:
        return [str(text or "")]
    out: list[str] = []
    for i in range(0, len(words), stride):
        piece = " ".join(words[i : i + win_words])
        if piece:
            out.append(piece)
        if i + win_words >= len(words):
            break
    return out


def extract_premise_claims(question: str) -> list[str]:
    q = str(question or "").strip()
    hyps = [q]
    m_en = re.search(
        r"(?:since|given that|assuming that)\s+(?:the paper\s+(?:establishes|proves|shows)\s+that\s+)?([^,?]+)",
        q,
        flags=re.I,
    )
    if m_en and len(m_en.group(1).strip()) >= 12:
        hyps.append(m_en.group(1).strip())
    m_ko = re.search(
        r"(?:논문이|본 연구가)?\s*(.+?)(?:다고|라고)\s*(?:입증했으므로|보였으므로|가정할 때|전제할 때)",
        q,
    )
    if m_ko and len(m_ko.group(1).strip()) >= 8:
        hyps.append(m_ko.group(1).strip())
    return hyps


class _BaseSlidingWindowNLI:
    backend_name: str = "base"

    def predict_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[dict[str, float]]:
        raise NotImplementedError

    def score_question_against_evidence(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        hyps = extract_premise_claims(question)
        meta_pairs: list[tuple[str, str, str, int]] = []
        for ev in evidence:
            for w in sliding_windows(ev.get("text", "")):
                for h in hyps:
                    meta_pairs.append((w, h, ev.get("evidence_id", ""), int(ev.get("page", 0))))
        if not meta_pairs:
            return {
                "max_entail": 0.0,
                "max_contra": 0.0,
                "window_count": 0,
                "latency_ms": 0.0,
                "top_contra_evidence_id": None,
                "top_contra_page": None,
                "engine": self.backend_name,
            }
        scores = self.predict_pairs([(m[0], m[1]) for m in meta_pairs])
        best_c_idx = max(range(len(scores)), key=lambda idx: scores[idx]["contra"])
        return {
            "max_entail": round(max(s["entail"] for s in scores), 4),
            "max_contra": round(scores[best_c_idx]["contra"], 4),
            "window_count": len(meta_pairs),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "top_contra_evidence_id": meta_pairs[best_c_idx][2],
            "top_contra_page": meta_pairs[best_c_idx][3],
            "engine": self.backend_name,
        }


class ScientificNLIPyTorch(_BaseSlidingWindowNLI):
    """Strict FP32 PyTorch runner for mDeBERTa-v3 Scientific NLI (padding-invariant on GPU/CPU)."""

    def __init__(self, model_ref: str | Path = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device: str = "auto"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_ref = str(model_ref)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.model_ref)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_ref, dtype=torch.float32
        ).to(self.device).eval()
        id2lbl = {int(k): str(v).lower() for k, v in (getattr(self.model.config, "id2label", None) or {}).items()}
        self.idx_entail = next((i for i, v in id2lbl.items() if "entail" in v), 0)
        self.idx_neutral = next((i for i, v in id2lbl.items() if "neutral" in v), 1)
        self.idx_contra = next((i for i, v in id2lbl.items() if "contra" in v), 2)
        self.backend_name = f"fp32_pytorch_{self.device}"

    def predict_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[dict[str, float]]:
        import torch
        import torch.nn.functional as F

        if not pairs:
            return []
        results: list[dict[str, float]] = []
        i = 0
        cur_bs = batch_size
        while i < len(pairs):
            sub = pairs[i : i + cur_bs]
            try:
                enc = self.tok(
                    [p[0] for p in sub],
                    [p[1] for p in sub],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    probs = F.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
                for row in probs:
                    if not all(math.isfinite(x) for x in row):
                        results.append({"entail": 0.5, "neutral": 0.5, "contra": 0.0})
                    else:
                        results.append(
                            {
                                "entail": float(row[self.idx_entail]),
                                "neutral": float(row[self.idx_neutral]),
                                "contra": float(row[self.idx_contra]),
                            }
                        )
                i += len(sub)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                else:
                    self.model.to("cpu")
                    self.device = "cpu"
                    self.backend_name = "fp32_pytorch_cpu"
        return results

    def check_padding_invariance(self) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        prem = "Our estimator does NOT eliminate the bias floor in finite samples; the bias remains strictly positive across all tested regimes."
        hyp = "Since the paper establishes that the estimator eliminates the bias floor, what is its variance?"
        probs_by_pad = {}
        for pad_to in (None, 384):
            kw: dict[str, Any] = {"truncation": True, "max_length": 512, "return_tensors": "pt"}
            if pad_to:
                kw["padding"] = "max_length"
                kw["max_length"] = pad_to
            enc = self.tok(prem, hyp, **kw).to(self.device)
            with torch.inference_mode():
                p = F.softmax(self.model(**enc).logits, dim=-1)[0].cpu().tolist()
            probs_by_pad[str(pad_to)] = float(p[self.idx_contra])
        diff = abs(probs_by_pad["None"] - probs_by_pad["384"])
        return {
            "unpadded_contra": round(probs_by_pad["None"], 6),
            "padded_384_contra": round(probs_by_pad["384"], 6),
            "abs_diff": round(diff, 6),
            "passed": bool(diff <= 0.02),
        }


class ScientificNLIOnnx(_BaseSlidingWindowNLI):
    """CPU ONNX INT8 runner for mDeBERTa-v3 Scientific NLI with built-in padding-invariance gate."""

    backend_name = "onnx_int8_cpu"

    def __init__(self, model_dir: str | Path, num_threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        onnx_path = self.model_dir / "model_quantized.onnx"
        tok_json = self.model_dir / "tokenizer.json"
        cfg_path = self.model_dir / "config.json"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX INT8 model not found: {onnx_path}")
        if not tok_json.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tok_json}")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        id2lbl = {int(k): str(v).lower() for k, v in (cfg.get("id2label") or {}).items()}
        self.idx_entail = next((i for i, v in id2lbl.items() if "entail" in v), 0)
        self.idx_neutral = next((i for i, v in id2lbl.items() if "neutral" in v), 1)
        self.idx_contra = next((i for i, v in id2lbl.items() if "contra" in v), 2)
        self.pad_id = int(cfg.get("pad_token_id", 0))

        self.tokenizer = Tokenizer.from_file(str(tok_json))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=self.pad_id, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(num_threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.model_size_mib = round(onnx_path.stat().st_size / (1024 ** 2), 2)

    def _run_raw(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }
        logits = self.session.run(None, feed)[0]
        return _softmax(np.asarray(logits, dtype=np.float64))

    def predict_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[dict[str, float]]:
        if not pairs:
            return []
        results: list[dict[str, float]] = []
        for i in range(0, len(pairs), batch_size):
            sub = pairs[i : i + batch_size]
            encs = self.tokenizer.encode_batch([(p[0], p[1]) for p in sub])
            input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
            attention_mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
            probs = self._run_raw(input_ids, attention_mask)
            for row in probs:
                if not np.all(np.isfinite(row)):
                    results.append({"entail": 0.5, "neutral": 0.5, "contra": 0.0})
                else:
                    results.append(
                        {
                            "entail": float(row[self.idx_entail]),
                            "neutral": float(row[self.idx_neutral]),
                            "contra": float(row[self.idx_contra]),
                        }
                    )
        return results

    def check_padding_invariance(self) -> dict[str, Any]:
        """Detect DeBERTa-v2 TorchScript ONNX relative-position padding drift."""
        prem = "Our estimator does NOT eliminate the bias floor in finite samples; the bias remains strictly positive across all tested regimes."
        hyp = "Since the paper establishes that the estimator eliminates the bias floor, what is its variance?"
        enc = self.tokenizer.encode(prem, hyp)
        ids = np.asarray([enc.ids], dtype=np.int64)
        mask = np.asarray([enc.attention_mask], dtype=np.int64)
        p_unpad = float(self._run_raw(ids, mask)[0][self.idx_contra])

        pad_len = max(0, 384 - ids.shape[1])
        ids_p = np.pad(ids, ((0, 0), (0, pad_len)), constant_values=self.pad_id)
        mask_p = np.pad(mask, ((0, 0), (0, pad_len)), constant_values=0)
        p_pad = float(self._run_raw(ids_p, mask_p)[0][self.idx_contra])
        diff = abs(p_unpad - p_pad)
        return {
            "unpadded_contra": round(p_unpad, 6),
            "padded_384_contra": round(p_pad, 6),
            "abs_diff": round(diff, 6),
            "passed": bool(diff <= 0.02),
        }


class ConformalGate:
    """Zero-pickle evaluator for reports/calibrated_gate.json (Paper-Grouped 5-Fold OOF)."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.feature_names: list[str] = list(payload["feature_names"])
        self.mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
        self.coef = np.asarray(payload["coef"], dtype=np.float64)
        self.intercept = float(payload["intercept"])
        self.thresholds: dict[str, float] = {
            str(k): float(v) for k, v in payload.get("conformal_thresholds_by_alpha", {}).items()
        }
        self.nli_contra_threshold = float(payload.get("nli_contra_threshold", 0.95))
        self.oof_aurc = float(payload.get("oof_aurc", 0.0))

    @classmethod
    def from_file(cls, path: str | Path) -> "ConformalGate":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, features: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
        x = np.asarray([float(features.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)
        safe_scale = np.where(self.scale == 0.0, 1.0, self.scale)
        z = (x - self.mean) / safe_scale
        logit = float(np.dot(z, self.coef) + self.intercept)
        prob = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        alpha_key = f"{alpha:.2f}"
        threshold = self.thresholds.get(alpha_key, float(self.payload.get("conformal_threshold", 0.8727)))
        return {
            "probability": round(prob, 4),
            "alpha": round(alpha, 2),
            "threshold": round(threshold, 4),
            "accepted": bool(prob >= threshold),
            "oof_aurc": self.oof_aurc,
        }
