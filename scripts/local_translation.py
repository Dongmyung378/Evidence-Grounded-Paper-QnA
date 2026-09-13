"""Pinned local NLLB English-to-Korean translation adapter."""

import hashlib
import json
from pathlib import Path

from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config" / "translation.json"


def load_translation_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported translation config schema")
    if config.get("provider") != "local_nllb":
        raise ValueError("Unsupported translation provider")
    for key in ("model", "revision", "source_language", "target_language"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"translation.{key} must be a non-empty string")
    if config["source_language"] != "eng_Latn":
        raise ValueError("translation source language must be eng_Latn")
    if config["target_language"] != "kor_Hang":
        raise ValueError("translation target language must be kor_Hang")
    if config.get("trust_remote_code") is not False:
        raise ValueError("translation trust_remote_code must be false")
    if config.get("use_safetensors") is not True:
        raise ValueError("translation model must use safetensors")
    if config.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("translation device must be auto, cpu, or cuda")
    if config.get("dtype") not in {"auto", "float16", "float32"}:
        raise ValueError("translation dtype must be auto, float16, or float32")
    for key in ("max_input_tokens", "max_output_tokens", "num_beams", "cpu_threads"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"translation.{key} must be a positive integer")
    if config.get("do_sample") is not False:
        raise ValueError("translation sampling must remain disabled")
    return config


def translation_config_fingerprint(config):
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LocalEnglishKoreanTranslator:
    """Load NLLB once and translate selected source sentences as one batch."""

    provider_name = "local_nllb"

    def __init__(self, config_path=CONFIG_PATH, local_files_only=False):
        self.config_path = Path(config_path)
        self.config = load_translation_config(self.config_path)
        self.config_fingerprint = translation_config_fingerprint(self.config)
        self.model_name = self.config["model"]
        self.local_files_only = bool(local_files_only)
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Local translation requires torch, transformers, and sentencepiece."
            ) from error
        requested_device = self.config["device"]
        self.device = (
            "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        )
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("translation.device=cuda but CUDA is unavailable")
        requested_dtype = self.config["dtype"]
        if requested_dtype == "auto":
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, requested_dtype)
        common = {
            "revision": self.config["revision"],
            "trust_remote_code": self.config["trust_remote_code"],
            "local_files_only": self.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.config["source_language"],
            **common,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            use_safetensors=self.config["use_safetensors"],
            **common,
        )
        self.device_fallback_reason = None
        if self.device == "cuda" and requested_device == "auto":
            free_bytes, _ = torch.cuda.mem_get_info()
            required = self.model.get_memory_footprint() * 1.35 + 256 * 1024**2
            if free_bytes < required:
                self.device = "cpu"
                self.dtype = torch.float32
                self.model = self.model.to(dtype=self.dtype)
                self.device_fallback_reason = "insufficient_free_cuda_memory"
        if self.device == "cpu":
            torch.set_num_threads(
                min(self.config["cpu_threads"], torch.get_num_threads())
            )
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def translate(self, sentences):
        if not isinstance(sentences, list) or not sentences:
            raise ValueError("sentences must be a non-empty list")
        if any(
            not isinstance(sentence, str) or not sentence.strip()
            for sentence in sentences
        ):
            raise ValueError("every translation input must be non-empty text")
        encoded = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["max_input_tokens"],
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        target_id = self.tokenizer.convert_tokens_to_ids(
            self.config["target_language"]
        )
        with self._torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_length=self.config["max_output_tokens"],
                num_beams=self.config["num_beams"],
                do_sample=self.config["do_sample"],
            )
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        ]

    def metadata(self):
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "revision": self.config["revision"],
            "license": self.config["license"],
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device_fallback_reason": self.device_fallback_reason,
            "config_path": str(
                self.config_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "config_fingerprint": self.config_fingerprint,
        }
