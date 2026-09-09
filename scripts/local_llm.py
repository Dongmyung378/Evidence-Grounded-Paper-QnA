"""Local Hugging Face text-generation adapter for Day 30."""

import hashlib
import json
from pathlib import Path

from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config" / "generation.json"


def load_generation_config(path=CONFIG_PATH):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    return validate_generation_config(config)


def validate_generation_config(config):
    assert config["schema_version"] == 1
    assert config["roadmap_day"] == 30
    assert config["provider"] == "local_transformers"
    assert isinstance(config["model"], str) and config["model"]
    assert isinstance(config["revision"], str) and config["revision"]
    assert config["trust_remote_code"] is False
    assert config["device"] in {"auto", "cpu", "cuda"}
    assert config["dtype"] in {"auto", "float16", "float32"}
    assert config["max_input_tokens"] > 0
    assert config["max_new_tokens"] > 0
    assert config["do_sample"] is False
    assert config["validation_attempts"] >= 1
    if "cpu_validation_attempts" in config:
        assert 1 <= config["cpu_validation_attempts"] <= config["validation_attempts"]
    assert config["safe_fallback_on_validation_error"] is True
    if "max_time_seconds" in config:
        assert config["max_time_seconds"] > 0
    if "cpu_threads" in config:
        assert isinstance(config["cpu_threads"], int) and config["cpu_threads"] > 0
    return config


def generation_config_fingerprint(config):
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LocalTransformersLLM:
    """Load one compact instruct model and reuse it across questions."""

    provider_name = "local_transformers"

    def __init__(self, config_path=CONFIG_PATH, local_files_only=False):
        self.config_path = Path(config_path)
        self.config = load_generation_config(self.config_path)
        self.config_fingerprint = generation_config_fingerprint(self.config)
        self.model_name = self.config["model"]
        self.local_files_only = local_files_only
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local generation requires torch and transformers. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        requested_device = self.config["device"]
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("generation.device=cuda but CUDA is unavailable")

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            **common,
        )
        self.device_fallback_reason = None
        if self.device == "cuda" and requested_device == "auto":
            free_bytes, _ = torch.cuda.mem_get_info()
            required = self.model.get_memory_footprint() * 1.5 + 512 * 1024**2
            if free_bytes < required:
                self.device = "cpu"
                self.dtype = torch.float32
                self.model = self.model.to(dtype=self.dtype)
                self.device_fallback_reason = "insufficient_free_cuda_memory"
        if self.device == "cpu":
            torch.set_num_threads(
                min(self.config.get("cpu_threads", 4), torch.get_num_threads())
            )
        self._torch = torch
        try:
            self.model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            if self.device != "cuda" or requested_device != "auto":
                raise
            self._move_to_cpu("cuda_out_of_memory_during_loading")
        self.model.eval()
        self._torch = torch

    def _move_to_cpu(self, reason):
        self.device = "cpu"
        self.dtype = self._torch.float32
        self.model.to(device="cpu", dtype=self.dtype)
        self.device_fallback_reason = reason
        self._torch.set_num_threads(
            min(self.config.get("cpu_threads", 4), self._torch.get_num_threads())
        )
        self._torch.cuda.empty_cache()

    def generate(self, messages, response_schema=None):
        """Generate one raw assistant response; validation stays in the Q&A layer."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens > self.config["max_input_tokens"]:
            raise ValueError(
                f"prompt has {input_tokens} tokens, exceeding max_input_tokens="
                f"{self.config['max_input_tokens']}"
            )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        generation_args = {
            "max_new_tokens": self.config["max_new_tokens"],
            "do_sample": self.config["do_sample"],
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.get("max_time_seconds"):
            generation_args["max_time"] = self.config["max_time_seconds"]
        try:
            with self._torch.inference_mode():
                output = self.model.generate(**encoded, **generation_args)
        except self._torch.cuda.OutOfMemoryError:
            if self.device != "cuda" or self.config["device"] != "auto":
                raise
            encoded = {name: tensor.to("cpu") for name, tensor in encoded.items()}
            self._move_to_cpu("cuda_out_of_memory_during_generation")
            with self._torch.inference_mode():
                output = self.model.generate(**encoded, **generation_args)
        generated = output[0, input_tokens:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def metadata(self):
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "revision": self.config["revision"],
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device_fallback_reason": self.device_fallback_reason,
            "config_path": str(self.config_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "config_fingerprint": self.config_fingerprint,
        }
