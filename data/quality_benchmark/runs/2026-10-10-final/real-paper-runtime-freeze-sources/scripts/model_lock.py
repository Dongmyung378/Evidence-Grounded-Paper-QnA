"""Load and validate the final model identity lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config" / "model_lock.json"
EXPECTED_ROLES = {"embedding", "reranker", "translation", "legacy_generator"}


def load_model_lock(path: Path = CONFIG_PATH) -> dict:
    path = Path(path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("status") != "frozen":
        raise ValueError("Unsupported or unfrozen model lock")
    if lock.get("seed") != 378:
        raise ValueError("The model lock seed must remain 378")
    models = lock.get("models") or {}
    if set(models) != EXPECTED_ROLES:
        raise ValueError("The model lock does not define the required roles")
    for role, model in models.items():
        if not isinstance(model.get("name"), str) or not model["name"].strip():
            raise ValueError(f"Model name is missing for role: {role}")
        revision = model.get("revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError(f"Model revision is not a full commit hash: {role}")
    return lock


def locked_model(role: str, path: Path = CONFIG_PATH) -> dict:
    lock = load_model_lock(path)
    if role not in lock["models"]:
        raise ValueError(f"Unknown model lock role: {role}")
    return dict(lock["models"][role])


def model_lock_fingerprint(path: Path = CONFIG_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
