"""Environment-backed settings for the FastAPI runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .service import Settings


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _boolean(environment: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def _integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def settings_from_environment(
    environment: Mapping[str, str] | None = None,
) -> Settings:
    """Build validated runtime settings without embedding environment values in code."""

    values = os.environ if environment is None else environment
    defaults = Settings()
    data_dir = values.get("PAPER_QNA_DATA_DIR", "").strip()
    return Settings(
        data_dir=Path(data_dir) if data_dir else defaults.data_dir,
        max_upload_bytes=defaults.max_upload_bytes,
        seed=_integer(
            values,
            "PAPER_QNA_SEED",
            defaults.seed,
            minimum=0,
        ),
        max_pending_jobs=_integer(
            values,
            "PAPER_QNA_MAX_PENDING_JOBS",
            defaults.max_pending_jobs,
            minimum=1,
        ),
        model_local_files_only=_boolean(
            values,
            "PAPER_QNA_MODEL_LOCAL_FILES_ONLY",
            defaults.model_local_files_only,
        ),
        prepare_question_runtime=_boolean(
            values,
            "PAPER_QNA_PREPARE_MODELS",
            True,
        ),
    )
