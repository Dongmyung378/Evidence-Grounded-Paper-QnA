"""Verify measured historical source bytes without attributing them to new code."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/historical_evaluation.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_source(relative, expected_hash, snapshots, root=ROOT):
    """Return a byte-verified current source or an explicitly retained snapshot."""
    root = Path(root).resolve()
    current = (root / relative).resolve()
    if not current.is_relative_to(root):
        raise ValueError("Frozen source path escapes the project")
    if current.is_file() and sha256(current) == expected_hash:
        return current
    archived = snapshots.get(relative)
    if not archived:
        raise ValueError(f"Frozen source changed without a historical snapshot: {relative}")
    snapshot = (root / archived).resolve()
    baseline_root = (root / "data/baselines").resolve()
    if not snapshot.is_relative_to(baseline_root):
        raise ValueError("Historical source snapshot must stay inside data/baselines")
    if not snapshot.is_file() or sha256(snapshot) != expected_hash:
        raise ValueError(f"Historical source snapshot hash mismatch: {relative}")
    return snapshot


def load_historical_provenance():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("scope") != "historical_evaluation_only":
        raise ValueError("Unsupported historical evaluation provenance")
    frozen = ROOT / config["frozen_manifest"]
    if sha256(frozen) != config["frozen_manifest_sha256"]:
        raise ValueError("The historical evaluation manifest changed")
    return config


def verify_frozen_source(relative, expected_hash):
    config = load_historical_provenance()
    return resolve_source(relative, expected_hash, config["source_snapshots"])
