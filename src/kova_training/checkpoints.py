"""Checksum manifests and newest-valid checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointManifest:
    step: int
    base_revision: str
    dataset_hash: str
    code_revision: str
    files: dict[str, str]


REQUIRED_TRAINER_FILES = frozenset(
    {
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "kova_state.json",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    checkpoint: Path,
    *,
    step: int,
    base_revision: str,
    dataset_hash: str,
    code_revision: str,
) -> CheckpointManifest:
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    files = {
        str(path.relative_to(checkpoint)): sha256(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if not files:
        raise ValueError("checkpoint has no files")
    manifest = CheckpointManifest(step, base_revision, dataset_hash, code_revision, files)
    target = checkpoint / "manifest.json"
    temporary = checkpoint / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(asdict(manifest), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return manifest


def validate_checkpoint(checkpoint: Path) -> CheckpointManifest | None:
    try:
        data = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        manifest = CheckpointManifest(**data)
        if not manifest.files:
            return None
        for relative, expected in manifest.files.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return None
            path = checkpoint / relative_path
            if not path.is_file() or sha256(path) != expected:
                return None
        return manifest
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def validate_resumable_checkpoint(checkpoint: Path) -> CheckpointManifest | None:
    """Validate integrity plus the state required for a real training resume."""

    manifest = validate_checkpoint(checkpoint)
    if manifest is None:
        return None
    names = {Path(relative).name for relative in manifest.files}
    has_adapter = "adapter_model.safetensors" in names or "adapter_model.bin" in names
    has_rng = any(name.startswith("rng_state") and name.endswith(".pth") for name in names)
    if not has_adapter or not has_rng or not REQUIRED_TRAINER_FILES.issubset(names):
        return None
    return manifest


def newest_valid_checkpoint(root: Path) -> tuple[Path, CheckpointManifest] | None:
    candidates: list[tuple[Path, CheckpointManifest]] = []
    if not root.exists():
        return None
    for path in root.iterdir():
        if not path.is_dir():
            continue
        manifest = validate_checkpoint(path)
        if manifest is not None:
            candidates.append((path, manifest))
    return max(candidates, key=lambda item: item[1].step) if candidates else None
