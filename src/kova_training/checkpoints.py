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
            path = checkpoint / relative
            if not path.is_file() or sha256(path) != expected:
                return None
        return manifest
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


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
