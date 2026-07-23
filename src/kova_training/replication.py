"""Checksum-verified private Hugging Face checkpoint replication."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from .checkpoints import sha256, validate_resumable_checkpoint

DENIED_NAMES = frozenset({".env", "credentials.json", "hf_token", "secret.txt", "token.txt"})


class HubApi(Protocol):
    def upload_folder(self, **kwargs: Any) -> Any: ...

    def upload_file(self, **kwargs: Any) -> Any: ...


Download = Callable[..., str]


def assert_checkpoint_safe(checkpoint: Path, secret_values: Iterable[str] = ()) -> None:
    secrets = tuple(value for value in secret_values if value)
    for path in checkpoint.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint)
        if any(part.lower() in DENIED_NAMES for part in relative.parts):
            raise ValueError("checkpoint contains a credential-like filename")
        if secrets and path.stat().st_size <= 1_000_000:
            content = path.read_bytes()
            if any(secret.encode() in content for secret in secrets):
                raise ValueError("checkpoint contains a protected value")


def replicate_checkpoint(
    checkpoint: Path,
    *,
    repo_id: str,
    token: str,
    api: HubApi | None = None,
    download: Download | None = None,
) -> str:
    """Upload one valid checkpoint, verify its manifest, then advance LATEST."""

    if not token:
        raise ValueError("HF_TOKEN is required for private checkpoint replication")
    manifest = validate_resumable_checkpoint(checkpoint)
    if manifest is None:
        raise ValueError("checkpoint is not checksum-valid and fully resumable")
    assert_checkpoint_safe(checkpoint, (token,))

    if api is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]

        api = api or HfApi(token=token)
        download = download or hf_hub_download

    remote_path = f"recovery/checkpoint-{manifest.step}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(checkpoint),
        path_in_repo=remote_path,
        token=token,
        commit_message=f"checkpoint: replicate step {manifest.step}",
    )
    downloaded = Path(
        download(
            repo_id=repo_id,
            repo_type="model",
            filename=f"{remote_path}/manifest.json",
            token=token,
            force_download=True,
        )
    )
    if sha256(downloaded) != sha256(checkpoint / "manifest.json"):
        raise RuntimeError("off-instance manifest verification failed")

    marker = {
        "base_revision": manifest.base_revision,
        "checkpoint": remote_path,
        "dataset_hash": manifest.dataset_hash,
        "manifest_sha256": sha256(checkpoint / "manifest.json"),
        "step": manifest.step,
    }
    with tempfile.TemporaryDirectory() as temporary:
        marker_path = Path(temporary) / "LATEST.json"
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=str(marker_path),
            path_in_repo="recovery/LATEST.json",
            token=token,
            commit_message=f"checkpoint: promote step {manifest.step}",
        )
    return remote_path
