from pathlib import Path
from typing import Any

import pytest

from kova_training.checkpoints import write_manifest
from kova_training.replication import replicate_checkpoint


class FakeApi:
    def __init__(self) -> None:
        self.folder_calls: list[dict[str, Any]] = []
        self.file_calls: list[dict[str, Any]] = []

    def upload_folder(self, **kwargs: Any) -> None:
        self.folder_calls.append(kwargs)

    def upload_file(self, **kwargs: Any) -> None:
        self.file_calls.append(kwargs)


def resumable_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint-50"
    checkpoint.mkdir()
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state_0.pth",
        "kova_state.json",
    ):
        (checkpoint / name).write_text(name, encoding="utf-8")
    write_manifest(
        checkpoint,
        step=50,
        base_revision="base-sha",
        dataset_hash="data-sha",
        code_revision="code-sha",
    )
    return checkpoint


def test_replication_promotes_only_after_manifest_round_trip(tmp_path: Path) -> None:
    checkpoint = resumable_checkpoint(tmp_path)
    api = FakeApi()

    def download(**kwargs: Any) -> str:
        assert kwargs["filename"] == "recovery/checkpoint-50/manifest.json"
        return str(checkpoint / "manifest.json")

    remote = replicate_checkpoint(
        checkpoint,
        repo_id="owner/private",
        token="protected-test-value",
        api=api,
        download=download,
    )
    assert remote == "recovery/checkpoint-50"
    assert len(api.folder_calls) == 1
    assert len(api.file_calls) == 1
    assert api.file_calls[0]["path_in_repo"] == "recovery/LATEST.json"


def test_replication_rejects_secret_bearing_checkpoint(tmp_path: Path) -> None:
    checkpoint = resumable_checkpoint(tmp_path)
    (checkpoint / "notes.txt").write_text("protected-test-value", encoding="utf-8")
    write_manifest(
        checkpoint,
        step=50,
        base_revision="base-sha",
        dataset_hash="data-sha",
        code_revision="code-sha",
    )
    with pytest.raises(ValueError, match="protected value"):
        replicate_checkpoint(
            checkpoint,
            repo_id="owner/private",
            token="protected-test-value",
            api=FakeApi(),
            download=lambda **_: str(checkpoint / "manifest.json"),
        )
