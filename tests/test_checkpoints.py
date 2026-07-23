import json
from pathlib import Path

from kova_training.checkpoints import (
    newest_valid_checkpoint,
    validate_checkpoint,
    validate_resumable_checkpoint,
    write_manifest,
)


def create_checkpoint(root: Path, step: int, content: str) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "trainer.json").write_text(content, encoding="utf-8")
    write_manifest(
        checkpoint,
        step=step,
        base_revision="base-sha",
        dataset_hash="data-sha",
        code_revision="code-sha",
    )
    return checkpoint


def test_newest_checksum_valid_checkpoint_wins(tmp_path: Path) -> None:
    older = create_checkpoint(tmp_path, 50, "old")
    newer = create_checkpoint(tmp_path, 100, "new")
    assert validate_checkpoint(older) is not None
    selected = newest_valid_checkpoint(tmp_path)
    assert selected is not None
    assert selected[0] == newer


def test_corrupt_newest_checkpoint_falls_back(tmp_path: Path) -> None:
    older = create_checkpoint(tmp_path, 50, "old")
    newer = create_checkpoint(tmp_path, 100, "new")
    (newer / "trainer.json").write_text("corrupt", encoding="utf-8")
    selected = newest_valid_checkpoint(tmp_path)
    assert selected is not None
    assert selected[0] == older


def test_resumable_checkpoint_requires_complete_training_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-50"
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
    assert validate_resumable_checkpoint(checkpoint) is not None
    (checkpoint / "optimizer.pt").unlink()
    assert validate_resumable_checkpoint(checkpoint) is None


def test_manifest_cannot_escape_checkpoint_directory(tmp_path: Path) -> None:
    checkpoint = create_checkpoint(tmp_path, 50, "safe")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {"../outside": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_checkpoint(checkpoint) is None
