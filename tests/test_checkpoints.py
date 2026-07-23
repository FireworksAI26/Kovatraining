from pathlib import Path

from kova_training.checkpoints import (
    newest_valid_checkpoint,
    validate_checkpoint,
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
