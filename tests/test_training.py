import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from kova_training.config import ConfigurationError
from kova_training.training import AdapterPlan, build_handoff


def test_handoff_is_code_only_and_pinned() -> None:
    handoff = build_handoff()
    handoff.validate()
    assert handoff.mode == "code_only"
    assert handoff.execution_authorized is False
    assert handoff.adapter.rank == 64
    assert handoff.smoke_steps >= 100
    assert len(handoff.model_revision) == 40
    assert len(handoff.modelopt_revision) == 40


def test_handoff_serializes_without_credentials(tmp_path) -> None:
    from kova_training.config import CampaignConfig
    from kova_training.training import write_handoff

    output = tmp_path / "handoff.json"
    write_handoff(CampaignConfig(), output)
    data = json.loads(output.read_text())
    assert "HF_TOKEN" in data["required_environment"]
    assert "token" not in data


def test_adapter_targets_cannot_expand_to_experts() -> None:
    with pytest.raises(ConfigurationError, match="attention projections"):
        replace(AdapterPlan(), target_modules=("q_proj", "down_proj")).validate()


def test_training_configs_are_parseable_and_pinned() -> None:
    root = Path(__file__).parents[1]
    configs = list((root / "training" / "configs").rglob("*.yaml"))
    assert configs
    assert all(isinstance(yaml.safe_load(path.read_text()), dict) for path in configs)

    lock = json.loads((root / "training" / "modelopt.lock.json").read_text())
    handoff = build_handoff()
    assert lock["repository"] == handoff.modelopt_repository
    assert lock["revision"] == handoff.modelopt_revision
