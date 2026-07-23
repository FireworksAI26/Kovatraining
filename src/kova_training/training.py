"""Immutable handoff metadata for a code-only training continuation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CampaignConfig, ConfigurationError

MODEL_OPT_REPOSITORY = "https://github.com/NVIDIA/Model-Optimizer.git"
MODEL_OPT_REVISION = "ec87a82927d003986d44fb7f4fa8b3d10c31b095"
ATTENTION_TARGETS = (
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)


@dataclass(frozen=True)
class AdapterPlan:
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    use_dora: bool = False
    target_modules: tuple[str, ...] = ATTENTION_TARGETS

    def validate(self) -> None:
        if not 1 <= self.rank <= 128:
            raise ConfigurationError("smoke adapter rank must be between 1 and 128")
        if self.alpha < self.rank:
            raise ConfigurationError("LoRA alpha must be at least the adapter rank")
        if not 0 <= self.dropout < 1:
            raise ConfigurationError("LoRA dropout must be in [0, 1)")
        if self.target_modules != ATTENTION_TARGETS:
            raise ConfigurationError("smoke test may target only pinned GLM attention projections")


@dataclass(frozen=True)
class TrainingHandoff:
    mode: str
    model_id: str
    model_revision: str
    recovery_repo: str
    modelopt_repository: str
    modelopt_revision: str
    adapter: AdapterPlan
    context_length: int
    smoke_steps: int
    save_steps: int
    save_hours: int
    required_environment: tuple[str, ...]
    execution_authorized: bool
    known_unverified_gates: tuple[str, ...]

    def validate(self) -> None:
        self.adapter.validate()
        if self.mode != "code_only":
            raise ConfigurationError("handoff mode must remain code_only")
        if not 16_384 <= self.context_length <= 32_768:
            raise ConfigurationError("smoke context length must stay between 16K and 32K")
        if self.smoke_steps < 100 or self.save_steps != 50 or self.save_hours != 2:
            raise ConfigurationError("smoke/checkpoint policy does not match campaign gates")
        if self.execution_authorized:
            raise ConfigurationError("a repository handoff must not authorize GPU execution")
        if len(self.modelopt_revision) != 40:
            raise ConfigurationError("ModelOpt must be pinned to a commit")
        if not self.known_unverified_gates:
            raise ConfigurationError("unverified launch gates must be explicit")


def build_handoff(config: CampaignConfig | None = None) -> TrainingHandoff:
    campaign = config or CampaignConfig()
    campaign.validate()
    handoff = TrainingHandoff(
        mode="code_only",
        model_id=campaign.model_id,
        model_revision=campaign.model_revision,
        recovery_repo=campaign.recovery_repo,
        modelopt_repository=MODEL_OPT_REPOSITORY,
        modelopt_revision=MODEL_OPT_REVISION,
        adapter=AdapterPlan(),
        context_length=16_384,
        smoke_steps=campaign.smoke_steps,
        save_steps=campaign.checkpoint_steps,
        save_hours=campaign.checkpoint_hours,
        required_environment=(
            "HF_TOKEN",
            "KOVA_CODE_REVISION",
            "KOVA_DATASET_HASH",
            "KOVA_LIVE_HOURLY_COST",
            "KOVA_RESOURCE_STARTED_AT",
            "KOVA_STARTING_CREDIT",
        ),
        execution_authorized=False,
        known_unverified_gates=(
            "live Daytona balance and eight-GPU quota",
            "same-host RTX PRO 6000 topology and NCCL",
            "GLM-5.2 ModelOpt NVFP4 adapter compatibility",
            "LoRA full-state stop-and-resume equivalence",
            "off-instance checkpoint round-trip",
        ),
    )
    handoff.validate()
    return handoff


def write_handoff(config: CampaignConfig, output: Path) -> None:
    handoff = build_handoff(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(asdict(handoff), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
