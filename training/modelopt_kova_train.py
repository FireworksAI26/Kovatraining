# Adapted from NVIDIA Model Optimizer examples/llm_qat/train.py.
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""KovaDev GLM NVFP4 adapter smoke entrypoint.

Run this from the repository root with the pinned ModelOpt llm_qat example on
PYTHONPATH. It never provisions or deletes cloud resources.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_OPT_EXAMPLE = Path(
    os.environ.get("KOVA_MODELOPT_EXAMPLE", ".third_party/modelopt/examples/llm_qat")
).resolve()
if not (MODEL_OPT_EXAMPLE / "arguments.py").is_file():
    raise RuntimeError("KOVA_MODELOPT_EXAMPLE does not point to the pinned llm_qat example")
sys.path.insert(0, str(MODEL_OPT_EXAMPLE))

import modelopt.torch.opt as mto  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from arguments import get_training_args  # type: ignore[import-not-found]  # noqa: E402
from modelopt.torch.quantization.plugins.transformers_trainer import QATTrainer  # noqa: E402
from peft import LoraConfig, TaskType  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from utils import (  # type: ignore[import-not-found]  # noqa: E402
    get_metrics_with_perplexity,
    make_supervised_data_module,
)

from kova_training.budget import (  # noqa: E402
    BudgetAction,
    BudgetLedger,
    LedgerState,
    decide_budget_action,
    estimate_spend,
)
from kova_training.checkpoints import write_manifest  # noqa: E402
from kova_training.config import CampaignConfig  # noqa: E402
from kova_training.replication import replicate_checkpoint  # noqa: E402
from kova_training.training import ATTENTION_TARGETS, AdapterPlan  # noqa: E402

mto.enable_huggingface_checkpointing()


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required protected environment variable is missing: {name}")
    return value


class CampaignCallback(TrainerCallback):
    """Time-based saves, budget stops, manifests, and private replication."""

    def __init__(self, config: CampaignConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.last_save = time.monotonic()
        self.resource_started_at = datetime.fromisoformat(
            required_environment("KOVA_RESOURCE_STARTED_AT").replace("Z", "+00:00")
        )
        if self.resource_started_at.tzinfo is None:
            raise RuntimeError("KOVA_RESOURCE_STARTED_AT must include a timezone")
        if self.resource_started_at > datetime.now(UTC):
            raise RuntimeError("KOVA_RESOURCE_STARTED_AT cannot be in the future")
        live_hourly_cost = float(required_environment("KOVA_LIVE_HOURLY_COST"))
        if live_hourly_cost <= 0:
            raise RuntimeError("KOVA_LIVE_HOURLY_COST must be positive")
        self.buffered_hourly_cost = live_hourly_cost * config.cost_buffer
        maximum_cost = self.buffered_hourly_cost * config.smoke_ttl_minutes / 60
        if maximum_cost > config.budget.smoke_cap:
            raise RuntimeError("live price can exceed the $250 smoke cap before TTL cleanup")
        self.ledger = BudgetLedger(
            Path(os.environ.get("KOVA_BUDGET_LEDGER", ".campaign/ledger.json"))
        )
        starting_credit = float(required_environment("KOVA_STARTING_CREDIT"))
        if starting_credit < config.maximum_smoke_cost:
            raise RuntimeError("starting credit is below the buffered smoke maximum")
        if not self.ledger.path.exists():
            self.ledger.save(LedgerState.initial(starting_credit))

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state, kwargs
        now = time.monotonic()
        elapsed = (datetime.now(UTC) - self.resource_started_at).total_seconds()
        ledger = self.ledger.load()
        ledger = replace(
            ledger,
            estimated_spend=estimate_spend(elapsed, self.buffered_hourly_cost),
        )
        self.ledger.save(ledger)
        budget_action = decide_budget_action(ledger, self.config.budget)
        if now - self.last_save >= self.config.checkpoint_hours * 3_600:
            control.should_save = True
        if budget_action in {BudgetAction.CHECKPOINT_AND_STOP, BudgetAction.DELETE_NOW}:
            control.should_save = budget_action is BudgetAction.CHECKPOINT_AND_STOP
            control.should_training_stop = True
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del control, kwargs
        if not state.is_world_process_zero:
            return None
        checkpoint = self.output_dir / f"checkpoint-{state.global_step}"
        kova_state = {
            "data_cursor": {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "seed": args.data_seed if args.data_seed is not None else args.seed,
            },
            "spending_estimate": self.ledger.load().estimated_spend,
        }
        (checkpoint / "kova_state.json").write_text(
            json.dumps(kova_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_manifest(
            checkpoint,
            step=state.global_step,
            base_revision=self.config.model_revision,
            dataset_hash=required_environment("KOVA_DATASET_HASH"),
            code_revision=required_environment("KOVA_CODE_REVISION"),
        )
        replicate_checkpoint(
            checkpoint,
            repo_id=self.config.recovery_repo,
            token=required_environment("HF_TOKEN"),
        )
        ledger = self.ledger.load()
        self.ledger.save(replace(ledger, validated_checkpoint_off_instance=True))
        self.last_save = time.monotonic()
        return None


def validate_model_targets(model: Any) -> None:
    leaf_names = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
    missing = set(ATTENTION_TARGETS) - leaf_names
    if missing:
        raise RuntimeError(f"GLM attention target modules are missing: {sorted(missing)}")


def validate_trainable_parameters(model: Any) -> None:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError("adapter attachment produced no trainable parameters")
    if any("lora_" not in name for name in names):
        raise RuntimeError("a non-adapter parameter is trainable")
    if any("mlp.experts" in name or ".gate." in name for name in names):
        raise RuntimeError("routing or expert parameters must remain frozen during smoke")


def phase_settings(training_args: Any) -> str:
    phase = os.environ.get("KOVA_SMOKE_PHASE", "").strip()
    if phase == "stop":
        training_args.max_steps = 60
        training_args.resume_from_checkpoint = None
    elif phase == "resume":
        checkpoint = required_environment("KOVA_RESUME_CHECKPOINT")
        training_args.max_steps = 110
        training_args.resume_from_checkpoint = checkpoint
    else:
        raise RuntimeError("KOVA_SMOKE_PHASE must be 'stop' or 'resume'")
    return phase


def train() -> None:
    config = CampaignConfig()
    config.validate()
    AdapterPlan().validate()
    required_environment("HF_TOKEN")
    required_environment("KOVA_DATASET_HASH")
    required_environment("KOVA_CODE_REVISION")

    model_args, training_args, data_args, distill_args = get_training_args()
    if distill_args.distill:
        raise RuntimeError("distillation is not permitted during the smoke test")
    if model_args.model_name_or_path != config.model_id:
        raise RuntimeError("smoke test must use the pinned NVFP4 checkpoint")
    if Path(training_args.output_dir).resolve() == Path("/"):
        raise RuntimeError("output directory cannot be the filesystem root")
    phase = phase_settings(training_args)

    output_dir = Path(training_args.output_dir)
    last_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
    if phase == "stop" and last_checkpoint is not None:
        raise RuntimeError("stop phase requires an empty output directory")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
    )
    if getattr(model.config, "model_type", None) != "glm_moe_dsa":
        raise RuntimeError("checkpoint architecture is not glm_moe_dsa")
    quantization = getattr(model.config, "quantization_config", {})
    if quantization.get("quant_method") != "modelopt":
        raise RuntimeError("checkpoint is not the pinned ModelOpt quantization")
    model.config.use_cache = False
    validate_model_targets(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        model_max_length=model_args.model_max_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    training_args.lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=list(ATTENTION_TARGETS),
        task_type=TaskType.CAUSAL_LM,
        use_dora=False,
    )
    data_module = make_supervised_data_module(data_args, tokenizer)
    callback = CampaignCallback(config, output_dir)
    trainer = QATTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[callback],
        **data_module,
    )
    validate_trainable_parameters(trainer.model)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if phase == "stop" and trainer.state.global_step != 60:
        raise RuntimeError("planned stop phase did not reach step 60")
    if phase == "resume" and trainer.state.global_step < 110:
        raise RuntimeError("resume phase did not complete at least 110 total steps")
    metrics = get_metrics_with_perplexity(trainer.evaluate())
    (output_dir / f"{phase}-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainer.save_state()
    trainer.save_model(str(output_dir / f"{phase}-final"))


if __name__ == "__main__":
    train()
