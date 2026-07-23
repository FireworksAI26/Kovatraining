"""Atomic spend ledger and fail-closed budget decisions."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from .config import BudgetPolicy


class BudgetAction(StrEnum):
    CONTINUE = "continue"
    WARN = "warn"
    CHECKPOINT_AND_STOP = "checkpoint_and_stop"
    DELETE_NOW = "delete_now"


@dataclass(frozen=True)
class LedgerState:
    starting_credit: float
    estimated_spend: float
    last_observed_credit: float | None
    observed_at: str
    validated_checkpoint_off_instance: bool = False

    @classmethod
    def initial(cls, starting_credit: float) -> LedgerState:
        if starting_credit <= 0:
            raise ValueError("starting credit must be positive")
        return cls(
            starting_credit=starting_credit,
            estimated_spend=0.0,
            last_observed_credit=starting_credit,
            observed_at=datetime.now(UTC).isoformat(),
        )


class BudgetLedger:
    """Persists spend state with replace-on-success semantics."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> LedgerState:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return LedgerState(**data)

    def save(self, state: LedgerState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(asdict(state), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)


def decide_budget_action(state: LedgerState, policy: BudgetPolicy) -> BudgetAction:
    """Return the most conservative action justified by current spend."""

    spend = state.estimated_spend
    if spend >= policy.emergency_stop or spend >= policy.absolute_cap:
        return BudgetAction.DELETE_NOW
    if spend >= policy.graceful_stop:
        return BudgetAction.CHECKPOINT_AND_STOP
    if spend >= policy.warn_at[0]:
        return BudgetAction.WARN
    if spend >= policy.smoke_cap and not state.validated_checkpoint_off_instance:
        return BudgetAction.DELETE_NOW
    return BudgetAction.CONTINUE


def estimate_spend(elapsed_seconds: float, buffered_hourly_cost: float) -> float:
    if elapsed_seconds < 0 or buffered_hourly_cost <= 0:
        raise ValueError("elapsed time and hourly cost are invalid")
    return elapsed_seconds / 3_600 * buffered_hourly_cost
