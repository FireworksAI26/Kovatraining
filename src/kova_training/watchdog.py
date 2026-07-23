"""Training-health watchdog decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .budget import BudgetAction, LedgerState, decide_budget_action
from .config import BudgetPolicy


class HealthAction(StrEnum):
    CONTINUE = "continue"
    RESUME_ONCE = "resume_once"
    CHECKPOINT_AND_DELETE = "checkpoint_and_delete"
    DELETE_NOW = "delete_now"


@dataclass(frozen=True)
class HealthSample:
    heartbeat_at: datetime
    consecutive_failures: int = 0
    has_nan: bool = False
    oom: bool = False
    deadlocked: bool = False
    checkpoint_valid: bool = False
    checkpoint_replicated: bool = False


def decide_health_action(
    sample: HealthSample,
    ledger: LedgerState,
    policy: BudgetPolicy,
    *,
    now: datetime | None = None,
    idle_timeout_seconds: int = 900,
) -> HealthAction:
    current = now or datetime.now(UTC)
    budget_action = decide_budget_action(ledger, policy)
    if budget_action is BudgetAction.DELETE_NOW:
        return HealthAction.DELETE_NOW
    if budget_action is BudgetAction.CHECKPOINT_AND_STOP:
        return HealthAction.CHECKPOINT_AND_DELETE
    idle_seconds = (current - sample.heartbeat_at).total_seconds()
    if idle_seconds > idle_timeout_seconds or sample.has_nan or sample.deadlocked:
        return (
            HealthAction.CHECKPOINT_AND_DELETE
            if sample.checkpoint_valid
            else HealthAction.DELETE_NOW
        )
    if sample.oom or sample.consecutive_failures >= 2:
        return (
            HealthAction.CHECKPOINT_AND_DELETE
            if sample.checkpoint_valid
            else HealthAction.DELETE_NOW
        )
    if (
        sample.consecutive_failures == 1
        and sample.checkpoint_valid
        and sample.checkpoint_replicated
    ):
        return HealthAction.RESUME_ONCE
    return HealthAction.CONTINUE
