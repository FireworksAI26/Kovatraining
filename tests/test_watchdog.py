from dataclasses import replace
from datetime import UTC, datetime, timedelta

from kova_training.budget import LedgerState
from kova_training.config import BudgetPolicy
from kova_training.watchdog import HealthAction, HealthSample, decide_health_action

NOW = datetime(2026, 7, 23, tzinfo=UTC)


def sample(**changes: object) -> HealthSample:
    return replace(HealthSample(heartbeat_at=NOW), **changes)


def test_one_failure_resumes_only_from_replicated_checkpoint() -> None:
    action = decide_health_action(
        sample(consecutive_failures=1, checkpoint_valid=True, checkpoint_replicated=True),
        LedgerState.initial(5_000),
        BudgetPolicy(),
        now=NOW,
    )
    assert action is HealthAction.RESUME_ONCE


def test_repeated_failure_stops_and_deletes() -> None:
    action = decide_health_action(
        sample(consecutive_failures=2, checkpoint_valid=True),
        LedgerState.initial(5_000),
        BudgetPolicy(),
        now=NOW,
    )
    assert action is HealthAction.CHECKPOINT_AND_DELETE


def test_stale_worker_without_checkpoint_is_deleted() -> None:
    action = decide_health_action(
        sample(heartbeat_at=NOW - timedelta(minutes=16)),
        LedgerState.initial(5_000),
        BudgetPolicy(),
        now=NOW,
    )
    assert action is HealthAction.DELETE_NOW


def test_emergency_budget_always_deletes() -> None:
    ledger = replace(LedgerState.initial(5_000), estimated_spend=4_500)
    assert (
        decide_health_action(sample(checkpoint_valid=True), ledger, BudgetPolicy(), now=NOW)
        is HealthAction.DELETE_NOW
    )
