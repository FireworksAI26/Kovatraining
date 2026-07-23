from dataclasses import replace
from pathlib import Path

from kova_training.budget import (
    BudgetAction,
    BudgetLedger,
    LedgerState,
    decide_budget_action,
    estimate_spend,
)
from kova_training.config import BudgetPolicy


def test_budget_actions_are_fail_closed() -> None:
    policy = BudgetPolicy()
    state = LedgerState.initial(5_000)
    assert decide_budget_action(state, policy) is BudgetAction.CONTINUE
    assert decide_budget_action(replace(state, estimated_spend=3_200), policy) is BudgetAction.WARN
    assert (
        decide_budget_action(replace(state, estimated_spend=3_950), policy)
        is BudgetAction.CHECKPOINT_AND_STOP
    )
    assert (
        decide_budget_action(replace(state, estimated_spend=4_500), policy)
        is BudgetAction.DELETE_NOW
    )


def test_smoke_cap_deletes_without_replicated_checkpoint() -> None:
    state = replace(LedgerState.initial(5_000), estimated_spend=250)
    assert decide_budget_action(state, BudgetPolicy()) is BudgetAction.DELETE_NOW


def test_ledger_round_trip(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "ledger.json")
    state = LedgerState.initial(5_000)
    ledger.save(state)
    assert ledger.load() == state
    assert not (tmp_path / "ledger.json.tmp").exists()


def test_spend_estimate_uses_buffered_rate() -> None:
    assert estimate_spend(3_600, 46.416) == 46.416
