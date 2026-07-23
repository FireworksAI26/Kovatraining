"""Validated campaign policy and resource configuration."""

from __future__ import annotations

from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when a campaign policy is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class ResourceRequest:
    """The only supported smoke-test resource shape."""

    gpu_count: int = 8
    gpu_type: str = "RTX-PRO-6000"
    cpu: int = 64
    memory_gib: int = 500
    disk_gib: int = 2_000


@dataclass(frozen=True)
class BudgetPolicy:
    """Campaign spending limits in US dollars."""

    smoke_cap: float = 250.0
    planned_cap: float = 4_000.0
    warn_at: tuple[float, float] = (3_000.0, 3_500.0)
    graceful_stop: float = 3_900.0
    emergency_stop: float = 4_500.0
    absolute_cap: float = 5_000.0
    protected_reserve: float = 150.0

    def validate(self) -> None:
        values = (
            self.smoke_cap,
            self.planned_cap,
            *self.warn_at,
            self.graceful_stop,
            self.emergency_stop,
            self.absolute_cap,
            self.protected_reserve,
        )
        if any(value <= 0 for value in values):
            raise ConfigurationError("budget values must be positive")
        if self.warn_at != tuple(sorted(self.warn_at)):
            raise ConfigurationError("warning thresholds must be sorted")
        if not (
            self.warn_at[-1]
            < self.graceful_stop
            <= self.planned_cap
            < self.emergency_stop
            < self.absolute_cap
        ):
            raise ConfigurationError("budget stop thresholds are not fail-closed")
        if self.protected_reserve >= self.smoke_cap:
            raise ConfigurationError("protected reserve must be smaller than the smoke cap")


@dataclass(frozen=True)
class CampaignConfig:
    """Immutable configuration for one production campaign."""

    resources: ResourceRequest = ResourceRequest()
    budget: BudgetPolicy = BudgetPolicy()
    model_id: str = "nvidia/GLM-5.2-NVFP4"
    model_revision: str = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
    recovery_repo: str = "Kovacreations/KovaDev-Coder"
    smoke_ttl_minutes: int = 290
    expected_hourly_cost: float = 38.68
    cost_buffer: float = 1.20
    checkpoint_steps: int = 50
    checkpoint_hours: int = 2
    smoke_steps: int = 110

    @property
    def buffered_hourly_cost(self) -> float:
        return self.expected_hourly_cost * self.cost_buffer

    @property
    def maximum_smoke_cost(self) -> float:
        return self.buffered_hourly_cost * self.smoke_ttl_minutes / 60

    def validate(self) -> None:
        self.budget.validate()
        if self.resources.gpu_count != 8:
            raise ConfigurationError("the GLM smoke test requires exactly eight GPUs")
        if self.resources.gpu_type != "RTX-PRO-6000":
            raise ConfigurationError("the smoke test is pinned to RTX-PRO-6000")
        if min(self.resources.cpu, self.resources.memory_gib, self.resources.disk_gib) <= 0:
            raise ConfigurationError("resource requests must be positive")
        if self.smoke_ttl_minutes <= 0:
            raise ConfigurationError("smoke TTL must be positive")
        if self.expected_hourly_cost <= 0 or self.cost_buffer < 1:
            raise ConfigurationError("cost forecast must be positive and buffered")
        if self.maximum_smoke_cost > self.budget.smoke_cap:
            raise ConfigurationError("smoke TTL can exceed the $250 hard cap")
        if self.checkpoint_steps <= 0 or self.checkpoint_hours <= 0:
            raise ConfigurationError("checkpoint intervals must be positive")
        if self.smoke_steps < 100:
            raise ConfigurationError("smoke test must complete at least 100 optimizer steps")
        if not self.model_revision or self.model_revision == "main":
            raise ConfigurationError("model revision must be immutable")
