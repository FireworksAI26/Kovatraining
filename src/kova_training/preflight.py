"""Authenticated read-only Daytona campaign preflight."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import CampaignConfig


@dataclass(frozen=True)
class PreflightReport:
    checked_at: str
    active_sandboxes: int
    account_scope_verified: bool
    live_quota_verified: bool
    live_balance: float | None
    requested_gpu_count: int
    requested_gpu_type: str
    maximum_smoke_cost: float
    blockers: tuple[str, ...]

    @property
    def ready_for_billable_smoke(self) -> bool:
        return not self.blockers

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


async def run_daytona_preflight(config: CampaignConfig) -> PreflightReport:
    """Read current resources and fail closed when account evidence is unavailable."""

    from daytona import AsyncDaytona
    from daytona_api_client_async import OrganizationsApi

    config.validate()
    blockers: list[str] = []
    async with AsyncDaytona() as daytona:
        sandboxes = [sandbox async for sandbox in daytona.list()]
        if sandboxes:
            blockers.append("existing Daytona sandboxes must be reconciled before launch")

        scope_verified = False
        quota_verified = False
        balance: float | None = None
        try:
            organizations = await OrganizationsApi(daytona._api_client).list_organizations(
                _request_timeout=20
            )
            scope_verified = bool(organizations)
            if organizations:
                usage = await OrganizationsApi(daytona._api_client).get_organization_usage_overview(
                    organizations[0].id, _request_timeout=20
                )
                quota_verified = any(
                    region.total_gpu_quota >= config.resources.gpu_count
                    and region.max_memory_per_gpu_sandbox is not None
                    and region.max_disk_per_gpu_sandbox is not None
                    for region in usage.region_usage
                )
        except Exception:
            blockers.append("API key cannot read live Daytona organization quota")

        # Daytona's sandbox API does not expose billing credit. A protected,
        # account-scoped source must provide it before any create call.
        blockers.append("live Daytona starting balance is unavailable")
        if not scope_verified:
            blockers.append("Daytona organization scope is unverified")
        if not quota_verified:
            blockers.append("eight-GPU live quota is unverified")

    return PreflightReport(
        checked_at=datetime.now(UTC).isoformat(),
        active_sandboxes=len(sandboxes),
        account_scope_verified=scope_verified,
        live_quota_verified=quota_verified,
        live_balance=balance,
        requested_gpu_count=config.resources.gpu_count,
        requested_gpu_type=config.resources.gpu_type,
        maximum_smoke_cost=round(config.maximum_smoke_cost, 2),
        blockers=tuple(dict.fromkeys(blockers)),
    )
