"""Command-line entry point for safe campaign operations."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .config import CampaignConfig
from .preflight import run_daytona_preflight
from .training import write_handoff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kova-campaign")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate", help="validate immutable campaign policy")
    preflight = subcommands.add_parser("preflight", help="run read-only Daytona checks")
    preflight.add_argument("--output", type=Path, default=Path(".campaign/preflight.json"))
    handoff = subcommands.add_parser("handoff", help="write the code-only training handoff")
    handoff.add_argument("--output", type=Path, default=Path(".campaign/handoff.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CampaignConfig()
    if args.command == "validate":
        config.validate()
        print(
            json.dumps(
                {
                    "buffered_hourly_cost": round(config.buffered_hourly_cost, 2),
                    "maximum_smoke_cost": round(config.maximum_smoke_cost, 2),
                    "status": "valid",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "preflight":
        report = asyncio.run(run_daytona_preflight(config))
        report.save(args.output)
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0 if report.ready_for_billable_smoke else 2
    if args.command == "handoff":
        write_handoff(config, args.output)
        print(json.dumps({"output": str(args.output), "status": "code_only"}, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
