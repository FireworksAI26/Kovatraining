"""Clone the exact NVIDIA ModelOpt source used by the handoff."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path(".third_party/modelopt"))
    args = parser.parse_args()
    lock = json.loads((Path(__file__).parent / "modelopt.lock.json").read_text())
    if args.destination.exists():
        raise SystemExit(f"destination already exists: {args.destination}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            lock["repository"],
            str(args.destination),
        ]
    )
    run(["git", "checkout", "--detach", lock["revision"]], cwd=args.destination)
    actual = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],
        cwd=args.destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != lock["revision"]:
        raise SystemExit("ModelOpt revision verification failed")
    print(actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
