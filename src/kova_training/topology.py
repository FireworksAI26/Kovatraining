"""Fail-closed same-host GPU and NCCL probe for the smoke-test node."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_GPUS = 8
MINIMUM_VRAM_MIB = 90 * 1024


class TopologyError(RuntimeError):
    """Raised when the delivered GPU node does not meet the campaign gate."""


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    uuid: str
    memory_mib: int
    compute_capability: str


@dataclass(frozen=True)
class TopologyReport:
    checked_at: str
    hostname: str
    world_size: int
    gpus: tuple[Gpu, ...]
    same_host: bool
    peer_access: bool
    nccl_all_reduce: bool
    topology: str
    passed: bool


def _run(command: Sequence[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed, non-shell diagnostic commands
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout[:64_000]


def query_gpus() -> tuple[Gpu, ...]:
    output = _run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )
    gpus: list[Gpu] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise TopologyError("unexpected nvidia-smi inventory format")
        gpus.append(Gpu(int(fields[0]), fields[1], fields[2], int(fields[3]), fields[4]))
    if len(gpus) != EXPECTED_GPUS:
        raise TopologyError(f"expected {EXPECTED_GPUS} GPUs, found {len(gpus)}")
    if len({gpu.uuid for gpu in gpus}) != EXPECTED_GPUS:
        raise TopologyError("GPU UUIDs are not unique")
    for gpu in gpus:
        if "RTX PRO 6000" not in gpu.name or "Blackwell" not in gpu.name:
            raise TopologyError("delivered GPU model is not RTX PRO 6000 Blackwell")
        if gpu.memory_mib < MINIMUM_VRAM_MIB:
            raise TopologyError("delivered GPU has less than 90 GiB VRAM")
    return tuple(gpus)


def run_probe() -> TopologyReport:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.distributed as dist  # type: ignore[import-not-found]
    except ImportError as error:
        raise TopologyError("PyTorch with CUDA/NCCL is required") from error

    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise TopologyError("CUDA and NCCL must both be available")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != EXPECTED_GPUS:
        raise TopologyError(f"torchrun world size must be {EXPECTED_GPUS}")
    torch.cuda.set_device(local_rank)

    hostnames: list[str | None] = [None] * world_size
    dist.all_gather_object(hostnames, socket.gethostname())
    same_host = len(set(hostnames)) == 1

    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    nccl_ok = float(value.item()) == expected

    peer_ok = True
    if rank == 0:
        peer_ok = all(
            torch.cuda.can_device_access_peer(left, right)
            for left in range(EXPECTED_GPUS)
            for right in range(EXPECTED_GPUS)
            if left != right
        )
    flag = torch.tensor(int(peer_ok), device=f"cuda:{local_rank}")
    dist.broadcast(flag, src=0)
    peer_ok = bool(flag.item())

    gpus = query_gpus()
    topology = _run(("nvidia-smi", "topo", "-m")) if rank == 0 else ""
    report = TopologyReport(
        checked_at=datetime.now(UTC).isoformat(),
        hostname=socket.gethostname(),
        world_size=world_size,
        gpus=gpus,
        same_host=same_host,
        peer_access=peer_ok,
        nccl_all_reduce=nccl_ok,
        topology=topology,
        passed=same_host and peer_ok and nccl_ok and len(gpus) == EXPECTED_GPUS,
    )
    dist.barrier()
    if not report.passed:
        raise TopologyError("GPU topology gate failed")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kova-gpu-probe")
    parser.add_argument("--output", type=Path, default=Path(".campaign/topology.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe()
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(args.output), "passed": report.passed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
