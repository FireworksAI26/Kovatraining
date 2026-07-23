from unittest.mock import Mock, patch

import pytest

from kova_training.topology import TopologyError, query_gpus


def inventory(count: int = 8, memory: int = 98304) -> str:
    return "\n".join(
        f"{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU-{index}, {memory}, 12.0"
        for index in range(count)
    )


@patch("kova_training.topology.subprocess.run")
def test_gpu_inventory_requires_exact_expected_shape(run: Mock) -> None:
    run.return_value = Mock(stdout=inventory())
    gpus = query_gpus()
    assert len(gpus) == 8
    assert all(gpu.memory_mib >= 90 * 1024 for gpu in gpus)


@patch("kova_training.topology.subprocess.run")
def test_gpu_inventory_fails_closed_on_too_few_gpus(run: Mock) -> None:
    run.return_value = Mock(stdout=inventory(count=7))
    with pytest.raises(TopologyError, match="expected 8"):
        query_gpus()
