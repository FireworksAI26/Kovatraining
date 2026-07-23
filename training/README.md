# GPU training handoff

This directory is a **code-only handoff**. It does not provision a VM, authorize
spend, or start training. The next agent must satisfy the live balance/quota,
topology, dataset, budget-watchdog, and compatibility gates before running it.

## What is pinned

- Checkpoint: `nvidia/GLM-5.2-NVFP4`
- Revision: `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`
- NVIDIA ModelOpt 0.45.0 source: see `modelopt.lock.json`
- Initial adapter: rank 64, alpha 128, attention projections only
- Context: 16,384 tokens
- Smoke sequence: stop after step 60, restore step 50, continue through step 110

The upstream NVIDIA checkpoint is an inference-optimized ModelOpt artifact. The
repository includes a real QLoRA attempt, but GLM-5.2 NVFP4 training, FSDP2,
adapter export, and exact LoRA optimizer resume are **not proven** until the
hard-capped smoke test passes. NVIDIA's pinned example itself warns that LoRA
resume is not supported; this handoff deliberately treats successful resume as
a go/no-go experiment rather than an assumption.

## Preparation (no cloud provisioning)

Use a CUDA/PyTorch image compatible with the delivered Blackwell driver. Keep
the image's CUDA-matched `torch` wheel, then install the pinned training stack:

```bash
python -m pip install -r training/requirements-gpu.txt
python -m pip install -e .
python training/bootstrap_modelopt.py
```

Freeze a provenance-checked Hugging Face `DatasetDict` at
`/data/kova/frozen-smoke`, record its SHA-256 in `KOVA_DATASET_HASH`, and replace
the placeholder only if the frozen path differs. Benchmark tasks, patches,
hidden tests, and near-duplicates must remain excluded.

## Mandatory GPU gate

Run this inside the already-created candidate node before downloading weights:

```bash
torchrun --standalone --nproc-per-node=8 -m kova_training.topology \
  --output .campaign/topology.json
```

It requires eight unique RTX PRO 6000 Blackwell GPUs with at least 90 GiB each,
one hostname, full peer access, and a passing NCCL all-reduce. Failure must cause
the external lifecycle controller to delete the node.

## Two-phase smoke test

The protected environment must supply `HF_TOKEN`; do not place it in a command,
file, URL, log, or checkpoint. Set the non-secret state values
`KOVA_STARTING_CREDIT`, `KOVA_LIVE_HOURLY_COST`, `KOVA_RESOURCE_STARTED_AT`,
`KOVA_DATASET_HASH`, and `KOVA_CODE_REVISION`. The resource start timestamp must
include timezone information so model-download time counts toward the cap. Point
`KOVA_MODELOPT_EXAMPLE` at the pinned checkout's `examples/llm_qat` directory.

Phase one trains to step 60 and produces/replicates checkpoint 50:

```bash
KOVA_SMOKE_PHASE=stop accelerate launch \
  --config-file training/configs/accelerate/fsdp2_glm.yaml \
  training/modelopt_kova_train.py \
  --config training/configs/train/smoke_qlora.yaml
```

Terminate the process deliberately, validate `checkpoint-50/manifest.json` and
the private `recovery/LATEST.json`, then resume:

```bash
KOVA_SMOKE_PHASE=resume \
KOVA_RESUME_CHECKPOINT=/checkpoints/kova-smoke/checkpoint-50 \
accelerate launch \
  --config-file training/configs/accelerate/fsdp2_glm.yaml \
  training/modelopt_kova_train.py \
  --config training/configs/train/smoke_qlora.yaml
```

Do not start production merely because the script exits successfully. Compare
loss and parameter checksums against an uninterrupted control, confirm optimizer,
scheduler, RNG, global step, and data order restoration, run the frozen tiny
development evaluation, verify checkpoint download, and record measured peak
VRAM/RAM/disk/throughput/cost. Any mismatch stops the GLM NVFP4 path within the
$250 smoke cap.

## Deliberately not included

- A Daytona `create` call or any automatic GPU provisioning
- Production SFT/DPO/RLVR launch authorization
- Benchmark data or answers
- Secrets or generated datasets
- A claim that NVFP4 adapter training is supported

The external watchdog must continue to own VM deletion because an in-process
trainer cannot stop billing if its host freezes.
