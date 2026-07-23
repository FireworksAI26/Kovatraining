# KovaTraining

Fail-closed infrastructure and recovery controls for the KovaDev coding-model
adapter campaign.

The repository does not make a billable Daytona request unless a fresh preflight
can prove the account scope, live quota, starting balance, and an empty resource
inventory. The initial campaign action is a hard-capped smoke test, not production
training.

## Local validation

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/kova-campaign validate
.venv/bin/pytest
```

Authenticated read-only preflight:

```bash
.venv/bin/kova-campaign preflight
```

The command exits with status 2 and records its blockers when the Daytona key
cannot read organization quota or a protected starting-balance source is absent.

## Code-only training handoff

The repository includes a pinned, non-provisioning GPU handoff in
[`training/`](training/README.md). It contains the eight-GPU NCCL probe, GLM
attention-only rank-64 adapter smoke entrypoint, forced stop/resume sequence,
budget callback, checksum manifests, and private checkpoint replication.

Generate a machine-readable summary without starting training:

```bash
.venv/bin/kova-campaign handoff
```

No cloud `create` call exists in this repository. The training path remains
unverified until a future agent passes the live account, topology, and $250
smoke gates described in the handoff.
