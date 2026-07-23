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
