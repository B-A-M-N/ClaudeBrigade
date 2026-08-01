# FreeInference LiteLLM Package

## Purpose

This package implements the standalone integration CLI, catalog synchronization,
endpoint selection policy, and compatibility harness.

## Ownership

Provider integration logic under the package’s public Python API and console
entry point.

## Local Contracts

- `sync.py` consumes authenticated model catalogs and produces deterministic
  metadata/configuration.
- `endpoint_policy.py` selects endpoints according to explicit/default/cache
  policy and records why a choice was made.
- `harness.py` performs explicitly requested compatibility checks and must not
  leak request/response bodies into reports.
- `cli.py` is orchestration only; preserve library-testable behavior in the
  underlying modules.

## Work Guidance

Use typed bounded inputs, stable digests, and explicit errors. Avoid hidden
network calls during import or ordinary local commands. Preserve independence
from the main router’s state database.

## Verification

```bash
cd integrations/freeinference-litellm && uv run --extra test pytest -q
```

## Child DOX Index

No child directories. Generated `__pycache__` and egg-info files are not source.
