# Integration Scripts

## Purpose

This directory contains the standalone kit’s explicit model synchronization
and operator diagnostic commands.

## Ownership

Catalog sync and diagnostics for the FreeInference integration.

## Local Contracts

- Scripts may read `FREEINFERENCE_API_KEY` only from the operator environment.
- `sync_models.py` must preserve deterministic catalog/config output and must
  not perform inference probes implicitly.
- `doctor.py` reports configuration and compatibility state without printing
  secrets, prompts, responses, or tool arguments.

## Work Guidance

Keep scripts callable from the repository root or integration directory only
when their documented paths support it. Use bounded HTTP timeouts and return
nonzero status for actionable failures.

## Verification

```bash
cd integrations/freeinference-litellm && uv run --extra test pytest -q
```

Run `doctor.py` or `sync_models.py` only with an explicitly supplied
`FREEINFERENCE_API_KEY`; both are live catalog operations rather than
credential-free help commands.

## Child DOX Index

No child directories. Each Python file is an operator command.
