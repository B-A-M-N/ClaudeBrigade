# Integration Package Source

## Purpose

This directory owns the installable Python source for the standalone
FreeInference LiteLLM kit.

## Ownership

Integration package behavior, CLI entry points, catalog synchronization,
endpoint policy, and compatibility harnesses.

## Local Contracts

- Keep this package independent of ClaudeBrigade SQLite state and router
  singletons.
- Public artifacts may contain model metadata and aggregate usage only; never
  prompts, completions, tool arguments, or credentials.
- Endpoint selection must respect explicit overrides/configured defaults before
  cache-based `auto` selection and must not silently fall back during probes.

## Work Guidance

Keep provider HTTP concerns bounded and testable. Put policy in explicit
functions/classes rather than CLI-only branches. Update `pyproject.toml` and
the lockfile through the integration package workflow when dependencies change.

## Verification

```bash
cd integrations/freeinference-litellm && uv run --extra test pytest -q
```

## Child DOX Index

| Path | Purpose |
|------|---------|
| `fi_litellm/` | Python package implementation |
