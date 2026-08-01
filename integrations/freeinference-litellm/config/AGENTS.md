# Integration Configuration

## Purpose

This directory owns the standalone LiteLLM proxy configuration template used
by the FreeInference integration kit.

## Ownership

Provider-specific local proxy configuration, not ClaudeBrigade route state.

## Local Contracts

- Keep the proxy loopback-only and preserve explicit `openai/<model-id>` model
  namespaces.
- Credentials are injected through the operator environment; never put keys in
  YAML.
- Generated or operator-overridden values must not be mistaken for the main
  router’s certified catalog.

## Work Guidance

Change config in lockstep with the package’s sync/doctor behavior and contract
tests. Preserve deterministic YAML generation and explicit endpoint policy.

## Verification

```bash
cd integrations/freeinference-litellm && uv run --extra test pytest -q
```

## Child DOX Index

No child directories. `litellm.yaml` is the local template.
