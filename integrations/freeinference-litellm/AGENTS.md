# FreeInference LiteLLM Integration Kit

## Purpose

This subtree owns the separately usable, local BYOK integration for
FreeInference. It discovers models available to a user-owned key and exposes
loopback-only OpenAI and Anthropic-compatible LiteLLM endpoints.

## Ownership

ClaudeBrigade integration tooling; provider credentials remain owned by the
user and are never committed or sent to Claude Code.

## Local Contracts

- `FREEINFERENCE_API_KEY` is read only by synchronization/proxy processes.
- `LITELLM_MASTER_KEY` authenticates local clients to the loopback proxy.
- FreeInference models use the explicit `openai/<model-id>` namespace upstream.
- `sync` consumes the live `/v1/models` catalog and writes deterministic YAML.
- `auto` endpoint selection uses fresh, token-weighted cache-read evidence;
  explicit endpoint selection and configured defaults take precedence.
- Reports contain compatibility results and aggregate usage only; they must
  not contain prompts, responses, tool arguments, or API keys.

## Work Guidance

Keep the kit independent of ClaudeBrigade’s SQLite state. Use bounded HTTP
timeouts, no automatic inference probes, and no silent model fallback while
compatibility is being measured.

## Verification

```bash
uv run --extra test pytest -q
```

Run live probes only when an operator explicitly requests them.

## Child DOX Index

| Path | Purpose |
|------|---------|
| `config/` | Loopback LiteLLM configuration template |
| `scripts/` | Explicit catalog synchronization and operator diagnostics |
| `src/` | Installable Python package and CLI implementation |
| `tests/` | Offline integration-kit contract tests |
