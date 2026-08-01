# Integration Tests

## Purpose

This directory verifies the standalone FreeInference LiteLLM kit’s offline
contracts and deterministic policy behavior.

## Ownership

Integration-kit regression evidence, separate from the main router test suite.

## Local Contracts

- Tests must not require a live provider key, live inference, or external
  network access.
- Assert deterministic config/catalog output, credential isolation, endpoint
  precedence, and report redaction.
- Keep fixtures free of prompts, completions, real credentials, and private
  provider payloads.

## Work Guidance

Prefer local HTTP fakes and pure policy fixtures. Add a test for every new
provider response shape or endpoint-selection rule.

## Verification

```bash
cd integrations/freeinference-litellm && uv run --extra test pytest -q
```

## Child DOX Index

No child directories. `test_contract.py` is the current contract suite.
