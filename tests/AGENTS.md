# Verification Suite

## Purpose

This subtree contains the offline tests for routing, registry/configuration,
state/workflows, MCP authorization, hooks, provider admission, LiteLLM,
fastpath, sidecars, launch behavior, and shadow-worktree integration.

## Ownership

Regression evidence for the application. Tests should prove contracts and
failure paths, not merely exercise happy-path syntax.

## Local Contracts

- Tests use isolated temporary SQLite/config fixtures and must not write to
  the user’s runtime state or canonical repository.
- No test should require live provider inference, provider credentials, or
  network availability. Mock HTTP and local fixtures instead.
- Tests that mutate workflow state must include run/epoch/principal scope and,
  for controller-only operations, create the active controller binding.
- New authority, lifecycle, retry, workspace, or completion behavior needs a
  negative test for wrong actor/resource, replay, stale state, or failure.
- Preserve the project import convention: `enhanced_router` is loaded from
  `router/`, and hook modules are loaded from `hooks/` via `conftest.py`.

## Work Guidance

Prefer one focused test module per subsystem and deterministic fake clocks or
HTTP transports where timing matters. Avoid broad fixture changes that hide a
missing authorization or lifecycle precondition. Split long-running/hanging
integration groups when diagnosing failures.

## Verification

```bash
pytest -q tests/test_router.py tests/test_registry.py tests/test_state.py
pytest -q tests/test_mcp_control.py tests/test_mcp_transport.py
pytest -q tests/test_provider_admission.py tests/test_litellm_supervisor.py
pytest -q tests/test_ledger_and_guard.py tests/test_fingerprint.py
git diff --check
```

## Child DOX Index

No child directories. Test modules are grouped by subsystem in their names.
