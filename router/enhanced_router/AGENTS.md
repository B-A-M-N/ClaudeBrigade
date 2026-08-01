# Enhanced Router Runtime

## Purpose

This package is the authoritative runtime control plane: FastAPI transport,
registry-backed route resolution, SQLite workflow state, provider admission,
LiteLLM supervision, MCP control, native lifecycle correlation, sidecars, and
shadow-worktree integration.

## Ownership

The router runtime owns request admission, route and execution authority, and
durable orchestration state. Claude Code owns the native tool loop and native
Agent process lifecycle; hooks report and guard that lifecycle.

## Local Contracts

- Read `ARCHITECTURE.md` before changing routing, provider admission,
  scheduling, endpoint selection, workspace integration, or completion state.
- Route bindings are immutable within an epoch. Never re-resolve an existing
  binding from current YAML.
- Mutating native actions require a claimed action, attached child identity,
  owned active shadow workspace, and mutation lease. The canonical checkout is
  changed only by the integration path.
- MCP mutations require an authenticated principal, run/epoch/resource scope,
  and the capability appropriate to the operation. A controller binding alone
  is not a substitute for an action claim where one is required.
- Native, sidecar, and controller actions share persisted execution evidence;
  only native actions are represented as Claude Code native agents.
- Model-qualified agent names and public aliases come from the registry
  manifest. Do not add parallel hardcoded identity maps.
- `state.py` remains a large transactional boundary for now. Keep changes
  localized and preserve existing transaction/invariant tests; do not perform a
  broad state/repository refactor in this workstream without explicit approval.

## Work Guidance

Prefer a narrow repository method or service seam over duplicating SQL at call
sites. Use typed transition/status validation, exact run and epoch scoping,
fail-closed behavior for authority or workspace uncertainty, and persisted
events for lifecycle changes. Route all provider calls through the common
admission/deadline/retry/usage path. Avoid live provider calls in tests.

When adding a model or specialist identity, update the relevant YAML schema,
registry validation, launch manifest projection, and focused tests together.

## Verification

```bash
pyright router/enhanced_router/ hooks/
pytest -q tests/test_state.py tests/test_mcp_control.py tests/test_mcp_transport.py
pytest -q tests/test_router.py tests/test_registry.py tests/test_provider_admission.py
pytest -q tests/test_sidecar_executor.py tests/test_litellm_supervisor.py
git diff --check
```

## Child DOX Index

No child directories. Module ownership is summarized in the root guide and
the architecture handoff.
