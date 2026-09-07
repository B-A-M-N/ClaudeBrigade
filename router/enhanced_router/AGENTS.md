# Enhanced Router Runtime

## Purpose

This package is the authoritative runtime control plane: FastAPI transport,
registry-backed route resolution, SQLite workflow state, provider admission,
LiteLLM supervision, MCP control, native lifecycle correlation, native
sidecar workers, coprocessors, and shadow-worktree integration.

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
- Controller planning phases are scheduler actions, not model-backed worker
  routes. They must remain discoverable before a controller binding exists;
  the claim transaction still enforces the immutable run resource policy.
- Native role, native sidecar, coprocessor, and controller actions share
  persisted execution evidence. Native sidecars are represented as Claude
  Code native agents; only coprocessors use the bounded router-owned call
  executor.
- Router-owned fastpath route/verify jobs are persisted `coprocessor_call`
  executions. Legacy `sidecar_call` rows remain readable during migration;
  keep their advisory authority boundary intact.
- Agent names and public aliases come from the registry manifest and remain
  model-neutral; backing model/provider/endpoint identity lives in the bound
  route snapshot. Do not add parallel hardcoded identity maps.
- Native sidecar definitions are independently selected from `sidecars.yaml`;
  their model/provider route, tools, permissions, and worktree policy must not
  be inferred from a native role profile when a phase names a sidecar.
- Native model slots and durable role lanes are additive. A slot projection
  must not delete or rename the stable role aliases.
- Every native sidecar public model alias is individually routable, but the
  first request must match its claimed native worker/action identity. Route
  it through normal provider request admission; never bypass provider limits.
- Saved role/controller inference profiles may include ordered fallbacks. A
  fallback is a new route candidate, not permission to mutate an existing
  immutable binding.
- Provider API keys are read inside the router from the OS credential store;
  the owner-only `providers.env` file is a compatibility fallback. Do not
  reintroduce provider-key exports into the Claude Code launcher environment.
- Explicit catalog refreshes may persist non-secret model metadata in
  `discovered_models.yaml`. Discovered entries are sidecar/read-only by
  default and must not gain native role or certification authority implicitly.
- The configuration wizard uses target-specific model choices and concrete
  `(provider_id, model_id, endpoint_id)` route identity across controller,
  roles, fallbacks, fastpath, and sidecars. Certification evidence must remain
  scoped to that route, target role, and probe protocol.
- Provider discovery receives an exact catalog URL. Only the registry fallback
  from an inference endpoint base may append `/models`; never append it to an
  explicit discovery URL.
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
