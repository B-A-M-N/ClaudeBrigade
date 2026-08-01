# ClaudeBrigade — Agent Working Guide

> Current architecture handoff: read [ARCHITECTURE.md](ARCHITECTURE.md)
> before changing routing, provider admission, workflow scheduling, or
> endpoint selection. This guide contains historical command/path details;
> the architecture handoff is authoritative where the two differ.

This document describes the essential knowledge for an agent to work effectively in the ClaudeBrigade repository.

---

## Project Overview

**ClaudeBrigade** is a loopback model router for Claude Code. It runs a FastAPI server on `127.0.0.1:8787` that Claude Code connects to. The router:

- Routes the active main-thread model through a registry-backed, immutable controller binding; Anthropic passthrough is explicit configuration
- Routes **role aliases** (`anthropic-brigade-recon`, `anthropic-brigade-implementer`, `anthropic-brigade-adversary`, `anthropic-brigade-repairer`) through a controlled pipeline:
  - Direct-Anthropic backends (e.g., LongCat at `api.longcat.chat/anthropic`)
  - LiteLLM proxy backends (local or remote models via Ollama, OpenRouter, etc.)
- Maintains **immutable agent bindings** per epoch via SQLite state
- Runs an MCP control server for management tools
- Includes `integrations/freeinference-litellm/`, a reusable local BYOK LiteLLM kit that discovers the user's FreeInference models
- Gives mutating native agents isolated Git shadow worktrees and integrates
  only validated changesets back into the canonical checkout

**Key architectural components:**
- `router/enhanced_router/` — FastAPI app, routing logic, state, registry, backends, LiteLLM supervisor, MCP control
- `config/` — YAML configs for models, profiles, workflows, providers,
  sidecars, fastpath, and generated discovered model catalogs
- `hooks/` — Claude Code hooks for guardrails, audit, session lifecycle, completion
- `agents/` — Agent definition files (injected via `--agents` flag at launch)
- `bin/` — Launcher scripts, including `claude-brigade-config` for credentials,
  catalogs, saved sidecars, and inference profiles
- `tests/` — pytest coverage for routing, state, hooks, provider admission,
  fastpath, LiteLLM, and shadow-worktree integration (run the suite for the
  current count)

---

## Essential Commands

### Development

```bash
# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_router.py -v          # specific module
python -m pytest tests/ -k "test_resolve" -v      # pattern match

# Run with coverage
python -m pytest tests/ --cov=enhanced_router --cov-report=term-missing

# Type checking (pyright)
pyright router/enhanced_router/

# Linting (ruff)
ruff check router/enhanced_router/
ruff format router/enhanced_router/
```

### Installation / Deployment

```bash
# From source directory
./install.sh

# Installs to:
#   ~/.local/share/claude-brigade/router/     (application + venv)
#   ~/.claude-brigade/                        (profile: settings, agents, hooks)
#   ~/.config/claude-brigade/                 (models.yaml, profiles.yaml, tokens)
#   ~/.cache/claude-brigade/                  (logs, PID, MCP config)
#   ~/.local/state/claude-brigade/            (SQLite state.db)
#   ~/.local/bin/                             (claude-brigade, claude-brigade-doctor, ...)
```

### Runtime Diagnostics

```bash
claude-brigade-doctor                    # health check
claude-brigade-config                     # interactive credentials/catalogs/profiles/sidecars
claude-brigade-router-stop               # stop background router
cat ~/.cache/claude-brigade/router.log   # router logs (rotated at 10MB, 5 backups)
CLAUDE_CONFIG_DIR=~/.claude-brigade claude doctor
CLAUDE_CONFIG_DIR=~/.claude-brigade claude auth status --text
```

### Configuration Files (User-Editable)

```bash
# Models, profiles, workflows
~/.config/claude-brigade/models.yaml
~/.config/claude-brigade/profiles.yaml
~/.config/claude-brigade/workflows.yaml
~/.config/claude-brigade/providers.yaml     # provider limits and deadlines
~/.config/claude-brigade/sidecars.yaml      # independent sidecar policies
~/.config/claude-brigade/discovered_models.yaml # generated catalog metadata

# Provider keys (LONGCAT_API_KEY, OPENROUTER_API_KEY, etc.)
# Preferred: OS credential store managed by claude-brigade-config.
~/.config/claude-brigade/providers.env      # legacy fallback, mode 0600

# Secrets (auto-generated, mode 0600)
~/.config/claude-brigade/router.token
~/.config/claude-brigade/litellm.token
```

---

## Code Organization

### Router Package (`router/enhanced_router/`)

| File | Responsibility |
|------|----------------|
| `app.py` | FastAPI app, lifespan, `/v1/messages`, `/v1/models`, `/healthz`, MCP mount |
| `routing.py` | `resolve_request()` — role alias → epoch → route → binding → `ResolvedRoute` |
| `state.py` | `RouteState` — schema/migrations, `_new_conn`, and the handful of methods that span more than one repository's tables; mixes in ~20 `*_state.py` repository modules (see below) for everything else |
| `*_state.py` | One `XRepository` mixin per bounded concern (e.g. `workflow_phase_state.py`, `agent_execution_state.py`, `mutation_lease_state.py`, `shadow_workspace_state.py`, `provider_reservation_state.py`, `run_orchestration_state.py`) — each only touches its own tables through `self._new_conn()`; `RouteState` inherits from all of them so cross-repository `self.method()` calls resolve via normal MRO regardless of which mixin defines the method |
| `registry.py` | `ModelRegistry` — loads YAML, validates cross-refs, deterministic `recommend()` |
| `config_models.py` | Pydantic models: `ModelSpec`, `ProfileSpec`, `ModelCapabilities`, etc. |
| `backends.py` | Proxy implementations: `proxy_anthropic_passthrough`, `proxy_direct_anthropic`, `proxy_litellm_messages` |
| `base.py` | Path constants, compatibility role sets, registry-backed agent authorization helpers, and hop-by-hop headers |
| `litellm_config.py` | Generates LiteLLM YAML config from registry; atomic write + digest |
| `litellm_supervisor.py` | `LiteLLMSupervisor` — blue-green child process lifecycle (generations, deployments) |
| `mcp_control.py` | Authenticated local control surface for models, profiles, routes, task phases, runnable actions, fastpath proposal disposition, and status |
| `mcp_transport.py` | Auth wrapper for MCP StreamableHTTP |
| `policy.py` | Heuristics for tier classification (trivial/normal/cross-cutting/high-risk) |
| `agents_json.py` | Validates and generates the native `--agents` manifest, including registry-qualified specialists |

### Configuration (`config/`)

- **`models.yaml`** — Model definitions: `display_name`, `backend` (`direct-anthropic` | `litellm`), `upstream_model`/`litellm_model`, `api_base`/`api_base_env`/`api_key_env`, `capabilities`, `allowed_roles`, `enabled`
- **`providers.yaml`** — Provider-wide admission limits; FreeInference defaults to `max_concurrency: 4` across controller, agents, and transports
- **`fastpath.yaml`** — Optional DiffusionGemma route/verify sidecar; advisory and read-only, never an authority for endpoints, merges, or completion
- **`sidecars.yaml`** — Independent bounded sidecar model/provider policies
- **`discovered_models.yaml`** — Generated non-secret provider catalog metadata
- **`profiles.yaml`** — Maps 4 roles (`recon`, `implementer`, `adversary`, `repairer`) to model IDs
- **`workflows.yaml`** — Maps workflow names (`normal`, `cross-cutting`, `high-risk`, `trivial`) to `default_profile`

### Hooks (`hooks/`)

| Hook | Purpose |
|------|---------|
| `guard_tool.py` | PreToolUse: blocks unauthorized subagents, read-only roles from Write/Edit/Bash-mutation, records ledger |
| `audit_tool.py` | PostToolUse: fingerprints workspace, logs Mutation events, test execution, AgentResult |
| `completion_guard.py` | Stop: enforces completion evidence (recon + adversary for cross-cutting, 2× adversary for high-risk) |
| `session_start.py` | SessionStart: creates/resumes a run, registers the canonical workspace, and writes session metadata; task epochs begin after prompt intake |
| `session_end.py` | SessionEnd: closes run, cleanup |
| `statusline.py` | StatusLine: shows run/epoch/agent context |
| `workspace_fingerprint.py` | SHA-256 of tracked + untracked files (for mutation detection) |

### Agent Definitions (`agents/`)

Injected at launch via `--agents` flag (outranks project-level files):

- `brigade-recon.md` — read-only repository investigation
- `brigade-implementer.md` — default source mutation
- `brigade-adversary.md` — independent implementation reviewer
- `brigade-repairer.md` — fixes accepted findings
- `controller-direct.md` — exceptional trivial edits by controller
- `controller-append.md` — appended to controller system prompt

Mutating definitions request native Claude Code worktree isolation. The
`audit_agent.py` lifecycle hook records the actual child worktree, extracts a
persisted changeset on stop, and removes the child worktree. Green changesets
pass deterministic `git apply --check` before application; yellow candidates
  become explicit `controller_integration` actions returned by MCP, and red
  candidates block completion until the main controller resolves them. No
  hook commits, stashes, resets, or silently merges conflicts. The main
  controller must claim an integration action before approving yellow or
  resolving red; a controller binding alone is not authorization.

---

## Architecture & Data Flow

### Current design rules

- The active Claude Code main model is the controller. Controller is a runtime
  role and may be FreeInference; it is not a hardcoded Claude/vendor identity.
- FreeInference is local BYOK. The provider key stays in the router process.
- `FREEINFERENCE_MAX_CONCURRENCY` defaults to `4` and applies to controller,
  visible agents, direct provider streams, and LiteLLM provider traffic.
- FreeInference endpoint `auto` selection chooses the freshest eligible,
  token-weighted cache winner after certification. Context limits are not
  bundled; provider catalog metadata is authoritative when supplied.
- Fixed routes pin one endpoint. Explicit `managed-group` routes pin a logical
  LiteLLM deployment group and policy while LiteLLM chooses an equivalent
  physical deployment.
- The router cannot replay a denied native `Agent` call. The controller must
  call MCP `get_runnable_actions`, claim the exact action with
  `claim_runnable_action`, spawn only that admitted action, and call it again
  after terminal lifecycle events. The PreToolUse hook consumes claims once;
  unclaimed Agent calls are denied.
- MCP is the authenticated local control surface for model/profile/route and
  workflow operations. It does not perform hidden specialist fan-out.
- DiffusionGemma route proposals must be accepted or rejected by the active
  main controller through MCP. Yellow/red shadow candidates likewise require
  the active controller binding and an exact claimed integration action.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete handoff and remaining
implementation checklist.

### Request Flow (Role Alias)

```
Claude Code POST /v1/messages
  → app.py:messages()
  → resolve_request(public_model, run_id, client_agent_id)
      1. ROLE_MODEL_ALIASES maps alias → role
      2. Requires run_id + client_agent_id (409 if missing)
      3. Get active epoch for run (409 if none)
      4. Check existing agent_binding (immutable once created)
         → If exists: reconstruct route from PINNED fields (backend, model_id, endpoint/group policy, catalog_generation, etc.)
         → If not: get role_route for epoch → ModelSpec from registry → validate → bind atomically
      5. Return ResolvedRoute(kind, role, model_id, upstream_model, api_base, litellm_base_url, ...)
  → Dispatch to backend:
      - ANTHROPIC_PASSTHROUGH → proxy_anthropic_passthrough() → explicitly configured Anthropic model
      - DIRECT_ANTHROPIC → proxy_direct_anthropic() → upstream_model (e.g., LongCat)
      - LITELLM → proxy_litellm_messages() → active LiteLLM child (fixed alias or managed group alias)
```

### Immutable Binding Guarantee

Once an agent is bound to a route in an epoch:
- Fixed binding pins: `backend`, `model_id`, endpoint, `upstream_model`, `api_base`, `registry_hash`, `catalog_generation`, and `litellm_model_name`
- Managed-group binding pins: logical model, approved deployment set, group policy digest, registry hash, LiteLLM generation, and group alias; the actual provider deployment is recorded per request
- **Registry changes do not affect bound agents** — they continue on the pinned deployment
- New agents spawned after a route change get the new route
- Epoch closure releases all bindings

### LiteLLM Supervisor (Blue-Green)

```
Registry change → reload_catalog()
  1. Create new generation (staging) with new config digest
  2. Start LiteLLM child on free port
  3. Health probe /health
  4. Activate generation (old → draining)
  5. Old child drained when all its bindings released → killed
```

### State Machine (SQLite)

Tables: `runs`, `epochs`, `role_routes`, `agent_bindings`, `route_events`, `model_health`, `litellm_generations`, `litellm_deployments`

Key invariants (enforced by unique indexes):
- One active epoch per run (`uq_active_epoch`)
- One active binding per agent per run (`uq_active_agent_binding`)

---

## Naming Conventions & Patterns

### Model IDs (registry keys)
- Lowercase with hyphens: `longcat-2`, `qwen-local`, `glm-review`
- Used in `models.yaml`, `profiles.yaml`, routes

### Role Aliases (public-facing)
- `anthropic-brigade-{recon,implementer,adversary,repairer}`
- Stable aliases are exposed by the registry manifest; model-qualified aliases
  are declared in profile specialist metadata.

### Agent Types (internal)
- `brigade-recon`, `brigade-implementer`, `brigade-adversary`, `brigade-repairer`, `controller-direct`
- Static compatibility names are supplemented by registry-backed manifest
  authorization through `base.py` helpers.

### Epoch IDs
- Format: `ep_{profile}_{N}` (e.g., `ep_hybrid_001`)
- Generated by `session_start` hook

### Run IDs
- UUID v4, created by launcher

### Environment Variables (Launcher)
| Variable | Purpose |
|----------|---------|
| `CLAUDE_BRIGADE_RUN_ID` | Passed to hooks for session correlation |
| `ENHANCED_ROUTER_TOKEN` | Router auth (from `router.token`) |
| `BRIGADE_LITELLM_KEY` | LiteLLM internal auth (from `litellm.token`) |
| `LONGCAT_API_KEY` / `OPENROUTER_API_KEY` / etc. | Router-only provider keys; preferred source is the OS credential store, with `providers.env` as fallback |
| `ANTHROPIC_UPSTREAM` | Override Anthropic base URL |
| `LONGCAT_UPSTREAM` | Override LongCat base URL |

---

## Testing Approach

- **Isolated fixtures**: Each test gets a fresh `tmp_path` with patched `DEFAULT_DB_PATH` and registry singleton
- **No external dependencies**: All tests use mocks or local SQLite; no network calls
- **Test categories**:
  - `test_router.py` — payload normalization, endpoints
  - `test_alias_routing.py` — role alias resolution, binding immutability
  - `test_e2e.py` — full request path with state + registry
  - `test_state.py` — SQLite CRUD, migrations, invariants
  - `test_registry.py` — YAML loading, cross-ref validation, recommendation
  - `test_litellm_supervisor.py` — process lifecycle, health probes
  - `test_litellm_config.py` — config generation, digests, atomic write
  - `test_litellm_proxy.py` — dispatch, header sanitization
  - `test_ledger_and_guard.py` — hook behaviors, mutation detection
  - `test_launcher.py` — port handling, provider key isolation
  - `test_config_models.py` — Pydantic validation
  - `test_agents_json.py` — agent bundle rendering

### Running a Single Test with Debug

```bash
python -m pytest tests/test_e2e.py::test_resolve_role_alias_creates_binding -v -s
```

---

## Important Gotchas

### 1. Package Import Path
The package is `enhanced_router` but lives under `router/enhanced_router/`. Tests and hooks add `router/` and `hooks/` to `sys.path` via `conftest.py` and `sys.path.insert(0, ...)`.

### 2. Singleton State & Registry
`RouteState` and `ModelRegistry` are module-level singletons (`_state`, `_registry_instance`). Tests patch them via:
```python
import enhanced_router.state as state_module
state_module._state = None  # forces re-init with patched DB_PATH
```

### 3. Immutable Bindings Are Not Re-Resolved
When debugging routing: if an agent already has a binding, `resolve_request` **never re-reads the registry**. It reconstructs from the pinned binding fields. This is by design.

### 4. LiteLLM Deployment Pinning
Bindings store `catalog_generation`. If that generation is fully drained (no deployments), the binding becomes unusable (503). This is intentional — agents must not silently switch deployments.

### 5. Header Stripping
Internal headers (`x-enhanced-token`, `x-brigade-run-id`) are stripped in `sanitize_upstream_headers()`. **Do not rely on them reaching upstream.**

### 6. Role Alias vs Controller Model
The four `anthropic-brigade-*` aliases are role-routed. Main-thread model IDs are resolved through the registry and pinned as controller bindings; only models explicitly configured as `anthropic-passthrough` reach Anthropic.

### 7. Mutation Detection in Hooks
Read-only Bash authorization uses a narrow observational allowlist; regex
classification is diagnostic only. The `audit_tool.py` pre/post fingerprint
diff remains the mutation evidence, and fingerprinting must recalculate the
mutable workspace rather than cache by epoch.

### 8. Shadow-worktree integration

The canonical checkout is registered once per run. Cross-run ownership of the
same checkout is rejected. Worker patches are computed from the captured dirty
baseline to the worker state, so existing user edits are preserved. Path
overlap is conservative: same-file overlap is yellow even when a textual
merge might be possible. A failed `git apply --check` is a conflict and is
never force-applied.

### 8. Profile References
`ProfileSpec` validates that all four roles reference **existing model IDs** from `models.yaml`. Using a role name as a model ID (e.g., `recon: "recon"`) is rejected.

### 9. `direct-anthropic` Backend Requirements
- Must have `upstream_model` (e.g., `LongCat-2.0`)
- Must have `api_key_env` set in environment (checked at resolve time)

### 10. `litellm` Backend Requirements
- Must have `litellm_model` (e.g., `ollama/qwen3-coder-next`)
- `api_base` optional (defaults to `http://127.0.0.1:11434` for Ollama)
- `api_key_env` optional (for OpenRouter, etc.)

---

## Common Tasks

### Add a New Model
1. Edit `config/models.yaml` (source) and/or `~/.config/claude-brigade/models.yaml` (runtime)
2. Add entry with all required fields
3. Run `claude-brigade-router-stop` then restart to pick up changes (or use MCP `reload_catalog`)

### Add a New Profile
1. Edit `config/profiles.yaml` / `~/.config/claude-brigade/profiles.yaml`
2. Reference existing model IDs for all four roles
3. Validate: `python -c "from enhanced_router.registry import ModelRegistry; r=ModelRegistry(); r.load_models(); r.load_profiles(); r._validate_cross_refs(); print('OK')"`

### Change Role Route at Runtime
Use MCP tool `set_role_route`:
```json
{
  "run_id": "...",
  "epoch_id": "...",
  "role": "implementer",
  "model_id": "qwen-local",
  "source": "manual",
  "reason": "switch to local model"
}
```

### Debug Routing for a Run
```bash
# Via MCP
get_route_status { "run_id": "...", "epoch_id": "..." }

# Or inspect SQLite directly
sqlite3 ~/.local/state/claude-brigade/state.db \
  "SELECT * FROM role_routes WHERE run_id='...';"
sqlite3 ~/.local/state/claude-brigade/state.db \
  "SELECT * FROM agent_bindings WHERE run_id='...';"
```

### Test a Hook Locally
```bash
echo '{"tool_name": "Write", "tool_input": {"file_path": "test.py", "content": "x=1"}, "session_id": "test", "agent_id": "agent-1"}' | python hooks/guard_tool.py
```

---

## Security Primitives (BrokeLLM-Derived)

- **Symlink protection**: `router.token`, `litellm.token`, `providers.env` rejected if symlinks. Writes use `O_NOFOLLOW | O_EXCL`.
- **Restrictive umask**: Launcher sets `umask 077` before creating runtime files.
- **Expanded env key ban list**: All common provider keys (`OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, etc.) unset before Claude Code starts — only available to router process.
- **Launch audit trail**: Every invocation writes JSON event to `$CACHE_DIR/audit.jsonl` (mode 0600).
- **Log rotation**: Router log auto-rotates at 10 MB with 5 backups.

---

## Milestone Tracking (Implementation Status)

All 9 milestones complete (see `bin/plan.md` and `meta/M*.md`):
- M01: Generic role agents & stable aliases
- M02: Model & profile registry
- M03: SQLite route state
- M04: MCP control server
- M05: Gateway role-alias routing
- M06: LiteLLM child service
- M07: Hook & evidence-ledger integration
- M08: Workflow engine
- M09: Branding, migration, documentation, E2E

---

## Files to Check When Debugging

| Issue | Check |
|-------|-------|
| Router won't start | `~/.cache/claude-brigade/router.log`, `claude-brigade-doctor` |
| Role alias 409 | `x-brigade-run-id` / `x-claude-code-agent-id` headers present? Active epoch exists? |
| Binding not updating | Existing binding pins old route — expected behavior |
| LiteLLM 503 | `catalog_generation` in binding has no active deployment |
| Model not in `/v1/models` | Check `models.yaml` → `enabled: true`, backend config valid |
| Hook deny | `guard_tool.py` logs to stderr; check agent_type vs `MUTATORS`/`ALLOWED_SUBAGENTS` |

---

## Key Invariants to Preserve

1. **One active epoch per run** — enforced by `uq_active_epoch` unique index
2. **One active binding per agent per run** — enforced by `uq_active_agent_binding`
3. **Bindings are immutable** — route reconstructed from pinned fields only
4. **Registry cross-refs valid** — profile model IDs must exist in models
5. **Provider keys isolated** — never leaked to Claude Code process
6. **Token files never symlinks** — `O_NOFOLLOW` on read/write
7. **Hop-by-hop headers stripped** — per RFC 9113 §8.2.2
8. **Completion requires evidence** — recon + adversary (cross-cutting), 2× adversary (high-risk)

---

## Version & Dependencies

- **Python**: >=3.10
- **Core deps**: `fastapi`, `httpx`, `mcp`, `PyYAML`, `keyring`, `uvicorn`, `litellm[proxy]==1.93.0`
- **Test deps**: `pytest`, `pytest-asyncio`, `pytest-cov`
- **Package**: `enhanced_router` (installed editable in venv)
- **Entry point**: `claude-brigade` (bash launcher)

## Child DOX Index

| Path | Purpose |
|------|---------|
| `router/` | Python runtime package and its enhanced router implementation |
| `hooks/` | Claude Code lifecycle, guardrail, audit, and completion hooks |
| `agents/` | Native visible agent definitions injected at launch |
| `config/` | Model, provider, profile, and workflow configuration |
| `bin/` | Launcher, doctor, login, and router lifecycle commands |
| `tests/` | Offline unit, integration, hook, and fixture verification |
| `meta/` | Milestone and implementation handoff records |
| `Targets/` | Provider/target planning notes, not runtime configuration |
| `integrations/` | Separately usable integrations and their local contracts |
