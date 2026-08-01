# P0 Implementation Plan

## Current handoff status (2026-07-31)

The design has been corrected to separate four concerns that older plan text
mixed together:

1. The active Claude Code main model is the runtime controller; no vendor is
   hardcoded as controller.
2. Claude-facing workers are visible native Agent invocations. The router
   cannot replay a denied Agent tool call, so provider capacity is exposed
   through the cooperative MCP `get_runnable_actions` step.
3. Logical model identity is static. Fixed routes pin one certified endpoint;
   explicitly configured managed groups pin a LiteLLM group and allow dynamic
   selection among equivalent deployments.
4. DiffusionGemma is a router-owned advisory sidecar, not a normal worker or
   workflow authority.

The following implementation pieces are present in the current worktree:

- FreeInference BYOK provider loading, live catalog sync, provider-supplied
  context metadata, and adjustable `FREEINFERENCE_MAX_CONCURRENCY` (default 4).
- Controller bindings, endpoint certification/cache selection, binding
  configuration identity, fixed/managed-group schema foundations, and pinned
  LiteLLM deployment lookup.
- Shared upstream HTTP client/header filtering, incremental SSE usage parsing,
  bounded stream deadlines, provider request admission, and circuit state;
  managed groups conservatively reserve all candidate provider slots, including
  FreeInference's shared limit of 4.
- Native visible model-qualified agents, SQLite execution/finding state,
  mutation leases, task intake, phase snapshots, and MCP model/workflow
  controls.

This is not yet a release-complete claim. The remaining implementation order
is maintained below and in `ARCHITECTURE.md`.

## Shadow-worktree integration status

The current worktree now includes the core safe merge path:

- native mutating agents request Claude Code worktree isolation;
- the canonical checkout baseline (HEAD, tracked dirty patch, and untracked
  files) is captured without stash/commit/reset;
- worker changes are persisted as binary changesets relative to that dirty
  baseline;
- same-file overlap is conservatively classified yellow, invalid changesets
  red, and disjoint valid changes green;
- green candidates pass `git apply --check` against the unchanged canonical
  baseline before application;
- DiffusionGemma may advise escalation but cannot authorize filesystem changes;
- yellow/red candidates are returned as `controller_integration` actions to the
  active main-thread controller, and completion remains blocked until the
  controller approves, retries, resolves, or discards them.

The remaining shadow work is end-to-end native Claude hook coverage and
controller-driven semantic conflict-resolution exercise. The controller must
claim the returned integration action before approving yellow or resolving
red; the router does not pretend to perform an automatic three-way merge.

Fastpath route recommendations follow the same authority boundary: the main
controller may inspect, accept/reject, and optionally apply logical model
routes through authenticated MCP operations. DiffusionGemma cannot choose a
physical endpoint, authorize a merge, or change workflow state directly.

## Dependency Order

```
P0-3 (env parser) → P0-1 (endpoint pinning) → P0-2 (auth contract)
                                          ↓
P0-4 (workflow engine) → P0-5 (phase snapshots) → P0-6 (workflow config)
                                          ↓
P0-7 (findings lifecycle) ← P0-5
                                          ↓
P0-8 (litellm crash fix)  (independent, can parallelize)
```

---

## P0-1: Endpoint Pinning & Credential Isolation

**Objective:** Prevent LongCat credentials from ever reaching Anthropic endpoint.

**Files:**
- `config/models.yaml` — pin LongCat to its Anthropic-compatible endpoint
- `router/enhanced_router/routing.py:_resolve_api_base` — require explicit endpoint, fail-closed
- `router/enhanced_router/backends.py:proxy_direct_anthropic` — remove Anthropic fallback, validate pinned host
- `.env.example` — add `LONGCAT_API_BASE` requirement
- `README.md` — update setup instructions

**Required Behavior:**
- `direct-anthropic` models MUST have explicit `api_base` or `api_base_env` that resolves at startup
- `proxy_direct_anthropic()` returns `503 provider_endpoint_unavailable` before any outbound request if endpoint missing
- Endpoint validation uses `urllib.parse.urlsplit()` comparing scheme+host+port against pinned value
- LongCat credential NEVER sent to non-pinned hostname
- Redirect responses not followed with provider credentials

**Invariants:**
- Zero outbound requests without validated pinned endpoint
- Credential never appears in headers to unpinned host
- Startup validation fails fast if LongCat endpoint missing

**Acceptance Tests:**
1. Default config resolves LongCat to pinned endpoint, not anthropic.com
2. Missing `LONGCAT_API_BASE` → startup validation error, no request sent
3. Mock server on wrong host receives zero requests with LongCat key
4. Redirect response with LongCat key not followed

**Prohibited:**
- Fallback to `ANTHROPIC_UPSTREAM` or `https://api.anthropic.com`
- String-replace host comparison
- Deferred validation until first request

**Depends on:** P0-3 (env parser must load `LONGCAT_API_BASE` reliably)

---

## P0-2: Provider Auth Contract

**Objective:** Explicit, validated auth spec per model; no inference from backend type.

**Files:**
- `router/enhanced_router/config_models.py` — extend `ModelSpec` with `auth` field
- `router/enhanced_router/backends.py:resolve_provider_auth` — use model's auth spec only
- `config/models.yaml` — add `auth` to each model

**ModelSpec Extension:**
```yaml
auth:
  type: bearer | x-api-key | custom-header | none
  header: "Authorization" | "x-api-key" | "x-custom-header"
  prefix: "Bearer " | "" | "Custom "
  env: "ENV_VAR_NAME"  # required unless type: none
```

**Required Behavior:**
- Backend uses ONLY the model's declared auth spec
- Startup validates: env var exists, type/header/prefix consistency
- Exactly one auth header emitted per request
- `direct-anthropic` backend does NOT infer `x-api-key` from backend type

**Invariants:**
- Zero requests with inferred auth
- Zero requests with multiple auth headers
- Zero requests with missing required credentials
- OAuth credentials never leak to provider

**Acceptance Tests:**
1. Mock provider receives exactly one expected auth header per model
2. Missing auth env → startup validation error
3. `type: bearer` + `header: x-api-key` rejected at validation
4. OAuth token from Claude never appears in outbound request

**Prohibited:**
- Backend-type-based auth inference
- Dual-header emission
- Optional env var for required auth

**Depends on:** P0-1 (endpoint pinning must work first)

---

## P0-3: Hardened Env Parser Integration

**Objective:** Single source of truth for env loading; launcher uses hardened parser.

**Files:**
- `router/enhanced_router/env_parser.py` — existing hardened parser (keep)
- `bin/claude-brigade` — replace bash parser with Python bootstrap
- `README.md` — update examples

**Required Behavior:**
- Launcher invokes `python -m router.enhanced_router.env_parser` (or bootstrap module)
- Parser emits NUL-delimited `KEY=VALUE` or JSON to stdout
- Launcher consumes without `eval` or `source`
- All parser validations apply: permissions, ownership, quotes, comments, null bytes, control chars

**Invariants:**
- Single parser codebase
- Zero `eval`/`source` of env file
- Parser validation failures block launch

**Acceptance Tests:**
1. `LONGCAT_API_KEY='quoted_value'` → exports `quoted_value` (no quotes)
2. `KEY=value #comment` → exports `value` (no comment)
3. `KEY=` → exports empty string
4. World-readable `.env` → launch blocked
5. Symlink `.env` → launch blocked
6. Null byte in value → launch blocked
7. Duplicate keys → last wins, logged
8. All README examples parse correctly

**Prohibited:**
- Maintaining two parsers
- Bash `while IFS='=' read` loop
- `export "$key=$value"` from unparsed input

**Depended on by:** P0-1, P0-2 (both need reliable env loading)

---

## P0-4: Authoritative Workflow Engine

**Objective:** Workflow selection and phase enforcement driven by state machine, not assistant text.

**Files:**
- `hooks/session_start.py` — call `begin_task` MCP op instead of creating epoch directly
- `hooks/user_prompt_submit.py` — call `begin_task` if no active task
- `hooks/completion_guard.py` — call `validate_completion` MCP op, reject on mismatch
- `router/enhanced_router/policy.py` — `classify_scope()` becomes authoritative classifier
- `router/enhanced_router/mcp_control.py` — add MCP operations:
  - `begin_task(epoch_id, prompt, workspace_fingerprint) → {workflow_id, phases, first_actions}`
  - `get_task_state(epoch_id) → TaskState`
  - `start_phase(epoch_id, phase_id, agent_id) → PhaseInstance`
  - `complete_phase(epoch_id, phase_id, evidence_json) → void`
  - `skip_conditional_phase(epoch_id, phase_id, reason) → void`
  - `record_finding(epoch_id, phase_id, finding) → finding_id`
  - `adjudicate_finding(finding_id, disposition, reason) → void`
  - `validate_completion(epoch_id, final_diff) → {allowed: bool, required_tier: str, missing: []}`
- `router/enhanced_router/state.py` — implement above operations with transactions

**Required Behavior:**
1. `session_start` → `begin_task` captures baseline fingerprint, classifies scope, selects workflow, persists immutable phase plan
2. Controller receives first permissible actions from `begin_task` response
3. Each phase transition via `start_phase`/`complete_phase` against immutable snapshot
4. `completion_guard` → `validate_completion` classifies final diff, computes minimum required tier, rejects if executed tier < required
5. Footer becomes generated summary; not source of truth

**Invariants:**
- Workflow selected before any mutating action
- Phase transitions only via MCP ops against persisted snapshot
- Completion validated against final diff, not assistant text
- No phase can start before dependencies satisfied
- Required phases cannot be skipped

**Acceptance Tests:**
1. `session_start` with cross-cutting prompt → `begin_task` returns `high-risk` workflow
2. Controller attempts mutating action before `start_phase(implement)` → rejected
3. `complete_phase` without evidence → rejected
4. Skip required phase → rejected
5. Final diff classified lower than actual changes → completion rejected
6. Footer text ignored; state-derived summary matches persisted phases

**Prohibited:**
- Footer-based completion validation
- Natural-language workflow selection
- Phase dependencies enforced only in YAML

**Depends on:** P0-3 (env for config), P0-5 (phase snapshots must exist)

---

## P0-5: Immutable Phase Snapshots

**Objective:** Phase governance persisted at task start; transitions enforced against snapshot.

**Files:**
- `router/enhanced_router/state.py` — new tables, constraints, methods

**New Tables:**
```sql
workflow_phase_instances (
  epoch_id, phase_id, ordinal, required, mutating,
  allowed_roles_json, controller_actor,
  dependencies_json, condition_json, parallel_group,
  specification_hash, status, started_at, completed_at,
  evidence_json, agent_id,
  PRIMARY KEY (epoch_id, phase_id),
  FOREIGN KEY (epoch_id) REFERENCES epochs(epoch_id)
)

-- Constraints:
-- UNIQUE(epoch_id, phase_id)
-- CHECK: mutating phases not concurrent (application-enforced)
-- CHECK: required phases status != 'skipped'
-- CHECK: completed phases have non-null evidence_json
-- CHECK: dependencies satisfied before status='running'
```

**Methods:**
- `create_phase_snapshot(epoch_id, workflow_def) → void` — called by `begin_task`
- `start_phase(epoch_id, phase_id, agent_id) → void` — validates deps, status, mutating exclusion
- `complete_phase(epoch_id, phase_id, evidence_json) → void` — validates evidence, sets completed_at
- `skip_conditional_phase(epoch_id, phase_id, reason) → void` — validates condition false, not required
- `get_phase_snapshot(epoch_id, phase_id) → PhaseInstance`
- `validate_workflow_definition(workflow_def) → void` — called at registry load

**Workflow Definition Validation (at registry load):**
- Unique phase IDs
- All dependencies exist
- Acyclic dependency graph
- Valid roles exist in agent registry
- Valid conditions reference epoch fields
- Valid parallel groups
- Exactly one controller actor where required

**Invariants:**
- Phase semantics immutable after `begin_task`
- DB constraints prevent invalid transitions
- Workflow def validated once at startup

**Acceptance Tests:**
1. Snapshot created with all semantic fields at `begin_task`
2. `start_phase` blocks if dependency incomplete
3. `start_phase` blocks if another mutating phase running
4. `complete_phase` without evidence → constraint violation
5. Skip required phase → constraint violation
6. Invalid workflow def at startup → load failure

**Prohibited:**
- Caller-supplied phase spec at `start_phase`
- Missing `required`/`mutating` in snapshot
- Runtime-only dependency checks

**Depended on by:** P0-4, P0-6, P0-7

---

## P0-6: High-Risk Workflow Correction

**Objective:** Workflow config matches controller policy and completion guard expectations.

**Files:**
- `config/workflows.yaml` — redefine `high-risk` workflow
- `agents/controller-append.md` — update controller policy to match
- `hooks/completion_guard.py` — remove heuristic inference

**New Workflow Definition:**
```yaml
high-risk:
  phases:
    - id: recon
      roles: [recon]
      required: true
    - id: design-review
      roles: [adversary]
      required: true
      depends_on: [recon]
    - id: implementation
      roles: [implementer]
      required: true
      depends_on: [design-review]
      mutating: true
    - id: implementation-review
      roles: [adversary]
      required: true
      depends_on: [implementation]
      distinct_agent_from: [design-review]
    - id: repair
      roles: [repairer]
      required: false
      depends_on: [implementation-review]
      condition: "has_accepted_findings"
    - id: verification
      roles: [adversary]
      required: true
      depends_on: [repair, implementation-review]
      condition: "always_or_repair_done"
```

**Controller Policy Update:**
- Explicitly references `design-review` and `implementation-review` as separate required adversary phases
- `distinct_agent_from` enforced by `agent_id` check in phase snapshot

**Completion Guard:**
- Removes heuristic "pre/post implementation" inference from ledger
- Uses `validate_completion` MCP op (P0-4)

**Invariants:**
- Two distinct adversary phases required for high-risk
- Distinct agent IDs enforced
- Workflow YAML is single source of truth

**Acceptance Tests:**
1. `high-risk` workflow loads with all phases
2. `design-review` and `implementation-review` have distinct `agent_id` when completed
3. Skip `design-review` → rejected
4. Same agent for both adversary phases → rejected
5. Controller policy references match workflow YAML exactly

**Prohibited:**
- Heuristic phase classification in completion guard
- Controller policy diverging from workflow YAML

**Depends on:** P0-4, P0-5

---

## P0-7: Structured Findings Lifecycle

**Objective:** Durable, reviewable findings with explicit controller adjudication.

**Files:**
- `router/enhanced_router/state.py` — new `findings` table and methods

**New Table:**
```sql
findings (
  finding_id TEXT PRIMARY KEY,
  epoch_id TEXT NOT NULL,
  source_phase_id TEXT NOT NULL,
  source_agent_id TEXT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('critical','high','medium','low','info')),
  category TEXT NOT NULL,
  description TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  disposition TEXT CHECK (disposition IN ('accepted','rejected','duplicate','waived')),
  disposition_reason TEXT,
  accepted_at TIMESTAMP,
  repair_agent_id TEXT,
  resolution_evidence_json TEXT,
  verification_status TEXT CHECK (verification_status IN ('pending','verified','failed')),
  FOREIGN KEY (epoch_id) REFERENCES epochs(epoch_id)
)
```

**Methods:**
- `record_finding(epoch_id, phase_id, agent_id, finding) → finding_id`
- `adjudicate_finding(finding_id, disposition, reason) → void`
- `assign_repair(finding_id, repair_agent_id) → void`
- `resolve_finding(finding_id, resolution_evidence_json) → void`
- `verify_finding(finding_id, status) → void`
- `get_open_findings(epoch_id) → List[Finding]`
- `get_findings_for_repair(epoch_id) → List[Finding]` — only accepted, unverified

**Controller Contract (enforced in completion):**
- Every finding from adversary phase MUST be explicitly adjudicated
- `accepted` findings MUST be repaired and verified before completion
- `rejected` findings with `severity IN ('critical','high')` MUST have `disposition_reason`
- `waived` findings MUST have `disposition_reason`
- Completion rejected if any accepted finding unverified

**Invariants:**
- Findings immutable after creation (only disposition/resolution/verification mutate)
- Adversary cannot modify own findings after phase complete
- Controller cannot complete with open accepted findings

**Acceptance Tests:**
1. Adversary records finding → persisted with all fields
2. Controller adjudicates → disposition + reason persisted
3. Accepted finding without repair → completion rejected
4. Critical rejected finding without reason → completion rejected
5. Repair assigns agent → `repair_agent_id` set
6. Resolution evidence stored → `resolution_evidence_json`
7. Verification updates status → completion allowed only when all verified

**Prohibited:**
- Free-text "Accepted-Findings: resolved" in footer
- Findings only in assistant messages
- Implicit acceptance

**Depends on:** P0-5 (phase snapshot for `source_phase_id`)

---

## P0-8: LiteLLM Crash TypeError Fix

**Objective:** Fix `TypeError` in crash monitor when child exits.

**Files:**
- `router/enhanced_router/state.py` — add `termination_reason` column + method param
- `router/enhanced_router/litellm_supervisor.py` — fix call site, harden monitor

**State.py Changes:**
```sql
ALTER TABLE litellm_deployments ADD COLUMN termination_reason TEXT;
```

```python
def update_litellm_deployment(self, dep_id: str, status: str, pid: int | None = None, termination_reason: str | None = None) -> None:
    # ...
```

**Supervisor Changes:**
- Fix call: `update_litellm_deployment(dep_id, status="failed", pid=pid, termination_reason=...)`
- Wrap state updates in monitor so bookkeeping failure doesn't kill supervisor
- Store monitor task handles for cleanup
- Cancel and await on shutdown
- Expose exit code/reason via health/status endpoint

**Invariants:**
- Supervisor never crashes due to state update failure
- Child exit always recorded with reason
- Clean shutdown cancels monitors

**Acceptance Tests:**
1. Child exits 0 → deployment status=stopped, termination_reason="exit:0"
2. Child exits 137 → status=failed, termination_reason="signal:SIGKILL"
3. State DB error during update → supervisor logs, continues monitoring
4. Supervisor shutdown → monitors cancelled and awaited
5. Health endpoint shows last exit code/reason

**Prohibited:**
- Bare `except:` swallowing errors
- Missing `termination_reason` column
- Monitor task leaks

**Independent:** Can parallelize with other P0 items
