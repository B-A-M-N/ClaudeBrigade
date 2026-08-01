# ClaudeBrigade continuation handoff

This is a short resume point for future work. The longer design contract is
in [ARCHITECTURE.md](ARCHITECTURE.md); the retained audit work is tracked in
[P0_IMPLEMENTATION_PLAN.md](P0_IMPLEMENTATION_PLAN.md).

## Product decisions that must not regress

- The active model in Claude Code's main thread is the controller. Controller
  is a runtime role, not a Sonnet, Claude, DeepSeek, or vendor identity.
- FreeInference is a local user-owned BYOK integration. The provider key stays
  in the router/LiteLLM process; Claude Code receives only a local router key.
- FreeInference's operator-facing concurrency default is **4**. It applies to
  the controller as well as visible workers and to both direct Anthropic-format
  and LiteLLM/OpenAI-format requests. It is adjustable with
  `FREEINFERENCE_MAX_CONCURRENCY`.
- FreeInference context and output limits are not invented in bundled YAML.
  `/v1/models` discovery may supply them; absent provider metadata remains
  unknown.
- A logical model may have fixed endpoint routing or an explicitly configured
  managed LiteLLM deployment group. Fixed routes pin an endpoint. Managed
  groups pin the logical model, approved deployments, generation, and policy;
  LiteLLM may select an equivalent provider deployment per request.
- For FreeInference, an automatic endpoint choice ranks only fresh, certified,
  healthy endpoints and then uses token-weighted cache-read rate. Explicit
  endpoint configuration wins. Existing fixed bindings never silently move.
- Every substantive specialist must be a visible native Claude Code Agent.
  The router cannot replay a denied native Agent call, so capacity planning is
  cooperative: the controller asks `get_runnable_actions`, claims the exact
  returned action with `claim_runnable_action`, spawns only that action, and
  asks again after terminal lifecycle events. Unclaimed native Agent calls
  are denied because Claude Code cannot replay them later.
- MCP is intentional. It is the local authenticated control surface for model
  discovery, profiles, routes, task phases, runnable actions, and status. It is
  not the specialist execution engine and must not become hidden fan-out.
- DiffusionGemma is an internal read-only advisory sidecar for route/verify
  packets. It cannot mutate, waive required review, lower a deterministic
  workflow tier, or certify completion.

## Implemented in the current working tree

- Controller bindings and model-agnostic controller resolution.
- FreeInference provider configuration, live catalog discovery, provider
  metadata merge, and adjustable concurrency.
- Cache-aware endpoint selection and capability-specific certification records.
- Fixed and managed-group model schemas plus shared LiteLLM aliases for groups.
- SQLite row-factory repair, v30 configuration-hash migration, v31 managed
  binding fields, corrected execution foreign key migration, and route snapshot
  identity.
- Pinned LiteLLM deployment lookup fused into binding reads.
- Shared external HTTP client and shared response-header filtering.
- Incremental SSE usage parsing and bounded stream/admission cleanup.
- Provider request/agent admission and circuit-state enforcement for routed
  requests, with durable reservation records for native-agent planning. A
  managed group persists all candidate provider IDs and reserves the set
  atomically; this is conservative but preserves FreeInference's max of 4.
- Model-qualified native agents, hook execution records, mutation leases,
  workflow phase snapshots, task intake, fastpath schemas, and MCP status/
  runnable-action operations.
- Git-backed shadow worktrees for native mutators: canonical dirty-baseline
  capture, cross-run checkout protection, persisted binary changesets,
  overlap classification, DiffusionGemma merge-risk advice, deterministic
  `git apply --check` preflight, green integration, and controller-visible
  yellow/red escalation actions. The active main controller is the only
  workflow actor expected to approve yellow candidates or retry/discard red
  candidates; it must claim the exact integration action first. Unresolved
  candidates block completion.
- Local `integrations/freeinference-litellm` package with model sync and a
  provider compatibility-harness starting point.
- Controller-only fastpath proposal disposition: the main controller can
  inspect, accept/reject, and optionally apply logical route recommendations;
  fastpath cannot select a physical endpoint or change state directly.

## Remaining implementation order

1. Complete trusted actual-deployment telemetry for multi-provider LiteLLM
   groups. Admission is now conservative and atomic across all candidate
   providers; later callback integration can release unused candidates earlier.
2. Exercise the cooperative native-Agent assignment path against real Claude
   Code hook payloads, including provider-capacity exhaustion and terminal
   cleanup; claims now prevent denied calls from becoming dead queued work.
3. Enforce every persisted phase contract on production hooks: actor,
   provider requirements, deadlines, turn budgets, result schemas, quorum,
   fallback policy, and mutation lease.
4. Finish end-to-end native Claude hook coverage for workspace/changeset
   isolation and exercise the explicit yellow-candidate controller resolution
   flow. Green integration is implemented and deterministic; semantic conflict
   resolution is not automatic, and red/yellow candidates remain visible as
   claimed controller actions until they are approved, retried, or discarded.
5. Complete MCP authorization and status surfaces for every mutating tool;
   include endpoint, provider, cache, queue, circuit, and actual deployment
   identity in statusline/health output.
6. Finish LiteLLM generation monitoring, active-binding drain reconciliation,
   and readiness behavior.
7. Expand local fixture tests for tool calls, parallel tools, tool-result
   continuation, structured output, cancellation, SSE usage, 401/429/503, and
   generation pinning.
8. Run opt-in FreeInference certification and paired OpenAI-vs-Anthropic
   endpoint benchmarks. Never run live inference probes automatically.
9. Reconcile installer assets, release allowlist, `SHA256SUMS`, and all docs
   only after source/config edits are complete.

## Verification commands

From the repository root:

```bash
pytest -q tests/
ruff check router/enhanced_router hooks tests
pyright router/enhanced_router hooks
python -m py_compile router/enhanced_router/*.py hooks/*.py
PYTHONPATH=integrations/freeinference-litellm/src pytest -q integrations/freeinference-litellm/tests
git diff --check HEAD
```

The current root suite reaches 411 passed and 1 skipped. The integrity
manifest is an explicit release allowlist and must be refreshed after final
source/config edits.
