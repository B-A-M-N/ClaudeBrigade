# ClaudeBrigade architecture and implementation handoff

This document is the current design contract for ClaudeBrigade. It is written
for the local, user-owned BYOK deployment in this repository. It supersedes
older descriptions that assumed a permanent Sonnet, DeepSeek, or LongCat
controller.

## Product boundary

ClaudeBrigade is a loopback proxy and control plane for Claude Code. The
router runs on `127.0.0.1` and never receives provider infrastructure
credentials from a provider itself. The user supplies provider keys through
the interactive configuration CLI. The preferred store is the OS keyring;
the owner-readable `providers.env` file is a compatibility fallback for
headless installations. The router owns loaded keys and Claude Code gets only
a per-run local router credential.

The controller/native-agent inference profile and sidecar profile are
independent configuration lanes. A named inference profile chooses the
controller and native roles, including ordered model/provider fallbacks. A
named sidecar chooses a bounded read-only model call. Multiple named key slots
may be configured for one provider and rotated round-robin independently of
model/provider fallback. A deployment can therefore change models, providers,
or credentials without conflating those decisions.

The optional `integrations/freeinference-litellm` package is a reusable local
LiteLLM integration kit. It discovers the models available to the user's
FreeInference key, generates a pinned local LiteLLM configuration, and runs
compatibility probes. It is not a FreeInference deployment component.

## Controller invariant

The controller is the model active in Claude Code's main thread. Controller is
a runtime role, not a vendor identity. It may be a Claude model, a
FreeInference model, or another registry model that has current controller
certification.

Main request resolution is:

```text
Claude Code active model
  -> registry model identity
  -> certified route/deployment policy
  -> immutable controller binding for the client session
  -> configured backend
```

Only a model explicitly configured as `anthropic-passthrough` may go to the
Anthropic upstream. A FreeInference controller uses the same local router
credential boundary as every other local client:

```text
Claude Code --local router token--> ClaudeBrigade --FreeInference key--> provider
```

The FreeInference key must not be placed in Claude Code's environment.

## Logical models and deployments

ClaudeBrigade routes logical models. A logical model can expose more than one
certified deployment.

### Fixed route

The router selects one endpoint when an agent or controller is first bound and
pins its endpoint, backend, generation, configuration hash, and certification
evidence. This is the default and is required for non-equivalent protocols,
sidecars, experiments, and provider-specific tool behavior.

### Managed deployment group

An explicitly configured `routing_mode: managed-group` model is bound to a
logical LiteLLM group and an immutable deployment policy. LiteLLM may choose
among the certified equivalent LiteLLM deployments inside that group. The
logical model and approved deployment set remain stable for the agent's
lifetime; the physical deployment may vary per request.

The actual deployment must be recovered from trusted LiteLLM telemetry or
response metadata and recorded against the request. A group must never hide
provider identity from usage, health, concurrency, or audit state.

Sidecars such as DiffusionGemma remain endpoint-bound advisory services. They
do not become Claude Code agents and cannot mutate workflow authority. Bounded
sidecar actions are nevertheless first-class persisted executions: they use
the same provider admission, deadlines, retry policy, result validation, and
execution-event surface as native workers.

## FreeInference endpoint policy

Unless an operator explicitly supplies an endpoint override or configures a
managed group, endpoint selection is:

```text
explicit endpoint override
  -> certification and capability eligibility
  -> provider health and credential availability
  -> fresh token-weighted cache evidence
  -> highest cache-read rate
  -> configured default when evidence is insufficient
```

Cache rate is calculated from tokens, not from an average of request
percentages:

```text
sum(cache_read_tokens) / sum(input_tokens_total)
```

The selected endpoint is pinned for a fixed route. A new observation affects
new bindings or a new epoch; it never silently moves an existing fixed agent.

FreeInference context and output limits are not prefilled in the bundled
configuration. The live `/v1/models` catalog may supply those values during
an explicit catalog sync. Provider metadata is authoritative; missing
metadata remains unknown.

## Provider concurrency and deadlines

FreeInference has one operator-facing concurrency setting:

```text
FREEINFERENCE_MAX_CONCURRENCY=4
```

The bundled provider configuration also defaults to `max_concurrency: 4`.
The setting applies to both native-agent admission and upstream request
admission. It includes a FreeInference controller, direct Anthropic-format
FreeInference requests, and LiteLLM/OpenAI-format FreeInference requests.
The two internal counters remain separate so the implementation can evolve
their policies without changing the operator-facing knob.

Request permits are held from dispatch through the complete response stream,
including cleanup. Cancellation, client disconnect, timeout, malformed SSE,
and upstream errors must all release the permit.

For a managed group spanning more than one provider, the binding persists the
candidate provider set. Because LiteLLM may choose the physical deployment
only after Brigade dispatches the request, admission reserves one slot from
each candidate atomically and releases the set together. This is deliberately
conservative: it cannot let FreeInference exceed its configured limit of four,
although it may temporarily use less aggregate capacity. Trusted LiteLLM
deployment metadata is used afterward for actual-provider accounting when
available.

Provider settings are configurable in `config/providers.yaml` and may be
overridden through the provider-specific environment variable. FreeInference
credentials remain process-local.

The standard Claude subscription passthrough has no registry provider row, so
the router assigns it a synthetic `anthropic` admission lane. It uses the
same request permit, retry/circuit, deadline, stream accounting, and release
path as configured external providers. An explicit `anthropic` provider entry
may override its default concurrency policy.

Provider discovery is catalog-driven. The bundled provider definitions include
OpenRouter plus the configured Kilo/Crush, OpenCode Zen/Go, Cline, NVIDIA NIM,
FreeInference, FreeTheAI, Requesty, and FreeModel routes. The interactive CLI
can refresh compatible `/models` catalogs and stores discovered model metadata
separately from hand-authored registry authority. Search in the CLI filters
the loaded catalog by model ID, description, and provider.

## Execution lanes and native-agent ownership

Every repository-tool or mutation specialist is a visible native Claude Code
Agent. Bounded read-only inference may use a visible Brigade sidecar instead.
The router binds, authenticates, routes, meters, and records both; it does not
secretly fan out to several models and synthesize a hidden answer.

Model-qualified agents identify their logical assignment, for example:

```text
brigade-fi-qwen-scout
brigade-fi-kimi-implementer
brigade-fi-glm-adversary
```

The native lifecycle hooks create and close authoritative SQLite execution
records. JSONL is diagnostic evidence only. The scheduler distinguishes three
execution kinds:

```text
native_agent       visible Claude Code Agent with repository tools
sidecar_call       bounded router-owned structured specialist call
controller_action  controller-only integration/adjudication operation
```

Only the first kind is presented as a native Claude Code agent. Sidecars are
shown explicitly in orchestration status and have no direct filesystem or
workflow-authority capability.

The registry owns model-qualified native identities. Profile specialist
entries may declare a `native_agent_name` and `public_model_alias`; the
launcher projects those entries into the Claude Code `--agents` manifest from
the stable role definition. Routing, scheduling, hooks, `/v1/models`, and
status use the same manifest rather than parallel model-name tables.

## Shadow workspaces and integration

Mutating native agents run in Claude Code's isolated worktrees when the agent
definition requests `isolation: worktree`. `SessionStart` registers the
canonical checkout and captures its `HEAD`, tracked dirty patch, and untracked
baseline without stashing or committing the user's work. `SubagentStart`
records the actual child worktree; `SubagentStop` extracts a persisted binary
changeset and removes the child worktree.

Integration is deliberately conservative:

```text
changeset extraction
  -> path-overlap classification
  -> green: advisory check, git apply --check, then apply
  -> yellow: retain persisted patch and escalate to the main controller
  -> red: retain candidate and escalate recovery to the main controller
```

The final deterministic guard is `git apply --check` against the unchanged
canonical baseline. A failed preflight is never force-applied. Existing user
changes are preserved because worker patches are generated from the canonical
dirty baseline to the worker state, not from `HEAD` to the worker state.
The router never commits, stashes, resets, or silently performs a three-way
merge. DiffusionGemma may advise that a green candidate needs review, but it
cannot authorize an integration or resolve a conflict. Yellow and red
candidates become explicit `controller_integration` actions in
`get_runnable_actions`; the active main-thread controller must approve a
deterministically preflighted yellow patch or resolve/discard/retry a red one
after claiming that exact controller action. A binding proves the operation is
on the main controller path; the action claim proves it was scheduled. Both
are required before completion can proceed.

This is working foundation code, not a claim that arbitrary semantic
conflicts are automatically resolved. Conflict resolution remains an explicit
controller/integration action followed by final verification on the canonical
workspace.

## Cooperative scheduling boundary

ClaudeBrigade cannot replay a denied Claude Code `Agent` tool call. Therefore
the database must not be treated as an executable spawn queue.

The controller asks MCP for `get_runnable_actions`, receives only actions that
are ready under the phase DAG and current provider capacity, claims the exact
action with `claim_runnable_action`, and invokes the returned native agent
name. The PreToolUse hook consumes that claim once and attaches the spawn
intent to the child lifecycle. An unclaimed Agent call is denied. After a
specialist reaches a terminal lifecycle state, the controller asks again. A
capacity-denied Agent call is rejected without creating a dead queued spawn
intent.

Native actions still require cooperative controller spawning because Claude
Code cannot replay a denied Agent call. Sidecar actions are different: the
router owns their bounded executor and can invoke, cancel, retry, and recover
them through persisted execution records. A sidecar must be invoked through
the claimed action rather than being hidden behind an ordinary model request.

## Workflow authority

`SessionStart` creates or resumes a run and captures session metadata. It does
not freeze a default task before a prompt exists. `UserPromptSubmit` creates a
task intake, extracts deterministic risk signals, and may request an advisory
fastpath proposal. The controller accepts or replaces the proposal before
mutation.

Workflow phase instances persist their dependencies, required actor, actual
starter, roles, execution kind, parallelism, attempt budgets, quorum,
deadlines, turn budget, result schema, and fallback policy. Phase transitions
read the persisted instance rather than trusting a caller-supplied workflow
definition. Execution statuses follow a bounded transition machine, and
quality quorum counts only schema/evidence-valid results accepted by the
controller.

Mutation authority is separate from route identity. A mutating execution must
hold the active workspace mutation lease, and the pre-tool hook rejects
mutations without that lease.

## Fastpath boundary

DiffusionGemma is a router-owned, read-only advisory sidecar for bounded route
recommendations and quick verification. It is not a standard worker, cannot
select uncertified endpoints, cannot lower a deterministic tier, cannot waive
review, and cannot mark findings resolved.

Fastpath failures bypass to deterministic/controller routing. Fastpath PASS is
advisory and never satisfies a mandatory adversarial phase. Any mutation after
verification invalidates the verification evidence. The current fastpath hook
has a short relevance wait and receives a queued job receipt; the router owns
the detached bounded call and persists the proposal against its intake. The
legacy fastpath transport is still separate from the general persisted
sidecar-execution lane, so it must not be treated as a native worker or as
completion evidence.

## Credential and protocol rules

- Provider keys are loaded from the OS keyring when available, with the
  hardened `providers.env` parser as a compatibility fallback, and isolated
  from Claude Code.
- Multiple key slots for one provider remain separate router/LiteLLM
  deployments under the same logical model group. Round-robin key rotation is
  independent from ordered model/provider fallback.
- Secret values never appear in YAML model/sidecar configuration, catalog
  metadata, statusline output, or audit logs. Only slot names and availability
  are persisted outside the credential backend.
- Internal router and Brigade headers are stripped before upstream dispatch.
- Supported provider protocol headers are forwarded only under endpoint policy.
- LiteLLM uses explicit provider namespaces such as `openai/model` for generic
  OpenAI-compatible upstreams.
- LiteLLM generation, configuration hash, and binding policy are immutable
  evidence. Old generations drain only after their pinned bindings, active
  requests, and active streams are released or explicitly failed and
  reconciled.
- All outbound model backends use shared admission/deadline/retry and stream
  accounting paths; provider deployment attribution accepts only trusted exact
  deployment IDs.

## Current implementation status

Implemented in the current working tree:

- model-agnostic controller resolution and controller bindings;
- provider configuration with adjustable FreeInference concurrency;
- live catalog synchronization and provider-supplied context metadata;
- certified, cache-aware fixed endpoint selection;
- fixed and managed-group model schema/compiler foundations;
- SQLite row factory, configuration-hash persistence, endpoint evidence, and
  pinned LiteLLM deployment lookup fusion;
- one shared external HTTP client and response-header copier;
- bounded streaming deadlines, incremental SSE usage parsing, and provider
  admission/circuit state, including atomic managed-group reservations;
- Claude subscription passthrough requests admitted and accounted through the
  shared synthetic `anthropic` provider lane;
- model-qualified visible agents, execution lifecycle hooks, mutation leases,
  task intake, phase snapshots, and controller MCP controls;
- Git-backed shadow workspaces, dirty-baseline preservation, persisted worker
  patches, deterministic overlap classification, green preflight/apply, and
  controller-visible yellow/red integration candidates;
- cooperative `get_runnable_actions` and orchestration-status operations;
- persisted `native_agent`, `sidecar_call`, and `controller_action` execution
  kinds, with bounded sidecar invocation, cancellation, retry, result schema
  validation, and ordered execution events;
- actor-scoped MCP principals with run/epoch/resource checks, ephemeral
  controller capabilities, and separate worker result capabilities;
- atomic native claim-to-child attachment with spawn correlation, terminal
  claim outcomes, provider-reservation handoff, and lifecycle reconciliation;
- fail-closed mutating workspace ownership, canonical-generation advancement,
  and crash-recoverable integration journaling;
- persisted completion tokens bound to the current workspace fingerprint and
  route snapshot, fail-closed completion validation, state-backed statusline
  visibility, and profile readiness reporting;
- registry-backed model-qualified native-agent manifest generation and
  dynamic role-alias/model binding projection;
- detached fastpath route jobs with bounded prompt-intake latency, persisted
  proposal correlation, and shutdown cancellation;
- read-only agent definitions without Bash, observational Bash allowlisting as
  a fallback guard, and SQLite-backed session-resume marker rehydration;
- controller-only route-proposal disposition operations; DiffusionGemma
  recommendations remain advisory until the active main controller explicitly
  accepts or rejects them;
- local FreeInference/LiteLLM integration kit with dynamic model sync.

Remaining work before a production-quality proving ground:

1. Complete trusted actual-deployment telemetry for multi-provider LiteLLM
   groups and reduce the conservative all-candidate reservation when the
   installed LiteLLM callback can identify the deployment before upstream
   dispatch.
2. Complete end-to-end native Claude hook coverage for shadow worktrees and
   exercise the controller's claimed yellow/red resolution flow. The green
   changeset path, persisted patches, overlap classification, deterministic
   preflight, and controller-action gate are implemented; semantic conflict
   resolution is intentionally not automatic.
3. Move detached fastpath route/verify jobs onto the general persisted sidecar
   executor so they share execution events, cancellation, retry, and status
   visibility with other bounded specialists.
4. Expand the local fixture integration suite for tools, parallel tools,
   tool-result continuation, structured output, cancellation, SSE usage, 401,
   429, 503, and generation pinning.
5. Run opt-in FreeInference paired endpoint certification and benchmark runs;
   do not run live inference probes automatically.
6. Reconcile all documentation, release manifests, installer assets, and
   generated hashes after source/config work is final.
7. Split the large `RouteState` implementation into domain repositories only
   after the current invariants have remained stable through the proving-ground
   tests. This is intentionally deferred; this pass keeps transactional
   changes localized and does not attempt a broad state rewrite.

This status is intentionally not a release claim. The repository should be
considered an active proving-ground implementation until the remaining list
and the end-to-end tests are complete.
