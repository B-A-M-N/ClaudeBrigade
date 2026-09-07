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

ClaudeBrigade has four additive configuration layers:

1. Native model lanes (`main`, `haiku`, `sonnet`, `opus`, and `fable`, plus
   the `background` and optional `custom` lanes) compile to Claude Code's
   model configuration. `background` projects to Claude Code's
   `ANTHROPIC_SMALL_FAST_MODEL` route; `custom` projects to the concrete
   operator-selected router alias and is never emitted as `model: custom`.
   A selected inference profile binds those lanes to logical models and
   provider deployments.
2. Durable role lanes (`controller`, `recon`, `implementer`, `adversary`, and
   `repairer`) remain the stable compatibility surface for ordinary routing.
3. Named native sidecar agents are separately configured workers. They select
   their own concrete provider/model/endpoint route, native identity, tools,
   permissions, worktree policy, and workflow position. They never select or
   consume the inference profile's model lanes. They are full Claude Code
   Agents, not bounded model calls.
4. Coprocessors are the small router-owned structured-call mechanism used for
   bounded advisory work such as DiffusionGemma routing. They have no tools,
   filesystem access, or workflow authority.

The inference profile and sidecar-agent profile are independent configuration
surfaces. A profile may provide both role routes and native slot/agent
projections, while a sidecar profile selects additional native workers with
independent provider/model/endpoint routes. Multiple named key slots
may be configured for one provider and rotated independently of model/provider
fallback. A deployment can therefore change models, providers, credentials,
roles, and sidecar workers without conflating those decisions.

Every native sidecar has two identities: a backwards-compatible configuration
selector (the YAML mapping key) and a stable semantic `worker_id`. Workflow
actions, manifests, MCP claims, lifecycle records, and capability snapshots
use the semantic worker ID. The backing model/provider/endpoint is route data,
not worker identity; changing that route must not rename a worker or change
its authority contract. Legacy configuration selectors remain accepted during
migration.

Bundled workflow phases use semantic worker IDs. A sidecar profile may still
select agents by legacy configuration selector because that profile is a
launch/resource allow-list, but scheduling resolves the selected worker to
one unique semantic identity before creating an action or execution.

Launch presets may also select a workflow composition. `fi-flow-proven` keeps
the hardened sidecar-led flow available as its own explicit choice. The
`claude-augmented` preset selects the additive composition: native slot agents
own the primary implementation, repair, and critical-review lanes, while the
independently routed sidecars provide grounding, conditional senior review,
completion diagnosis, correction, and the final critical gate. This does not
rebind a slot to a sidecar model and does not create a second scheduler; MCP
and SQLite remain authoritative for claims, dependencies, capacity, evidence,
and completion.

Each workflow declares a composition contract: `native-only` uses Claude Code
native workers, `sidecar-only` uses independently routed Brigade sidecars,
`native-augmented` requires both execution planes, and `adaptive` is reserved
for operator-authored workflows that intentionally choose per-phase execution.
The contract is validated when configuration loads and is captured by the
immutable workflow/epoch selection; it is not inferred from model names or
from a worker's prompt.

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

Bounded coprocessors such as DiffusionGemma remain endpoint-bound advisory
services. They do not become Claude Code agents and cannot mutate workflow
authority. Native sidecar agents are different: they enter the normal native
Agent lifecycle and may receive worktree-backed mutation capability when the
selected definition explicitly grants it.

Provider reservations persist the logical `model_id` that caused the
reservation. Admission evaluates the latest model-health record both when a
reservation is created and when queued work is promoted. A fresh failed check
or a health record older than the provider's `health_max_age_seconds` expires
the candidate instead of silently granting it a provider slot. Missing health
is admitted only when `allow_untested_models` is enabled. This applies equally
to controller, native sidecar, and bounded coprocessor routes.

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

Every configured provider has its own operator-facing concurrency policy.
FreeInference additionally exposes one environment override:

```text
FREEINFERENCE_MAX_CONCURRENCY=4
```

The bundled provider configuration also defaults to `max_concurrency: 4`.
The setting applies to both native-agent admission and upstream request
admission. The default allocation is one reserved controller lane and at most
three worker lanes. It includes the FreeInference controller, native role and
sidecar agents, direct Anthropic-format requests, LiteLLM/OpenAI-format
requests, and bounded coprocessors. The two internal counters remain separate
so the implementation can evolve their policies without changing the
operator-facing knob.

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
credentials remain process-local. OpenRouter, NVIDIA NIM, FreeInference.org,
and other OpenAI-compatible free-model providers use the same outbound
connection path: router-owned credentials, provider-specific admission,
bounded connect/read/write deadlines, transient retry, circuit state, and
stream-safe cleanup. Catalog refresh uses an explicit configured URL exactly;
only an endpoint-base fallback may append `/models`, and never when the base
already ends in `/models`.

The interactive configuration CLI recognizes credentials from the current
environment, router-materialized keyring slots, the OS keyring itself, and
the locked `providers.env` compatibility file. This keeps catalog refresh
usable before router startup while keeping secret values out of YAML, logs,
and Claude Code's environment. Transport failures without an HTTP response
(DNS, refused connections, TLS errors, timeouts, and protocol resets) feed the
same provider circuit breaker as transient 408/429/5xx responses.

### Run-wide resource policy

Provider limits are necessary but not sufficient for task admission. Each
task snapshots its workflow resource policy into `runs.resource_policy_json`
when the epoch is created. The policy covers active native workers,
mutators, reviewers, bounded coprocessors, shadow worktrees, reserved
tokens, and an optional deadline. It is immutable for that run/epoch.

The runnable-action wave planner uses the policy as a diagnostic filter, but
the claim transaction rechecks it under `BEGIN IMMEDIATE`. This prevents
concurrent controllers from exceeding a run-wide limit between planning and
spawn. A claim is admitted only when the intersection is available:

```text
provider capacity
  ∩ run resource policy
  ∩ phase parallelism / attempt budget
  ∩ package ownership and worktree limits
  ∩ token budget / deadline
```

The policy does not replace provider admission. A provider may still reject a
route even when the run has capacity, and a run may reject a route even when
the provider has spare capacity. MCP status exposes both provider-level and
run-level blockers so the controller can choose a smaller wave or wait.

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
Agent, including configured sidecar agents. Bounded structured inference uses
the separate coprocessor lane. The router binds, authenticates, routes,
meters, and records both; it does not secretly fan out to several models and
synthesize a hidden answer.

Native agent identities identify their logical assignment, independently of
the backing model, for example:

```text
brigade-sidecar-scout
brigade-sidecar-coding-worker
brigade-sidecar-correctness-reviewer
```

The native lifecycle hooks create and close authoritative SQLite execution
records. JSONL is diagnostic evidence only. The scheduler distinguishes four
execution kinds:

```text
native_agent       visible Claude Code Agent with repository tools
coprocessor_call   bounded router-owned structured specialist call
sidecar_call       legacy name accepted only while old rows migrate
controller_action  controller-only integration/adjudication operation
```

Native sidecar agents use `native_agent` with a `worker_kind` and `worker_id`
that identify their independent configuration. Only `coprocessor_call` is a
router-owned bounded call. Both are shown explicitly in orchestration status.
Each native sidecar also has an individually routable public model alias. A
request using that alias must match the claimed native sidecar identity and
attached execution; it cannot be used as an unclaimed direct model escape
hatch. The resolved sidecar request still enters the selected provider's
normal request admission and therefore remains subject to that provider's
configured concurrency limit.

### Automatic coprocessor feedback

When `feedback_monitor` is enabled, Claude Code's `PostToolBatch` hook first
performs a local watched-tool check. Edit/Write/NotebookEdit batches are
classified as the semantic `post_edit_checkpoint`; observational shell
traffic remains a generic `post_tool_batch`. Only relevant batches reach the
authenticated loopback checkpoint endpoint. SQLite then applies evidence
deduplication, cooldown, per-execution call budgets, and feedback
parallelism. A fresh checkpoint starts one bounded coprocessor execution and
returns a pending receipt or, when ready, an advisory result through hook
`additionalContext` before the next model turn. Ready results are delivered
once at a later checkpoint when the original hook times out. A completed
coprocessor response is not accepted evidence: the main controller must
explicitly adjudicate it through MCP before it can satisfy a workflow phase.
Delivery and later adoption/rejection are persisted separately from transport
completion so the system can measure whether advice helped or caused rework.
Outcome telemetry records the concrete model/provider route, latency, token use,
estimated cost, adoption disposition, resulting actions/changesets, later
validation, and harm classification. After the configured minimum sample,
adoption and harm thresholds can automatically move a coprocessor from
`automatic` to its configured degraded mode (`advisory-on-request`,
`shadow-only`, or `disabled`); MCP exposes the metrics used for that decision.

The monitor does not issue a model request for every MCP event. The actual
coprocessor request uses the same provider admission, retry, deadline, usage,
and release path as ordinary inference. A conservative worker-budget gate
defers feedback when the provider is already full, protecting the controller
lane. Feedback cannot authorize tools, mutation, integration, completion, or
route changes. Sentinel delivery is wrapped as non-authoritative evidence;
its suggested next step is explicitly an untrusted hypothesis, and the
controller must independently assess the current workspace before using it.
Invalid JSON, incomplete evidence, stale packets, and results outside the
declared capability contract abstain or fail. Router-owned
executions use one terminalization path for completion, failure, timeout,
cancellation, claim release, route telemetry, and phase re-evaluation.
Provider-specific concurrency limits remain authoritative for every
coprocessor route; FreeInference's configured limit of four is not a global
limit for other providers. Merge-risk verification is queued asynchronously
before deterministic integration; a late fail/escalate result becomes a
durable controller finding rather than delaying or reversing a deterministic
green apply.

The selected `sidecar_profile` also has a launch-level
`coprocessors_enabled` kill switch. Turning it off removes bounded
coprocessor phases and automatic feedback for that launch while leaving native
sidecar workers and their independently configured routes available. Individual
coprocessors, feedback monitoring, and fastpath retain their own `enabled`
controls. `sidecars.yaml` has the same top-level switch for launches that do
not select a named sidecar profile. The strictest applicable switch wins.

The registry owns model-neutral native identities. Profile specialist entries
may declare a semantic `native_agent_name` and `public_model_alias`; their
backing model/provider/endpoint remains route data. The launcher projects
those entries into the Claude Code `--agents` manifest from the stable role
definition. Routing, scheduling, hooks, `/v1/models`, and status use the same
manifest rather than parallel model-name tables.

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
intent. `get_runnable_action_wave` may return several independent native
workers at once; each action has a persisted package identity and claim.

Native actions still require cooperative controller spawning because Claude
Code cannot replay a denied Agent call. Native sidecar actions use this same
path and are invoked with their returned native agent identity. Coprocessor
actions are different: the router owns their bounded executor and can invoke,
cancel, retry, and recover them through persisted execution records. A native
sidecar must never be routed through the coprocessor executor.

## Workflow authority

`SessionStart` creates or resumes a run and captures session metadata. It does
not freeze a default task before a prompt exists. `UserPromptSubmit` creates a
task intake, extracts deterministic risk signals, and may request an advisory
fastpath proposal. The controller accepts or replaces the proposal before
mutation.

Workflow phase instances persist their dependencies, required actor, actual
starter, roles, execution kind, parallelism, attempt budgets, quorum,
deadlines, turn budget, result schema, fallback policy, and adaptive launch
policy. `minimum_first` phases launch only their initial evidence wave, then
add a replica or retry after terminal evidence shows that the configured
quorum is still unmet. `all_packages` phases launch distinct persisted
packages as capacity permits. Phase transitions
read the persisted instance rather than trusting a caller-supplied workflow
definition. Execution statuses follow a bounded transition machine, and
quality quorum counts only schema/evidence-valid results accepted by the
controller.

Mutation authority is separate from route identity. A mutating execution must
hold the active workspace mutation lease, and the pre-tool hook rejects
mutations without that lease.

### Contract, package, and escalation state

Controller-managed tasks publish an immutable task contract before mutation.
The normalized SQLite coverage ledger stores mandatory requirements, ambiguity
state, evidence links, and coverage audits. Implementation fanout is over
persisted `work_packages`, not over duplicate copies of the whole task. Each
package carries its objective, path scope, requirement IDs, acceptance
criteria, required tests, dependency IDs, and contract digest. The scheduler
does not emit generic concurrent mutator slots when no package contract exists.

Adaptive escalation is a persisted policy decision. The pure escalation
policy classifies observed failures, missing coverage, security-path changes,
repair loops, and evidence gaps; the controller may then apply one atomic
epoch escalation. Applying it cancels stale action claims without rewriting
existing work, records the decision, and forces a fresh runnable-action query.
No escalation may downgrade an already selected workflow or silently discard
worker evidence. Escalation pauses mutation. The controller cannot acknowledge
the stronger graph until the compensating recon/design/risk-review phases have
completed; only then may repair-or-replan mutation resume.

Controller phases are also first-class runnable actions. Contract publication,
package planning, adjudication, verification, and integration remain on the
main controller path; native workers and independent sidecars remain native
Agent executions; bounded coprocessors remain router-owned advisory calls.
The statusline and MCP orchestration projection expose these as distinct
states: executed, accepted, integrated, and verified are not interchangeable.

Package planning is a real controller action, not an implicit worker route.
It may be claimed before the controller has created a model binding because
the action authorizes durable planning work rather than an inference request.
When a phase declares `produces: work_packages`, completion requires at least
one downstream package with requirement IDs, path scope, acceptance
criteria, required tests, and a contract digest. A mutating phase declaring
`fanout_from: work_packages` is not runnable until those persisted packages
exist; it never receives a generic whole-task mutator slot.

Selected slot bindings are persisted at epoch creation. A later profile
reload cannot change the model or capability contract of an already claimed
native role or sidecar execution. The action snapshot, rendered native agent
definition, and first routed request must agree on native identity, the
selected Claude Code slot for slot workers or the independent
provider/model/endpoint route for sidecars, capability digest, and work
package.

## Fastpath boundary

DiffusionGemma is a router-owned, read-only advisory coprocessor for bounded route
recommendations and quick verification. It is not a standard worker, cannot
select uncertified endpoints, cannot lower a deterministic tier, cannot waive
review, and cannot mark findings resolved.

Fastpath failures bypass to deterministic/controller routing. Fastpath PASS is
advisory and never satisfies a mandatory adversarial phase. Any mutation after
verification invalidates the verification evidence. The current fastpath hook
has a short relevance wait and receives a queued job receipt; the router owns
the detached bounded call and persists the proposal against its intake. The
fastpath request protocol remains specialized OpenAI-compatible JSON, but its
job lifecycle is a persisted `coprocessor_call` execution with cancellation and
status visibility. It must not be treated as a native worker or as completion
evidence.

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
- OpenRouter, NVIDIA NIM, FreeInference.org, and other OpenAI-compatible
  provider connections with exact catalog URLs, pre-start credential
  detection, bounded transient discovery retries, and transport-failure
  circuit feedback;
- certified, cache-aware fixed endpoint selection;
- fixed and managed-group model schema/compiler foundations;
- SQLite row factory, configuration-hash persistence, endpoint evidence, and
  pinned LiteLLM deployment lookup fusion;
- one shared external HTTP client per event loop with failed-pool recycling
  and shutdown cleanup, plus response-header copying;
- bounded streaming deadlines, incremental SSE usage parsing, and provider
  admission/circuit state, including atomic managed-group reservations;
- OpenAI-compatible fastpath transport routed through shared admission,
  provider retry/circuit, deadline, and usage accounting;
- Claude subscription passthrough requests admitted and accounted through the
  shared synthetic `anthropic` provider lane;
- model-qualified visible agents, execution lifecycle hooks, mutation leases,
  task intake, phase snapshots, and controller MCP controls;
- Git-backed shadow workspaces, dirty-baseline preservation, persisted worker
  patches, deterministic overlap classification, green preflight/apply, and
  controller-visible yellow/red integration candidates;
- cooperative `get_runnable_actions` and orchestration-status operations;
- persisted `native_agent`, `coprocessor_call`, legacy `sidecar_call`, and
  `controller_action` execution kinds, with bounded coprocessor invocation,
  cancellation, retry, result-schema validation, and ordered execution events;
- actor-scoped MCP principals with run/epoch/resource checks, ephemeral
  controller capabilities, and separate worker result capabilities;
- atomic native claim-to-child attachment with spawn correlation, terminal
  claim outcomes, provider-reservation handoff, and lifecycle reconciliation;
- fail-closed mutating workspace ownership, canonical-generation advancement,
  and crash-recoverable integration journaling;
- mutation-lease acquisition that lazily reclaims a lease from a crashed
  holder (stale heartbeat) instead of permanently locking the workspace for
  writes, the same lazy-expiry-on-read shape used for runnable action claims;
- red shadow-integration candidates (invalid patch, unauthorized files, or a
  failed integration preflight) automatically escalate to an accepted
  finding so `evaluate_condition`'s `accepted_findings` gate sees them
  instead of only living in the integration-candidates table; resolved when
  the controller discards or retries the candidate;
- `RouteState` is decomposed into ~20 single-purpose repository mixin
  modules (`router/enhanced_router/*_state.py`), each covering one bounded
  concern (workflow phases, agent executions, shadow workspaces, mutation
  leases, provider reservations, run orchestration, and so on) and mixed
  into `RouteState` via ordinary multiple inheritance; `state.py` itself
  retains only schema/migrations and the handful of methods that
  deliberately span more than one repository's tables;
- opt-in cost-aware model recommendation (`RecommendationConstraints.prefer_low_cost`)
  and a profile-load diagnostic flagging profiles that assign the same model
  to every implemented role, which would make adversarial review a same-model
  self-review;
- persisted completion tokens bound to the current workspace fingerprint and
  route snapshot, fail-closed completion validation, state-backed statusline
  visibility, and profile readiness reporting;
- registry-backed model-qualified native-agent manifest generation and
  dynamic role-alias/model binding projection;
- detached fastpath route/verify jobs using persisted sidecar executions,
  bounded prompt-intake latency, proposal/verification correlation, and
  shutdown cancellation, with bounded retry reconstruction after router
  restart;
- read-only agent definitions without Bash, observational Bash allowlisting as
  a fallback guard, and SQLite-backed session-resume marker rehydration;
- controller-only route-proposal disposition operations; DiffusionGemma
  recommendations remain advisory until the active main controller explicitly
  accepts or rejects them;
- local FreeInference/LiteLLM integration kit with dynamic model sync.
- an explicit certification boundary that normalizes sanitized FreeInference
  contract reports into route-qualified endpoint certification rows; report
  publication is operator-invoked and never implied by catalog discovery or
  a successful transport response;
- controller-owned task contracts, requirement/evidence coverage, persisted
  work-package contracts, dependency-aware package waves, adaptive escalation
  decisions, all-package quality gates, and unified orchestration/status
  projections;
- read-only `claude-brigade status` Rich output with package ownership,
  worker dispositions, requirement progress, wait reasons, blockers, and
  escalation state; it never starts Claude Code or creates a run;

Remaining work before a production-quality proving ground:

1. The LiteLLM 1.93.0 deployment-filter callback is implemented behind
   ``BRIGADE_LITELLM_DISPATCH_FILTER=1`` and is fail-closed for malformed or
   out-of-policy managed-group metadata. Run the contract harness against the
   actual child/upstream pair before enabling it. Do not reduce conservative
   all-candidate reservations until live benchmarks prove that admission and
   callback selection agree. The router still records requested candidates
   plus validated physical deployment attribution from response metadata;
   unknown identities remain explicitly untrusted and never change admission.
2. Complete end-to-end native Claude hook coverage for shadow worktrees and
   exercise the controller's claimed yellow/red resolution flow. The green
   changeset path, persisted patches, overlap classification, deterministic
   preflight, and controller-action gate are implemented; semantic conflict
   resolution is intentionally not automatic.
3. Exercise detached-fastpath restart recovery against a live streaming
   fixture. Startup now reconciles every still-active detached job
   immediately, records ``orphaned_after_restart`` and ``orphaned_at``,
   closes in-flight route attempts, and keeps the orphan from consuming a
   fallback candidate.
4. The local transport fixture suite now covers configured 429/503 retries,
   non-retryable 401 responses, split SSE usage accounting, stream cleanup on
   cancellation, and parallel tool-result continuation payloads. Remaining
   proving work is provider-specific integration coverage for structured
   output, cancellation, streaming usage, and generation pinning through the
   actual LiteLLM child and certified upstream adapters.
5. Run opt-in FreeInference paired endpoint certification and benchmark runs;
   publish sanitized contract reports through the certification boundary;
   do not run live inference probes automatically. The report bridge now
   records capability-specific evidence, a stable evidence digest, harness
   version, and protocol version in authoritative SQLite state.
6. Reconcile all documentation, release manifests, installer assets, and
   generated hashes after source/config work is final.
7. Broader operational proving remains: live pre-dispatch deployment callback
   coverage varies by installed LiteLLM version, and end-to-end provider
   fixture coverage still needs expansion. Run-level `max_estimated_cost`
   admission is now enforced alongside active-agent, token, and deadline
   limits; actual billing precision still depends on provider usage telemetry.

This status is intentionally not a release claim. The repository should be
considered an active proving-ground implementation until the remaining list
and the end-to-end tests are complete.
