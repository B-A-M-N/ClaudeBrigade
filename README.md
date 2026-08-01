# ClaudeBrigade: Model-Agnostic Controller + Visible Native Agents

This installs a second command, `claude-brigade`, backed by a separate Claude Code profile. Your normal `claude` command and `~/.claude` profile are not edited.

The launcher scripts are named `claude-brigade` and friends by default. For backward compatibility during migration, the install also produces `claude-enhanced` symlinks in `~/.local/bin/`. Use `claude-brigade` as the preferred name for all documentation and daily use.

## What you get

```text
ordinary task
  -> active Claude Code main model: controller
     -> visible, model-qualified native agents when evidence or mutation is needed
     -> controller reviews the stable diff and adjudicates findings
     -> controller runs final verification and a workspace-hash gate

Claude Code -> 127.0.0.1:8787 loopback router
  active main model           -> registry-selected controller route
  visible native agents       -> immutable logical model/deployment policy
  FreeInference               -> local provider pool with shared admission

Mutating native agents use Claude Code worktree isolation. Their changes are
captured as persisted shadow changesets. Disjoint valid changesets are
preflighted with `git apply --check` before application; same-file or invalid
changes are escalated to the active main controller as yellow/red integration
actions. DiffusionGemma may advise, but never merges or resolves a conflict.
```

You give it normal requests. The active controller chooses the tier and invokes native subagents itself.

The launcher owns agent definitions, controller prompt, hooks, MCP, and settings. Use `--model` or `--controller-model` to select the active controller.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current controller, logical
model/deployment, FreeInference cache-selection, concurrency, and cooperative
scheduling contract. MCP is the authenticated local control surface for
changing models, profiles, endpoint overrides, and workflow state; it is not
the specialist execution engine.

## Local FreeInference BYOK

FreeInference can be used as the controller or as a visible specialist
provider without Anthropic credentials. Put the user-owned key in the local
`providers.env` file and keep Claude Code pointed at the loopback router. The
default FreeInference concurrency is four and can be adjusted with:

```bash
FREEINFERENCE_MAX_CONCURRENCY=4
```

Unless an endpoint is explicitly configured, certified FreeInference
deployments are ranked by fresh token-weighted cache-read rate. Context and
output limits are not bundled for FreeInference; an explicit catalog sync
uses the limits returned by the provider API when present.

FreeInference is treated as one provider pool even when the request transport
differs. A FreeInference controller, a direct Anthropic-format request, and a
LiteLLM/OpenAI-format request all consume the same provider concurrency
budget. The default is four concurrent requests/agent admissions and is
operator-adjustable:

```yaml
providers:
  freeinference:
    max_concurrency: 4
```

or, for a launch-specific override:

```bash
FREEINFERENCE_MAX_CONCURRENCY=4 claude-brigade
```

The provider key remains in the router/LiteLLM process. Claude Code receives
only a per-run local router token, so a FreeInference-only session does not
require Anthropic authentication.

For a standalone local LiteLLM integration kit, see
[`integrations/freeinference-litellm`](integrations/freeinference-litellm/README.md).

## Install

```bash
./install.sh
```

The script installs to:

- `~/.local/share/claude-brigade/` — application code and Python virtual environment
- `~/.claude-brigade/` — Claude Code profile (settings, agent definitions, hooks)
- `~/.config/claude-brigade/` — YAML configs and token files
- `~/.cache/claude-brigade/` — router logs, PID files, MCP config
- `~/.local/state/claude-brigade/` — SQLite route state
- `~/.local/bin/` — `claude-brigade`, `claude-brigade-doctor`, etc. (plus `claude-enhanced*` symlinks for backward compatibility)

Edit the protected key file:

```bash
nano ~/.config/claude-brigade/providers.env
```

Use this format:

```bash
FREEINFERENCE_API_KEY='your_freeinference_key_here'
LONGCAT_API_KEY='your_key_here'
LONGCAT_API_BASE='https://api.longcat.chat/anthropic'
```

Authenticate the separate profile only when the selected arrangement needs a
Claude/Anthropic route. FreeInference-only sessions use the local router token
and the user-owned provider key:

```bash
claude-brigade-login
claude-brigade-doctor
```

Then use it like ordinary Claude Code:

```bash
cd /path/to/repository
claude-brigade
```

## Workflow Engine (M08)

ClaudeBrigade creates a task intake from the submitted request, extracts
deterministic minimum-risk signals, and may ask the controller to accept or
replace an advisory route proposal. It then runs a phase-based workflow:

| Tier | Phases |
|------|--------|
| **trivial** | No subagents — controller answers or applies a safe edit directly via `controller-direct`. |
| **normal** | `implementation` — controller writes a contract, configured implementer mutates source, controller reviews the diff. |
| **cross-cutting** | `recon` -> `implementation` -> `adversarial-review` -> `repair` (conditional) -> `verification`. Recon must run before any mutation, and one completed adversary review must complete before verification. |
| **high-risk** | Same phases as cross-cutting, but two independently spawned adversaries must approve before repair begins. A design-phase adversary runs before implementation. |

Phases have explicit `depends_on` edges and are enforced by the `completion_guard` hook. Each phase can be assigned specific roles (`recon`, `implementer`, `adversary`, `repairer`) and carries a `mutation` flag controlling tool access. The active controller always drives orchestration and adjudication.

Configuration lives in `workflows.yaml`:

```yaml
workflows:
  cross-cutting:
    default_profile: hybrid
    phases:
      - id: recon                roles: [recon]      required: true   mutation: false
      - id: implementation       roles: [implementer] depends_on: [recon] mutation: true
      - id: adversarial-review   roles: [adversary]  depends_on: [implementation] mutation: false
      - id: repair               roles: [repairer]   depends_on: [adversarial-review] conditional: accepted_findings mutation: true
      - id: verification         actor: controller   depends_on: [repair]  mutation: false
```

## Enforced controls

- Only the generated, model-qualified native agents in the launch manifest are
  spawnable from the main controller.
- Agent definitions are injected with native `--agents` session configuration,
  which outranks project-level agent files and prevents accidental name
  collisions or role replacement.
- Native Agent calls remain visible and are admitted cooperatively through
  `get_runnable_actions`; the router cannot replay a denied Agent call.
- Only implementer/repairer executions and the exceptional `controller-direct`
  path may use file-write tools, and mutating executions require the active
  workspace mutation lease.
- Read-only roles are blocked from obvious mutating shell commands. This is a guardrail, not an operating-system sandbox.
- Agent start/stop lifecycle evidence is recorded without prompts or source content.
- Cross-cutting completion requires recon plus one completed fresh adversary.
- High-risk completion requires recon plus two separately spawned adversaries.
- The final completion hook checks role lifecycles, unresolved findings, `git diff --check`, and a SHA-256 fingerprint of tracked changes plus untracked files.

### Cooperative native-agent scheduling

Claude Code does not replay an `Agent` call that the router denied. The
controller therefore asks the local control plane for runnable actions, claims
the exact returned action, and only then invokes that native agent:

```text
get_runnable_actions
  -> claim_runnable_action
  -> invoke the returned native agent
  -> terminal hook releases capacity
  -> get_runnable_actions again
```

An unclaimed or capacity-denied Agent call is rejected without creating a
dead queued spawn intent. The router records admission and lifecycle state,
but it does not secretly fan out to models or synthesize hidden specialist
answers.

### Shadow worktrees and merge safety

Mutating native agents use Claude Code worktree isolation when their generated
definition requests it. ClaudeBrigade captures the canonical dirty baseline,
extracts each worker changeset, classifies path overlap, and preserves the
user's existing changes.

- **Green:** a disjoint changeset passes deterministic `git apply --check` and
  may be applied to the unchanged canonical baseline.
- **Yellow:** same-file or uncertain overlap is retained as a persisted
  candidate and becomes a controller integration action.
- **Red:** an invalid patch or conflict is retained for controller-directed
  retry, repair, or discard.

The router never force-applies a failed patch, commits, stashes, resets, or
silently performs a three-way merge. DiffusionGemma may provide bounded,
read-only merge-risk advice, but it cannot merge, resolve a conflict, waive a
required review, or mark completion. Yellow and red candidates require the
active main controller to claim the exact integration action and decide the
next step. Completion remains blocked while they are unresolved.

## Configuration

The YAML files under `~/.config/claude-brigade/` control routing, profiling, and provider admission:

### `models.yaml`

Defines every model ClaudeBrigade can reach. Each entry has:

- `display_name` — human-readable name
- `backend` — `direct-anthropic` (e.g., LongCat) or `litellm` (local models via Ollama, OpenRouter, etc.)
- `upstream_model` (for `direct-anthropic`) or `litellm_model` (for `litellm`)
- `api_base` / `api_base_env` / `api_key_env` — connection and credential details
- `capabilities` — tool/protocol certification and role capabilities. Provider
  context/output limits are not prefilled; live provider catalog metadata is
  authoritative when available.
- `allowed_roles` — which brigade roles may use this model
- `enabled` — defaults to `true`; set `false` to disable without deleting

For a logical model with more than one provider deployment, use explicit
endpoint policy. A fixed route pins one certified endpoint. An explicitly
configured `routing_mode: managed-group` binds a logical LiteLLM group and
approved deployment policy; LiteLLM may then select an equivalent deployment
within that group while reporting the actual provider for telemetry.

For FreeInference, `endpoint: auto` means: require current certification and
health, then prefer the endpoint with the highest fresh token-weighted cache
read rate. For example, if Qwen3.6-35B has materially better cache reuse at
the OpenAI-compatible endpoint than at the Anthropic-compatible endpoint,
new fixed bindings select OpenAI. Existing bindings remain pinned.

### `profiles.yaml`

Maps the four brigade roles (`recon`, `implementer`, `adversary`, `repairer`) to model IDs defined in `models.yaml`. Profiles let you swap the entire team configuration:

```yaml
profiles:
  hybrid:
    recon: qwen-local
    implementer: longcat-2
    adversary: longcat-2
    repairer: longcat-2
```

The router validates at load time that every role reference points to an existing, enabled model.

### `workflows.yaml`

Maps workflow tiers (`trivial`, `normal`, `cross-cutting`, `high-risk`) to a `default_profile` and an ordered list of phases with role assignments, dependency edges, and mutation flags.

### `providers.yaml`

Provider limits apply across every endpoint of a provider. For FreeInference,
the default shared concurrency is four upstream request streams. The same
setting is also used for provider agent admission. This includes a
FreeInference-backed main controller; requests to an Anthropic passthrough
model do not consume the FreeInference pool.

Adjust the installed value in `~/.config/claude-brigade/providers.yaml`:

```yaml
providers:
  freeinference:
    max_concurrency: 4
```

For a temporary launch-specific override, set `FREEINFERENCE_MAX_CONCURRENCY`
in `providers.env`. The router consumes this setting, while Claude Code does
not receive the FreeInference key.

### Runtime commands

```bash
# Change role-to-model route at runtime via MCP
set_role_route { "run_id": "...", "role": "implementer", "model_id": "glm-review", "endpoint": "auto" }

# Reload the model catalog (triggers blue-green LiteLLM deployment)
reload_catalog
```

MCP is an authenticated, loopback-only management surface for model
discovery, profiles, route/endpoint policy, task phases, runnable-action
claims, proposals, and orchestration status. It is not the specialist
execution engine. Native Claude Code Agent invocations remain the visible
execution boundary.

## Security

### Environment isolation

- Provider API keys (`LONGCAT_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, etc.) are loaded only into the router process from `providers.env`.
- `FREEINFERENCE_MAX_CONCURRENCY` is a non-secret router setting; it changes the shared FreeInference admission limit without exposing `FREEINFERENCE_API_KEY` to Claude Code.
- The launcher explicitly unsets inherited API-key, OAuth-token, cloud-provider, model-override, and global-subagent-model environment variables before starting Claude Code.
- This prevents a shell-level override from silently replacing the selected controller or forcing every subagent onto the wrong model.

### Safe env-file parsing (env_parser.py)

The `providers.env` file is parsed by a strict, non-executing dotenv parser with:

- **Symlink rejection** — file is rejected outright when it is a symlink.
- **Ownership validation** — file must be owned by the current user or root.
- **Mode checks** — group/world writable or executable bits cause rejection.
- **Null-byte and encoding rejection** — prevents injection through binary or malformed content.
- **Key allowlist** — optional set of permitted variable names; unknown keys raise an error.

### File-based process locking (lock.py)

The launcher uses POSIX `flock(2)` to prevent concurrent invocations from racing on port selection, PID file creation, and token generation.

### Symlink protection

Token files (`router.token`, `litellm.token`) and `providers.env` are rejected when they are symlinks. File writes use `O_NOFOLLOW | O_EXCL` flags to prevent symlink-based token substitution.

### Restrictive umask

The launcher sets `umask 077` before creating any runtime files, ensuring owner-only permissions by default.

### Launch audit trail

Every invocation writes a structured JSON event to `$CACHE_DIR/audit.jsonl` (mode `0600`) with timestamp, run ID, PID, provider keys loaded, and a truncated hash of the router token.

### Log rotation

The router log auto-rotates at 10 MB with 5 retained backups, preventing unbounded disk usage.

Because the profile is intentionally separate, user-level skills, plugins, and memory stored only under `~/.claude` are not automatically copied. Project-level `.claude` configuration and `CLAUDE.md` files still load normally.

## Router behavior

The router preserves the Claude Code message/tool contract while routing each
logical model through its certified backend. It isolates incoming local
credentials, injects only the selected provider credential, applies the
endpoint's protocol/header policy, streams upstream SSE, and records bounded
route/admission/usage telemetry without logging prompts, tool results, or
provider keys.

FreeInference endpoint selection is cache-aware, but compatibility and health
come first. Cache rates are token-weighted and observations are tied to the
model, endpoint, provider, configuration, and harness identity. A better
observation affects new bindings; it does not silently move a fixed route.

Context limits are never guessed by the router. When the provider catalog
returns context or output metadata, the live catalog may publish it. When the
provider omits it, the value remains unknown and the request follows the
provider's own enforcement behavior.

### LiteLLM Proxy (Blue-Green)

ClaudeBrigade runs a local LiteLLM proxy child process to serve requests routed to `litellm` backend models (Ollama, OpenRouter, etc.). The proxy is managed by a supervisor that implements **blue-green deployment**:

1. When `models.yaml` changes and the catalog is reloaded, the supervisor creates a new **generation** with a freshly generated LiteLLM YAML config.
2. It starts a new LiteLLM child process on a free port and health-probes `/health`.
3. On success, the old child enters **draining** — it continues serving in-flight requests from bindings pinned to the old generation.
4. When the old child has no active bindings, it is terminated and the new generation becomes active.

This ensures zero-downtime model catalog updates. Fixed bindings pin
`catalog_generation` and endpoint identity. Managed-group bindings pin the
logical group, approved deployment set, generation, and policy digest while
allowing LiteLLM to select among equivalent deployments according to the
configured fallback policy.

## Isolation

- Normal Claude profile: `~/.claude`
- Enhanced Claude profile: `~/.claude-brigade`
- Provider keys: `~/.config/claude-brigade/providers.env` with mode `0600`
- Router token: `~/.config/claude-brigade/router.token` with mode `0600`
- LiteLLM key: `~/.config/claude-brigade/litellm.token` with mode `0600`
- Launch audit trail: `$CACHE_DIR/audit.jsonl` with mode `0600`
- Router log: `$CACHE_DIR/router.log` with mode `0600`, auto-rotated at 10 MB (5 backups)

### Security primitives (BrokeLLM-derived)

- **Symlink protection**: Token files (`router.token`, `litellm.token`) and `providers.env` are rejected when they are symlinks. File writes use `O_NOFOLLOW | O_EXCL` flags to prevent symlink-based token substitution.
- **Restrictive umask**: The launcher sets `umask 077` before creating any runtime files, ensuring owner-only permissions by default.
- **Expanded env key ban list**: All common provider API keys (`OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`, etc.) are unset before Claude Code starts — they are only available to the router process.
- **Launch audit trail**: Every invocation writes a structured JSON event to `audit.jsonl` with timestamp, run ID, PID, provider keys loaded, and a truncated hash of the router token.
- **Log rotation**: The router log auto-rotates at 10 MB with 5 retained backups, preventing unbounded disk usage.

## Diagnostics

```bash
claude-brigade-doctor
cat ~/.cache/claude-brigade/router.log
CLAUDE_CONFIG_DIR=~/.claude-brigade claude doctor
CLAUDE_CONFIG_DIR=~/.claude-brigade claude auth status --text
claude-brigade-router-stop
```

After changing a provider key or provider limit, run
`claude-brigade-router-stop` before the next launch so the router restarts with
the updated local configuration.

## Important limitations

This is a local compatibility and orchestration layer, not a hosted provider
service or an Anthropic-supported non-Claude configuration. Provider protocol
support and model capabilities can change. A model is not controller- or
mutation-eligible merely because it appears in a live catalog; the required
endpoint certification must be current.

Automatic semantic conflict resolution is intentionally out of scope for the
router. The main controller remains responsible for yellow/red shadow-tree
integration decisions, followed by deterministic verification on the
canonical workspace.
