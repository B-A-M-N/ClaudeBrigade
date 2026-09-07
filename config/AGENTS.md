# Runtime Configuration

## Purpose

This subtree is the checked-in source configuration for models, providers,
native model slots, role profiles, native sidecar workers, coprocessors,
workflows, and the optional advisory fastpath.

## Ownership

Operator-facing policy inputs. Runtime state, secrets, credentials, and
generated LiteLLM deployment files belong outside this subtree.

## Local Contracts

- `models.yaml` defines logical models, backends, capabilities, allowed roles,
  endpoint declarations, and certification metadata.
- `profiles.yaml` maps recon/implementer/adversary/repairer roles to model IDs;
  specialist launch identities are declared here when they are model-specific.
- `workflows.yaml` defines persisted phase contracts, dependencies, actors,
  execution kinds, fanout/attempt/quorum policy, fallback behavior, and the
  immutable run-wide resource policy snapshot used for action admission.
  Each workflow also declares `composition_mode` (`native-only`,
  `sidecar-only`, `native-augmented`, or `adaptive`) so the selected execution
  planes are validated as a contract rather than inferred from model names.
- `providers.yaml` defines admission limits, deadlines, retry policy, provider
  endpoints, and discovery settings. OpenRouter, Kilo/Crush, Cline,
  OpenCode Zen/Go, FreeTheAI, Requesty, FreeModel, and NVIDIA NIM are bundled.
  `health_max_age_seconds` and `allow_untested_models` control whether model
  health records are fresh enough for reservation admission.
  It carries `provider_schema_version`; installer upgrades merge missing
  router-owned metadata (including discovery defaults) into an existing user
  file while preserving user endpoints, limits, additions, and removals.
- `fastpath.yaml` is advisory/read-only. It cannot choose physical endpoints,
  lower a deterministic tier, mutate files, or authorize completion.
- `sidecars.yaml` stores both independently configured native sidecar agents
  and bounded coprocessors. Native sidecars select their own concrete
  provider/model/endpoint route, native identity, tools, permissions,
  worktree policy, and workflow role. They do not select or consume the
  Claude Code `main`/`sonnet`/`haiku`/`opus`/`fable` lanes plus the
  `background` (`ANTHROPIC_SMALL_FAST_MODEL`) and optional `custom` lanes.
  Coprocessors
  select only bounded request limits and structured-call policy. New bounded
  definitions use `coprocessors`; old `sidecars` keys are migration input
  only.
- `sidecar_profiles.yaml` may set `coprocessors_enabled: false` for a launch
  bundle. That disables bounded coprocessor workflow phases and automatic
  feedback for the selected launch without disabling native sidecar workers;
  individual coprocessors, feedback monitoring, and fastpath still honor their
  own `enabled` settings.
- `sidecars.yaml` also accepts the top-level `coprocessors_enabled` master
  switch for launches without a named sidecar profile. The global switch
  affects bounded MCP calls only; native sidecar agents and fastpath remain
  independently configurable. The strictest applicable switch wins.
- Each native sidecar's `public_model_alias` is an independently routable
  inference handle. It remains bound to the sidecar's configured provider and
  is still constrained by that provider's concurrency/admission policy.
- Each native sidecar also declares a stable semantic `worker_id`. Workflow
  phases and runtime lifecycle records should use that identity; model names
  belong only to the sidecar route. Existing YAML mapping keys remain valid
  migration selectors. Sidecar profiles may retain those selectors as their
  launch allow-list, but workflow phases should use semantic worker IDs.
- `profiles.yaml` may define `slots` (`main`, `background`, `haiku`, `sonnet`,
  `opus`, `fable`, and optional `custom`) and named native `agents` in
  addition to the durable role routes. `custom` is always projected as its
  concrete router alias; never emit `model: custom`.
  Slot bindings do not replace the role aliases.
- `profiles.yaml` is the saved Claude Code inference configuration. Each role
  can specify a primary model/endpoint and ordered `fallback_models`; an
  optional `controller_model` is applied by `claude-brigade --brigade-profile`.
- The interactive `claude-brigade-config` command stores API keys in the OS
  credential store when available. `providers.env` is a locked-down legacy
  fallback and migration source; never put values in YAML, source,
  command-line arguments, or logs.
- `discovered_models.yaml` is generated account/provider catalog metadata. It
  contains no secrets and grants no native role, tool, mutation, or
  certification authority until an operator explicitly configures that.
- Provider-discovered route identity is the concrete tuple
  `(provider_id, model_id, endpoint_id)`; a catalog URL is exact when supplied
  under `discovery.url`, and `/models` is appended only to an inference base
  URL used as a fallback.
- Configuration must not contain API keys or local token contents. Use the
  operator configuration directory for secrets and environment values.

## Work Guidance

Use IDs already present in `models.yaml`; preserve exact role/capability
compatibility and endpoint certification semantics. Prefer adding explicit
fields validated by `config_models.py` over implicit naming conventions. After
configuration edits, inspect profile readiness and registry cross-reference
errors before changing runtime code.

## Verification

```bash
pytest -q tests/test_config_models.py tests/test_registry.py tests/test_policy.py
claude-brigade-doctor
```

## Child DOX Index

No child directories. YAML files are the local policy modules.
