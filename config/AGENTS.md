# Runtime Configuration

## Purpose

This subtree is the checked-in source configuration for models, providers,
profiles, workflows, independent sidecars, and the optional advisory fastpath.

## Ownership

Operator-facing policy inputs. Runtime state, secrets, credentials, and
generated LiteLLM deployment files belong outside this subtree.

## Local Contracts

- `models.yaml` defines logical models, backends, capabilities, allowed roles,
  endpoint declarations, and certification metadata.
- `profiles.yaml` maps recon/implementer/adversary/repairer roles to model IDs;
  specialist launch identities are declared here when they are model-specific.
- `workflows.yaml` defines persisted phase contracts, dependencies, actors,
  execution kinds, fanout/attempt/quorum policy, and fallback behavior.
- `providers.yaml` defines admission limits, deadlines, retry policy, provider
  endpoints, and discovery settings. OpenRouter, Kilo/Crush, Cline,
  OpenCode Zen/Go, FreeTheAI, Requesty, FreeModel, and NVIDIA NIM are bundled.
- `fastpath.yaml` is advisory/read-only. It cannot choose physical endpoints,
  lower a deterministic tier, mutate files, or authorize completion.
- `sidecars.yaml` stores independent bounded specialist definitions. A sidecar
  selects its own logical `model_id`, endpoint, timeout, packet/output bounds,
  and prompt; it does not inherit the Claude Code role profile unless a
  workflow phase omits `sidecar` for compatibility.
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
