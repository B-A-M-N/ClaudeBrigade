# Runtime Configuration

## Purpose

This subtree is the checked-in source configuration for models, providers,
profiles, workflows, and the optional advisory fastpath.

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
  endpoints, and discovery settings.
- `fastpath.yaml` is advisory/read-only. It cannot choose physical endpoints,
  lower a deterministic tier, mutate files, or authorize completion.
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
