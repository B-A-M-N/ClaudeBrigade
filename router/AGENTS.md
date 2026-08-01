# Router Runtime

## Purpose

This subtree owns the installable Python package used by the loopback
ClaudeBrigade router. It contains the runtime implementation and packaging
metadata; it does not own user-editable YAML or Claude Code hook behavior.

## Ownership

Runtime routing and orchestration implementation.

## Local Contracts

- The import package is `enhanced_router`, even though its source directory is
  `router/enhanced_router/`.
- Public model requests enter through the FastAPI app and must preserve
  identity-first routing, immutable bindings, provider admission, and the
  authenticated MCP boundary described in `../ARCHITECTURE.md`.
- `enhanced_router.egg-info/` is generated packaging metadata. Update it only
  through the project’s packaging/install workflow.
- Do not place provider credentials, tokens, or runtime SQLite state here.

## Work Guidance

Read the architecture handoff before changing routing, endpoint selection,
workflow scheduling, provider admission, LiteLLM lifecycle, or shadow
integration. Keep HTTP adapters thin and preserve the shared outbound
execution semantics. Add a focused regression test with every contract change.

## Verification

```bash
pyright router/enhanced_router/
pytest -q tests/test_router.py tests/test_registry.py
git diff --check
```

## Child DOX Index

| Path | Purpose |
|------|---------|
| `enhanced_router/` | FastAPI router, state, registry, backends, MCP, and orchestration runtime |
