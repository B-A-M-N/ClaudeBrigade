# Integrations

## Purpose

This subtree contains separately usable provider/integration kits that extend
ClaudeBrigade without becoming part of the router’s SQLite authority.

## Ownership

Integration packaging, provider-specific synchronization, compatibility
probes, and local proxy configuration.

## Local Contracts

- Integrations must keep provider credentials local and out of Claude Code,
  source control, prompts, reports, and test fixtures.
- An integration may produce catalog/configuration artifacts, but route
  authority and run/workflow state remain in `router/enhanced_router/`.
- Each integration owns its own dependency lock, tests, and operational
  README; do not silently import its runtime into the main router.

## Work Guidance

Use bounded HTTP timeouts, deterministic output, explicit live-probe opt-in,
and clear failure reporting. Keep provider-specific assumptions inside the
integration boundary and expose stable artifacts to the router only through
documented configuration/catalog contracts.

## Verification

Run the nearest integration guide’s checks from the integration directory.

## Child DOX Index

| Path | Purpose |
|------|---------|
| `freeinference-litellm/` | Standalone FreeInference BYOK LiteLLM kit |
