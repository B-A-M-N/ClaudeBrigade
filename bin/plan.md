# ClaudeBrigade Implementation Plan

## Overview

Replace the LongCat-specific controller with a generic Brigade architecture backed by a YAML model/profile registry, SQLite route state, an MCP control server, gateway role-alias routing, LiteLLM provider integration, and a workflow engine.

## Profiles

### FreeClaudeOAI (development)

| Role | Model |
|------|-------|
| main controller | deepseek-v4-flash |
| planner | deepseek-v4-flash |
| fast | qwen3.6-35b |
| subagent (all) | qwen3.6-35b |

### ClaudeBrigade (target)

| Role | Model |
|------|-------|
| controller | deepseek-v4-pro (fallback: deepseek-v4-flash) |
| recon | qwen3.6-35b-a3b |
| implementer | qwen3.6-35b-a3b |
| test_engineer | qwen3.6-35b-a3b |
| secondary_reviewer | qwen3.6-35b-a3b |
| repairer | qwen3.6-35b-a3b |
| final_reviewer | deepseek-v4-pro |
| adjudicator | deepseek-v4-pro |
| verifier | deepseek-v4-pro |

## Milestones

### Milestone 1 — Generic role agents and stable aliases

Rename LongCat-specific agents to generic Brigade roles. Replace LongCat-specific descriptions with generic ones. Update agent JSON validation, hook role allowlists, and tests.

Tracking: `meta/M01-role-migration.md`

### Milestone 2 — Model and profile registry

YAML schemas for models, profiles, and role assignments. Pydantic models with cross-reference validation. Configuration hashing and deterministic recommendation.

Tracking: `meta/M02-registry.md`

### Milestone 3 — SQLite route state

State machine design. Schema and migrations. CRUD operations. Role-route updates with immutable agent binding. Run and epoch isolation.

Tracking: `meta/M03-route-state.md`

### Milestone 4 — MCP control server

FastMCP server with auth middleware. Tools: list_models, recommend_model, set_role_route, get_route_status, select_profile. Integration tests.

Tracking: `meta/M04-mcp-control.md`

### Milestone 5 — Gateway role-alias routing

Refactor app.py. Alias recognition and role resolution. Agent-ID binding. Internal header stripping. /v1/models updates. Routing tests.

Tracking: `meta/M05-gateway-routing.md`

### Milestone 6 — LiteLLM child service

Provider classes and compatibility certification. LiteLLM config generation. Process supervision. Health checks. Private authentication. Provider fixture harness.

Tracking: `meta/M06-litellm.md`

### Milestone 7 — Hook and evidence-ledger integration

Guard, audit, session, and completion hooks for the Brigade architecture. Role lifecycle tracking. Evidence-ledger output format.

Tracking: `meta/M07-hooks.md`

### Milestone 8 — Workflow engine

Workflow schema and phase semantics. Phase-state tracking. Transition validation. Dependency rules. Escalation and recovery mechanics. Workflow execution tests.

Tracking: `meta/M08-workflow-engine.md`

### Milestone 9 — Branding, migration, documentation, E2E

Rename deliverables. Migration guide. Documentation. End-to-end tests. SHA256SUMS update. Release packaging.

Tracking: `meta/M09-release.md`

## Progress

- [x] M01: Generic role agents and stable aliases — `meta/M01-role-migration.md`
- [x] M02: Model and profile registry — `meta/M02-registry.md`
- [x] M03: SQLite route state — `meta/M03-route-state.md`
- [x] M04: MCP control server — `meta/M04-mcp-control.md`
- [x] M05: Gateway role-alias routing — `meta/M05-gateway-routing.md`
- [x] M06: LiteLLM child service — `meta/M06-litellm.md`
- [x] M07: Hook and evidence-ledger integration — `meta/M07-hooks.md`
- [x] M08: Workflow engine — `meta/M08-workflow-engine.md`
- [x] M09: Branding, migration, documentation, E2E
