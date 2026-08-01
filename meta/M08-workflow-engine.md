# M08: Workflow Engine

## Status
COMPLETE — all state machine, phase lifecycle, dependency validation, and hook integration done

## Implementation
- `config/workflows.yaml`: Four workflow tiers with full phase specs
- `config_models.py`: `WorkflowSpec`, `WorkflowPhase`, `TierPolicy` Pydantic models
- `state.py`: `workflow_phases` SQLite table, SCHEMA_VERSION 11, `_migrate_v11`
- `state.py`: `initialize_workflow_phases`, `get_workflow_phases`, `get_active_phase`, `start_phase`, `complete_phase`, `skip_phase`, `validate_phase_transition`
- `state.py`: `ensure_workflow_phases` auto-calls from `session_start.py` and `user_prompt_submit.py` hooks
- `registry.py`: `get_workflow()` method, typed `WorkflowSpec` objects in `load_workflows()`
- Tests: 18+ tests across phase lifecycle, dependency chains, failed-dep blocking, skipped-dep allowing, empty phase list, nonexistent phase errors, migration idempotency, multi-phase E2E lifecycle

## Not implemented (out of scope for current milestone)
- Agent orchestrator (reads WorkflowSpec and spawns subagents per phase)
- Auto-escalation between tiers based on workspace diff signals