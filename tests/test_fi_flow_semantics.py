"""Regression tests for the fi-flow policy layered on native Claude runtime."""

from __future__ import annotations

import json

from enhanced_router.config_models import WorkflowPhase
from enhanced_router.state import RouteState


def test_fi_flow_result_contract_rejects_blocked_verdict() -> None:
    phase = WorkflowPhase(
        id="critical-gate",
        sidecar_agent="glm-critical-gate",
        execution_kind="native_agent",
        roles=["adjudicator"],
        result_contract={
            "schema_id": "glm_critical_gate_v1",
            "success_verdicts": ["APPROVED"],
            "blocked_verdicts": ["BLOCKED"],
        },
    )
    assert phase.result_contract is not None
    assert phase.result_contract.success_verdicts == ["APPROVED"]


def test_canonical_mutation_reopens_fi_flow_signoffs(tmp_path) -> None:
    state = RouteState(tmp_path / "state.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "freeinference")
    state.register_main_workspace(
        workspace_id="main-1", run_id="run-1", epoch_id="ep-1",
        path=str(tmp_path), base_sha="a" * 40, dirty_patch_hash="b" * 64,
    )
    state.initialize_workflow_phases(
        "run-1", "ep-1", [
            {"id": "post-grounding", "roles": ["recon"], "required": True},
            {"id": "critical-gate", "roles": ["adjudicator"], "required": True},
        ],
    )
    state.start_phase("run-1", "ep-1", "post-grounding", actor="recon")
    state.complete_phase(
        "run-1", "ep-1", "post-grounding",
        result_evidence=json.dumps({"verdict": "PASS"}),
    )
    token = state.prepare_completion_token("run-1", "ep-1", "c" * 64)

    state.advance_canonical_workspace(
        workspace_id="main-1",
        expected_generation=0,
        expected_dirty_hash="b" * 64,
        applied_changeset_id="changeset-1",
        new_base_sha="d" * 40,
        new_dirty_hash="e" * 64,
    )

    phases = {item["phase_id"]: item for item in state.get_workflow_phases("run-1", "ep-1")}
    assert phases["post-grounding"]["status"] == "pending"
    assert phases["post-grounding"]["invalidation_reason"]
    consumed = state.consume_completion_token(
        "run-1", "ep-1", token["token"], "c" * 64, token["route_snapshot_sha256"],
    )
    assert consumed["valid"] is False
    assert "already consumed" in consumed["reason"]
