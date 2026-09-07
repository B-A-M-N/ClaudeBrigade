from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from enhanced_router.escalation_policy import EscalationInputs, evaluate_escalation
from enhanced_router.presentation import orchestration_snapshot
from enhanced_router.run_orchestration_state import resolve_effective_workflow
from enhanced_router.state import RouteState, WorkflowPhaseStateError


def test_empty_workflow_graph_is_not_reported_complete() -> None:
    class EmptyState:
        def get_workflow_phases(self, *_args: object) -> list[dict]:
            return []

        def get_agent_executions(self, *_args: object, **_kwargs: object) -> list[dict]:
            return []

        def get_work_packages(self, *_args: object) -> list[dict]:
            return []

        def get_requirement_coverage(self, *_args: object) -> dict:
            return {"complete": True, "covered": 0, "total": 0}

        def get_ambiguities(self, *_args: object) -> list[dict]:
            return []

        def get_escalation_state(self, *_args: object) -> dict:
            return {"epoch": {"status": "active"}}

    snapshot = orchestration_snapshot(EmptyState(), "run-1", "ep-1")
    assert snapshot["completion_state"]["project_complete"] is False
    assert "workflow phases have not been initialized" in snapshot["blockers"]


def test_forced_workflow_cannot_downgrade_committed_minimum_tier() -> None:
    assert resolve_effective_workflow("normal", "cross-cutting", None) == "cross-cutting"
    custom = type("Spec", (), {"tier": "high-risk"})()
    assert resolve_effective_workflow("custom-review", "high-risk", custom) == "custom-review"


def test_untiered_custom_workflow_cannot_bypass_committed_minimum() -> None:
    with pytest.raises(ValueError, match="does not declare a tier"):
        resolve_effective_workflow("custom-review", "cross-cutting", None)


def test_contract_requirement_coverage_is_durable(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "contracts.db")
    state.create_run("run-1", session_id="session-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")

    contract = state.publish_task_contract(
        "run-1", "ep-1", {"objective": "ship feature", "dod": ["tests pass"]}
    )
    requirement = state.add_requirement(
        "run-1", "ep-1", "The feature has a regression test", category="tests"
    )
    state.update_requirement(requirement["requirement_id"], status="satisfied")
    state.link_requirement_evidence(
        requirement["requirement_id"], evidence_kind="test", evidence_ref="pytest:test_feature", valid=True
    )

    fresh = RouteState(tmp_path / "contracts.db")
    coverage = fresh.get_requirement_coverage("run-1", "ep-1")
    assert coverage["complete"] is True
    assert coverage["covered"] == 1
    assert fresh.get_task_contract("run-1", "ep-1")["contract_digest"] == contract["contract_digest"]


def test_requirement_coverage_requires_declared_evidence_kinds_and_audit(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "coverage-audit.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.publish_task_contract("run-1", "ep-1", {"objective": "ship", "dod": ["tests"]})
    requirement = state.add_requirement(
        "run-1",
        "ep-1",
        "The feature is tested",
        acceptance={"required_evidence": ["test", "audit"]},
    )
    state.update_requirement(requirement["requirement_id"], status="satisfied")
    state.link_requirement_evidence(
        requirement["requirement_id"], evidence_kind="test", evidence_ref="pytest:feature", valid=True,
    )
    incomplete = state.get_requirement_coverage("run-1", "ep-1")
    assert incomplete["complete"] is False
    assert incomplete["requirements"][0]["missing_evidence"] == ["audit"]

    state.link_requirement_evidence(
        requirement["requirement_id"], evidence_kind="audit", evidence_ref="audit:final", valid=True,
    )
    complete = state.get_requirement_coverage("run-1", "ep-1")
    assert complete["complete"] is True
    audit = state.record_coverage_audit("run-1", "ep-1", complete=True, auditor="controller")
    assert audit["complete"] == 1
    assert audit["contract_version"] == 1
    assert state.get_latest_coverage_audit("run-1", "ep-1")["audit_id"] == audit["audit_id"]


def test_unknown_requirement_evidence_does_not_count_as_coverage(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "unknown-evidence.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.publish_task_contract("run-1", "ep-1", {"objective": "ship", "dod": ["tests"]})
    requirement = state.add_requirement("run-1", "ep-1", "The feature is tested")
    state.update_requirement(requirement["requirement_id"], status="satisfied")
    state.link_requirement_evidence(
        requirement["requirement_id"],
        evidence_kind="test",
        evidence_ref="pytest:pending",
        valid=None,
    )

    coverage = state.get_requirement_coverage("run-1", "ep-1")
    assert coverage["complete"] is False
    assert coverage["missing_mandatory"] == [requirement["requirement_id"]]


def test_approved_contract_rejects_new_requirements(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "immutable-contract.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.publish_task_contract("run-1", "ep-1", {"objective": "ship", "dod": ["tests"]})
    state.add_requirement("run-1", "ep-1", "The feature is tested")
    state.approve_task_contract("run-1", "ep-1")

    with pytest.raises(ValueError, match="approved task contracts are immutable"):
        state.add_requirement("run-1", "ep-1", "A late requirement")


def test_work_packages_are_explicit_and_dependency_aware(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "packages.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "cross-cutting", "hybrid")
    state.publish_work_package("run-1", "ep-1", "implementation", "API package", package_id="pkg-api")
    state.publish_work_package(
        "run-1", "ep-1", "implementation", "storage package", package_id="pkg-storage",
        dependencies=["pkg-api"],
    )
    assert [p["package_id"] for p in state.get_ready_work_packages("run-1", "ep-1", "implementation")] == ["pkg-api"]
    state.update_work_package("pkg-api", status="completed")
    assert [p["package_id"] for p in state.get_ready_work_packages("run-1", "ep-1", "implementation")] == ["pkg-storage"]


def test_escalation_policy_escalates_incomplete_high_risk_work() -> None:
    decision = evaluate_escalation(EscalationInputs(
        current_tier="normal", changed_security_paths=True, missing_requirements=1,
    ))
    assert decision.should_escalate is True
    assert decision.to_tier == "high-risk"
    assert decision.reasons


def test_escalation_appends_compensating_phases_and_requires_ack(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "escalation.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    decision = evaluate_escalation(EscalationInputs(
        current_tier="normal", changed_security_paths=True,
    ))
    result = state.escalate_epoch("run-1", "ep-1", decision)
    assert result["epoch"]["mutation_paused"] == 1
    phases = state.get_workflow_phases("run-1", "ep-1")
    assert any(item["phase_template_id"] == "escalation-recon" for item in phases)
    assert any(item["phase_template_id"] == "repair-or-replan" for item in phases)
    with pytest.raises(ValueError, match="compensating review"):
        state.acknowledge_escalation("run-1", "ep-1")
    for template, actor in (
        ("escalation-recon", "recon"),
        ("retroactive-design-review", "adversary"),
        ("implementation-risk-review", "adversary"),
    ):
        phase = next(item for item in state.get_workflow_phases("run-1", "ep-1")
                     if item["phase_template_id"] == template)
        state.start_phase("run-1", "ep-1", phase["phase_id"], actor=actor)
        state.complete_phase(
            "run-1", "ep-1", phase["phase_id"],
            result_evidence='{"reviewed":true}',
        )
    assert state.acknowledge_escalation("run-1", "ep-1")["epoch"]["mutation_paused"] == 0


def test_phase_launch_policy_persists_in_new_and_existing_databases(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "launch-policy.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    phases = state.initialize_workflow_phases(
        "run-1", "ep-1", [{
            "id": "review",
            "roles": ["adversary"],
            "launch_policy": "minimum_first",
            "initial_fanout": 1,
            "maximum_replicas": 2,
        }],
    )
    assert phases[0]["launch_policy"] == "minimum_first"
    fresh = RouteState(tmp_path / "launch-policy.db")
    assert fresh.get_workflow_phases("run-1", "ep-1")[0]["launch_policy"] == "minimum_first"


def test_begin_task_preserves_configured_package_launch_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The authoritative task-start transaction must keep workflow fanout policy."""
    from enhanced_router.registry import ModelRegistry
    import enhanced_router.registry as registry_module

    registry = ModelRegistry(Path(__file__).resolve().parents[1] / "config")
    registry.load_models()
    registry.load_profiles()
    registry.load_workflows()
    registry.load_providers()
    monkeypatch.setattr(registry_module, "get_registry", lambda: registry)

    state = RouteState(tmp_path / "begin-task-launch-policy.db")
    result = state.begin_task(
        "run-1",
        session_id="session-1",
        cwd=str(tmp_path),
        workflow_id="cross-cutting",
        profile_id="hybrid",
        prompt="implement the requested change",
    )

    phases = {item["phase_id"]: item for item in result["phases"]}
    persisted = state.get_workflow_phases("run-1", result["epoch_id"])
    persisted_by_id = {item["phase_id"]: item for item in persisted}
    assert persisted_by_id["implementation"]["launch_policy"] == "all_packages"
    assert persisted_by_id["implementation"]["fanout_from"] == "work_packages"
    assert phases["implementation"]["status"] == "pending"


def test_fastpath_plan_updates_only_unbound_pending_phases(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "fastpath-plan.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "cross-cutting", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [
            {"id": "recon", "roles": ["recon"], "min_fanout": 1, "max_fanout": 1},
            {"id": "implementation", "roles": ["implementer"], "min_fanout": 1, "max_fanout": 1},
        ],
    )

    result = state.apply_fastpath_plan(
        "run-1", "ep-1", {
            "resolved_routes": {
                "recon": {"model": "qwen", "fanout": 2},
                "implementer": {"model": "deepseek", "fanout": 2},
            },
            "parallel_groups": [["recon", "implementer"]],
        },
    )

    assert result["applied"] is True
    assert {item["phase_id"] for item in result["changes"]} == {"recon", "implementation"}
    phases = {item["phase_id"]: item for item in state.get_workflow_phases("run-1", "ep-1")}
    assert phases["implementation"]["min_fanout"] == 2
    assert phases["implementation"]["maximum_replicas"] == 2
    assert phases["implementation"]["parallel_group"] == "fastpath-group-1"


def test_changeset_objectively_escalates_infrastructure_mutation(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "changeset-escalation.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.create_workspace(
        workspace_id="main-workspace",
        run_id="run-1",
        epoch_id="ep-1",
        kind="main",
        path=str(tmp_path),
        base_sha="base",
        dirty_patch_hash="clean",
    )
    state.create_changeset(
        changeset_id="changeset-router",
        execution_id="exec-1",
        workspace_id="main-workspace",
        base_sha="base",
        patch_digest="digest",
        changed_files=["router/enhanced_router/routing.py"],
        result={"validation": {"valid": True}},
        status="validated",
    )

    result = state.auto_escalate_after_changeset(
        "run-1", "ep-1", "changeset-router",
    )
    assert result["applied"] is True
    assert result["decision"]["to_tier"] == "cross-cutting"
    assert state.get_active_epoch("run-1")["mutation_paused"] == 1


def test_completion_gate_reclassifies_latest_integrated_changeset(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "completion-reclassify.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.create_workspace(
        workspace_id="main-workspace",
        run_id="run-1",
        epoch_id="ep-1",
        kind="main",
        path=str(tmp_path),
        base_sha="base",
        dirty_patch_hash="clean",
    )
    state.create_changeset(
        changeset_id="changeset-security",
        execution_id="exec-1",
        workspace_id="main-workspace",
        base_sha="base",
        patch_digest="digest",
        changed_files=["config/providers.env"],
        result={"validation": {"valid": True}},
        status="validated",
    )
    state.advance_canonical_workspace(
        workspace_id="main-workspace",
        expected_generation=0,
        expected_dirty_hash="clean",
        applied_changeset_id="changeset-security",
        new_base_sha="base-2",
        new_dirty_hash="dirty-2",
    )

    result = state.validate_completion(
        "run-1",
        "ep-1",
        {"Workflow-ID": "normal", "Workflow-Tier": "normal"},
    )
    assert result["valid"] is False
    assert "escalation" in result
    assert state.get_active_epoch("run-1")["workflow_id"] == "high-risk"


def test_verification_gate_reclassifies_before_start(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "verification-reclassify.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{"id": "verification", "actor": "controller", "required": True}],
    )
    state.create_workspace(
        workspace_id="main-workspace",
        run_id="run-1",
        epoch_id="ep-1",
        kind="main",
        path=str(tmp_path),
        base_sha="base",
        dirty_patch_hash="clean",
    )
    state.create_changeset(
        changeset_id="changeset-infra",
        execution_id="exec-1",
        workspace_id="main-workspace",
        base_sha="base",
        patch_digest="digest",
        changed_files=["router/enhanced_router/routing.py"],
        result={"validation": {"valid": True}},
        status="validated",
    )
    state.advance_canonical_workspace(
        workspace_id="main-workspace",
        expected_generation=0,
        expected_dirty_hash="clean",
        applied_changeset_id="changeset-infra",
        new_base_sha="base-2",
        new_dirty_hash="dirty-2",
    )

    with pytest.raises(WorkflowPhaseStateError, match="escalated before verification"):
        state.start_phase("run-1", "ep-1", "verification", actor="controller")
    assert state.get_active_epoch("run-1")["workflow_id"] == "cross-cutting"


def test_review_repair_cycles_are_persisted_as_phase_instances(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "phase-cycles.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "cross-cutting", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [
            {"id": "adversarial-review", "roles": ["adversary"], "max_attempts": 3},
            {"id": "repair", "roles": ["repairer"], "mutation": True,
             "depends_on": ["adversarial-review"]},
        ],
    )

    first = state.append_phase_instance(
        "run-1", "ep-1", "adversarial-review",
        dependencies=["repair"],
        supersedes_phase_id="adversarial-review",
        trigger_event="accepted-finding:f-1",
    )
    second = state.append_phase_instance(
        "run-1", "ep-1", "adversarial-review",
        dependencies=[first["phase_id"]],
        supersedes_phase_id=first["phase_id"],
        trigger_event="repair-completed:repair-1",
    )

    assert first["phase_id"] == "adversarial-review#1"
    assert second["phase_id"] == "adversarial-review#2"
    assert first["phase_template_id"] == "adversarial-review"
    assert second["iteration"] == 2
    assert second["supersedes_phase_id"] == first["phase_id"]
    assert second["trigger_event"] == "repair-completed:repair-1"
    assert second["dependencies_json"] == '["adversarial-review#1"]'
    assert second["status"] == "pending"


def test_execution_limits_terminalize_before_next_tool_call(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "execution-limits.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{"id": "implementation", "roles": ["implementer"], "turn_budget": 1}],
    )
    state.start_phase("run-1", "ep-1", "implementation")
    state.create_agent_execution(
        "exec-1", "run-1", "ep-1", "agent-1", "implementer", "model-a",
        phase_id="implementation",
        capability_snapshot={"can_mutate": False},
    )
    state.update_agent_execution("exec-1", tool_call_count=1)

    result = state.enforce_execution_limits(
        "exec-1", run_id="run-1", epoch_id="ep-1",
    )
    assert result["allowed"] is False
    assert result["error_class"] == "turn_budget_exhausted"
    assert state.get_agent_execution("exec-1")["status"] == "timeout"


def test_token_reconciliation_preserves_live_reservations(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "token-live.db")
    state.create_run("run-token-live")
    state.reserve_token_budget(
        reservation_id="tokens:live",
        run_id="run-token-live",
        epoch_id=None,
        action_id=None,
        execution_id=None,
        estimated_tokens=100,
    )

    assert state.reconcile_token_reservations("run-token-live") == 0
    active = state.get_token_reservations("run-token-live", active_only=True)
    assert active[0]["state"] == "reserved"


def test_native_result_acceptance_is_separate_from_lifecycle(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "quality.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.create_agent_execution(
        "exec-1", "run-1", "ep-1", "agent-1", "recon", "model-a",
        execution_kind="native_agent",
    )
    state.update_agent_execution("exec-1", status="completed", evidence_valid=True)
    execution = state.adjudicate_native_result(
        run_id="run-1", epoch_id="ep-1", execution_id="exec-1",
        disposition="accepted", quality_score=0.9,
    )
    assert execution is not None
    assert execution["accepted_by_controller"] == 1
    assert execution["result_disposition"] == "accepted"


def test_nontrivial_mutation_is_blocked_until_contract_approval(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "contract-gate.db")
    state.create_run("run-1", session_id="session-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.set_role_route("run-1", "ep-1", "implementer", "model-a", "manual")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{"id": "implementation", "roles": ["implementer"], "mutation": True}],
    )

    actions = state.get_runnable_actions("run-1", "ep-1")
    contract_action = next(item for item in actions if item["action_kind"] == "controller_contract")
    assert contract_action["required_action"].startswith("publish a complete objective")
    assert not any(item.get("role") == "implementer" for item in actions)

    state.claim_runnable_action("run-1", "ep-1", contract_action["action_id"])
    state.publish_task_contract(
        "run-1", "ep-1", {"objective": "ship the feature", "dod": ["tests pass"]}
    )
    requirement = state.add_requirement("run-1", "ep-1", "The feature is implemented")
    approved = state.approve_task_contract("run-1", "ep-1")
    assert approved["status"] == "approved"
    assert state.get_task_contract("run-1", "ep-1")["status"] == "approved"
    assert requirement["status"] == "open"


def test_unadjudicated_native_result_cannot_close_phase(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "native-quality.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{"id": "recon", "roles": ["recon"]}],
    )
    state.start_phase("run-1", "ep-1", "recon")
    state.create_agent_execution(
        "exec-1", "run-1", "ep-1", "agent-1", "recon", "model-a", phase_id="recon",
    )
    state.update_agent_execution("exec-1", status="completed", evidence_valid=True)
    assert state.complete_phase_if_ready("run-1", "ep-1", "recon")["status"] == "active"
    accepted = state.adjudicate_native_result(
        run_id="run-1", epoch_id="ep-1", execution_id="exec-1",
        disposition="accepted", quality_score=0.9,
    )
    assert accepted is not None
    assert state.get_workflow_phases("run-1", "ep-1")[0]["status"] == "completed"


def test_all_packages_completion_requires_accepted_evidence_for_each_package(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "all-packages.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "cross-cutting", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{
            "id": "implementation",
            "roles": ["implementer"],
            "mutation": False,
            "completion_mode": "all_packages",
            "min_fanout": 1,
            "max_fanout": 2,
        }],
    )
    state.publish_work_package(
        "run-1", "ep-1", "implementation", "API package", package_id="pkg-api",
    )
    state.publish_work_package(
        "run-1", "ep-1", "implementation", "Storage package", package_id="pkg-storage",
    )
    state.start_phase("run-1", "ep-1", "implementation", actor="implementer")

    for execution_id, package_id in (("exec-api", "pkg-api"), ("exec-storage", "pkg-storage")):
        state.create_agent_execution(
            execution_id, "run-1", "ep-1", execution_id, "implementer", "model-a",
            phase_id="implementation", package_id=package_id,
            capability_snapshot={"can_mutate": False},
        )

    state.update_agent_execution("exec-api", status="completed", evidence_valid=True)
    state.adjudicate_native_result(
        run_id="run-1", epoch_id="ep-1", execution_id="exec-api",
        disposition="accepted", quality_score=0.9,
    )
    assert state.get_workflow_phases("run-1", "ep-1")[0]["status"] == "active"

    state.update_agent_execution("exec-storage", status="completed", evidence_valid=True)
    state.adjudicate_native_result(
        run_id="run-1", epoch_id="ep-1", execution_id="exec-storage",
        disposition="accepted", quality_score=0.9,
    )
    assert state.get_workflow_phases("run-1", "ep-1")[0]["status"] == "completed"


def test_mutating_package_requires_approved_contract_requirement_and_scope(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "package-contract.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{"id": "implementation", "mutation": True}],
    )
    state.publish_task_contract("run-1", "ep-1", {"objective": "ship", "dod": ["tests"]})
    requirement = state.add_requirement("run-1", "ep-1", "Implement the feature")
    state.approve_task_contract("run-1", "ep-1")
    with pytest.raises(ValueError, match="at least one task requirement"):
        state.publish_work_package(
            "run-1", "ep-1", "implementation", "missing requirement", path_scope=["src/**"],
        )
    with pytest.raises(ValueError, match="unknown dependency"):
        state.publish_work_package(
            "run-1", "ep-1", "implementation", "bad dependency",
            path_scope=["src/**"], requirement_ids=[requirement["requirement_id"]],
            dependencies=["missing"], acceptance=["feature behavior passes"],
            required_tests=["pytest tests/test_feature.py"],
        )
    package = state.publish_work_package(
        "run-1", "ep-1", "implementation", "valid package", path_scope=["src/**"],
        requirement_ids=[requirement["requirement_id"]],
        acceptance=["feature behavior passes"],
        required_tests=["pytest tests/test_feature.py"],
    )
    assert package["status"] == "ready"


def test_mutating_package_requires_acceptance_and_tests(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "package-evidence.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{"id": "implementation", "mutation": True}],
    )
    state.publish_task_contract("run-1", "ep-1", {"objective": "ship"})
    requirement = state.add_requirement("run-1", "ep-1", "Implement the feature")
    state.approve_task_contract("run-1", "ep-1")
    with pytest.raises(ValueError, match="acceptance criterion"):
        state.publish_work_package(
            "run-1", "ep-1", "implementation", "missing acceptance",
            path_scope=["src/**"], requirement_ids=[requirement["requirement_id"]],
            required_tests=["pytest tests/test_feature.py"],
        )
    with pytest.raises(ValueError, match="required test"):
        state.publish_work_package(
            "run-1", "ep-1", "implementation", "missing test",
            path_scope=["src/**"], requirement_ids=[requirement["requirement_id"]],
            acceptance=["feature behavior passes"],
        )


def test_controller_package_plan_cannot_complete_without_downstream_packages(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "package-plan.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [
            {"id": "package-plan", "actor": "controller", "produces": "work_packages"},
            {"id": "implementation", "mutation": True, "depends_on": ["package-plan"]},
        ],
    )
    state.publish_task_contract("run-1", "ep-1", {"objective": "ship"})
    requirement = state.add_requirement("run-1", "ep-1", "Implement the feature")
    state.approve_task_contract("run-1", "ep-1")
    state.start_phase("run-1", "ep-1", "package-plan", actor="controller")
    with pytest.raises(WorkflowPhaseStateError, match="downstream work package"):
        state.complete_phase("run-1", "ep-1", "package-plan", result_evidence="plan recorded")
    state.publish_work_package(
        "run-1", "ep-1", "implementation", "implement feature",
        path_scope=["src/**"], requirement_ids=[requirement["requirement_id"]],
        acceptance=["feature behavior passes"],
        required_tests=["pytest tests/test_feature.py"],
    )
    assert state.complete_phase(
        "run-1", "ep-1", "package-plan", result_evidence="plan recorded",
    )["status"] == "completed"


def test_controller_package_plan_is_runnable_without_controller_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller planning is a scheduler action, not a worker route."""
    state = RouteState(tmp_path / "package-plan-action.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "trivial", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [
            {"id": "package-plan", "actor": "controller", "produces": "work_packages"},
        ],
    )

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

    action = next(
        item for item in state.get_runnable_actions("run-1", "ep-1")
        if item["phase_id"] == "package-plan"
    )
    assert action["action_kind"] == "controller_action"
    assert action["requires_main_controller"] is True
    assert action["model_id"] == "controller"


def test_finding_mutations_are_scoped_inside_the_update(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "finding-scope.db")
    state.create_run("run-1")
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.create_run("run-2")
    state.create_epoch("run-2", "ep-2", "normal", "hybrid")
    state.create_finding("finding-1", "run-1", "ep-1", "real issue")

    assert state.adjudicate_finding(
        "finding-1", "accepted", run_id="run-2", epoch_id="ep-2",
    ) is None
    assert state.resolve_finding(
        "finding-1", "verified", run_id="run-2", epoch_id="ep-2",
    ) is None
    finding = state.get_finding("finding-1")
    assert finding is not None
    assert finding["disposition"] == "pending"
    assert finding["verification_status"] == "pending"


def test_controller_action_completion_releases_token_reservation(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "controller-token-release.db")
    state.create_run("run-1", token_budget=8_192)
    state.create_epoch("run-1", "ep-1", "trivial", "hybrid")
    state.initialize_workflow_phases(
        "run-1", "ep-1", [{"id": "planning", "actor": "controller"}],
    )
    action = next(
        item for item in state.get_runnable_actions("run-1", "ep-1")
        if item["phase_id"] == "planning"
    )
    state.claim_runnable_action("run-1", "ep-1", action["action_id"])
    state.complete_phase("run-1", "ep-1", "planning", result_evidence="done")
    state.finish_controller_actions_for_phase("run-1", "ep-1", "planning")

    assert state.get_token_reservations("run-1", active_only=True) == []
