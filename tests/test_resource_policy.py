from __future__ import annotations

from pathlib import Path

import pytest

from enhanced_router.state import RouteState
from enhanced_router.state_errors import WorkflowStateError


def test_run_resource_policy_counts_native_and_coprocessor_work(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "resources.db")
    state.create_run(
        "run-1",
        resource_policy={
            "max_active_native_agents": 1,
            "max_active_mutators": 1,
            "max_active_reviewers": 1,
            "max_active_coprocessors": 1,
            "max_active_worktrees": 1,
        },
    )
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")

    state.create_agent_execution(
        "exec-native",
        "run-1",
        "ep-1",
        "native-1",
        "implementer",
        "model-a",
        execution_kind="native_agent",
        capability_snapshot={"can_mutate": False},
    )
    capacity = state.get_run_resource_capacity("run-1", "ep-1")
    assert capacity["active"]["active_native_agents"] == 1
    assert capacity["remaining"]["active_native_agents"] == 0

    # A coprocessor is accounted independently from native workers, but still
    # consumes the run-wide coprocessor lane.
    state.create_agent_execution(
        "exec-coprocessor",
        "run-1",
        "ep-1",
        "coprocessor-1",
        "coprocessor",
        "model-b",
        execution_kind="coprocessor_call",
    )
    capacity = state.get_run_resource_capacity("run-1", "ep-1")
    assert capacity["active"]["active_coprocessors"] == 1
    assert capacity["remaining"]["active_coprocessors"] == 0
    assert capacity["remaining"]["active_native_agents"] == 0


def test_reserved_token_policy_is_enforced_before_claim_budget(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "reserved-tokens.db")
    state.create_run(
        "run-1",
        resource_policy={"max_reserved_tokens": 100},
    )

    first = state.reserve_token_budget(
        reservation_id="tokens-1",
        run_id="run-1",
        epoch_id="ep-1",
        action_id="action-1",
        execution_id="pending-1",
        estimated_tokens=80,
    )
    assert first["state"] == "reserved"

    second = state.reserve_token_budget(
        reservation_id="tokens-2",
        run_id="run-1",
        epoch_id="ep-1",
        action_id="action-2",
        execution_id="pending-2",
        estimated_tokens=30,
    )
    assert second["state"] == "unavailable"
    assert "reserved-token limit" in second["reason"]


def test_resource_policy_is_immutable_after_run_registration(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "immutable-policy.db")
    first = state.create_run(
        "run-1",
        resource_policy={"max_active_native_agents": 2},
    )
    second = state.create_run(
        "run-1",
        resource_policy={"max_active_native_agents": 99},
    )
    assert first["resource_policy"]["max_active_native_agents"] == 2
    assert second["resource_policy"]["max_active_native_agents"] == 2


def test_run_native_cap_applies_across_provider_routes(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "cross-provider-cap.db")
    state.create_run("run-1", resource_policy={"max_active_native_agents": 1})
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.create_agent_execution(
        "exec-provider-a", "run-1", "ep-1", "agent-a", "recon", "model-a",
        execution_kind="native_agent", provider_id="provider-a",
    )

    with pytest.raises(WorkflowStateError, match="native agent limit"):
        state.assert_run_resource_capacity(
            "run-1", "ep-1",
            {
                "action_kind": "native_agent",
                "role": "implementer",
                "model_id": "model-b",
                "provider_id": "provider-b",
            },
        )


def test_estimated_cost_budget_counts_terminal_spend_and_next_action(tmp_path: Path) -> None:
    state = RouteState(tmp_path / "estimated-cost.db")
    state.create_run("run-1", resource_policy={"max_estimated_cost": 1.0})
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.create_agent_execution(
        "exec-1", "run-1", "ep-1", "agent-1", "recon", "model-a",
        execution_kind="native_agent", capability_snapshot={"can_mutate": False},
    )
    state.update_agent_execution("exec-1", status="completed", estimated_cost=0.75)

    capacity = state.get_run_resource_capacity("run-1", "ep-1")
    assert capacity["active"]["estimated_cost"] == pytest.approx(0.75)
    assert capacity["remaining"]["estimated_cost"] == pytest.approx(0.25)

    with pytest.raises(WorkflowStateError, match="estimated cost"):
        state.assert_run_resource_capacity(
            "run-1", "ep-1",
            {
                "action_kind": "coprocessor_call",
                "role": "coprocessor",
                "model_id": "model-b",
                "estimated_cost": 0.30,
            },
        )
