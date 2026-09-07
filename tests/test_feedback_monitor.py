"""Automatic coprocessor checkpoint monitor tests."""

from __future__ import annotations

import io
import json
from pathlib import Path

from enhanced_router.config_models import FeedbackMonitorSpec
from enhanced_router.state import RouteState
from enhanced_router.feedback_service import _format_feedback, _semantic_trigger


def test_feedback_monitor_hot_path_skips_unwatched_batch(monkeypatch, tmp_path: Path):
    import feedback_monitor

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_BRIGADE_RUN_ID", "run-1")
    called = False

    def fail_request(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unwatched batches must not call the router")

    monkeypatch.setattr(feedback_monitor, "_request", fail_request)
    monkeypatch.setattr(feedback_monitor.sys, "stdin", io.StringIO(json.dumps({
        "hook_event_name": "PostToolBatch",
        "session_id": "session-1",
        "tool_name": "Read",
    })))

    assert feedback_monitor.main() == 0
    assert called is False


def test_feedback_monitor_emits_additional_context(monkeypatch, tmp_path: Path, capsys):
    import feedback_monitor

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_BRIGADE_RUN_ID", "run-1")
    session_dir = tmp_path / "claude-brigade" / "sessions" / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "active_epoch_id.txt").write_text("ep-1", encoding="utf-8")
    monkeypatch.setattr(
        feedback_monitor,
        "_request",
        lambda *args, **kwargs: {
            "status": "completed",
            "feedback": "Check the error path before continuing.",
        },
    )
    monkeypatch.setattr(feedback_monitor.sys, "stdin", io.StringIO(json.dumps({
        "hook_event_name": "PostToolBatch",
        "session_id": "session-1",
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/app.py"},
    })))

    assert feedback_monitor.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["hookEventName"] == "PostToolBatch"
    assert "error path" in output["hookSpecificOutput"]["additionalContext"]


def test_feedback_monitor_marks_mutation_as_post_edit_checkpoint():
    import feedback_monitor

    assert feedback_monitor._checkpoint_name({"tool_name": "Edit"}) == "post_edit_checkpoint"
    assert feedback_monitor._checkpoint_name({"tool_name": "Write"}) == "post_edit_checkpoint"
    assert feedback_monitor._checkpoint_name({"tool_name": "Bash"}) == "post_tool_batch"


def test_sentinel_feedback_is_neutral_and_non_authoritative():
    rendered = _format_feedback(
        {
            "alert": "escalate",
            "why": "A persistence file changed.",
            "suggested_next": "invoke_controller",
            "confidence": 0.86,
            "evidence_refs": ["diff"],
        },
        "post-edit-sentinel",
    )
    assert "NON-AUTHORITATIVE" in rendered
    assert "Authority: none" in rendered
    assert "Independence rule" in rendered
    assert "untrusted hypothesis" in rendered


def test_feedback_monitor_policy_defaults_are_bounded():
    policy = FeedbackMonitorSpec(enabled=True)
    assert policy.max_parallelism == 1
    assert policy.max_calls_per_execution == 3
    assert policy.wait_seconds == 6


def test_feedback_semantic_trigger_skips_observational_bash():
    assert _semantic_trigger(
        {"tool_input": {"command": "git status --short"}}, {"Bash"}
    ) is False
    assert _semantic_trigger(
        {"tool_input": {"command": "pytest -q"}}, {"Bash"}
    ) is True
    assert _semantic_trigger(
        {"tool_results": [{"exit_code": 1}]}, {"Bash"}
    ) is True


def test_feedback_checkpoint_claim_deduplicates_and_budgets(tmp_path: Path):
    state = RouteState(tmp_path / "state.db")
    state.create_run("run-1", session_id="session-1", cwd=str(tmp_path))
    first = state.claim_feedback_checkpoint(
        feedback_id="feedback-1",
        run_id="run-1",
        epoch_id="ep-1",
        execution_key="exec-1",
        claude_agent_id="agent-1",
        action_id="action-1",
        checkpoint="post_tool_batch",
        coprocessor_id="verification-reviewer",
        provider_id="freeinference",
        evidence_digest="digest-1",
        packet_digest="packet-1",
        cooldown_seconds=0,
        max_calls=2,
        max_parallelism=1,
    )
    assert first["decision"] == "claimed"

    duplicate = state.claim_feedback_checkpoint(
        feedback_id="feedback-2",
        run_id="run-1",
        epoch_id="ep-1",
        execution_key="exec-1",
        claude_agent_id="agent-1",
        action_id="action-1",
        checkpoint="post_tool_batch",
        coprocessor_id="verification-reviewer",
        provider_id="freeinference",
        evidence_digest="digest-1",
        packet_digest="packet-1",
        cooldown_seconds=0,
        max_calls=2,
        max_parallelism=1,
    )
    assert duplicate["decision"] == "in_flight"

    state.complete_feedback("feedback-1", status="completed", result={"decision": "escalate"})
    same_completed = state.claim_feedback_checkpoint(
        feedback_id="feedback-3",
        run_id="run-1",
        epoch_id="ep-1",
        execution_key="exec-1",
        claude_agent_id="agent-1",
        action_id="action-1",
        checkpoint="post_tool_batch",
        coprocessor_id="verification-reviewer",
        provider_id="freeinference",
        evidence_digest="digest-1",
        packet_digest="packet-1",
        cooldown_seconds=0,
        max_calls=2,
        max_parallelism=1,
    )
    assert same_completed["decision"] == "duplicate"
    assert same_completed["feedback"] == {"decision": "escalate"}


def test_feedback_results_are_ready_then_delivered_once(tmp_path: Path):
    state = RouteState(tmp_path / "state.db")
    state.create_run("run-1", session_id="session-1", cwd=str(tmp_path))
    state.claim_feedback_checkpoint(
        feedback_id="feedback-ready",
        run_id="run-1",
        epoch_id="ep-1",
        execution_key="exec-1",
        claude_agent_id="agent-1",
        action_id=None,
        checkpoint="post_tool_batch",
        coprocessor_id="reviewer",
        provider_id="provider-a",
        evidence_digest="digest-ready",
        packet_digest="packet-ready",
        cooldown_seconds=0,
        max_calls=2,
        max_parallelism=1,
        parent_execution_id="exec-1",
    )
    state.complete_feedback(
        "feedback-ready",
        status="completed",
        result={"decision": "continue"},
        feedback_text="advisory",
    )
    ready = state.get_ready_feedback(
        "run-1", "ep-1", execution_key="exec-1", parent_execution_id="exec-1",
    )
    assert ready is not None
    assert ready["delivery_status"] == "ready"
    delivered = state.mark_feedback_delivered("feedback-ready", consumer_turn_id="turn-1")
    assert delivered is not None
    assert delivered["delivery_status"] == "delivered"
    assert state.get_ready_feedback(
        "run-1", "ep-1", execution_key="exec-1", parent_execution_id="exec-1",
    ) is None


def test_feedback_stale_running_claim_is_reconciled(tmp_path: Path):
    state = RouteState(tmp_path / "state.db")
    state.create_run("run-1", session_id="session-1", cwd=str(tmp_path))
    state.claim_feedback_checkpoint(
        feedback_id="feedback-stale",
        run_id="run-1",
        epoch_id="ep-1",
        execution_key="exec-1",
        claude_agent_id=None,
        action_id=None,
        checkpoint="post_tool_batch",
        coprocessor_id="reviewer",
        provider_id="provider-a",
        evidence_digest="digest-stale",
        packet_digest="packet-stale",
        cooldown_seconds=0,
        max_calls=2,
        max_parallelism=1,
        lease_seconds=1,
    )
    conn = state._new_conn()
    try:
        conn.execute(
            "UPDATE feedback_checkpoints SET lease_expires_at='2000-01-01T00:00:00+00:00' "
            "WHERE feedback_id='feedback-stale'"
        )
        conn.commit()
    finally:
        conn.close()
    assert state.reconcile_feedback() == 1
    row = state.get_feedback("feedback-stale")
    assert row is not None
    assert row["status"] == "failed"
    assert row["delivery_status"] == "expired"


def test_feedback_adjudication_records_concrete_successful_route(tmp_path: Path):
    state = RouteState(tmp_path / "state.db")
    state.create_run("run-1", session_id="session-1", cwd=str(tmp_path))
    state.create_epoch("run-1", "ep-1", "normal", "hybrid")
    state.start_detached_coprocessor_execution(
        run_id="run-1",
        epoch_id="ep-1",
        execution_id="exec-feedback",
        phase_id="feedback:reviewer",
        role="coprocessor:reviewer",
        model_id="primary-model",
        provider_id="provider-a",
        packet={"workspace_digest": "digest-1"},
    )
    state.start_route_attempt(
        attempt_id="exec-feedback:route:1",
        run_id="run-1",
        epoch_id="ep-1",
        execution_id="exec-feedback",
        candidate_index=1,
        model_id="fallback-model",
        provider_id="provider-b",
        endpoint_id="endpoint-b",
        route_digest="route-b",
    )
    state.finish_route_attempt(
        "exec-feedback:route:1",
        status="succeeded",
        usage={"input_tokens": 11, "output_tokens": 7, "estimated_cost": 0.02},
    )
    state.update_agent_execution("exec-feedback", status="completed")
    state.claim_feedback_checkpoint(
        feedback_id="feedback-outcome",
        run_id="run-1",
        epoch_id="ep-1",
        execution_key="exec-feedback",
        claude_agent_id="sidecar:exec-feedback",
        action_id=None,
        checkpoint="post_tool_batch",
        coprocessor_id="reviewer",
        provider_id="provider-a",
        evidence_digest="evidence-1",
        packet_digest="packet-1",
        cooldown_seconds=0,
        max_calls=1,
        max_parallelism=1,
        parent_execution_id="exec-feedback",
        prompt_version="review-v2",
        schema_version="feedback-v1",
    )
    state.link_feedback_execution("feedback-outcome", "exec-feedback")
    state.complete_feedback(
        "feedback-outcome",
        status="completed",
        result={"decision": "repair"},
        feedback_text="repair",
    )
    outcome = state.adjudicate_feedback(
        "feedback-outcome",
        disposition="adopted",
        adopted_finding_ids=["finding-1"],
        resulting_action_ids=["action-1"],
        task_class="normal",
    )
    assert outcome is not None
    conn = state._new_conn()
    try:
        row = conn.execute(
            "SELECT model_id, provider_id, prompt_version, schema_version, "
            "input_tokens, output_tokens, disposition "
            "FROM coprocessor_outcomes WHERE feedback_id=?",
            ("feedback-outcome",),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert tuple(row) == (
        "fallback-model", "provider-b", "review-v2", "feedback-v1", 11, 7, "adopted",
    )
    metrics = state.get_coprocessor_outcome_metrics(
        "reviewer", task_class="normal", run_id="run-1",
    )
    assert metrics["evaluated_outcomes"] == 1
    assert metrics["adoption_rate"] == 1.0
