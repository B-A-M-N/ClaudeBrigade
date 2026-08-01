"""Tests for router-owned bounded sidecar execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from enhanced_router.sidecar_executor import SidecarExecutor
from enhanced_router.state import RouteState, WorkflowStateError


def _setup_sidecar(state: RouteState, monkeypatch: pytest.MonkeyPatch) -> dict:
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    state.set_role_route("r1", "ep-1", "recon", "model-a", "manual")
    state.initialize_workflow_phases(
        "r1", "ep-1", [{
            "id": "recon", "roles": ["recon"], "execution_kind": "sidecar_call",
            "max_attempts": 2,
        }],
    )
    state.start_phase("r1", "ep-1", "recon")

    class FakeRegistry:
        providers: dict = {}

        @staticmethod
        def get_model(model_id: str) -> SimpleNamespace:
            return SimpleNamespace(provider_id=None)

    import enhanced_router.registry as registry_module
    monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())
    action = next(item for item in state.get_runnable_actions("r1", "ep-1")
                  if item["action_kind"] == "sidecar_call")
    return state.claim_runnable_action("r1", "ep-1", action["action_id"])


@pytest.mark.asyncio
async def test_sidecar_completes_and_persists_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = RouteState(tmp_path / "state.db")
    claim = _setup_sidecar(state, monkeypatch)
    executor = SidecarExecutor(state)

    async def fake_request(**kwargs):
        return ({"content": [{"type": "text", "text": '{"verdict":"pass"}'}]}, 200, {
            "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
        })

    executor._request = fake_request  # type: ignore[method-assign]
    execution = await executor.invoke(
        run_id="r1", epoch_id="ep-1", action_id=claim["action_id"],
        claim_token=claim["claim_token"], packet={"task": "review"},
    )
    final = await executor.wait(execution["execution_id"])
    assert final is not None
    assert final["status"] == "completed"
    assert final["result_json"] == '{"verdict":"pass"}'
    assert [event["event_type"] for event in state.get_execution_events(
        "r1", "ep-1", execution["execution_id"],
    )] == ["started", "running", "completed"]
    await executor.shutdown()


@pytest.mark.asyncio
async def test_sidecar_retry_uses_bounded_original_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state = RouteState(tmp_path / "state.db")
    claim = _setup_sidecar(state, monkeypatch)
    executor = SidecarExecutor(state)
    attempts = 0

    async def fake_request(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ({"error": "temporary"}, 503, {})
        return ({"content": [{"type": "text", "text": '{"ok":true}'}]}, 200, {})

    executor._request = fake_request  # type: ignore[method-assign]
    first = await executor.invoke(
        run_id="r1", epoch_id="ep-1", action_id=claim["action_id"],
        claim_token=claim["claim_token"], packet={"task": "retry-me"},
    )
    first_final = await executor.wait(first["execution_id"])
    assert first_final is not None
    assert first_final["status"] == "failed"
    second = await executor.retry(first["execution_id"])
    second_final = await executor.wait(second["execution_id"])
    assert second_final is not None
    assert second_final["status"] == "completed"
    await executor.shutdown()


@pytest.mark.asyncio
async def test_detached_sidecar_persists_lifecycle_and_is_cancellable(tmp_path: Path):
    state = RouteState(tmp_path / "state.db")
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    executor = SidecarExecutor(state)

    async def fake_runner() -> dict:
        await asyncio.sleep(10)
        return {"decision": "pass"}

    execution = await executor.invoke_detached(
        run_id="r1",
        epoch_id="ep-1",
        execution_id="fp-test",
        phase_id="fastpath:verify",
        role="fastpath",
        model_id="diffusiongemma",
        provider_id="freeinference",
        packet={"verification_id": "v1"},
        timeout_seconds=30,
        runner=fake_runner,
    )
    assert execution["execution_kind"] == "sidecar_call"
    assert execution["claude_agent_id"] == "sidecar:fp-test"
    cancelled = await executor.cancel("fp-test")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    await executor.wait("fp-test")
    final = state.get_agent_execution("fp-test")
    assert final is not None
    assert final["status"] == "cancelled"
    assert [event["event_type"] for event in state.get_execution_events(
        "r1", "ep-1", "fp-test",
    )] == ["started", "cancelled"]
    await executor.shutdown()


@pytest.mark.asyncio
async def test_detached_bypass_is_not_recorded_as_a_valid_success(tmp_path: Path):
    """A validation_status='bypassed' dict result must not look identical to
    a real success in the ledger -- previously any dict result got
    schema_valid=True, evidence_valid=True, quality_score=1.0 unconditionally,
    so a policy bypass (e.g. app.py's failure_policy='bypass' path) was
    indistinguishable from a genuinely accepted proposal."""
    state = RouteState(tmp_path / "state.db")
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    executor = SidecarExecutor(state)

    async def bypass_runner() -> dict:
        return {
            "validation_status": "bypassed",
            "validation_reason": "endpoint not certified",
            "confidence": 0.0,
        }

    await executor.invoke_detached(
        run_id="r1", epoch_id="ep-1", execution_id="fp-bypass",
        phase_id="fastpath:route", role="fastpath", model_id="diffusiongemma",
        provider_id="freeinference", packet={"proposal_id": "p1"},
        timeout_seconds=5, runner=bypass_runner,
    )
    await executor.wait("fp-bypass")
    bypassed = state.get_agent_execution("fp-bypass")
    assert bypassed is not None
    assert bypassed["status"] == "completed"
    assert bypassed["schema_valid"] == 0
    assert bypassed["evidence_valid"] == 0
    assert bypassed["quality_score"] == 0.0

    async def accepted_runner() -> dict:
        return {"validation_status": "accepted_for_controller_review", "routes": {}}

    await executor.invoke_detached(
        run_id="r1", epoch_id="ep-1", execution_id="fp-accepted",
        phase_id="fastpath:route", role="fastpath", model_id="diffusiongemma",
        provider_id="freeinference", packet={"proposal_id": "p2"},
        timeout_seconds=5, runner=accepted_runner,
    )
    await executor.wait("fp-accepted")
    accepted = state.get_agent_execution("fp-accepted")
    assert accepted is not None
    assert accepted["status"] == "completed"
    assert accepted["schema_valid"] == 1
    assert accepted["evidence_valid"] == 1
    assert accepted["quality_score"] == 1.0
    await executor.shutdown()


@pytest.mark.asyncio
async def test_detached_fastpath_failure_can_retry_while_router_is_alive(tmp_path: Path):
    state = RouteState(tmp_path / "state.db")
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    executor = SidecarExecutor(state)
    attempts = 0

    async def flaky_runner() -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary provider failure")
        return {"decision": "escalate"}

    first = await executor.invoke_detached(
        run_id="r1", epoch_id="ep-1", execution_id="fp-failure",
        phase_id="fastpath:verify", role="fastpath", model_id="diffusiongemma",
        provider_id="freeinference", packet={"verification_id": "v1"},
        timeout_seconds=30, runner=flaky_runner,
    )
    await executor.wait(first["execution_id"])
    assert state.get_agent_execution(first["execution_id"])["status"] == "failed"

    retry = await executor.retry(first["execution_id"])
    await executor.wait(retry["execution_id"])
    assert state.get_agent_execution(retry["execution_id"])["status"] == "completed"
    assert attempts == 2
    with pytest.raises(WorkflowStateError, match="not retryable"):
        await executor.retry(retry["execution_id"])
    await executor.shutdown()


@pytest.mark.asyncio
async def test_detached_fastpath_retry_reconstructs_after_executor_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    state = RouteState(tmp_path / "state.db")
    state.create_run("r1")
    state.create_epoch("r1", "ep-1", "normal", "hybrid")
    first_executor = SidecarExecutor(state)

    async def failed_runner() -> dict:
        raise RuntimeError("provider unavailable")

    first = await first_executor.invoke_detached(
        run_id="r1", epoch_id="ep-1", execution_id="fp-restart",
        phase_id="fastpath:verify", role="fastpath", model_id="diffusiongemma",
        provider_id="freeinference", packet={"verification_id": "v-restart"},
        timeout_seconds=30, runner=failed_runner,
    )
    await first_executor.wait(first["execution_id"])
    await first_executor.shutdown()

    import enhanced_router.app as app_module
    import enhanced_router.registry as registry_module

    async def recovered_runner(packet: dict) -> dict:
        assert packet["verification_id"] == "v-restart"
        return {"decision": "escalate", "validation_status": "advisory"}

    class FakeFastpathConfig:
        timeout_seconds = 30

    monkeypatch.setattr(app_module, "_run_fastpath_verify", recovered_runner)
    monkeypatch.setattr(
        registry_module,
        "get_registry",
        lambda: SimpleNamespace(fastpath=FakeFastpathConfig()),
    )

    restarted_executor = SidecarExecutor(state)
    retry = await restarted_executor.retry(first["execution_id"])
    await restarted_executor.wait(retry["execution_id"])
    final = state.get_agent_execution(retry["execution_id"])
    assert final is not None
    assert final["status"] == "completed"
    assert final["parent_execution_id"] == first["execution_id"]
    assert final["retry_count"] == 1
    await restarted_executor.shutdown()
