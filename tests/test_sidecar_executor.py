"""Tests for router-owned bounded sidecar execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from enhanced_router.sidecar_executor import SidecarExecutor
from enhanced_router.state import RouteState


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
    await asyncio.sleep(0.02)
    final = state.get_agent_execution(execution["execution_id"])
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
    await asyncio.sleep(0.02)
    assert state.get_agent_execution(first["execution_id"])["status"] == "failed"
    second = await executor.retry(first["execution_id"])
    await asyncio.sleep(0.02)
    assert state.get_agent_execution(second["execution_id"])["status"] == "completed"
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
    await executor.shutdown()
