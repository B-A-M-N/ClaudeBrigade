from __future__ import annotations

import asyncio
from time import monotonic

import pytest

from enhanced_router.provider_admission import AdmissionTimeout, ProviderAdmissionManager, ProviderLimits


@pytest.mark.asyncio
async def test_provider_agent_admission_is_fifo() -> None:
    manager = ProviderAdmissionManager({"freeinference": ProviderLimits(max_active_agents=1, max_queued_agents=2)})
    await manager.reserve_agent("freeinference", "first")
    second = asyncio.create_task(manager.reserve_agent("freeinference", "second"))
    third = asyncio.create_task(manager.reserve_agent("freeinference", "third"))
    await asyncio.sleep(0)
    assert manager.snapshot("freeinference")["queued_agents"] == 2
    await manager.release_agent("first")
    await asyncio.wait_for(second, 1)
    assert not third.done()
    await manager.release_agent("second")
    await asyncio.wait_for(third, 1)


@pytest.mark.asyncio
async def test_cancelled_request_releases_queue_entry() -> None:
    manager = ProviderAdmissionManager({"freeinference": ProviderLimits(max_inflight_requests=1)})
    await manager.acquire_request("freeinference", "first")
    waiting = asyncio.create_task(manager.acquire_request("freeinference", "second", deadline=0.0))
    with pytest.raises(AdmissionTimeout):
        await waiting
    assert manager.snapshot("freeinference")["queued_requests"] == 0


@pytest.mark.asyncio
async def test_freeinference_shared_request_limit_is_four() -> None:
    """Direct and LiteLLM routes share one provider-wide request pool."""
    manager = ProviderAdmissionManager(
        {"freeinference": ProviderLimits(max_concurrency=4)}
    )

    for request_id in ("direct-1", "direct-2", "litellm-1", "controller-1"):
        await manager.acquire_request("freeinference", request_id)

    fifth = asyncio.create_task(
        manager.acquire_request("freeinference", "controller-2", deadline=monotonic() + 0.05)
    )
    await asyncio.sleep(0)
    snapshot = manager.snapshot("freeinference")
    assert snapshot["active_requests"] == 4
    assert snapshot["queued_requests"] == 1
    assert snapshot["limits"]["max_inflight_requests"] == 4  # type: ignore[index]

    await manager.release_request("direct-1")
    await asyncio.wait_for(fifth, 1)
    assert manager.snapshot("freeinference")["active_requests"] == 4


@pytest.mark.asyncio
async def test_queued_provider_does_not_block_other_provider() -> None:
    manager = ProviderAdmissionManager({
        "a": ProviderLimits(max_concurrency=1, queue_timeout_seconds=1),
        "b": ProviderLimits(max_concurrency=1, queue_timeout_seconds=1),
    })

    await manager.acquire_request("a", "a1")
    queued = asyncio.create_task(manager.acquire_request("a", "a2"))
    await asyncio.sleep(0)

    await asyncio.wait_for(
        manager.acquire_request("b", "b1"),
        timeout=0.05,
    )

    assert not queued.done()
    assert manager.snapshot("b")["active_requests"] == 1

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued


@pytest.mark.asyncio
async def test_managed_group_admission_is_atomic_and_releases_all_candidates() -> None:
    manager = ProviderAdmissionManager({
        "freeinference": ProviderLimits(max_concurrency=1, queue_timeout_seconds=1),
        "openrouter": ProviderLimits(max_concurrency=1, queue_timeout_seconds=1),
    })

    await manager.acquire_request("freeinference", "fi-blocker")
    with pytest.raises(AdmissionTimeout):
        await manager.acquire_request_group(
            ("freeinference", "openrouter"), "group-1", deadline=monotonic() + 0.05,
        )
    assert manager.snapshot("openrouter")["active_requests"] == 0


@pytest.mark.asyncio
async def test_managed_group_waiter_is_not_starved_by_single_provider_queue() -> None:
    manager = ProviderAdmissionManager({
        "a": ProviderLimits(max_concurrency=1, queue_timeout_seconds=1),
        "b": ProviderLimits(max_concurrency=1, queue_timeout_seconds=1),
    })
    await manager.acquire_request("a", "a1")
    await manager.acquire_request("b", "b1")
    group = asyncio.create_task(
        manager.acquire_request_group(("a", "b"), "group", deadline=monotonic() + 1)
    )
    await asyncio.sleep(0)

    single = asyncio.create_task(
        manager.acquire_request("a", "a2", deadline=monotonic() + 1)
    )
    await asyncio.sleep(0)
    await manager.release_request("b1")
    await manager.release_request("a1")
    await asyncio.wait_for(group, 1)
    assert manager.snapshot("a")["active_requests"] == 1
    assert manager.snapshot("b")["active_requests"] == 1
    assert not single.done()

    await manager.release_request_group(("a", "b"), "group")
    await asyncio.wait_for(single, 1)

    await manager.release_request("fi-blocker")
    await manager.acquire_request_group(("freeinference", "openrouter"), "group-2")
    assert manager.snapshot("freeinference")["active_requests"] == 1
    assert manager.snapshot("openrouter")["active_requests"] == 1
    await manager.release_request_group(("freeinference", "openrouter"), "group-2")
    assert manager.snapshot("freeinference")["active_requests"] == 0
    assert manager.snapshot("openrouter")["active_requests"] == 0


@pytest.mark.asyncio
async def test_configure_updates_existing_provider_state() -> None:
    manager = ProviderAdmissionManager(
        {"freeinference": ProviderLimits(max_inflight_requests=1)}
    )
    await manager.acquire_request("freeinference", "first")
    waiting = asyncio.create_task(
        manager.acquire_request("freeinference", "second", deadline=monotonic() + 1)
    )
    await asyncio.sleep(0)

    manager.configure(
        "freeinference",
        ProviderLimits(max_inflight_requests=2),
    )
    await asyncio.wait_for(waiting, 1)
    assert manager.snapshot("freeinference")["active_requests"] == 2


def test_snapshots_are_json_safe() -> None:
    manager = ProviderAdmissionManager(
        {"freeinference": ProviderLimits(max_concurrency=4)}
    )
    snapshot = manager.snapshots()["freeinference"]
    assert snapshot["limits"] == {
        "max_concurrency": 4,
        "max_active_agents": 4,
        "max_inflight_requests": 4,
        "max_queued_agents": 8,
        "queue_timeout_seconds": 20.0,
    }
    assert snapshot["queued_group_requests"] == 0


def test_freeinference_concurrency_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from enhanced_router.provider_admission import apply_concurrency_env_override

    monkeypatch.setenv("FREEINFERENCE_MAX_CONCURRENCY", "2")
    limits = apply_concurrency_env_override(
        ProviderLimits(max_concurrency=4),
        "FREEINFERENCE_MAX_CONCURRENCY",
    )

    assert limits.max_concurrency == 2
    assert limits.max_active_agents == 2
    assert limits.max_inflight_requests == 2
