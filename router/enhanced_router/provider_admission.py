"""Fair provider-scoped agent and request admission.

This is intentionally separate from transport code. A provider limit applies
across direct Anthropic-compatible and LiteLLM/OpenAI-compatible deployments
of the same provider.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from time import monotonic


REQUEST_LANES = frozenset({
    "controller", "critical", "verification", "worker", "feedback", "fastpath",
})
_LANE_PRIORITY = {
    "controller": 0,
    "critical": 1,
    "verification": 2,
    "worker": 3,
    "feedback": 4,
    "fastpath": 5,
}


class AdmissionTimeout(TimeoutError):
    """A call waited longer than its configured admission deadline."""


@dataclass(frozen=True)
class ProviderLimits:
    max_concurrency: int | None = None
    max_active_agents: int = 4
    max_inflight_requests: int = 4
    max_queued_agents: int = 8
    queue_timeout_seconds: float = 20.0
    controller_reserve: int = 0
    max_worker_concurrency: int | None = None
    priority_policy: str = "fifo"

    def __post_init__(self) -> None:
        if self.max_concurrency is not None:
            if self.max_concurrency < 1:
                raise ValueError("max_concurrency must be positive")
            object.__setattr__(self, "max_active_agents", self.max_concurrency)
            object.__setattr__(self, "max_inflight_requests", self.max_concurrency)
        if self.controller_reserve >= self.max_active_agents:
            raise ValueError("controller_reserve must be below max_active_agents")
        if self.max_worker_concurrency is None:
            object.__setattr__(
                self,
                "max_worker_concurrency",
                max(1, self.max_active_agents - self.controller_reserve),
            )
        if self.max_worker_concurrency is not None and self.max_worker_concurrency > self.max_active_agents:
            raise ValueError("max_worker_concurrency cannot exceed max_active_agents")


def apply_concurrency_env_override(
    limits: ProviderLimits,
    env_name: str | None,
    environ: Mapping[str, str] | None = None,
) -> ProviderLimits:
    """Apply an optional positive-integer override to both provider limits."""
    if not env_name:
        return limits
    source = environ if environ is not None else os.environ
    raw_override = source.get(env_name, "").strip()
    if not raw_override:
        return limits
    try:
        override = int(raw_override)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be a positive integer when set") from exc
    if override < 1:
        raise RuntimeError(f"{env_name} must be a positive integer when set")
    return replace(
        limits,
        max_concurrency=override,
        max_active_agents=override,
        max_inflight_requests=override,
        max_worker_concurrency=max(1, override - limits.controller_reserve),
    )


@dataclass
class _Waiter:
    item_id: str
    deadline: float
    future: asyncio.Future[None]
    lane: str = "worker"
    enqueued_at: float = 0.0


@dataclass
class _GroupWaiter:
    provider_ids: tuple[str, ...]
    item_id: str
    deadline: float
    future: asyncio.Future[None]
    lane: str = "worker"
    enqueued_at: float = 0.0


@dataclass
class _ProviderState:
    limits: ProviderLimits
    active_agents: set[str] = field(default_factory=set)
    active_requests: set[str] = field(default_factory=set)
    request_lanes: dict[str, str] = field(default_factory=dict)
    agent_queue: deque[_Waiter] = field(default_factory=deque)
    request_queue: deque[_Waiter] = field(default_factory=deque)
    circuit_state: str = "healthy"
    circuit_open_until: float | None = None
    circuit_failures: int = 0
    half_open_probe: str | None = None


class ProviderAdmissionManager:
    """FIFO admission manager with cancellation and dynamic resize."""

    def __init__(self, providers: dict[str, ProviderLimits] | None = None) -> None:
        self._providers = dict(providers or {})
        self._states: dict[str, _ProviderState] = {}
        self._group_queue: deque[_GroupWaiter] = deque()
        self._lock = asyncio.Lock()

    def configure(self, provider_id: str, limits: ProviderLimits) -> None:
        self._providers[provider_id] = limits
        state = self._states.get(provider_id)
        if state is not None:
            state.limits = limits
            self._pump_group_queue()
            self._pump(state, state.agent_queue, state.active_agents, limits.max_active_agents)
            self._pump(state, state.request_queue, state.active_requests, limits.max_inflight_requests)

    def _state(self, provider_id: str) -> _ProviderState:
        state = self._states.get(provider_id)
        if state is None:
            state = _ProviderState(self._providers.get(provider_id, ProviderLimits()))
            self._states[provider_id] = state
        return state

    async def reserve_agent(self, provider_id: str, call_id: str, deadline: float | None = None) -> None:
        state = self._state(provider_id)
        await self._acquire(
            state,
            state.agent_queue,
            state.active_agents,
            call_id,
            state.limits.max_active_agents,
            state.limits.max_queued_agents,
            deadline,
        )

    async def acquire_request(
        self,
        provider_id: str,
        request_id: str,
        deadline: float | None = None,
        *,
        lane: str = "worker",
    ) -> None:
        if lane not in REQUEST_LANES:
            raise ValueError(f"unknown provider request lane: {lane}")
        state = self._state(provider_id)
        await self._admit_circuit(state, request_id)
        try:
            await self._acquire(
                state,
                state.request_queue,
                state.active_requests,
                request_id,
                state.limits.max_inflight_requests,
                None,
                deadline,
                lane=lane,
            )
        except BaseException:
            async with self._lock:
                if state.half_open_probe == request_id:
                    state.half_open_probe = None
            raise

    async def acquire_request_group(
        self,
        provider_ids: tuple[str, ...] | list[str],
        request_id: str,
        deadline: float | None = None,
        *,
        lane: str = "worker",
    ) -> None:
        """Reserve one request slot from every possible group provider.

        LiteLLM does not expose the selected deployment before it dispatches
        a managed-group request.  Reserving the candidate providers together
        is deliberately conservative: it guarantees that a FreeInference
        deployment in a multi-provider group cannot exceed its configured
        limit.  The reservation is atomic and released from every provider;
        it never holds one provider while waiting on another.
        """
        ordered = tuple(sorted(set(provider_ids)))
        if lane not in REQUEST_LANES:
            raise ValueError(f"unknown provider request lane: {lane}")
        if not ordered:
            return
        if len(ordered) == 1:
            await self.acquire_request(ordered[0], request_id, deadline, lane=lane)
            return
        limits = [self._state(provider_id).limits for provider_id in ordered]
        end = deadline if deadline is not None else monotonic() + min(
            limit.queue_timeout_seconds for limit in limits
        )
        loop = asyncio.get_running_loop()
        waiter = _GroupWaiter(
            provider_ids=ordered,
            item_id=request_id,
            deadline=end,
            future=loop.create_future(),
            lane=lane,
            enqueued_at=monotonic(),
        )
        async with self._lock:
            self._group_queue.append(waiter)
            self._pump_group_queue()
        try:
            remaining = max(0.0, waiter.deadline - monotonic())
            await asyncio.wait_for(waiter.future, timeout=remaining)
        except asyncio.CancelledError:
            async with self._lock:
                try:
                    self._group_queue.remove(waiter)
                except ValueError:
                    pass
                self._pump_group_queue()
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            async with self._lock:
                try:
                    self._group_queue.remove(waiter)
                except ValueError:
                    pass
                self._pump_group_queue()
            raise AdmissionTimeout(
                f"managed provider-group admission timed out for {request_id}"
            ) from exc

    def _pump_group_queue(self) -> None:
        """Admit ready groups without holding partial provider capacity."""
        now = monotonic()
        for waiter in tuple(self._group_queue):
            if waiter.future.cancelled() or waiter.deadline <= now:
                self._group_queue.remove(waiter)
                if not waiter.future.done():
                    waiter.future.set_exception(
                        AdmissionTimeout(
                            f"managed provider-group admission timed out for {waiter.item_id}"
                        )
                    )
                continue
            states = [self._state(provider_id) for provider_id in waiter.provider_ids]
            ready = True
            for state in states:
                if state.circuit_state == "open":
                    if state.circuit_open_until is not None and now < state.circuit_open_until:
                        ready = False
                        break
                    state.circuit_state = "half-open"
                if state.circuit_state == "half-open" and state.half_open_probe not in {None, waiter.item_id}:
                    ready = False
                    break
                if not self._request_capacity_available(state, waiter.lane):
                    ready = False
                    break
            if not ready:
                continue
            self._group_queue.remove(waiter)
            for state in states:
                state.active_requests.add(waiter.item_id)
                state.request_lanes[waiter.item_id] = waiter.lane
                if state.circuit_state == "half-open":
                    state.half_open_probe = waiter.item_id
            if not waiter.future.done():
                waiter.future.set_result(None)

    async def _admit_circuit(self, state: _ProviderState, request_id: str) -> None:
        """Gate new upstream requests on provider-wide circuit state."""
        async with self._lock:
            now = monotonic()
            if state.circuit_state == "open":
                if state.circuit_open_until is not None and now < state.circuit_open_until:
                    raise AdmissionTimeout("provider circuit is open")
                state.circuit_state = "half-open"
            if state.circuit_state == "half-open":
                if state.half_open_probe not in {None, request_id}:
                    raise AdmissionTimeout("provider circuit is probing recovery")
                state.half_open_probe = request_id

    async def retry_allowed(
        self, provider_ids: tuple[str, ...] | list[str], request_id: str,
    ) -> bool:
        """Check current circuits before a retry without taking new capacity."""
        async with self._lock:
            for provider_id in sorted(set(provider_ids)):
                state = self._state(provider_id)
                now = monotonic()
                if state.circuit_state == "open":
                    if state.circuit_open_until is not None and now < state.circuit_open_until:
                        return False
                    state.circuit_state = "half-open"
                if state.circuit_state == "half-open" \
                        and state.half_open_probe not in {None, request_id}:
                    return False
            return True

    async def record_response(
        self,
        provider_id: str,
        request_id: str,
        status_code: int,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        """Update provider-wide health from an upstream response.

        Two bounded rate/unavailability responses open the circuit.  A single
        half-open probe is allowed after the delay and either closes or
        re-opens it.  The response status is intentionally provider-wide so
        direct Anthropic and LiteLLM deployments share the same backoff.
        """
        async with self._lock:
            state = self._state(provider_id)
            if status_code in {408, 429, 500, 502, 503, 504, 529}:
                self._record_circuit_failure(
                    state,
                    retry_after_seconds=retry_after_seconds,
                )
                return
            if 200 <= status_code < 300:
                state.circuit_failures = 0
                state.circuit_state = "healthy"
                state.circuit_open_until = None
                state.half_open_probe = None

    async def record_transport_failure(
        self,
        provider_ids: tuple[str, ...] | list[str],
        request_id: str,
    ) -> None:
        """Feed a connection/timeout failure into provider circuit state.

        A request can fail before an HTTP response exists (DNS failure,
        refused connection, TLS failure, read timeout, or remote protocol
        reset). Those failures are provider health signals too. For a managed
        group the physical provider is unknown, so all reserved candidates are
        conservatively marked; admission already reserved all of them.
        """
        del request_id  # retained for a stable call-site/correlation contract
        async with self._lock:
            for provider_id in sorted(set(provider_ids)):
                self._record_circuit_failure(self._state(provider_id))

    @staticmethod
    def _record_circuit_failure(
        state: _ProviderState,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        state.circuit_failures += 1
        if state.circuit_state == "half-open" or state.circuit_failures >= 2:
            delay = (
                retry_after_seconds
                if retry_after_seconds is not None
                else min(30.0, 2.0 ** state.circuit_failures)
            )
            state.circuit_state = "open"
            state.circuit_open_until = monotonic() + max(1.0, delay)
            state.half_open_probe = None
        else:
            state.circuit_state = "degraded"

    async def _acquire(
        self,
        state: _ProviderState,
        queue: deque[_Waiter],
        active: set[str],
        item_id: str,
        capacity: int,
        max_queue: int | None,
        deadline: float | None,
        *,
        lane: str = "agent",
    ) -> None:
        async with self._lock:
            if queue is state.request_queue and self._group_queue:
                self._pump_group_queue()
            if item_id in active:
                return
            if not queue and (
                (lane == "agent" and len(active) < capacity)
                or (lane != "agent" and self._request_capacity_available(state, lane))
            ):
                active.add(item_id)
                if queue is state.request_queue:
                    state.request_lanes[item_id] = lane
                return
            if max_queue is not None and len(queue) >= max_queue:
                raise AdmissionTimeout(f"provider queue is full for {item_id}")
            loop = asyncio.get_running_loop()
            waiter = _Waiter(
                item_id=item_id,
                deadline=deadline if deadline is not None else monotonic() + state.limits.queue_timeout_seconds,
                future=loop.create_future(),
                lane=lane,
                enqueued_at=monotonic(),
            )
            queue.append(waiter)

        try:
            remaining = max(0.0, waiter.deadline - monotonic())
            await asyncio.wait_for(waiter.future, timeout=remaining)
        except asyncio.CancelledError:
            async with self._lock:
                try:
                    queue.remove(waiter)
                except ValueError:
                    pass
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            async with self._lock:
                try:
                    queue.remove(waiter)
                except ValueError:
                    pass
            raise AdmissionTimeout(f"provider admission timed out for {item_id}") from exc

    @staticmethod
    def _request_capacity_available(state: _ProviderState, lane: str) -> bool:
        if len(state.active_requests) >= state.limits.max_inflight_requests:
            return False
        if lane in {"controller", "critical", "verification"}:
            return True
        worker_count = sum(
            current_lane not in {"controller", "critical", "verification"}
            for current_lane in state.request_lanes.values()
        )
        return worker_count < int(state.limits.max_worker_concurrency or state.limits.max_inflight_requests)

    def _pump(self, state: _ProviderState, queue: deque[_Waiter], active: set[str], capacity: int) -> None:
        while queue and len(active) < capacity:
            # Preserve FIFO within a lane, but do not let queued workers
            # consume the controller reserve or block a critical request.
            candidates = [
                candidate for candidate in queue
                if queue is not state.request_queue
                or self._request_capacity_available(state, candidate.lane)
            ]
            if not candidates:
                return
            if queue is state.request_queue and state.limits.priority_policy == "strict":
                # Priority is strict among currently eligible lanes, while a
                # five-second age bonus prevents a permanently queued worker
                # from starving under a sustained controller stream.
                now = monotonic()
                waiter = min(
                    candidates,
                    key=lambda candidate: (
                        _LANE_PRIORITY.get(candidate.lane, 99)
                        - (1 if now - candidate.enqueued_at >= 5.0 else 0),
                        candidate.enqueued_at,
                    ),
                )
            else:
                waiter = candidates[0]
            if waiter is None:
                return
            queue.remove(waiter)
            if waiter.future.cancelled() or waiter.deadline <= monotonic():
                if not waiter.future.done():
                    waiter.future.set_exception(AdmissionTimeout(f"provider admission timed out for {waiter.item_id}"))
                continue
            active.add(waiter.item_id)
            if queue is state.request_queue:
                state.request_lanes[waiter.item_id] = waiter.lane
            if not waiter.future.done():
                waiter.future.set_result(None)

    async def release_agent(self, call_id: str) -> None:
        async with self._lock:
            for state in self._states.values():
                if call_id in state.active_agents:
                    state.active_agents.remove(call_id)
                    self._pump(state, state.agent_queue, state.active_agents, state.limits.max_active_agents)
                    return

    async def release_request(self, request_id: str) -> None:
        async with self._lock:
            for state in self._states.values():
                if request_id in state.active_requests:
                    state.active_requests.remove(request_id)
                    state.request_lanes.pop(request_id, None)
                    if state.half_open_probe == request_id:
                        state.half_open_probe = None
                    self._pump_group_queue()
                    self._pump(state, state.request_queue, state.active_requests, state.limits.max_inflight_requests)
                    return

    async def release_request_group(
        self, provider_ids: tuple[str, ...] | list[str], request_id: str,
    ) -> None:
        """Release a conservative managed-group reservation from all providers."""
        ordered = tuple(sorted(set(provider_ids)))
        if len(ordered) <= 1:
            await self.release_request(request_id)
            return
        async with self._lock:
            for provider_id in ordered:
                state = self._states.get(provider_id)
                if state is None or request_id not in state.active_requests:
                    continue
                state.active_requests.remove(request_id)
                state.request_lanes.pop(request_id, None)
                if state.half_open_probe == request_id:
                    state.half_open_probe = None
            self._pump_group_queue()
            for provider_id in ordered:
                state = self._states.get(provider_id)
                if state is None:
                    continue
                self._pump(
                    state, state.request_queue, state.active_requests,
                    state.limits.max_inflight_requests,
                )

    async def cancel(self, item_id: str) -> bool:
        async with self._lock:
            for waiter in tuple(self._group_queue):
                if waiter.item_id == item_id:
                    self._group_queue.remove(waiter)
                    if not waiter.future.done():
                        waiter.future.cancel()
                    self._pump_group_queue()
                    return True
            for state in self._states.values():
                for queue in (state.agent_queue, state.request_queue):
                    for waiter in tuple(queue):
                        if waiter.item_id == item_id:
                            queue.remove(waiter)
                            if not waiter.future.done():
                                waiter.future.cancel()
                            return True
        return False

    async def resize(self, provider_id: str, new_limits: ProviderLimits) -> None:
        async with self._lock:
            self._providers[provider_id] = new_limits
            state = self._state(provider_id)
            state.limits = new_limits
            self._pump_group_queue()
            self._pump(state, state.agent_queue, state.active_agents, new_limits.max_active_agents)
            self._pump(state, state.request_queue, state.active_requests, new_limits.max_inflight_requests)

    def snapshot(self, provider_id: str) -> dict[str, object]:
        state = self._state(provider_id)
        queued_groups = sum(
            provider_id in waiter.provider_ids for waiter in self._group_queue
        )
        limits = asdict(state.limits)
        # Keep the long-standing JSON shape for providers that use the
        # default policy; emit the lane controls when an operator configures
        # a non-default reserve or priority policy.
        if limits.get("controller_reserve") == 0:
            limits.pop("controller_reserve", None)
        if limits.get("max_worker_concurrency") == state.limits.max_active_agents:
            limits.pop("max_worker_concurrency", None)
        if limits.get("priority_policy") == "fifo":
            limits.pop("priority_policy", None)
        return {
            "provider_id": provider_id,
            "active_agents": len(state.active_agents),
            "active_requests": len(state.active_requests),
            "active_request_lanes": {
                lane: sum(current == lane for current in state.request_lanes.values())
                for lane in sorted(set(state.request_lanes.values()))
            },
            "queued_agents": len(state.agent_queue),
            "queued_requests": len(state.request_queue),
            "queued_group_requests": queued_groups,
            "limits": limits,
            "circuit": {
                "state": state.circuit_state,
                "open_until_monotonic": state.circuit_open_until,
                "failure_count": state.circuit_failures,
            },
        }

    def snapshots(self) -> dict[str, dict[str, object]]:
        """Return JSON-safe admission state for all configured providers."""
        return {
            provider_id: self.snapshot(provider_id)
            for provider_id in sorted(self._providers)
        }
