from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import enhanced_router.backends as backends
from enhanced_router.backends import BackendType, ResolvedRoute


class _FakeResponse:
    def __init__(
        self,
        request: httpx.Request,
        *,
        body: bytes = b"{}",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        fail_read: bool = False,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.request = request
        self.body = body
        self.fail_read = fail_read
        self.chunks = chunks or [body]
        self.closed = False

    async def aread(self) -> bytes:
        if self.fail_read:
            raise httpx.ReadError("upstream reset while reading response", request=self.request)
        return self.body

    async def aclose(self) -> None:
        self.closed = True

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests = 0

    def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
        return httpx.Request(method, url, **kwargs)

    async def send(self, request: httpx.Request, *, stream: bool = False) -> object:
        del stream
        self.requests += 1
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result(request) if callable(result) else result


def _registry(monkeypatch: pytest.MonkeyPatch, *, attempts: int = 2) -> None:
    provider = SimpleNamespace(
        retry=SimpleNamespace(
            max_attempts=attempts,
            retryable_statuses=[408, 429, 500, 502, 503, 504, 529],
            max_backoff_seconds=1.0,
        ),
        deadlines=SimpleNamespace(
            time_to_first_token_seconds=5.0,
            stream_idle_seconds=5.0,
            request_wall_seconds=30.0,
        ),
    )
    monkeypatch.setattr(
        "enhanced_router.registry.get_registry",
        lambda: SimpleNamespace(providers={"freeinference": provider}),
    )


def _route() -> ResolvedRoute:
    return ResolvedRoute(
        kind=BackendType.LITELLM,
        model_id="provider/model",
        provider_id="freeinference",
        api_base="https://free.example/v1",
    )


@pytest.mark.asyncio
async def test_transport_retry_replaces_broken_keepalive_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry(monkeypatch)
    first = _FakeClient([httpx.ConnectError("stale connection")])
    second = _FakeClient([lambda request: _FakeResponse(request)])
    recycled: list[object] = []

    async def no_failure(*_args: object, **_kwargs: object) -> None:
        return None

    async def allowed(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(backends._provider_admission, "record_transport_failure", no_failure)
    monkeypatch.setattr(backends._provider_admission, "retry_allowed", allowed)
    monkeypatch.setattr(backends, "_record_provider_response", no_failure)
    monkeypatch.setattr(backends.random, "uniform", lambda *_args: 0.0)
    monkeypatch.setattr(backends.asyncio, "sleep", no_failure)

    def recycle(client: object) -> object:
        recycled.append(client)
        return second

    monkeypatch.setattr(backends, "recycle_upstream_client", recycle)
    request = await backends._send_with_provider_retry(
        first,
        lambda: first.build_request("POST", "https://free.example/v1/chat/completions"),
        _route(),
        backends.time.perf_counter(),
        "request-1",
    )

    assert request.status_code == 200
    assert first.requests == 1
    assert second.requests == 1
    assert recycled == [first]


@pytest.mark.asyncio
async def test_non_streaming_body_reset_is_retried_before_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _registry(monkeypatch)
    first = _FakeClient([lambda request: _FakeResponse(request, fail_read=True)])
    second = _FakeClient([lambda request: _FakeResponse(request, body=b'{"ok":true}')])

    async def no_failure(*_args: object, **_kwargs: object) -> None:
        return None

    async def allowed(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(backends._provider_admission, "record_transport_failure", no_failure)
    monkeypatch.setattr(backends._provider_admission, "retry_allowed", allowed)
    monkeypatch.setattr(backends, "_record_provider_response", no_failure)
    monkeypatch.setattr(backends.random, "uniform", lambda *_args: 0.0)
    monkeypatch.setattr(backends.asyncio, "sleep", no_failure)
    monkeypatch.setattr(backends, "recycle_upstream_client", lambda _client: second)

    response, content = await backends._request_non_streaming_with_provider_retry(
        first,
        lambda: first.build_request("POST", "https://free.example/v1/chat/completions"),
        _route(),
        backends.time.perf_counter(),
        "request-2",
    )

    assert response.status_code == 200
    assert content == b'{"ok":true}'
    assert first.requests == 1
    assert second.requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_retryable_provider_status_recycles_pool_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    _registry(monkeypatch)
    first = _FakeClient([
        lambda request: _FakeResponse(
            request,
            status_code=status,
            headers={"retry-after": "0"},
        ),
    ])
    second = _FakeClient([lambda request: _FakeResponse(request, body=b'{"ok":true}')])
    recycled: list[object] = []

    async def no_failure(*_args: object, **_kwargs: object) -> None:
        return None

    async def allowed(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(backends._provider_admission, "record_transport_failure", no_failure)
    monkeypatch.setattr(backends._provider_admission, "retry_allowed", allowed)
    monkeypatch.setattr(backends, "_record_provider_response", no_failure)
    monkeypatch.setattr(backends.random, "uniform", lambda *_args: 0.0)
    monkeypatch.setattr(backends.asyncio, "sleep", no_failure)

    def recycle(client: object) -> object:
        recycled.append(client)
        return second

    monkeypatch.setattr(backends, "recycle_upstream_client", recycle)
    response = await backends._send_with_provider_retry(
        first,
        lambda: first.build_request("POST", "https://free.example/v1/messages"),
        _route(),
        backends.time.perf_counter(),
        "request-status",
    )

    assert response.status_code == 200
    assert first.requests == 1
    assert second.requests == 1
    assert recycled == [first]


@pytest.mark.asyncio
async def test_authentication_status_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    _registry(monkeypatch)
    client = _FakeClient([
        lambda request: _FakeResponse(request, status_code=401),
        lambda request: _FakeResponse(request, status_code=200),
    ])
    async def record_response(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(backends, "_record_provider_response", record_response)

    response = await backends._send_with_provider_retry(
        client,
        lambda: client.build_request("POST", "https://free.example/v1/messages"),
        _route(),
        backends.time.perf_counter(),
        "request-auth",
    )

    assert response.status_code == 401
    assert client.requests == 1


@pytest.mark.asyncio
async def test_stream_fixture_preserves_split_sse_usage_and_releases_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    request = httpx.Request("POST", "https://free.example/v1/messages")
    response = _FakeResponse(
        request,
        chunks=[
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":12}}}\n',
            b'data: {"type":"message_delta","usage":{"output_tokens":3}}\n\n',
            b'data: {"type":"message_delta","usage":{"output_tokens":2}}\n',
            b'data: [DONE]\n',
        ],
    )
    released: list[str] = []
    usage: list[tuple[dict[str, object], bool]] = []

    async def release(_route: object, request_id: str | None) -> None:
        released.append(request_id or "")

    monkeypatch.setattr(backends, "_release_provider_request", release)
    monkeypatch.setattr(
        backends,
        "_record_endpoint_usage",
        lambda _route, _request_id, payload, _started, *, succeeded: usage.append((payload, succeeded)),
    )
    monkeypatch.setattr(backends, "_provider_stream_deadlines", lambda _route: (5.0, 5.0, 30.0))

    chunks = [
        chunk async for chunk in backends._stream_upstream_with_admission(
            response, "stream-1", route, backends.time.perf_counter()
        )
    ]

    assert b"".join(chunks).count(b"data:") == 4
    assert response.closed is True
    assert released == ["stream-1"]
    assert usage == [({"usage": {"input_tokens": 12, "output_tokens": 3}}, True)]


@pytest.mark.asyncio
async def test_stream_cancellation_closes_response_and_releases_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    request = httpx.Request("POST", "https://free.example/v1/messages")
    response = _FakeResponse(request, chunks=[b"data: {\"type\":\"message_start\"}\n"])
    released: list[str] = []
    failures: list[str] = []

    async def release(_route: object, request_id: str | None) -> None:
        released.append(request_id or "")

    monkeypatch.setattr(backends, "_release_provider_request", release)
    monkeypatch.setattr(
        backends,
        "_record_execution_transport_failure",
        lambda _route, error_class, _reason: failures.append(error_class),
    )
    monkeypatch.setattr(backends, "_provider_stream_deadlines", lambda _route: (5.0, 5.0, 30.0))

    generator = backends._stream_upstream_with_admission(
        response, "stream-cancel", route, backends.time.perf_counter()
    )
    assert await generator.__anext__() == response.chunks[0]
    await generator.aclose()

    assert response.closed is True
    assert released == ["stream-cancel"]
