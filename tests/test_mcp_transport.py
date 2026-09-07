"""Actor and resource authentication tests for the MCP transport boundary."""

from __future__ import annotations

import asyncio
import tempfile

from enhanced_router import mcp_transport
from enhanced_router.mcp_control import _get_current_principal
from enhanced_router.state import RouteState


def _request(app, headers: dict[str, str], seen: list[object]) -> list[dict]:
    async def run() -> list[dict]:
        messages: list[dict] = []

        async def inner(scope, receive, send):
            seen.append(_get_current_principal())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        wrapped = app(inner)
        scope = {
            "type": "http",
            "client": ("127.0.0.1", 1234),
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await wrapped(scope, receive, send)
        return messages

    return asyncio.run(run())


def test_controller_capability_authenticates_principal(monkeypatch):
    state = RouteState(tempfile.mktemp())
    state.create_run("run-1", session_id="session-1", controller_capability="secret")
    monkeypatch.setenv("ENHANCED_ROUTER_TOKEN", "router")
    monkeypatch.setattr(mcp_transport, "ROUTER_TOKEN", "router")
    seen: list[object] = []

    messages = _request(
        lambda inner: mcp_transport.authenticated_mcp_app(inner, state),
        {
            "X-Enhanced-Token": "router",
            "X-Brigade-Run-Id": "run-1",
            "X-Brigade-Principal-Kind": "controller",
            "X-Brigade-Controller-Capability": "secret",
        },
        seen,
    )

    assert messages[0]["status"] == 200
    principal = seen[0]
    assert principal is not None
    assert principal.principal_kind == "controller"
    assert principal.run_id == "run-1"
    assert principal.authenticated is True


def test_missing_controller_capability_is_rejected(monkeypatch):
    state = RouteState(tempfile.mktemp())
    state.create_run("run-1", controller_capability="secret")
    monkeypatch.setenv("ENHANCED_ROUTER_TOKEN", "router")
    monkeypatch.setattr(mcp_transport, "ROUTER_TOKEN", "router")
    seen: list[object] = []

    messages = _request(
        lambda inner: mcp_transport.authenticated_mcp_app(inner, state),
        {
            "X-Enhanced-Token": "router",
            "X-Brigade-Run-Id": "run-1",
            "X-Brigade-Principal-Kind": "controller",
        },
        seen,
    )

    assert messages[0]["status"] == 401
    assert not seen


def test_error_response_disconnect_is_swallowed():
    async def run() -> None:
        async def send(_message):
            raise ConnectionResetError("client disconnected")

        await mcp_transport._send_error(send, 401, "gone")

    asyncio.run(run())
