"""ASGI auth middleware for the ClaudeBrigade MCP control server."""

from __future__ import annotations

import hmac
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from enhanced_router.mcp_control import (
    McpPrincipal,
    _brigade_run_id_var,
    _mcp_principal_var,
)
from enhanced_router.state import RouteState

LOGGER = logging.getLogger("claude-enhanced-router")

# The token is loaded at call time from the environment
ROUTER_TOKEN: str | None = None


def _get_token() -> str:
    global ROUTER_TOKEN
    if ROUTER_TOKEN is None:
        ROUTER_TOKEN = os.environ.get("ENHANCED_ROUTER_TOKEN", "")
        if not ROUTER_TOKEN:
            raise RuntimeError(
                "ENHANCED_ROUTER_TOKEN is not set. "
                "The MCP control server cannot start without authentication. "
                "Ensure the router token exists and is readable."
            )
    return ROUTER_TOKEN


def authenticated_mcp_app(
    inner_app: Callable[..., Awaitable[Any]],
    state: RouteState | None,
) -> Callable[..., Awaitable[Any]]:
    """Wrap the MCP app with request-level authentication.

    Requirements (in order):
    1. Loopback source IP
    2. X-Enhanced-Token matches router token (constant-time comparison)
    3. X-Brigade-Run-Id header present and run exists in DB
    """

    async def auth_wrapper(scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await inner_app(scope, receive, send)
            return

        # Extract headers
        headers = dict(scope.get("headers", []))

        def _h(name: str) -> str | None:
            val = headers.get(name.lower().encode())
            return val.decode() if val else None

        # 1. Loopback check
        client_addr = scope.get("client")
        if client_addr:
            host = client_addr[0]
            if host not in ("127.0.0.1", "::1", "localhost"):
                await _send_error(send, 403, "loopback access only")
                return

        token = _get_token()
        provided_token = _h("x-enhanced-token")

        # 2. Token validation (fail closed — always compare when token is loaded)
        if not provided_token:
            await _send_error(send, 401, "missing x-enhanced-token")
            return
        if not hmac.compare_digest(provided_token, token):
            await _send_error(send, 401, "invalid router token")
            return

        # 3. Run ID validation
        run_id = _h("x-brigade-run-id")
        if not run_id:
            await _send_error(send, 400, "missing x-brigade-run-id header")
            return

        from enhanced_router.state import get_state
        current_state = state or get_state()
        run = current_state.get_run(run_id)
        if run is None:
            await _send_error(send, 404, f"run {run_id} not found")
            return
        if run.get("closed_at"):
            await _send_error(send, 410, f"run {run_id} is closed")
            return

        principal_kind = (_h("x-brigade-principal-kind") or "").strip().lower()
        if principal_kind not in {"controller", "worker", "sidecar"}:
            await _send_error(send, 401, "missing or invalid x-brigade-principal-kind")
            return

        session_id = _h("x-brigade-session-id")
        agent_id = _h("x-claude-code-agent-id")
        execution_id = _h("x-brigade-execution-id")
        credential = _h("x-brigade-controller-capability")

        if principal_kind == "controller":
            if not credential or not current_state.verify_controller_capability(
                run_id, credential, session_id=session_id,
            ):
                await _send_error(send, 401, "invalid controller capability")
                return
            capabilities = frozenset({
                "read_routes",
                "claim_native_action",
                "report_worker_result",
                "adjudicate_finding",
                "integrate_changeset",
                "complete_workflow",
                "invoke_sidecar",
                "cancel_execution",
                "retry_execution",
            })
        else:
            if not agent_id or not execution_id:
                await _send_error(
                    send, 401,
                    "worker and sidecar principals require agent and execution identity",
                )
                return
            active = current_state.get_active_epoch(run_id)
            execution = (
                current_state.get_agent_execution_scoped(
                    run_id, str(active["epoch_id"]), execution_id,
                )
                if active is not None else None
            )
            if (
                execution is None
                or execution.get("claude_agent_id") != agent_id
                or execution.get("status") not in {
                    "started", "running", "streaming", "verifying",
                }
            ):
                await _send_error(send, 403, "execution is not owned by the authenticated actor")
                return
            capabilities = frozenset({"read_routes", "report_worker_result"})

        # Store both contexts for tools — use token-based resets.
        run_context_token = _brigade_run_id_var.set(run_id)
        principal_context_token = _mcp_principal_var.set(McpPrincipal(
            principal_kind=principal_kind,
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            execution_id=execution_id,
            allowed_capabilities=capabilities,
            credential_id=credential,
            authenticated=True,
        ))

        try:
            await inner_app(scope, receive, send)
        finally:
            _mcp_principal_var.reset(principal_context_token)
            _brigade_run_id_var.reset(run_context_token)

    return auth_wrapper


async def _send_error(send: Callable, status: int, detail: str) -> None:
    """Send an error response via ASGI send."""
    body = json.dumps(
        {"error": detail}, separators=(",", ":")
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
