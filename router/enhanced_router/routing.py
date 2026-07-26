"""Request resolving against RouteState -- role alias -> binding or epoch route."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import HTTPException

from enhanced_router.backends import BackendType, ResolvedRoute

logger = logging.getLogger("claude-enhanced-router")

ROLE_MODEL_ALIASES = {
    "anthropic-brigade-recon": "recon",
    "anthropic-brigade-implementer": "implementer",
    "anthropic-brigade-adversary": "adversary",
    "anthropic-brigade-repairer": "repairer",
}


def resolve_request(
    public_model: str,
    run_id: str | None,
    claude_agent_id: str | None,
) -> ResolvedRoute:
    """Resolve a Claude Code request to a backend route.

    If *public_model* is a role alias (one of ``ROLE_MODEL_ALIASES``):
      - Requires *run_id* and *claude_agent_id* (raise 409 if missing)
      - Gets the active epoch for *run_id* (raise 409 if none)
      - Checks for an existing binding for (run_id, claude_agent_id)
        - If bound: return the immutable ResolvedRoute from the binding
        - If not bound: resolve current epoch route, create binding via
          RouteState.bind_agent(), return ResolvedRoute

    Otherwise (standard Claude model id):
      - Return ResolvedRoute(kind=ANTHROPIC_PASSTHROUGH)
    """
    role = ROLE_MODEL_ALIASES.get(public_model)

    if role is None:
        # Standard Claude model -- passthrough
        return ResolvedRoute(kind=BackendType.ANTHROPIC_PASSTHROUGH, model_id=public_model)

    # Role alias path
    if not run_id or not claude_agent_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Role alias '{public_model}' requires run_id and claude_agent_id headers. "
                "Start a new session to get these."
            ),
        )

    # Lazily import RouteState to avoid import before package install
    try:
        from enhanced_router.state import get_state
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RouteState is not available. Ensure enhanced_router is installed.",
        )

    state = get_state()

    # Get active epoch
    active_epoch = state.get_active_epoch(run_id)
    if active_epoch is None:
        raise HTTPException(
            status_code=409,
            detail=f"No active epoch for run {run_id}. Create one via session_start hook.",
        )

    # Check existing binding
    existing = state.get_agent_binding(run_id, claude_agent_id)
    if existing:
        return ResolvedRoute(
            kind=BackendType.ANTHROPIC_PASSTHROUGH,
            model_id=existing["model_id"],
            agent_binding_id=existing["binding_id"],
        )

    # Resolve current epoch route for the role
    route = state.get_role_route(run_id, active_epoch["epoch_id"], role)
    if route is None:
        raise HTTPException(
            status_code=404,
            detail=f"No route defined for role '{role}' in epoch {active_epoch['epoch_id']}",
        )

    # Create binding
    binding_id = state.bind_agent(
        run_id=run_id,
        claude_agent_id=claude_agent_id,
        epoch_id=active_epoch["epoch_id"],
        role=role,
        model_id=route["model_id"],
        route_version=route["version"],
    )

    return ResolvedRoute(
        kind=BackendType.ANTHROPIC_PASSTHROUGH,
        model_id=route["model_id"],
        agent_binding_id=binding_id,
    )
