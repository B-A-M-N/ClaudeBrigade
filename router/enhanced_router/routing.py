"""Request resolving against RouteState + ModelRegistry.

A role alias (anthropic-brigade-*) is resolved through:
  1. Active epoch lookup
  2. Agent binding check (immutable once bound)
  3. ModelSpec lookup from the registry to determine backend kind and upstream model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import HTTPException

from enhanced_router.backends import BackendType, ResolvedRoute, ROLE_MODEL_ALIASES
from enhanced_router.registry import ModelRegistry
from enhanced_router.state import RouteState

logger = logging.getLogger("claude-enhanced-router")

# ROLE_MODEL_ALIASES is defined in backends.py as the single source of truth.
# Import it there to avoid duplicate definitions.


def resolve_request(
    public_model: str,
    run_id: str | None,
    claude_agent_id: str | None,
) -> ResolvedRoute:
    """Resolve a Claude Code request to a backend route.

    If *public_model* is a role alias:
      - Requires run_id and claude_agent_id (409 if missing)
      - Gets the active epoch (409 if none)
      - Checks existing binding → immutable if bound
      - Otherwise resolves epoch route, looks up ModelSpec for backend type,
        creates binding, returns ResolvedRoute with correct kind and upstream

    Otherwise:
      - Return ANTHROPIC_PASSTHROUGH for standard Claude model IDs
    """
    role = ROLE_MODEL_ALIASES.get(public_model)

    if role is None:
        return ResolvedRoute(kind=BackendType.ANTHROPIC_PASSTHROUGH, model_id=public_model)

    # Role alias path — require identity headers
    if not run_id or not claude_agent_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Role alias '{public_model}' requires x-brigade-run-id and "
                "x-claude-code-agent-id headers. Start a new session."
            ),
        )

    # Lazy imports to avoid circular import at module load time
    try:
        from enhanced_router.state import get_state
        from enhanced_router.registry import get_registry
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="enhanced_router package is not available. Ensure it is installed.",
        )

    state: RouteState = get_state()
    registry: ModelRegistry = get_registry()

    # Active epoch
    active_epoch = state.get_active_epoch(run_id)
    if active_epoch is None:
        raise HTTPException(
            status_code=409,
            detail=f"No active epoch for run {run_id}. Create one via the session_start hook.",
        )

    epoch_id = active_epoch["epoch_id"]

    # Check existing binding (immutable if active)
    existing = state.get_agent_binding(run_id, claude_agent_id)
    if existing is not None:
        # Verify the binding still belongs to the active epoch
        if existing.get("epoch_id") != epoch_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Agent {claude_agent_id} is bound to epoch "
                    f"{existing['epoch_id']} but active epoch is {epoch_id}. "
                    "Close the current epoch first."
                ),
            )
        # Verify the bound role matches the requested alias
        if existing.get("role") != role:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Agent {claude_agent_id} is bound as '{existing.get('role')}' "
                    f"but request uses alias for '{role}'."
                ),
            )

        model_id = existing["model_id"]
        spec = registry.get_model(model_id)

        if not spec.enabled:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_id}' bound to agent {claude_agent_id} is disabled.",
            )

        kind = _backend_kind(spec.backend)
        return ResolvedRoute(
            kind=kind,
            role=role,
            model_id=model_id,
            upstream_model=spec.upstream_model or spec.litellm_model,
            api_base=spec.api_base,
            agent_binding_id=existing["binding_id"],
            route_version=existing.get("route_version"),
            registry_hash=registry.registry_hash(),
            catalog_generation=existing.get("catalog_generation"),
            litellm_model_name=existing.get("litellm_model_name"),
            litellm_base_url=spec.api_base,
        )

    # No existing binding — resolve current epoch route
    route = state.get_role_route(run_id, epoch_id, role)
    if route is None:
        raise HTTPException(
            status_code=404,
            detail=f"No route defined for role '{role}' in epoch {epoch_id}",
        )

    model_id = route["model_id"]
    spec = registry.get_model(model_id)

    # Validate the model against the role
    if not spec.enabled:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' for role '{role}' is disabled.",
        )
    if role not in spec.allowed_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' does not allow role '{role}'.",
        )
    if role in ("implementer", "repairer") and not spec.capabilities.mutation:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' has mutation=false but role '{role}' requires mutation capability.",
        )

    # Validate credential availability for direct-anthropic backends
    if spec.backend == "direct-anthropic":
        import os
        if spec.api_key_env:
            if not os.environ.get(spec.api_key_env):
                raise HTTPException(
                    status_code=503,
                    detail=f"API key env var '{spec.api_key_env}' for model '{model_id}' is not set.",
                )

    kind = _backend_kind(spec.backend)
    upstream_model = spec.upstream_model or spec.litellm_model
    route_version = route["version"]
    reg_hash = registry.registry_hash()

    # Create immutable binding with all pinning fields
    binding_id = state.bind_agent(
        run_id=run_id,
        claude_agent_id=claude_agent_id,
        epoch_id=epoch_id,
        role=role,
        model_id=model_id,
        route_version=route_version,
        backend=spec.backend,
        registry_hash=reg_hash,
        catalog_generation=None,
        litellm_model_name=spec.litellm_model,
        upstream_model=spec.upstream_model,
        api_base=spec.api_base,
    )

    return ResolvedRoute(
        kind=kind,
        role=role,
        model_id=model_id,
        upstream_model=upstream_model,
        api_base=spec.api_base,
        agent_binding_id=binding_id,
        route_version=route_version,
        registry_hash=reg_hash,
        catalog_generation=None,
        litellm_model_name=spec.litellm_model,
        litellm_base_url=spec.api_base,
    )


def _backend_kind(backend: str) -> BackendType:
    """Map the registry backend string to a BackendType enum."""
    return {
        "direct-anthropic": BackendType.DIRECT_ANTHROPIC,
        "litellm": BackendType.LITELLM,
    }.get(backend, BackendType.ANTHROPIC_PASSTHROUGH)