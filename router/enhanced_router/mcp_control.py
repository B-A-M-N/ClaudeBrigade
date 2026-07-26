"""FastMCP control server for ClaudeBrigade model routing."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from enhanced_router.config_models import ModelSpec
from enhanced_router.registry import ModelRegistry
from enhanced_router.state import RouteState, get_state

# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------

control_mcp = FastMCP(
    name="ClaudeBrigade Model Control",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)

# ---------------------------------------------------------------------------
# MCP DTOs (sanitized — never expose ModelSpec internals)
# ---------------------------------------------------------------------------


def _sanitize_model(model_id: str, spec: ModelSpec) -> dict[str, Any]:
    """Return a safe subset of model metadata with no backend configuration."""
    return {
        "model_id": model_id,
        "display_name": spec.display_name,
        "capabilities": {
            "tools": spec.capabilities.tools,
            "mutation": spec.capabilities.mutation,
            "context_tokens": spec.capabilities.context_tokens,
            "reasoning": spec.capabilities.reasoning,
            "local": spec.capabilities.local,
        },
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@control_mcp.tool()
async def list_models(
    role: str | None = None,
    healthy_only: bool = True,
) -> list[dict[str, Any]]:
    """List available models, optionally filtered by role and health."""
    registry: ModelRegistry = _get_registry()
    results: list[dict[str, Any]] = []

    if role:
        candidates = registry.models_for_role(role)
        for mid, spec in candidates:
            results.append(_sanitize_model(mid, spec))
    else:
        for mid, spec in registry.models.items():
            results.append(_sanitize_model(mid, spec))

    return results


@control_mcp.tool()
async def recommend_model(
    role: str,
    required_context_tokens: int | None = None,
    local_only: bool = False,
    requires_tools: bool = True,
) -> dict[str, Any]:
    """Recommend a model for a role using deterministic ranking."""
    from enhanced_router.config_models import RecommendationConstraints

    registry: ModelRegistry = _get_registry()
    constraints = RecommendationConstraints(
        role=role,
        required_context_tokens=required_context_tokens,
        local_only=local_only,
        requires_tools=requires_tools,
    )
    ranked = registry.recommend(role, constraints=constraints)
    return {
        "role": role,
        "recommendations": [
            {"model_id": r.model_id, "score": r.score, "reason": r.reason}
            for r in ranked
        ],
    }


@control_mcp.tool()
async def set_role_route(
    role: str,
    model_id: str,
    reason: str,
) -> dict[str, Any]:
    """Set the backing model for a worker role.

    Takes effect on the **next** subagent spawn. Already-running agents are
    unaffected.
    """
    registry: ModelRegistry = _get_registry()

    # Validate model exists in registry
    try:
        spec = registry.get_model(model_id)
    except KeyError:
        return {"changed": False, "error": f"Unknown model: {model_id}"}

    # Validate model is enabled
    if not spec.enabled:
        return {
            "changed": False,
            "error": f"Model '{model_id}' is disabled",
        }

    # Validate role is allowed for this model
    if role not in spec.allowed_roles:
        return {
            "changed": False,
            "error": (
                f"Model '{model_id}' does not allow role '{role}'"
            ),
        }

    # Validate mutation capability for mutation roles
    if role in ("implementer", "repairer") and not spec.capabilities.mutation:
        return {
            "changed": False,
            "error": (
                f"Model '{model_id}' has mutation=false "
                f"but role '{role}' requires mutation"
            ),
        }

    # Validate tools capability
    if not spec.capabilities.tools:
        return {
            "changed": False,
            "error": (
                f"Model '{model_id}' has tools=false "
                f"but role '{role}' requires tools"
            ),
        }

    # Validate credentials available for direct-anthropic backends
    if spec.backend == "direct-anthropic":
        import os

        if spec.api_key_env and not os.environ.get(spec.api_key_env):
            return {
                "changed": False,
                "error": (
                    f"API key env var '{spec.api_key_env}' "
                    f"for model '{model_id}' is not set"
                ),
            }

    # If all validations pass, proceed with the route change
    state: RouteState = get_state()
    run_id = _get_current_run_id()

    if not run_id:
        return {"changed": False, "error": "No active run ID"}

    active = state.get_active_epoch(run_id)
    if not active:
        return {"changed": False, "error": "No active epoch"}

    state.set_role_route(
        run_id=run_id,
        epoch_id=active["epoch_id"],
        role=role,
        model_id=model_id,
        source="mcp",
        reason=reason,
    )
    return {
        "changed": True,
        "effective_for": "next_subagent_spawn",
        "active_agents_unchanged": True,
    }


@control_mcp.tool()
async def get_route_status() -> dict[str, Any]:
    """Return the current route configuration and active binding count."""
    state: RouteState = get_state()
    run_id = _get_current_run_id()

    if not run_id:
        return {"error": "No active run ID"}

    active = state.get_active_epoch(run_id)
    if not active:
        return {"error": "No active epoch"}

    routes = state.get_epoch_routes(run_id, active["epoch_id"])
    bindings = state.get_active_bindings(run_id, active["epoch_id"])

    return {
        "epoch_id": active["epoch_id"],
        "workflow_id": active["workflow_id"],
        "profile_id": active.get("profile_id"),
        "routes": routes,
        "active_bindings_count": len(bindings),
        "registry_hash": _get_registry().registry_hash(),
    }


@control_mcp.tool()
async def select_profile(
    profile_id: str,
    reason: str,
) -> dict[str, Any]:
    """Apply a profile, setting all four role routes atomically."""
    state: RouteState = get_state()
    run_id = _get_current_run_id()

    if not run_id:
        return {"changed": False, "error": "No active run ID"}

    active = state.get_active_epoch(run_id)
    if not active:
        return {"changed": False, "error": "No active epoch"}

    try:
        state.set_profile_routes_atomic(
            run_id=run_id,
            epoch_id=active["epoch_id"],
            profile_id=profile_id,
            reason=reason,
        )
    except (KeyError, ValueError) as exc:
        return {"changed": False, "error": str(exc)}

    return {
        "changed": True,
        "profile_id": profile_id,
        "effective_for": "next_subagent_spawn",
        "active_agents_unchanged": True,
    }


@control_mcp.tool()
async def reset_role_route(
    role: str,
    reason: str,
) -> dict[str, Any]:
    """Reset a role route to the epoch's profile default."""
    state: RouteState = get_state()
    run_id = _get_current_run_id()

    if not run_id:
        return {"changed": False, "error": "No active run ID"}

    active = state.get_active_epoch(run_id)
    if not active:
        return {"changed": False, "error": "No active epoch"}

    profile_id = active.get("profile_id", "hybrid")
    registry: ModelRegistry = _get_registry()
    profile = registry.get_profile(profile_id)
    default_model = getattr(profile, role)

    state.set_role_route(
        run_id=run_id,
        epoch_id=active["epoch_id"],
        role=role,
        model_id=default_model,
        source="mcp-reset",
        reason=reason,
    )
    return {
        "changed": True,
        "model_id": default_model,
        "profile_id": profile_id,
        "effective_for": "next_subagent_spawn",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_registry_instance: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        import pathlib

        cfg_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "config"
        _registry_instance = ModelRegistry(cfg_dir)
        _registry_instance.load_models()
        _registry_instance.load_profiles()
        _registry_instance.load_workflows()
    return _registry_instance


# Context variable set by the MCP auth transport
from contextvars import ContextVar

_brigade_run_id_var: ContextVar[str | None] = ContextVar(
    "_brigade_run_id", default=None
)


def _get_current_run_id() -> str | None:
    return _brigade_run_id_var.get()


def set_current_run_id(run_id: str | None) -> None:
    """Set the run ID for the current request context.

    If *run_id* is ``None``, the context variable is reset to its default.
    """
    if run_id is not None:
        _brigade_run_id_var.set(run_id)
    else:
        _brigade_run_id_var.set(None)