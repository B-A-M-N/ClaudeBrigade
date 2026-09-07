from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from enhanced_router.config_models import ModelSpec
from enhanced_router.registry import ModelRegistry, get_registry as _get_registry
from enhanced_router.state import (
    RouteState,
    WorkflowPhaseStateError,
    WorkflowStateError,
    get_state,
)

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
            "controller_eligible": spec.capabilities.controller_eligible,
            "write_tool_certified": spec.capabilities.write_tool_certified,
            "read_tool_certified": spec.capabilities.read_tool_certified,
        },
        "provider_id": spec.provider_id,
        "availability": spec.availability,
        "endpoints": {
            endpoint_id: {
                "backend": endpoint.backend,
                "protocol": endpoint.protocol,
                "availability": endpoint.availability,
                "certified": endpoint.certified,
            }
            for endpoint_id, endpoint in spec.endpoints.items()
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
    """List available models, optionally filtered by role and health.

    When *healthy_only* is True (default), only models whose latest health
    record shows reachable=1, authenticated=1, and compatible=1 are returned.
    Models without a health record are considered untested and excluded.
    """
    registry: ModelRegistry = _get_registry()
    results: list[dict[str, Any]] = []

    candidates: list[tuple[str, "ModelSpec"]]
    if role:
        candidates = registry.models_for_role(role)
    else:
        candidates = [(mid, spec) for mid, spec in registry.models.items()]

    for mid, spec in candidates:
        if healthy_only:
            health = get_state().get_model_health(mid)
            if health is None or not (health["reachable"] and health["authenticated"] and health["compatible"]):
                continue
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
    endpoint: str = "auto",
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

    if endpoint != "auto":
        try:
            from enhanced_router.endpoint_selection import select_endpoint
            select_endpoint(
                model_id, spec, get_state(), explicit_endpoint=endpoint,
                require_certified=True, provider_id=spec.provider_id,
                configuration_hash=registry.registry_hash(),
                required_capabilities=("messages", "streaming", "tools"),
            )
        except ValueError as exc:
            return {"changed": False, "error": f"Endpoint '{endpoint}' is not eligible: {exc}"}

    # Validate mutation capability for mutation roles
    if role in ("implementer", "repairer") and spec.capabilities.write_tool_certified is not True:
        return {
            "changed": False,
            "error": f"Model '{model_id}' is not write-tool certified for role '{role}'",
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

        credential_available = bool(os.environ.get(spec.api_key_env)) if spec.api_key_env else True
        if spec.api_key_env:
            try:
                from enhanced_router.credential_store import resolve_loaded

                credential_available = bool(resolve_loaded(spec.api_key_env))
            except Exception:
                pass
        if spec.api_key_env and not credential_available:
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
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", controller_only=True,
    )
    if authorization_error:
        return {"changed": False, "error": authorization_error}

    active = state.get_active_epoch(run_id)
    if not active:
        return {"changed": False, "error": "No active epoch"}

    result = state.set_role_route(
        run_id=run_id,
        epoch_id=active["epoch_id"],
        role=role,
        model_id=model_id,
        source="mcp",
        reason=reason,
        endpoint_override=None if endpoint == "auto" else endpoint,
    )
    # Create a route snapshot for the response
    snapshot = state.create_route_snapshot(
        run_id, active["epoch_id"], purpose="mcp-route-change"
    )
    return {
        "changed": True,
        "run_id": run_id,
        "epoch_id": active["epoch_id"],
        "role": role,
        "model_id": model_id,
        "endpoint": endpoint,
        "route_version": result.get("version"),
        "effective_for": "next_subagent_spawn",
        "active_agents_unchanged": True,
        "route_snapshot_sha256": snapshot,
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
    registry = _get_registry()
    endpoint_observations: dict[str, dict[str, dict]] = {}
    for role, target in routes.items():
        model_id = target.get("model_id")
        if not isinstance(model_id, str) or model_id not in registry.models:
            continue
        spec = registry.get_model(model_id)
        endpoint_observations[role] = state.get_endpoint_observations(
            model_id,
            provider_id=spec.provider_id,
            configuration_hash=registry.registry_hash(),
        )

    return {
        "epoch_id": active["epoch_id"],
        "workflow_id": active["workflow_id"],
        "profile_id": active.get("profile_id") or None,
        "routes": routes,
        "endpoint_observations": endpoint_observations,
        "active_bindings_count": len(bindings),
        "registry_hash": _get_registry().registry_hash(),
    }


@control_mcp.tool()
async def get_runnable_actions() -> dict[str, Any]:
    """Return native-agent actions that the controller may start now.

    Claude Code owns native ``Agent`` spawning.  The router therefore exposes
    a cooperative scheduling step instead of pretending a denied tool call
    can be replayed later by SQLite.
    """
    state: RouteState = get_state()
    run_id = _get_current_run_id()
    if not run_id:
        return {"ready": False, "error": "No active run ID", "actions": []}
    authorization_error = _require_capability(
        state, run_id, "read_routes",
    )
    if authorization_error:
        return {"ready": False, "error": authorization_error, "actions": []}
    active = state.get_active_epoch(run_id)
    if not active:
        return {"ready": False, "error": "No active epoch", "actions": []}
    actions = state.get_runnable_actions(run_id, active["epoch_id"])
    return {
        "ready": True,
        "run_id": run_id,
        "epoch_id": active["epoch_id"],
        "actions": actions,
        "capacity": state.get_run_resource_capacity(run_id, str(active["epoch_id"])),
        "next_poll": "after a native agent reaches a terminal lifecycle state",
    }


@control_mcp.tool()
async def get_runnable_action_wave(limit: int | None = None) -> dict[str, Any]:
    """Return all independently packageable native actions in one bounded wave."""
    state: RouteState = get_state()
    run_id = _get_current_run_id()
    if not run_id:
        return {"ready": False, "error": "No active run ID", "actions": []}
    authorization_error = _require_capability(state, run_id, "read_routes")
    if authorization_error:
        return {"ready": False, "error": authorization_error, "actions": []}
    active = state.get_active_epoch(run_id)
    if not active:
        return {"ready": False, "error": "No active epoch", "actions": []}
    try:
        actions = state.get_runnable_action_wave(
            run_id, str(active["epoch_id"]), limit=limit
        )
    except (ValueError, WorkflowStateError) as exc:
        return {"ready": False, "error": str(exc), "actions": []}
    provider_worker_limits: dict[str, int] = {}
    try:
        from enhanced_router.registry import get_registry

        registry = get_registry()
        for action in actions:
            provider_id = str(action.get("provider_id") or "").strip()
            provider = registry.providers.get(provider_id) if provider_id else None
            if provider is not None:
                provider_worker_limits[provider_id] = int(
                    provider.limits.max_worker_concurrency or 0
                )
    except Exception:
        # Action generation already failed closed when a provider cannot be
        # resolved. Keep this response compatible with older registry
        # doubles used by embedded callers.
        provider_worker_limits = {}
    return {
        "ready": True,
        "run_id": run_id,
        "epoch_id": active["epoch_id"],
        "actions": actions,
        # ``worker_limit`` is retained for older controllers, but it is not
        # the authority when multiple providers are present.  Consumers
        # should use provider_worker_limits and the run-wide capacity snapshot.
        "worker_limit": None,
        "provider_worker_limits": provider_worker_limits,
        "capacity": state.get_run_resource_capacity(run_id, str(active["epoch_id"])),
        "next_poll": "after a wave changes terminal state or capacity",
    }


@control_mcp.tool()
async def claim_runnable_action(action_id: str) -> dict[str, Any]:
    """Claim one returned action before spawning or resolving it.

    Claude Code owns the child lifecycle and cannot replay a denied tool call.
    The main controller must claim an action first. Native claims are consumed
    exactly once by the PreToolUse hook for the matching agent name; controller
    integration claims are consumed by the yellow/red resolution operation.
    """
    state: RouteState = get_state()
    run_id = _get_current_run_id()
    if not run_id:
        return {"claimed": False, "error": "No active run ID"}
    authorization_error = _require_capability(
        state, run_id, "claim_native_action", controller_only=True,
    )
    if authorization_error:
        return {"claimed": False, "error": authorization_error}
    active = state.get_active_epoch(run_id)
    if not active:
        return {"claimed": False, "error": "No active epoch"}
    try:
        claim = state.claim_runnable_action(
            run_id, str(active["epoch_id"]), action_id,
        )
    except (ValueError, WorkflowStateError, WorkflowPhaseStateError) as exc:
        return {"claimed": False, "error": str(exc)}
    return {"claimed": True, "run_id": run_id, "epoch_id": active["epoch_id"], **claim}


@control_mcp.tool()
async def get_action_contract(action_id: str) -> dict[str, Any]:
    """Return the exact native worker contract for a runnable action."""
    state: RouteState = get_state()
    run_id = _get_current_run_id()
    if not run_id:
        return {"found": False, "error": "No active run ID"}
    authorization_error = _require_capability(
        state, run_id, "read_routes", controller_only=True,
    )
    if authorization_error:
        return {"found": False, "error": authorization_error}
    active = state.get_active_epoch(run_id)
    if not active:
        return {"found": False, "error": "No active epoch"}
    actions = state.get_runnable_actions(
        run_id, str(active["epoch_id"]), include_claimed=True,
    )
    action = next((item for item in actions if item.get("action_id") == action_id), None)
    if action is None:
        action = next(
            (
                item for item in state.get_runnable_action_wave(
                    run_id, str(active["epoch_id"]), include_claimed=True
                )
                if item.get("action_id") == action_id
            ),
            None,
        )
    if action is None:
        return {"found": False, "error": f"Action '{action_id}' not found"}
    return {
        "found": True,
        "action_id": action_id,
        "native_agent_name": action.get("native_agent_name"),
        "worker_kind": action.get("worker_kind"),
        "worker_id": action.get("worker_id"),
        "agent_id": action.get("agent_id"),
        "native_slot": action.get("native_slot"),
        "expected_model_alias": action.get("expected_model_alias"),
        "package_id": action.get("package_id"),
        "package_contract_digest": action.get("package_contract_digest"),
        "prompt_contract_digest": action.get("prompt_contract_digest"),
        "package_contract_version": action.get("package_contract_version"),
        "package_display_name": action.get("package_display_name"),
        "package_summary": action.get("package_summary"),
        "display_name": action.get("display_name") or action.get("package_display_name"),
        "display_summary": action.get("display_summary") or action.get("package_summary"),
        "controller_action_kind": action.get("controller_action_kind"),
        "produces": action.get("produces"),
        "fanout_from": action.get("fanout_from"),
        "launch_policy": action.get("launch_policy"),
        "initial_fanout": action.get("initial_fanout"),
        "maximum_replicas": action.get("maximum_replicas"),
        "required_successes": action.get("required_successes"),
        "required_action": action.get("required_action"),
        "prompt_contract": action.get("prompt") or {},
        "path_scope": action.get("path_scope", []),
        "prohibited_paths": action.get("prohibited_paths", []),
        "priority_class": action.get("priority_class"),
        "model_id": action.get("model_id"),
        "capability_snapshot": action.get("capability_snapshot") or {},
        "workspace_policy": action.get("workspace_policy", "none"),
        "background": bool(action.get("background", True)),
        "prompt": action.get("prompt") or {
            "phase_id": action.get("phase_id"),
            "role": action.get("role"),
            "acceptance_criteria": [],
            "required_tests": [],
        },
    }


@control_mcp.tool()
async def invoke_coprocessor(
    action_id: str,
    claim_token: str,
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Start a bounded coprocessor for a claimed coprocessor action."""
    state = get_state()
    run_id = _get_current_run_id()
    if not run_id:
        return {"started": False, "error": "No active run ID"}
    authorization_error = _require_capability(
        state, run_id, "invoke_coprocessor", controller_only=True,
    )
    if authorization_error:
        return {"started": False, "error": authorization_error}
    active = state.get_active_epoch(run_id)
    if active is None:
        return {"started": False, "error": "No active epoch"}
    try:
        from enhanced_router.coprocessor_executor import get_coprocessor_executor
        execution = await get_coprocessor_executor(state).invoke(
            run_id=run_id,
            epoch_id=str(active["epoch_id"]),
            action_id=action_id,
            claim_token=claim_token,
            packet=packet,
        )
    except (ValueError, WorkflowStateError) as exc:
        return {"started": False, "error": str(exc)}
    return {
        "started": True,
        "run_id": run_id,
        "epoch_id": str(active["epoch_id"]),
        "execution_id": execution["execution_id"],
        "execution": execution,
    }


@control_mcp.tool()
async def adjudicate_coprocessor_result(
    run_id: str,
    epoch_id: str,
    execution_id: str,
    disposition: str,
    reason: str = "",
    evidence_valid: bool | None = None,
    quality_score: float | None = None,
    accepted_finding_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Explicitly accept or reject a completed bounded coprocessor result.

    Transport success and schema validity never authorize workflow progress;
    only this controller-owned operation can set acceptance evidence.
    """
    state = get_state()
    authorization_error = _require_capability(
        state, run_id, "adjudicate_coprocessor_result", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"adjudicated": False, "error": authorization_error}
    principal = _get_current_principal()
    adjudicated_by = str(principal.agent_id if principal and principal.agent_id else "controller")
    try:
        result = state.adjudicate_coprocessor_result(
            run_id=run_id,
            epoch_id=epoch_id,
            execution_id=execution_id,
            disposition=disposition,
            reason=reason,
            evidence_valid=evidence_valid,
            quality_score=quality_score,
            accepted_finding_ids=accepted_finding_ids,
            adjudicated_by=adjudicated_by,
        )
    except (ValueError, WorkflowStateError) as exc:
        return {"adjudicated": False, "error": str(exc)}
    if result is None:
        return {"adjudicated": False, "error": "coprocessor execution not found"}
    return {"adjudicated": True, "execution": result}


@control_mcp.tool()
async def adjudicate_native_result(
    run_id: str,
    epoch_id: str,
    execution_id: str,
    disposition: str,
    reason: str = "",
    evidence_valid: bool | None = None,
    quality_score: float | None = None,
    accepted_finding_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Accept or reject a completed native worker result for quality-gated phases."""
    state = get_state()
    error = _require_capability(
        state, run_id, "adjudicate_native_result", epoch_id=epoch_id,
        controller_only=True,
    )
    if error:
        return {"adjudicated": False, "error": error}
    principal = _get_current_principal()
    adjudicated_by = str(principal.agent_id if principal and principal.agent_id else "controller")
    try:
        result = state.adjudicate_native_result(
            run_id=run_id, epoch_id=epoch_id, execution_id=execution_id,
            disposition=disposition, reason=reason, evidence_valid=evidence_valid,
            quality_score=quality_score, accepted_finding_ids=accepted_finding_ids,
            adjudicated_by=adjudicated_by,
        )
    except (ValueError, WorkflowStateError) as exc:
        return {"adjudicated": False, "error": str(exc)}
    return {"adjudicated": result is not None, "execution": result}


@control_mcp.tool()
async def adjudicate_feedback(
    run_id: str,
    epoch_id: str,
    feedback_id: str,
    disposition: str,
    reason: str = "",
    adopted_finding_ids: list[str] | None = None,
    rejected_finding_ids: list[str] | None = None,
    resulting_action_ids: list[str] | None = None,
    resulting_changeset_ids: list[str] | None = None,
    later_validation: dict[str, Any] | None = None,
    harm_class: str | None = None,
    quality_score: float | None = None,
    quality_dimensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record whether delivered automatic advice helped, harmed or was ignored."""
    state = get_state()
    authorization_error = _require_capability(
        state, run_id, "adjudicate_feedback", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"adjudicated": False, "error": authorization_error}
    feedback = state.get_feedback(feedback_id)
    if feedback is None or str(feedback.get("run_id")) != run_id or str(feedback.get("epoch_id")) != epoch_id:
        return {"adjudicated": False, "error": "feedback is outside the requested run and epoch"}
    try:
        result = state.adjudicate_feedback(
            feedback_id,
            disposition=disposition,
            reason=reason,
            adopted_finding_ids=adopted_finding_ids,
            rejected_finding_ids=rejected_finding_ids,
            resulting_action_ids=resulting_action_ids,
            resulting_changeset_ids=resulting_changeset_ids,
            later_validation=later_validation,
            harm_class=harm_class,
            quality_score=quality_score,
            quality_dimensions=quality_dimensions,
            task_class=str(feedback.get("checkpoint") or ""),
        )
    except ValueError as exc:
        return {"adjudicated": False, "error": str(exc)}
    return {"adjudicated": result is not None, "feedback": result}


@control_mcp.tool()
async def get_coprocessor_metrics(
    run_id: str,
    epoch_id: str,
    coprocessor_id: str,
    task_class: str = "",
) -> dict[str, Any]:
    """Return measured feedback outcomes for operator/controller decisions."""
    state = get_state()
    authorization_error = _require_capability(
        state, run_id, "read_routes", epoch_id=epoch_id, controller_only=True,
    )
    if authorization_error:
        return {"available": False, "error": authorization_error}
    active = state.get_active_epoch(run_id)
    if active is None or str(active.get("epoch_id")) != epoch_id:
        return {"available": False, "error": "epoch is not active for this run"}
    return {
        "available": True,
        "metrics": state.get_coprocessor_outcome_metrics(
            coprocessor_id,
            task_class=task_class or None,
            run_id=run_id,
        ),
    }


async def invoke_specialist(
    action_id: str,
    claim_token: str,
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility alias for ``invoke_coprocessor``."""
    return await invoke_coprocessor(action_id, claim_token, packet)


@control_mcp.tool()
async def request_feedback_checkpoint(
    checkpoint: str,
    packet: dict[str, Any],
    action_id: str | None = None,
) -> dict[str, Any]:
    """Request the current automatic-feedback checkpoint policy.

    Hooks use the authenticated loopback endpoint because they must return
    ``additionalContext`` to Claude Code synchronously.  This MCP tool calls
    the same service for controllers or workers that explicitly want to
    inspect/trigger a checkpoint.
    """
    state = get_state()
    run_id = _get_current_run_id()
    principal = _get_current_principal()
    if not run_id or principal is None:
        return {"status": "ignored", "reason": "MCP principal or run ID is missing"}
    if not ("read_routes" in principal.allowed_capabilities
            or "report_worker_result" in principal.allowed_capabilities):
        return {"status": "denied", "reason": "MCP principal cannot request feedback"}
    from enhanced_router.feedback_service import request_feedback_checkpoint as run_checkpoint

    body = dict(packet)
    body.update({
        "run_id": run_id,
        "epoch_id": body.get("epoch_id"),
        "checkpoint": checkpoint,
        "agent_id": body.get("agent_id") or principal.agent_id,
        "action_id": action_id or body.get("action_id"),
    })
    return await run_checkpoint(state=state, body=body)


@control_mcp.tool()
async def get_execution(
    run_id: str,
    epoch_id: str,
    execution_id: str,
) -> dict[str, Any]:
    """Read one execution and enforce run/epoch ownership."""
    state = get_state()
    authorization_error = _require_capability(
        state, run_id, "read_routes", epoch_id=epoch_id,
    )
    if authorization_error:
        return {"error": authorization_error}
    principal = _get_current_principal()
    if principal is not None and principal.principal_kind != "controller":
        if principal.execution_id != execution_id:
            return {"error": "execution is not owned by the authenticated principal"}
    execution = state.get_agent_execution_scoped(run_id, epoch_id, execution_id)
    return execution or {"error": f"Execution '{execution_id}' not found"}


@control_mcp.tool()
async def get_execution_events(
    run_id: str,
    epoch_id: str,
    execution_id: str,
    after_seq: int = 0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read ordered execution events after a sequence number."""
    state = get_state()
    authorization_error = _require_capability(
        state, run_id, "read_routes", epoch_id=epoch_id,
    )
    if authorization_error:
        return [{"error": authorization_error}]
    principal = _get_current_principal()
    if principal is not None and principal.principal_kind != "controller" \
            and principal.execution_id != execution_id:
        return [{"error": "execution is not owned by the authenticated principal"}]
    return state.get_execution_events(
        run_id, epoch_id, execution_id, after_seq=after_seq, limit=limit,
    )


@control_mcp.tool()
async def cancel_execution(
    run_id: str,
    epoch_id: str,
    execution_id: str,
) -> dict[str, Any]:
    """Cancel a running bounded coprocessor execution."""
    state = get_state()
    authorization_error = _require_capability(
        state, run_id, "cancel_execution", epoch_id=epoch_id, controller_only=True,
    )
    if authorization_error:
        return {"cancelled": False, "error": authorization_error}
    execution = state.get_agent_execution_scoped(run_id, epoch_id, execution_id)
    if execution is None:
        return {"cancelled": False, "error": f"Execution '{execution_id}' not found"}
    if execution.get("execution_kind") not in {"sidecar_call", "coprocessor_call"}:
        return {"cancelled": False, "error": "only coprocessor executions can be cancelled here"}
    from enhanced_router.coprocessor_executor import get_coprocessor_executor
    result = await get_coprocessor_executor(state).cancel(execution_id)
    return {"cancelled": True, "execution": result}


@control_mcp.tool()
async def retry_execution(
    run_id: str,
    epoch_id: str,
    execution_id: str,
) -> dict[str, Any]:
    """Retry a failed coprocessor within its bounded retry budget."""
    state = get_state()
    authorization_error = _require_capability(
        state, run_id, "retry_execution", epoch_id=epoch_id, controller_only=True,
    )
    if authorization_error:
        return {"started": False, "error": authorization_error}
    execution = state.get_agent_execution_scoped(run_id, epoch_id, execution_id)
    if execution is None:
        return {"started": False, "error": f"Execution '{execution_id}' not found"}
    from enhanced_router.coprocessor_executor import get_coprocessor_executor
    try:
        retried = await get_coprocessor_executor(state).retry(execution_id)
    except (ValueError, WorkflowStateError) as exc:
        return {"started": False, "error": str(exc)}
    return {"started": True, "execution": retried, "retry_of": execution_id}


@control_mcp.tool()
async def get_orchestration_status() -> dict[str, Any]:
    """Return controller-visible phase, execution, queue, and provider state."""
    state: RouteState = get_state()
    run_id = _get_current_run_id()
    if not run_id:
        return {"ready": False, "error": "No active run ID"}
    active = state.get_active_epoch(run_id)
    if not active:
        return {"ready": False, "run_id": run_id, "error": "No active epoch"}
    epoch_id = str(active["epoch_id"])
    from enhanced_router.backends import provider_admission_snapshots

    supervisor = get_litellm_supervisor()
    litellm = None
    if supervisor is not None:
        litellm = {
            "runtime": supervisor.deployment_telemetry()
            if hasattr(supervisor, "deployment_telemetry") else None,
            "deployments": state.get_litellm_deployment_telemetry(limit=20),
            "request_attributions": state.get_litellm_request_attributions(limit=50),
        }
    from enhanced_router.presentation import orchestration_snapshot

    return {
        "ready": True,
        "run_id": run_id,
        "epoch_id": epoch_id,
        "phases": state.get_workflow_phases(run_id, epoch_id),
        "executions": state.get_agent_executions(run_id, epoch_id=epoch_id),
        "provider_reservations": state.get_provider_reservations(
            run_id=run_id, epoch_id=epoch_id,
        ),
        "provider_admission": provider_admission_snapshots(),
        "litellm": litellm,
        "runnable_actions": state.get_runnable_actions(run_id, epoch_id),
        "orchestration": orchestration_snapshot(state, run_id, epoch_id),
    }


@control_mcp.tool()
async def get_orchestration_plan(run_id: str, epoch_id: str) -> dict[str, Any]:
    """Return the durable execution plan without creating or claiming work.

    Native workflow drivers use this as a read-only projection.  SQLite/MCP
    remains authoritative for phase readiness, package ownership, action
    claims, provider admission, and completion; the projection never becomes
    a second scheduler.
    """
    state: RouteState = get_state()
    error = _authorize_explicit_run(run_id)
    if error:
        return {"ready": False, "error": error}
    capability_error = _require_capability(
        state, run_id, "read_routes", epoch_id=epoch_id,
    )
    if capability_error:
        return {"ready": False, "error": capability_error}
    active = state.get_active_epoch(run_id)
    if active is None or str(active.get("epoch_id")) != str(epoch_id):
        return {"ready": False, "error": "epoch is not active for this run"}
    from enhanced_router.backends import provider_admission_snapshots
    from enhanced_router.presentation import orchestration_snapshot
    orchestration = orchestration_snapshot(state, run_id, epoch_id)
    wave = state.get_runnable_action_wave(run_id, epoch_id, limit=3)
    return {
        "ready": True,
        "plan_version": "orchestration-plan-v1",
        "scheduler_authority": "claudebrigade-mcp",
        "run_id": run_id,
        "epoch_id": epoch_id,
        "workflow_id": active.get("workflow_id"),
        "escalation": state.get_escalation_state(run_id, epoch_id),
        "phases": state.get_workflow_phases(run_id, epoch_id),
        "work_packages": state.get_work_packages(run_id, epoch_id),
        "runnable_actions": wave,
        "wave": {
            "count": len(wave),
            "action_ids": [item.get("action_id") for item in wave],
            "package_ids": sorted({
                str(item.get("package_id"))
                for item in wave
                if item.get("package_id")
            }),
            "worker_kinds": sorted({
                str(item.get("worker_kind") or item.get("action_kind"))
                for item in wave
            }),
        },
        "capacity": state.get_run_resource_capacity(run_id, epoch_id),
        "provider_admission": provider_admission_snapshots(),
        "coverage": state.get_requirement_coverage(run_id, epoch_id),
        "orchestration": orchestration,
        "wait_reasons": orchestration.get("wait_reasons", []),
    }


@control_mcp.tool()
async def get_escalation_state(run_id: str, epoch_id: str) -> dict[str, Any]:
    """Return the durable escalation state for one run/epoch."""
    state = get_state()
    error = _authorize_explicit_run(run_id)
    if error:
        return {"ready": False, "error": error}
    active = state.get_active_epoch(run_id)
    if active is None or str(active.get("epoch_id")) != str(epoch_id):
        return {"ready": False, "error": "epoch is not active for this run"}
    return state.get_escalation_state(run_id, epoch_id)


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
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", controller_only=True,
    )
    if authorization_error:
        return {"changed": False, "error": authorization_error}

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

    authorization_error = _require_capability(
        state, run_id, "complete_workflow", controller_only=True,
    )
    if authorization_error:
        return {"changed": False, "error": authorization_error}

    active = state.get_active_epoch(run_id)
    if not active:
        return {"changed": False, "error": "No active epoch"}

    profile_id = active.get("profile_id") or "hybrid"
    registry: ModelRegistry = _get_registry()
    profile = registry.get_profile(profile_id)
    target = profile.route_target(role)
    default_model = target.model

    state.set_role_route(
        run_id=run_id,
        epoch_id=active["epoch_id"],
        role=role,
        model_id=default_model,
        source="mcp-reset",
        reason=reason,
        endpoint_override=None if target.endpoint == "auto" else target.endpoint,
    )
    return {
        "changed": True,
        "model_id": default_model,
        "profile_id": profile_id,
        "effective_for": "next_subagent_spawn",
    }


@control_mcp.tool()
async def record_binding_command(
    run_id: str,
    epoch_id: str,
    command_type: str,
    reason: str,
    claude_session_id: str = "",
    claude_agent_id: str | None = None,
    expected_binding_version: int | None = None,
    requested_model_id: str | None = None,
    requested_role: str | None = None,
) -> dict[str, Any]:
    """Record a binding command without applying it.

    Use for logging / audit trails (release events, manual commands, etc.).
    The command remains ``'pending'`` until applied via ``apply_binding_command``.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"recorded": False, "error": authorization_error}
    authorization_error = _require_capability(
        get_state(), run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"recorded": False, "error": authorization_error}
    from enhanced_router.state import VALID_COMMAND_TYPES

    if command_type not in VALID_COMMAND_TYPES:
        return {
            "recorded": False,
            "error": (
                f"Invalid command_type {command_type!r}. "
                f"Must be one of {VALID_COMMAND_TYPES}"
            ),
        }

    import uuid

    state: RouteState = get_state()
    command_id = f"cmd-{uuid.uuid4().hex[:12]}"

    try:
        result = state.record_binding_command(
            command_id=command_id,
            run_id=run_id,
            epoch_id=epoch_id,
            command_type=command_type,
            actor_type="mcp",
            reason=reason,
            claude_session_id=claude_session_id or "",
            claude_agent_id=claude_agent_id or "",
            expected_binding_version=expected_binding_version,
            requested_model_id=requested_model_id,
            requested_role=requested_role,
        )
        return {
            "recorded": True,
            "command_id": command_id,
            "status": result["status"],
        }
    except Exception as exc:
        return {"recorded": False, "error": str(exc)}


@control_mcp.tool()
async def apply_binding_command(
    command_id: str,
    actor_type: str,
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Apply a pending binding command, recording the actor.

    Changes the command status from ``'pending'`` to ``'applied'``.
    """
    state: RouteState = get_state()
    try:
        command = state.get_binding_command_by_id(command_id)
        if command is None:
            return {"applied": False, "error": "binding command not found"}
        authorization_error = _require_capability(
            state, str(command["run_id"]), "complete_workflow",
            epoch_id=str(command["epoch_id"]), controller_only=True,
        )
        if authorization_error:
            return {"applied": False, "error": authorization_error}
        result = state.apply_binding_command(
            command_id=command_id,
            actor_type=actor_type,
            actor_id=actor_id or "",
        )
        return {
            "applied": True,
            "command_id": command_id,
            "status": result["status"],
            "command_type": result["command_type"],
            "reason": result["reason"],
        }
    except ValueError as exc:
        return {"applied": False, "error": str(exc)}
    except Exception as exc:
        return {"applied": False, "error": str(exc)}


@control_mcp.tool()
async def reload_catalog(
    reason: str,
) -> dict[str, Any]:
    """Reload the LiteLLM catalog from the current model registry.

    Compiles the current ``models.yaml`` into a new LiteLLM proxy
    configuration, starts a new generation child process, and atomically
    activates it.  The previous generation is drained and eventually
    killed once its pinned agent bindings release.

    Requires the ``LiteLLMSupervisor`` to be initialised (set via
    ``set_litellm_supervisor``).
    """
    global _litellm_supervisor_instance
    state = get_state()
    run_id = _get_current_run_id()
    if not run_id:
        return {"changed": False, "generation": None, "error": "No active run ID"}
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", controller_only=True,
    )
    if authorization_error:
        return {"changed": False, "generation": None, "error": authorization_error}
    if _litellm_supervisor_instance is None:
        return {
            "changed": False,
            "generation": None,
            "error": "LiteLLM supervisor not initialised. Ensure BRIGADE_LITELLM_KEY is set.",
        }

    registry: ModelRegistry = _get_registry()
    registry.reload()
    reg_hash = registry.registry_hash()

    from enhanced_router.litellm_config import generate_litellm_config

    referenced_ids = registry.referenced_model_ids_for_active_runs(state)
    config_text = generate_litellm_config(registry.models, referenced_ids=referenced_ids)

    try:
        result = await _litellm_supervisor_instance.reload(
            registry_hash=reg_hash,
            models=registry.models,
            config_text=config_text,
            reason=reason or "mcp-reload",
            referenced_ids=referenced_ids,
        )
        if result.get("changed"):
            return {
                "changed": True,
                "generation": result["generation"],
                "port": result.get("port"),
                "previous_generation": result.get("previous_generation"),
                "registry_hash": reg_hash,
            }
        return {
            "changed": False,
            "generation": result["generation"],
            "reason": result.get("reason", "no change"),
            "registry_hash": reg_hash,
        }
    except Exception as exc:
        return {
            "changed": False,
            "generation": None,
            "error": str(exc),
            "registry_hash": reg_hash,
        }


@control_mcp.tool()
async def begin_task(
    run_id: str,
    session_id: str = "",
    cwd: str = ".",
    workflow_id: str = "normal",
    profile_id: str = "hybrid",
    signals: list[str] | None = None,
    force_workflow: str | None = None,
) -> dict[str, Any]:
    """Authoritative task start — classifies, selects workflow, persists phases.

    Delegates to RouteState.begin_task() for atomic run + epoch +
    phase creation.  Returns the immutable task contract.

    Returns the contract dict with: run_id, epoch_id, workflow_id,
    profile_id, phases, specification_hash.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"error": authorization_error}
    authorization_error = _require_capability(
        get_state(), run_id, "complete_workflow", controller_only=True,
        require_binding=False,
    )
    if authorization_error:
        return {"error": authorization_error}
    state: RouteState = get_state()
    try:
        result = state.begin_task(
            run_id=run_id,
            session_id=session_id,
            cwd=cwd,
            workflow_id=workflow_id,
            profile_id=profile_id,
            signals=signals or [],
            force_workflow=force_workflow,
        )
        return result
    except ValueError as exc:
        return {"error": str(exc)}


@control_mcp.tool()
async def get_task_state(
    run_id: str,
    epoch_id: str,
) -> dict[str, Any]:
    """Return the current task state: phases, statuses, and completion summary."""
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"error": authorization_error}
    state: RouteState = get_state()
    phases = state.get_workflow_phases(run_id, epoch_id)

    all_done = all(p["status"] in ("completed", "skipped") for p in phases)
    current = [p for p in phases if p["status"] == "active"]

    return {
        "phase_count": len(phases),
        "completed": sum(1 for p in phases if p["status"] == "completed"),
        "skipped": sum(1 for p in phases if p["status"] == "skipped"),
        "active": [p["phase_id"] for p in current],
        "pending": [p["phase_id"] for p in phases if p["status"] == "pending"],
        "all_phases_complete": all_done,
        "phases": [
            {
                "phase_id": p["phase_id"],
                "status": p["status"],
                "required": bool(p["required"]),
                "mutating": bool(p["mutating"]),
                "actor": p["actor"],
            }
            for p in phases
        ],
    }


@control_mcp.tool()
async def publish_task_contract(
    run_id: str,
    epoch_id: str,
    contract: dict[str, Any],
    source: str = "controller",
    replace_unapproved: bool = False,
) -> dict[str, Any]:
    """Publish the scope/acceptance contract for an epoch.

    Before approval, the controller may replace an earlier intake/draft
    contract.  Once approved, the contract is immutable for the epoch.
    """
    state = get_state()
    error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True,
    )
    if error:
        return {"published": False, "error": error}
    try:
        record = state.publish_task_contract(
            run_id, epoch_id, contract, source=source,
            replace_draft=replace_unapproved,
        )
        return {"published": True, "contract": record}
    except (ValueError, KeyError) as exc:
        return {"published": False, "error": str(exc)}


@control_mcp.tool()
async def approve_task_contract(run_id: str, epoch_id: str, approved_by: str = "controller") -> dict[str, Any]:
    """Approve the published contract before package mutation is admitted."""
    state = get_state()
    error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True,
    )
    if error:
        return {"approved": False, "error": error}
    try:
        record = state.approve_task_contract(run_id, epoch_id, approved_by=approved_by)
        contract_action_id = f"contract:{run_id}:{epoch_id}"
        # If approval was performed through the scheduler's first-class
        # contract action, close that claim as part of the same controller
        # operation.  Direct state/API callers remain compatible when no
        # contract action was claimed.
        consumed = state.consume_controller_action(run_id, epoch_id, contract_action_id)
        if consumed is not None:
            state.finish_controller_action(run_id, epoch_id, contract_action_id, "completed")
        return {"approved": True, "contract": record, "action_id": contract_action_id if consumed else None}
    except ValueError as exc:
        return {"approved": False, "error": str(exc)}


@control_mcp.tool()
async def add_requirement(
    run_id: str,
    epoch_id: str,
    statement: str,
    category: str = "functional",
    mandatory: bool = True,
    acceptance: dict[str, Any] | None = None,
    requirement_id: str | None = None,
    risk: str = "normal",
) -> dict[str, Any]:
    """Add one requirement to the controller-owned coverage ledger."""
    state = get_state()
    error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True,
    )
    if error:
        return {"added": False, "error": error}
    try:
        requirement = state.add_requirement(
            run_id, epoch_id, statement, category=category, mandatory=mandatory,
            acceptance=acceptance, requirement_id=requirement_id, risk=risk,
        )
        return {"added": True, "requirement": requirement}
    except ValueError as exc:
        return {"added": False, "error": str(exc)}


@control_mcp.tool()
async def update_requirement(requirement_id: str, status: str, reason: str = "") -> dict[str, Any]:
    """Adjudicate requirement coverage status."""
    state = get_state()
    requirement = None
    # The ID is globally unique; derive scope from the row before auth.
    conn = state._new_conn()
    try:
        row = conn.execute("SELECT run_id, epoch_id FROM requirements WHERE requirement_id=?", (requirement_id,)).fetchone()
        if row is None:
            return {"updated": False, "error": "requirement not found"}
        error = _require_capability(state, str(row[0]), "complete_workflow", epoch_id=str(row[1]), controller_only=True)
        if error:
            return {"updated": False, "error": error}
    finally:
        conn.close()
    try:
        requirement = state.update_requirement(requirement_id, status=status, reason=reason)
        return {"updated": requirement is not None, "requirement": requirement}
    except ValueError as exc:
        return {"updated": False, "error": str(exc)}


@control_mcp.tool()
async def link_requirement_evidence(
    requirement_id: str,
    evidence_kind: str,
    evidence_ref: str,
    evidence_digest: str | None = None,
    valid: bool | None = None,
) -> dict[str, Any]:
    """Attach a test, changeset, finding, or audit artifact to a requirement."""
    state = get_state()
    conn = state._new_conn()
    try:
        row = conn.execute("SELECT run_id, epoch_id FROM requirements WHERE requirement_id=?", (requirement_id,)).fetchone()
        if row is None:
            return {"linked": False, "error": "requirement not found"}
        error = _require_capability(state, str(row[0]), "complete_workflow", epoch_id=str(row[1]), controller_only=True)
        if error:
            return {"linked": False, "error": error}
    finally:
        conn.close()
    return {"linked": True, "evidence": state.link_requirement_evidence(
        requirement_id, evidence_kind=evidence_kind, evidence_ref=evidence_ref,
        evidence_digest=evidence_digest, valid=valid,
    )}


@control_mcp.tool()
async def get_requirement_coverage(run_id: str, epoch_id: str) -> dict[str, Any]:
    """Return the authoritative requirement/evidence coverage projection."""
    error = _authorize_explicit_run(run_id)
    if error:
        return {"error": error}
    return get_state().get_requirement_coverage(run_id, epoch_id)


@control_mcp.tool()
async def record_coverage_audit(
    run_id: str,
    epoch_id: str,
    complete: bool,
    missing: list[str] | None = None,
    auditor: str = "controller",
    contract_version: int | None = None,
    workspace_generation: int | None = None,
    workspace_digest: str | None = None,
) -> dict[str, Any]:
    """Persist the controller's requirement-coverage audit.

    A successful coverage audit is evidence that the current contract and
    canonical workspace were checked together.  It is intentionally separate
    from ``get_requirement_coverage``: computed coverage is not itself an
    attestation that a controller performed the final audit.
    """
    state = get_state()
    error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True,
    )
    if error:
        return {"recorded": False, "error": error}
    try:
        audit = state.record_coverage_audit(
            run_id,
            epoch_id,
            complete=complete,
            missing=missing,
            auditor=auditor,
            contract_version=contract_version,
            workspace_generation=workspace_generation,
            workspace_digest=workspace_digest,
        )
    except ValueError as exc:
        return {"recorded": False, "error": str(exc)}
    return {"recorded": True, "audit": audit}


@control_mcp.tool()
async def add_ambiguity(
    run_id: str,
    epoch_id: str,
    question: str,
    options: list[str] | None = None,
    ambiguity_id: str | None = None,
) -> dict[str, Any]:
    """Record an unresolved contract question before implementation."""
    state = get_state()
    error = _require_capability(state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True)
    if error:
        return {"added": False, "error": error}
    try:
        return {"added": True, "ambiguity": state.add_ambiguity(
            run_id, epoch_id, question, options=options, ambiguity_id=ambiguity_id,
        )}
    except ValueError as exc:
        return {"added": False, "error": str(exc)}


@control_mcp.tool()
async def resolve_ambiguity(
    ambiguity_id: str,
    resolution: str,
    status: str = "resolved",
) -> dict[str, Any]:
    """Resolve or defer a recorded contract ambiguity."""
    state = get_state()
    conn = state._new_conn()
    try:
        row = conn.execute("SELECT run_id, epoch_id FROM ambiguities WHERE ambiguity_id=?", (ambiguity_id,)).fetchone()
        if row is None:
            return {"resolved": False, "error": "ambiguity not found"}
        error = _require_capability(state, str(row[0]), "complete_workflow", epoch_id=str(row[1]), controller_only=True)
        if error:
            return {"resolved": False, "error": error}
    finally:
        conn.close()
    try:
        result = state.resolve_ambiguity(ambiguity_id, resolution, status=status)
        return {"resolved": result is not None, "ambiguity": result}
    except ValueError as exc:
        return {"resolved": False, "error": str(exc)}


@control_mcp.tool()
async def publish_work_packages(
    run_id: str,
    epoch_id: str,
    phase_id: str,
    packages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish disjoint package contracts consumed by the native scheduler."""
    state = get_state()
    error = _require_capability(state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True)
    if error:
        return {"published": False, "error": error}
    published: list[dict[str, Any]] = []
    try:
        for package in packages:
            published.append(state.publish_work_package(
                run_id, epoch_id, phase_id, str(package.get("objective") or ""),
                package_id=package.get("package_id"), path_scope=package.get("path_scope"),
                requirement_ids=package.get("requirements"), dependencies=package.get("dependencies"),
                acceptance=package.get("acceptance"), required_tests=package.get("required_tests"),
                prohibited_paths=package.get("prohibited_paths"), contract_digest=package.get("contract_digest"),
                contract_version=int(package.get("contract_version") or 1),
                prompt_contract=package.get("prompt_contract"),
                display_name=package.get("display_name"), summary=package.get("summary"),
                risk=str(package.get("risk") or "normal"), can_run_parallel=bool(package.get("can_run_parallel", True)),
            ))
        return {"published": True, "packages": published}
    except (ValueError, KeyError) as exc:
        return {"published": False, "error": str(exc)}


@control_mcp.tool()
async def get_work_packages(run_id: str, epoch_id: str, phase_id: str | None = None) -> dict[str, Any]:
    """List persisted package contracts and their statuses."""
    error = _authorize_explicit_run(run_id)
    if error:
        return {"error": error}
    return {"packages": get_state().get_work_packages(run_id, epoch_id, phase_id)}


@control_mcp.tool()
async def evaluate_escalation(
    run_id: str,
    epoch_id: str,
    failed_tests: int = 0,
    unresolved_findings: int = 0,
    missing_requirements: int = 0,
    changed_files: int = 0,
    changed_security_paths: bool = False,
    repeated_repairs: int = 0,
    provider_failures: int = 0,
    evidence_incomplete: bool = False,
    deadline_pressure: bool = False,
) -> dict[str, Any]:
    """Evaluate an adaptive escalation without applying it."""
    from enhanced_router.escalation_policy import EscalationInputs, evaluate_escalation as decide
    state = get_state()
    active = state.get_active_epoch(run_id)
    current = str(
        (active or {}).get("escalation_level")
        or (active or {}).get("workflow_id")
        or "normal"
    )
    decision = decide(EscalationInputs(
        current_tier=current, failed_tests=failed_tests, unresolved_findings=unresolved_findings,
        missing_requirements=missing_requirements, changed_files=changed_files,
        changed_security_paths=changed_security_paths, repeated_repairs=repeated_repairs,
        provider_failures=provider_failures, evidence_incomplete=evidence_incomplete,
        deadline_pressure=deadline_pressure,
    ))
    return {"decision": decision.__dict__}


@control_mcp.tool()
async def apply_escalation(run_id: str, epoch_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    """Apply a previously evaluated escalation through the controller authority."""
    from enhanced_router.escalation_policy import EscalationDecision
    state = get_state()
    error = _require_capability(state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True)
    if error:
        return {"applied": False, "error": error}
    try:
        parsed = EscalationDecision(
            should_escalate=bool(decision.get("should_escalate")),
            from_tier=str(decision["from_tier"]), to_tier=str(decision["to_tier"]),
            reasons=tuple(str(item) for item in decision.get("reasons", [])),
            policy_digest=str(decision["policy_digest"]),
        )
        return {"applied": True, "state": state.escalate_epoch(run_id, epoch_id, parsed)}
    except (KeyError, ValueError) as exc:
        return {"applied": False, "error": str(exc)}


@control_mcp.tool()
async def escalate_workflow(
    run_id: str, epoch_id: str, decision: dict[str, Any],
) -> dict[str, Any]:
    """Apply an escalation; explicit name for controller workflow clients."""
    return await apply_escalation(run_id, epoch_id, decision)


@control_mcp.tool()
async def acknowledge_escalation(
    run_id: str, epoch_id: str, reason: str = "",
) -> dict[str, Any]:
    """Acknowledge compensating review and resume admitted mutation."""
    state = get_state()
    error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id, controller_only=True,
    )
    if error:
        return {"acknowledged": False, "error": error}
    principal = _get_current_principal()
    actor = str(principal.agent_id if principal and principal.agent_id else "controller")
    try:
        result = state.acknowledge_escalation(
            run_id, epoch_id, acknowledged_by=actor,
        )
        return {"acknowledged": True, "reason": reason, "state": result}
    except (KeyError, ValueError, WorkflowStateError) as exc:
        return {"acknowledged": False, "error": str(exc)}


def _proposal_route_targets(proposal: dict) -> dict[str, tuple[str, str]]:
    """Extract model/endpoint targets without granting fastpath authority."""
    raw = json.loads(str(proposal.get("parsed_proposal_json") or "{}"))
    # Detached proposals retain both the model-facing compact ``routes`` and
    # the router-resolved ``resolved_routes``.  Prefer the latter: the MCP
    # controller may accept a proposal long after the original candidate map
    # has left memory, but it must never trust an unresolvable candidate alias.
    routes = raw.get("resolved_routes") or raw.get("routes")
    if not isinstance(routes, dict):
        raise ValueError("route proposal has no routes object")
    targets: dict[str, tuple[str, str]] = {}
    for role, target in routes.items():
        if role not in {"recon", "implementer", "adversary", "repairer"}:
            raise ValueError(f"route proposal contains unknown role '{role}'")
        if not isinstance(target, dict):
            raise ValueError(f"route proposal target for '{role}' is not an object")
        model_id = target.get("model") or target.get("preferred_logical_model")
        if not model_id and target.get("candidate_id"):
            raise ValueError(
                f"route proposal target for '{role}' has an unresolved candidate "
                "and cannot be applied"
            )
        endpoint = target.get("endpoint", "auto")
        if not isinstance(model_id, str) or not model_id:
            continue
        if endpoint != "auto":
            raise ValueError(
                "fastpath cannot choose a physical endpoint; use an explicit "
                "controller route change for endpoint overrides"
            )
        targets[role] = (model_id, "auto")
    return targets


@control_mcp.tool()
async def get_route_proposal(proposal_id: str) -> dict[str, Any]:
    """Return a fastpath proposal for the authenticated main controller."""
    run_id = _get_current_run_id()
    if not run_id:
        return {"found": False, "error": "No active run ID"}
    state = get_state()
    authorization_error = _require_capability(state, run_id, "read_routes")
    if authorization_error:
        return {"found": False, "error": authorization_error}
    proposal = state.get_route_proposal_for_run(proposal_id, run_id)
    if proposal is None:
        return {"found": False, "error": "route proposal not found"}
    return {"found": True, "proposal": proposal}


@control_mcp.tool()
async def accept_route_proposal(
    proposal_id: str,
    reason: str,
    apply_routes: bool = False,
    apply_plan: bool | None = None,
) -> dict[str, Any]:
    """Accept an advisory proposal, optionally applying routes and its plan.

    DiffusionGemma never applies routes itself.  Only the authenticated main
    controller may accept a proposal, and endpoint selection remains in the
    normal certified/cache-aware resolver.

    ``apply_plan`` defaults to ``apply_routes`` for compatibility with the
    original one-switch MCP contract: a controller that explicitly approves
    route application also gets the proposal's still-pending fanout hints.
    """
    run_id = _get_current_run_id()
    if not run_id:
        return {"accepted": False, "error": "No active run ID"}
    state = get_state()
    controller_error = _require_main_controller_binding(
        state, run_id, capability="complete_workflow",
    )
    if controller_error:
        return {"accepted": False, "error": controller_error}
    proposal = state.get_route_proposal_for_run(proposal_id, run_id)
    if proposal is None:
        return {"accepted": False, "error": "route proposal not found"}
    try:
        targets = _proposal_route_targets(proposal)
        active = state.get_active_epoch(run_id)
        if active is None:
            raise ValueError("no active epoch for proposal")
        epoch_id = str(active["epoch_id"])
        registry = _get_registry()
        parsed_proposal = json.loads(str(proposal.get("parsed_proposal_json") or "{}"))
        if not isinstance(parsed_proposal, dict):
            raise ValueError("route proposal payload is not an object")
        proposal_registry_hash = parsed_proposal.get("registry_hash")
        if proposal_registry_hash and str(proposal_registry_hash) != registry.registry_hash():
            raise ValueError(
                "route proposal was generated from a stale registry/catalog snapshot"
            )
        proposal_candidate_digest = parsed_proposal.get("candidate_set_digest")
        if proposal_candidate_digest:
            from enhanced_router.fastpath import build_route_candidates
            import hashlib

            offered_candidates, _ = build_route_candidates(
                registry, ["recon", "implementer", "adversary", "repairer"],
            )
            current_candidate_digest = hashlib.sha256(
                json.dumps(
                    {
                        "registry_hash": registry.registry_hash(),
                        "candidates": offered_candidates,
                        "minimum_tier": parsed_proposal.get("minimum_tier", "normal"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if str(proposal_candidate_digest) != current_candidate_digest:
                raise ValueError(
                    "route proposal was generated from a stale candidate-set snapshot"
                )
        should_apply_plan = apply_routes if apply_plan is None else apply_plan
        plan_result: dict[str, Any] = {"applied": False, "changes": []}
        if should_apply_plan:
            plan_result = state.apply_fastpath_plan(
                run_id, epoch_id, parsed_proposal,
            )
        if apply_routes:
            for role, (model_id, endpoint) in targets.items():
                spec = registry.get_model(model_id)
                if not spec.enabled or role not in spec.allowed_roles:
                    raise ValueError(
                        f"model '{model_id}' is not enabled for role '{role}'"
                    )
                if role in {"implementer", "repairer"} and spec.capabilities.write_tool_certified is not True:
                    raise ValueError(
                        f"model '{model_id}' is not write-tool certified for '{role}'"
                    )
                state.set_role_route(
                    run_id, epoch_id, role, model_id,
                    source="controller-fastpath-accept",
                    reason=reason,
                    endpoint_override=None if endpoint == "auto" else endpoint,
                )
        result = state.set_route_proposal_disposition(
            proposal_id, "accepted", reason=reason, epoch_id=epoch_id,
        )
        return {
            "accepted": result is not None,
            "proposal": result,
            "applied_routes": apply_routes,
            "applied_plan": plan_result["applied"],
            "plan_changes": plan_result["changes"],
            "proposed_workflow_tier": parsed_proposal.get("workflow_tier"),
            "routes": targets,
        }
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return {"accepted": False, "error": str(exc)}


@control_mcp.tool()
async def reject_route_proposal(proposal_id: str, reason: str) -> dict[str, Any]:
    """Reject an advisory fastpath proposal from the authenticated controller."""
    run_id = _get_current_run_id()
    if not run_id:
        return {"rejected": False, "error": "No active run ID"}
    state = get_state()
    controller_error = _require_main_controller_binding(
        state, run_id, capability="complete_workflow",
    )
    if controller_error:
        return {"rejected": False, "error": controller_error}
    proposal = state.get_route_proposal_for_run(proposal_id, run_id)
    if proposal is None:
        return {"rejected": False, "error": "route proposal not found"}
    result = state.set_route_proposal_disposition(
        proposal_id, "rejected", reason=reason,
    )
    return {"rejected": result is not None, "proposal": result}


@control_mcp.tool()
async def validate_completion(
    run_id: str,
    epoch_id: str,
    workflow_tier: str = "",
    accepted_findings: str = "none",
    route_snapshot_sha256: str = "",
    workflow_id: str = "",
    session_dir: str | None = None,
) -> dict[str, Any]:
    """Validate completion evidence against authoritative state.

    Checks workflow tier consistency, phase sequence, findings
    lifecycle, workspace fingerprint, and route snapshot.

    Returns dict with: valid (bool), reason (str|None),
    snapshot_sha256 (str|None).
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"valid": False, "reason": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"valid": False, "reason": authorization_error}
    parsed = {
        "Workflow-Tier": workflow_tier,
        "Workflow-ID": workflow_id,
        "Accepted-Findings": accepted_findings,
        "Route-Snapshot-SHA256": route_snapshot_sha256,
    }
    sd = Path(session_dir) if session_dir else None
    result = state.validate_completion(run_id, epoch_id, parsed, session_dir=sd)
    return result


@control_mcp.tool()
async def prepare_completion(
    run_id: str,
    epoch_id: str,
    workspace_fingerprint: str,
    ttl_seconds: int = 300,
) -> dict[str, Any]:
    """Issue a one-time completion attestation bound to current state."""
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"prepared": False, "error": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"prepared": False, "error": authorization_error}
    try:
        token = state.prepare_completion_token(
            run_id, epoch_id, workspace_fingerprint, ttl_seconds=ttl_seconds,
        )
    except (ValueError, WorkflowStateError) as exc:
        return {"prepared": False, "error": str(exc)}
    return {"prepared": True, **token}


@control_mcp.tool()
async def get_integration_candidates(
    run_id: str,
    epoch_id: str | None = None,
    disposition: str | None = None,
) -> list[dict[str, Any]]:
    """List shadow-worktree changesets awaiting controller integration."""
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return [{"error": authorization_error}]
    state = get_state()
    authorization_error = (
        _require_capability(state, run_id, "read_routes", epoch_id=epoch_id)
        if epoch_id is not None
        else _require_capability(state, run_id, "read_routes")
    )
    if authorization_error:
        return [{"error": authorization_error}]
    return state.get_integration_candidates(
        run_id=run_id, epoch_id=epoch_id, disposition=disposition,
    )


@control_mcp.tool()
async def integrate_shadow_changeset(
    run_id: str,
    epoch_id: str,
    changeset_id: str,
    controller_approval: bool = False,
) -> dict[str, Any]:
    """Integrate a validated shadow changeset after deterministic preflight.

    Green candidates may be applied automatically.  Yellow candidates require
    explicit controller approval and still must pass ``git apply --check``;
    red candidates are never applied by this operation.  The operation never
    commits, stashes, resets, or performs a three-way merge in the user's
    checkout.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"integrated": False, "error": authorization_error}

    state = get_state()
    controller_error = _require_capability(
        state, run_id, "integrate_changeset", epoch_id=epoch_id,
        controller_only=True,
    )
    if controller_error:
        return {"integrated": False, "error": controller_error}
    candidates = state.get_integration_candidates(run_id=run_id, epoch_id=epoch_id)
    candidate = next(
        (item for item in candidates
         if item.get("changeset_id") == changeset_id),
        None,
    )
    if candidate is None:
        return {"integrated": False, "error": "integration candidate not found"}
    disposition = str(candidate.get("disposition"))
    if disposition == "red":
        return {"integrated": False, "error": "red integration candidate requires recovery"}
    if disposition == "yellow" and not controller_approval:
        return {"integrated": False, "error": "yellow candidate requires controller approval"}
    integration_action_id: str | None = None
    if disposition == "yellow" and controller_approval:
        controller_error = _require_main_controller_binding(
            state, run_id, epoch_id, capability="integrate_changeset",
        )
        if controller_error:
            return {"integrated": False, "error": controller_error}
        action_error = _consume_controller_integration_action(
            state, run_id, epoch_id, str(candidate["candidate_id"]),
        )
        if action_error:
            return {"integrated": False, "error": action_error}
        integration_action_id = f"integration:{candidate['candidate_id']}"

    integration_outcome = "failed"
    try:
        changeset_record = state.get_changeset(changeset_id)
        if changeset_record is None:
            return {"integrated": False, "error": "changeset not found"}
        if changeset_record.get("status") not in {"validated", "proposed"}:
            return {
                "integrated": False,
                "error": f"changeset is already {changeset_record.get('status')}",
            }
        patch_blob = changeset_record.get("patch_blob")
        if not isinstance(patch_blob, (bytes, bytearray)):
            return {"integrated": False, "error": "changeset has no persisted patch"}
        workspace_id = str(changeset_record.get("workspace_id"))
        workspace = state.get_workspace(workspace_id)
        run = state.get_run(run_id)
        if workspace is None or run is None or not run.get("cwd"):
            return {"integrated": False, "error": "canonical workspace state is unavailable"}

        from enhanced_router.shadow_worktree import Changeset, ShadowWorktreeManager, escalate_red_candidate

        try:
            validation = json.loads(str(changeset_record.get("result_json") or "{}"))
            validation = validation.get("validation", validation)
            changed_files = tuple(
                json.loads(str(changeset_record.get("changed_files_json") or "[]"))
            )
            changeset = Changeset(
                changeset_id=changeset_id,
                execution_id=str(changeset_record.get("execution_id")),
                workspace_id=workspace_id,
                base_sha=str(changeset_record.get("base_sha")),
                patch_digest=str(changeset_record.get("patch_digest")),
                patch=bytes(patch_blob),
                changed_files=changed_files,
                validation=validation,
                parent_canonical_generation=(
                    int(changeset_record["parent_canonical_generation"])
                    if changeset_record.get("parent_canonical_generation") is not None
                    else None
                ),
            )
            manager = ShadowWorktreeManager(str(run["cwd"]))
            main_rows = state.get_workspaces(
                run_id=run_id, epoch_id=epoch_id, kind="main", status="active",
            )
            if not main_rows:
                return {"integrated": False, "error": "canonical workspace is not active"}
            result = manager.integrate_green(
                state=state, run_id=run_id, epoch_id=epoch_id, changeset=changeset,
                expected_dirty_patch_hash=str(
                    main_rows[0].get("current_dirty_hash")
                    or main_rows[0].get("dirty_patch_hash")
                    or ""
                ),
            )
            state.update_workspace_status(workspace_id, "merged")
            integration_outcome = "completed"
            # Reclassify the canonical result as well as the worker shadow
            # changeset.  A collection of individually small packages can
            # become cross-subsystem or security-sensitive only after they
            # are combined in the canonical workspace.
            try:
                escalation = state.auto_escalate_after_changeset(
                    run_id, epoch_id, changeset_id, apply=True,
                )
            except Exception as exc:
                # Integration is already durable at this point.  Preserve the
                # successful apply and surface escalation evaluation failure as
                # follow-up evidence rather than mislabeling the changeset red.
                escalation = {"applied": False, "error": str(exc)}
            return {"integrated": True, **result, "escalation": escalation}
        except Exception as exc:
            state.mark_integration_candidate(
                changeset_id, disposition="red", validation={"preflight_error": str(exc)},
            )
            escalate_red_candidate(
                state, run_id=run_id, epoch_id=epoch_id,
                candidate_id=str(candidate["candidate_id"]), changeset_id=changeset_id,
                reason=str(exc), evidence={"preflight_error": str(exc)},
            )
            return {"integrated": False, "error": str(exc), "conflict": True}
    finally:
        if integration_action_id:
            state.finish_controller_action(
                run_id, epoch_id, integration_action_id, integration_outcome,
            )


@control_mcp.tool()
async def resolve_shadow_candidate(
    run_id: str,
    epoch_id: str,
    changeset_id: str,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    """Record the main controller's resolution of a red/yellow candidate.

    ``decision=discard`` closes a candidate after the controller has handled
    the conflict or decided the worker result is not usable.  ``decision=retry``
    leaves it pending so the controller can request a new worker execution.
    This operation does not apply filesystem changes; integration remains a
    separate deterministic preflight operation.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"resolved": False, "error": authorization_error}
    if decision not in {"discard", "retry"}:
        return {"resolved": False, "error": "decision must be 'discard' or 'retry'"}
    state = get_state()
    controller_error = _require_main_controller_binding(
        state, run_id, epoch_id, capability="complete_workflow",
    )
    if controller_error:
        return {"resolved": False, "error": controller_error}
    candidates = state.get_integration_candidates(run_id=run_id, epoch_id=epoch_id)
    candidate = next(
        (item for item in candidates if item.get("changeset_id") == changeset_id),
        None,
    )
    if candidate is None:
        return {"resolved": False, "error": "integration candidate not found"}
    action_error = _consume_controller_integration_action(
        state, run_id, epoch_id, str(candidate["candidate_id"]),
    )
    if action_error:
        return {"resolved": False, "error": action_error}
    integration_action_id = f"integration:{candidate['candidate_id']}"
    was_red = str(candidate.get("disposition")) == "red"
    resolution_outcome = "failed"
    try:
        if decision == "retry":
            state.mark_integration_candidate(
                changeset_id, disposition="pending",
                validation={"controller_decision": "retry", "reason": reason},
            )
            if was_red:
                _resolve_red_candidate_finding(state, str(candidate["candidate_id"]), reason)
            resolution_outcome = "completed"
            return {
                "resolved": True, "status": "pending",
                "next_action": "request a new worker execution",
            }
        state.mark_integration_candidate(
            changeset_id, disposition="resolved",
            validation={"controller_decision": "discard", "reason": reason},
        )
        if was_red:
            _resolve_red_candidate_finding(state, str(candidate["candidate_id"]), reason)
        changeset = state.get_changeset(changeset_id)
        if changeset:
            state.mark_changeset_rejected(changeset_id)
            workspace_id = str(changeset.get("workspace_id"))
            state.update_workspace_status(workspace_id, "discarded")
        resolution_outcome = "completed"
        return {"resolved": True, "status": "resolved", "decision": "discard"}
    finally:
        state.finish_controller_action(
            run_id, epoch_id, integration_action_id, resolution_outcome,
        )


@control_mcp.tool()
async def append_phase_instance(
    run_id: str,
    epoch_id: str,
    template_phase_id: str,
    dependencies: list[str] | None = None,
    supersedes_phase_id: str | None = None,
    trigger_event: str | None = None,
    phase_id: str | None = None,
) -> dict[str, Any]:
    """Append a controller-authorized review/repair phase instance.

    This is the durable mechanism for a second adversarial review or repair
    cycle.  It clones the persisted phase semantics while creating a new
    identity such as ``adversarial-review#2``; ordinary execution retries do
    not masquerade as a new semantic cycle.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"appended": False, "error": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"appended": False, "error": authorization_error}
    try:
        phase = state.append_phase_instance(
            run_id, epoch_id, template_phase_id,
            dependencies=dependencies,
            supersedes_phase_id=supersedes_phase_id,
            trigger_event=trigger_event,
            phase_id=phase_id,
        )
        return {"appended": True, "phase": phase}
    except (ValueError, KeyError) as exc:
        return {"appended": False, "error": str(exc)}


@control_mcp.tool()
async def start_phase(
    run_id: str,
    epoch_id: str,
    phase_id: str,
    actor: str = "",
    evaluate_condition: bool = True,
) -> dict[str, Any]:
    """Start a workflow phase: set status='active'.

    When *evaluate_condition* is True and the phase has a conditional
    expression, the condition is evaluated first.  If the condition is
    not satisfied (e.g. no accepted findings for a repair phase), the
    phase is auto-skipped instead of activated.

    Returns the phase dict with the resulting status.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"error": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"error": authorization_error}

    phases = state.get_workflow_phases(run_id, epoch_id)
    phase_spec = next((p for p in phases if p["phase_id"] == phase_id), None)
    if phase_spec is None:
        return {"error": f"Phase '{phase_id}' not found"}

    condition_json = phase_spec.get("condition_json")
    if evaluate_condition and condition_json:
        import json as _json
        condition = _json.loads(condition_json)
        if condition:
            result = state.skip_conditional_phase(run_id, epoch_id, phase_id, condition)
            return {
                "phase_id": phase_id,
                "status": result["status"],
                "condition_evaluated": True,
                "skipped": result["status"] == "skipped",
                "reason": result.get("result_evidence", ""),
            }

    try:
        principal = _get_current_principal()
        started_by = (
            principal.agent_id or principal.session_id or principal.credential_id or ""
            if principal is not None else ""
        )
        result = state.start_phase(
            run_id, epoch_id, phase_id, actor=actor, principal=started_by,
        )
        return {"phase_id": phase_id, "status": result["status"], "condition_evaluated": False}
    except WorkflowPhaseStateError as exc:
        return {"error": str(exc)}


@control_mcp.tool()
async def complete_phase(
    run_id: str,
    epoch_id: str,
    phase_id: str,
    result_evidence: str = "",
    error: str = "",
) -> dict[str, Any]:
    """Complete a workflow phase with result evidence."""
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"error": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"error": authorization_error}
    try:
        result = state.complete_phase(run_id, epoch_id, phase_id, result_evidence=result_evidence, error=error)
        state.finish_controller_actions_for_phase(
            run_id, epoch_id, phase_id,
            "failed" if error else "completed",
        )
        return {"phase_id": phase_id, "status": result["status"]}
    except WorkflowPhaseStateError as exc:
        return {"error": str(exc)}


@control_mcp.tool()
async def skip_conditional_phase(
    run_id: str,
    epoch_id: str,
    phase_id: str,
    condition: str,
) -> dict[str, Any]:
    """Evaluate a conditional phase and skip it if the condition is not met.

    For example, the ``repair`` phase uses condition ``accepted_findings``:
    if no accepted findings exist, it is auto-skipped.

    Returns the phase dict with the resulting status.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"error": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"error": authorization_error}
    try:
        result = state.skip_conditional_phase(run_id, epoch_id, phase_id, condition)
        return {
            "phase_id": phase_id,
            "status": result["status"],
            "condition_evaluated": True,
            "skipped": result["status"] == "skipped",
        }
    except WorkflowPhaseStateError as exc:
        return {"error": str(exc)}


@control_mcp.tool()
async def record_finding(
    run_id: str,
    epoch_id: str,
    finding_id: str,
    description: str,
    severity: str = "medium",
    category: str = "",
    source_phase_id: str | None = None,
    source_agent_id: str | None = None,
    evidence_json: str | None = None,
) -> dict[str, Any]:
    """Record a structured finding from an adversarial review.

    Returns the created finding record.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"recorded": False, "error": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "report_worker_result", epoch_id=epoch_id,
    )
    if authorization_error:
        return {"recorded": False, "error": authorization_error}
    principal = _get_current_principal()
    if (
        principal is not None
        and principal.principal_kind != "controller"
        and principal.agent_id != source_agent_id
    ):
        return {"recorded": False, "error": "finding source is not owned by the authenticated agent"}
    return state.create_finding(
        finding_id=finding_id,
        run_id=run_id,
        epoch_id=epoch_id,
        description=description,
        severity=severity,
        category=category,
        source_phase_id=source_phase_id,
        source_agent_id=source_agent_id,
        evidence_json=evidence_json,
    )


@control_mcp.tool()
async def adjudicate_finding(
    run_id: str,
    epoch_id: str,
    finding_id: str,
    disposition: str,
    reason: str = "",
    dispositioned_by: str = "",
    repair_agent_id: str | None = None,
    repair_phase_id: str | None = None,
) -> dict[str, Any]:
    """Accept, reject, waive, or mark duplicate a finding.

    Accepted findings with a *repair_agent_id* are flagged for repair.
    Valid dispositions: accepted, rejected, duplicate, waived.
    """
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "adjudicate_finding", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"error": authorization_error}
    if state.get_finding_scoped(run_id, epoch_id, finding_id) is None:
        return {"error": f"Finding '{finding_id}' not found in the requested epoch"}
    result = state.adjudicate_finding(
        finding_id=finding_id,
        disposition=disposition,
        run_id=run_id,
        epoch_id=epoch_id,
        reason=reason,
        dispositioned_by=dispositioned_by,
        repair_agent_id=repair_agent_id,
        repair_phase_id=repair_phase_id,
    )
    if result is None:
        return {"error": f"Finding '{finding_id}' not found"}
    return result


@control_mcp.tool()
async def get_findings(
    run_id: str,
    epoch_id: str | None = None,
    disposition: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List findings for a run, optionally filtered by epoch, disposition, or verification status."""
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return [{"error": authorization_error}]
    state: RouteState = get_state()
    authorization_error = (
        _require_capability(state, run_id, "read_routes", epoch_id=epoch_id)
        if epoch_id is not None
        else _require_capability(state, run_id, "read_routes")
    )
    if authorization_error:
        return [{"error": authorization_error}]
    return state.get_findings(run_id, epoch_id=epoch_id, disposition=disposition, status=status)


@control_mcp.tool()
async def resolve_finding(
    run_id: str,
    epoch_id: str,
    finding_id: str,
    verification_status: str,
    resolution_evidence_json: str = "{}",
) -> dict[str, Any]:
    """Record resolution evidence and verification result for a finding.

    Valid statuses: verified, failed, irrelevant.
    """
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "adjudicate_finding", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"error": authorization_error}
    if state.get_finding_scoped(run_id, epoch_id, finding_id) is None:
        return {"error": f"Finding '{finding_id}' not found in the requested epoch"}
    result = state.resolve_finding(
        finding_id=finding_id,
        verification_status=verification_status,
        resolution_evidence_json=resolution_evidence_json,
        run_id=run_id,
        epoch_id=epoch_id,
    )
    if result is None:
        return {"error": f"Finding '{finding_id}' not found"}
    return result


@control_mcp.tool()
async def create_agent_execution(
    run_id: str,
    epoch_id: str,
    execution_id: str,
    claude_agent_id: str,
    role: str,
    model_id: str,
    phase_id: str | None = None,
    binding_id: int | None = None,
) -> dict[str, Any]:
    """Record the start of an agent execution (spawned subagent).

    Links the Claude agent ID, role, model, binding, and workflow phase.
    """
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return {"error": authorization_error}
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "complete_workflow", epoch_id=epoch_id,
        controller_only=True,
    )
    if authorization_error:
        return {"error": authorization_error}
    return state.create_agent_execution(
        execution_id=execution_id,
        run_id=run_id,
        epoch_id=epoch_id,
        claude_agent_id=claude_agent_id,
        role=role,
        model_id=model_id,
        phase_id=phase_id,
        binding_id=binding_id,
    )


@control_mcp.tool()
async def update_agent_execution(
    run_id: str,
    epoch_id: str,
    execution_id: str,
    status: str | None = None,
    result_type: str | None = None,
    result_summary: str | None = None,
    output_hash: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Update an agent execution with completion data.

    Valid statuses: started, running, completed, failed, timeout, cancelled.
    When status is completed/failed/timeout/cancelled, completed_at is set.

    ``tool_call_count`` and ``total_tokens`` are deliberately not settable
    here. ``tool_call_count`` gates each workflow phase's turn budget
    (create_agent_execution sums it across a phase's executions), and the
    router's own PreToolUse hook already maintains it authoritatively via
    increment_execution_tool_calls -- a self-reported value here would let a
    worker understate its own usage and bypass that budget. Real per-request
    token usage is likewise recorded authoritatively from provider responses
    via record_execution_metrics_for_binding; nothing in the router itself
    relies on a self-reported cumulative total.
    """
    state: RouteState = get_state()
    authorization_error = _require_capability(
        state, run_id, "report_worker_result", epoch_id=epoch_id,
    )
    if authorization_error:
        return {"error": authorization_error}
    execution = state.get_agent_execution_scoped(run_id, epoch_id, execution_id)
    if execution is None:
        return {"error": f"Execution '{execution_id}' not found in the requested epoch"}
    principal = _get_current_principal()
    if (
        principal is not None
        and principal.principal_kind != "controller"
        and principal.execution_id != execution_id
    ):
        return {"error": "execution is not owned by the authenticated principal"}
    try:
        result = state.update_agent_execution(
            execution_id=execution_id,
            status=status,
            result_type=result_type,
            result_summary=result_summary,
            output_hash=output_hash,
            error=error,
        )
    except (ValueError, WorkflowStateError) as exc:
        return {"error": str(exc)}
    if result is None:
        return {"error": f"Execution '{execution_id}' not found"}
    return result


@control_mcp.tool()
async def get_agent_executions(
    run_id: str,
    epoch_id: str | None = None,
    phase_id: str | None = None,
    role: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List agent executions for a run with optional filters."""
    authorization_error = _authorize_explicit_run(run_id)
    if authorization_error:
        return [{"error": authorization_error}]
    state: RouteState = get_state()
    authorization_error = (
        _require_capability(state, run_id, "read_routes", epoch_id=epoch_id)
        if epoch_id is not None
        else _require_capability(state, run_id, "read_routes")
    )
    if authorization_error:
        return [{"error": authorization_error}]
    return state.get_agent_executions(
        run_id, epoch_id=epoch_id, phase_id=phase_id, role=role, status=status,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_litellm_supervisor_instance: Any | None = None


def set_litellm_supervisor(supervisor: Any) -> None:
    """Set the LiteLLM supervisor instance for catalog reload.

    Called during app startup after the supervisor is initialized.
    """
    global _litellm_supervisor_instance
    _litellm_supervisor_instance = supervisor


def get_litellm_supervisor() -> Any | None:
    """Return the current LiteLLM supervisor instance, or None."""
    return _litellm_supervisor_instance


# Context variable set by the MCP auth transport


@dataclass(frozen=True)
class McpPrincipal:
    """Authenticated actor context for one MCP request."""

    principal_kind: str
    run_id: str
    session_id: str | None = None
    agent_id: str | None = None
    execution_id: str | None = None
    allowed_capabilities: frozenset[str] = frozenset()
    credential_id: str | None = None
    authenticated: bool = False


_mcp_principal_var: ContextVar[McpPrincipal | None] = ContextVar(
    "_mcp_principal", default=None,
)

_brigade_run_id_var: ContextVar[str | None] = ContextVar(
    "_brigade_run_id", default=None
)


def _get_current_run_id() -> str | None:
    return _brigade_run_id_var.get()


def _get_current_principal() -> McpPrincipal | None:
    return _mcp_principal_var.get()


def set_current_principal(principal: McpPrincipal | None) -> None:
    """Set the in-process principal used by MCP tools and tests."""
    _mcp_principal_var.set(principal)
    _brigade_run_id_var.set(principal.run_id if principal else None)


def _authorize_explicit_run(run_id: str) -> str | None:
    """Reject cross-run access when transport authentication supplied a run."""
    current = _get_current_run_id()
    if current is not None and run_id != current:
        return f"run_id '{run_id}' does not match the authenticated run"
    principal = _get_current_principal()
    if principal is not None and principal.run_id != run_id:
        return f"run_id '{run_id}' does not match the authenticated principal"
    return None


def _require_capability(
    state: RouteState,
    run_id: str,
    capability: str,
    *,
    epoch_id: str | None = None,
    controller_only: bool = False,
    require_binding: bool = True,
) -> str | None:
    """Authorize a state-changing MCP operation against its actor context."""
    error = _authorize_explicit_run(run_id)
    if error:
        return error
    principal = _get_current_principal()
    if principal is None:
        return "MCP principal is missing"
    if controller_only and principal.principal_kind != "controller":
        return "operation requires the authenticated main controller"
    if capability not in principal.allowed_capabilities:
        return f"MCP principal lacks capability '{capability}'"
    run = state.get_run(run_id)
    if run is None:
        return f"run {run_id!r} was not found"
    if epoch_id is not None:
        active = state.get_active_epoch(run_id)
        if active is None or str(active["epoch_id"]) != str(epoch_id):
            return "target epoch is not the active epoch for this run"
    if principal.authenticated and principal.principal_kind == "controller":
        # The transport has already checked the raw capability, but verify it
        # again against the run row before every mutation so a capability
        # cannot be replayed across runs or epochs.
        if not principal.credential_id or not state.verify_controller_capability(
            run_id, principal.credential_id, session_id=principal.session_id,
        ):
            return "controller capability is invalid or expired"
    if controller_only and require_binding:
        session_id = str(run.get("claude_session_id") or "")
        if not session_id or state.get_controller_binding(run_id, session_id) is None:
            return (
                "controller-only operation requires the active main controller binding"
            )
    return None


def _require_main_controller_binding(
    state: RouteState, run_id: str, epoch_id: str | None = None,
    capability: str = "adjudicate_finding",
) -> str | None:
    """Require an authenticated controller binding for adjudication."""
    return _require_capability(
        state, run_id, capability, epoch_id=epoch_id,
        controller_only=True,
    )


def _consume_controller_integration_action(
    state: RouteState, run_id: str, epoch_id: str, candidate_id: str,
) -> str | None:
    """Require a controller-planned integration action before resolution."""
    action_id = f"integration:{candidate_id}"
    if state.consume_controller_action(run_id, epoch_id, action_id) is None:
        return (
            "integration action is not claimed; the main controller must call "
            "get_runnable_actions and claim_runnable_action first"
        )
    return None


def _resolve_red_candidate_finding(state: RouteState, candidate_id: str, reason: str) -> None:
    """Close out the finding escalate_red_candidate opened for this candidate.

    Called from resolve_shadow_candidate once the controller has actually
    handled a red candidate (discard or retry) -- a no-op if no finding was
    ever created for it (resolve_finding on an unknown finding_id just
    returns None, it doesn't raise).
    """
    state.resolve_finding(
        f"shadow-conflict-{candidate_id}", "irrelevant",
        resolution_evidence_json=json.dumps(
            {"controller_resolution": reason}, sort_keys=True,
        ),
    )


def set_current_run_id(run_id: str | None) -> None:
    """Set the run ID for the current request context.

    If *run_id* is ``None``, the context variable is reset to its default.
    """
    if run_id is not None:
        set_current_principal(McpPrincipal(
            principal_kind="controller",
            run_id=run_id,
            allowed_capabilities=frozenset({
                "read_routes", "claim_native_action", "report_worker_result",
                "adjudicate_finding", "integrate_changeset", "complete_workflow",
                "adjudicate_coprocessor_result", "adjudicate_feedback",
                "adjudicate_native_result",
                "get_coprocessor_metrics",
                "invoke_coprocessor", "invoke_sidecar",
                "cancel_execution", "retry_execution",
            }),
            authenticated=False,
        ))
    else:
        set_current_principal(None)
