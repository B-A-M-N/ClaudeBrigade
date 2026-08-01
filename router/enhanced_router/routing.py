"""Request resolving against RouteState + ModelRegistry.

A role alias (anthropic-brigade-*) is resolved through:
  1. Active epoch lookup
  2. Agent binding check (immutable once bound — reconstructed from pinned fields)
  3. ModelSpec lookup from the registry to determine backend kind and upstream model
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import HTTPException

from enhanced_router.backends import (
    BackendType,
    ResolvedRoute,
    RequestIdentity,
    ROLE_MODEL_ALIASES,
    UnsupportedBackendError,
)
from enhanced_router.registry import ModelRegistry
from enhanced_router.state import RouteState, ControllerModelError
from enhanced_router.endpoint_selection import select_endpoint

logger = logging.getLogger("claude-enhanced-router")


def resolved_route_from_binding(
    binding: dict,
    litellm_base_url: str | None = None,
) -> ResolvedRoute:
    """Convert a binding row from *state.get_agent_binding()* to ResolvedRoute.

    Includes EVERY pinned field:

    - backend -> kind
    - model_id
    - upstream_model
    - api_base
    - api_key_env
    - binding_id
    - route_version
    - registry_hash
    - catalog_generation
    - litellm_model_name
    - litellm_base_url (optional, resolved from deployment lookup)
    - role (from binding role mapped back to alias)
    """
    backend_map: dict[str, BackendType] = {
        "direct-anthropic": BackendType.DIRECT_ANTHROPIC,
        "litellm": BackendType.LITELLM,
        "anthropic-passthrough": BackendType.ANTHROPIC_PASSTHROUGH,
    }
    kind = backend_map.get(binding.get("backend", ""))
    if kind is None:
        raise UnsupportedBackendError(binding.get("backend", ""))

    role_raw = binding.get("role")

    cat_gen = binding.get("catalog_generation")

    allowed_raw = binding.get("allowed_deployments_json")
    try:
        allowed_deployments = tuple(json.loads(allowed_raw)) if allowed_raw else ()
    except (TypeError, ValueError):
        allowed_deployments = ()
    provider_ids_raw = binding.get("provider_ids_json")
    try:
        provider_ids = tuple(json.loads(provider_ids_raw)) if provider_ids_raw else ()
    except (TypeError, ValueError):
        provider_ids = ()
    return ResolvedRoute(
        kind=kind,
        role=role_raw,
        model_id=binding.get("model_id"),
        upstream_model=binding.get("upstream_model"),
        api_base=binding.get("api_base"),
        api_key_env=binding.get("api_key_env"),
        litellm_model_name=binding.get("litellm_model_name"),
        agent_binding_id=binding.get("binding_id"),
        route_version=binding.get("route_version"),
        registry_hash=binding.get("registry_hash"),
        catalog_generation=cat_gen,
        litellm_base_url=litellm_base_url,
        auth_spec_json=binding.get("auth_spec_json"),
        provider_id=binding.get("provider_id"),
        endpoint_id=binding.get("endpoint_id"),
        endpoint_selection_reason=binding.get("endpoint_selection_reason"),
        routing_mode=binding.get("routing_mode") or "fixed",
        deployment_group=binding.get("deployment_group"),
        allowed_deployments=allowed_deployments,
        provider_ids=provider_ids,
        deployment_policy_digest=binding.get("deployment_policy_digest"),
    )


def resolved_route_from_controller_binding(binding: dict) -> ResolvedRoute:
    """Reconstruct a controller route solely from its immutable binding."""
    backend_map = {
        "direct-anthropic": BackendType.DIRECT_ANTHROPIC,
        "litellm": BackendType.LITELLM,
        "anthropic-passthrough": BackendType.ANTHROPIC_PASSTHROUGH,
    }
    kind = backend_map.get(binding.get("backend", ""))
    if kind is None:
        raise UnsupportedBackendError(binding.get("backend", ""))
    allowed_raw = binding.get("allowed_deployments_json")
    try:
        allowed_deployments = tuple(json.loads(allowed_raw)) if allowed_raw else ()
    except (TypeError, ValueError):
        allowed_deployments = ()
    provider_ids_raw = binding.get("provider_ids_json")
    try:
        provider_ids = tuple(json.loads(provider_ids_raw)) if provider_ids_raw else ()
    except (TypeError, ValueError):
        provider_ids = ()
    return ResolvedRoute(
        kind=kind,
        model_id=binding.get("registry_model_id"),
        upstream_model=binding.get("upstream_model"),
        api_base=binding.get("api_base"),
        api_key_env=binding.get("api_key_env"),
        litellm_model_name=binding.get("litellm_model_name"),
        registry_hash=binding.get("registry_hash"),
        catalog_generation=binding.get("catalog_generation"),
        auth_spec_json=binding.get("auth_spec_json"),
        provider_id=binding.get("provider_id"),
        endpoint_id=binding.get("endpoint_id"),
        endpoint_selection_reason=binding.get("endpoint_selection_reason"),
        routing_mode=binding.get("routing_mode") or "fixed",
        deployment_group=binding.get("deployment_group"),
        allowed_deployments=allowed_deployments,
        provider_ids=provider_ids,
        deployment_policy_digest=binding.get("deployment_policy_digest"),
        controller_binding_id=binding.get("binding_id"),
    )


def _resolve_api_base(api_base: str | None, api_base_env: str | None) -> str | None:
    """Resolve api_base, optionally reading from api_base_env.

    P0-1: For direct-anthropic models, both api_base and api_base_env must be
    non-empty. Returning None will cause resolve_request to fail with 503.
    """
    if api_base is not None:
        return api_base
    if api_base_env is not None:
        return os.environ.get(api_base_env)
    return None


def resolve_litellm_base_url(
    state: RouteState,
    catalog_generation: int | None,
) -> str | None:
    """Resolve LiteLLM base URL from a catalog generation, or None."""
    if catalog_generation is None:
        return None
    dep = state.get_litellm_deployment_for_generation(catalog_generation)
    if dep is None:
        return None
    return f"http://127.0.0.1:{dep['port']}"


# ---------------------------------------------------------------------------
# Identity-first request resolution
# ---------------------------------------------------------------------------


def resolve_request(
    *,
    identity: RequestIdentity,
    public_model: str,
) -> ResolvedRoute:
    """Resolve a Claude Code request to a backend route using identity-first dispatch.

    Resolution order (P0-1):

    1. Extract identity from the request.
    2. When ``claude_agent_id`` is present, look up an existing active binding FIRST.
    3. If found, use it regardless of the incoming model string.
       Record any model mismatch as evidence (logged).
    4. If no binding exists:
       a. Require a valid Brigade role alias in ``public_model``.
       b. Resolve the role route from the active epoch.
       c. Validate model health, role compatibility, and catalog generation.
       d. Atomically create and return the complete authoritative binding.
    5. If no agent ID exists, treat as controller/utility traffic:
       a. Enforce the configured controller model policy.
       b. Return ANTHROPIC_PASSTHROUGH for standard Claude model IDs.
    """
    # Lazy imports to avoid circular import at module load time
    try:
        from enhanced_router.state import get_state
        from enhanced_router.registry import get_registry
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="enhanced_router package is not available. Ensure it is installed.",
        )

    registry: ModelRegistry = get_registry()
    state: RouteState = get_state()
    role = registry.role_model_aliases().get(public_model, ROLE_MODEL_ALIASES.get(public_model))

    # Reject malformed native-agent requests before opening the SQLite state
    # database.  This keeps the identity contract deterministic even when the
    # router has not yet been installed with a writable state directory.
    if role is not None and (not identity.claude_agent_id or not identity.run_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Role alias '{public_model}' requires x-brigade-run-id and "
                "x-claude-code-agent-id headers."
            ),
        )

    # ------------------------------------------------------------------
    # 1. Binding-first lookup (P0-1): check for existing binding BEFORE
    #    examining the model string.
    # ------------------------------------------------------------------
    existing: dict | None = None
    if identity.claude_agent_id and identity.run_id:
        existing = state.get_agent_binding(identity.run_id, identity.claude_agent_id)

    if existing is not None:
        # Existing binding found — use it regardless of incoming model
        if role is not None and existing.get("role") != role:
            # Model mismatch: log it but still use the existing binding
            logger.warning(
                "Binding model mismatch: agent=%s bound_role=%s request_role=%s "
                "(using existing binding, ignoring mismatch)",
                identity.claude_agent_id, existing.get("role"), role,
            )

        cat_gen = existing.get("catalog_generation")
        port = existing.get("litellm_port")
        if cat_gen is not None and port is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Pinned LiteLLM generation {cat_gen} is no longer available. "
                    f"The agent was bound to a deployment that has been fully drained."
                ),
            )
        litellm_base_url = f"http://127.0.0.1:{port}" if port is not None else None

        resolved = resolved_route_from_binding(existing, litellm_base_url=litellm_base_url)
        return resolved

    # ------------------------------------------------------------------
    # 2. Subagent with existing binding, no identity — fail closed (P0-3)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 3. An claude_agent_id is present but no binding exists — create a new
    #    role binding from the requested alias.
    # ------------------------------------------------------------------
    if identity.claude_agent_id:
        if role is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Agent ID '{identity.claude_agent_id}' present but model "
                    f"'{public_model}' is not a Brigade role alias. "
                    "Subagent requests must use a role alias."
                ),
            )

        assert identity.run_id is not None  # guaranteed by check above

        # Active epoch
        active_epoch = state.get_active_epoch(identity.run_id)
        if active_epoch is None:
            raise HTTPException(
                status_code=409,
                detail=f"No active epoch for run {identity.run_id}.",
            )
        epoch_id = active_epoch["epoch_id"]

        # Resolve role route from epoch
        route = state.get_role_route(identity.run_id, epoch_id, role)
        if route is None:
            raise HTTPException(
                status_code=404,
                detail=f"No route defined for role '{role}' in epoch {epoch_id}",
            )

        model_id_value = registry.role_model_bindings().get(public_model, route["model_id"])
        if not isinstance(model_id_value, str) or not model_id_value:
            raise HTTPException(status_code=500, detail=f"Role '{role}' has no model binding")
        model_id = model_id_value
        assignment = state.get_spawn_assignment(
            identity.run_id, epoch_id, str(identity.claude_agent_id),
        )
        if assignment is None:
            pending = state.get_unattached_spawn_claim_for_role(
                identity.run_id, epoch_id, role,
            )
            if pending is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "native agent lifecycle is not attached; wait for "
                        "SubagentStart before the first model request"
                    ),
                )
        if assignment is not None:
            if assignment.get("role") != role or assignment.get("model_id") != model_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "native agent assignment mismatch: the first request must use "
                        f"role={assignment.get('role')!r}, model={assignment.get('model_id')!r}"
                    ),
                )
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
        if role in ("implementer", "repairer") and spec.capabilities.write_tool_certified is not True:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' is not write-tool certified for role '{role}'.",
            )

        reg_hash = registry.registry_hash()
        endpoint_override = route.get("endpoint_override") or route.get("endpoint_id")
        managed_group = spec.routing_mode == "managed-group" and not endpoint_override
        managed_deployments: tuple[str, ...] = ()
        managed_provider_ids: set[str | None] = set()
        if managed_group:
            eligible = []
            for candidate_id in sorted(spec.endpoints):
                try:
                    eligible.append(select_endpoint(
                        model_id,
                        spec,
                        state,
                        explicit_endpoint=candidate_id,
                        require_certified=True,
                        provider_id=spec.endpoints[candidate_id].provider_id or spec.provider_id,
                        configuration_hash=reg_hash,
                        required_capabilities=("messages", "streaming", "tools"),
                    ))
                except ValueError:
                    continue
            if not eligible:
                raise HTTPException(status_code=503, detail=f"Model '{model_id}' has no eligible managed deployment")
            selected = eligible[0]
            managed_deployments = tuple(item.endpoint_id for item in eligible)
            managed_provider_ids = {
                item.spec.provider_id or spec.provider_id for item in eligible
            }
        else:
            try:
                selected = select_endpoint(
                    model_id,
                    spec,
                    state,
                    explicit_endpoint=endpoint_override,
                    # Production bindings may only use a certified endpoint.
                    # The legacy single-endpoint adapter is explicitly
                    # certified for local/operator-owned models; provider
                    # deployments must have capability-specific evidence.
                    require_certified=True,
                    provider_id=spec.provider_id,
                    configuration_hash=reg_hash,
                    required_capabilities=("messages", "streaming", "tools"),
                )
            except ValueError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        endpoint = selected.spec

        # Resolve api_base from the selected, immutable endpoint.
        resolved_api_base = _resolve_api_base(endpoint.api_base, endpoint.api_base_env)

        # Validate credential availability
        if endpoint.backend == "direct-anthropic":
            # P0-1: Require an explicit endpoint at resolve time too (defense in depth).
            if not resolved_api_base:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Model '{model_id}' (direct-anthropic) has no resolved endpoint. "
                        f"Set api_base or api_base_env='{spec.api_base_env or '...'}'."
                    ),
                )
            if endpoint.api_key_env and not os.environ.get(endpoint.api_key_env):
                raise HTTPException(
                    status_code=503,
                    detail=f"API key env var '{endpoint.api_key_env}' for model '{model_id}' is not set.",
                )

        kind = _backend_kind(endpoint.backend)
        upstream_model = endpoint.upstream_model or endpoint.litellm_model
        route_version = route["version"]
        # Resolve LiteLLM deployment
        catalog_generation: int | None = None
        litellm_base_url: str | None = None
        litellm_model_name: str | None = None
        if kind == BackendType.LITELLM:
            litellm_model_name = (
                f"brigade-{spec.deployment_group or model_id}"
                if managed_group
                else f"brigade-{model_id}--{selected.endpoint_id}" if spec.endpoints else f"brigade-{model_id}"
            )
            dep = state.get_active_litellm_deployment()
            if dep is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No active LiteLLM deployment available for model '{model_id}'. "
                        "Wait for the LiteLLM catalog to finish loading."
                    ),
                )
            catalog_generation = dep["generation"]
            litellm_base_url = f"http://127.0.0.1:{dep['port']}"

        # Serialize auth spec for immutable pinning (P0-2)
        auth_spec_json: str | None = None
        if endpoint.auth is not None:
            auth_spec_json = json.dumps(endpoint.auth.model_dump())

        # Atomically bind or return existing (race-safe)
        binding_dict, _ = state.bind_or_get_agent(
            run_id=identity.run_id,
            claude_agent_id=str(identity.claude_agent_id),
            epoch_id=epoch_id,
            role=role,
            model_id=model_id,
            route_version=route_version,
            backend=endpoint.backend,
            registry_hash=reg_hash,
            catalog_generation=catalog_generation,
            litellm_model_name=litellm_model_name,
            upstream_model=upstream_model,
            api_base=resolved_api_base,
            api_key_env=endpoint.api_key_env,
            auth_spec_json=auth_spec_json,
            endpoint_id=None if managed_group else selected.endpoint_id,
            endpoint_selection_reason=(
                f"managed deployment group '{spec.deployment_group or model_id}'"
                if managed_group else selected.reason
            ),
            endpoint_policy_json=json.dumps(spec.endpoint_policy.model_dump()),
            certification_id=None if managed_group else endpoint.certification_id,
            provider_id=(
                endpoint.provider_id or spec.provider_id
                if not managed_group
                else next(iter(managed_provider_ids)) if len(managed_provider_ids) == 1 else None
            ),
            provider_ids_json=json.dumps(
                sorted(provider_id for provider_id in managed_provider_ids if provider_id)
                if managed_group
                else ([endpoint.provider_id or spec.provider_id] if (endpoint.provider_id or spec.provider_id) else [])
            ),
            configuration_hash=reg_hash,
            routing_mode="managed-group" if managed_group else "fixed",
            deployment_group=spec.deployment_group or model_id if managed_group else None,
            allowed_deployments_json=json.dumps(managed_deployments) if managed_group else None,
            deployment_policy_digest=reg_hash if managed_group else None,
            claude_session_id=identity.claude_session_id,
            claude_parent_agent_id=identity.claude_parent_agent_id,
        )

        resolved = resolved_route_from_binding(binding_dict, litellm_base_url=litellm_base_url)
        return resolved

    # ------------------------------------------------------------------
    # 4. No agent ID — controller/utility traffic
    # ------------------------------------------------------------------
    if role is not None:
        # A role alias without agent identity is always rejected (P0-3)
        raise HTTPException(
            status_code=409,
            detail=(
                f"Role alias '{public_model}' requires x-claude-code-agent-id. "
                "The controller may not use subagent role aliases directly."
            ),
        )

    # 4. Main-thread controller traffic. A controller is a runtime role, not
    # a vendor identity: resolve the selected public model through the same
    # registry and pin it for the client session.
    if identity.run_id and identity.claude_session_id:
        existing_controller = state.get_controller_binding(
            identity.run_id, identity.claude_session_id
        )
        if existing_controller is not None:
            cat_gen = existing_controller.get("catalog_generation")
            port = existing_controller.get("litellm_port")
            if cat_gen is not None and port is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Pinned controller LiteLLM generation {cat_gen} is unavailable",
                )
            litellm_base_url = f"http://127.0.0.1:{port}" if port is not None else None
            route = resolved_route_from_controller_binding(existing_controller)
            if route.kind == BackendType.LITELLM:
                route = ResolvedRoute(**{**route.__dict__, "litellm_base_url": litellm_base_url})
            return route

    reg_hash = registry.registry_hash()
    try:
        controller_spec = registry.get_model(public_model)
    except KeyError:
        # Preserve standalone compatibility for callers without a run while
        # requiring configured models for a bound controller session.
        if not identity.run_id:
            return ResolvedRoute(kind=BackendType.ANTHROPIC_PASSTHROUGH, model_id=public_model)
        raise HTTPException(status_code=404, detail=f"Controller model '{public_model}' is not in the registry")

    if not controller_spec.enabled:
        raise HTTPException(status_code=503, detail=f"Controller model '{public_model}' is disabled")
    if identity.run_id:
        try:
            state.validate_controller_model(identity.run_id, public_model)
        except ControllerModelError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    managed_group = controller_spec.routing_mode == "managed-group"
    managed_deployments: tuple[str, ...] = ()
    managed_provider_ids: set[str | None] = set()
    if managed_group:
        eligible = []
        for candidate_id in sorted(controller_spec.endpoints):
            try:
                eligible.append(select_endpoint(
                    public_model,
                    controller_spec,
                    state,
                    explicit_endpoint=candidate_id,
                    require_certified=True,
                    provider_id=controller_spec.endpoints[candidate_id].provider_id or controller_spec.provider_id,
                    configuration_hash=reg_hash,
                    required_capabilities=("messages", "streaming", "tools"),
                ))
            except ValueError:
                continue
        if not eligible:
            raise HTTPException(status_code=503, detail=f"Controller model '{public_model}' has no eligible managed deployment")
        selected = eligible[0]
        managed_deployments = tuple(item.endpoint_id for item in eligible)
        managed_provider_ids = {
            item.spec.provider_id or controller_spec.provider_id for item in eligible
        }
    else:
        try:
            selected = select_endpoint(
                public_model,
                controller_spec,
                state,
                    # Standard Claude passthrough is already an explicit
                    # backend boundary, not a provider deployment selected by
                    # Brigade.  All other controller deployments require the
                    # same endpoint certification as worker routes.
                    require_certified=controller_spec.backend != "anthropic-passthrough",
                provider_id=controller_spec.provider_id,
                configuration_hash=reg_hash,
                required_capabilities=("messages", "streaming", "tools"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    endpoint = selected.spec
    resolved_api_base = _resolve_api_base(endpoint.api_base, endpoint.api_base_env)
    if endpoint.backend == "direct-anthropic" and not resolved_api_base:
        raise HTTPException(status_code=503, detail=f"No endpoint configured for controller '{public_model}'")
    if endpoint.api_key_env and not os.environ.get(endpoint.api_key_env):
        raise HTTPException(status_code=503, detail=f"API key env var '{endpoint.api_key_env}' is not set")

    kind = _backend_kind(endpoint.backend)
    catalog_generation: int | None = None
    litellm_base_url: str | None = None
    litellm_model_name: str | None = None
    if kind == BackendType.LITELLM:
        dep = state.get_active_litellm_deployment()
        if dep is None:
            raise HTTPException(status_code=503, detail="No active LiteLLM deployment for controller")
        catalog_generation = dep["generation"]
        litellm_base_url = f"http://127.0.0.1:{dep['port']}"
        litellm_model_name = (
            f"brigade-{controller_spec.deployment_group or public_model}"
            if managed_group
            else f"brigade-{public_model}--{selected.endpoint_id}" if controller_spec.endpoints else f"brigade-{public_model}"
        )

    auth_spec_json = json.dumps(endpoint.auth.model_dump()) if endpoint.auth else None
    if identity.run_id and identity.claude_session_id:
        provider_id = (
            endpoint.provider_id or controller_spec.provider_id
            if not managed_group
            else next(iter(managed_provider_ids)) if len(managed_provider_ids) == 1 else None
        )
        reservation_id = f"controller:{identity.run_id}:{identity.claude_session_id}"
        reservation_created = False
        if provider_id:
            provider = registry.providers.get(provider_id)
            if provider is not None:
                existing_reservation = next(
                    (
                        item for item in state.get_provider_reservations(provider_id)
                        if item["reservation_id"] == reservation_id
                        and item["state"] in {"queued", "reserved"}
                    ),
                    None,
                )
                if existing_reservation is None:
                    reservation = state.reserve_provider_agent(
                        reservation_id=reservation_id,
                        run_id=identity.run_id,
                        epoch_id=(state.get_active_epoch(identity.run_id) or {}).get("epoch_id"),
                        provider_id=provider_id,
                        execution_id=f"controller:{identity.claude_session_id}",
                        lane="controller",
                        max_active=provider.limits.max_active_agents,
                        reason="main controller binding",
                    )
                    reservation_created = True
                else:
                    reservation = existing_reservation
                if reservation["state"] != "reserved":
                    if reservation_created:
                        state.release_provider_reservation(reservation_id, state="cancelled")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Provider '{provider_id}' has no active controller capacity",
                    )
        try:
            binding, is_new_controller = state.bind_or_get_controller(
                run_id=identity.run_id,
                client_session_id=identity.claude_session_id,
                public_model=public_model,
                registry_model_id=public_model,
                backend=endpoint.backend,
                upstream_model=endpoint.upstream_model or endpoint.litellm_model,
                provider_id=(
                    provider_id if not managed_group
                    else next(iter(managed_provider_ids)) if len(managed_provider_ids) == 1 else None
                ),
                provider_ids_json=json.dumps(
                    sorted(provider_id for provider_id in managed_provider_ids if provider_id)
                    if managed_group
                    else ([provider_id] if provider_id else [])
                ),
                api_base=resolved_api_base,
                catalog_generation=catalog_generation,
                registry_hash=reg_hash,
                certification_id=endpoint.certification_id,
                auth_spec_json=auth_spec_json,
                api_key_env=endpoint.api_key_env,
                endpoint_id=None if managed_group else selected.endpoint_id,
                endpoint_selection_reason=(
                    f"managed deployment group '{controller_spec.deployment_group or public_model}'"
                    if managed_group else selected.reason
                ),
                endpoint_policy_json=json.dumps(controller_spec.endpoint_policy.model_dump()),
                litellm_model_name=litellm_model_name,
                configuration_hash=reg_hash,
                routing_mode="managed-group" if managed_group else "fixed",
                deployment_group=controller_spec.deployment_group or public_model if managed_group else None,
                allowed_deployments_json=json.dumps(managed_deployments) if managed_group else None,
                deployment_policy_digest=reg_hash if managed_group else None,
            )
        except Exception:
            if reservation_created and provider_id:
                state.release_provider_reservation(reservation_id, state="cancelled")
            raise
        if not is_new_controller and reservation_created and provider_id:
            state.release_provider_reservation(reservation_id, state="cancelled")
        route = resolved_route_from_controller_binding(binding)
        if kind == BackendType.LITELLM:
            route = ResolvedRoute(**{**route.__dict__, "litellm_base_url": litellm_base_url, "litellm_model_name": litellm_model_name})
        return route

    return ResolvedRoute(
        kind=kind,
        model_id=public_model,
        upstream_model=endpoint.upstream_model or endpoint.litellm_model,
        api_base=resolved_api_base,
        api_key_env=endpoint.api_key_env,
        registry_hash=reg_hash,
        catalog_generation=catalog_generation,
        litellm_model_name=litellm_model_name,
        litellm_base_url=litellm_base_url,
        auth_spec_json=auth_spec_json,
        provider_id=endpoint.provider_id or controller_spec.provider_id,
        endpoint_id=selected.endpoint_id,
        endpoint_selection_reason=selected.reason,
    )


def _backend_kind(backend: str) -> BackendType:
    """Map the registry backend string to a BackendType enum.

    Raises ``UnsupportedBackendError`` for unknown backend values
    instead of silently defaulting to ANTHROPIC_PASSTHROUGH.
    """
    result = {
        "direct-anthropic": BackendType.DIRECT_ANTHROPIC,
        "litellm": BackendType.LITELLM,
        "anthropic-passthrough": BackendType.ANTHROPIC_PASSTHROUGH,
    }.get(backend)
    if result is None:
        raise UnsupportedBackendError(backend)
    return result
