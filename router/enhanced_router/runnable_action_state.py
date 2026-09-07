"""Runnable-action claim/scheduling core, split out of state.py.

Twenty-second increment of the incremental extraction out of ``RouteState``
-- the last and largest slice of "Run lifecycle". This is the controller's
scheduling planner (``get_runnable_actions``) and atomic claimer
(``claim_runnable_action``), plus controller-action bookkeeping, orphan
reconciliation on startup/crash-recovery, and spawn-intent tracking. Per an
investigation pass before this extraction, this is deliberately kept as one
module rather than split further: ``get_runnable_actions`` and
``claim_runnable_action`` share the exact same claim-shape contract and
``action_id`` scheme, and ``reconcile_lifecycle`` sweeps native and sidecar
orphans plus reservations/intents together in one pass.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import json
import hashlib
import logging
import secrets
import sqlite3
import fnmatch
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from enhanced_router.route_ladder import candidate_dict, route_key, route_digest
from enhanced_router.state_errors import WorkflowStateError

LOGGER = logging.getLogger("claude-enhanced-router.workflow")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow_age(max_age_seconds: int) -> str:
    """Return an ISO-8601 timestamp that is *max_age_seconds* in the past."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    return cutoff.isoformat()


def _role_route_fallback_candidates(route: dict) -> list[dict]:
    """Normalize a role_routes row's fallback ladder to a list of
    {"model", "endpoint"} dicts.

    Prefers the newer fallback_routes_json (per-candidate endpoint) and
    falls back to the legacy fallback_models_json (bare model IDs, endpoint
    always "auto") for rows written before that column existed.
    """
    fallback_routes = route.get("fallback_routes")
    if not isinstance(fallback_routes, list):
        try:
            fallback_routes = json.loads(route.get("fallback_routes_json") or "[]")
        except (TypeError, ValueError):
            fallback_routes = []
    fallback_routes = [candidate_dict(c) for c in fallback_routes if isinstance(c, dict) and c.get("model")]
    if fallback_routes:
        return fallback_routes
    try:
        legacy_models = json.loads(route.get("fallback_models_json") or "[]")
    except (TypeError, ValueError):
        legacy_models = []
    return [
        {"model": m, "endpoint": "auto"}
        for m in legacy_models if isinstance(m, str) and m
    ]


def _route_fallback_candidates(route: dict) -> list[dict]:
    """Read a native sidecar/coprocessor ladder or a role-route ladder."""
    configured = route.get("fallback_routes")
    if isinstance(configured, list) and configured:
        return [candidate_dict(item) for item in configured if isinstance(item, dict)]
    return _role_route_fallback_candidates(route)


def _path_scopes_overlap(left: list[str], right: list[str]) -> bool:
    """Conservatively detect package path overlap before mutators launch."""
    for raw_left in left or []:
        left_path = str(raw_left or "").strip().replace("\\", "/").rstrip("/")
        if not left_path:
            continue
        for raw_right in right or []:
            right_path = str(raw_right or "").strip().replace("\\", "/").rstrip("/")
            if not right_path:
                continue
            if fnmatch.fnmatch(left_path, right_path) or fnmatch.fnmatch(right_path, left_path):
                return True
            if not any(char in left_path + right_path for char in "*?["):
                if left_path == right_path or left_path.startswith(right_path + "/") or right_path.startswith(left_path + "/"):
                    return True
    return False


_ROUTE_ADVANCE_ERROR_CLASSES = {
    "provider_unavailable",
    "rate_limited",
    "authentication_failed",
    "endpoint_unhealthy",
    "request_timeout",
    "transport_error",
}

_NON_PROVIDER_FAILURE_ERROR_CLASSES = {
    "cancelled_by_controller",
    "user_cancelled",
    "policy_rejected",
    "orphaned_after_restart",
}


def _execution_advances_route(execution: dict) -> bool:
    """Return whether a terminal execution should consume a ladder slot."""
    status = str(execution.get("status") or "").lower()
    if status in {"cancelled", "orphaned"}:
        return False
    if status not in {"failed", "timeout", "timed_out"}:
        return False
    error_class = str(execution.get("error_class") or "").lower()
    # Legacy rows without an error class retain provider-failure behavior;
    # classified policy/task failures do not advance a provider ladder.
    return (
        not error_class
        or error_class in _ROUTE_ADVANCE_ERROR_CLASSES
    ) and error_class not in _NON_PROVIDER_FAILURE_ERROR_CLASSES


def _action_slot(
    run_id: str,
    epoch_id: str,
    phase_id: str,
    role: str,
    native_agent_name: str,
    executions: list[dict],
    claims: dict[str, dict],
    max_attempts: int,
    *,
    include_claimed: bool,
) -> tuple[int, str, str] | None:
    """Return the first persisted phase/package slot still available.

    Action identity must not depend on the number of rows visible in a
    concurrent read. Completed executions and claimed actions carry their
    package slot explicitly; legacy executions without a package are assigned
    deterministic low slots only as a migration fallback.
    """
    prefix = f"{phase_id}:{role}:slot-"

    def slot_from_package(value: object) -> int | None:
        text = str(value or "")
        marker = ":slot-"
        if marker not in text:
            return None
        suffix = text.rsplit(marker, 1)[1].split(":", 1)[0]
        return int(suffix) if suffix.isdigit() else None

    occupied: set[int] = set()
    legacy_count = 0
    for execution in executions:
        slot = slot_from_package(execution.get("package_id"))
        if slot is None:
            slot = legacy_count
            legacy_count += 1
        occupied.add(slot)

    claimed_slots: set[int] = set()
    for claim in claims.values():
        if (
            claim.get("phase_id") == phase_id
            and claim.get("role") == role
            and claim.get("status") in {"claimed", "consumed"}
        ):
            slot = slot_from_package(claim.get("action_id"))
            if slot is not None:
                claimed_slots.add(slot)
                occupied.add(slot)

    for slot in range(max(1, int(max_attempts))):
        package_id = f"{prefix}{slot}"
        action_id = (
            f"action:{run_id}:{epoch_id}:{package_id}:"
            f"{native_agent_name}:attempt-1"
        )
        if slot in occupied:
            claim = claims.get(action_id)
            # A live, unattached claim is still the runnable action when the
            # caller explicitly asks for claimed work. A consumed claim whose
            # execution is terminal, however, occupies its package slot and
            # must advance to the next stable retry slot.
            if claim is None or claim.get("status") != "claimed" or not include_claimed:
                continue
        if action_id in claims and not include_claimed:
            continue
        return slot, package_id, action_id
    return None


class RunnableActionRepository(RepositoryMixin):
    """Mixin providing the runnable-action claim/scheduling core.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does), plus a large set of already-extracted
    cross-section methods (workflow phases, agent executions, controller
    bindings, integration candidates, role routes, run registry, provider
    reservations) that all resolve through the mixin's normal method
    resolution order.
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def _active_action_claims(
        self, run_id: str, epoch_id: str,
    ) -> dict[str, dict]:
        """Return non-terminal action claims, expiring stale claims first.

        Expiring the claim row alone is not enough: a TTL-expired claim may
        still hold a 'reserved' provider_reservations row.  Without releasing
        it here, that capacity slot leaks for the life of the process --
        reconcile_lifecycle only runs at startup, so it would not otherwise
        be freed until the next restart.
        """
        conn = self._new_conn()
        try:
            now = _utcnow()
            expiring = conn.execute(
                "SELECT action_id, reservation_id, provider_id FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND status='claimed' AND expires_at < ?",
                (run_id, epoch_id, now),
            ).fetchall()
            conn.execute(
                "UPDATE runnable_action_claims SET status='expired' "
                "WHERE run_id=? AND epoch_id=? AND status='claimed' AND expires_at < ?",
                (run_id, epoch_id, now),
            )
            conn.commit()
            rows = conn.execute(
                "SELECT * FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND status IN ('claimed','consumed')",
                (run_id, epoch_id),
            ).fetchall()
            result = {str(row["action_id"]): dict(row) for row in rows}
        finally:
            conn.close()
        released_provider_ids: set[str] = set()
        for row in expiring:
            reservation_id = row["reservation_id"]
            if not reservation_id:
                continue
            released = self.release_provider_reservation(str(reservation_id), "expired")
            if released and released.get("provider_id"):
                released_provider_ids.add(str(released["provider_id"]))
        if released_provider_ids:
            from enhanced_router.registry import get_registry

            registry = get_registry()
            for provider_id in released_provider_ids:
                provider = registry.providers.get(provider_id)
                if provider:
                    self.admit_provider_agents(provider_id, provider.limits.max_active_agents)
        return result

    def get_runnable_actions(
        self, run_id: str, epoch_id: str, *, include_claimed: bool = False,
    ) -> list[dict]:
        """Return controller-visible native actions that may be spawned now.

        This is planning, not execution.  It intentionally never creates a
        queued reservation: a denied Claude Code Agent call cannot be replayed
        by SQLite.  The controller asks again after a terminal lifecycle event.
        """
        from enhanced_router.registry import get_registry

        self.advance_conditional_phases(run_id, epoch_id)
        registry = get_registry()
        actions: list[dict] = []
        epoch = self.get_active_epoch(run_id)
        mutation_paused = bool(
            epoch
            and str(epoch.get("epoch_id")) == str(epoch_id)
            and (
                epoch.get("mutation_paused")
                or str(epoch.get("escalation_state") or "") == "escalated"
            )
        )
        claims = self._active_action_claims(run_id, epoch_id)
        def allowed_bounded_calls(profile_id: str | None) -> dict[str, Any]:
            resolver = getattr(registry, "resolve_coprocessors", None)
            if callable(resolver):
                resolved = resolver(profile_id)
                return cast(dict[str, Any], resolved) if isinstance(resolved, dict) else {}
            # Compatibility with lightweight registry doubles and older
            # installed registries that still expose the legacy name.
            legacy_resolver = getattr(registry, "resolve_sidecars", None)
            if callable(legacy_resolver):
                resolved = legacy_resolver(profile_id)
                return cast(dict[str, Any], resolved) if isinstance(resolved, dict) else {}
            return {}
        active_workflow = self.get_active_epoch(run_id)
        workflow_tier = str(
            (active_workflow or {}).get("workflow_id")
            or (active_workflow or {}).get("minimum_tier")
            or "normal"
        )
        task_contract = self.get_task_contract(run_id, epoch_id)
        contract_approved = bool(
            task_contract is not None and task_contract.get("status") == "approved"
        )
        # Non-trivial mutation cannot start against the intake-only contract.
        # Surface one durable controller action so the controller can publish
        # requirements, resolve ambiguities, approve the contract, and then
        # re-query the same scheduler.  Trivial controller-direct work keeps
        # its intentionally lightweight path.
        requires_contract = workflow_tier != "trivial" and not contract_approved
        if requires_contract and any(
            bool(phase.get("mutating"))
            and str(phase.get("status") or "") not in {"skipped", "completed", "failed"}
            for phase in self.get_workflow_phases(run_id, epoch_id)
        ):
            contract_action_id = f"contract:{run_id}:{epoch_id}"
            contract_claim = claims.get(contract_action_id)
            if contract_claim is None or include_claimed:
                controller_model = "controller"
                run = self.get_run(run_id)
                if run and run.get("claude_session_id"):
                    binding = self.get_controller_binding(run_id, str(run["claude_session_id"]))
                    if binding and binding.get("registry_model_id"):
                        controller_model = str(binding["registry_model_id"])
                contract_action: dict[str, Any] = {
                    "action_id": contract_action_id,
                    "action_kind": "controller_contract",
                    "controller_action_kind": "controller_contract",
                    "phase_id": "controller-contract",
                    "role": "controller",
                    "native_agent_name": "controller",
                    "model_id": controller_model,
                    "endpoint": "auto",
                    "provider_id": None,
                    "status": "ready",
                    "requires_main_controller": True,
                    "required_action": (
                        "publish a complete objective, mandatory requirements, acceptance criteria, "
                        "and resolved ambiguities; then approve the task contract"
                    ),
                    "contract_status": task_contract.get("status") if task_contract else None,
                    "contract_version": task_contract.get("version") if task_contract else None,
                }
                if contract_claim is not None:
                    contract_action.update({
                        "status": str(contract_claim["status"]),
                        "claim_token": contract_claim["claim_token"],
                        "reservation_id": contract_claim.get("reservation_id"),
                        "intent_id": contract_claim.get("intent_id"),
                    })
                actions.append(contract_action)
        for candidate in self.get_integration_candidates(run_id=run_id, epoch_id=epoch_id):
            disposition = str(candidate.get("disposition"))
            if disposition not in {"yellow", "red", "pending"}:
                continue
            action_id = f"integration:{candidate['candidate_id']}"
            claim = claims.get(action_id)
            if claim is not None and not include_claimed:
                continue
            controller_model = "controller"
            run = self.get_run(run_id)
            if run and run.get("claude_session_id"):
                binding = self.get_controller_binding(
                    run_id, str(run["claude_session_id"])
                )
                if binding and binding.get("registry_model_id"):
                    controller_model = str(binding["registry_model_id"])
            action = {
                "action_id": action_id,
                "action_kind": "controller_integration",
                "requires_main_controller": True,
                "candidate_id": candidate["candidate_id"],
                "changeset_id": candidate["changeset_id"],
                "phase_id": "controller-integration",
                "role": "controller",
                "native_agent_name": "controller",
                "model_id": controller_model,
                "disposition": disposition,
                "required_action": (
                    "inspect and resolve the conflict; approve only after deterministic preflight"
                ),
                "status": "escalated",
            }
            if claim is not None:
                action.update({
                    "status": str(claim["status"]),
                    "claim_token": claim["claim_token"],
                    "reservation_id": claim.get("reservation_id"),
                    "intent_id": claim.get("intent_id"),
                })
            actions.append(action)
        for phase in self.get_ready_phases(run_id, epoch_id) + self.get_active_phases(run_id, epoch_id):
            if mutation_paused and bool(phase.get("mutating")):
                continue
            if requires_contract and bool(phase.get("mutating")):
                continue
            phase_packages = self.get_work_packages(
                run_id, epoch_id, str(phase["phase_id"])
            )
            if (
                bool(phase.get("mutating"))
                and str(phase.get("fanout_from") or "") == "work_packages"
                and not phase_packages
            ):
                # A package-fanout mutation is never allowed to fall back to
                # a whole-task generic worker. The controller must publish
                # explicit, scoped packages first.
                continue
            actor_contract = str(phase.get("required_actor") or phase.get("actor") or "")
            controller_phase = actor_contract == "controller"
            if (
                requires_contract
                and controller_phase
                and str(phase.get("produces") or "") == "work_packages"
            ):
                continue
            if actor_contract and not controller_phase:
                continue
            # Controller phases are first-class actions.  A pending phase is
            # intentionally visible so the controller can claim the action;
            # claiming it starts the phase atomically before the controller
            # performs the work.
            if controller_phase and phase.get("status") not in {"pending", "active"}:
                continue
            allowed_roles = ["controller"] if controller_phase else json.loads(
                phase.get("allowed_roles_json") or "[]"
            )
            executions = self.get_agent_executions(
                run_id, epoch_id=epoch_id, phase_id=phase["phase_id"]
            )
            active_count = sum(
                item.get("status") in {"started", "running", "streaming", "verifying"}
                for item in executions
            )
            max_parallelism = int(
                phase.get("max_parallelism") or phase.get("max_fanout") or 1
            )
            max_attempts = int(
                phase.get("max_attempts") or phase.get("max_fanout") or 1
            )
            max_attempts_per_model = phase.get("max_attempts_per_model")
            fallback_policy = str(phase.get("fallback_policy") or "").strip().lower()
            if active_count >= max_parallelism or len(executions) >= max_attempts:
                continue
            execution_kind = str(phase.get("execution_kind") or "native_agent")
            phase_agent_id = str(phase.get("agent_id") or "").strip() or None
            sidecar_id = str(phase.get("sidecar_id") or "").strip() or None
            sidecar_agent_id = str(phase.get("sidecar_agent_id") or "").strip() or None
            coprocessor_id = str(phase.get("coprocessor_id") or "").strip() or None
            sidecar_spec = None
            sidecar_agent = None
            coprocessor_route = None
            run_row = self.get_run(run_id)
            sidecar_profile_id = run_row.get("sidecar_profile_id") if run_row else None
            if execution_kind == "sidecar_call" and sidecar_id:
                try:
                    sidecar_spec = registry.get_sidecar(sidecar_id)
                except KeyError:
                    continue
                if not sidecar_spec.enabled:
                    continue
                # A run's sidecar_profile_id bounds which (possibly
                # resource-costly) sidecars it may invoke -- a workflow
                # phase naming a sidecar outside that bound must never
                # become runnable, or the bound is advisory in name only.
                if sidecar_id not in allowed_bounded_calls(sidecar_profile_id):
                    continue
            if sidecar_agent_id:
                try:
                    sidecar_agent = registry.resolve_sidecar_agent(
                        sidecar_agent_id,
                        sidecar_profile_id,
                    )
                except KeyError:
                    continue
                if not sidecar_agent.enabled:
                    continue
                if execution_kind != "native_agent":
                    continue
                if phase.get("mutating") and not sidecar_agent.can_mutate:
                    continue
            if sidecar_agent is not None:
                max_parallelism = min(max_parallelism, int(sidecar_agent.max_parallelism))
            if sidecar_agent is not None and not allowed_roles:
                allowed_roles = [role for role in sidecar_agent.roles if role in {
                    "recon", "implementer", "adversary", "repairer"
                }]
            for role in allowed_roles:
                route: dict[str, Any]
                controller_binding = None
                if controller_phase:
                    run = self.get_run(run_id)
                    session_id = str(run.get("claude_session_id")) if run and run.get("claude_session_id") else ""
                    controller_binding = self.get_controller_binding(run_id, session_id) if session_id else None
                    # Controller-owned phases are executable scheduler actions,
                    # not model-backed worker actions.  A controller binding is
                    # useful metadata when one exists, but package planning and
                    # other controller phases must remain discoverable before
                    # the main request has created that binding.
                    if controller_binding and controller_binding.get("registry_model_id"):
                        route = {
                            "endpoint_override": controller_binding.get("endpoint_id"),
                            "model_id": controller_binding["registry_model_id"],
                        }
                    else:
                        route = {
                            "endpoint_override": "auto",
                            "model_id": "controller",
                        }
                elif sidecar_agent is not None:
                    route = {
                        "endpoint_override": (
                            None if sidecar_agent.endpoint == "auto" else sidecar_agent.endpoint
                        ),
                        "model_id": sidecar_agent.model_id,
                        "provider_id": sidecar_agent.provider_id,
                        "fallback_routes": [
                            candidate_dict(item)
                            for item in getattr(sidecar_agent, "fallback_routes", [])
                        ],
                        "version": 0,
                    }
                elif sidecar_spec is not None:
                    route = {
                        "endpoint_override": (
                            None if sidecar_spec.endpoint == "auto" else sidecar_spec.endpoint
                        ),
                        "model_id": sidecar_spec.model_id,
                        "provider_id": getattr(sidecar_spec, "provider_id", None),
                        "fallback_routes": [
                            candidate_dict(item)
                            for item in getattr(sidecar_spec, "fallback_routes", [])
                        ],
                        "version": 0,
                    }
                elif coprocessor_id:
                    try:
                        coprocessor_route = registry.get_coprocessor(coprocessor_id)
                    except KeyError:
                        continue
                    if (
                        not coprocessor_route.enabled
                        or coprocessor_id not in allowed_bounded_calls(sidecar_profile_id)
                    ):
                        continue
                    route = {
                        "endpoint_override": (
                            None if coprocessor_route.endpoint == "auto" else coprocessor_route.endpoint
                        ),
                        "model_id": coprocessor_route.model_id,
                        "provider_id": coprocessor_route.provider_id,
                        "fallback_routes": [
                            candidate_dict(item)
                            for item in getattr(coprocessor_route, "fallback_routes", [])
                        ],
                        "version": 0,
                    }
                else:
                    route = self.get_role_route(run_id, epoch_id, role)
                    if not route:
                        continue
                # Fallback ladders and per-model attempt caps are keyed by
                # role_routes (one route per role), so the failure/attempt
                # counts used to index into them must also be role-scoped.
                # `executions` is phase-wide -- every configured phase today
                # has exactly one role, so this is a no-op change for the
                # current config, but a multi-role phase would otherwise mix
                # another role's failures into this role's fallback index.
                role_executions = [item for item in executions if item.get("role") == role]
                if role_executions and any(
                    _execution_advances_route(item)
                    for item in role_executions
                ) and not fallback_policy:
                    route_fallbacks = _role_route_fallback_candidates(route)
                    if not (
                        execution_kind == "native_agent"
                        and route_fallbacks
                    ):
                        continue
                if (
                    execution_kind == "native_agent"
                    and not controller_phase
                ):
                    fallback_routes = _route_fallback_candidates(route)
                    fallback_models = [c["model"] for c in fallback_routes]
                    failed_attempts = sum(
                        _execution_advances_route(item)
                        for item in role_executions
                    )
                    if fallback_models and failed_attempts > len(fallback_models):
                        continue
                    fallback_index = int(failed_attempts) - 1
                    if failed_attempts > 0 and 0 <= fallback_index < len(fallback_routes):
                        candidate = fallback_routes[fallback_index]
                        fallback_model = candidate.get("model")
                        if isinstance(fallback_model, str) and fallback_model:
                            route = dict(route)
                            route["model_id"] = fallback_model
                            route["endpoint_override"] = candidate.get("endpoint") or None
                            route["provider_id"] = candidate.get("provider_id")
                            route["candidate_index"] = fallback_index + 1
                model_id = str(route["model_id"])
                coprocessor = None
                try:
                    model = registry.get_model(model_id)
                except KeyError:
                    # Controller actions can be planned before a controller
                    # request creates an immutable model binding.  They are
                    # scheduler work, so the synthetic ``controller`` route
                    # does not need a registry model or provider.
                    if controller_phase:
                        model = None
                    else:
                        continue
                provider_value = (
                    route.get("provider_id")
                    or
                    (controller_binding or {}).get("provider_id")
                    or (getattr(sidecar_agent, "provider_id", None) if sidecar_agent is not None else None)
                    or (getattr(sidecar_spec, "provider_id", None) if sidecar_spec is not None else None)
                    or (getattr(coprocessor_route, "provider_id", None) if coprocessor_route is not None else None)
                    or (model.provider_id if model else None)
                )
                provider_id = str(provider_value) if provider_value else None
                if max_attempts_per_model is not None:
                    current_route_key = route_key({
                        "model": model_id,
                        "endpoint": route.get("endpoint_override") or "auto",
                        "provider_id": provider_id,
                    })
                    route_attempts = sum(
                        route_key({
                            "model": item.get("model_id") or "",
                            "endpoint": item.get("endpoint_id") or "auto",
                            "provider_id": item.get("provider_id"),
                        }) == current_route_key
                        for item in role_executions
                    )
                    if route_attempts >= int(max_attempts_per_model):
                        continue
                provider = registry.providers.get(provider_id) if provider_id else None
                if controller_phase:
                    execution_kind = "controller_action"
                if execution_kind not in {
                    "native_agent", "controller_action", "sidecar_call", "coprocessor_call"
                }:
                    continue
                if execution_kind == "coprocessor_call" and coprocessor_id:
                    try:
                        coprocessor = registry.get_coprocessor(coprocessor_id)
                    except KeyError:
                        continue
                    if not coprocessor.enabled:
                        continue
                if (
                    execution_kind == "native_agent"
                    and provider_id is not None
                    and provider
                    and not self.provider_agent_capacity_available(
                        provider_id,
                        provider.limits.max_active_agents,
                        lane="worker",
                        lane_limit=provider.limits.max_worker_concurrency,
                    )
                ):
                    continue
                worker_entry: dict[str, Any] | None = None
                native_name = "controller-direct" if controller_phase else (
                    sidecar_agent.native_agent_name
                    if sidecar_agent is not None
                    else registry.native_agent_name(model_id, role)
                    if hasattr(registry, "native_agent_name")
                    else f"brigade-{role}"
                )
                if execution_kind == "sidecar_call" and not controller_phase:
                    native_name = f"sidecar-{role}"
                resolve_worker = getattr(registry, "resolve_native_worker", None)
                if callable(resolve_worker) and sidecar_agent is not None:
                    resolved_worker = resolve_worker(
                        sidecar_agent_id, model_id, role,
                        sidecar_profile_id=sidecar_profile_id,
                    )
                    if isinstance(resolved_worker, dict):
                        worker_entry = resolved_worker
                elif callable(resolve_worker) and execution_kind == "native_agent" and not controller_phase:
                    resolved_worker = resolve_worker(phase_agent_id, model_id, role)
                    if isinstance(resolved_worker, dict):
                        worker_entry = resolved_worker
                if worker_entry is not None:
                    native_name = str(worker_entry["native_agent_name"])
                effective_can_mutate = bool(
                    worker_entry is not None
                    and worker_entry.get("can_mutate") is True
                    and phase.get("mutating")
                )
                slot_info = _action_slot(
                    str(run_id), str(epoch_id), str(phase["phase_id"]), str(role),
                    native_name, executions, claims, max_attempts,
                    include_claimed=include_claimed,
                )
                if slot_info is None:
                    continue
                slot_index, package_id, action_id = slot_info
                claim = claims.get(action_id)
                action = {
                    "action_id": action_id,
                    # Controller phases are first-class controller actions.
                    # They must not create a native spawn intent or consume a
                    # worker reservation merely because older callers used
                    # the native-agent discriminator for every action.
                    "action_kind": "controller_action" if controller_phase else execution_kind,
                    "legacy_action_kind": "native_agent" if controller_phase else execution_kind,
                    "controller_action_kind": (
                        "controller_" + str(phase["phase_id"]).replace("-", "_")
                        if controller_phase else None
                    ),
                    "display_name": (
                        "Controller " + str(phase["phase_id"]).replace("-", " ").title()
                        if controller_phase else None
                    ),
                    "display_summary": (
                        (
                            "Publish disjoint work packages for downstream mutation, "
                            "then publish the planning evidence"
                        )
                        if controller_phase and str(phase.get("produces") or "") == "work_packages"
                        else "Perform the controller-owned phase and publish its evidence"
                        if controller_phase else None
                    ),
                    "progress_total": 1 if controller_phase else None,
                    "progress_completed": 0 if controller_phase else None,
                    "phase_id": phase["phase_id"],
                    "role": role,
                    "native_agent_name": native_name,
                    "model_id": model_id,
                    "endpoint": route.get("endpoint_override") or "auto",
                    "provider_id": provider_id,
                    "candidate_index": int(route.get("candidate_index") or 0),
                    "route_digest": route_digest({
                        "model": model_id,
                        "endpoint": route.get("endpoint_override") or "auto",
                        "provider_id": provider_id,
                    }),
                    "status": "ready",
                    "max_fanout": phase.get("max_fanout"),
                    "max_parallelism": phase.get("max_parallelism") or phase.get("max_fanout") or 1,
                    "launch_policy": phase.get("launch_policy") or "minimum_first",
                    "initial_fanout": phase.get("initial_fanout") or phase.get("min_fanout") or 1,
                    "maximum_replicas": phase.get("maximum_replicas") or phase.get("max_fanout") or 1,
                    "required_successes": phase.get("required_successes") or max(
                        int(phase.get("min_fanout") or 1),
                        int(phase.get("quality_quorum") or 1),
                    ),
                    "completion_mode": phase.get("completion_mode") or "quorum",
                    "execution_count": len(executions),
                    "active_execution_count": active_count,
                    "accepted_execution_count": sum(
                        item.get("accepted_by_controller") in {True, 1}
                        for item in executions
                    ),
                    "fanout_from": phase.get("fanout_from"),
                    "produces": phase.get("produces"),
                    "required_action": (
                        "publish_work_packages for the downstream mutating phase, "
                        "then complete this controller phase with evidence"
                        if controller_phase and str(phase.get("produces") or "") == "work_packages"
                        else None
                    ),
                    "package_id": package_id,
                    "current_fanout": slot_index,
                    "requires_main_controller": controller_phase,
                    "worker_kind": "controller" if controller_phase else (
                        "sidecar_agent" if sidecar_agent is not None else "native_role"
                    ),
                    "worker_id": (
                        (sidecar_agent.worker_id if sidecar_agent is not None else None)
                        or sidecar_agent_id
                        or phase_agent_id
                    ),
                    "agent_id": phase_agent_id,
                    "native_slot": None,
                    "expected_model_alias": None,
                    "priority_class": "implementation" if effective_can_mutate else "worker",
                    "can_mutate": effective_can_mutate,
                    "workspace_policy": (
                        "worktree" if effective_can_mutate
                        else "none"
                    ),
                }
                if worker_entry is not None:
                    action.update({
                        "native_agent_name": worker_entry["native_agent_name"],
                        "worker_kind": (
                            "sidecar_agent" if sidecar_agent is not None
                            else str(worker_entry.get("source_kind") or "native_role")
                        ),
                        "worker_id": (
                            worker_entry.get("worker_id")
                            or sidecar_agent_id
                            or worker_entry.get("agent_id")
                            or worker_entry.get("source_id")
                        ),
                        "agent_id": worker_entry.get("agent_id"),
                        "native_slot": worker_entry.get("slot"),
                        "expected_model_alias": worker_entry.get("model_alias"),
                        "priority_class": (
                            "critical" if worker_entry.get("slot") == "fable"
                            else "verification" if "verification" in (worker_entry.get("roles") or [])
                            else "implementation" if action["can_mutate"]
                            else "worker"
                        ),
                    "tool_policy": {
                            "tools": list(worker_entry.get("tools") or []),
                            "disallowed_tools": list(worker_entry.get("disallowed_tools") or []),
                            "can_mutate": action["can_mutate"],
                        },
                        "capability_snapshot": {
                            "roles": list(worker_entry.get("roles") or [role]),
                            "tools": list(worker_entry.get("tools") or []),
                            "disallowed_tools": list(worker_entry.get("disallowed_tools") or []),
                            "can_mutate": action["can_mutate"],
                            "isolation": action["workspace_policy"],
                            "may_spawn_agents": bool(worker_entry.get("may_spawn_agents")),
                            "may_integrate": bool(worker_entry.get("may_integrate")),
                            "may_adjudicate": bool(worker_entry.get("may_adjudicate")),
                            "counts_as_implementation": bool(worker_entry.get("counts_as_implementation")),
                        },
                        "background": bool(worker_entry.get("background", True)),
                        "max_turns": worker_entry.get("max_turns"),
                    })
                if sidecar_spec is not None:
                    action.update({
                        "sidecar_id": sidecar_id,
                        "sidecar_mode": sidecar_spec.mode,
                        "timeout_seconds": sidecar_spec.timeout_seconds,
                        "max_packet_bytes": sidecar_spec.max_packet_bytes,
                        "max_output_tokens": sidecar_spec.max_output_tokens,
                    })
                if coprocessor is not None:
                    action.update({
                        "coprocessor_id": coprocessor_id,
                        "coprocessor_mode": coprocessor.mode,
                        "timeout_seconds": coprocessor.timeout_seconds,
                        "max_packet_bytes": coprocessor.max_packet_bytes,
                        "max_output_tokens": coprocessor.max_output_tokens,
                    })
                if claim is not None:
                    action.update({
                        "status": str(claim["status"]),
                        "claim_token": claim["claim_token"],
                        "reservation_id": claim.get("reservation_id"),
                        "intent_id": claim.get("intent_id"),
                    })
                actions.append(action)
        # The single-action API remains compatible with existing controllers.
        # Controllers that can launch a worker wave use
        # ``get_runnable_action_wave`` below, which expands persisted phase
        # capacity into stable package/slot IDs.
        return actions

    def get_runnable_action_wave(
        self,
        run_id: str,
        epoch_id: str,
        *,
        limit: int | None = None,
        include_claimed: bool = False,
    ) -> list[dict]:
        """Return a bounded wave of independently claimable native actions.

        The legacy API intentionally emits one retry action per phase. This
        API expands only phases that advertise parallel capacity. Synthetic
        package IDs are deterministic until a controller publishes explicit
        work packages, so concurrent callers never derive identity from a
        mutable execution count.
        """
        claims = self._active_action_claims(run_id, epoch_id)
        wave: list[dict] = []
        provider_budget: dict[str, int] = {}
        provider_used: dict[str, int] = {}
        active_package_scopes: list[list[str]] = []
        all_packages = self.get_work_packages(run_id, epoch_id)
        phase_rows = {
            str(item.get("phase_id")): item
            for item in self.get_workflow_phases(run_id, epoch_id)
        }
        for package in all_packages:
            phase = phase_rows.get(str(package.get("phase_id"))) or {}
            if bool(phase.get("mutating")) and str(package.get("status") or "") in {"claimed", "running"}:
                active_package_scopes.append(list(package.get("path_scope") or []))
        selected_package_scopes: list[list[str]] = []
        try:
            from enhanced_router.registry import get_registry

            registry = get_registry()
        except Exception:
            registry = None
        resource_snapshot = self.get_run_resource_capacity(run_id, epoch_id)
        resource_policy = resource_snapshot.get("policy") or {}
        resource_counts = dict(resource_snapshot.get("active") or {})
        resource_deadline_exceeded = bool(resource_snapshot.get("deadline_exceeded"))
        for action in self.get_runnable_actions(
            run_id, epoch_id, include_claimed=True
        ):
            if action.get("action_kind") != "native_agent":
                projected = dict(resource_counts)
                self._increment_counts(projected, action)
                if self._capacity_reasons(
                    resource_policy,
                    projected,
                    deadline_exceeded=resource_deadline_exceeded,
                ):
                    continue
                wave.append(action)
                resource_counts = projected
                continue
            packages = self.get_ready_work_packages(
                run_id, epoch_id, str(action["phase_id"])
            )
            # Mutation fanout is package fanout.  A phase without persisted
            # package contracts is allowed one generic action only; this
            # prevents several workers from receiving the same whole-task
            # prompt and racing over the same files.
            launch_policy = str(action.get("launch_policy") or "minimum_first")
            execution_count = int(action.get("execution_count") or 0)
            active_execution_count = int(action.get("active_execution_count") or 0)
            accepted_execution_count = int(action.get("accepted_execution_count") or 0)
            required_successes = int(action.get("required_successes") or 1)
            if accepted_execution_count >= required_successes:
                # The phase already has enough controller-accepted evidence.
                # Do not spend another worker request merely because its
                # configured maximum fanout is larger than its quorum.
                continue
            if launch_policy == "minimum_first" and active_execution_count:
                # Minimum-first launches one evidence-producing attempt at a
                # time.  A later attempt is considered only after the current
                # attempt reaches terminal state or is rejected.
                continue
            if action.get("can_mutate") and packages:
                package_slots = [
                    package for package in packages
                    if bool(package.get("can_run_parallel", True))
                    or len(packages) == 1
                ]
            elif action.get("can_mutate") and str(action.get("fanout_from") or "") == "work_packages":
                package_slots = []
            elif action.get("can_mutate"):
                package_slots = [None]
            else:
                if packages:
                    package_slots = packages
                else:
                    # Read-only replicas are safe to synthesize.  Their
                    # identities remain stable and are bounded by the phase
                    # launch policy; mutators never use this path.
                    replica_limit = int(
                        action.get("maximum_replicas")
                        or action.get("max_fanout")
                        or 1
                    )
                    package_slots = [None] * max(1, replica_limit)
            if launch_policy == "minimum_first":
                initial_fanout = int(action.get("initial_fanout") or 1)
                maximum_replicas = int(
                    action.get("maximum_replicas")
                    or action.get("max_fanout")
                    or initial_fanout
                )
                desired = initial_fanout if execution_count == 0 else min(
                    maximum_replicas, execution_count + 1,
                )
                package_slots = package_slots[:max(1, desired)]
            maximum = min(
                len(package_slots),
                int(action.get("max_parallelism") or action.get("max_fanout") or 1),
            )
            start = int(action.get("current_fanout") or 0)
            if maximum <= start:
                continue
            for slot in range(start, maximum):
                # The legacy projection normally carries slot-0 as
                # ``package_id``. Reusing it for every expanded slot would
                # make concurrent wave entries share an action identity.
                package = package_slots[slot]
                if action.get("can_mutate") and package is not None:
                    package_scope = list(package.get("path_scope") or [])
                    if any(_path_scopes_overlap(package_scope, existing_scope)
                           for existing_scope in [*active_package_scopes, *selected_package_scopes]):
                        continue
                package_id = str(
                    (package or {}).get("package_id")
                    or action.get("work_package_id")
                    or f"{action['phase_id']}:{action['role']}:slot-{slot}"
                )
                clone = dict(action)
                clone.update({
                    "package_id": package_id,
                    "current_fanout": slot,
                    "candidate_index": action.get("candidate_index", 0),
                    "action_id": (
                        f"action:{run_id}:{epoch_id}:{action['phase_id']}:{package_id}:"
                        f"{action['native_agent_name']}:attempt-1"
                    ),
                })
                if package is not None:
                    clone.update({
                        "work_package_id": package["package_id"],
                        "package_contract_digest": package.get("contract_digest"),
                        "prompt_contract_digest": package.get("prompt_contract_digest")
                        or package.get("contract_digest"),
                        "package_contract_version": package.get("contract_version", 1),
                        "package_objective": package.get("objective"),
                        "package_display_name": package.get("display_name") or package.get("objective"),
                        "package_summary": package.get("summary") or package.get("objective"),
                        "path_scope": package.get("path_scope", []),
                        "requirements": package.get("requirements", []),
                        "acceptance": package.get("acceptance", []),
                        "required_tests": package.get("required_tests", []),
                        "prohibited_paths": package.get("prohibited_paths", []),
                        "prompt": package.get("prompt_contract") or {
                            "objective": package.get("objective"),
                            "path_scope": package.get("path_scope", []),
                            "acceptance": package.get("acceptance", []),
                            "required_tests": package.get("required_tests", []),
                            "prohibited_paths": package.get("prohibited_paths", []),
                        },
                    })
                claim = claims.get(clone["action_id"])
                if claim is not None and not include_claimed:
                    continue
                if claim is not None:
                    clone.update({
                        "status": str(claim["status"]),
                        "claim_token": claim["claim_token"],
                        "reservation_id": claim.get("reservation_id"),
                        "intent_id": claim.get("intent_id"),
                    })
                provider_id = str(clone.get("provider_id") or "").strip()
                if provider_id and registry is not None:
                    provider = registry.providers.get(provider_id)
                    if provider is not None and provider_id not in provider_budget:
                        provider_budget[provider_id] = self.provider_agent_capacity_remaining(
                            provider_id,
                            provider.limits.max_active_agents,
                            lane="worker",
                            lane_limit=provider.limits.max_worker_concurrency,
                        )
                    if (
                        provider_id in provider_budget
                        and provider_used.get(provider_id, 0) >= provider_budget[provider_id]
                    ):
                        continue
                projected = dict(resource_counts)
                self._increment_counts(projected, clone)
                if self._capacity_reasons(
                    resource_policy,
                    projected,
                    deadline_exceeded=resource_deadline_exceeded,
                ):
                    continue
                wave.append(clone)
                resource_counts = projected
                if action.get("can_mutate") and package is not None:
                    selected_package_scopes.append(list(package.get("path_scope") or []))
                if provider_id in provider_budget:
                    provider_used[provider_id] = provider_used.get(provider_id, 0) + 1
                if limit is not None and len(wave) >= limit:
                    return wave[:limit]
        return wave[:limit] if limit is not None else wave

    def claim_runnable_action(
        self,
        run_id: str,
        epoch_id: str,
        action_id: str,
        *,
        ttl_seconds: int = 90,
    ) -> dict:
        """Claim one controller-planned native action before spawning it.

        This is deliberately separate from provider admission.  The claim
        prevents duplicate native spawns; the durable provider reservation
        ensures the claim only succeeds when the worker can start now.  A
        denied native Agent call is never queued for later replay.
        """
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        actions = self.get_runnable_actions(run_id, epoch_id, include_claimed=True)
        action = next(
            (item for item in actions if item.get("action_id") == action_id),
            None,
        )
        if action is None:
            action = next(
                (
                    item for item in self.get_runnable_action_wave(
                        run_id, epoch_id, include_claimed=True
                    )
                    if item.get("action_id") == action_id
                ),
                None,
            )
        if action is None:
            raise WorkflowStateError(
                f"runnable action {action_id!r} is no longer available"
            )
        if action.get("controller_action_kind") or str(action.get("action_kind") or "").startswith("controller_"):
            phase = next(
                (item for item in self.get_workflow_phases(run_id, epoch_id)
                 if item.get("phase_id") == action.get("phase_id")),
                None,
            )
            if phase is not None and phase.get("status") == "pending":
                self.start_phase(
                    run_id, epoch_id, str(action["phase_id"]),
                    actor="controller", principal="runnable-action-claim",
                )
                actions = self.get_runnable_actions(run_id, epoch_id, include_claimed=True)
                action = next((item for item in actions if item.get("action_id") == action_id), action)
        if action.get("status") in {"claimed", "consumed"}:
            return action

        run = self.get_run(run_id)
        token_reservation_id: str | None = None
        if run and run.get("token_budget") is not None:
            token_reservation_id = f"tokens:{action_id}"
            estimated_tokens = int(
                action.get("estimated_tokens")
                or action.get("max_output_tokens")
                or 4096
            )
            token_reservation = self.reserve_token_budget(
                reservation_id=token_reservation_id,
                run_id=run_id,
                epoch_id=epoch_id,
                action_id=action_id,
                execution_id=f"pending:{action_id}",
                estimated_tokens=max(1, estimated_tokens),
            )
            if token_reservation.get("state") != "reserved":
                raise WorkflowStateError(
                    str(token_reservation.get("reason") or "token budget unavailable")
                )

        claim_token = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        reservation_id = f"action:{action_id}"
        intent_id = f"intent:{action_id}"
        provider_id = action.get("provider_id")
        reservation: dict | None = None
        if provider_id and action.get("action_kind") == "native_agent" and not action.get("controller_action_kind"):
            try:
                from enhanced_router.registry import get_registry

                provider = get_registry().providers.get(str(provider_id))
                if provider is None:
                    raise WorkflowStateError(f"provider {provider_id!r} is not configured")
                reservation = self.reserve_provider_agent(
                    reservation_id=reservation_id,
                    run_id=run_id,
                    epoch_id=epoch_id,
                    provider_id=str(provider_id),
                    execution_id=f"pending:{action_id}",
                    lane="worker",
                    max_active=provider.limits.max_active_agents,
                    lane_limit=provider.limits.max_worker_concurrency,
                    deadline_at=expires_at,
                    reason=f"action-claim:{action_id}",
                    enqueue=False,
                    model_id=action.get("model_id"),
                )
                if reservation is None or reservation.get("state") != "reserved":
                    raise WorkflowStateError(
                        f"provider {provider_id!r} has no capacity for action {action_id!r}"
                    )
            except Exception:
                if token_reservation_id:
                    self.release_token_reservation(token_reservation_id, "cancelled")
                raise

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # The wave planner is advisory. Re-check the run-wide policy while
            # holding the claim transaction so concurrent controllers cannot
            # exceed the same run's native/mutator/reviewer/coprocessor or
            # worktree ceilings between planning and claim.
            self._assert_action_capacity(conn, run_id, epoch_id, action)
            epoch_row = conn.execute(
                "SELECT mutation_paused, escalation_state FROM epochs "
                "WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if (
                action.get("can_mutate")
                and epoch_row is not None
                and (
                    bool(epoch_row[0])
                    or str(epoch_row[1] or "") == "escalated"
                )
            ):
                conn.rollback()
                if reservation is not None:
                    self.release_provider_reservation(reservation_id, "cancelled")
                if token_reservation_id:
                    self.release_token_reservation(token_reservation_id, "cancelled")
                raise WorkflowStateError(
                    "mutating action admission is paused pending escalation acknowledgment"
                )
            existing = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (action_id,),
            ).fetchone()
            if existing is not None and existing["status"] in {"claimed", "consumed"}:
                conn.rollback()
                if reservation is not None:
                    self.release_provider_reservation(reservation_id, "cancelled")
                if token_reservation_id:
                    self.release_token_reservation(token_reservation_id, "cancelled")
                return dict(existing)
            pending = conn.execute(
                "SELECT c.action_id FROM runnable_action_claims AS c "
                "LEFT JOIN spawn_intents AS i ON i.intent_id=c.intent_id "
                "WHERE c.run_id=? AND c.epoch_id=? AND c.native_agent_name=? AND c.role=? "
                "AND c.status IN ('claimed','consumed') AND c.claude_agent_id IS NULL "
                "AND c.action_id != ? AND COALESCE(i.package_id, '') = COALESCE(?, '') LIMIT 1",
                (
                    run_id, epoch_id, action["native_agent_name"], action["role"],
                    action_id, action.get("package_id"),
                ),
            ).fetchone()
            if pending is not None:
                conn.rollback()
                if reservation is not None:
                    self.release_provider_reservation(reservation_id, "cancelled")
                if token_reservation_id:
                    self.release_token_reservation(token_reservation_id, "cancelled")
                raise WorkflowStateError(
                    "a native action with the same name and role is already awaiting "
                    f"lifecycle attachment: {pending[0]}"
                )
            action_kind = str(action.get("action_kind") or "native_agent")
            values = (
                action_id, run_id, epoch_id, action["phase_id"], action["role"],
                action["native_agent_name"], action["model_id"], action_kind,
                action.get("endpoint") or "auto", provider_id,
                action.get("route_digest"), action.get("candidate_index", 0),
                token_reservation_id,
                claim_token, reservation_id if reservation is not None else None,
                intent_id if action_kind == "native_agent" else None,
                "claimed", _utcnow(), _utcnow(), expires_at,
            )
            if existing is None:
                conn.execute(
                    "INSERT INTO runnable_action_claims "
                    "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, action_kind, "
                    "endpoint_id, provider_id, route_digest, candidate_index, token_reservation_id, "
                    "claim_token, reservation_id, intent_id, status, created_at, claimed_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
            else:
                conn.execute(
                    "UPDATE runnable_action_claims SET run_id=?, epoch_id=?, phase_id=?, role=?, "
                    "native_agent_name=?, model_id=?, action_kind=?, endpoint_id=?, provider_id=?, "
                    "route_digest=?, candidate_index=?, token_reservation_id=?, claim_token=?, "
                    "reservation_id=?, intent_id=?, status='claimed', claimed_at=?, "
                    "expires_at=?, consumed_at=NULL, claude_agent_id=NULL, execution_id=NULL WHERE action_id=?",
                    (run_id, epoch_id, action["phase_id"], action["role"],
                     action["native_agent_name"], action["model_id"], action_kind,
                     action.get("endpoint") or "auto", provider_id, action.get("route_digest"),
                     action.get("candidate_index", 0), token_reservation_id,
                     claim_token, reservation_id if reservation is not None else None,
                     intent_id if action_kind == "native_agent" else None,
                     values[18], values[19], action_id),
                )
            policy_json = json.dumps(
                {
                    "action_id": action_id,
                    "claim_token": claim_token,
                    "worker_kind": action.get("worker_kind", "native_role"),
                    "worker_id": action.get("worker_id"),
                    "can_mutate": bool(action.get("can_mutate")),
                    "workspace_policy": action.get("workspace_policy", "none"),
                    "background": bool(action.get("background", True)),
                    "tool_policy": action.get("tool_policy") or {},
                    "max_turns": action.get("max_turns"),
                    "agent_definition_id": action.get("agent_id"),
                    "native_slot": action.get("native_slot"),
                    "expected_model_alias": action.get("expected_model_alias"),
                    "priority_class": action.get("priority_class"),
                    "package_id": action.get("package_id"),
                    "package_contract_digest": action.get("package_contract_digest"),
                    "prompt_contract_digest": action.get("prompt_contract_digest"),
                },
                separators=(",", ":"),
            )
            capability_json = json.dumps(
                action.get("capability_snapshot") or action.get("tool_policy") or {},
                sort_keys=True,
                separators=(",", ":"),
            )
            capability_digest = hashlib.sha256(capability_json.encode()).hexdigest()
            intent_exists = conn.execute(
                "SELECT 1 FROM spawn_intents WHERE intent_id=?", (intent_id,)
            ).fetchone()
            if action.get("action_kind") != "native_agent":
                pass
            elif intent_exists is None:
                conn.execute(
                    "INSERT INTO spawn_intents "
                    "(intent_id, run_id, epoch_id, phase_id, native_agent_name, role, model_id, "
                    "endpoint_id, provider_id, route_digest, candidate_index, worker_kind, worker_id, capability_snapshot_json, "
                    "prompt_contract_digest, workspace_policy, background, package_id, agent_definition_id, native_slot, "
                    "public_model_alias, capability_digest, priority_class, status, created_at, policy_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?)",
                    (intent_id, run_id, epoch_id, action["phase_id"], action["native_agent_name"],
                     action["role"], action["model_id"], action.get("endpoint") or "auto", provider_id,
                     action.get("route_digest"), action.get("candidate_index", 0),
                     action.get("worker_kind", "native_role"), action.get("worker_id"),
                     capability_json, action.get("prompt_contract_digest"),
                     action.get("workspace_policy", "none"),
                     1 if action.get("background", True) else 0,
                     action.get("package_id"), action.get("agent_id"),
                     action.get("native_slot"), action.get("expected_model_alias"),
                     capability_digest, action.get("priority_class"),
                     _utcnow(), policy_json),
                )
            elif action.get("action_kind") == "native_agent":
                conn.execute(
                    "UPDATE spawn_intents SET status='planned', spawned_at=NULL, completed_at=NULL, "
                    "claude_agent_id=NULL, worker_kind=?, worker_id=?, capability_snapshot_json=?, "
                    "endpoint_id=?, provider_id=?, route_digest=?, candidate_index=?, "
                    "prompt_contract_digest=?, workspace_policy=?, background=?, package_id=?, agent_definition_id=?, native_slot=?, "
                    "public_model_alias=?, capability_digest=?, priority_class=?, policy_json=? WHERE intent_id=?",
                    (action.get("worker_kind", "native_role"), action.get("worker_id"),
                     capability_json,
                     action.get("endpoint") or "auto", provider_id, action.get("route_digest"),
                     action.get("candidate_index", 0),
                     action.get("prompt_contract_digest"), action.get("workspace_policy", "none"),
                     1 if action.get("background", True) else 0,
                     action.get("package_id"), action.get("agent_id"),
                     action.get("native_slot"), action.get("expected_model_alias"),
                     capability_digest, action.get("priority_class"), policy_json, intent_id),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?", (action_id,)
            ).fetchone()
            assert result is not None
            package_id = str(action.get("package_id") or "")
            if package_id:
                try:
                    self.update_work_package(
                        package_id,
                        status="claimed",
                        reason=f"action claimed: {action_id}",
                    )
                except Exception:
                    LOGGER.debug("unable to mark package %s claimed", package_id, exc_info=True)
            return {**action, **dict(result), "status": "claimed"}
        except Exception:
            conn.rollback()
            if reservation is not None:
                self.release_provider_reservation(reservation_id, "cancelled")
            if token_reservation_id:
                self.release_token_reservation(token_reservation_id, "cancelled")
            raise
        finally:
            conn.close()

    def consume_controller_action(
        self, run_id: str, epoch_id: str, action_id: str,
    ) -> dict | None:
        """Consume a claimed controller-integration action exactly once."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=? "
                "AND run_id=? AND epoch_id=? AND role='controller' "
                "AND status='claimed' AND expires_at >= ?",
                (action_id, run_id, epoch_id, _utcnow()),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            now = _utcnow()
            conn.execute(
                "UPDATE runnable_action_claims SET status='consumed', consumed_at=? "
                "WHERE action_id=? AND status='claimed'",
                (now, action_id),
            )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (action_id,),
            ).fetchone()
            return dict(result) if result is not None else None
        finally:
            conn.close()

    def finish_controller_action(
        self,
        run_id: str,
        epoch_id: str,
        action_id: str,
        status: str,
    ) -> dict | None:
        """Terminalize a consumed controller-integration action.

        Controller integration is a two-step operation: the controller first
        consumes the claim immediately before acting, then the integration or
        resolution operation records its outcome.  Keeping the claim in
        ``consumed`` after that outcome would make it look active forever and
        would prevent the same candidate from being claimed again after a
        failed integration or an explicit retry decision.
        """
        if status not in {"completed", "failed", "cancelled", "orphaned"}:
            raise ValueError(f"invalid controller action status: {status}")
        result_row = None
        token_reservation_id: str | None = None
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            updated = conn.execute(
                "UPDATE runnable_action_claims SET status=?, consumed_at=COALESCE(consumed_at, ?) "
                "WHERE action_id=? AND run_id=? AND epoch_id=? AND role='controller' "
                "AND (action_kind='controller_integration' OR action_kind LIKE 'controller_%') "
                "AND status='consumed'",
                (status, now, action_id, run_id, epoch_id),
            )
            if updated.rowcount != 1:
                conn.rollback()
            else:
                reservation = conn.execute(
                    "SELECT token_reservation_id FROM runnable_action_claims "
                    "WHERE action_id=? AND run_id=? AND epoch_id=?",
                    (action_id, run_id, epoch_id),
                ).fetchone()
                token_reservation_id = (
                    str(reservation[0]) if reservation is not None and reservation[0] else None
                )
                conn.commit()
                result_row = conn.execute(
                    "SELECT * FROM runnable_action_claims WHERE action_id=?",
                    (action_id,),
                ).fetchone()
        finally:
            conn.close()
        if token_reservation_id:
            self.release_token_reservation(
                token_reservation_id,
                "consumed" if status == "completed" else "released",
            )
        return dict(result_row) if result_row is not None else None

    def finish_controller_actions_for_phase(
        self,
        run_id: str,
        epoch_id: str,
        phase_id: str,
        status: str = "completed",
    ) -> int:
        """Close the controller claim attached to a completed phase.

        Controller phases have no native child lifecycle to emit a stop hook,
        so phase completion must terminalize their consumed claim explicitly.
        This prevents a completed planning/adjudication action from continuing
        to consume run-wide capacity.
        """
        if status not in {"completed", "failed", "cancelled", "orphaned"}:
            raise ValueError(f"invalid controller action status: {status}")
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT action_id, status FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND phase_id=? AND role='controller' "
                "AND action_kind LIKE 'controller_%' "
                "AND status IN ('claimed','consumed')",
                (run_id, epoch_id, phase_id),
            ).fetchall()
        finally:
            conn.close()
        finished = 0
        for row in rows:
            action_id = str(row[0])
            if str(row[1]) == "claimed":
                self.consume_controller_action(run_id, epoch_id, action_id)
            if self.finish_controller_action(run_id, epoch_id, action_id, status) is not None:
                finished += 1
        return finished

    # start_sidecar_execution, start_detached_sidecar_execution,
    # append_execution_event, prepare_sidecar_retry, get_execution_events
    # live in SidecarExecutionRepository (sidecar_execution_state.py),
    # mixed in below.

    # consume_runnable_action_for_spawn, attach_spawned_agent,

    # get_pending_spawn_claim, get_unattached_spawn_claim_for_role,

    # fail_spawn_claim, get_spawn_assignment, finish_spawn_assignment live in

    # NativeSpawnAttachRepository (native_spawn_attach_state.py), mixed in

    # below.

    def cancel_action_claims(self, run_id: str) -> int:
        """Cancel uncompleted cooperative actions during session teardown."""
        conn = self._new_conn()
        reservation_ids: list[str] = []
        token_reservation_ids: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
            "SELECT intent_id, reservation_id, token_reservation_id FROM runnable_action_claims "
                "WHERE run_id=? AND status IN ('claimed','consumed')",
                (run_id,),
            ).fetchall()
            reservation_ids = [str(row[1]) for row in rows if row[1]]
            token_reservation_ids = [str(row[2]) for row in rows if row[2]]
            now = _utcnow()
            result = conn.execute(
                "UPDATE runnable_action_claims SET status='cancelled', consumed_at=COALESCE(consumed_at, ?) "
                "WHERE run_id=? AND status IN ('claimed','consumed')",
                (now, run_id),
            )
            for row in rows:
                if row[0]:
                    conn.execute(
                        "UPDATE spawn_intents SET status='cancelled', completed_at=? "
                        "WHERE intent_id=? AND status IN ('planned','spawned')",
                        (now, row[0]),
                    )
            conn.commit()
            count = int(result.rowcount)
        finally:
            conn.close()
        for reservation_id in reservation_ids:
            self.release_provider_reservation(reservation_id, "cancelled")
        for reservation_id in token_reservation_ids:
            self.release_token_reservation(reservation_id, "cancelled")
        return count

    def reconcile_lifecycle(
        self, run_id: str | None = None, *, max_age_seconds: int = 900,
        detached_max_age_seconds: int | None = None,
    ) -> dict[str, int]:
        """Reconcile claims, spawn intents, and reservations left by crashes.

        Only records older than ``max_age_seconds`` are considered orphaned so
        a normal lifecycle race is not mistaken for a crash.  The operation is
        idempotent and returns counts suitable for health/status reporting.
        ``detached_max_age_seconds`` is an independent startup control: the
        router invokes it with zero because detached coprocessors have no
        native child lifecycle and any active row belongs to the prior
        process.
        """
        cutoff = _utcnow_age(max_age_seconds)
        detached_cutoff = _utcnow_age(
            max_age_seconds if detached_max_age_seconds is None
            else detached_max_age_seconds
        )
        conn = self._new_conn()
        reservation_ids: list[str] = []
        provider_ids: set[str] = set()
        orphaned_execution_ids: list[str] = []
        detached_execution_ids: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            run_clause = "" if run_id is None else " AND run_id=?"
            run_params: tuple[object, ...] = () if run_id is None else (run_id,)
            # Detached sidecar executions (fastpath jobs: start_detached_sidecar_execution)
            # have no backing runnable_action_claims row, so the claim-based
            # orphan detection below never sees them. A router restart mid-job
            # would otherwise leave them 'started'/'running' forever -- and
            # even an explicit retry() call requires a terminal failed/timeout
            # status first, so nothing could ever recover them.
            detached = conn.execute(
                "SELECT execution_id FROM agent_executions "
                "WHERE execution_kind IN ('sidecar_call','coprocessor_call') AND status IN ('started','running') "
                "AND started_at < ?" + run_clause + " AND execution_id NOT IN "
                "(SELECT execution_id FROM runnable_action_claims WHERE execution_id IS NOT NULL)",
                (detached_cutoff, *run_params),
            ).fetchall()
            detached_execution_ids = [str(row[0]) for row in detached]
            expired = conn.execute(
                "SELECT reservation_id, provider_id FROM provider_reservations "
                "WHERE state IN ('queued','reserved') AND deadline_at IS NOT NULL "
                "AND deadline_at < ?" + run_clause,
                (cutoff, *run_params),
            ).fetchall()
            reservation_ids = [str(row[0]) for row in expired if row[0]]
            provider_ids.update(str(row[1]) for row in expired if row[1])
            claims = conn.execute(
                "SELECT c.action_id, c.reservation_id, c.intent_id, r.provider_id, c.execution_id, "
                "c.token_reservation_id "
                "FROM runnable_action_claims AS c LEFT JOIN provider_reservations AS r "
                "ON r.reservation_id=c.reservation_id "
                "WHERE c.status IN ('claimed','consumed') AND "
                "COALESCE(c.consumed_at, c.claimed_at) < "
                "?" + (" AND c.run_id=?" if run_id is not None else ""),
                (cutoff, *run_params),
            ).fetchall()
            claim_ids = [str(row[0]) for row in claims]
            claim_reservations = [str(row[1]) for row in claims if row[1]]
            token_reservation_ids = [str(row[5]) for row in claims if row[5]]
            provider_ids.update(str(row[3]) for row in claims if row[3])
            reservation_ids.extend(claim_reservations)
            orphaned_execution_ids = [str(row[4]) for row in claims if row[4]]
            claim_count = 0
            for row in claims:
                action_id, reservation_id, intent_id, _provider_id, _execution_id, _token_reservation_id = row
                conn.execute(
                    "UPDATE runnable_action_claims SET status='orphaned', "
                    "consumed_at=COALESCE(consumed_at, ?) WHERE action_id=? "
                    "AND status IN ('claimed','consumed')",
                    (_utcnow(), action_id),
                )
                if intent_id:
                    conn.execute(
                        "UPDATE spawn_intents SET status='failed', completed_at=? "
                        "WHERE intent_id=? AND status IN ('planned','spawned')",
                        (_utcnow(), intent_id),
                    )
                claim_count += 1
            reservation_count = conn.execute(
                "UPDATE provider_reservations SET state='expired', released_at=? "
                "WHERE state IN ('queued','reserved') AND deadline_at IS NOT NULL "
                "AND deadline_at < ?" + run_clause,
                (_utcnow(), cutoff, *run_params),
            ).rowcount
            conn.commit()
            result = {
                "claims_orphaned": claim_count,
                "reservations_expired": int(reservation_count),
                "spawn_intents_orphaned": len(claim_ids),
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if provider_ids:
            from enhanced_router.registry import get_registry
            registry = get_registry()
            for provider_id in provider_ids:
                provider = registry.providers.get(provider_id)
                if provider:
                    self.admit_provider_agents(provider_id, provider.limits.max_active_agents)
        for reservation_id in token_reservation_ids:
            self.release_token_reservation(reservation_id, "expired")
        executions_orphaned = 0
        for execution_id in orphaned_execution_ids:
            try:
                self.update_agent_execution(
                    execution_id=execution_id, status="timeout",
                    error="orphaned: no terminal report before crash-recovery cutoff",
                    error_class="orphaned_after_restart",
                    orphaned_at=_utcnow(),
                )
                executions_orphaned += 1
            except WorkflowStateError:
                # Already terminal (it finished right before reconciliation
                # ran) -- nothing to reconcile, not a failure.
                pass
        # Close route attempts that were in flight when the router died.  In
        # particular this makes detached fastpath executions retryable and
        # prevents an attempt from remaining perpetually ``started`` in
        # telemetry after its execution is reconciled to timeout.
        route_attempts_reconciled = 0
        try:
            route_attempts_reconciled = self.reconcile_route_attempts(
                [*orphaned_execution_ids, *detached_execution_ids],
                error_class="orphaned_after_restart",
                error="route attempt was interrupted by router restart",
            )
        except Exception:
            # Lifecycle recovery must not turn a best-effort telemetry write
            # into a startup outage; the execution timeout above remains
            # authoritative.
            LOGGER.exception("failed to reconcile route attempts")
        detached_orphaned = 0
        for execution_id in detached_execution_ids:
            try:
                self.update_agent_execution(
                    execution_id=execution_id, status="timeout",
                    error="orphaned: detached sidecar job had no terminal report before "
                          "crash-recovery cutoff",
                    error_class="orphaned_after_restart",
                    orphaned_at=_utcnow(),
                )
                detached_orphaned += 1
            except WorkflowStateError:
                pass
        result["executions_orphaned"] = executions_orphaned
        result["detached_executions_orphaned"] = detached_orphaned
        result["route_attempts_reconciled"] = route_attempts_reconciled
        return result

    def create_spawn_intent(self, **values: object) -> dict:
        required = {"intent_id", "run_id", "epoch_id", "native_agent_name", "role", "status"}
        missing = required - values.keys()
        if missing:
            raise ValueError(f"missing spawn intent fields: {sorted(missing)}")
        defaults = {
            "slot": "inherit", "model_id": None, "endpoint_id": None,
            "provider_id": None, "execution_kind": "subagent", "policy_json": "{}",
            "phase_id": None, "fallback_of": None,
        }
        defaults.update(values)
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO spawn_intents (intent_id, run_id, epoch_id, phase_id, native_agent_name,"
                " role, slot, model_id, endpoint_id, provider_id, execution_kind, status, created_at,"
                " fallback_of, policy_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (defaults["intent_id"], defaults["run_id"], defaults["epoch_id"], defaults["phase_id"],
                 defaults["native_agent_name"], defaults["role"], defaults["slot"], defaults["model_id"],
                 defaults["endpoint_id"], defaults["provider_id"], defaults["execution_kind"],
                 defaults["status"], _utcnow(), defaults["fallback_of"], defaults["policy_json"]),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM spawn_intents WHERE intent_id=?", (defaults["intent_id"],)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def update_spawn_intent(self, intent_id: str, status: str, **values: object) -> dict | None:
        if status not in {"planned", "queued", "spawned", "completed", "failed", "cancelled"}:
            raise ValueError("invalid spawn intent status")
        sets = ["status=?"]
        params: list[object] = [status]
        for column in (
            "spawned_at", "completed_at", "fallback_of", "policy_json",
            "claude_agent_id",
        ):
            if column in values:
                sets.append(f"{column}=?")
                params.append(values[column])
        if status == "spawned" and "spawned_at" not in values:
            sets.append("spawned_at=?")
            params.append(_utcnow())
        if status in {"completed", "failed", "cancelled"} and "completed_at" not in values:
            sets.append("completed_at=?")
            params.append(_utcnow())
        params.append(intent_id)
        conn = self._new_conn()
        try:
            conn.execute(f"UPDATE spawn_intents SET {', '.join(sets)} WHERE intent_id=?", params)
            conn.commit()
            row = conn.execute("SELECT * FROM spawn_intents WHERE intent_id=?", (intent_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()



    # store_provider_catalog_entries, get_provider_catalog_snapshots,
    # get_provider_catalog live in LiteLLMGenerationRepository
    # (litellm_state.py) now -- these are LiteLLM catalog persistence, not
    # run lifecycle; they only ever sat here by historical accident.
