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

import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from enhanced_router.state_errors import WorkflowStateError


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
    fallback_routes = [c for c in fallback_routes if isinstance(c, dict) and c.get("model")]
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


class RunnableActionRepository:
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
        claims = self._active_action_claims(run_id, epoch_id)
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
            actor_contract = str(phase.get("required_actor") or phase.get("actor") or "")
            controller_phase = actor_contract == "controller"
            if actor_contract and not controller_phase:
                continue
            if controller_phase and phase.get("status") != "active":
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
            sidecar_id = str(phase.get("sidecar_id") or "").strip() or None
            sidecar_spec = None
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
                run_row = self.get_run(run_id)
                sidecar_profile_id = run_row.get("sidecar_profile_id") if run_row else None
                if sidecar_id not in registry.resolve_sidecars(sidecar_profile_id):
                    continue
            for role in allowed_roles:
                controller_binding = None
                if controller_phase:
                    run = self.get_run(run_id)
                    session_id = str(run.get("claude_session_id")) if run and run.get("claude_session_id") else ""
                    controller_binding = self.get_controller_binding(run_id, session_id) if session_id else None
                    if not controller_binding or not controller_binding.get("registry_model_id"):
                        continue
                    route = {
                        "endpoint_override": controller_binding.get("endpoint_id"),
                        "model_id": controller_binding["registry_model_id"],
                    }
                elif sidecar_spec is not None:
                    route = {
                        "endpoint_override": (
                            None if sidecar_spec.endpoint == "auto" else sidecar_spec.endpoint
                        ),
                        "model_id": sidecar_spec.model_id,
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
                    item.get("status") in {"failed", "timeout", "timed_out", "cancelled", "orphaned"}
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
                    fallback_routes = _role_route_fallback_candidates(route)
                    fallback_models = [c["model"] for c in fallback_routes]
                    failed_attempts = sum(
                        item.get("status") in {"failed", "timeout", "timed_out", "cancelled", "orphaned"}
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
                model_id = str(route["model_id"])
                if max_attempts_per_model is not None:
                    model_attempts = sum(
                        item.get("model_id") == model_id for item in role_executions
                    )
                    if model_attempts >= int(max_attempts_per_model):
                        continue
                model = registry.get_model(model_id)
                provider_value = (
                    (controller_binding or {}).get("provider_id") or model.provider_id
                ) if model else None
                provider_id = str(provider_value) if provider_value else None
                provider = registry.providers.get(provider_id) if provider_id else None
                if controller_phase:
                    execution_kind = "native_agent"
                if execution_kind not in {"native_agent", "sidecar_call"}:
                    continue
                if (
                    execution_kind == "native_agent"
                    and provider_id is not None
                    and provider
                    and not self.provider_agent_capacity_available(
                        provider_id, provider.limits.max_active_agents
                    )
                ):
                    continue
                native_name = "controller-direct" if controller_phase else (
                    registry.native_agent_name(model_id, role)
                    if hasattr(registry, "native_agent_name")
                    else f"brigade-{role}"
                )
                if execution_kind == "sidecar_call" and not controller_phase:
                    native_name = f"sidecar-{role}"
                action_id = (
                    f"action:{run_id}:{epoch_id}:{phase['phase_id']}:{role}:{len(executions)}"
                )
                claim = claims.get(action_id)
                if claim is not None and not include_claimed:
                    continue
                action = {
                    "action_id": action_id,
                    "action_kind": execution_kind,
                    "phase_id": phase["phase_id"],
                    "role": role,
                    "native_agent_name": native_name,
                    "model_id": model_id,
                    "endpoint": route.get("endpoint_override") or "auto",
                    "provider_id": provider_id,
                    "status": "ready",
                    "max_fanout": phase.get("max_fanout"),
                    "current_fanout": len(executions),
                    "requires_main_controller": controller_phase,
                }
                if sidecar_spec is not None:
                    action.update({
                        "sidecar_id": sidecar_id,
                        "sidecar_mode": sidecar_spec.mode,
                        "timeout_seconds": sidecar_spec.timeout_seconds,
                        "max_packet_bytes": sidecar_spec.max_packet_bytes,
                        "max_output_tokens": sidecar_spec.max_output_tokens,
                    })
                if claim is not None:
                    action.update({
                        "status": str(claim["status"]),
                        "claim_token": claim["claim_token"],
                        "reservation_id": claim.get("reservation_id"),
                        "intent_id": claim.get("intent_id"),
                    })
                actions.append(action)
        # One action per role/phase is enough for the controller to ask again;
        # min_fanout is represented by repeated claims, not hidden fanout.
        return actions

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
            raise WorkflowStateError(
                f"runnable action {action_id!r} is no longer available"
            )
        if action.get("status") in {"claimed", "consumed"}:
            return action

        run = self.get_run(run_id)
        token_budget = run.get("token_budget") if run else None
        if token_budget is not None:
            conn = self._new_conn()
            try:
                spent = conn.execute(
                    "SELECT COALESCE(SUM(total_tokens), 0) FROM agent_executions WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
            finally:
                conn.close()
            if int(spent) >= int(token_budget):
                raise WorkflowStateError(
                    f"run {run_id!r} has exhausted its token budget "
                    f"({spent} spent, budget {token_budget})"
                )

        claim_token = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        reservation_id = f"action:{action_id}"
        intent_id = f"intent:{action_id}"
        provider_id = action.get("provider_id")
        reservation: dict | None = None
        if provider_id and action.get("action_kind") == "native_agent":
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
                deadline_at=expires_at,
                reason=f"action-claim:{action_id}",
                enqueue=False,
                model_id=action.get("model_id"),
            )
            if reservation.get("state") != "reserved":
                raise WorkflowStateError(
                    f"provider {provider_id!r} has no capacity for action {action_id!r}"
                )

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (action_id,),
            ).fetchone()
            if existing is not None and existing["status"] in {"claimed", "consumed"}:
                conn.rollback()
                if reservation is not None:
                    self.release_provider_reservation(reservation_id, "cancelled")
                return dict(existing)
            pending = conn.execute(
                "SELECT action_id FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND native_agent_name=? AND role=? "
                "AND status IN ('claimed','consumed') AND claude_agent_id IS NULL "
                "AND action_id != ? LIMIT 1",
                (
                    run_id, epoch_id, action["native_agent_name"], action["role"],
                    action_id,
                ),
            ).fetchone()
            if pending is not None:
                conn.rollback()
                if reservation is not None:
                    self.release_provider_reservation(reservation_id, "cancelled")
                raise WorkflowStateError(
                    "a native action with the same name and role is already awaiting "
                    f"lifecycle attachment: {pending[0]}"
                )
            action_kind = str(action.get("action_kind") or "native_agent")
            values = (
                action_id, run_id, epoch_id, action["phase_id"], action["role"],
                action["native_agent_name"], action["model_id"], action_kind, provider_id,
                claim_token, reservation_id if reservation is not None else None,
                intent_id if action_kind == "native_agent" else None,
                "claimed", _utcnow(), _utcnow(), expires_at,
            )
            if existing is None:
                conn.execute(
                    "INSERT INTO runnable_action_claims "
                    "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, action_kind, "
                    "provider_id, claim_token, reservation_id, intent_id, status, created_at, "
                    "claimed_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
            else:
                conn.execute(
                    "UPDATE runnable_action_claims SET run_id=?, epoch_id=?, phase_id=?, role=?, "
                    "native_agent_name=?, model_id=?, action_kind=?, provider_id=?, claim_token=?, "
                    "reservation_id=?, intent_id=?, status='claimed', claimed_at=?, "
                    "expires_at=?, consumed_at=NULL, claude_agent_id=NULL, execution_id=NULL WHERE action_id=?",
                    (run_id, epoch_id, action["phase_id"], action["role"],
                     action["native_agent_name"], action["model_id"], action_kind, provider_id,
                     claim_token, reservation_id if reservation is not None else None,
                     intent_id if action_kind == "native_agent" else None,
                     values[14], values[15], action_id),
                )
            policy_json = json.dumps(
                {"action_id": action_id, "claim_token": claim_token},
                separators=(",", ":"),
            )
            intent_exists = conn.execute(
                "SELECT 1 FROM spawn_intents WHERE intent_id=?", (intent_id,)
            ).fetchone()
            if action.get("action_kind") != "native_agent":
                pass
            elif intent_exists is None:
                conn.execute(
                    "INSERT INTO spawn_intents "
                    "(intent_id, run_id, epoch_id, phase_id, native_agent_name, role, model_id, "
                    "provider_id, status, created_at, policy_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                    "'planned', ?, ?)",
                    (intent_id, run_id, epoch_id, action["phase_id"], action["native_agent_name"],
                     action["role"], action["model_id"], provider_id, _utcnow(), policy_json),
                )
            elif action.get("action_kind") == "native_agent":
                conn.execute(
                    "UPDATE spawn_intents SET status='planned', spawned_at=NULL, completed_at=NULL, "
                    "claude_agent_id=NULL, policy_json=? WHERE intent_id=?",
                    (policy_json, intent_id),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?", (action_id,)
            ).fetchone()
            assert result is not None
            return {**action, **dict(result), "status": "claimed"}
        except Exception:
            conn.rollback()
            if reservation is not None:
                self.release_provider_reservation(reservation_id, "cancelled")
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
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            updated = conn.execute(
                "UPDATE runnable_action_claims SET status=?, consumed_at=COALESCE(consumed_at, ?) "
                "WHERE action_id=? AND run_id=? AND epoch_id=? AND role='controller' "
                "AND action_kind='controller_integration' AND status='consumed'",
                (status, now, action_id, run_id, epoch_id),
            )
            if updated.rowcount != 1:
                conn.rollback()
                return None
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (action_id,),
            ).fetchone()
            return dict(result) if result is not None else None
        finally:
            conn.close()

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
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT intent_id, reservation_id FROM runnable_action_claims "
                "WHERE run_id=? AND status IN ('claimed','consumed')",
                (run_id,),
            ).fetchall()
            reservation_ids = [str(row[1]) for row in rows if row[1]]
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
        return count

    def reconcile_lifecycle(
        self, run_id: str | None = None, *, max_age_seconds: int = 900,
    ) -> dict[str, int]:
        """Reconcile claims, spawn intents, and reservations left by crashes.

        Only records older than ``max_age_seconds`` are considered orphaned so
        a normal lifecycle race is not mistaken for a crash.  The operation is
        idempotent and returns counts suitable for health/status reporting.
        """
        cutoff = _utcnow_age(max_age_seconds)
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
                "WHERE execution_kind='sidecar_call' AND status IN ('started','running') "
                "AND started_at < ?" + run_clause + " AND execution_id NOT IN "
                "(SELECT execution_id FROM runnable_action_claims WHERE execution_id IS NOT NULL)",
                (cutoff, *run_params),
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
                "SELECT c.action_id, c.reservation_id, c.intent_id, r.provider_id, c.execution_id "
                "FROM runnable_action_claims AS c LEFT JOIN provider_reservations AS r "
                "ON r.reservation_id=c.reservation_id "
                "WHERE c.status IN ('claimed','consumed') AND "
                "COALESCE(c.consumed_at, c.claimed_at) < "
                "?" + (" AND c.run_id=?" if run_id is not None else ""),
                (cutoff, *run_params),
            ).fetchall()
            claim_ids = [str(row[0]) for row in claims]
            claim_reservations = [str(row[1]) for row in claims if row[1]]
            provider_ids.update(str(row[3]) for row in claims if row[3])
            reservation_ids.extend(claim_reservations)
            orphaned_execution_ids = [str(row[4]) for row in claims if row[4]]
            claim_count = 0
            for row in claims:
                action_id, reservation_id, intent_id, _provider_id, _execution_id = row
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
        executions_orphaned = 0
        for execution_id in orphaned_execution_ids:
            try:
                self.update_agent_execution(
                    execution_id=execution_id, status="timeout",
                    error="orphaned: no terminal report before crash-recovery cutoff",
                )
                executions_orphaned += 1
            except WorkflowStateError:
                # Already terminal (it finished right before reconciliation
                # ran) -- nothing to reconcile, not a failure.
                pass
        detached_orphaned = 0
        for execution_id in detached_execution_ids:
            try:
                self.update_agent_execution(
                    execution_id=execution_id, status="timeout",
                    error="orphaned: detached sidecar job had no terminal report before "
                          "crash-recovery cutoff",
                )
                detached_orphaned += 1
            except WorkflowStateError:
                pass
        result["executions_orphaned"] = executions_orphaned
        result["detached_executions_orphaned"] = detached_orphaned
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
