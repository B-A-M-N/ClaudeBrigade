"""Native-agent spawn claim/attach lifecycle, split out of state.py.

Twentieth increment of the incremental extraction out of ``RouteState`` --
another slice of "Run lifecycle". Mirrors the sidecar execution module
(``sidecar_execution_state.py``) but for native Claude Code subagents: a
runnable action claim is consumed into a spawn intent, the spawned process
then attaches its real ``claude_agent_id`` back to that claim (checked
against the active binding so a hook cannot claim a different identity than
the router actually bound), and the assignment lifecycle tracks completion.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from enhanced_router.state_errors import WorkflowStateError


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class NativeSpawnAttachRepository(RepositoryMixin):
    """Mixin providing native-agent spawn claim/attach persistence.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does), plus ``_select_agent_binding`` (from
    ``BindingRepository``), ``release_provider_reservation``, and
    ``admit_provider_agents`` (both from ``ProviderReservationRepository``)
    -- all already extracted and resolving through the mixin's normal
    method resolution order.
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def consume_runnable_action_for_spawn(
        self,
        run_id: str,
        epoch_id: str,
        native_agent_name: str,
        *,
        action_id: str | None = None,
        claude_agent_id: str | None = None,
        spawn_call_id: str | None = None,
    ) -> dict | None:
        """Attach a native Agent tool call to a claim before child identity exists.

        PreToolUse only knows the parent tool-call identity.  It records that
        correlation key and leaves the claim pending until SubagentStart can
        attach the actual child.  The explicit ``claude_agent_id`` argument is
        retained for older in-process callers that already have the child ID.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            clauses = [
                "run_id=?", "epoch_id=?", "native_agent_name=?",
                "status='claimed'", "expires_at >= ?",
            ]
            params: list[object] = [run_id, epoch_id, native_agent_name, _utcnow()]
            if action_id:
                clauses.append("action_id=?")
                params.append(action_id)
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE " + " AND ".join(clauses)
                + " ORDER BY claimed_at, action_id LIMIT 1", params,
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            now = _utcnow()
            if claude_agent_id:
                conn.execute(
                    "UPDATE runnable_action_claims SET status='consumed', consumed_at=?, "
                    "claude_agent_id=?, spawn_call_id=COALESCE(?, spawn_call_id) "
                    "WHERE action_id=? AND status='claimed'",
                    (now, claude_agent_id, spawn_call_id, row["action_id"]),
                )
            else:
                conn.execute(
                    "UPDATE runnable_action_claims SET spawn_call_id=? "
                    "WHERE action_id=? AND status='claimed'",
                    (spawn_call_id, row["action_id"]),
                )
            if row["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status=?, spawned_at=?, claude_agent_id=? "
                    "WHERE intent_id=? AND status='planned'",
                    (
                        "spawned" if claude_agent_id else "planned",
                        now if claude_agent_id else None,
                        claude_agent_id,
                        row["intent_id"],
                    ),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (row["action_id"],),
            ).fetchone()
            if result is None:
                return None
            result_dict = dict(result)
            if result_dict.get("intent_id"):
                intent = conn.execute(
                    "SELECT worker_kind, worker_id, agent_definition_id, native_slot, "
                    "public_model_alias, capability_digest, priority_class, package_id "
                    "FROM spawn_intents WHERE intent_id=?",
                    (result_dict["intent_id"],),
                ).fetchone()
                if intent is not None:
                    result_dict.update(dict(intent))
            return result_dict
        finally:
            conn.close()

    def attach_spawned_agent(
        self,
        *,
        run_id: str,
        epoch_id: str,
        native_agent_name: str,
        role: str,
        model_id: str,
        endpoint_id: str | None = None,
        claude_agent_id: str,
        execution_id: str,
        workspace_id: str | None,
        phase_id: str | None,
        provider_id: str | None = None,
        route_digest: str | None = None,
        candidate_index: int | None = None,
        spawn_call_id: str | None = None,
        claim_token: str | None = None,
        binding_id: int | None = None,
        actor_kind: str = "subagent",
        execution_kind: str = "subagent",
    ) -> dict:
        """Atomically bind a claimed native spawn to its child execution.

        This is the lifecycle seam between PreToolUse and SubagentStart.  It
        refuses ambiguous marker matches, creates the authoritative execution,
        attaches the provider reservation, and only then marks the claim as
        consumed.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            clauses = [
                "run_id=?", "epoch_id=?", "native_agent_name=?", "role=?",
                "status='claimed'", "expires_at >= ?",
            ]
            params: list[object] = [
                run_id, epoch_id, native_agent_name, role, _utcnow(),
            ]
            if claim_token:
                clauses.append("claim_token=?")
                params.append(claim_token)
            if spawn_call_id:
                clauses.append("spawn_call_id=?")
                params.append(spawn_call_id)
            rows = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE "
                + " AND ".join(clauses)
                + " ORDER BY claimed_at, action_id",
                params,
            ).fetchall()
            if len(rows) != 1:
                conn.rollback()
                if not rows:
                    raise WorkflowStateError(
                        f"no live claimed native action matches {native_agent_name!r}"
                    )
                raise WorkflowStateError(
                    "native spawn correlation is ambiguous; refusing to choose a claim"
                )
            claim = rows[0]
            if claim["claude_agent_id"] not in (None, claude_agent_id):
                raise WorkflowStateError("native action is already attached to another child")
            if str(claim["model_id"]) != str(model_id):
                raise WorkflowStateError("native action model does not match its claim")
            claimed_endpoint = str(claim["endpoint_id"] or "auto")
            supplied_endpoint = str(endpoint_id or "auto")
            if claimed_endpoint != supplied_endpoint:
                raise WorkflowStateError("native action endpoint does not match its claim")
            if claim["provider_id"] != provider_id:
                raise WorkflowStateError("native action provider does not match its claim")
            if claim["route_digest"] and route_digest and claim["route_digest"] != route_digest:
                raise WorkflowStateError("native action route digest does not match its claim")
            if claim["candidate_index"] is not None and candidate_index is not None \
                    and int(claim["candidate_index"]) != int(candidate_index):
                raise WorkflowStateError("native action candidate does not match its claim")

            intent = None
            capability_snapshot: dict = {}
            if claim["intent_id"]:
                intent = conn.execute(
                    "SELECT * FROM spawn_intents WHERE intent_id=?",
                    (claim["intent_id"],),
                ).fetchone()
                if intent is not None:
                    try:
                        capability_snapshot = json.loads(
                            str(intent["capability_snapshot_json"] or "{}")
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        capability_snapshot = {}
            can_mutate = (
                bool(capability_snapshot.get("can_mutate"))
                if capability_snapshot
                else role in {"implementer", "repairer", "controller"}
            )

            existing_execution = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if existing_execution is not None:
                if (
                    existing_execution["run_id"] != run_id
                    or existing_execution["epoch_id"] != epoch_id
                    or existing_execution["claude_agent_id"] != claude_agent_id
                ):
                    raise WorkflowStateError("execution ID is already owned by another lifecycle")
                conn.rollback()
                return {
                    "claim": dict(claim),
                    "execution": dict(existing_execution),
                }

            if phase_id is not None:
                phase = conn.execute(
                    "SELECT status, allowed_roles_json, required_actor, max_fanout, "
                    "max_parallelism, max_attempts "
                    "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()
                if phase is None or phase[0] != "active":
                    raise WorkflowStateError("native action phase is not active")
                allowed_roles = json.loads(phase[1] or "[]")
                if role not in allowed_roles and not (role == "controller" and phase[2] == "controller"):
                    raise WorkflowStateError("native action role is not allowed in its phase")
                active_count = conn.execute(
                    "SELECT COUNT(*) FROM agent_executions "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=? "
                    "AND status NOT IN ('completed','failed','timeout','cancelled')",
                    (run_id, epoch_id, phase_id),
                ).fetchone()[0]
                attempt_count = conn.execute(
                    "SELECT COUNT(*) FROM agent_executions "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()[0]
                if int(active_count) >= int(phase[4] or phase[3] or 1):
                    raise WorkflowStateError("native action exceeds phase parallelism")
                if int(attempt_count) >= int(phase[5] or phase[3] or 1):
                    raise WorkflowStateError("native action exceeds phase attempt budget")

            if binding_id is None:
                existing_binding = self._select_agent_binding(
                    conn, run_id, claude_agent_id,
                )
                if existing_binding is not None:
                    binding_id = int(existing_binding["binding_id"])
            if binding_id is not None:
                binding = conn.execute(
                    "SELECT run_id, epoch_id, claude_agent_id, role, model_id, released_at, "
                    "provider_id, endpoint_id, route_digest, candidate_index "
                    "FROM agent_bindings WHERE binding_id=?",
                    (binding_id,),
                ).fetchone()
                if binding is None:
                    raise WorkflowStateError(f"unknown agent binding {binding_id}")
                if (
                    binding[0] != run_id or binding[1] != epoch_id
                    or binding[2] != claude_agent_id
                    or binding[3] != role or binding[4] != model_id
                ):
                    raise WorkflowStateError("native execution identity does not match its binding")
                if binding[5] is not None:
                    raise WorkflowStateError("native agent binding is already released")
                if provider_id is not None and binding[6] is not None and provider_id != binding[6]:
                    raise WorkflowStateError("native execution provider does not match its binding")
                if binding[7] is not None and str(binding[7] or "auto") != supplied_endpoint:
                    raise WorkflowStateError("native execution endpoint does not match its binding")
                if binding[8] and claim["route_digest"] and binding[8] != claim["route_digest"]:
                    raise WorkflowStateError("native execution route does not match its binding")

            independence_key = hashlib.sha256(
                json.dumps({
                    "model_id": model_id,
                    "provider_id": provider_id,
                    "role": role,
                    "phase_id": phase_id,
                    "parent_execution_id": None,
                    "workspace_id": workspace_id,
                }, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

            if workspace_id is not None:
                workspace = conn.execute(
                    "SELECT kind, status, owner_execution_id FROM workspaces WHERE workspace_id=?",
                    (workspace_id,),
                ).fetchone()
                if can_mutate:
                    if workspace is None or workspace[0] != "shadow" or workspace[1] != "active" \
                            or workspace[2] != execution_id:
                        raise WorkflowStateError(
                            "mutating native execution is not attached to its active shadow workspace"
                        )
            elif can_mutate:
                raise WorkflowStateError("mutating native execution requires a shadow workspace")

            now = _utcnow()
            conn.execute(
                "INSERT INTO agent_executions "
                "(execution_id, run_id, epoch_id, claude_agent_id, role, model_id, phase_id, "
                "binding_id, status, actor_kind, execution_kind, provider_id, endpoint_id, route_digest, "
                "candidate_index, workspace_id, independence_key, "
                "worker_kind, worker_id, capability_snapshot_json, tool_policy_digest, "
                "prompt_contract_digest, workspace_policy, background, package_id, agent_definition_id, "
                "native_slot, public_model_alias, capability_digest, priority_class) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'started', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    execution_id, run_id, epoch_id, claude_agent_id, role, model_id,
                    phase_id, binding_id, actor_kind, execution_kind, provider_id, endpoint_id,
                    route_digest or claim["route_digest"],
                    candidate_index if candidate_index is not None else claim["candidate_index"],
                    workspace_id, independence_key,
                    (intent["worker_kind"] if intent is not None else None),
                    (intent["worker_id"] if intent is not None else None),
                    json.dumps(capability_snapshot, separators=(",", ":")),
                    (intent["tool_policy_digest"] if intent is not None else None),
                    (intent["prompt_contract_digest"] if intent is not None else None),
                    (intent["workspace_policy"] if intent is not None else None),
                    (intent["background"] if intent is not None else None),
                    (intent["package_id"] if intent is not None else None),
                    (intent["agent_definition_id"] if intent is not None else None),
                    (intent["native_slot"] if intent is not None else None),
                    (intent["public_model_alias"] if intent is not None else None),
                    (intent["capability_digest"] if intent is not None else None),
                    (intent["priority_class"] if intent is not None else None),
                ),
            )
            claim_update = conn.execute(
                "UPDATE runnable_action_claims SET status='consumed', consumed_at=?, "
                "claude_agent_id=?, execution_id=? WHERE action_id=? AND status='claimed'",
                (now, claude_agent_id, execution_id, claim["action_id"]),
            )
            if claim_update.rowcount != 1:
                raise WorkflowStateError("native action claim changed during attachment")
            if claim["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status='spawned', spawned_at=?, claude_agent_id=? "
                    "WHERE intent_id=? AND status IN ('planned','spawned')",
                    (now, claude_agent_id, claim["intent_id"]),
                )
            if claim["reservation_id"]:
                conn.execute(
                    "UPDATE provider_reservations SET execution_id=? "
                    "WHERE reservation_id=? AND execution_id=?",
                    (execution_id, claim["reservation_id"], f"pending:{claim['action_id']}"),
                )
            conn.commit()
            final_claim = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (claim["action_id"],),
            ).fetchone()
            final_execution = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            assert final_claim is not None and final_execution is not None
            return {"claim": dict(final_claim), "execution": dict(final_execution)}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_pending_spawn_claim(
        self, run_id: str, epoch_id: str, native_agent_name: str, role: str,
    ) -> dict | None:
        """Return the only unbound claim for a native spawn, if one exists."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT c.*, i.worker_kind, i.worker_id, i.capability_snapshot_json, "
                "i.workspace_policy, i.background, i.package_id "
                "FROM runnable_action_claims AS c LEFT JOIN spawn_intents AS i "
                "ON i.intent_id=c.intent_id WHERE c.run_id=? AND c.epoch_id=? "
                "AND c.native_agent_name=? AND c.role=? AND c.status='claimed' "
                "AND expires_at >= ? ORDER BY claimed_at, action_id",
                (run_id, epoch_id, native_agent_name, role, _utcnow()),
            ).fetchall()
            if len(rows) > 1:
                raise WorkflowStateError(
                    "native spawn correlation is ambiguous; refusing to choose a claim"
                )
            return dict(rows[0]) if rows else None
        finally:
            conn.close()

    def get_unattached_spawn_claim_for_role(
        self, run_id: str, epoch_id: str, role: str,
    ) -> dict | None:
        """Return the pending native claim for a role, if lifecycle is racing."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE run_id=? AND epoch_id=? "
                "AND role=? AND status='claimed' AND claude_agent_id IS NULL "
                "AND expires_at >= ? ORDER BY claimed_at, action_id",
                (run_id, epoch_id, role, _utcnow()),
            ).fetchall()
            if len(rows) > 1:
                raise WorkflowStateError(
                    "multiple native actions are awaiting attachment for this role"
                )
            return dict(rows[0]) if rows else None
        finally:
            conn.close()

    def fail_spawn_claim(
        self, action_id: str, *, status: str = "failed", reason: str = "",
    ) -> dict | None:
        """Terminally close an unattached spawn claim and its intent."""
        if status not in {"failed", "timed_out", "cancelled", "orphaned"}:
            raise ValueError("invalid unattached spawn claim status")
        conn = self._new_conn()
        reservation_id: str | None = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=? "
                "AND status IN ('claimed','consumed')", (action_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            reservation_id = str(row["reservation_id"]) if row["reservation_id"] else None
            now = _utcnow()
            conn.execute(
                "UPDATE runnable_action_claims SET status=?, consumed_at=COALESCE(consumed_at, ?) "
                "WHERE action_id=? AND status IN ('claimed','consumed')",
                (status, now, action_id),
            )
            if row["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status='failed', completed_at=?, policy_json=? "
                    "WHERE intent_id=? AND status IN ('planned','spawned')",
                    (
                        now,
                        json.dumps({"failure": reason}, separators=(",", ":")),
                        row["intent_id"],
                    ),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?", (action_id,)
            ).fetchone()
            result_dict = dict(result) if result else None
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if reservation_id:
            reservation_state = {
                "timed_out": "expired",
                "orphaned": "expired",
                "failed": "cancelled",
            }.get(status, status)
            released = self.release_provider_reservation(reservation_id, reservation_state)
            if released:
                from enhanced_router.registry import get_registry
                provider = get_registry().providers.get(str(released.get("provider_id")))
                if provider:
                    self.admit_provider_agents(
                        str(released["provider_id"]), provider.limits.max_active_agents,
                    )
        return result_dict

    def get_spawn_assignment(
        self, run_id: str, epoch_id: str, claude_agent_id: str,
    ) -> dict | None:
        """Return the consumed action identity for a native agent, if any."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE run_id=? AND epoch_id=? "
                "AND claude_agent_id=? AND status='consumed' "
                "ORDER BY consumed_at DESC LIMIT 1",
                (run_id, epoch_id, claude_agent_id),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def finish_spawn_assignment(
        self,
        run_id: str,
        epoch_id: str,
        claude_agent_id: str,
        status: str,
    ) -> dict | None:
        """Close the action ledger for every native-agent terminal outcome."""
        if status not in {"completed", "failed", "timeout", "cancelled", "error"}:
            raise ValueError(f"invalid native-agent terminal status: {status}")
        claim_status = {
            "completed": "completed",
            "failed": "failed",
            "timeout": "timed_out",
            "cancelled": "cancelled",
            "error": "failed",
        }[status]
        intent_status = "completed" if status == "completed" else "failed"
        conn = self._new_conn()
        token_reservation_id: str | None = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE run_id=? AND epoch_id=? "
                "AND claude_agent_id=? AND status IN ('consumed','claimed') "
                "ORDER BY consumed_at DESC, claimed_at DESC LIMIT 1",
                (run_id, epoch_id, claude_agent_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            token_reservation_id = str(row["token_reservation_id"]) if row["token_reservation_id"] else None
            now = _utcnow()
            # Older embedded callers may still consume a claim with a child
            # ID before creating an authoritative execution.  Preserve their
            # response compatibility while the production lifecycle (which
            # always sets execution_id through attach_spawned_agent) receives
            # the explicit terminal status.
            terminal_status = "consumed" if (
                claim_status == "completed" and row["execution_id"] is None
            ) else claim_status
            conn.execute(
                "UPDATE runnable_action_claims SET status=?, consumed_at=COALESCE(consumed_at, ?) "
                "WHERE action_id=? AND status IN ('consumed','claimed')",
                (terminal_status, now, row["action_id"]),
            )
            if row["intent_id"]:
                conn.execute(
                    "UPDATE spawn_intents SET status=?, completed_at=? WHERE intent_id=? "
                    "AND status IN ('planned','spawned')",
                    (intent_status, now, row["intent_id"]),
                )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=?",
                (row["action_id"],),
            ).fetchone()
            if token_reservation_id:
                self.release_token_reservation(
                    token_reservation_id,
                    "consumed" if status == "completed" else "released",
                )
            return dict(result) if result is not None else None
        finally:
            conn.close()
