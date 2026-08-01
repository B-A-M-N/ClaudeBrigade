"""Sidecar execution ledger + generic execution-event persistence, split out
of state.py.

Nineteenth increment of the incremental extraction out of ``RouteState`` --
another slice of "Run lifecycle". Sidecars have no Claude Code child
process, so they use a synthetic ``sidecar:<execution_id>`` agent identity
while sharing the same ``agent_executions``/``execution_events`` ledger as
native workers. ``append_execution_event``/``get_execution_events`` are
generic execution-event CRUD used by both flows -- ``agent_execution_state.py``
(native-agent lifecycle, already extracted) also calls
``self.append_execution_event``, which keeps working unchanged via the
mixin's normal method resolution order.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from enhanced_router.state_errors import WorkflowStateError


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SidecarExecutionRepository:
    """Mixin providing sidecar execution + execution-event persistence.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def start_sidecar_execution(
        self,
        *,
        run_id: str,
        epoch_id: str,
        action_id: str,
        claim_token: str,
        execution_id: str,
        packet: dict,
    ) -> dict:
        """Atomically consume a sidecar claim and create its execution.

        Sidecars have no Claude Code child process.  They therefore use a
        synthetic agent identity, while retaining the same claim-to-execution
        correlation used by native workers.
        """
        encoded_packet = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        if len(encoded_packet.encode("utf-8")) > 64_000:
            raise WorkflowStateError("sidecar packet exceeds the 64 KiB bound")
        sidecar_agent_id = f"sidecar:{execution_id}"
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            claim = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE action_id=? AND run_id=? "
                "AND epoch_id=? AND action_kind='sidecar_call' AND claim_token=? "
                "AND status='claimed' AND expires_at >= ?",
                (action_id, run_id, epoch_id, claim_token, _utcnow()),
            ).fetchone()
            if claim is None:
                raise WorkflowStateError("sidecar claim is missing, expired, or already consumed")
            phase = conn.execute(
                "SELECT status, allowed_roles_json, max_fanout, provider_requirements_json, "
                "max_parallelism, max_attempts "
                "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, claim["phase_id"]),
            ).fetchone()
            if phase is None or phase[0] != "active":
                raise WorkflowStateError("sidecar action phase is not active")
            allowed_roles = json.loads(phase[1] or "[]")
            if claim["role"] not in allowed_roles:
                raise WorkflowStateError("sidecar role is not allowed in its phase")
            if claim["provider_id"] not in json.loads(phase[3] or "[]") and json.loads(phase[3] or "[]"):
                raise WorkflowStateError("sidecar provider is not permitted by its phase")
            active_count = conn.execute(
                "SELECT COUNT(*) FROM agent_executions WHERE run_id=? AND epoch_id=? "
                "AND phase_id=? AND status NOT IN ('completed','failed','timeout','cancelled')",
                (run_id, epoch_id, claim["phase_id"]),
            ).fetchone()[0]
            attempt_count = conn.execute(
                "SELECT COUNT(*) FROM agent_executions WHERE run_id=? AND epoch_id=? "
                "AND phase_id=?",
                (run_id, epoch_id, claim["phase_id"]),
            ).fetchone()[0]
            if int(active_count) >= int(phase[4] or phase[2] or 1):
                raise WorkflowStateError("sidecar action exceeds phase parallelism")
            if int(attempt_count) >= int(phase[5] or phase[2] or 1):
                raise WorkflowStateError("sidecar action exceeds phase attempt budget")
            existing = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if existing is not None:
                if existing["run_id"] != run_id or existing["epoch_id"] != epoch_id:
                    raise WorkflowStateError("execution ID is owned by another run")
                conn.rollback()
                return dict(existing)
            now = _utcnow()
            conn.execute(
                "INSERT INTO agent_executions "
                "(execution_id, run_id, epoch_id, claude_agent_id, role, model_id, phase_id, "
                "status, actor_kind, execution_kind, provider_id, independence_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'started', 'sidecar', 'sidecar_call', ?, ?)",
                (
                    execution_id, run_id, epoch_id, sidecar_agent_id, claim["role"],
                    claim["model_id"], claim["phase_id"], claim["provider_id"],
                    hashlib.sha256(
                        f"sidecar:{claim['model_id']}:{claim['role']}:{claim['phase_id']}".encode()
                    ).hexdigest(),
                ),
            )
            updated = conn.execute(
                "UPDATE runnable_action_claims SET status='consumed', consumed_at=?, "
                "claude_agent_id=?, execution_id=? WHERE action_id=? AND status='claimed'",
                (now, sidecar_agent_id, execution_id, action_id),
            )
            if updated.rowcount != 1:
                raise WorkflowStateError("sidecar claim changed during execution start")
            conn.execute(
                "INSERT INTO execution_events "
                "(execution_id, run_id, epoch_id, seq, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, 1, 'started', ?, ?)",
                (execution_id, run_id, epoch_id, encoded_packet, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def start_detached_sidecar_execution(
        self,
        *,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        phase_id: str,
        role: str,
        model_id: str,
        provider_id: str | None,
        packet: dict,
        parent_execution_id: str | None = None,
        retry_count: int = 0,
    ) -> dict:
        """Create a persisted router-owned advisory sidecar execution.

        Fastpath jobs are created by the router itself rather than by a
        workflow claim.  They still use the same execution ledger and event
        stream as claimed sidecars, but are scoped to an existing run/epoch
        and cannot become native-agent or mutation work.
        """
        encoded_packet = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        if len(encoded_packet.encode("utf-8")) > 64_000:
            raise WorkflowStateError("detached sidecar packet exceeds the 64 KiB bound")
        sidecar_agent_id = f"sidecar:{execution_id}"
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            epoch = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if owned is None or epoch is None:
                raise WorkflowStateError("detached sidecar run or epoch is not active")
            if parent_execution_id is not None:
                parent = conn.execute(
                    "SELECT run_id, epoch_id FROM agent_executions WHERE execution_id=?",
                    (parent_execution_id,),
                ).fetchone()
                if parent is None or parent[0] != run_id or parent[1] != epoch_id:
                    raise WorkflowStateError("detached sidecar retry parent is out of scope")
            existing = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if existing is not None:
                if existing["run_id"] != run_id or existing["epoch_id"] != epoch_id:
                    raise WorkflowStateError("execution ID is owned by another run")
                conn.rollback()
                return dict(existing)
            now = _utcnow()
            independence_key = hashlib.sha256(
                json.dumps({
                    "model_id": model_id,
                    "provider_id": provider_id,
                    "role": role,
                    "phase_id": phase_id,
                    "execution_id": execution_id,
                }, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            conn.execute(
                "INSERT INTO agent_executions "
                "(execution_id, run_id, epoch_id, claude_agent_id, role, model_id, phase_id, "
                "status, actor_kind, execution_kind, provider_id, retry_count, "
                "parent_execution_id, independence_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'started', 'sidecar', 'sidecar_call', ?, ?, ?, ?)",
                (execution_id, run_id, epoch_id, sidecar_agent_id, role, model_id,
                 phase_id, provider_id, retry_count, parent_execution_id, independence_key),
            )
            conn.execute(
                "INSERT INTO execution_events "
                "(execution_id, run_id, epoch_id, seq, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, 1, 'started', ?, ?)",
                (execution_id, run_id, epoch_id, encoded_packet, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append_execution_event(
        self,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        event_type: str,
        payload: dict | None = None,
    ) -> dict:
        """Append an ordered, scoped execution event."""
        payload_json = json.dumps(payload or {}, separators=(",", ":"), ensure_ascii=False)
        if len(payload_json.encode("utf-8")) > 64_000:
            raise ValueError("execution event payload exceeds the 64 KiB bound")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                "SELECT 1 FROM agent_executions WHERE execution_id=? AND run_id=? AND epoch_id=?",
                (execution_id, run_id, epoch_id),
            ).fetchone()
            if owned is None:
                raise WorkflowStateError("execution is not owned by the requested run and epoch")
            seq = int(conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM execution_events WHERE execution_id=?",
                (execution_id,),
            ).fetchone()[0])
            now = _utcnow()
            conn.execute(
                "INSERT INTO execution_events "
                "(execution_id, run_id, epoch_id, seq, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (execution_id, run_id, epoch_id, seq, event_type, payload_json, now),
            )
            conn.commit()
            return {
                "execution_id": execution_id, "run_id": run_id, "epoch_id": epoch_id,
                "seq": seq, "event_type": event_type, "payload": payload or {},
                "created_at": now,
            }
        finally:
            conn.close()

    def prepare_sidecar_retry(self, execution_id: str) -> dict:
        """Create one bounded retry claim from a failed sidecar execution."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            execution = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=?", (execution_id,)
            ).fetchone()
            if execution is None:
                raise WorkflowStateError("sidecar execution was not found")
            if execution["execution_kind"] != "sidecar_call":
                raise WorkflowStateError("only sidecar executions can be retried")
            if execution["status"] not in {"failed", "timeout", "cancelled"}:
                raise WorkflowStateError("sidecar execution is not in a retryable terminal state")
            claim = conn.execute(
                "SELECT * FROM runnable_action_claims WHERE execution_id=? "
                "AND action_kind='sidecar_call' ORDER BY claimed_at DESC LIMIT 1",
                (execution_id,),
            ).fetchone()
            if claim is None:
                raise WorkflowStateError("sidecar execution has no originating claim")
            prior_retries = conn.execute(
                "SELECT COUNT(*) FROM runnable_action_claims WHERE action_id LIKE ?",
                (f"{claim['action_id']}:retry:%",),
            ).fetchone()[0]
            if int(prior_retries) >= 2:
                raise WorkflowStateError("sidecar retry budget exhausted")
            event = conn.execute(
                "SELECT payload_json FROM execution_events WHERE execution_id=? AND seq=1",
                (execution_id,),
            ).fetchone()
            if event is None:
                raise WorkflowStateError("sidecar input packet is unavailable for retry")
            packet = json.loads(str(event[0]))
            retry_action_id = f"{claim['action_id']}:retry:{secrets.token_hex(4)}"
            retry_token = secrets.token_urlsafe(24)
            now = _utcnow()
            expires = (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat()
            conn.execute(
                "INSERT INTO runnable_action_claims "
                "(action_id, run_id, epoch_id, phase_id, role, native_agent_name, model_id, "
                "action_kind, provider_id, claim_token, status, created_at, claimed_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'sidecar_call', ?, ?, 'claimed', ?, ?, ?)",
                (retry_action_id, execution["run_id"], execution["epoch_id"], execution["phase_id"],
                 execution["role"], claim["native_agent_name"], execution["model_id"],
                 execution["provider_id"], retry_token, now, now, expires),
            )
            conn.commit()
            return {
                "run_id": execution["run_id"], "epoch_id": execution["epoch_id"],
                "action_id": retry_action_id, "claim_token": retry_token, "packet": packet,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_execution_events(
        self,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        *,
        after_seq: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        """Read execution events only within the requested run and epoch."""
        limit = max(1, min(int(limit), 500))
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM execution_events WHERE execution_id=? AND run_id=? "
                "AND epoch_id=? AND seq>? ORDER BY seq LIMIT ?",
                (execution_id, run_id, epoch_id, max(0, int(after_seq)), limit),
            ).fetchall()
            result: list[dict] = []
            for row in rows:
                item = dict(row)
                try:
                    item["payload"] = json.loads(item.pop("payload_json"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    item["payload"] = {"malformed": True}
                result.append(item)
            return result
        finally:
            conn.close()
