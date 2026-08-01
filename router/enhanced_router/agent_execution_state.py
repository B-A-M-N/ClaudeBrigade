"""Agent execution ledger persistence, split out of state.py.

Fourteenth increment of the incremental extraction out of ``RouteState``.
This is the authoritative record of every native/subagent/sidecar
execution: identity is checked against the active binding so a hook cannot
claim a different model or role than the router actually bound, and
per-execution tool-call/token counters are only ever incremented by the
router itself (``increment_execution_tool_calls`` explicitly documents that
caller-supplied counters are not trusted).

Uses ``self._select_agent_binding`` (from ``BindingRepository``, already
extracted) and ``self.append_execution_event`` (still defined directly on
``RouteState`` at the time of this extraction) -- both resolve through the
mixin's normal method resolution order, exactly like every earlier
increment's cross-section calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone

from enhanced_router.state_errors import WorkflowStateError

logger = logging.getLogger("claude-enhanced-router")

_EXECUTION_STATUSES = frozenset((
    "started", "running", "completed", "failed", "timeout", "cancelled",
))
_EXECUTION_TERMINAL_STATUSES = frozenset((
    "completed", "failed", "timeout", "cancelled",
))
_EXECUTION_TRANSITIONS = {
    "started": frozenset(("running", "completed", "failed", "timeout", "cancelled")),
    "running": frozenset(("completed", "failed", "timeout", "cancelled")),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentExecutionRepository:
    """Mixin providing agent execution ledger persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does), plus ``_select_agent_binding`` and
    ``append_execution_event`` (mixed into ``RouteState`` from
    ``BindingRepository`` and the not-yet-extracted shadow workspace
    section, respectively).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def create_agent_execution(
        self,
        execution_id: str,
        run_id: str,
        epoch_id: str,
        claude_agent_id: str,
        role: str,
        model_id: str,
        *,
        phase_id: str | None = None,
        binding_id: int | None = None,
        actor_kind: str = "subagent",
        execution_kind: str = "subagent",
        provider_id: str | None = None,
        endpoint_id: str | None = None,
        transport: str | None = None,
        configuration_hash: str | None = None,
        parent_execution_id: str | None = None,
        workspace_id: str | None = None,
        independence_key: str | None = None,
    ) -> dict:
        """Record the start of an authoritative execution.

        When a binding is supplied, all identity fields are checked against
        it. Hooks cannot create a record that claims a different model or
        role than the router actually bound.
        """
        conn = self._new_conn()
        try:
            if role in {"implementer", "repairer", "controller"}:
                if not workspace_id:
                    raise ValueError("mutating agent execution requires a shadow workspace")
                workspace_row = conn.execute(
                    "SELECT kind, status, owner_execution_id FROM workspaces WHERE workspace_id=?",
                    (workspace_id,),
                ).fetchone()
                if (
                    workspace_row is None
                    or workspace_row[0] != "shadow"
                    or workspace_row[1] != "active"
                    or workspace_row[2] != execution_id
                ):
                    raise ValueError(
                        "mutating agent execution must own an active shadow workspace"
                    )
            # A native lifecycle hook can run before the first model request
            # creates a binding.  When an active binding already exists, use
            # it as the authoritative identity instead of trusting caller
            # supplied model/provider fields.
            if binding_id is None:
                existing_binding = self._select_agent_binding(
                    conn, run_id, claude_agent_id,
                )
                if existing_binding is not None:
                    binding_id = int(existing_binding["binding_id"])

            if binding_id is not None:
                binding = conn.execute(
                    "SELECT run_id, epoch_id, claude_agent_id, role, model_id, released_at, "
                    "provider_id, endpoint_id, configuration_hash "
                    "FROM agent_bindings WHERE binding_id=?", (binding_id,)
                ).fetchone()
                if binding is None:
                    raise ValueError(f"unknown agent binding {binding_id}")
                if binding[0] != run_id or binding[1] != epoch_id or binding[2] != claude_agent_id:
                    raise ValueError("agent execution does not match its binding scope")
                if binding[3] != role or binding[4] != model_id:
                    raise ValueError("agent execution identity does not match its binding")
                if binding[5] is not None:
                    raise ValueError("agent binding is already released")
                if provider_id is not None and binding[6] is not None and provider_id != binding[6]:
                    raise ValueError("agent execution provider does not match its binding")
                if endpoint_id is not None and binding[7] is not None and endpoint_id != binding[7]:
                    raise ValueError("agent execution endpoint does not match its binding")
                if (
                    configuration_hash is not None
                    and binding[8] is not None
                    and configuration_hash != binding[8]
                ):
                    raise ValueError("agent execution configuration does not match its binding")

            if independence_key is None:
                independence_key = hashlib.sha256(
                    json.dumps({
                        "model_id": model_id,
                        "provider_id": provider_id,
                        "endpoint_id": endpoint_id,
                        "role": role,
                        "phase_id": phase_id,
                        "parent_execution_id": parent_execution_id,
                    }, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()

            if phase_id is not None:
                phase = conn.execute(
                    "SELECT status, required_actor, allowed_roles_json, max_fanout, "
                    "distinct_agent_from_json, provider_requirements_json, "
                    "max_duration_seconds, turn_budget, max_parallelism, max_attempts "
                    "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()
                if phase is None:
                    raise ValueError(f"unknown workflow phase {phase_id!r}")
                if phase[0] != "active":
                    raise ValueError(
                        f"workflow phase {phase_id!r} is {phase[0]}, not active"
                    )
                if phase[6] is not None:
                    started_at = conn.execute(
                        "SELECT started_at FROM workflow_phases "
                        "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                        (run_id, epoch_id, phase_id),
                    ).fetchone()[0]
                    if started_at:
                        elapsed = (
                            datetime.now(timezone.utc)
                            - datetime.fromisoformat(str(started_at)).astimezone(timezone.utc)
                        ).total_seconds()
                        if elapsed > int(phase[6]):
                            raise ValueError(
                                f"workflow phase {phase_id!r} exceeded its deadline"
                            )
                provider_requirements = json.loads(phase[5] or "[]")
                if provider_requirements and provider_id not in provider_requirements:
                    raise ValueError(
                        f"provider {provider_id!r} is not permitted by workflow phase {phase_id!r}"
                    )
                allowed_roles = json.loads(phase[2] or "[]")
                actor = str(phase[1] or "")
                if role not in allowed_roles and not (role == "controller" and actor == "controller"):
                    raise ValueError(
                        f"role {role!r} is not allowed in workflow phase {phase_id!r}"
                    )
                if actor and actor not in {role, "controller"}:
                    raise ValueError(
                        f"workflow phase {phase_id!r} requires actor {actor!r}"
                    )
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
                if int(active_count) >= int(phase[8] or phase[3] or 1):
                    raise ValueError(
                        f"workflow phase {phase_id!r} exceeded max fanout/parallelism"
                    )
                if int(attempt_count) >= int(phase[9] or phase[3] or 1):
                    raise ValueError(f"workflow phase {phase_id!r} exceeded max attempts")
                if phase[7] is not None:
                    used_turns = conn.execute(
                        "SELECT COALESCE(SUM(tool_call_count), 0) FROM agent_executions "
                        "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                        (run_id, epoch_id, phase_id),
                    ).fetchone()[0]
                    if int(used_turns or 0) >= int(phase[7]):
                        raise ValueError(
                            f"workflow phase {phase_id!r} exhausted turn budget {phase[7]}"
                        )
                distinct_refs = json.loads(phase[4] or "[]")
                if distinct_refs:
                    placeholders = ",".join("?" for _ in distinct_refs)
                    prior = conn.execute(
                        "SELECT claude_agent_id, independence_key FROM agent_executions "
                        f"WHERE run_id=? AND epoch_id=? AND phase_id IN ({placeholders})",
                        [run_id, epoch_id, *distinct_refs],
                    ).fetchall()
                    if any(
                        (independence_key and row[1] == independence_key)
                        or (not row[1] and row[0] == claude_agent_id)
                        for row in prior
                    ):
                        raise ValueError(
                            f"agent {claude_agent_id!r} violates distinct_agent_from "
                            f"for workflow phase {phase_id!r}"
                        )
            conn.execute(
                """INSERT INTO agent_executions
                   (execution_id, run_id, epoch_id, claude_agent_id, role, model_id,
                    phase_id, binding_id, status, actor_kind, execution_kind, provider_id,
                    endpoint_id, transport, configuration_hash, parent_execution_id,
                    workspace_id, independence_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'started', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (execution_id, run_id, epoch_id, claude_agent_id, role, model_id,
                 phase_id, binding_id, actor_kind, execution_kind, provider_id,
                 endpoint_id, transport, configuration_hash, parent_execution_id,
                 workspace_id, independence_key),
            )
            conn.commit()
            return self.get_agent_execution(execution_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def get_agent_execution(self, execution_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_agent_execution_scoped(
        self, run_id: str, epoch_id: str, execution_id: str,
    ) -> dict | None:
        """Return an execution only when both run and epoch own it."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE execution_id=? "
                "AND run_id=? AND epoch_id=?",
                (execution_id, run_id, epoch_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_agent_execution(
        self,
        execution_id: str,
        *,
        status: str | None = None,
        result_type: str | None = None,
        result_summary: str | None = None,
        output_hash: str | None = None,
        error: str | None = None,
        tool_call_count: int | None = None,
        total_tokens: int | None = None,
        result_json: str | None = None,
        request_count: int | None = None,
        retry_count: int | None = None,
        input_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        output_tokens: int | None = None,
        ttft_ms: float | None = None,
        wall_time_ms: float | None = None,
        error_class: str | None = None,
        schema_valid: bool | None = None,
        evidence_valid: bool | None = None,
        accepted_by_controller: bool | None = None,
        quality_score: float | None = None,
        verdict: str | None = None,
        confidence: float | None = None,
    ) -> dict | None:
        """Update an agent execution while enforcing lifecycle transitions.

        ``timed_out`` is accepted as the public spelling and stored as the
        legacy schema's ``timeout`` value.  Metrics are deliberately written
        to ``error_class``; the human-readable ``error`` field is preserved.
        """
        conn = self._new_conn()
        try:
            current = conn.execute(
                "SELECT status FROM agent_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if current is None:
                return None
            sets = ["updated_at = datetime('now')"]
            params: list = []
            if status is not None:
                normalized_status = "timeout" if status == "timed_out" else status
                if normalized_status not in _EXECUTION_STATUSES:
                    raise ValueError(f"invalid agent execution status: {status}")
                current_status = str(current[0])
                if current_status in _EXECUTION_TERMINAL_STATUSES:
                    if normalized_status != current_status:
                        raise WorkflowStateError(
                            f"terminal execution cannot transition from {current_status!r} "
                            f"to {normalized_status!r}"
                        )
                elif normalized_status not in _EXECUTION_TRANSITIONS.get(current_status, frozenset()):
                    raise WorkflowStateError(
                        f"illegal execution transition {current_status!r} -> {normalized_status!r}"
                    )
                sets.append("status = ?")
                params.append(normalized_status)
                if normalized_status in _EXECUTION_TERMINAL_STATUSES:
                    sets.append("completed_at = datetime('now')")
            if result_type is not None:
                sets.append("result_type = ?")
                params.append(result_type)
            if result_summary is not None:
                sets.append("result_summary = ?")
                params.append(result_summary)
            if output_hash is not None:
                sets.append("output_hash = ?")
                params.append(output_hash)
            if error is not None:
                sets.append("error = ?")
                params.append(error)
            if tool_call_count is not None:
                sets.append("tool_call_count = ?")
                params.append(tool_call_count)
            if total_tokens is not None:
                sets.append("total_tokens = ?")
                params.append(total_tokens)
            for column, value in (
                ("result_json", result_json), ("request_count", request_count),
                ("retry_count", retry_count), ("input_tokens", input_tokens),
                ("cache_read_tokens", cache_read_tokens), ("cache_write_tokens", cache_write_tokens),
                ("output_tokens", output_tokens), ("ttft_ms", ttft_ms),
                ("wall_time_ms", wall_time_ms), ("error_class", error_class),
                ("schema_valid", None if schema_valid is None else int(schema_valid)),
                ("evidence_valid", None if evidence_valid is None else int(evidence_valid)),
                ("accepted_by_controller", None if accepted_by_controller is None else int(accepted_by_controller)),
                ("quality_score", quality_score), ("verdict", verdict),
                ("confidence", confidence),
            ):
                if value is not None:
                    sets.append(f"{column} = ?")
                    params.append(value)
            params.append(execution_id)
            conn.execute(
                f"UPDATE agent_executions SET {', '.join(sets)} WHERE execution_id = ?",
                params,
            )
            conn.commit()
            return self.get_agent_execution(execution_id)
        finally:
            conn.close()

    def increment_execution_tool_calls(
        self, execution_id: str, *, run_id: str | None = None,
        epoch_id: str | None = None, delta: int = 1,
    ) -> dict | None:
        """Atomically increment tool calls; caller counters are not trusted."""
        if delta < 1:
            raise ValueError("tool-call increment must be positive")
        conn = self._new_conn()
        try:
            clauses = ["execution_id=?"]
            params: list[object] = [execution_id]
            if run_id is not None:
                clauses.append("run_id=?")
                params.append(run_id)
            if epoch_id is not None:
                clauses.append("epoch_id=?")
                params.append(epoch_id)
            conn.execute(
                "UPDATE agent_executions SET tool_call_count=COALESCE(tool_call_count, 0)+?, "
                "updated_at=datetime('now') WHERE " + " AND ".join(clauses),
                [delta, *params],
            )
            conn.commit()
            return self.get_agent_execution(execution_id)
        finally:
            conn.close()

    def increment_execution_requests(self, binding_id: int | None) -> dict | None:
        """Correlate an admitted model request with its execution ledger row."""
        if binding_id is None:
            return None
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE agent_executions SET request_count=COALESCE(request_count, 0)+1, "
                "updated_at=datetime('now') WHERE binding_id=?",
                (binding_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE binding_id=? "
                "ORDER BY started_at DESC LIMIT 1", (binding_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def record_execution_metrics_for_binding(
        self,
        binding_id: int | None,
        *,
        input_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        output_tokens: int = 0,
        ttft_ms: float | None = None,
        wall_time_ms: float | None = None,
    ) -> dict | None:
        """Merge non-sensitive request metrics into the bound execution."""
        if binding_id is None:
            return None
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE agent_executions SET input_tokens=COALESCE(input_tokens, 0)+?, "
                "cache_read_tokens=COALESCE(cache_read_tokens, 0)+?, "
                "cache_write_tokens=COALESCE(cache_write_tokens, 0)+?, "
                "output_tokens=COALESCE(output_tokens, 0)+?, "
                "ttft_ms=COALESCE(ttft_ms, ?), wall_time_ms=COALESCE(?, wall_time_ms), "
                "updated_at=datetime('now') WHERE binding_id=?",
                (input_tokens, cache_read_tokens, cache_write_tokens, output_tokens,
                 ttft_ms, wall_time_ms, binding_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM agent_executions WHERE binding_id=? "
                "ORDER BY started_at DESC LIMIT 1", (binding_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def record_execution_failure_for_binding(
        self, binding_id: int | None, *, error_class: str, error: str,
    ) -> dict | None:
        """Persist a transport failure against the active bound execution."""
        if binding_id is None:
            return None
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT execution_id, run_id, epoch_id FROM agent_executions "
                "WHERE binding_id=? AND status IN ('started','running') "
                "ORDER BY started_at DESC LIMIT 1", (binding_id,),
            ).fetchone()
            if row is None:
                return None
            now = _utcnow()
            conn.execute(
                "UPDATE agent_executions SET status='failed', completed_at=?, error=?, "
                "error_class=?, updated_at=? WHERE execution_id=? AND status IN ('started','running')",
                (now, error[:500], error_class, now, row[0]),
            )
            conn.commit()
            result = self.get_agent_execution(str(row[0]))
            try:
                self.append_execution_event(
                    str(row[1]), str(row[2]), str(row[0]), "failed",
                    {"error_class": error_class, "reason": error[:500]},
                )
            except Exception:
                logger.debug("failed to append transport failure event", exc_info=True)
            return result
        finally:
            conn.close()

    def get_agent_executions(
        self,
        run_id: str,
        epoch_id: str | None = None,
        phase_id: str | None = None,
        role: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """List agent executions with optional filters."""
        conn = self._new_conn()
        try:
            parts = ["SELECT * FROM agent_executions WHERE run_id = ?"]
            params: list = [run_id]
            if epoch_id:
                parts.append("AND epoch_id = ?")
                params.append(epoch_id)
            if phase_id:
                parts.append("AND phase_id = ?")
                params.append(phase_id)
            if role:
                parts.append("AND role = ?")
                params.append(role)
            if status:
                parts.append("AND status = ?")
                params.append(status)
            parts.append("ORDER BY started_at DESC")
            rows = conn.execute(" ".join(parts), params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
