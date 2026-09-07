"""Durable checkpoint ledger for automatic coprocessor feedback."""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class FeedbackRepository(RepositoryMixin):
    """Persist dedupe, cooldown, and in-flight feedback checkpoint state."""

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover
        raise NotImplementedError

    def claim_feedback_checkpoint(
        self,
        *,
        feedback_id: str,
        run_id: str,
        epoch_id: str,
        execution_key: str,
        claude_agent_id: str | None,
        action_id: str | None,
        checkpoint: str,
        coprocessor_id: str,
        provider_id: str | None,
        evidence_digest: str,
        packet_digest: str,
        cooldown_seconds: float,
        max_calls: int,
        max_parallelism: int,
        parent_execution_id: str | None = None,
        prompt_version: str | None = None,
        schema_version: str | None = None,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        """Atomically decide whether this checkpoint may start a call."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            # A router restart must not leave a stale running row consuming
            # the only feedback lane forever.
            conn.execute(
                "UPDATE feedback_checkpoints SET status='failed', delivery_status='expired', "
                "error=COALESCE(error, 'feedback execution lease expired'), "
                "orphaned_at=COALESCE(orphaned_at, ?), completed_at=COALESCE(completed_at, ?) "
                "WHERE run_id=? AND epoch_id=? AND status='running' "
                "AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
                (now, now, run_id, epoch_id, now),
            )
            latest = conn.execute(
                "SELECT * FROM feedback_checkpoints WHERE run_id=? AND epoch_id=? "
                "AND execution_key=? AND checkpoint=? AND coprocessor_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (run_id, epoch_id, execution_key, checkpoint, coprocessor_id),
            ).fetchone()
            if latest is not None:
                if str(latest["evidence_digest"]) == evidence_digest:
                    status = str(latest["status"])
                    conn.commit()
                    return {
                        "decision": "duplicate" if status == "completed" else "in_flight",
                        "feedback_id": str(latest["feedback_id"]),
                        "status": status,
                        "feedback": _decode_json(latest["result_json"]),
                        "feedback_text": latest["feedback_text"],
                    }
                last_time = _parse_time(latest["created_at"])
                if last_time is not None and (
                    datetime.now(timezone.utc) - last_time
                ) < timedelta(seconds=max(0.0, cooldown_seconds)):
                    conn.commit()
                    return {
                        "decision": "cooldown",
                        "feedback_id": str(latest["feedback_id"]),
                        "status": str(latest["status"]),
                    }

            calls = int(conn.execute(
                "SELECT COUNT(*) FROM feedback_checkpoints WHERE run_id=? AND epoch_id=? "
                "AND execution_key=? AND checkpoint=? AND coprocessor_id=?",
                (run_id, epoch_id, execution_key, checkpoint, coprocessor_id),
            ).fetchone()[0])
            if calls >= max(1, int(max_calls)):
                conn.commit()
                return {"decision": "budget", "status": "deferred"}

            in_flight = int(conn.execute(
                "SELECT COUNT(*) FROM feedback_checkpoints WHERE run_id=? AND epoch_id=? "
                "AND coprocessor_id=? AND status='running' "
                "AND (lease_expires_at IS NULL OR lease_expires_at >= ?)",
                (run_id, epoch_id, coprocessor_id, now),
            ).fetchone()[0])
            if in_flight >= max(1, int(max_parallelism)):
                conn.commit()
                return {"decision": "parallelism", "status": "deferred"}

            lease_expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=max(1, int(lease_seconds)))
            ).isoformat()
            conn.execute(
                "INSERT INTO feedback_checkpoints "
                "(feedback_id, run_id, epoch_id, execution_key, claude_agent_id, action_id, "
                "checkpoint, coprocessor_id, provider_id, evidence_digest, packet_digest, "
                "parent_execution_id, prompt_version, schema_version, status, delivery_status, "
                "attempt, created_at, started_at, heartbeat_at, lease_expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 'pending', ?, ?, ?, ?, ?)",
                (
                    feedback_id, run_id, epoch_id, execution_key, claude_agent_id,
                    action_id, checkpoint, coprocessor_id, provider_id, evidence_digest,
                    packet_digest, parent_execution_id, prompt_version, schema_version,
                    calls + 1, now, now, now,
                    lease_expires_at,
                ),
            )
            conn.commit()
            return {"decision": "claimed", "feedback_id": feedback_id, "status": "running"}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def link_feedback_execution(
        self, feedback_id: str, coprocessor_execution_id: str,
    ) -> dict[str, Any] | None:
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE feedback_checkpoints SET coprocessor_execution_id=? "
                "WHERE feedback_id=?",
                (coprocessor_execution_id, feedback_id),
            )
            conn.commit()
            return self.get_feedback(feedback_id)
        finally:
            conn.close()

    def complete_feedback(
        self,
        feedback_id: str,
        *,
        status: str,
        result: Any = None,
        feedback_text: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if status not in {"completed", "failed", "deferred"}:
            raise ValueError(f"invalid feedback status: {status}")
        result_json = (
            json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            if result is not None else None
        )
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE feedback_checkpoints SET status=?, delivery_status=?, result_json=?, "
                "feedback_text=?, error=?, completed_at=?, finalized_at=?, heartbeat_at=? "
                "WHERE feedback_id=?",
                (
                    status,
                    "ready" if status == "completed" else "expired",
                    result_json,
                    feedback_text,
                    error,
                    _utcnow(),
                    _utcnow(),
                    _utcnow(),
                    feedback_id,
                ),
            )
            conn.commit()
            return self.get_feedback(feedback_id)
        finally:
            conn.close()

    def get_ready_feedback(
        self,
        run_id: str,
        epoch_id: str,
        *,
        execution_key: str | None = None,
        parent_execution_id: str | None = None,
        current_workspace_digest: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the oldest relevant result that has not been delivered."""
        conn = self._new_conn()
        try:
            clauses = [
                "run_id=?", "epoch_id=?", "status='completed'",
                "delivery_status='ready'",
            ]
            params: list[Any] = [run_id, epoch_id]
            if execution_key is not None:
                clauses.append("execution_key=?")
                params.append(execution_key)
            if parent_execution_id is not None:
                clauses.append("parent_execution_id=?")
                params.append(parent_execution_id)
            rows = conn.execute(
                "SELECT * FROM feedback_checkpoints WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at ASC LIMIT 32",
                params,
            ).fetchall()
            for row in rows:
                # Feedback generated against an older workspace generation is
                # not safe to inject into a later turn.  The original packet
                # is persisted as the first execution event, so this check
                # does not depend on an in-memory task registry.
                if current_workspace_digest and row["coprocessor_execution_id"]:
                    event = conn.execute(
                        "SELECT payload_json FROM execution_events "
                        "WHERE execution_id=? ORDER BY seq LIMIT 1",
                        (row["coprocessor_execution_id"],),
                    ).fetchone()
                    original_digest = None
                    if event is not None:
                        packet = _decode_json(event[0])
                        if isinstance(packet, dict):
                            original_digest = packet.get("workspace_digest")
                    if original_digest and str(original_digest) != str(current_workspace_digest):
                        now = _utcnow()
                        conn.execute(
                            "UPDATE feedback_checkpoints SET delivery_status='expired', "
                            "disposition='ignored_stale', disposition_reason=?, "
                            "finalized_at=?, orphaned_at=COALESCE(orphaned_at, ?) "
                            "WHERE feedback_id=? AND delivery_status='ready'",
                            (
                                "workspace generation changed before delivery",
                                now, now, row["feedback_id"],
                            ),
                        )
                        conn.commit()
                        continue
                result = dict(row)
                result["result"] = _decode_json(result.get("result_json"))
                return result
            return None
        finally:
            conn.close()

    def mark_feedback_delivered(
        self, feedback_id: str, *, consumer_turn_id: str | None = None,
    ) -> dict[str, Any] | None:
        conn = self._new_conn()
        try:
            now = _utcnow()
            conn.execute(
                "UPDATE feedback_checkpoints SET delivery_status='delivered', "
                "delivered_at=?, consumer_turn_id=?, delivery_count=COALESCE(delivery_count, 0)+1 "
                "WHERE feedback_id=? AND status='completed' AND delivery_status='ready'",
                (now, consumer_turn_id, feedback_id),
            )
            conn.commit()
            return self.get_feedback(feedback_id)
        finally:
            conn.close()

    def heartbeat_feedback(self, feedback_id: str, *, lease_seconds: int = 300) -> bool:
        conn = self._new_conn()
        try:
            now = datetime.now(timezone.utc)
            lease = (now + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()
            updated = conn.execute(
                "UPDATE feedback_checkpoints SET heartbeat_at=?, lease_expires_at=? "
                "WHERE feedback_id=? AND status='running'",
                (now.isoformat(), lease, feedback_id),
            )
            conn.commit()
            return updated.rowcount == 1
        finally:
            conn.close()

    def reconcile_feedback(self) -> int:
        """Reconcile feedback rows after a router restart or task crash."""
        conn = self._new_conn()
        changed = 0
        try:
            now = _utcnow()
            rows = conn.execute(
                "SELECT feedback_id, coprocessor_execution_id, lease_expires_at "
                "FROM feedback_checkpoints WHERE status='running'"
            ).fetchall()
            for row in rows:
                execution = None
                if row[1]:
                    execution = conn.execute(
                        "SELECT status, result_json FROM agent_executions WHERE execution_id=?",
                        (row[1],),
                    ).fetchone()
                stale = bool(row[2] and str(row[2]) < now)
                if execution is None and stale:
                    conn.execute(
                        "UPDATE feedback_checkpoints SET status='failed', delivery_status='expired', "
                        "error=COALESCE(error, 'feedback execution orphaned'), orphaned_at=?, "
                        "completed_at=COALESCE(completed_at, ?), finalized_at=COALESCE(finalized_at, ?) "
                        "WHERE feedback_id=?",
                        (now, now, now, row[0]),
                    )
                    changed += 1
                elif execution is not None and str(execution[0]) == "completed":
                    if execution[1] is not None:
                        conn.execute(
                            "UPDATE feedback_checkpoints SET status='completed', delivery_status='ready', "
                            "result_json=COALESCE(result_json, ?), feedback_text=COALESCE(feedback_text, ?), "
                            "completed_at=COALESCE(completed_at, ?), finalized_at=COALESCE(finalized_at, ?) "
                            "WHERE feedback_id=? AND status='running'",
                            (str(execution[1]), str(execution[1])[:8_000], now, now, row[0]),
                        )
                        changed += 1
                elif execution is not None and str(execution[0]) in {
                    "failed", "timeout", "cancelled",
                }:
                    conn.execute(
                        "UPDATE feedback_checkpoints SET status='failed', delivery_status='expired', "
                        "error=COALESCE(error, 'linked coprocessor execution failed'), "
                        "completed_at=COALESCE(completed_at, ?), finalized_at=COALESCE(finalized_at, ?) "
                        "WHERE feedback_id=? AND status='running'",
                        (now, now, row[0]),
                    )
                    changed += 1
            conn.commit()
            return changed
        finally:
            conn.close()

    def adjudicate_feedback(
        self,
        feedback_id: str,
        *,
        disposition: str,
        reason: str = "",
        adopted_finding_ids: list[str] | None = None,
        rejected_finding_ids: list[str] | None = None,
        resulting_action_ids: list[str] | None = None,
        resulting_changeset_ids: list[str] | None = None,
        later_validation: Any = None,
        harm_class: str | None = None,
        task_class: str | None = None,
        quality_score: float | None = None,
        quality_dimensions: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        allowed = {
            "adopted", "partially_adopted", "rejected_incorrect",
            "rejected_irrelevant", "superseded", "ignored_stale",
            "ignored_timeout", "not_evaluated",
        }
        if disposition not in allowed:
            raise ValueError(f"invalid coprocessor feedback disposition: {disposition}")
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feedback_checkpoints WHERE feedback_id=?", (feedback_id,)
            ).fetchone()
            if row is None:
                return None
            now = _utcnow()

            # The checkpoint's configured provider is only the requested
            # primary.  Attribute the outcome to the concrete route that
            # actually returned the result, including a fallback provider.
            execution_row = None
            successful_attempt = None
            if row["coprocessor_execution_id"]:
                execution_row = conn.execute(
                    "SELECT model_id, provider_id FROM agent_executions "
                    "WHERE execution_id=?",
                    (row["coprocessor_execution_id"],),
                ).fetchone()
                successful_attempt = conn.execute(
                    "SELECT model_id, provider_id, endpoint_id, usage_json, "
                    "started_at, finished_at FROM route_attempts "
                    "WHERE execution_id=? AND status='succeeded' "
                    "ORDER BY candidate_index DESC LIMIT 1",
                    (row["coprocessor_execution_id"],),
                ).fetchone()
            actual_model_id = (
                successful_attempt[0] if successful_attempt is not None
                else execution_row[0] if execution_row is not None else None
            )
            actual_provider_id = (
                successful_attempt[1] if successful_attempt is not None
                else execution_row[1] if execution_row is not None else row["provider_id"]
            )
            latency_ms = None
            usage: dict[str, Any] = {}
            if successful_attempt is not None:
                try:
                    started = datetime.fromisoformat(str(successful_attempt[4]))
                    finished = datetime.fromisoformat(str(successful_attempt[5]))
                    latency_ms = max(0.0, (finished - started).total_seconds() * 1000.0)
                except (TypeError, ValueError, AttributeError):
                    latency_ms = None
                try:
                    decoded_usage = json.loads(str(successful_attempt[3] or "{}"))
                    if isinstance(decoded_usage, dict):
                        usage = decoded_usage
                except (TypeError, ValueError, json.JSONDecodeError):
                    usage = {}
            def encoded(value: list[str] | None) -> str:
                return json.dumps(value or [], separators=(",", ":"))
            conn.execute(
                "UPDATE feedback_checkpoints SET disposition=?, disposition_reason=?, "
                "adopted_finding_ids_json=?, rejected_finding_ids_json=?, resulting_action_ids_json=?, "
                "resulting_changeset_ids_json=?, later_validation_json=?, harm_class=?, finalized_at=? "
                "WHERE feedback_id=?",
                (
                    disposition, reason[:2000], encoded(adopted_finding_ids),
                    encoded(rejected_finding_ids), encoded(resulting_action_ids),
                    encoded(resulting_changeset_ids),
                    json.dumps(later_validation, separators=(",", ":"))
                    if later_validation is not None else None,
                    harm_class, now, feedback_id,
                ),
            )
            outcome_id = f"outcome_{feedback_id}_{now.replace(':', '').replace('+', '')}"
            conn.execute(
                "INSERT INTO coprocessor_outcomes "
                "(outcome_id, feedback_id, run_id, epoch_id, coprocessor_id, model_id, provider_id, "
                "prompt_version, schema_version, task_class, disposition, disposition_reason, adopted_finding_ids_json, "
                "rejected_finding_ids_json, resulting_action_ids_json, resulting_changeset_ids_json, "
                "later_validation_json, harm_class, latency_ms, input_tokens, output_tokens, estimated_cost, "
                "quality_score, quality_dimensions_json, finalized_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    outcome_id, feedback_id, row["run_id"], row["epoch_id"],
                    row["coprocessor_id"], actual_model_id,
                    actual_provider_id, row["prompt_version"], row["schema_version"],
                    task_class, disposition, reason[:2000],
                    encoded(adopted_finding_ids), encoded(rejected_finding_ids),
                    encoded(resulting_action_ids), encoded(resulting_changeset_ids),
                    json.dumps(later_validation, separators=(",", ":"))
                    if later_validation is not None else None,
                    harm_class,
                    latency_ms,
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("estimated_cost"),
                    quality_score,
                    json.dumps(quality_dimensions, separators=(",", ":"))
                    if quality_dimensions is not None else None,
                    now, now,
                ),
            )
            conn.commit()
            return self.get_feedback(feedback_id)
        finally:
            conn.close()

    def get_coprocessor_outcome_metrics(
        self,
        coprocessor_id: str,
        *,
        task_class: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Summarize evaluated outcomes for promotion/degradation decisions."""
        conn = self._new_conn()
        try:
            clauses = ["coprocessor_id=?"]
            params: list[Any] = [coprocessor_id]
            if task_class:
                clauses.append("task_class=?")
                params.append(task_class)
            if run_id:
                clauses.append("run_id=?")
                params.append(run_id)
            row = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN disposition NOT IN ('not_evaluated', 'ignored_timeout', 'ignored_stale') THEN 1 ELSE 0 END) AS evaluated, "
                "SUM(CASE WHEN disposition IN ('adopted', 'partially_adopted') THEN 1 ELSE 0 END) AS adopted, "
                "SUM(CASE WHEN harm_class IS NOT NULL AND harm_class NOT IN ('', 'none') THEN 1 ELSE 0 END) AS harmed, "
                "AVG(latency_ms) AS mean_latency_ms, SUM(COALESCE(estimated_cost, 0)) AS estimated_cost, "
                "SUM(COALESCE(input_tokens, 0)) AS input_tokens, SUM(COALESCE(output_tokens, 0)) AS output_tokens "
                "FROM coprocessor_outcomes WHERE " + " AND ".join(clauses),
                params,
            ).fetchone()
            total = int(row["total"] or 0) if row is not None else 0
            evaluated = int(row["evaluated"] or 0) if row is not None else 0
            adopted = int(row["adopted"] or 0) if row is not None else 0
            harmed = int(row["harmed"] or 0) if row is not None else 0
            return {
                "coprocessor_id": coprocessor_id,
                "task_class": task_class,
                "total_outcomes": total,
                "evaluated_outcomes": evaluated,
                "adopted_outcomes": adopted,
                "harmed_outcomes": harmed,
                "adoption_rate": adopted / evaluated if evaluated else None,
                "harm_rate": harmed / evaluated if evaluated else None,
                "mean_latency_ms": float(row["mean_latency_ms"]) if row is not None and row["mean_latency_ms"] is not None else None,
                "estimated_cost": float(row["estimated_cost"] or 0) if row is not None else 0.0,
                "input_tokens": int(row["input_tokens"] or 0) if row is not None else 0,
                "output_tokens": int(row["output_tokens"] or 0) if row is not None else 0,
            }
        finally:
            conn.close()

    def get_feedback(self, feedback_id: str) -> dict[str, Any] | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feedback_checkpoints WHERE feedback_id=?",
                (feedback_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["result"] = _decode_json(result.get("result_json"))
            for field in (
                "adopted_finding_ids_json", "rejected_finding_ids_json",
                "resulting_action_ids_json", "resulting_changeset_ids_json",
                "later_validation_json",
            ):
                result[field.removesuffix("_json")] = _decode_json(result.get(field))
            return result
        finally:
            conn.close()


def _decode_json(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
