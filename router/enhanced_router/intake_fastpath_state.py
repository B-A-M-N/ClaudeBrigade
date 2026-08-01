"""Task intake / route proposal / fastpath verification persistence, split
out of state.py.

Sixteenth increment of the incremental extraction out of ``RouteState`` --
the first slice of the (much larger) "Run lifecycle" section. These seven
methods are called only from outside ``RouteState`` (fastpath routing, the
MCP control surface) and never from another ``RouteState`` method, so this
is a zero-cross-reference, zero-risk extraction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow_plus(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class IntakeFastpathRepository:
    """Mixin providing task-intake/route-proposal/fastpath persistence.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def create_task_intake(
        self,
        *,
        intake_id: str,
        run_id: str,
        session_id: str,
        prompt: str,
        request_kind: str,
        repository_features: dict,
        deterministic_signals: list[str],
        minimum_tier: str,
    ) -> dict:
        """Persist bounded intake facts before any workflow epoch is created."""
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO task_intakes "
                "(intake_id, run_id, session_id, prompt_digest, request_kind,"
                " repository_features_json, deterministic_signals_json, minimum_tier, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    intake_id,
                    run_id,
                    session_id,
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    request_kind,
                    json.dumps(repository_features, sort_keys=True, separators=(",", ":")),
                    json.dumps(sorted(set(deterministic_signals)), separators=(",", ":")),
                    minimum_tier,
                    _utcnow(),
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM task_intakes WHERE intake_id=?", (intake_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def reserve_route_proposal(
        self,
        *,
        proposal_id: str,
        intake_id: str,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        source: str,
        configuration_hash: str = "",
        candidate_digest: str = "",
        ttl_seconds: int = 30,
    ) -> dict:
        """Reserve a route-proposal row before inference starts.

        Makes proposal_id resolvable via get_route_proposal immediately --
        previously the row was only INSERTed after the detached fastpath
        model call finished, so a caller handed a proposal_id in a "queued"
        response could get "not found" for however long inference took.
        Idempotent (INSERT OR IGNORE): a caller retrying after a transport
        failure without knowing whether the first attempt's write landed
        just gets the existing reservation back, not a constraint error.
        """
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO route_proposals "
                "(proposal_id, intake_id, run_id, epoch_id, execution_id, source, status,"
                " parsed_proposal_json, validation_status, configuration_hash, candidate_digest,"
                " expires_at, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'queued', '{}', 'pending', ?, ?, ?, ?)",
                (proposal_id, intake_id, run_id, epoch_id, execution_id, source,
                 configuration_hash, candidate_digest, _utcnow_plus(ttl_seconds), _utcnow()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def mark_route_proposal_running(self, proposal_id: str) -> dict | None:
        """Transition a reserved proposal from queued to running.

        CAS-guarded: returns None (not the row) when the proposal wasn't
        actually still queued -- otherwise a caller could mistake "the
        UPDATE matched nothing" for "the transition succeeded."
        """
        conn = self._new_conn()
        try:
            cursor = conn.execute(
                "UPDATE route_proposals SET status='running' WHERE proposal_id=? AND status='queued'",
                (proposal_id,),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def complete_route_proposal(
        self,
        proposal_id: str,
        *,
        parsed_proposal: dict,
        validation_status: str,
        validation_reason: str = "",
        fastpath_model_id: str | None = None,
        fastpath_endpoint_id: str | None = None,
        raw_output_digest: str | None = None,
        confidence: float | None = None,
    ) -> dict | None:
        """Transition a reserved proposal to 'completed' with its real result.

        CAS-guarded (status IN ('queued','running')): a late/duplicate
        completion -- e.g. a retried detached job racing the original --
        cannot clobber a row that already reached a terminal state.
        """
        conn = self._new_conn()
        try:
            cursor = conn.execute(
                "UPDATE route_proposals SET status='completed', parsed_proposal_json=?,"
                " validation_status=?, validation_reason=?, fastpath_model_id=?,"
                " fastpath_endpoint_id=?, raw_output_digest=?, confidence=?, completed_at=?"
                " WHERE proposal_id=? AND status IN ('queued','running')",
                (json.dumps(parsed_proposal, sort_keys=True, separators=(",", ":")),
                 validation_status, validation_reason, fastpath_model_id, fastpath_endpoint_id,
                 raw_output_digest, confidence, _utcnow(), proposal_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def fail_route_proposal(self, proposal_id: str, reason: str) -> dict | None:
        """Transition a reserved proposal to 'failed' (CAS-guarded, see complete_route_proposal)."""
        conn = self._new_conn()
        try:
            cursor = conn.execute(
                "UPDATE route_proposals SET status='failed', validation_status='failed',"
                " validation_reason=?, completed_at=?"
                " WHERE proposal_id=? AND status IN ('queued','running')",
                (reason[:500], _utcnow(), proposal_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_route_proposal(
        self,
        *,
        proposal_id: str,
        intake_id: str,
        run_id: str,
        epoch_id: str,
        source: str,
        parsed_proposal: dict,
        validation_status: str,
        validation_reason: str = "",
        fastpath_model_id: str | None = None,
        fastpath_endpoint_id: str | None = None,
        raw_output_digest: str | None = None,
        confidence: float | None = None,
    ) -> dict:
        """Convenience wrapper for synchronous callers (tests, non-detached
        proposals): reserve then immediately complete in one call."""
        self.reserve_route_proposal(
            proposal_id=proposal_id, intake_id=intake_id, run_id=run_id, epoch_id=epoch_id,
            execution_id=f"sync:{proposal_id}", source=source,
        )
        result = self.complete_route_proposal(
            proposal_id,
            parsed_proposal=parsed_proposal, validation_status=validation_status,
            validation_reason=validation_reason, fastpath_model_id=fastpath_model_id,
            fastpath_endpoint_id=fastpath_endpoint_id, raw_output_digest=raw_output_digest,
            confidence=confidence,
        )
        assert result is not None
        return result

    def _expire_route_proposal_if_stale(self, conn: sqlite3.Connection, proposal_id: str) -> None:
        conn.execute(
            "UPDATE route_proposals SET status='expired', completed_at=?"
            " WHERE proposal_id=? AND status IN ('queued','running') AND expires_at < ?",
            (_utcnow(), proposal_id, _utcnow()),
        )
        conn.commit()

    def get_route_proposal(self, proposal_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            self._expire_route_proposal_if_stale(conn, proposal_id)
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_task_intake(self, intake_id: str) -> dict | None:
        """Return one bounded task-intake record for control-plane checks."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM task_intakes WHERE intake_id=?",
                (intake_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_route_proposal_for_run(
        self, proposal_id: str, run_id: str,
    ) -> dict | None:
        """Return a proposal only when it belongs to *run_id*."""
        conn = self._new_conn()
        try:
            self._expire_route_proposal_if_stale(conn, proposal_id)
            row = conn.execute(
                "SELECT * FROM route_proposals WHERE proposal_id=? AND run_id=?",
                (proposal_id, run_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_route_proposal_outcomes(self, run_id: str | None = None) -> dict:
        """Aggregate route-proposal outcome telemetry, optionally scoped to
        one run.

        Built entirely from data the proposal lifecycle (reserve/mark_running/
        complete/fail + controller disposition) already records -- no new
        recording path or schema needed. Answers the questions that matter
        before trusting fastpath's route influence: how often does a
        proposal actually complete, how often is it accepted vs rejected vs
        bypassed vs left to expire unreviewed.
        """
        conn = self._new_conn()
        try:
            where = "WHERE run_id = ?" if run_id is not None else ""
            params = (run_id,) if run_id is not None else ()
            by_status = dict(conn.execute(
                f"SELECT status, COUNT(*) FROM route_proposals {where} GROUP BY status", params,
            ).fetchall())
            by_validation_status = dict(conn.execute(
                f"SELECT validation_status, COUNT(*) FROM route_proposals {where} GROUP BY validation_status",
                params,
            ).fetchall())
            by_disposition = dict(conn.execute(
                f"SELECT COALESCE(controller_disposition, 'undecided'), COUNT(*) "
                f"FROM route_proposals {where} GROUP BY COALESCE(controller_disposition, 'undecided')",
                params,
            ).fetchall())
            total = sum(by_status.values())
            completed_and_reviewable = by_validation_status.get("accepted_for_controller_review", 0)
            accepted = by_disposition.get("accepted", 0)
            acceptance_rate = (
                accepted / completed_and_reviewable if completed_and_reviewable else None
            )
            return {
                "run_id": run_id,
                "total": total,
                "by_status": by_status,
                "by_validation_status": by_validation_status,
                "by_disposition": by_disposition,
                "acceptance_rate": acceptance_rate,
            }
        finally:
            conn.close()

    def get_pending_route_proposals(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return completed, undispositioned, unexpired proposals for this
        run/epoch -- the set get_runnable_actions surfaces as a
        controller_route_review action, so a completed proposal is
        discoverable through the same mechanism the controller already
        polls, instead of only being reachable by proactively guessing
        its proposal_id.
        """
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM route_proposals WHERE run_id=? AND epoch_id=?"
                " AND status='completed' AND validation_status='accepted_for_controller_review'"
                " AND controller_disposition IS NULL AND expires_at >= ?",
                (run_id, epoch_id, _utcnow()),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def set_route_proposal_disposition(
        self, proposal_id: str, run_id: str, disposition: str, reason: str = "",
        epoch_id: str | None = None,
    ) -> dict | None:
        """CAS-guarded controller disposition of a route proposal.

        'accepted' requires the strict precondition a proposal must meet
        before its routes can actually be applied: status='completed'
        (inference finished), validation_status='accepted_for_controller_review'
        (not bypassed/failed), still scoped to *epoch_id* -- the CURRENTLY
        active epoch, so a late proposal generated for an earlier epoch of
        this run can't be applied during a later one -- not expired, and
        not already dispositioned. Any other disposition (e.g. 'rejected')
        only requires the proposal belong to this run and not already be
        dispositioned; rejecting is safe regardless of what state inference
        left the proposal in.
        """
        conn = self._new_conn()
        try:
            self._expire_route_proposal_if_stale(conn, proposal_id)
            if disposition == "accepted":
                if epoch_id is None:
                    raise ValueError("accepting a proposal requires its epoch_id")
                cursor = conn.execute(
                    "UPDATE route_proposals SET controller_disposition=?, controller_reason=?,"
                    " applied_epoch_id=?, applied_at=?"
                    " WHERE proposal_id=? AND run_id=? AND epoch_id=?"
                    " AND status='completed' AND validation_status='accepted_for_controller_review'"
                    " AND controller_disposition IS NULL AND expires_at >= ?",
                    (disposition, reason, epoch_id, _utcnow(), proposal_id, run_id, epoch_id, _utcnow()),
                )
            else:
                cursor = conn.execute(
                    "UPDATE route_proposals SET controller_disposition=?, controller_reason=?"
                    " WHERE proposal_id=? AND run_id=? AND controller_disposition IS NULL",
                    (disposition, reason, proposal_id, run_id),
                )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_fastpath_verification(self, **values: object) -> dict:
        required = {
            "verification_id", "run_id", "epoch_id", "contract_digest", "evidence_digest",
            "decision", "checks_json", "violations_json", "confidence", "policy_disposition",
        }
        missing = required - values.keys()
        if missing:
            raise ValueError(f"missing fastpath verification fields: {sorted(missing)}")
        conn = self._new_conn()
        try:
            columns = sorted(values)
            placeholders = ",".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO fastpath_verifications ({','.join(columns)}, created_at) "
                f"VALUES ({placeholders}, ?)",
                [values[column] for column in columns] + [_utcnow()],
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM fastpath_verifications WHERE verification_id=?",
                (values["verification_id"],),
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()
