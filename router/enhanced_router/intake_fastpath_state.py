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
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def create_route_proposal(
        self,
        *,
        proposal_id: str,
        intake_id: str,
        source: str,
        parsed_proposal: dict,
        validation_status: str,
        validation_reason: str = "",
        fastpath_model_id: str | None = None,
        fastpath_endpoint_id: str | None = None,
        raw_output_digest: str | None = None,
        confidence: float | None = None,
    ) -> dict:
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO route_proposals "
                "(proposal_id, intake_id, source, fastpath_model_id, fastpath_endpoint_id,"
                " raw_output_digest, parsed_proposal_json, confidence, validation_status,"
                " validation_reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (proposal_id, intake_id, source, fastpath_model_id, fastpath_endpoint_id,
                 raw_output_digest, json.dumps(parsed_proposal, sort_keys=True, separators=(",", ":")),
                 confidence, validation_status, validation_reason, _utcnow()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM route_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_route_proposal(self, proposal_id: str) -> dict | None:
        conn = self._new_conn()
        try:
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
        """Return a proposal only when its intake belongs to *run_id*."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT rp.* FROM route_proposals AS rp "
                "JOIN task_intakes AS ti ON ti.intake_id=rp.intake_id "
                "WHERE rp.proposal_id=? AND ti.run_id=?",
                (proposal_id, run_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def set_route_proposal_disposition(
        self, proposal_id: str, disposition: str, reason: str = "", epoch_id: str | None = None
    ) -> dict | None:
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE route_proposals SET controller_disposition=?, controller_reason=?,"
                " applied_epoch_id=COALESCE(?, applied_epoch_id), applied_at=COALESCE(?, applied_at)"
                " WHERE proposal_id=?",
                (disposition, reason, epoch_id, _utcnow() if epoch_id else None, proposal_id),
            )
            conn.commit()
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
