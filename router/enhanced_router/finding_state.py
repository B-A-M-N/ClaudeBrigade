"""Finding lifecycle persistence, split out of state.py.

Fourth increment of the incremental extraction out of ``RouteState``: these
seven methods only touch the ``findings`` table through
``self._new_conn()``, so they move to their own module as a mixin without
touching any call site. ``RouteState`` still exposes these methods under
their original names.
"""

from __future__ import annotations

import sqlite3


class FindingRepository:
    """Mixin providing finding lifecycle persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def create_finding(
        self,
        finding_id: str,
        run_id: str,
        epoch_id: str,
        description: str,
        *,
        severity: str = "medium",
        category: str = "",
        source_phase_id: str | None = None,
        source_agent_id: str | None = None,
        evidence_json: str | None = None,
    ) -> dict:
        """Record a new finding. Returns the created finding row."""
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO findings
                   (finding_id, run_id, epoch_id, severity, category, description,
                    source_phase_id, source_agent_id, evidence_json, disposition, verification_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'pending')""",
                (finding_id, run_id, epoch_id, severity, category, description,
                 source_phase_id, source_agent_id, evidence_json or "{}"),
            )
            conn.commit()
            return self.get_finding(finding_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def get_finding(self, finding_id: str) -> dict | None:
        """Return a single finding by finding_id."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM findings WHERE finding_id = ?", (finding_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_finding_scoped(
        self, run_id: str, epoch_id: str, finding_id: str,
    ) -> dict | None:
        """Return a finding only when it belongs to the requested epoch."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM findings WHERE finding_id=? AND run_id=? AND epoch_id=?",
                (finding_id, run_id, epoch_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_findings(
        self,
        run_id: str,
        epoch_id: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """List findings for a run, optionally filtered by epoch, disposition, or status."""
        conn = self._new_conn()
        try:
            parts = ["SELECT * FROM findings WHERE run_id = ?"]
            params: list[str] = [run_id]
            if epoch_id:
                parts.append("AND epoch_id = ?")
                params.append(epoch_id)
            if disposition:
                parts.append("AND disposition = ?")
                params.append(disposition)
            if status:
                parts.append("AND verification_status = ?")
                params.append(status)
            parts.append("ORDER BY created_at DESC")
            rows = conn.execute(" ".join(parts), params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def adjudicate_finding(
        self,
        finding_id: str,
        disposition: str,
        *,
        reason: str = "",
        dispositioned_by: str = "",
        repair_agent_id: str | None = None,
        repair_phase_id: str | None = None,
    ) -> dict | None:
        """Accept, reject, waive, or mark a finding as duplicate.

        Accepted findings with a repair_agent_id are flagged for resolution.
        """
        conn = self._new_conn()
        try:
            conn.execute(
                """UPDATE findings SET disposition=?, disposition_reason=?,
                   dispositioned_at=datetime('now'), dispositioned_by=?,
                   repair_agent_id=?, repair_phase_id=?, updated_at=datetime('now')
                   WHERE finding_id=?""",
                (disposition, reason, dispositioned_by, repair_agent_id,
                 repair_phase_id, finding_id),
            )
            conn.commit()
            return self.get_finding(finding_id)
        finally:
            conn.close()

    def resolve_finding(
        self,
        finding_id: str,
        verification_status: str,
        resolution_evidence_json: str = "{}",
    ) -> dict | None:
        """Record resolution evidence and verification result for a finding."""
        conn = self._new_conn()
        try:
            conn.execute(
                """UPDATE findings SET verification_status=?,
                   resolution_evidence_json=?, updated_at=datetime('now')
                   WHERE finding_id=?""",
                (verification_status, resolution_evidence_json, finding_id),
            )
            conn.commit()
            return self.get_finding(finding_id)
        finally:
            conn.close()

    def get_open_accepted_findings(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return findings that are accepted but not yet verified (for repair contracts)."""
        return self.get_findings(run_id, epoch_id=epoch_id, disposition="accepted", status="pending")
