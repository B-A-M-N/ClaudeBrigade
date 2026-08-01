"""Conditional-phase evaluation, split out of state.py.

Thirteenth increment of the incremental extraction out of ``RouteState``.
Depends on the workflow phase state machine (``get_workflow_phases``,
``WorkflowPhaseStateError``) and the finding repository
(``get_open_accepted_findings``) -- both already extracted -- imported
directly from their own modules rather than back through ``state.py``, to
avoid a circular import.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from enhanced_router.workflow_phase_state import WorkflowPhaseStateError


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConditionEvaluationRepository:
    """Mixin providing conditional-phase evaluation methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does), plus ``get_open_accepted_findings`` and
    ``get_workflow_phases`` (both mixed into ``RouteState`` from their own
    repositories).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def evaluate_condition(
        self, run_id: str, epoch_id: str, phase_id: str, condition: str | None,
    ) -> dict:
        """Evaluate whether a conditional phase should be activated or skipped.

        Supported conditions:

        - ``accepted_findings``: Requires at least one open accepted finding.
          If none exist, the phase is auto-skipped (not required).
        - ``None`` or empty: Condition is satisfied (no constraint).

        Returns dict with keys: satisfied (bool), reason (str), evidence (dict).
        """
        if not condition:
            return {"satisfied": True, "reason": "No condition", "evidence": {}}

        if condition == "accepted_findings":
            findings = self.get_open_accepted_findings(run_id, epoch_id)
            count = len(findings)
            if count > 0:
                return {
                    "satisfied": True,
                    "reason": f"{count} accepted open finding(s) require repair",
                    "evidence": {"accepted_finding_count": count},
                }
            return {
                "satisfied": False,
                "reason": "No accepted findings to repair — auto-skipping",
                "evidence": {"accepted_finding_count": 0},
            }

        return {
            "satisfied": False,
            "reason": f"Unknown condition '{condition}'",
            "evidence": {},
        }

    def skip_conditional_phase(
        self, run_id: str, epoch_id: str, phase_id: str, condition: str | None,
        reason: str = "",
    ) -> dict:
        """Evaluate and skip a conditional phase if its condition is not met.

        Returns the phase dict (with status 'skipped' if skipped,
        or unchanged status if condition is satisfied).
        """
        evaluation = self.evaluate_condition(run_id, epoch_id, phase_id, condition)
        if evaluation["satisfied"]:
            # Condition is satisfied — do not skip, phase stays pending
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)

        # Auto-skip: condition is not satisfied
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT status FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[0] == "skipped":
                phases = self.get_workflow_phases(run_id, epoch_id)
                return next(p for p in phases if p["phase_id"] == phase_id)
            if row[0] != "pending":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[0]}, cannot skip (must be 'pending')"
                )

            required = conn.execute(
                "SELECT required FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if required is not None and bool(required[0]):
                raise WorkflowPhaseStateError(
                    f"Required phase '{phase_id}' cannot be conditionally skipped"
                )
            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status='skipped', completed_at=?, result_evidence=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (now, json.dumps({
                    "skip_type": "conditional",
                    "condition": condition,
                    "reason": reason or evaluation.get("reason", ""),
                    "evidence": evaluation.get("evidence", {}),
                }),
                 run_id, epoch_id, phase_id),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()
