"""Controller model policy persistence, split out of state.py.

Tenth increment of the incremental extraction out of ``RouteState``.
``ControllerModelError`` moves here too (rather than staying in state.py)
so this module is self-contained; ``state.py`` re-exports it under its
original name (``from enhanced_router.state import ControllerModelError``)
so existing importers (``routing.py``, tests) are unaffected.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ControllerModelError(Exception):
    """Raised when a controller model is not permitted by policy."""


class ControllerPolicyRepository(RepositoryMixin):
    """Mixin providing controller model policy persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def upsert_controller_policy(
        self,
        run_id: str,
        model_ids: list[str],
        policy: str = "reject",
    ) -> None:
        """Store or update a controller model policy."""
        if policy not in ("allow", "reject"):
            raise ValueError(f"Invalid policy: {policy}. Must be 'allow' or 'reject'")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _utcnow()
            conn.execute(
                """INSERT INTO controller_policies (run_id, permitted_models, model_change_policy, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       permitted_models = excluded.permitted_models,
                       model_change_policy = excluded.model_change_policy,
                       updated_at = excluded.updated_at""",
                (run_id, json.dumps(model_ids), policy, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_controller_policy(self, run_id: str) -> dict | None:
        """Return the controller policy for *run_id*, or None.

        for the public API.
        """
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT run_id, permitted_models, model_change_policy, created_at, updated_at "
                "FROM controller_policies WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(
                ("run_id", "permitted_models", "model_change_policy", "created_at", "updated_at"), row
            ))
        finally:
            conn.close()

    def validate_controller_model(self, run_id: str, model_id: str) -> bool:
        """Check whether *model_id* is permitted under the controller policy.

        Returns ``True`` when the model is allowed (or no policy is set).
        Raises ``ControllerModelError`` when the model is rejected.
        """
        try:
            from enhanced_router.registry import get_registry
            spec = get_registry().get_model(model_id)
            if spec.enabled and (
                spec.capabilities.controller_eligible
                or spec.backend == "anthropic-passthrough"
            ):
                return True
        except Exception:
            pass
        policy = self.get_controller_policy(run_id)
        if policy is None:
            return True  # no policy means allow
        if policy["model_change_policy"] == "allow":
            return True
        permitted = json.loads(policy["permitted_models"])
        if model_id in permitted:
            return True
        raise ControllerModelError(f"Model '{model_id}' is not permitted by controller policy")
