"""Run registry (the ``runs`` table) persistence, split out of state.py.

Twenty-first increment of the incremental extraction out of ``RouteState``
-- another slice of "Run lifecycle". Touches only the ``runs`` table; the
authoritative record of a run's launch-time selection (inference profile,
sidecar profile, launch preset) and its controller capability credential.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRegistryRepository:
    """Mixin providing run-registry persistence methods.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    _RUN_COLUMNS = (
        "run_id", "claude_session_id", "cwd", "controller_capability_hash",
        "inference_profile_id", "sidecar_profile_id", "launch_preset_id",
        "token_budget", "created_at", "closed_at",
    )

    def create_run(
        self,
        run_id: str,
        session_id: str | None = None,
        cwd: str | None = None,
        controller_capability: str | None = None,
        inference_profile_id: str | None = None,
        sidecar_profile_id: str | None = None,
        launch_preset_id: str | None = None,
        token_budget: int | None = None,
    ) -> dict:
        """Insert run if not exists (idempotent). Returns run dict.

        ``token_budget`` is an optional cap on total tokens spent across
        every agent_execution in this run, enforced by
        RunnableActionRepository.claim_runnable_action. NULL/unset means
        unbounded, matching every run created before this field existed.
        """
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, claude_session_id, cwd, controller_capability_hash, "
                "inference_profile_id, sidecar_profile_id, launch_preset_id, "
                "token_budget, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    session_id,
                    cwd,
                    hashlib.sha256(controller_capability.encode("utf-8")).hexdigest()
                    if controller_capability else None,
                    inference_profile_id,
                    sidecar_profile_id,
                    launch_preset_id,
                    token_budget,
                    _utcnow(),
                ),
            )
            if controller_capability:
                conn.execute(
                    "UPDATE runs SET controller_capability_hash=COALESCE(controller_capability_hash, ?) "
                    "WHERE run_id=?",
                    (
                        hashlib.sha256(controller_capability.encode("utf-8")).hexdigest(),
                        run_id,
                    ),
                )
            if session_id is not None or cwd is not None:
                conn.execute(
                    "UPDATE runs SET claude_session_id=COALESCE(claude_session_id, ?), "
                    "cwd=COALESCE(cwd, ?) WHERE run_id=?",
                    (session_id, cwd, run_id),
                )
            if inference_profile_id is not None or sidecar_profile_id is not None or launch_preset_id is not None:
                conn.execute(
                    "UPDATE runs SET "
                    "inference_profile_id=COALESCE(inference_profile_id, ?), "
                    "sidecar_profile_id=COALESCE(sidecar_profile_id, ?), "
                    "launch_preset_id=COALESCE(launch_preset_id, ?) WHERE run_id=?",
                    (inference_profile_id, sidecar_profile_id, launch_preset_id, run_id),
                )
            conn.commit()
            row = conn.execute(
                f"SELECT {', '.join(self._RUN_COLUMNS)} FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Failed to create run {run_id}")
            return dict(zip(self._RUN_COLUMNS, row))
        finally:
            conn.close()

    def set_run_selection(
        self,
        run_id: str,
        *,
        inference_profile_id: str | None = None,
        sidecar_profile_id: str | None = None,
        launch_preset_id: str | None = None,
    ) -> dict | None:
        """Persist this run's launch selection as the authoritative record.

        Unlike create_run's COALESCE-guarded insert (which only fills a value
        the first time), this always overwrites -- the launcher calls it once
        config selection is final, so a later call intentionally replaces an
        earlier default. Only hooks reading get_run() should ever be treated
        as authoritative for routing; a launcher-exported environment
        variable only reaches the single process tree that inherited it,
        which breaks down the moment a router daemon is shared by more than
        one concurrent run.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE runs SET inference_profile_id=COALESCE(?, inference_profile_id), "
                "sidecar_profile_id=COALESCE(?, sidecar_profile_id), "
                "launch_preset_id=COALESCE(?, launch_preset_id) WHERE run_id=?",
                (inference_profile_id, sidecar_profile_id, launch_preset_id, run_id),
            )
            if cursor.rowcount == 0:
                conn.rollback()
                return None
            conn.commit()
            row = conn.execute(
                f"SELECT {', '.join(self._RUN_COLUMNS)} FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return dict(zip(self._RUN_COLUMNS, row)) if row is not None else None
        finally:
            conn.close()

    def get_run(self, run_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                f"SELECT {', '.join(self._RUN_COLUMNS)} FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(zip(self._RUN_COLUMNS, row))
        finally:
            conn.close()

    def active_run_selections(self) -> list[dict]:
        """Return inference/sidecar profile selections for every open run.

        A run is "open" while ``closed_at`` is unset. Used to bound the
        LiteLLM child process and agent manifest to what runs currently in
        flight can actually select, instead of every profile ever saved.
        """
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT inference_profile_id, sidecar_profile_id FROM runs "
                "WHERE closed_at IS NULL",
            ).fetchall()
            return [
                {"inference_profile_id": row[0], "sidecar_profile_id": row[1]}
                for row in rows
            ]
        finally:
            conn.close()

    def verify_controller_capability(
        self, run_id: str, capability: str, *, session_id: str | None = None,
    ) -> bool:
        """Validate the ephemeral controller credential for one run."""
        run = self.get_run(run_id)
        if run is None or run.get("closed_at"):
            return False
        if session_id and run.get("claude_session_id") != session_id:
            return False
        expected = str(run.get("controller_capability_hash") or "")
        actual = hashlib.sha256(capability.encode("utf-8")).hexdigest()
        return bool(expected) and hmac.compare_digest(expected, actual)

    def close_run(self, run_id: str) -> None:
        """Set closed_at. Idempotent."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE runs SET closed_at = ? WHERE run_id = ?",
                (_utcnow(), run_id),
            )
            conn.commit()
        finally:
            conn.close()
