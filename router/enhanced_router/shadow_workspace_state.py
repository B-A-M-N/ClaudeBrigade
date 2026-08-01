"""Shadow workspace / changeset / integration lifecycle, split out of
state.py.

Fifteenth increment of the incremental extraction out of ``RouteState``.
Covers the copy-on-write shadow workspace model for mutating agents: git
worktree registration, canonical workspace advancement (compare-and-set on
generation + dirty hash), the integration journal that serializes concurrent
``git apply`` attempts against one canonical workspace, changesets, and
integration candidate dispositions.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from enhanced_router.state_errors import WorkflowStateError


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ShadowWorkspaceRepository:
    """Mixin providing shadow workspace/changeset/integration persistence.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def create_workspace(
        self,
        *,
        workspace_id: str,
        run_id: str,
        epoch_id: str,
        kind: str,
        path: str,
        base_sha: str | None,
        dirty_patch_hash: str | None,
        status: str = "active",
        owner_execution_id: str | None = None,
        baseline_untracked_files: list[str] | None = None,
        parent_canonical_generation: int | None = None,
        parent_dirty_patch_hash: str | None = None,
    ) -> dict:
        """Register a main, shadow, or integration Git workspace."""
        if kind not in {"main", "shadow", "integration"}:
            raise ValueError(f"invalid workspace kind: {kind}")
        if status not in {"active", "ready", "merged", "discarded", "failed"}:
            raise ValueError(f"invalid workspace status: {status}")
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO workspaces
                   (workspace_id, run_id, epoch_id, kind, path, base_sha,
                    dirty_patch_hash, current_base_sha, current_dirty_hash,
                    parent_canonical_generation, parent_dirty_patch_hash,
                    status, owner_execution_id, baseline_untracked_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (workspace_id, run_id, epoch_id, kind, path, base_sha,
                 dirty_patch_hash, base_sha, dirty_patch_hash,
                 parent_canonical_generation, parent_dirty_patch_hash,
                 status, owner_execution_id,
                 json.dumps(sorted(baseline_untracked_files or [])), _utcnow()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def register_main_workspace(
        self,
        *,
        workspace_id: str,
        run_id: str,
        epoch_id: str,
        path: str,
        base_sha: str,
        dirty_patch_hash: str,
        baseline_untracked_files: list[str] | None = None,
    ) -> dict:
        """Register one canonical checkout and reject cross-run ownership."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=? AND kind='main' "
                "AND status IN ('active','ready') LIMIT 1",
                (workspace_id,),
            ).fetchone()
            if row is not None:
                existing = dict(row)
                if existing["run_id"] != run_id:
                    raise WorkflowStateError(
                        f"canonical workspace is already active for run {existing['run_id']}"
                    )
                current_base_sha = existing.get("current_base_sha") or existing.get("base_sha")
                current_dirty_hash = existing.get("current_dirty_hash") or existing.get("dirty_patch_hash")
                if current_base_sha != base_sha or current_dirty_hash != dirty_patch_hash:
                    raise WorkflowStateError("canonical workspace baseline changed during the run")
                if epoch_id and existing.get("epoch_id") != epoch_id and existing.get("epoch_id") == "session-intake":
                    conn.execute(
                        "UPDATE workspaces SET epoch_id=? WHERE workspace_id=?",
                        (epoch_id, workspace_id),
                    )
                    existing["epoch_id"] = epoch_id
                conn.commit()
                return existing
            conn.execute(
                """INSERT INTO workspaces
                   (workspace_id, run_id, epoch_id, kind, path, base_sha,
                    dirty_patch_hash, current_base_sha, current_dirty_hash,
                    canonical_generation, status, baseline_untracked_json, created_at)
                   VALUES (?, ?, ?, 'main', ?, ?, ?, ?, ?, 0, 'active', ?, ?)""",
                (workspace_id, run_id, epoch_id, path, base_sha, dirty_patch_hash,
                 base_sha, dirty_patch_hash,
                 json.dumps(sorted(baseline_untracked_files or [])), _utcnow()),
            )
            conn.commit()
            created = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            assert created is not None
            return dict(created)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def validate_execution_workspace(
        self, workspace_id: str | None, execution_id: str, role: str,
    ) -> dict:
        """Require mutating executions to own an active shadow workspace."""
        if role not in {"implementer", "repairer", "controller"}:
            return {"valid": True, "workspace": None}
        if not workspace_id:
            return {"valid": False, "reason": "mutating execution has no workspace"}
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            return {"valid": False, "reason": "workspace is not registered"}
        if workspace.get("kind") != "shadow":
            return {"valid": False, "reason": "mutating execution workspace is not a shadow"}
        if workspace.get("status") != "active":
            return {"valid": False, "reason": "mutating execution workspace is not active"}
        if workspace.get("owner_execution_id") != execution_id:
            return {"valid": False, "reason": "workspace belongs to another execution"}
        return {"valid": True, "workspace": workspace}

    def advance_canonical_workspace(
        self,
        *,
        workspace_id: str,
        expected_generation: int,
        expected_dirty_hash: str,
        applied_changeset_id: str,
        new_base_sha: str,
        new_dirty_hash: str,
    ) -> dict:
        """Advance the canonical workspace after one applied changeset."""
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=? AND kind='main' AND status='active'",
                (workspace_id,),
            ).fetchone()
            if row is None:
                raise WorkflowStateError("canonical workspace is not active")
            current_generation = int(row["canonical_generation"] or 0)
            current_hash = row["current_dirty_hash"] or row["dirty_patch_hash"]
            if current_generation != expected_generation or current_hash != expected_dirty_hash:
                raise WorkflowStateError("canonical workspace generation changed before advancement")
            conn.execute(
                "UPDATE workspaces SET canonical_generation=?, current_base_sha=?, "
                "current_dirty_hash=?, last_changeset_id=?, base_sha=?, dirty_patch_hash=? "
                "WHERE workspace_id=?",
                (
                    current_generation + 1, new_base_sha, new_dirty_hash,
                    applied_changeset_id, new_base_sha, new_dirty_hash, workspace_id,
                ),
            )
            conn.commit()
            result = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            assert result is not None
            return dict(result)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def begin_integration_journal(
        self,
        *,
        journal_id: str,
        run_id: str,
        epoch_id: str,
        workspace_id: str,
        changeset_id: str,
        expected_generation: int,
        expected_dirty_hash: str,
    ) -> dict:
        """Open the journal for one integration, serializing by workspace.

        The DB-level canonical_generation check in advance_canonical_workspace
        runs *after* the actual ``git apply`` has already mutated the
        worktree, so it cannot by itself prevent two concurrent integrations
        against the same canonical workspace from racing at the filesystem
        level.  Rejecting a second 'applying' journal for the same
        workspace_id here -- before any git apply happens -- is what
        actually serializes them.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            in_flight = conn.execute(
                "SELECT journal_id FROM integration_journal "
                "WHERE workspace_id=? AND status='applying'",
                (workspace_id,),
            ).fetchone()
            if in_flight is not None:
                conn.rollback()
                raise WorkflowStateError(
                    f"another integration is already applying to this canonical "
                    f"workspace: {in_flight[0]}"
                )
            existing = conn.execute(
                "SELECT journal_id FROM integration_journal WHERE journal_id=?",
                (journal_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO integration_journal "
                    "(journal_id, run_id, epoch_id, workspace_id, changeset_id, "
                    "expected_generation, expected_dirty_hash, status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'applying', ?)",
                    (
                        journal_id, run_id, epoch_id, workspace_id, changeset_id,
                        expected_generation, expected_dirty_hash, _utcnow(),
                    ),
                )
            else:
                # A retry of a previously terminal (failed) attempt for the
                # same changeset -- reopen it rather than silently reusing
                # the stale terminal row (INSERT OR IGNORE would have done
                # that, leaving the journal saying 'failed' while a fresh
                # git apply proceeded underneath it).
                conn.execute(
                    "UPDATE integration_journal SET run_id=?, epoch_id=?, workspace_id=?, "
                    "changeset_id=?, expected_generation=?, expected_dirty_hash=?, "
                    "status='applying', error=NULL, created_at=?, completed_at=NULL "
                    "WHERE journal_id=?",
                    (
                        run_id, epoch_id, workspace_id, changeset_id,
                        expected_generation, expected_dirty_hash, _utcnow(), journal_id,
                    ),
                )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM integration_journal WHERE journal_id=?", (journal_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def finish_integration_journal(
        self, journal_id: str, status: str, error: str | None = None,
    ) -> dict | None:
        if status not in {"completed", "reconciled", "failed"}:
            raise ValueError("invalid integration journal status")
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE integration_journal SET status=?, error=?, completed_at=? "
                "WHERE journal_id=? AND status='applying'",
                (status, error, _utcnow(), journal_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM integration_journal WHERE journal_id=?", (journal_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_pending_integration_journals(self, run_id: str | None = None) -> list[dict]:
        conn = self._new_conn()
        try:
            if run_id is None:
                rows = conn.execute(
                    "SELECT * FROM integration_journal WHERE status='applying' "
                    "ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM integration_journal WHERE status='applying' "
                    "AND run_id=? ORDER BY created_at", (run_id,)
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_workspace(self, workspace_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_workspaces(
        self,
        *,
        run_id: str | None = None,
        epoch_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        conn = self._new_conn()
        try:
            clauses: list[str] = []
            params: list[object] = []
            for column, value in (("run_id", run_id), ("epoch_id", epoch_id),
                                  ("kind", kind), ("status", status)):
                if value is not None:
                    clauses.append(f"{column}=?")
                    params.append(value)
            query = "SELECT * FROM workspaces"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY created_at, workspace_id"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def update_workspace_status(self, workspace_id: str, status: str) -> dict | None:
        if status not in {"active", "ready", "merged", "discarded", "failed"}:
            raise ValueError(f"invalid workspace status: {status}")
        conn = self._new_conn()
        try:
            released = _utcnow() if status in {"merged", "discarded", "failed"} else None
            conn.execute(
                "UPDATE workspaces SET status=?, released_at=COALESCE(?, released_at) "
                "WHERE workspace_id=?",
                (status, released, workspace_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def create_changeset(
        self,
        *,
        changeset_id: str,
        execution_id: str,
        workspace_id: str,
        base_sha: str,
        patch_digest: str,
        changed_files: list[str],
        result: dict,
        status: str,
        patch: bytes | None = None,
        parent_canonical_generation: int | None = None,
    ) -> dict:
        if status not in {"proposed", "validated", "rejected", "merged"}:
            raise ValueError(f"invalid changeset status: {status}")
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO execution_changesets
                   (changeset_id, execution_id, workspace_id, base_sha,
                    patch_digest, changed_files_json, result_json, patch_blob,
                    parent_canonical_generation, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (changeset_id, execution_id, workspace_id, base_sha, patch_digest,
                 json.dumps(changed_files, sort_keys=True),
                 json.dumps(result, sort_keys=True), patch,
                 parent_canonical_generation, status, _utcnow()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_changeset(self, changeset_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_changesets(
        self,
        *,
        run_id: str | None = None,
        epoch_id: str | None = None,
        execution_id: str | None = None,
    ) -> list[dict]:
        conn = self._new_conn()
        try:
            query = (
                "SELECT ec.* FROM execution_changesets ec "
                "JOIN workspaces w ON w.workspace_id=ec.workspace_id"
            )
            clauses: list[str] = []
            params: list[object] = []
            for expression, value in (("w.run_id", run_id), ("w.epoch_id", epoch_id),
                                      ("ec.execution_id", execution_id)):
                if value is not None:
                    clauses.append(f"{expression}=?")
                    params.append(value)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY ec.created_at, ec.changeset_id"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def mark_changeset_merged(self, changeset_id: str) -> dict | None:
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE execution_changesets SET status='merged' WHERE changeset_id=?",
                (changeset_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def mark_changeset_rejected(self, changeset_id: str) -> dict | None:
        """Close a changeset that the controller explicitly discarded."""
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE execution_changesets SET status='rejected' WHERE changeset_id=?",
                (changeset_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM execution_changesets WHERE changeset_id=?", (changeset_id,)
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def create_integration_candidate(
        self,
        *,
        candidate_id: str,
        run_id: str,
        epoch_id: str,
        changeset_id: str,
        overlap: dict,
        validation: dict,
        disposition: str,
        integration_execution_id: str | None = None,
    ) -> dict:
        if disposition not in {"green", "yellow", "red", "pending", "resolved"}:
            raise ValueError(f"invalid integration disposition: {disposition}")
        conn = self._new_conn()
        try:
            conn.execute(
                """INSERT INTO integration_candidates
                   (candidate_id, run_id, epoch_id, changeset_id, overlap_json,
                    validation_json, disposition, integration_execution_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (candidate_id, run_id, epoch_id, changeset_id,
                 json.dumps(overlap, sort_keys=True), json.dumps(validation, sort_keys=True),
                 disposition, integration_execution_id, _utcnow()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM integration_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def mark_integration_candidate(
        self, changeset_id: str, *, disposition: str, validation: dict | None = None,
    ) -> int:
        if disposition not in {"green", "yellow", "red", "pending", "resolved"}:
            raise ValueError(f"invalid integration disposition: {disposition}")
        conn = self._new_conn()
        try:
            sets = ["disposition=?"]
            params: list[object] = [disposition]
            if validation is not None:
                sets.append("validation_json=?")
                params.append(json.dumps(validation, sort_keys=True))
            params.append(changeset_id)
            result = conn.execute(
                f"UPDATE integration_candidates SET {', '.join(sets)} WHERE changeset_id=?",
                params,
            )
            conn.commit()
            return result.rowcount
        finally:
            conn.close()

    def get_integration_candidates(
        self, *, run_id: str, epoch_id: str | None = None,
        disposition: str | None = None,
    ) -> list[dict]:
        """Return controller-visible changeset integration decisions."""
        conn = self._new_conn()
        try:
            clauses = ["run_id=?"]
            params: list[object] = [run_id]
            if epoch_id is not None:
                clauses.append("epoch_id=?")
                params.append(epoch_id)
            if disposition is not None:
                clauses.append("disposition=?")
                params.append(disposition)
            rows = conn.execute(
                "SELECT * FROM integration_candidates WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at, candidate_id",
                params,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
