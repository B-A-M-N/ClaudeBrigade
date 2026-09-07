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

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from enhanced_router.state_errors import WorkflowStateError


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow_age(max_age_seconds: int) -> str:
    """Return an ISO-8601 timestamp that is *max_age_seconds* in the past."""
    return (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()


# Mirrors mutation_lease_state._STALE_LEASE_MAX_AGE_SECONDS: applied lazily
# inside get_workspaces (see _expire_stale_workspaces) rather than via a
# scheduled sweep, since nothing else periodically reaps a shadow workspace
# whose owning subagent died without a clean status transition -- only
# session_end.py's full-session-close cleanup ever caught it before this.
_STALE_SHADOW_WORKSPACE_MAX_AGE_SECONDS = 1_200


class ShadowWorkspaceRepository(RepositoryMixin):
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
            now = _utcnow()
            conn.execute(
                """INSERT INTO workspaces
                   (workspace_id, run_id, epoch_id, kind, path, base_sha,
                    dirty_patch_hash, current_base_sha, current_dirty_hash,
                    parent_canonical_generation, parent_dirty_patch_hash,
                    status, owner_execution_id, baseline_untracked_json, created_at,
                    heartbeat_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (workspace_id, run_id, epoch_id, kind, path, base_sha,
                 dirty_patch_hash, base_sha, dirty_patch_hash,
                 parent_canonical_generation, parent_dirty_patch_hash,
                 status, owner_execution_id,
                 json.dumps(sorted(baseline_untracked_files or [])), now, now),
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
            now = _utcnow()
            conn.execute(
                """INSERT INTO workspaces
                   (workspace_id, run_id, epoch_id, kind, path, base_sha,
                    dirty_patch_hash, current_base_sha, current_dirty_hash,
                    canonical_generation, status, baseline_untracked_json, created_at,
                    heartbeat_at)
                   VALUES (?, ?, ?, 'main', ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?)""",
                (workspace_id, run_id, epoch_id, path, base_sha, dirty_patch_hash,
                 base_sha, dirty_patch_hash,
                 json.dumps(sorted(baseline_untracked_files or [])), now, now),
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
            self._invalidate_completion_evidence_in_transaction(
                conn,
                run_id=str(row["run_id"]),
                epoch_id=str(row["epoch_id"]),
                generation=current_generation + 1,
                reason=f"canonical workspace advanced by changeset {applied_changeset_id}",
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

    @staticmethod
    def _invalidate_completion_evidence_in_transaction(
        conn: sqlite3.Connection,
        *,
        run_id: str,
        epoch_id: str,
        generation: int,
        reason: str,
    ) -> None:
        """Void signoffs after canonical mutation while retaining history.

        The execution ledger remains immutable evidence of what happened.  A
        completed review is not allowed to authorize a newer workspace, so
        its phase is reopened and its acceptance is cleared in the same
        transaction as the canonical-generation advance.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE completion_tokens SET consumed_at=COALESCE(consumed_at, ?) "
            "WHERE run_id=? AND epoch_id=? AND consumed_at IS NULL",
            (now, run_id, epoch_id),
        )
        conn.execute(
            "UPDATE coverage_audits SET complete=0, missing_json=?, created_at=? "
            "WHERE run_id=? AND epoch_id=? AND complete=1",
            (json.dumps(["canonical workspace changed"], separators=(",", ":")), now, run_id, epoch_id),
        )
        phase_rows = conn.execute(
            "SELECT phase_id, max_attempts FROM workflow_phases "
            "WHERE run_id=? AND epoch_id=? AND status IN ('active','completed','skipped')",
            (run_id, epoch_id),
        ).fetchall()
        markers = ("ground", "senior", "completion", "critical", "verification", "final", "audit")
        for phase in phase_rows:
            phase_id = str(phase[0]).lower()
            if not any(marker in phase_id for marker in markers):
                continue
            attempts = conn.execute(
                "SELECT COUNT(*) FROM agent_executions WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase[0]),
            ).fetchone()[0]
            conn.execute(
                "UPDATE workflow_phases SET status='pending', started_at=NULL, "
                "completed_at=NULL, result_evidence=NULL, error=?, "
                "invalidated_at=?, invalidation_reason=?, evidence_generation=NULL, "
                "evidence_digest=NULL, max_attempts=MAX(COALESCE(max_attempts, 1), ?) "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                ("stale completion evidence: " + reason, now, reason, int(attempts) + 1,
                 run_id, epoch_id, phase[0]),
            )
            conn.execute(
                "UPDATE agent_executions SET accepted_by_controller=NULL, "
                "evidence_valid=0, result_disposition='invalidated', "
                "adjudication_reason=?, adjudicated_at=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=? AND status='completed'",
                (reason, now, run_id, epoch_id, phase[0]),
            )

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

    def _expire_stale_workspaces(self) -> None:
        """Reclaim shadow workspaces whose owning subagent died without a
        clean status transition (SubagentStop/StopFailure), the same
        lazy-expiry-on-read shape as acquire_mutation_lease and
        _active_action_claims. Scoped to kind='shadow' only -- a stale
        'main'/'integration' workspace is a different lifecycle and must
        not be auto-discarded just because it's been active a while.
        """
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE workspaces SET status='discarded', released_at=? "
                "WHERE kind='shadow' AND status='active' AND heartbeat_at < ?",
                (_utcnow(), _utcnow_age(_STALE_SHADOW_WORKSPACE_MAX_AGE_SECONDS)),
            )
            conn.commit()
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
        self._expire_stale_workspaces()
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
            now = _utcnow()
            released = now if status in {"merged", "discarded", "failed"} else None
            conn.execute(
                "UPDATE workspaces SET status=?, released_at=COALESCE(?, released_at), "
                "heartbeat_at=? WHERE workspace_id=?",
                (status, released, now, workspace_id),
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
            result = dict(row) if row is not None else None
            if result and result.get("execution_id"):
                execution = conn.execute(
                    "SELECT package_id FROM agent_executions WHERE execution_id=?",
                    (result["execution_id"],),
                ).fetchone()
                if execution is not None and execution[0]:
                    conn.execute(
                        "UPDATE work_packages SET status='integrated', status_reason=?, updated_at=? "
                        "WHERE package_id=? AND status IN ('completed','running','claimed','ready')",
                        (f"changeset {changeset_id} integrated", _utcnow(), execution[0]),
                    )
                    conn.commit()
            return result
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
