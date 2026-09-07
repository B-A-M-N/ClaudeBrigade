"""Durable task-contract and requirement coverage state.

The workflow ledger answers *which phases ran*.  This repository answers the
more important completion question: *which requested requirements are
covered, by what evidence, and with what disposition?*  It deliberately
contains no model or prompt logic; controllers publish and adjudicate the
records through the authenticated control surface.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, cast

from enhanced_router.repository_base import RepositoryMixin


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ContractStateError(ValueError):
    """Raised when a task contract transition is invalid."""


class ContractRepository(RepositoryMixin):
    """Repository for contracts, requirements, ambiguities and evidence."""

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - host override
        raise NotImplementedError

    def publish_task_contract(
        self,
        run_id: str,
        epoch_id: str,
        contract: dict[str, Any],
        *,
        source: str = "controller",
        contract_id: str | None = None,
        replace_draft: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(contract, dict) or not contract:
            raise ContractStateError("task contract must be a non-empty object")
        contract_id = contract_id or f"contract:{run_id}:{epoch_id}"
        digest = _digest(contract)
        now = _utcnow()
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM task_contracts WHERE run_id=? AND epoch_id=? AND status IN ('published','approved') "
                "ORDER BY version DESC LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            if existing is not None and not replace_draft:
                if existing["contract_digest"] == digest:
                    conn.rollback()
                    return dict(existing)
                raise ContractStateError("an immutable task contract is already published for this epoch")
            if existing is not None and existing["status"] == "approved":
                raise ContractStateError("an approved task contract is immutable for this epoch")
            version = int(
                conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM task_contracts WHERE run_id=? AND epoch_id=?",
                    (run_id, epoch_id),
                ).fetchone()[0]
            ) + 1
            if existing is not None:
                contract_id = f"{contract_id}:v{version}"
            if replace_draft:
                conn.execute(
                    "UPDATE task_contracts SET status='superseded', superseded_at=? "
                    "WHERE run_id=? AND epoch_id=? AND status IN ('draft','published')",
                    (now, run_id, epoch_id),
                )
            conn.execute(
                "INSERT INTO task_contracts "
                "(contract_id,run_id,epoch_id,version,contract_json,contract_digest,status,source,created_at,published_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (contract_id, run_id, epoch_id, version, json.dumps(contract, sort_keys=True, separators=(",", ":")),
                 digest, "published", source, now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM task_contracts WHERE contract_id=?", (contract_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_task_contract(self, run_id: str, epoch_id: str) -> dict[str, Any] | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM task_contracts WHERE run_id=? AND epoch_id=? "
                "AND status NOT IN ('superseded','rejected') ORDER BY version DESC LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def approve_task_contract(
        self, run_id: str, epoch_id: str, *, approved_by: str = "controller"
    ) -> dict[str, Any]:
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM task_contracts WHERE run_id=? AND epoch_id=? "
                "AND status='published' ORDER BY version DESC LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            if row is None:
                raise ContractStateError("no published task contract exists")
            contract_payload = json.loads(str(row["contract_json"] or "{}"))
            if not isinstance(contract_payload, dict) or not str(contract_payload.get("objective") or "").strip():
                raise ContractStateError("task contract requires a non-empty objective before approval")
            tier_row = conn.execute(
                "SELECT workflow_id, minimum_tier FROM epochs "
                "WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            tier = str((tier_row[0] if tier_row else None) or (tier_row[1] if tier_row else None) or "normal")
            requirements = conn.execute(
                "SELECT requirement_id, mandatory, status FROM requirements "
                "WHERE run_id=? AND epoch_id=? ORDER BY requirement_id",
                (run_id, epoch_id),
            ).fetchall()
            if tier != "trivial" and not any(bool(item[1]) for item in requirements):
                raise ContractStateError(
                    "non-trivial task contracts require at least one mandatory requirement"
                )
            open_ambiguities = conn.execute(
                "SELECT COUNT(*) FROM ambiguities WHERE run_id=? AND epoch_id=? AND status='open'",
                (run_id, epoch_id),
            ).fetchone()[0]
            if int(open_ambiguities or 0):
                raise ContractStateError(
                    f"resolve {int(open_ambiguities)} open task ambiguity(ies) before approval"
                )
            now = _utcnow()
            conn.execute(
                "UPDATE task_contracts SET status='approved', approved_by=?, approved_at=? WHERE contract_id=?",
                (approved_by, now, row["contract_id"]),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM task_contracts WHERE contract_id=?", (row["contract_id"],)).fetchone()
            assert updated is not None
            return dict(updated)
        finally:
            conn.close()

    def add_requirement(
        self,
        run_id: str,
        epoch_id: str,
        statement: str,
        *,
        category: str = "functional",
        mandatory: bool = True,
        acceptance: dict[str, Any] | None = None,
        requirement_id: str | None = None,
        risk: str = "normal",
    ) -> dict[str, Any]:
        if not statement.strip():
            raise ContractStateError("requirement statement cannot be empty")
        contract = self.get_task_contract(run_id, epoch_id)
        if contract is None:
            raise ContractStateError("publish a task contract before adding requirements")
        if str(contract.get("status") or "") == "approved":
            raise ContractStateError(
                "approved task contracts are immutable; publish a replacement contract version"
            )
        requirement_id = requirement_id or f"req:{run_id}:{epoch_id}:{_digest(statement)[:12]}"
        now = _utcnow()
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO requirements "
                "(requirement_id,run_id,epoch_id,contract_id,statement,category,mandatory,risk,acceptance_json,status,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(requirement_id) DO UPDATE SET statement=excluded.statement, category=excluded.category, "
                "mandatory=excluded.mandatory, risk=excluded.risk, acceptance_json=excluded.acceptance_json, updated_at=excluded.updated_at",
                (requirement_id, run_id, epoch_id, contract["contract_id"], statement.strip(), category,
                 1 if mandatory else 0, risk, json.dumps(acceptance or {}, sort_keys=True, separators=(",", ":")),
                 "open", now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM requirements WHERE requirement_id=?", (requirement_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_requirements(self, run_id: str, epoch_id: str) -> list[dict[str, Any]]:
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM requirements WHERE run_id=? AND epoch_id=? ORDER BY requirement_id",
                (run_id, epoch_id),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def update_requirement(self, requirement_id: str, *, status: str, reason: str = "") -> dict[str, Any] | None:
        if status not in {"open", "in_progress", "satisfied", "waived", "blocked"}:
            raise ContractStateError(f"invalid requirement status: {status}")
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE requirements SET status=?, status_reason=?, updated_at=? WHERE requirement_id=?",
                (status, reason[:2000], _utcnow(), requirement_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM requirements WHERE requirement_id=?", (requirement_id,)).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def link_requirement_evidence(
        self,
        requirement_id: str,
        *,
        evidence_kind: str,
        evidence_ref: str,
        evidence_digest: str | None = None,
        valid: bool | None = None,
    ) -> dict[str, Any]:
        evidence_id = f"evidence:{requirement_id}:{_digest([evidence_kind, evidence_ref])[:16]}"
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO evidence_links "
                "(evidence_id,requirement_id,evidence_kind,evidence_ref,evidence_digest,valid,created_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(evidence_id) DO UPDATE SET valid=excluded.valid,evidence_digest=excluded.evidence_digest",
                (evidence_id, requirement_id, evidence_kind, evidence_ref, evidence_digest,
                 None if valid is None else (1 if valid else 0), _utcnow()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM evidence_links WHERE evidence_id=?", (evidence_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_requirement_coverage(self, run_id: str, epoch_id: str) -> dict[str, Any]:
        requirements = self.get_requirements(run_id, epoch_id)
        conn = self._new_conn()
        try:
            links = conn.execute(
                "SELECT e.* FROM evidence_links e JOIN requirements r ON r.requirement_id=e.requirement_id "
                "WHERE r.run_id=? AND r.epoch_id=? ORDER BY e.created_at",
                (run_id, epoch_id),
            ).fetchall()
            by_req: dict[str, list[dict[str, Any]]] = {}
            for row in links:
                by_req.setdefault(str(row["requirement_id"]), []).append(dict(row))
            covered = 0
            missing: list[str] = []
            items: list[dict[str, Any]] = []
            for req in requirements:
                evidence = by_req.get(str(req["requirement_id"]), [])
                # Evidence is optimistic only after an explicit validation
                # decision. ``NULL`` means pending/unknown and must never
                # satisfy a mandatory requirement or a final coverage audit.
                valid = [item for item in evidence if item.get("valid") in {1, True}]
                try:
                    acceptance = json.loads(str(req.get("acceptance_json") or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    acceptance = {}
                required_evidence = acceptance.get("required_evidence", []) if isinstance(acceptance, dict) else []
                required_evidence = [str(item) for item in required_evidence] if isinstance(required_evidence, list) else []
                valid_kinds = {str(item.get("evidence_kind")) for item in valid}
                missing_evidence = [kind for kind in required_evidence if kind not in valid_kinds]
                is_covered = (
                    req.get("status") in {"satisfied", "waived"}
                    and bool(valid)
                    and not missing_evidence
                )
                if is_covered:
                    covered += 1
                elif req.get("mandatory"):
                    missing.append(str(req["requirement_id"]))
                items.append({
                    **req,
                    "evidence": evidence,
                    "required_evidence": required_evidence,
                    "missing_evidence": missing_evidence,
                    "covered": is_covered,
                })
            return {
                "run_id": run_id, "epoch_id": epoch_id, "total": len(requirements),
                "covered": covered, "missing_mandatory": missing, "complete": not missing,
                "requirements": items,
            }
        finally:
            conn.close()

    def record_coverage_audit(
        self,
        run_id: str,
        epoch_id: str,
        *,
        complete: bool,
        missing: list[str] | None = None,
        auditor: str = "controller",
        contract_version: int | None = None,
        workspace_generation: int | None = None,
        workspace_digest: str | None = None,
    ) -> dict[str, Any]:
        coverage = self.get_requirement_coverage(run_id, epoch_id)
        actual_missing = [str(item) for item in coverage.get("missing_mandatory", [])]
        if complete and actual_missing:
            raise ContractStateError(
                "cannot record a complete coverage audit while mandatory requirements "
                f"remain uncovered: {', '.join(actual_missing)}"
            )
        if contract_version is None:
            contract = self.get_task_contract(run_id, epoch_id)
            contract_version = int(contract["version"]) if contract else None
        if workspace_generation is None or workspace_digest is None:
            # RouteState supplies get_workspaces through the repository mixin.
            # A legacy unit caller may not have a canonical workspace, so keep
            # these fields nullable rather than fabricating a generation.
            get_workspaces = getattr(self, "get_workspaces", None)
            if callable(get_workspaces):
                workspaces = get_workspaces(run_id=run_id, epoch_id=epoch_id, kind="main", status="active")
                if workspaces:
                    workspace_items = cast(list[Any], workspaces)
                    workspace = cast(dict[str, Any], workspace_items[0])
                    if workspace_generation is None:
                        workspace_generation = int(workspace.get("canonical_generation") or 0)
                    if workspace_digest is None:
                        workspace_digest = str(
                            workspace.get("current_dirty_hash")
                            or workspace.get("dirty_patch_hash")
                            or ""
                        ) or None
        audit_id = f"coverage:{run_id}:{epoch_id}:{_utcnow()}"
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO coverage_audits "
                "(audit_id,run_id,epoch_id,complete,missing_json,auditor,contract_version,"
                "workspace_generation,workspace_digest,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    audit_id, run_id, epoch_id, 1 if complete else 0,
                    json.dumps(actual_missing if complete else (missing or actual_missing), separators=(",", ":")),
                    auditor, contract_version, workspace_generation, workspace_digest, _utcnow(),
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM coverage_audits WHERE audit_id=?", (audit_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def get_latest_coverage_audit(self, run_id: str, epoch_id: str) -> dict[str, Any] | None:
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM coverage_audits WHERE run_id=? AND epoch_id=? "
                "ORDER BY created_at DESC, audit_id DESC LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def add_ambiguity(
        self, run_id: str, epoch_id: str, question: str,
        *, options: list[str] | None = None, ambiguity_id: str | None = None,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ContractStateError("ambiguity question cannot be empty")
        ambiguity_id = ambiguity_id or f"ambiguity:{run_id}:{epoch_id}:{_digest(question)[:12]}"
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO ambiguities (ambiguity_id,run_id,epoch_id,question,options_json,status,created_at) "
                "VALUES (?,?,?,?,?,'open',?) ON CONFLICT(ambiguity_id) DO UPDATE SET question=excluded.question,options_json=excluded.options_json",
                (ambiguity_id, run_id, epoch_id, question.strip(), json.dumps(options or [], separators=(",", ":")), _utcnow()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM ambiguities WHERE ambiguity_id=?", (ambiguity_id,)).fetchone()
            assert row is not None
            result = dict(row)
            result["options"] = json.loads(result.pop("options_json") or "[]")
            return result
        finally:
            conn.close()

    def resolve_ambiguity(
        self, ambiguity_id: str, resolution: str, *, resolved_by: str = "controller", status: str = "resolved"
    ) -> dict[str, Any] | None:
        if status not in {"resolved", "deferred"}:
            raise ContractStateError("ambiguity status must be resolved or deferred")
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE ambiguities SET resolution=?,status=?,resolved_by=?,resolved_at=? WHERE ambiguity_id=?",
                (resolution[:4000], status, resolved_by[:256], _utcnow(), ambiguity_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM ambiguities WHERE ambiguity_id=?", (ambiguity_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["options"] = json.loads(result.pop("options_json") or "[]")
            return result
        finally:
            conn.close()

    def get_ambiguities(self, run_id: str, epoch_id: str) -> list[dict[str, Any]]:
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM ambiguities WHERE run_id=? AND epoch_id=? ORDER BY created_at",
                (run_id, epoch_id),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["options"] = json.loads(item.pop("options_json") or "[]")
                result.append(item)
            return result
        finally:
            conn.close()
