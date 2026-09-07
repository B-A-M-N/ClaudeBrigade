"""Persisted, package-scoped implementation work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from enhanced_router.repository_base import RepositoryMixin


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_digest(package: dict[str, Any]) -> str:
    material = {
        key: package.get(key)
        for key in (
            "package_id", "phase_id", "objective", "path_scope", "requirements",
            "acceptance", "required_tests", "prohibited_paths", "dependencies",
            "contract_version", "prompt_contract",
        )
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class WorkPackageStateError(ValueError):
    """Raised when a package would violate persisted scheduling policy."""


class WorkPackageRepository(RepositoryMixin):
    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - host override
        raise NotImplementedError

    def publish_work_package(
        self,
        run_id: str,
        epoch_id: str,
        phase_id: str,
        objective: str,
        *,
        package_id: str | None = None,
        path_scope: list[str] | None = None,
        requirement_ids: list[str] | None = None,
        dependencies: list[str] | None = None,
        acceptance: list[str] | None = None,
        required_tests: list[str] | None = None,
        prohibited_paths: list[str] | None = None,
        contract_digest: str | None = None,
        contract_version: int = 1,
        prompt_contract: dict[str, Any] | None = None,
        display_name: str | None = None,
        summary: str | None = None,
        risk: str = "normal",
        can_run_parallel: bool = True,
    ) -> dict[str, Any]:
        if not objective.strip():
            raise WorkPackageStateError("work-package objective cannot be empty")
        package_id = package_id or f"pkg:{run_id}:{epoch_id}:{phase_id}:{hashlib.sha256(objective.encode()).hexdigest()[:12]}"
        if contract_version < 1:
            raise WorkPackageStateError("work-package contract_version must be positive")
        if prompt_contract is None:
            prompt_contract = {
                "objective": objective.strip(),
                "path_scope": path_scope or [],
                "acceptance": acceptance or [],
                "required_tests": required_tests or [],
                "prohibited_paths": prohibited_paths or [],
            }
        record: dict[str, Any] = {
            "package_id": package_id,
            "run_id": run_id,
            "epoch_id": epoch_id,
            "phase_id": phase_id,
            "objective": objective.strip(),
            "path_scope": path_scope or [],
            "requirements": requirement_ids or [],
            "dependencies": dependencies or [],
            "acceptance": acceptance or [],
            "required_tests": required_tests or [],
            "prohibited_paths": prohibited_paths or [],
            "contract_version": contract_version,
            "prompt_contract": prompt_contract,
        }
        prompt_contract_digest = hashlib.sha256(
            json.dumps(prompt_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        computed_digest = package_digest(record)
        if contract_digest is not None and str(contract_digest) != computed_digest:
            raise WorkPackageStateError(
                "work-package contract_digest does not match its persisted contract"
            )
        digest = computed_digest
        now = _utcnow()
        conn = self._new_conn()
        try:
            phase = conn.execute(
                "SELECT mutating FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            contract = conn.execute(
                "SELECT contract_id, version, status FROM task_contracts WHERE run_id=? AND epoch_id=? "
                "AND status IN ('published','approved') ORDER BY version DESC LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            if phase is not None and phase[0] and (contract is None or contract[2] != "approved"):
                raise WorkPackageStateError(
                    "mutating work packages require an approved task contract"
                )
            if phase is not None and phase[0] and contract is not None:
                if int(contract_version) != int(contract[1]):
                    raise WorkPackageStateError(
                        f"work package contract_version {contract_version} does not match "
                        f"approved contract version {contract[1]}"
                    )
                if not requirement_ids:
                    raise WorkPackageStateError(
                        "mutating work packages must claim at least one task requirement"
                    )
                requirement_ids = [str(item) for item in requirement_ids]
                placeholders = ",".join("?" for _ in requirement_ids)
                rows = conn.execute(
                    "SELECT requirement_id FROM requirements WHERE run_id=? AND epoch_id=? "
                    f"AND contract_id=? AND requirement_id IN ({placeholders})",
                    [run_id, epoch_id, contract[0], *requirement_ids],
                ).fetchall()
                known = {str(item[0]) for item in rows}
                missing_requirements = sorted(set(requirement_ids) - known)
                if missing_requirements:
                    raise WorkPackageStateError(
                        "work package references unknown task requirement(s): "
                        + ", ".join(missing_requirements)
                    )
            if phase is not None and phase[0] and not (path_scope or []):
                raise WorkPackageStateError(
                    "mutating work packages require an explicit non-empty path scope"
                )
            if phase is not None and phase[0] and not (acceptance or []):
                raise WorkPackageStateError(
                    "mutating work packages require at least one acceptance criterion"
                )
            if phase is not None and phase[0] and not (required_tests or []):
                raise WorkPackageStateError(
                    "mutating work packages require at least one required test or check"
                )
            dependencies = [str(item) for item in (dependencies or [])]
            if package_id in dependencies:
                raise WorkPackageStateError("a work package cannot depend on itself")
            if dependencies:
                placeholders = ",".join("?" for _ in dependencies)
                rows = conn.execute(
                    "SELECT package_id FROM work_packages WHERE run_id=? AND epoch_id=? "
                    f"AND package_id IN ({placeholders})",
                    [run_id, epoch_id, *dependencies],
                ).fetchall()
                known_dependencies = {str(item[0]) for item in rows}
                missing_dependencies = sorted(set(dependencies) - known_dependencies)
                if missing_dependencies:
                    raise WorkPackageStateError(
                        "work package references unknown dependency(ies): "
                        + ", ".join(missing_dependencies)
                    )
            conn.execute(
                "INSERT INTO work_packages "
                "(package_id,run_id,epoch_id,phase_id,objective,path_scope_json,requirement_ids_json,dependencies_json,"
                "acceptance_json,required_tests_json,prohibited_paths_json,contract_digest,contract_version,"
                "prompt_contract_json,prompt_contract_digest,display_name,summary,risk,can_run_parallel,status,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(package_id) DO UPDATE SET objective=excluded.objective,path_scope_json=excluded.path_scope_json,"
                "requirement_ids_json=excluded.requirement_ids_json,dependencies_json=excluded.dependencies_json,"
                "acceptance_json=excluded.acceptance_json,required_tests_json=excluded.required_tests_json,"
                "prohibited_paths_json=excluded.prohibited_paths_json,contract_digest=excluded.contract_digest,"
                "contract_version=excluded.contract_version,prompt_contract_json=excluded.prompt_contract_json,"
                "prompt_contract_digest=excluded.prompt_contract_digest,display_name=excluded.display_name,"
                "summary=excluded.summary,risk=excluded.risk,can_run_parallel=excluded.can_run_parallel,"
                "updated_at=excluded.updated_at",
                (package_id, run_id, epoch_id, phase_id, objective.strip(), json.dumps(path_scope or [], separators=(",", ":")),
                 json.dumps(requirement_ids or [], separators=(",", ":")), json.dumps(dependencies, separators=(",", ":")),
                 json.dumps(acceptance or [], separators=(",", ":")), json.dumps(required_tests or [], separators=(",", ":")),
                 json.dumps(prohibited_paths or [], separators=(",", ":")), digest, contract_version,
                 json.dumps(prompt_contract, sort_keys=True, separators=(",", ":")), prompt_contract_digest,
                 display_name or objective.strip(), summary or objective.strip(), risk, 1 if can_run_parallel else 0,
                 "ready", now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM work_packages WHERE package_id=?", (package_id,)).fetchone()
            assert row is not None
            return self._package_row(row)
        finally:
            conn.close()

    @staticmethod
    def _package_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        for column in (
            "path_scope_json", "requirement_ids_json", "dependencies_json", "acceptance_json",
            "required_tests_json", "prohibited_paths_json", "prompt_contract_json",
        ):
            key = column.removesuffix("_json")
            try:
                result[key] = json.loads(result.get(column) or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                result[key] = []
        if result.get("prompt_contract_json"):
            try:
                result["prompt_contract"] = json.loads(result["prompt_contract_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                result["prompt_contract"] = {}
        return result

    def get_work_package(self, package_id: str) -> dict[str, Any] | None:
        conn = self._new_conn()
        try:
            row = conn.execute("SELECT * FROM work_packages WHERE package_id=?", (package_id,)).fetchone()
            return self._package_row(row) if row is not None else None
        finally:
            conn.close()

    def get_work_packages(self, run_id: str, epoch_id: str, phase_id: str | None = None) -> list[dict[str, Any]]:
        conn = self._new_conn()
        try:
            if phase_id is None:
                rows = conn.execute(
                    "SELECT * FROM work_packages WHERE run_id=? AND epoch_id=? ORDER BY created_at, package_id",
                    (run_id, epoch_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM work_packages WHERE run_id=? AND epoch_id=? AND phase_id=? ORDER BY created_at, package_id",
                    (run_id, epoch_id, phase_id),
                ).fetchall()
            return [self._package_row(row) for row in rows]
        finally:
            conn.close()

    def get_ready_work_packages(self, run_id: str, epoch_id: str, phase_id: str) -> list[dict[str, Any]]:
        packages = self.get_work_packages(run_id, epoch_id, phase_id)
        by_id = {str(item["package_id"]): item for item in self.get_work_packages(run_id, epoch_id)}
        phase_by_id = {
            str(item.get("phase_id")): item
            for item in self.get_workflow_phases(run_id, epoch_id)
        }
        result: list[dict[str, Any]] = []
        for package in packages:
            if package.get("status") not in {"ready", "retry"}:
                continue
            dependencies_ready = True
            for dep in package.get("dependencies", []):
                dependency = by_id.get(str(dep))
                if dependency is None:
                    dependencies_ready = False
                    break
                dependency_phase = phase_by_id.get(str(dependency.get("phase_id"))) or {}
                dependency_status = str(dependency.get("status") or "")
                # A mutating package is not a safe dependency until its
                # changeset has been integrated into the canonical workspace.
                # Read-only package outputs may be consumed once their worker
                # has completed and its evidence is available.
                required_statuses = (
                    {"integrated"}
                    if bool(dependency_phase.get("mutating"))
                    else {"completed", "integrated"}
                )
                if dependency_status not in required_statuses:
                    dependencies_ready = False
                    break
            if dependencies_ready:
                result.append(package)
        return result

    def update_work_package(self, package_id: str, *, status: str, reason: str = "") -> dict[str, Any] | None:
        if status not in {"ready", "claimed", "running", "completed", "integrated", "blocked", "retry", "cancelled"}:
            raise WorkPackageStateError(f"invalid work-package status: {status}")
        conn = self._new_conn()
        try:
            conn.execute(
                "UPDATE work_packages SET status=?, status_reason=?, updated_at=? WHERE package_id=?",
                (status, reason[:2000], _utcnow(), package_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM work_packages WHERE package_id=?", (package_id,)).fetchone()
            return self._package_row(row) if row is not None else None
        finally:
            conn.close()
