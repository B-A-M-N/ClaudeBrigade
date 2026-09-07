"""Workflow phase state machine, split out of state.py.

Twelfth increment of the incremental extraction out of ``RouteState``.
``WorkflowPhaseStateError`` moves here too (rather than staying in
state.py) since it's the phase state machine's primary exception;
``state.py`` re-exports it under its original name (``from
enhanced_router.state import WorkflowPhaseStateError``) so existing
importers (``mcp_control.py``, tests) are unaffected.

This is the phase DAG/transition engine: dependency satisfaction, mutation
exclusivity, quality quorum, deadlines, and turn budgets are all enforced
here. Several methods call back into ``self.skip_conditional_phase`` and
``self.get_agent_executions`` -- both still defined directly on
``RouteState`` at the time of this extraction -- which keeps working
unchanged via the mixin's normal method resolution order, exactly like the
cross-section calls in every earlier increment.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_actor(actor: str | None) -> str:
    """Normalize legacy ``<role>-agent`` labels before exact comparison."""
    value = str(actor or "").strip().lower()
    if value.endswith("-agent"):
        return value[:-len("-agent")]
    return value


_QUALITY_GATED_EXECUTION_KINDS = frozenset(
    {"native_agent", "subagent", "agent", "coprocessor_call", "sidecar_call"}
)


class WorkflowPhaseStateError(Exception):
    """Raised when a workflow phase transition is invalid."""


class WorkflowPhaseRepository(RepositoryMixin):
    """Mixin providing the workflow phase state machine.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does).
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def initialize_workflow_phases(
        self, run_id: str, epoch_id: str, phases: list[dict],
    ) -> list[dict]:
        """Create phase rows from a list of phase dicts (from WorkflowSpec.phases).

        Each phase dict has keys: id, roles, required, mutation, depends_on, conditional.
        Persists full phase semantics including dependencies, roles, mutation flag, and
        a specification hash for immutability verification.
        Is idempotent: if rows already exist for (run_id, epoch_id), returns existing.
        """
        import hashlib
        import json

        conn = self._new_conn()
        try:
            if not phases:
                return []

            existing = conn.execute(
                "SELECT phase_id FROM workflow_phases WHERE run_id=? AND epoch_id=?",
                (run_id, epoch_id),
            ).fetchall()
            if existing:
                return self.get_workflow_phases(run_id, epoch_id)

            for phase in phases:
                canonical = json.dumps({k: v for k, v in sorted(phase.items())}, sort_keys=True)
                spec_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                conn.execute(
                    """INSERT INTO workflow_phases
                       (run_id, epoch_id, phase_id, status, actor, required, mutating,
                       allowed_roles_json, dependencies_json, condition_json, parallel_group,
                        specification_hash, ordinal, distinct_agent_from_json, max_duration_seconds,
                        turn_budget, provider_requirements_json, min_fanout, max_fanout, result_schema,
                        quality_quorum, fallback_policy, execution_kind, required_actor,
                        max_parallelism, required_successes, max_attempts, max_attempts_per_model, sidecar_id,
                        sidecar_agent_id, coprocessor_id, produces, fanout_from)
                       VALUES (
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?
                       )""",
                    (
                        run_id, epoch_id, phase["id"], "pending",
                        phase.get("actor", ""),
                        1 if phase.get("required", True) else 0,
                        1 if phase.get("mutation", False) else 0,
                        json.dumps(phase.get("roles", [])),
                        json.dumps(phase.get("depends_on", [])),
                        json.dumps(
                            phase.get("condition")
                            if phase.get("condition") is not None
                            else phase.get("conditional")
                        ) if phase.get("condition") is not None or phase.get("conditional") else None,
                        phase.get("parallel_group"),
                        spec_hash,
                        phase.get("ordinal"), json.dumps(phase.get("distinct_agent_from", [])),
                        phase.get("max_duration_seconds"), phase.get("turn_budget"),
                        json.dumps(phase.get("provider_requirements", [])), phase.get("min_fanout", 1),
                        phase.get("max_fanout", 1), phase.get("result_schema"),
                        phase.get("quality_quorum", 1), phase.get("fallback_policy"),
                        phase.get("execution_kind", "native_agent"), phase.get("actor", ""),
                        phase.get("max_parallelism"), phase.get("required_successes"),
                        phase.get("max_attempts") or phase.get("max_fanout", 1), phase.get("max_attempts_per_model"),
                        phase.get("sidecar") or phase.get("sidecar_id"),
                        phase.get("sidecar_agent") or phase.get("sidecar_agent_id"),
                        phase.get("coprocessor") or phase.get("coprocessor_id"),
                        phase.get("produces"), phase.get("fanout_from"),
                    ),
                )
                conn.execute(
                    "UPDATE workflow_phases SET agent_id=? "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (phase.get("agent_id"), run_id, epoch_id, phase["id"]),
                )
                conn.execute(
                    "UPDATE workflow_phases SET completion_mode=?, minimum_quality_score=?, "
                    "requires_controller_acceptance=?, initial_fanout=?, maximum_replicas=?, "
                    "hedge_delay_seconds=?, launch_policy=? "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (
                        phase.get("completion_mode", "quorum"), phase.get("minimum_quality_score"),
                        1 if phase.get("requires_controller_acceptance", True) else 0,
                        phase.get("initial_fanout"), phase.get("maximum_replicas"),
                        phase.get("hedge_delay_seconds"), phase.get("launch_policy", "minimum_first"),
                        run_id, epoch_id, phase["id"],
                    ),
                )
                if phase.get("result_contract") is not None:
                    conn.execute(
                        "UPDATE workflow_phases SET result_contract_json=? "
                        "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                        (
                            json.dumps(phase["result_contract"], sort_keys=True, separators=(",", ":")),
                            run_id, epoch_id, phase["id"],
                        ),
                    )
            conn.commit()
            return self.get_workflow_phases(run_id, epoch_id)
        finally:
            conn.close()

    def get_workflow_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return all phases for an epoch, ordered by phase_id."""
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM workflow_phases WHERE run_id=? AND epoch_id=? ORDER BY id",
                (run_id, epoch_id),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_active_phase(self, run_id: str, epoch_id: str) -> dict | None:
        """Return the currently active phase, or None."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=? AND status='active' ORDER BY id LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_active_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return all active phases; parallel read-only phases are valid."""
        return [
            phase for phase in self.get_workflow_phases(run_id, epoch_id)
            if phase.get("status") == "active"
        ]

    def apply_fastpath_plan(
        self,
        run_id: str,
        epoch_id: str,
        proposal: dict,
    ) -> dict:
        """Apply a validated fastpath fanout/parallelism hint before launch.

        Fastpath remains advisory: this method only changes still-pending phase
        launch parameters, never routes or completion evidence.  Once an
        action claim or execution exists for the epoch, the plan is stale and
        is rejected rather than silently changing the shape of an active run.
        """
        raw_routes = proposal.get("resolved_routes") or proposal.get("routes") or {}
        if not isinstance(raw_routes, dict):
            raise ValueError("fastpath proposal routes must be an object")
        parallel_groups = proposal.get("parallel_groups") or []
        if not isinstance(parallel_groups, list):
            raise ValueError("fastpath proposal parallel_groups must be a list")

        role_fanout: dict[str, int] = {}
        for role, target in raw_routes.items():
            if not isinstance(target, dict):
                continue
            try:
                fanout = int(target.get("fanout") or 1)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid fastpath fanout for role '{role}'") from exc
            if fanout < 1 or fanout > 4:
                raise ValueError(f"fastpath fanout for role '{role}' must be between 1 and 4")
            role_fanout[str(role)] = fanout

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            epoch = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if epoch is None:
                raise ValueError("active epoch not found for fastpath plan")
            bound = conn.execute(
                "SELECT 1 FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND status IN ('claimed','consumed') LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            execution = conn.execute(
                "SELECT 1 FROM agent_executions WHERE run_id=? AND epoch_id=? LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            if bound is not None or execution is not None:
                raise ValueError(
                    "fastpath plan is stale: work is already bound in this epoch"
                )

            phases = conn.execute(
                "SELECT phase_id, status, actor, allowed_roles_json, min_fanout, "
                "max_fanout, initial_fanout, maximum_replicas, max_parallelism, "
                "parallel_group FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=? ORDER BY id",
                (run_id, epoch_id),
            ).fetchall()
            changes: list[dict] = []
            group_for_role: dict[str, str] = {}
            for index, group in enumerate(parallel_groups):
                if not isinstance(group, list):
                    raise ValueError("fastpath parallel groups must contain role lists")
                for role in group:
                    group_for_role[str(role)] = f"fastpath-group-{index + 1}"

            for phase in phases:
                if phase[1] != "pending" or phase[2]:
                    continue
                try:
                    allowed_roles = json.loads(phase[3] or "[]")
                except (TypeError, ValueError, json.JSONDecodeError):
                    allowed_roles = []
                matched = [str(role) for role in allowed_roles if str(role) in role_fanout]
                if not matched:
                    continue
                fanout = max(role_fanout[role] for role in matched)
                current_min = int(phase[4] or 1)
                current_max = int(phase[5] or current_min)
                current_initial = int(phase[6] or current_min)
                current_replicas = int(phase[7] or current_max)
                current_parallelism = int(phase[8] or 1)
                group = next(
                    (group_for_role[role] for role in matched if role in group_for_role),
                    phase[9],
                )
                new_values = (
                    max(current_min, fanout),
                    max(current_max, fanout),
                    max(current_initial, fanout),
                    max(current_replicas, fanout),
                    max(current_parallelism, fanout),
                    group,
                )
                if new_values == (
                    current_min, current_max, current_initial, current_replicas,
                    current_parallelism, phase[9],
                ):
                    continue
                conn.execute(
                    "UPDATE workflow_phases SET min_fanout=?, max_fanout=?, "
                    "initial_fanout=?, maximum_replicas=?, max_parallelism=?, "
                    "parallel_group=? WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (*new_values, run_id, epoch_id, phase[0]),
                )
                changes.append({
                    "phase_id": phase[0],
                    "roles": matched,
                    "fanout": fanout,
                    "parallel_group": group,
                })
            conn.commit()
            return {"applied": True, "changes": changes}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append_phase_instance(
        self,
        run_id: str,
        epoch_id: str,
        template_phase_id: str,
        *,
        dependencies: list[str] | None = None,
        supersedes_phase_id: str | None = None,
        trigger_event: str | None = None,
        phase_id: str | None = None,
    ) -> dict:
        """Append a fresh semantic instance of a persisted phase template.

        A retry inside one execution is not a new review/repair cycle.  This
        operation creates a new phase identity, preserving the immutable
        template semantics while recording the iteration, predecessor, and
        trigger that caused the cycle.  It is intentionally controller-owned
        through MCP; workers cannot grow the workflow graph themselves.
        """
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            template = conn.execute(
                "SELECT * FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=? "
                "AND (phase_id=? OR phase_template_id=?) "
                "ORDER BY CASE WHEN phase_id=? THEN 0 ELSE 1 END, iteration DESC, id DESC LIMIT 1",
                (run_id, epoch_id, template_phase_id, template_phase_id, template_phase_id),
            ).fetchone()
            if template is None:
                raise WorkflowPhaseStateError(
                    f"phase template '{template_phase_id}' was not found"
                )
            template_row = dict(template)
            base_id = str(template_row.get("phase_template_id") or template_phase_id)
            latest = conn.execute(
                "SELECT COALESCE(MAX(iteration), 0) FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=? AND (phase_id=? OR phase_template_id=?)",
                (run_id, epoch_id, base_id, base_id),
            ).fetchone()
            iteration = int(latest[0] or 0) + 1
            new_phase_id = phase_id or f"{base_id}#{iteration}"
            if conn.execute(
                "SELECT 1 FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, new_phase_id),
            ).fetchone() is not None:
                raise WorkflowPhaseStateError(
                    f"phase instance '{new_phase_id}' already exists"
                )

            dependency_ids = (
                [str(item) for item in dependencies]
                if dependencies is not None
                else json.loads(template_row.get("dependencies_json") or "[]")
            )
            ordinal_row = conn.execute(
                "SELECT COALESCE(MAX(ordinal), 0) FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=?",
                (run_id, epoch_id),
            ).fetchone()
            values = {
                key: template_row.get(key)
                for key in template.keys()
                if key != "id"
            }
            values.update({
                "phase_id": new_phase_id,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "actor": "",
                "result_evidence": None,
                "error": None,
                "dependencies_json": json.dumps(dependency_ids, separators=(",", ":")),
                "ordinal": int(ordinal_row[0] or 0) + 1,
                "phase_template_id": base_id,
                "iteration": iteration,
                "supersedes_phase_id": supersedes_phase_id,
                "trigger_event": trigger_event,
            })
            # Older databases may lack newly introduced nullable lifecycle
            # fields until migrations run.  Restrict the insert to columns
            # that actually exist so a phase-cycle request remains safely
            # compatible with an upgraded-but-not-restarted router.
            columns = [row[1] for row in conn.execute(
                "PRAGMA table_info(workflow_phases)"
            ).fetchall() if row[1] != "id" and row[1] in values]
            placeholders = ",".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO workflow_phases ({','.join(columns)}) VALUES ({placeholders})",
                [values[column] for column in columns],
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, new_phase_id),
            ).fetchone()
            assert row is not None
            return dict(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_ready_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Return pending phases whose persisted dependencies are satisfied."""
        phases = self.get_workflow_phases(run_id, epoch_id)
        ready: list[dict] = []
        for phase in phases:
            if phase.get("status") != "pending":
                continue
            dependencies = json.loads(phase.get("dependencies_json") or "[]")
            phase_by_id = {item["phase_id"]: item for item in phases}
            if all(self._dependency_satisfied(phase_by_id.get(dep)) for dep in dependencies):
                ready.append(phase)
        return ready

    @staticmethod
    def _dependency_satisfied(phase: dict | None) -> bool:
        """Return whether a persisted phase may satisfy a dependency."""
        if phase is None:
            return False
        if phase.get("status") == "completed":
            return True
        if phase.get("status") != "skipped":
            return False
        try:
            evidence = json.loads(phase.get("result_evidence") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return evidence.get("skip_type") == "conditional"

    def advance_conditional_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Apply deterministic conditional skips and result branches."""
        changed: list[dict] = []
        findings = self.get_findings(run_id, epoch_id=epoch_id)
        has_accepted_findings = any(item.get("disposition") == "accepted" for item in findings)
        for phase in self.get_ready_phases(run_id, epoch_id):
            if phase.get("condition_json") is None:
                continue
            condition = json.loads(phase["condition_json"])
            if condition == "accepted_findings":
                condition = {"kind": "accepted_findings"}
            if not isinstance(condition, dict):
                continue
            kind = condition.get("kind")
            satisfied = False
            if kind == "accepted_findings":
                satisfied = has_accepted_findings
            elif kind in {"phase_result_equals", "phase_result_in"}:
                source = next(
                    (item for item in self.get_workflow_phases(run_id, epoch_id)
                     if item.get("phase_id") == condition.get("phase_id")),
                    None,
                )
                value = self._phase_result_field(run_id, epoch_id, source, condition.get("field"))
                satisfied = (
                    value == condition.get("value")
                    if kind == "phase_result_equals"
                    else value in set(condition.get("values") or [])
                )
            elif kind == "workflow_tier_at_least":
                tiers = ("trivial", "normal", "cross-cutting", "high-risk")
                requested = str(condition.get("value") or "")
                escalation = self.get_escalation_state(run_id, epoch_id)
                current = str(
                    escalation.get("escalation_level")
                    or escalation.get("workflow_id")
                    or "normal"
                )
                satisfied = (
                    requested in tiers
                    and current in tiers
                    and tiers.index(current) >= tiers.index(requested)
                )
            elif kind == "explicit_escalation":
                escalation = self.get_escalation_state(run_id, epoch_id)
                expected = str(condition.get("value") or "escalated")
                satisfied = str(escalation.get("escalation_state") or "") == expected
            elif kind == "workspace_generation_changed":
                source = next(
                    (item for item in self.get_workflow_phases(run_id, epoch_id)
                     if item.get("phase_id") == condition.get("phase_id")),
                    None,
                )
                workspaces = self.get_workspaces(run_id=run_id, epoch_id=epoch_id, kind="main")
                current_generation = int(
                    (next(
                        (item for item in workspaces
                         if item.get("status") in {"active", "ready", "merged"}),
                        {},
                    ).get("canonical_generation") or 0)
                )
                source_generation = (
                    int(str(source.get("evidence_generation")))
                    if source and source.get("evidence_generation") is not None
                    else None
                )
                satisfied = source_generation is not None and current_generation != source_generation
            if not satisfied:
                changed.append(self.skip_conditional_phase(
                    run_id, epoch_id, phase["phase_id"],
                    str(kind or "condition"),
                    reason=f"conditional {kind or 'condition'} was not satisfied",
                ))
        return changed

    def _phase_result_field(
        self, run_id: str, epoch_id: str, phase: dict | None, field: str | None,
    ) -> object | None:
        if phase is None or not field:
            return None
        candidates: list[object] = []
        try:
            if phase.get("result_evidence"):
                candidates.append(json.loads(str(phase["result_evidence"])))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        for execution in reversed(self.get_agent_executions(
            run_id, epoch_id=epoch_id, phase_id=str(phase.get("phase_id")),
        )):
            try:
                if execution.get("result_json"):
                    candidates.append(json.loads(str(execution["result_json"])))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        for candidate in candidates:
            if isinstance(candidate, dict) and field in candidate:
                return candidate[field]
        return None

    def prepare_agent_phase(
        self, run_id: str, epoch_id: str, role: str, agent_id: str,
    ) -> dict | None:
        """Start or join the ready phase that authorizes a native agent role."""
        self.advance_conditional_phases(run_id, epoch_id)
        phases = self.get_workflow_phases(run_id, epoch_id)
        active = [
            phase for phase in phases
            if phase.get("status") == "active"
            and (
                role in json.loads(phase.get("allowed_roles_json") or "[]")
                or (role == "controller" and phase.get("actor") == "controller")
            )
        ]
        if active:
            return sorted(active, key=lambda item: (item.get("ordinal") or 0, item["phase_id"]))[0]
        ready = [
            phase for phase in self.get_ready_phases(run_id, epoch_id)
            if (
                role in json.loads(phase.get("allowed_roles_json") or "[]")
                or (role == "controller" and phase.get("actor") == "controller")
            )
            and (not phase.get("actor") or (role == "controller" and phase.get("actor") == "controller"))
        ]
        if not ready:
            raise WorkflowPhaseStateError(
                f"no ready workflow phase permits role '{role}'"
            )
        phase = sorted(ready, key=lambda item: (item.get("ordinal") or 0, item["phase_id"]))[0]
        max_duration = phase.get("max_duration_seconds")
        if max_duration and phase.get("started_at"):
            started = datetime.fromisoformat(str(phase["started_at"]))
            if (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds() > max_duration:
                self.complete_phase(run_id, epoch_id, phase["phase_id"], error="phase deadline exceeded")
                raise WorkflowPhaseStateError(f"phase '{phase['phase_id']}' deadline exceeded")
        return self.start_phase(run_id, epoch_id, phase["phase_id"], actor=role)

    def complete_phase_if_ready(self, run_id: str, epoch_id: str, phase_id: str) -> dict | None:
        """Complete a phase after all assigned executions reach terminal state."""
        phase = next(
            (item for item in self.get_workflow_phases(run_id, epoch_id) if item["phase_id"] == phase_id),
            None,
        )
        if phase is None or phase.get("status") != "active":
            return phase
        executions = self.get_agent_executions(run_id, epoch_id=epoch_id, phase_id=phase_id)
        if not executions or any(
            item.get("status") in {"started", "running", "streaming", "verifying"}
            for item in executions
        ):
            return phase
        completed = [item for item in executions if item.get("status") == "completed"]
        worker_executions = [
            item for item in completed
            if str(item.get("execution_kind") or "") in _QUALITY_GATED_EXECUTION_KINDS
        ]
        contract = self._result_contract(phase)
        awaiting_adjudication = [
            item for item in worker_executions
            if item.get("accepted_by_controller") not in {True, 1}
            and item.get("result_disposition") is None
        ]
        if awaiting_adjudication:
            # A router-owned response is terminal transport state, not
            # workflow evidence.  Leave the phase active until the main
            # controller explicitly adjudicates it.
            return phase
        requires_acceptance = bool(phase.get("requires_controller_acceptance"))
        minimum_quality = phase.get("minimum_quality_score")
        successful = [
            item for item in completed
            if (
                item.get("schema_valid") in {True, 1}
                if phase.get("result_schema") else True
            )
            and item.get("evidence_valid") in {True, 1}
            and (
                item.get("accepted_by_controller") in {True, 1}
                if str(item.get("execution_kind") or "")
                in _QUALITY_GATED_EXECUTION_KINDS
                else not requires_acceptance or item.get("accepted_by_controller") in {True, 1}
            )
            and (
                not contract
                or item.get("schema_valid") in {True, 1}
            )
            and (minimum_quality is None or float(item.get("quality_score") or 0) >= float(minimum_quality))
            and self._execution_matches_result_contract(item, contract)
            and self._execution_evidence_is_current(run_id, epoch_id, item, contract)
        ]
        required_successes = int(
            phase.get("required_successes")
            or max(int(phase.get("min_fanout") or 1), int(phase.get("quality_quorum") or 1))
        )
        if len(successful) >= required_successes:
            completion_mode = str(phase.get("completion_mode") or "quorum")
            if completion_mode == "all_packages":
                packages = self.get_work_packages(run_id, epoch_id, phase_id)
                package_ids = {
                    str(item.get("package_id"))
                    for item in packages
                    if item.get("package_id")
                }
                successful_package_ids = {
                    str(item.get("package_id"))
                    for item in successful
                    if item.get("package_id")
                }
                incomplete_packages = [
                    str(item.get("package_id"))
                    for item in packages
                    if str(item.get("status") or "") not in {"completed", "integrated"}
                ]
                # Package fanout is an explicit completeness contract.  A
                # quorum of workers must not allow one package to stand in for
                # another, and a terminal execution is not enough while its
                # package remains retryable/blocked.
                if (
                    not package_ids
                    or package_ids - successful_package_ids
                    or incomplete_packages
                ):
                    return phase
            return self.complete_phase(
                run_id, epoch_id, phase_id,
                result_evidence=json.dumps({
                    "execution_ids": [item["execution_id"] for item in executions],
                    "completed": len(completed),
                    "successful": len(successful),
                }, separators=(",", ":")),
            )

        failed = [item for item in executions if item.get("status") != "completed"]
        max_attempts = int(phase.get("max_attempts") or phase.get("max_fanout") or 1)
        fallback_policy = str(phase.get("fallback_policy") or "").strip()
        if fallback_policy and len(executions) < max_attempts:
            # Keep the phase active so the scheduler can materialize the next
            # retry/fallback action.  Failure evidence remains in the ledger.
            return phase
        if failed or len(executions) >= max_attempts:
            if (
                failed
                and len(executions) < max_attempts
                and any(
                    str(item.get("execution_kind") or "")
                    in {"coprocessor_call", "sidecar_call"}
                    for item in executions
                )
            ):
                # Router-owned calls may be retried/fail over by the
                # coprocessor scheduler before the phase is failed.
                return phase
            first_failure = failed[0] if failed else executions[-1]
            return self.complete_phase(
                run_id, epoch_id, phase_id,
                error=f"execution failure: {first_failure.get('execution_id')}",
            )
        return phase

    @staticmethod
    def _result_contract(phase: dict) -> dict:
        try:
            value = json.loads(str(phase.get("result_contract_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _execution_matches_result_contract(execution: dict, contract: dict) -> bool:
        if not contract:
            return True
        try:
            result = json.loads(str(execution.get("result_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(result, dict):
            return False
        verdict = result.get("verdict", result.get("status", result.get("completion")))
        if verdict is None:
            return False
        success = {str(item) for item in contract.get("success_verdicts") or []}
        branch = {str(item) for item in contract.get("branch_verdicts") or []}
        blocked = {str(item) for item in contract.get("blocked_verdicts") or []}
        failure = {str(item) for item in contract.get("failure_verdicts") or []}
        if str(verdict) in blocked or str(verdict) in failure:
            return False
        return not success or str(verdict) in success or str(verdict) in branch

    def _execution_evidence_is_current(
        self, run_id: str, epoch_id: str, execution: dict, contract: dict,
    ) -> bool:
        if not contract.get("requires_current_generation") and not contract.get("requires_current_digest"):
            return True
        workspaces = self.get_workspaces(run_id=run_id, epoch_id=epoch_id, kind="main")
        active = next(
            (item for item in workspaces if item.get("status") in {"active", "ready", "merged"}),
            None,
        )
        if active is None:
            return False
        if contract.get("requires_current_generation"):
            if execution.get("workspace_generation") is None:
                return False
            if int(execution["workspace_generation"]) != int(active.get("canonical_generation") or 0):
                return False
        if contract.get("requires_current_digest"):
            expected = active.get("current_dirty_hash") or active.get("dirty_patch_hash")
            if not expected or str(execution.get("workspace_digest") or "") != str(expected):
                return False
        return True

    def start_phase(
        self, run_id: str, epoch_id: str, phase_id: str, actor: str = "",
        principal: str = "",
    ) -> dict:
        """Start a phase: set status='active', record started_at.

        All dependency and actor semantics are read from the immutable
        persisted phase instance.  Callers cannot provide a replacement
        phase definition at transition time.
        """
        conn = self._new_conn()
        try:
            # Check phase exists and is pending
            row = conn.execute(
                "SELECT id, status, dependencies_json, required_actor, mutating, "
                "phase_template_id, phase_id "
                "FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[1] != "pending":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[1]}, cannot start (must be 'pending')"
                )
            required_actor = str(row[3] or "")
            if required_actor and _canonical_actor(actor) != _canonical_actor(required_actor):
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' requires actor '{required_actor}', got '{actor}'"
                )
            if row[4]:
                active_mutation = conn.execute(
                    "SELECT phase_id FROM workflow_phases WHERE run_id=? AND epoch_id=? "
                    "AND status='active' AND mutating=1 LIMIT 1", (run_id, epoch_id)
                ).fetchone()
                if active_mutation is not None:
                    raise WorkflowPhaseStateError(
                        f"mutating phase '{active_mutation[0]}' is already active"
                    )

            # Reclassify the canonical result immediately before verification
            # gates.  This closes the gap where a late repair/integration was
            # persisted after the normal changeset hook but before the
            # controller started final review.
            gate_name = str(row[5] or row[6] or "").lower()
            if not row[4] and any(
                marker in gate_name for marker in ("verification", "final-audit", "completion")
            ):
                reclassify = getattr(self, "reclassify_before_gate", None)
                if callable(reclassify):
                    result = reclassify(run_id, epoch_id, gate=phase_id)
                    if isinstance(result, dict) and result.get("applied"):
                        raise WorkflowPhaseStateError(
                            "workflow tier escalated before verification; "
                            "controller must re-query the compensating phases"
                        )

            # Dependencies come from the immutable persisted phase snapshot;
            # caller-supplied definitions cannot bypass the DAG.
            dependencies = json.loads(row[2] or "[]")
            phase_rows = conn.execute(
                "SELECT phase_id, status, result_evidence FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=?",
                (run_id, epoch_id),
            ).fetchall()
            phase_by_id = {str(item[0]): dict(item) for item in phase_rows}
            unsatisfied = [
                dependency for dependency in dependencies
                if not self._dependency_satisfied(phase_by_id.get(dependency))
            ]
            if unsatisfied:
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' dependencies are not satisfied: {', '.join(unsatisfied)}"
                )

            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status='active', started_at=?, actor=?, "
                "started_by_actor=?, started_by_principal=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (now, actor, actor, principal or None, run_id, epoch_id, phase_id),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()

    def complete_phase(
        self, run_id: str, epoch_id: str, phase_id: str,
        result_evidence: str = "", error: str = "",
    ) -> dict:
        """Complete a phase: set status='completed' (or 'failed' if error given).

        Only 'active' phases can be completed. Auto-sets status to 'failed' when
        error is non-empty.
        """
        new_status = "failed" if error else "completed"
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT status FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[0] != "active":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[0]}, cannot complete (must be 'active')"
                )

            phase_row = conn.execute(
                "SELECT allowed_roles_json, min_fanout, max_fanout, quality_quorum, "
                "started_at, distinct_agent_from_json, max_duration_seconds, result_schema, "
                "requires_controller_acceptance, minimum_quality_score, turn_budget, max_attempts, "
                "produces, completion_mode "
                "FROM workflow_phases "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?", (run_id, epoch_id, phase_id)
            ).fetchone()
            if phase_row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if new_status == "completed" and not result_evidence.strip():
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' requires result_evidence before completion"
                )
            if new_status == "completed" and phase_row[6] and phase_row[4]:
                started = datetime.fromisoformat(str(phase_row[4]))
                elapsed = (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
                if elapsed > int(phase_row[6]):
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded its {phase_row[6]} second deadline"
                    )
            if new_status == "completed" and phase_row[7]:
                try:
                    schema = json.loads(str(phase_row[7]))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_schema is invalid JSON"
                    ) from exc
                try:
                    parsed_evidence = json.loads(result_evidence)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_evidence is not valid JSON"
                    ) from exc
                required_keys = schema.get("required", []) if isinstance(schema, dict) else []
                if not isinstance(parsed_evidence, dict) or not isinstance(required_keys, list):
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_evidence does not match its schema"
                    )
                missing = sorted(str(key) for key in required_keys if key not in parsed_evidence)
                if missing:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' result_evidence is missing: {', '.join(missing)}"
                    )
            if new_status == "completed":
                contract_row = conn.execute(
                    "SELECT result_contract_json FROM workflow_phases "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()
                if contract_row and contract_row[0]:
                    try:
                        contract = json.loads(str(contract_row[0]))
                        evidence = json.loads(result_evidence)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise WorkflowPhaseStateError(
                            f"Phase '{phase_id}' result does not satisfy its result contract"
                        ) from exc
                    verdict = evidence.get("verdict", evidence.get("status")) if isinstance(evidence, dict) else None
                    success = {str(item) for item in contract.get("success_verdicts") or []}
                    branch = {str(item) for item in contract.get("branch_verdicts") or []}
                    blocked = {str(item) for item in contract.get("blocked_verdicts") or []}
                    failure = {str(item) for item in contract.get("failure_verdicts") or []}
                    if verdict is not None and (str(verdict) in blocked or str(verdict) in failure or (success and str(verdict) not in success and str(verdict) not in branch)):
                        raise WorkflowPhaseStateError(
                            f"Phase '{phase_id}' cannot complete with verdict '{verdict}'"
                        )
            if new_status == "completed" and str(phase_row[12] or "") == "work_packages":
                package_count = conn.execute(
                    "SELECT COUNT(*) FROM work_packages "
                    "WHERE run_id=? AND epoch_id=? AND phase_id<>?",
                    (run_id, epoch_id, phase_id),
                ).fetchone()[0]
                if int(package_count or 0) == 0:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' must publish at least one downstream work package"
                    )
            executions = conn.execute(
                "SELECT claude_agent_id, independence_key, status, tool_call_count, "
                "schema_valid, evidence_valid, accepted_by_controller, quality_score "
                ", package_id "
                "FROM agent_executions "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?", (run_id, epoch_id, phase_id)
            ).fetchall()
            if new_status == "completed" and phase_row[10] and executions:
                observed_turns = sum(int(item[3] or 0) for item in executions)
                if observed_turns > int(phase_row[10]):
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded turn budget {phase_row[10]}"
                    )
            if new_status == "completed" and executions:
                completed = [item for item in executions if item[2] == "completed"]
                quality_eligible = [
                    item for item in completed
                    if (not phase_row[7] or item[4] == 1)
                    and item[5] == 1
                    and item[6] == 1
                    and (
                        phase_row[9] is None
                        or item[7] is not None and float(item[7]) >= float(phase_row[9])
                    )
                ]
                min_fanout = int(phase_row[1] or 1)
                max_fanout = int(phase_row[2] or 1)
                quorum = int(phase_row[3] or 1)
                max_attempts = int(phase_row[11] or max_fanout)
                if len(executions) > max_attempts:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded max attempts {max_attempts}"
                    )
                if len(completed) < min_fanout or len(quality_eligible) < quorum:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' requires {max(min_fanout, quorum)} quality-valid execution(s)"
                    )
                if str(phase_row[13] or "quorum") == "all_packages":
                    packages = conn.execute(
                        "SELECT package_id, status FROM work_packages "
                        "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                        (run_id, epoch_id, phase_id),
                    ).fetchall()
                    package_ids = {str(item[0]) for item in packages if item[0]}
                    successful_package_ids = {
                        str(item[8]) for item in quality_eligible if item[8]
                    }
                    incomplete = [
                        str(item[0]) for item in packages
                        if str(item[1] or "") not in {"completed", "integrated"}
                    ]
                    if (
                        not package_ids
                        or package_ids - successful_package_ids
                        or incomplete
                    ):
                        raise WorkflowPhaseStateError(
                            f"Phase '{phase_id}' requires accepted quality evidence for every work package"
                        )
                distinct_refs = json.loads(phase_row[5] or "[]")
                if distinct_refs:
                    prior = conn.execute(
                        "SELECT claude_agent_id, independence_key FROM agent_executions WHERE run_id=? AND epoch_id=? "
                        "AND phase_id IN (%s)" % ",".join("?" * len(distinct_refs)),
                        [run_id, epoch_id, *distinct_refs],
                    ).fetchall()
                    prior_keys = {item[1] for item in prior if item[1]}
                    prior_ids = {item[0] for item in prior if not item[1]}
                    if any(
                        (item[1] and item[1] in prior_keys)
                        or (not item[1] and item[0] in prior_ids)
                        for item in completed
                    ):
                        raise WorkflowPhaseStateError(
                            f"Phase '{phase_id}' violates distinct_agent_from"
                        )
            if new_status == "completed" and str(phase_row[13] or "quorum") == "all_packages" and not executions:
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' requires accepted quality evidence for every work package"
                )
            now = _utcnow()
            workspace = conn.execute(
                "SELECT canonical_generation, current_dirty_hash, dirty_patch_hash "
                "FROM workspaces WHERE run_id=? AND epoch_id=? AND kind='main' "
                "AND status IN ('active','ready','merged') ORDER BY created_at LIMIT 1",
                (run_id, epoch_id),
            ).fetchone()
            conn.execute(
                "UPDATE workflow_phases SET status=?, completed_at=?, result_evidence=?, error=?, "
                "evidence_generation=?, evidence_digest=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (
                    new_status, now, result_evidence, error,
                    int(workspace[0] or 0) if workspace is not None and new_status == "completed" else None,
                    str(workspace[1] or workspace[2] or "") if workspace is not None and new_status == "completed" else None,
                    run_id, epoch_id, phase_id,
                ),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()

    def skip_phase(
        self, run_id: str, epoch_id: str, phase_id: str, reason: str = "",
    ) -> dict:
        """Skip a non-required phase: set status='skipped'.

        Raises WorkflowPhaseStateError if phase is required.
        """
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT status, required FROM workflow_phases WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (run_id, epoch_id, phase_id),
            ).fetchone()
            if row is None:
                raise WorkflowPhaseStateError(f"Phase '{phase_id}' not found")
            if row[0] != "pending":
                raise WorkflowPhaseStateError(
                    f"Phase '{phase_id}' is {row[0]}, cannot skip (must be 'pending')"
                )
            if bool(row[1]):
                raise WorkflowPhaseStateError(
                    f"Required phase '{phase_id}' cannot be skipped"
                )

            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status='skipped', completed_at=?, result_evidence=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (now, json.dumps({"skip_type": "manual", "reason": reason}),
                 run_id, epoch_id, phase_id),
            )
            conn.commit()
            phases = self.get_workflow_phases(run_id, epoch_id)
            return next(p for p in phases if p["phase_id"] == phase_id)
        finally:
            conn.close()

    def validate_phase_transition(
        self, run_id: str, epoch_id: str, target_phase_id: str,
        phases_spec: list[dict] | None = None,
    ) -> dict:
        """Validate that *target_phase_id* can be started given current phase state.

        ``phases_spec`` is retained as a compatibility argument for older
        callers, but it is intentionally ignored.  Persisted phase instances
        are the only authority for dependencies and transition state.

        Returns dict: {"valid": True} or {"valid": False, "reason": "..."}

        Rules:
        1. Target phase must exist in phases_spec
        2. Target phase must be 'pending' (not already started/completed/skipped)
        3. All phases in depends_on must be 'completed'
        4. If a required dependency is 'failed', target cannot start
        """
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT phase_id, status, dependencies_json, result_evidence "
                "FROM workflow_phases WHERE run_id=? AND epoch_id=?",
                (run_id, epoch_id),
            ).fetchall()
            phase_rows = {str(row[0]): dict(row) for row in rows}
        finally:
            conn.close()

        # Check target is pending
        target = phase_rows.get(target_phase_id)
        if target is None:
            return {"valid": False, "reason": f"Phase '{target_phase_id}' has no state record"}
        if target["status"] != "pending":
            return {"valid": False, "reason": f"Phase '{target_phase_id}' is '{target['status']}', not 'pending'"}

        # Check dependencies
        for dep_id in json.loads(target.get("dependencies_json") or "[]"):
            dependency = phase_rows.get(dep_id)
            if dependency is None:
                return {"valid": False, "reason": f"Dependency phase '{dep_id}' has no state record"}
            if dependency["status"] == "failed":
                return {"valid": False, "reason": f"Dependency phase '{dep_id}' failed, cannot proceed"}
            if not self._dependency_satisfied(dependency):
                return {"valid": False, "reason": f"Dependency phase '{dep_id}' is not satisfied"}

        return {"valid": True}

    def ensure_workflow_phases(self, run_id: str, epoch_id: str) -> list[dict]:
        """Ensure workflow phases exist for the given run+epoch.

        Loads the WorkflowSpec from the registry (by workflow_id) and
        creates phase rows if they don't yet exist.
        """
        existing = self.get_workflow_phases(run_id, epoch_id)
        if existing:
            return existing

        # Get the epoch to find the workflow_id
        active = self.get_active_epoch(run_id)
        if not active or active.get("epoch_id") != epoch_id:
            return []

        workflow_id = active.get("workflow_id", "")
        if not workflow_id:
            return []

        # Load spec from registry
        from enhanced_router.registry import get_registry

        registry = get_registry()
        spec = registry.get_workflow(workflow_id)
        if spec is None:
            return []

        phases_data = [
            {
                "id": p.id,
                "roles": p.roles,
                "required": p.required,
                "mutation": p.mutation,
                "depends_on": p.depends_on,
                "conditional": p.conditional,
                "actor": p.actor or "",
                "distinct_agent_from": p.distinct_agent_from,
                "parallel_group": p.parallel_group,
                "ordinal": p.ordinal,
                "max_duration_seconds": p.max_duration_seconds,
                "turn_budget": p.turn_budget,
                "provider_requirements": p.provider_requirements,
                "min_fanout": p.min_fanout,
                "max_fanout": p.max_fanout,
                "result_schema": p.result_schema,
                "quality_quorum": p.quality_quorum,
                "fallback_policy": p.fallback_policy,
                "execution_kind": p.execution_kind,
            }
            for p in spec.phases
        ]

        return self.initialize_workflow_phases(run_id, epoch_id, phases_data)
