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


class WorkflowPhaseStateError(Exception):
    """Raised when a workflow phase transition is invalid."""


class WorkflowPhaseRepository:
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
                       max_parallelism, required_successes, max_attempts, max_attempts_per_model, sidecar_id)
                       VALUES (
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?, ?
                       )""",
                    (
                        run_id, epoch_id, phase["id"], "pending",
                        phase.get("actor", ""),
                        1 if phase.get("required", True) else 0,
                        1 if phase.get("mutation", False) else 0,
                        json.dumps(phase.get("roles", [])),
                        json.dumps(phase.get("depends_on", [])),
                        json.dumps(phase.get("conditional")) if phase.get("conditional") else None,
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
        """Apply only deterministic conditional skips whose dependencies are ready."""
        changed: list[dict] = []
        findings = self.get_findings(run_id, epoch_id=epoch_id)
        has_accepted_findings = any(item.get("disposition") == "accepted" for item in findings)
        for phase in self.get_ready_phases(run_id, epoch_id):
            if phase.get("condition_json") is None:
                continue
            condition = json.loads(phase["condition_json"])
            if condition == "accepted_findings" and not has_accepted_findings:
                changed.append(self.skip_conditional_phase(
                    run_id, epoch_id, phase["phase_id"],
                    "accepted_findings",
                    reason="conditional accepted_findings condition is satisfied by an empty accepted set",
                ))
        return changed

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
        if phase.get("result_schema"):
            successful = [
                item for item in completed
                if item.get("schema_valid") is True
                and item.get("evidence_valid") is not False
                and item.get("accepted_by_controller") is not False
            ]
        else:
            successful = [
                item for item in completed
                if item.get("evidence_valid") is not False
                and item.get("accepted_by_controller") is not False
            ]
        required_successes = int(
            phase.get("required_successes")
            or max(int(phase.get("min_fanout") or 1), int(phase.get("quality_quorum") or 1))
        )
        if len(successful) >= required_successes:
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
            first_failure = failed[0] if failed else executions[-1]
            return self.complete_phase(
                run_id, epoch_id, phase_id,
                error=f"execution failure: {first_failure.get('execution_id')}",
            )
        return phase

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
                "SELECT id, status, dependencies_json, required_actor, mutating "
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
                "turn_budget, max_attempts FROM workflow_phases "
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
            executions = conn.execute(
                "SELECT claude_agent_id, independence_key, status, tool_call_count, "
                "schema_valid, evidence_valid, accepted_by_controller, quality_score "
                "FROM agent_executions "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?", (run_id, epoch_id, phase_id)
            ).fetchall()
            if new_status == "completed" and phase_row[8] and executions:
                observed_turns = sum(int(item[3] or 0) for item in executions)
                if observed_turns > int(phase_row[8]):
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded turn budget {phase_row[8]}"
                    )
            if new_status == "completed" and executions:
                completed = [item for item in executions if item[2] == "completed"]
                if phase_row[7]:
                    quality_eligible = [
                        item for item in completed
                        if item[4] == 1 and item[5] != 0 and item[6] != 0
                    ]
                else:
                    quality_eligible = [
                        item for item in completed
                        if item[5] != 0 and item[6] != 0
                    ]
                min_fanout = int(phase_row[1] or 1)
                max_fanout = int(phase_row[2] or 1)
                quorum = int(phase_row[3] or 1)
                max_attempts = int(phase_row[9] or max_fanout)
                if len(executions) > max_attempts:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' exceeded max attempts {max_attempts}"
                    )
                if len(completed) < min_fanout or len(quality_eligible) < quorum:
                    raise WorkflowPhaseStateError(
                        f"Phase '{phase_id}' requires {max(min_fanout, quorum)} quality-valid execution(s)"
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
            now = _utcnow()
            conn.execute(
                "UPDATE workflow_phases SET status=?, completed_at=?, result_evidence=?, error=? "
                "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                (new_status, now, result_evidence, error, run_id, epoch_id, phase_id),
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
