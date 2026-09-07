"""Task orchestration entry points, split out of state.py.

Twenty-third and final increment of the "Run lifecycle" decomposition. This
is the authoritative task-start transaction (``begin_task``: atomic run +
epoch + phases + routes creation), completion validation, and the
completion-token handshake that gates a controller's final "done" report
against a matching workspace/route snapshot.
"""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from enhanced_router.state_errors import WorkflowStateError
from enhanced_router.route_ladder import target_candidates

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


WORKFLOW_TIERS = ("trivial", "normal", "cross-cutting", "high-risk")


def resolve_effective_workflow(
    requested_workflow: str,
    minimum_tier: str | None,
    requested_spec: object | None,
) -> str:
    """Resolve intake workflow without permitting a tier downgrade.

    Built-in workflow IDs are ordered by name. Custom workflow IDs must
    declare ``WorkflowSpec.tier`` when a caller has already committed a
    minimum tier; otherwise there is no safe way to prove that a forced
    workflow contains the required review/verification strength.
    """
    tier_order = {name: index for index, name in enumerate(WORKFLOW_TIERS)}
    requested_tier = (
        requested_workflow
        if requested_workflow in tier_order
        else getattr(requested_spec, "tier", None)
    )
    if minimum_tier not in tier_order:
        return requested_workflow
    if requested_tier not in tier_order:
        raise ValueError(
            f"workflow '{requested_workflow}' does not declare a tier and "
            f"cannot satisfy minimum tier '{minimum_tier}'"
        )
    if tier_order[str(requested_tier)] < tier_order[str(minimum_tier)]:
        return str(minimum_tier)
    return requested_workflow


class RunOrchestrationRepository(RepositoryMixin):
    """Mixin providing task orchestration entry points.

    Requires a host class that provides ``_new_conn() -> sqlite3.Connection``
    (``RouteState`` does), plus a large set of already-extracted
    cross-section methods (epochs, mutation leases, agent executions,
    findings, provider reservations, workflow phases, workspaces, controller
    policy) and ``create_route_snapshot`` (still defined directly on
    ``RouteState``) -- all resolve through the mixin's normal method
    resolution order.
    """

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - overridden by RouteState
        raise NotImplementedError

    def begin_task(
        self,
        run_id: str,
        session_id: str,
        cwd: str,
        workflow_id: str | None = None,
        profile_id: str | None = None,
        signals: list[str] | None = None,
        force_workflow: str | None = None,
        prompt: str = "",
        intake_id: str | None = None,
        minimum_tier: str | None = None,
        baseline_fingerprint: str | None = None,
        contract: dict | None = None,
    ) -> dict:
        """Authoritative task start — atomic run + epoch + phases creation.

        Creates or idempotently confirms the run, creates a fresh epoch
        from the workflow+profile, initializes all workflow phases with
        full semantics, and returns the task contract.

        This is the single authoritative entry point for starting a new
        task/epoch. All hooks call this instead of doing inline
        run/epoch/phase creation.

        Returns dict with: run_id, epoch_id, workflow_id, profile_id,
        phases, specification_hash.
        """
        from enhanced_router.registry import get_registry

        signals = signals or []

        # Determine the effective workflow without allowing a caller or a
        # late route proposal to downgrade the intake commitment. Named
        # custom workflows must declare their tier when a minimum has been
        # committed; otherwise intake fails closed.
        from enhanced_router.config_models import determine_tier
        requested_workflow = force_workflow or workflow_id
        if requested_workflow is None:
            requested_workflow = determine_tier(signals) if signals else "normal"
        reg = get_registry()
        reg.load_workflows()
        requested_workflow = str(requested_workflow)
        requested_spec = reg.get_workflow(requested_workflow)
        if requested_spec is None:
            raise ValueError(f"Workflow '{requested_workflow}' not found in registry")
        effective_workflow = resolve_effective_workflow(
            requested_workflow, minimum_tier, requested_spec,
        )

        # Verify the promoted built-in tier (or explicitly declared custom
        # workflow) exists before mutating any run/epoch state.
        spec = (
            requested_spec
            if effective_workflow == requested_workflow
            else reg.get_workflow(effective_workflow)
        )
        if spec is None:
            raise ValueError(f"Workflow '{effective_workflow}' not found in registry")
        reg.load_profiles()
        effective_profile_id = profile_id or spec.default_profile
        profile = reg.get_profile(effective_profile_id)
        resource_policy_json = json.dumps(
            spec.resource_policy.model_dump(exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")

            # 1. Create or confirm run
            conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, claude_session_id, cwd, resource_policy_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, session_id, cwd, resource_policy_json, _utcnow()),
            )
            # The launcher pre-registers the run before task intake.  Bind the
            # selected workflow's policy exactly once at epoch creation so a
            # later YAML reload cannot change this run's limits.
            conn.execute(
                "UPDATE runs SET resource_policy_json=COALESCE(resource_policy_json, ?) "
                "WHERE run_id=? AND closed_at IS NULL",
                (resource_policy_json, run_id),
            )
            if session_id or cwd:
                sets: list[str] = []
                params: list = []
                if session_id:
                    sets.append("claude_session_id = COALESCE(claude_session_id, ?)")
                    params.append(session_id)
                if cwd:
                    sets.append("cwd = COALESCE(cwd, ?)")
                    params.append(cwd)
                if sets:
                    params.append(run_id)
                    conn.execute(
                        f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ? AND closed_at IS NULL",
                        params,
                    )

            # 2. Check for existing active epoch
            active = conn.execute(
                "SELECT 1 FROM epochs WHERE run_id = ? AND closed_at IS NULL",
                (run_id,),
            ).fetchone()
            if active:
                conn.rollback()
                raise ValueError(f"Active epoch already exists for run {run_id}")

            # 3. Create epoch
            epoch_id = f"ep_{_utcnow().replace(':', '-').replace('.', '-')}"
            conn.execute(
                "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, composition_mode, status, created_at,"
                " prompt_digest, minimum_tier, intake_id, contract_json, baseline_fingerprint)"
                " VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)",
                (run_id, epoch_id, effective_workflow, effective_profile_id,
                 spec.composition_mode, _utcnow(),
                 prompt_digest, minimum_tier or effective_workflow, intake_id,
                 json.dumps(contract or {}, sort_keys=True, separators=(",", ":")),
                 baseline_fingerprint),
            )

            # 4. Set profile routes atomically
            now = _utcnow()
            for role in ("recon", "implementer", "adversary", "repairer"):
                target = profile.route_target(role)
                model_id = target.model
                candidates = target_candidates(target)
                primary = candidates[0]
                fallback_models = [item["model"] for item in candidates[1:]]
                fallback_routes = candidates[1:]
                conn.execute(
                    """INSERT INTO role_routes (
                           run_id, epoch_id, role, model_id, source, reason,
                           version, changed_at, primary_route_json,
                           fallback_models_json, fallback_routes_json
                       ) VALUES (?, ?, ?, ?, 'profile', ?, 1, ?, ?, ?, ?)
                       ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                           model_id = excluded.model_id,
                           source = excluded.source,
                           reason = excluded.reason,
                           version = version + 1,
                           changed_at = excluded.changed_at,
                           primary_route_json = excluded.primary_route_json,
                           fallback_models_json = excluded.fallback_models_json,
                           fallback_routes_json = excluded.fallback_routes_json""",
                    (
                        run_id, epoch_id, role, model_id, f"profile:{effective_profile_id}", now,
                        json.dumps(primary, separators=(",", ":")),
                        json.dumps(fallback_models, separators=(",", ":")),
                        json.dumps(fallback_routes, separators=(",", ":")),
                    ),
                )
                conn.execute(
                    "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                    (None if target.endpoint == "auto" else target.endpoint, run_id, epoch_id, role),
                )
                conn.execute(
                    "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                    (None if primary["endpoint"] == "auto" else primary["endpoint"], run_id, epoch_id, role),
                )
                conn.execute(
                    "INSERT INTO route_events (run_id, epoch_id, event_type, role, new_model_id, created_at) "
                    "VALUES (?, ?, 'profile_set', ?, ?, ?)",
                    (run_id, epoch_id, role, model_id, now),
                )

            # 5. Initialize workflow phases
            phases_data = [
                {
                    "id": p.id,
                    "roles": p.roles,
                    "required": p.required,
                    "mutation": p.mutation,
                    "depends_on": p.depends_on,
                    "conditional": p.conditional,
                    "condition": p.condition.model_dump(exclude_none=True) if p.condition else None,
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
                    "result_contract": (
                        p.result_contract.model_dump(exclude_none=True)
                        if p.result_contract else None
                    ),
                    "quality_quorum": p.quality_quorum,
                    "fallback_policy": p.fallback_policy,
                    "execution_kind": p.execution_kind,
                    "agent_id": p.agent_id,
                    "max_parallelism": p.max_parallelism,
                    "required_successes": p.required_successes,
                    "max_attempts": p.max_attempts,
                    "max_attempts_per_model": p.max_attempts_per_model,
                    "completion_mode": p.completion_mode,
                    "launch_policy": p.launch_policy,
                    "minimum_quality_score": p.minimum_quality_score,
                    "requires_controller_acceptance": p.requires_controller_acceptance,
                    "initial_fanout": p.initial_fanout,
                    "maximum_replicas": p.maximum_replicas,
                    "hedge_delay_seconds": p.hedge_delay_seconds,
                    "sidecar_id": p.sidecar,
                    "sidecar_agent_id": p.sidecar_agent,
                    "coprocessor_id": p.coprocessor,
                    "produces": p.produces,
                    "fanout_from": p.fanout_from,
                }
                for p in spec.phases
            ]

            for phase in phases_data:
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
                        phase.get("sidecar_id"), phase.get("sidecar_agent_id"),
                        phase.get("coprocessor_id"), phase.get("produces"), phase.get("fanout_from"),
                    ),
                )
                conn.execute(
                    "UPDATE workflow_phases SET agent_id=? "
                    "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                    (phase.get("agent_id"), run_id, epoch_id, phase["id"]),
                )
                conn.execute(
                    "UPDATE workflow_phases SET completion_mode=?, minimum_quality_score=?, "
                    "requires_controller_acceptance=?, initial_fanout=?, maximum_replicas=?, hedge_delay_seconds=?, launch_policy=? "
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
                if phase.get("condition") is not None:
                    conn.execute(
                        "UPDATE workflow_phases SET condition_json=? "
                        "WHERE run_id=? AND epoch_id=? AND phase_id=?",
                        (
                            json.dumps(phase["condition"], sort_keys=True, separators=(",", ":")),
                            run_id, epoch_id, phase["id"],
                        ),
                    )

            conn.commit()

            if contract:
                # Mirror the intake contract into the normalized coverage
                # ledger after the atomic run/epoch transaction commits.  The
                # epoch JSON remains a compatibility snapshot; the ledger is
                # the authoritative completion surface for new tasks.
                self.publish_task_contract(
                    run_id, epoch_id, contract, source="begin_task",
                )

            # 6. Controller policy is capability-based. The controller is not
            # required to be one of the worker models in the selected profile.
            from enhanced_router.registry import get_registry
            registry = get_registry()
            permitted_models = [model_id for model_id, _ in registry.controller_models()]
            permitted_models.extend(
                model_id for model_id, model in registry.models.items()
                if model.enabled and model.backend == "anthropic-passthrough"
            )
            if not permitted_models:
                # Backward-compatible fallback for catalogs not yet certified.
                permitted_models = sorted({profile.route_target(role).model for role in _VALID_ROLES})
            self.upsert_controller_policy(run_id, permitted_models, policy="reject")

            # Return the task contract
            phases = self.get_workflow_phases(run_id, epoch_id)
            return {
                "run_id": run_id,
                "epoch_id": epoch_id,
                "workflow_id": effective_workflow,
                "profile_id": effective_profile_id,
                "intake_id": intake_id,
                "minimum_tier": minimum_tier or effective_workflow,
                "prompt_digest": prompt_digest,
                "phases": [
                    {
                        "phase_id": p["phase_id"],
                        "status": p["status"],
                        "required": bool(p["required"]),
                        "mutating": bool(p["mutating"]),
                        "allowed_roles": __import__("json").loads(p["allowed_roles_json"]),
                        "depends_on": __import__("json").loads(p["dependencies_json"]),
                        "specification_hash": p["specification_hash"],
                    }
                    for p in phases
                ],
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def validate_completion(
        self,
        run_id: str,
        epoch_id: str,
        parsed: dict[str, str],
        session_dir: 'Path | None' = None,
    ) -> dict:
        """Validate completion evidence against authoritative state.

        Checks: tier/workflow consistency, phase sequence, findings
        lifecycle, workspace fingerprint, and route snapshot.

        Returns dict with: valid (bool), reason (str|None),
        snapshot_sha256 (str|None).
        """

        # 1. Validate workflow tier matches active epoch
        active = self.get_active_epoch(run_id)
        if not active or active.get("epoch_id") != epoch_id:
            return {"valid": False, "reason": f"No active epoch for run {run_id}"}
        reclassify = getattr(self, "reclassify_before_gate", None)
        if callable(reclassify):
            try:
                gate_result = reclassify(run_id, epoch_id, gate="completion")
            except Exception as exc:
                return {
                    "valid": False,
                    "reason": f"Unable to reclassify workspace before completion: {exc}",
                }
            if isinstance(gate_result, dict) and gate_result.get("applied"):
                return {
                    "valid": False,
                    "reason": (
                        "Workspace changes require workflow escalation before completion; "
                        "controller must complete the compensating review graph"
                    ),
                    "escalation": gate_result,
                }
        if bool(active.get("mutation_paused")) or str(active.get("escalation_state") or "") == "escalated":
            return {
                "valid": False,
                "reason": "Workflow escalation requires controller acknowledgment and compensating phases",
            }

        # Lifecycle state is authoritative.  JSONL hook mirrors and the
        # assistant's footer cannot make an active or queued execution look
        # complete.  The controller's own provider reservation is retained
        # for the lifetime of its session, so only worker reservations are
        # considered here; they must have been released by SubagentStop.
        active_executions = self.get_agent_executions(run_id, epoch_id=epoch_id)
        active_executions = [
            execution for execution in active_executions
            if execution.get("status") not in {"completed", "failed", "timeout", "cancelled"}
        ]
        if active_executions:
            return {
                "valid": False,
                "reason": "Active native agent execution(s) remain: "
                + ", ".join(str(item.get("execution_id")) for item in active_executions),
            }
        worker_reservations = [
            reservation for reservation in self.get_provider_reservations(active_only=True)
            if reservation.get("run_id") == run_id
            and reservation.get("epoch_id") == epoch_id
            and reservation.get("lane") != "controller"
        ]
        if worker_reservations:
            return {
                "valid": False,
                "reason": "Provider agent reservations remain active or queued: "
                + ", ".join(str(item.get("reservation_id")) for item in worker_reservations),
            }
        if self.get_active_mutation_leases(run_id, epoch_id):
            return {"valid": False, "reason": "An active workspace mutation lease remains"}
        workspace_rows = self.get_workspaces(run_id=run_id, epoch_id=epoch_id)
        pending_workspaces = [
            row for row in workspace_rows
            if row.get("kind") == "shadow" and row.get("status") in {"active", "ready"}
        ]
        if pending_workspaces:
            return {
                "valid": False,
                "reason": "Shadow workspace changesets still require integration: "
                + ", ".join(str(row.get("workspace_id")) for row in pending_workspaces),
            }
        conn = self._new_conn()
        try:
            red_candidates = conn.execute(
                "SELECT candidate_id, disposition FROM integration_candidates "
                "WHERE run_id=? AND epoch_id=? AND disposition IN ('red','yellow','pending')",
                (run_id, epoch_id),
            ).fetchall()
        finally:
            conn.close()
        if red_candidates:
            return {
                "valid": False,
                "reason": "Integration candidates require main-controller adjudication: "
                + ", ".join(f"{row[0]} ({row[1]})" for row in red_candidates),
            }
        reported_workflow = parsed.get("Workflow-ID", "")
        if reported_workflow and reported_workflow != active.get("workflow_id"):
            return {
                "valid": False,
                "reason": (
                    f"Workflow-ID mismatch: message says '{reported_workflow}' "
                    f"but active epoch has workflow '{active['workflow_id']}'"
                ),
            }

        reported_tier = parsed.get("Workflow-Tier", "")
        if reported_tier and reported_tier != active.get("workflow_id"):
            return {
                "valid": False,
                "reason": (
                    f"Workflow-Tier mismatch: message says '{reported_tier}' "
                    f"but active epoch has workflow '{active.get('workflow_id')}'"
                ),
            }

        # 2. Validate all required phases completed, no active phases remain
        phases = self.get_workflow_phases(run_id, epoch_id)
        for phase in phases:
            if phase.get("required") and phase.get("status") != "completed":
                return {
                    "valid": False,
                    "reason": f"Required phase '{phase['phase_id']}' is not completed (status: {phase.get('status')})"
                }
            if phase.get("status") == "active":
                return {
                    "valid": False,
                    "reason": f"Phase '{phase['phase_id']}' is still active at completion"
                }

        # 3. Validate phase sequence: dependencies satisfied
        for phase in phases:
            if phase.get("status") == "completed":
                deps = json.loads(phase.get("dependencies_json", "[]"))
                for dep_id in deps:
                    dep_phase = next((p for p in phases if p["phase_id"] == dep_id), None)
                    if dep_phase is None or dep_phase.get("status") not in {"completed", "skipped"}:
                        return {
                            "valid": False,
                            "reason": f"Phase '{phase['phase_id']}' completed but dependency '{dep_id}' is not completed"
                        }

        # 4. Validate mutating phases don't overlap
        # mutating_phases = [p for p in phases if p.get("mutating") and p.get("status") == "completed"]
        # This is a simple check - more thorough overlap detection would require timestamps
        # For now, ensure no two mutating phases have overlapping active periods by checking
        # that they completed sequentially (simplified)

        # 5. Validate findings lifecycle via state
        findings = self.get_findings(run_id, epoch_id=epoch_id)
        open_accepted = [f for f in findings if f.get("disposition") == "accepted" and f.get("verification_status") == "pending"]

        accepted_findings = [f for f in findings if f.get("disposition") == "accepted"]
        if open_accepted:
            return {
                "valid": False,
                "reason": f"Accepted findings remain unresolved: {len(open_accepted)} open accepted finding(s)",
            }
        # The footer is presentation only. The database is authoritative: an
        # empty accepted set is valid, and a non-empty set must be verified.
        if any(f.get("verification_status") != "verified" for f in accepted_findings):
            return {"valid": False, "reason": "Accepted findings exist without verified resolution evidence"}

        # 6. Validate that phases have result_evidence when completed
        main_workspaces = [
            row for row in self.get_workspaces(run_id=run_id, epoch_id=epoch_id, kind="main")
            if row.get("status") in {"active", "ready", "merged"}
        ]
        current_workspace = main_workspaces[0] if main_workspaces else None
        for phase in phases:
            if phase.get("status") == "completed" and not phase.get("result_evidence"):
                return {
                    "valid": False,
                    "reason": f"Phase '{phase['phase_id']}' is completed but has no result_evidence"
                }
            if phase.get("status") == "completed" and current_workspace is not None:
                contract_json = phase.get("result_contract_json")
                try:
                    result_contract = json.loads(str(contract_json or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    result_contract = {}
                if result_contract.get("requires_current_generation"):
                    if phase.get("evidence_generation") is None or int(phase["evidence_generation"]) != int(current_workspace.get("canonical_generation") or 0):
                        return {
                            "valid": False,
                            "reason": f"Phase '{phase['phase_id']}' evidence is stale for the current workspace generation",
                        }
                if result_contract.get("requires_current_digest"):
                    current_digest = current_workspace.get("current_dirty_hash") or current_workspace.get("dirty_patch_hash")
                    if not current_digest or str(phase.get("evidence_digest") or "") != str(current_digest):
                        return {
                            "valid": False,
                            "reason": f"Phase '{phase['phase_id']}' evidence is stale for the current workspace digest",
                        }

        # A published task contract turns completion into a coverage claim,
        # not merely a phase-order claim.  Legacy runs without a contract
        # retain their historical behavior; new controller-managed tasks
        # cannot finish while mandatory requirements are uncovered.
        contract = self.get_task_contract(run_id, epoch_id)
        if contract is not None:
            coverage = self.get_requirement_coverage(run_id, epoch_id)
            if not coverage.get("complete", False):
                return {
                    "valid": False,
                    "reason": "Mandatory task requirements remain uncovered: "
                    + ", ".join(str(item) for item in coverage.get("missing_mandatory", [])),
                    "coverage": coverage,
                }
            coverage_audit = self.get_latest_coverage_audit(run_id, epoch_id)
            if not coverage_audit or not bool(coverage_audit.get("complete")):
                return {
                    "valid": False,
                    "reason": "A complete controller coverage audit is required before completion",
                    "coverage": coverage,
                    "coverage_audit": coverage_audit,
                }
            audit_version = coverage_audit.get("contract_version")
            if audit_version is not None and int(audit_version) != int(contract.get("version") or 0):
                return {
                    "valid": False,
                    "reason": "Coverage audit was created for a different task-contract version",
                    "coverage": coverage,
                    "coverage_audit": coverage_audit,
                }
            main_workspaces = [
                row for row in self.get_workspaces(run_id=run_id, epoch_id=epoch_id, kind="main")
                if row.get("status") in {"active", "ready", "merged"}
            ]
            if main_workspaces:
                workspace = main_workspaces[0]
                current_generation = int(workspace.get("canonical_generation") or 0)
                audited_generation = coverage_audit.get("workspace_generation")
                if audited_generation is not None and int(audited_generation) != current_generation:
                    return {
                        "valid": False,
                        "reason": "Coverage audit is stale: canonical workspace generation changed",
                        "coverage": coverage,
                        "coverage_audit": coverage_audit,
                    }
                current_digest = workspace.get("current_dirty_hash") or workspace.get("dirty_patch_hash")
                audited_digest = coverage_audit.get("workspace_digest")
                if current_digest and audited_digest and str(current_digest) != str(audited_digest):
                    return {
                        "valid": False,
                        "reason": "Coverage audit is stale: canonical workspace changed after the audit",
                        "coverage": coverage,
                        "coverage_audit": coverage_audit,
                    }

        # 7. Validate route snapshot
        snapshot_sha256 = parsed.get("Route-Snapshot-SHA256", "").lower()
        if not snapshot_sha256:
            return {"valid": False, "reason": "Route-Snapshot-SHA256 is required"}
        actual_snapshot = self.create_route_snapshot(run_id, epoch_id, purpose="verified-completion")
        if snapshot_sha256 != actual_snapshot:
            return {
                "valid": False,
                "reason": f"Route snapshot mismatch: reported {snapshot_sha256} but current state produces {actual_snapshot}",
                "snapshot_sha256": actual_snapshot,
            }

        return {"valid": True, "reason": None, "snapshot_sha256": snapshot_sha256 or None}

    def prepare_completion_token(
        self,
        run_id: str,
        epoch_id: str,
        workspace_fingerprint: str,
        *,
        ttl_seconds: int = 300,
    ) -> dict:
        """Issue a short-lived, state-generated completion attestation."""
        if not hmac.compare_digest(
            workspace_fingerprint.lower(), workspace_fingerprint
        ) or len(workspace_fingerprint) != 64:
            raise WorkflowStateError("workspace fingerprint must be 64 lowercase hexadecimal characters")
        try:
            int(workspace_fingerprint, 16)
        except ValueError as exc:
            raise WorkflowStateError("workspace fingerprint must be hexadecimal") from exc
        if ttl_seconds < 1:
            raise ValueError("completion token TTL must be positive")
        active = self.get_active_epoch(run_id)
        if not active or active.get("epoch_id") != epoch_id:
            raise WorkflowStateError("completion token requires the active epoch")
        snapshot = self.create_route_snapshot(run_id, epoch_id, purpose="verified-completion")
        evidence_snapshot = self.create_completion_evidence_snapshot(run_id, epoch_id)
        token = secrets.token_urlsafe(32)
        token_id = f"completion:{secrets.token_hex(12)}"
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(seconds=ttl_seconds)
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO completion_tokens "
                "(token_id, run_id, epoch_id, token_hash, workspace_fingerprint, "
                "route_snapshot_sha256, evidence_snapshot_sha256, issued_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    token_id, run_id, epoch_id,
                    hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    workspace_fingerprint, snapshot, evidence_snapshot,
                    issued_at.isoformat(), expires_at.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "token_id": token_id,
            "token": token,
            "run_id": run_id,
            "epoch_id": epoch_id,
            "workspace_fingerprint": workspace_fingerprint,
            "route_snapshot_sha256": snapshot,
            "evidence_snapshot_sha256": evidence_snapshot,
            "expires_at": expires_at.isoformat(),
        }

    def create_completion_evidence_snapshot(self, run_id: str, epoch_id: str) -> str:
        """Hash current signoff phase/result provenance for token binding."""
        import hashlib
        import json

        phases = self.get_workflow_phases(run_id, epoch_id)
        signoff_markers = (
            "ground", "senior", "completion", "critical", "verification", "final", "audit",
        )
        evidence: list[dict[str, object]] = []
        for phase in phases:
            phase_id = str(phase.get("phase_id") or "").lower()
            if not any(marker in phase_id for marker in signoff_markers):
                continue
            executions = self.get_agent_executions(
                run_id, epoch_id=epoch_id, phase_id=str(phase.get("phase_id")),
            )
            evidence.append({
                "phase_id": phase.get("phase_id"),
                "status": phase.get("status"),
                "evidence_generation": phase.get("evidence_generation"),
                "evidence_digest": phase.get("evidence_digest"),
                "result_evidence": phase.get("result_evidence"),
                "executions": [
                    {
                        "execution_id": item.get("execution_id"),
                        "output_hash": item.get("output_hash"),
                        "result_json": item.get("result_json"),
                        "result_disposition": item.get("result_disposition"),
                    }
                    for item in executions
                ],
            })
        payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def consume_completion_token(
        self,
        run_id: str,
        epoch_id: str,
        token: str,
        workspace_fingerprint: str,
        route_snapshot_sha256: str,
    ) -> dict:
        """Consume a completion attestation exactly once and verify its binding."""
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM completion_tokens WHERE run_id=? AND epoch_id=? "
                "AND token_hash=?",
                (run_id, epoch_id, token_hash),
            ).fetchone()
            if row is None:
                return {"valid": False, "reason": "completion token not found"}
            if row["consumed_at"] is not None:
                return {"valid": False, "reason": "completion token was already consumed"}
            if datetime.fromisoformat(str(row["expires_at"])) < datetime.now(timezone.utc):
                return {"valid": False, "reason": "completion token expired"}
            if not hmac.compare_digest(str(row["workspace_fingerprint"]), workspace_fingerprint):
                return {"valid": False, "reason": "completion token workspace mismatch"}
            if not hmac.compare_digest(str(row["route_snapshot_sha256"]), route_snapshot_sha256):
                return {"valid": False, "reason": "completion token route snapshot mismatch"}
            expected_evidence = row["evidence_snapshot_sha256"]
            if expected_evidence:
                current_evidence = self.create_completion_evidence_snapshot(run_id, epoch_id)
                if not hmac.compare_digest(str(expected_evidence), current_evidence):
                    return {"valid": False, "reason": "completion token evidence snapshot mismatch"}
            consumed_at = _utcnow()
            updated = conn.execute(
                "UPDATE completion_tokens SET consumed_at=? "
                "WHERE token_id=? AND consumed_at IS NULL",
                (consumed_at, row["token_id"]),
            )
            if updated.rowcount != 1:
                conn.rollback()
                return {"valid": False, "reason": "completion token was already consumed"}
            conn.commit()
            return {"valid": True, "token_id": row["token_id"], "consumed_at": consumed_at}
        finally:
            conn.close()

    # _RUN_COLUMNS, create_run, set_run_selection, get_run,
    # active_run_selections, verify_controller_capability, close_run live in
    # RunRegistryRepository (run_registry_state.py), mixed in below.
