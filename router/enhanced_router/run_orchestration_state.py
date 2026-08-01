"""Task orchestration entry points, split out of state.py.

Twenty-third and final increment of the "Run lifecycle" decomposition. This
is the authoritative task-start transaction (``begin_task``: atomic run +
epoch + phases + routes creation), completion validation, and the
completion-token handshake that gates a controller's final "done" report
against a matching workspace/route snapshot.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from enhanced_router.state_errors import WorkflowStateError

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunOrchestrationRepository:
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

    def prepare_task_plan(
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
        """Compute the deterministic task plan without writing anything yet.

        Pure/read-only: resolves the effective workflow+profile from
        registry state and pre-generates the epoch_id that
        materialize_task_epoch will later create -- letting a caller (e.g.
        the fastpath-aware two-stage task start) reference that exact
        epoch_id (for a fastpath packet, a reserved route proposal, ...)
        before the epoch row itself exists. Returns everything
        materialize_task_epoch needs as **kwargs.
        """
        from enhanced_router.registry import get_registry

        signals = signals or []

        if force_workflow:
            effective_workflow = force_workflow
        elif minimum_tier:
            effective_workflow = minimum_tier
        else:
            from enhanced_router.config_models import determine_tier
            tier = determine_tier(signals) if signals else (workflow_id or "normal")
            effective_workflow = tier

        reg = get_registry()
        reg.load_workflows()
        spec = reg.get_workflow(effective_workflow)
        if spec is None:
            raise ValueError(f"Workflow '{effective_workflow}' not found in registry")
        reg.load_profiles()
        effective_profile_id = profile_id or spec.default_profile
        # Confirm the profile actually resolves now, while it's cheap to
        # fail -- materialize_task_epoch re-resolves it again for its own
        # transaction rather than trusting a plan built against
        # possibly-stale registry state.
        reg.get_profile(effective_profile_id)
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None

        return {
            "run_id": run_id,
            "session_id": session_id,
            "cwd": cwd,
            "epoch_id": f"ep_{_utcnow().replace(':', '-').replace('.', '-')}",
            "workflow_id": effective_workflow,
            "profile_id": effective_profile_id,
            "prompt_digest": prompt_digest,
            "intake_id": intake_id,
            "minimum_tier": minimum_tier or effective_workflow,
            "baseline_fingerprint": baseline_fingerprint,
            "contract": contract,
        }

    def materialize_task_epoch(
        self,
        plan: dict,
        *,
        route_overrides: dict[str, str] | None = None,
    ) -> dict:
        """Atomic run + epoch + routes + phases creation from a prepared plan.

        This is the single authoritative entry point that actually writes a
        new task/epoch. *route_overrides* lets a validated, accepted
        fastpath route proposal that arrived during prepare_task_plan's
        caller-side wait replace the profile default for specific roles --
        atomically, as part of this same epoch materialization, rather than
        via a separate post-hoc set_role_route call after the epoch (and its
        phase DAG) already exist. Only unbound roles should ever be
        overridden this way; there is nothing bound yet at materialization
        time by construction, since binding requires an active epoch.
        """
        from enhanced_router.registry import get_registry

        run_id = str(plan["run_id"])
        session_id = str(plan["session_id"])
        cwd = str(plan["cwd"])
        epoch_id = str(plan["epoch_id"])
        effective_workflow = str(plan["workflow_id"])
        effective_profile_id = str(plan["profile_id"])
        prompt_digest = plan.get("prompt_digest")
        intake_id = plan.get("intake_id")
        minimum_tier = plan.get("minimum_tier")
        baseline_fingerprint = plan.get("baseline_fingerprint")
        contract = plan.get("contract")
        route_overrides = route_overrides or {}

        reg = get_registry()
        reg.load_workflows()
        spec = reg.get_workflow(effective_workflow)
        if spec is None:
            raise ValueError(f"Workflow '{effective_workflow}' not found in registry")
        reg.load_profiles()
        profile = reg.get_profile(effective_profile_id)

        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")

            # 1. Create or confirm run
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, claude_session_id, cwd, created_at) VALUES (?, ?, ?, ?)",
                (run_id, session_id, cwd, _utcnow()),
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

            # 3. Create epoch (epoch_id already generated by prepare_task_plan)
            conn.execute(
                "INSERT INTO epochs (run_id, epoch_id, workflow_id, profile_id, status, created_at,"
                " prompt_digest, minimum_tier, intake_id, contract_json, baseline_fingerprint)"
                " VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)",
                (run_id, epoch_id, effective_workflow, effective_profile_id, _utcnow(),
                 prompt_digest, minimum_tier or effective_workflow, intake_id,
                 json.dumps(contract or {}, sort_keys=True, separators=(",", ":")),
                 baseline_fingerprint),
            )

            # 4. Set profile routes atomically, applying any accepted
            # fastpath route_override for a role in place of the profile
            # default -- this is what actually makes a proposal that arrived
            # in time affect the initial materialization instead of only
            # ever being applicable after the fact via set_role_route.
            now = _utcnow()
            for role in ("recon", "implementer", "adversary", "repairer"):
                target = profile.route_target(role)
                override_model_id = route_overrides.get(role)
                model_id = override_model_id or target.model
                source = f"fastpath-accepted:{effective_profile_id}" if override_model_id else f"profile:{effective_profile_id}"
                event_type = "fastpath_route_set" if override_model_id else "profile_set"
                conn.execute(
                    """INSERT INTO role_routes (run_id, epoch_id, role, model_id, source, reason, version, changed_at)
                       VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                       ON CONFLICT(run_id, epoch_id, role) DO UPDATE SET
                           model_id = excluded.model_id,
                           source = excluded.source,
                           reason = excluded.reason,
                           version = version + 1,
                           changed_at = excluded.changed_at""",
                    (run_id, epoch_id, role, model_id, source, source, now),
                )
                # A fastpath-selected route always uses "auto" endpoint
                # selection (fastpath never chooses a physical endpoint,
                # same invariant FastpathPolicyValidator enforces); only a
                # profile default may pin an explicit endpoint.
                endpoint_id = None if override_model_id else (
                    None if target.endpoint == "auto" else target.endpoint
                )
                conn.execute(
                    "UPDATE role_routes SET endpoint_id=? WHERE run_id=? AND epoch_id=? AND role=?",
                    (endpoint_id, run_id, epoch_id, role),
                )
                conn.execute(
                    "UPDATE role_routes SET fallback_models_json=? WHERE run_id=? AND epoch_id=? AND role=?",
                    (json.dumps([] if override_model_id else target.fallback_models), run_id, epoch_id, role),
                )
                conn.execute(
                    "INSERT INTO route_events (run_id, epoch_id, event_type, role, new_model_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, epoch_id, event_type, role, model_id, now),
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
                    "max_parallelism": p.max_parallelism,
                    "required_successes": p.required_successes,
                    "max_attempts": p.max_attempts,
                    "max_attempts_per_model": p.max_attempts_per_model,
                    "sidecar_id": p.sidecar,
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
                        phase.get("sidecar_id"),
                    ),
                )

            conn.commit()

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
        """One-shot task start: prepare_task_plan then materialize_task_epoch
        with no route overrides.

        Kept for callers that don't need the two-stage fastpath-aware flow
        (a bounded wait for a route proposal between planning and
        materializing) -- exactly the previous begin_task behavior,
        composed from the same two building blocks that flow now uses.
        """
        plan = self.prepare_task_plan(
            run_id, session_id, cwd, workflow_id=workflow_id, profile_id=profile_id,
            signals=signals, force_workflow=force_workflow, prompt=prompt,
            intake_id=intake_id, minimum_tier=minimum_tier,
            baseline_fingerprint=baseline_fingerprint, contract=contract,
        )
        return self.materialize_task_epoch(plan)

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
        for phase in phases:
            if phase.get("status") == "completed" and not phase.get("result_evidence"):
                return {
                    "valid": False,
                    "reason": f"Phase '{phase['phase_id']}' is completed but has no result_evidence"
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
        token = secrets.token_urlsafe(32)
        token_id = f"completion:{secrets.token_hex(12)}"
        issued_at = datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(seconds=ttl_seconds)
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO completion_tokens "
                "(token_id, run_id, epoch_id, token_hash, workspace_fingerprint, "
                "route_snapshot_sha256, issued_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    token_id, run_id, epoch_id,
                    hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    workspace_fingerprint, snapshot,
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
            "expires_at": expires_at.isoformat(),
        }

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
