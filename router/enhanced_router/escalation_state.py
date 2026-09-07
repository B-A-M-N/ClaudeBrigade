"""Durable escalation proposals and atomic epoch escalation."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any
from pathlib import PurePath
from fnmatch import fnmatch

from enhanced_router.escalation_policy import (
    EscalationDecision,
    EscalationInputs,
    evaluate_escalation,
)
from enhanced_router.policy import INFRA_GLOBS, SECURITY_GLOBS
from enhanced_router.repository_base import RepositoryMixin


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class EscalationRepository(RepositoryMixin):
    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - host override
        raise NotImplementedError

    def get_escalation_state(self, run_id: str, epoch_id: str) -> dict[str, Any]:
        conn = self._new_conn()
        try:
            epoch = conn.execute(
                "SELECT workflow_id, status, minimum_tier, escalation_level, escalation_state, "
                "escalation_reason, escalated_at, mutation_paused, escalation_generation "
                "FROM epochs WHERE run_id=? AND epoch_id=?", (run_id, epoch_id)
            ).fetchone()
            events = conn.execute(
                "SELECT * FROM escalation_events WHERE run_id=? AND epoch_id=? ORDER BY created_at",
                (run_id, epoch_id),
            ).fetchall()
            return {
                "run_id": run_id,
                "epoch_id": epoch_id,
                "epoch": dict(epoch) if epoch else None,
                "events": [dict(item) for item in events],
            }
        finally:
            conn.close()

    def propose_escalation(
        self, run_id: str, epoch_id: str, decision: EscalationDecision, *, status: str = "proposed"
    ) -> dict[str, Any]:
        escalation_id = f"esc:{uuid.uuid4()}"
        conn = self._new_conn()
        try:
            conn.execute(
                "INSERT INTO escalation_events "
                "(escalation_id,run_id,epoch_id,from_tier,to_tier,reason,signals_json,policy_digest,status,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (escalation_id, run_id, epoch_id, decision.from_tier, decision.to_tier,
                 "; ".join(decision.reasons), json.dumps(list(decision.reasons), separators=(",", ":")),
                 decision.policy_digest, status, _utcnow()),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM escalation_events WHERE escalation_id=?", (escalation_id,)).fetchone()
            assert row is not None
            return dict(row)
        finally:
            conn.close()

    def auto_escalate_after_changeset(
        self,
        run_id: str,
        epoch_id: str,
        changeset_id: str,
        *,
        apply: bool = True,
    ) -> dict[str, Any]:
        """Evaluate objective changeset signals and optionally apply escalation.

        This is the bridge between real workspace mutations and the durable
        escalation policy.  It uses only the persisted changeset file list,
        requirement coverage, findings, and phase history; model claims never
        decide whether a change is high-risk.
        """
        changeset = self.get_changeset(changeset_id)
        if changeset is None:
            raise ValueError(f"unknown changeset {changeset_id!r}")
        try:
            changed_files = json.loads(str(changeset.get("changed_files_json") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            changed_files = []
        changed_files = [str(item) for item in changed_files if item]

        def matches(path: str, patterns: list[str]) -> bool:
            return any(
                fnmatch(path, pattern.replace("**", "*"))
                or (pattern.startswith("**/") and fnmatch(path, pattern[3:]))
                for pattern in patterns
            )

        changed_security = any(matches(path, SECURITY_GLOBS) for path in changed_files)
        changed_infra = any(matches(path, INFRA_GLOBS) for path in changed_files)
        subsystems = {PurePath(path).parts[0] for path in changed_files if PurePath(path).parts}
        minimum_tier = (
            "high-risk" if changed_security
            else "cross-cutting" if changed_infra or len(subsystems) >= 2
            else "normal" if changed_files
            else None
        )
        coverage = self.get_requirement_coverage(run_id, epoch_id)
        findings = self.get_findings(run_id, epoch_id=epoch_id)
        phases = self.get_workflow_phases(run_id, epoch_id)
        active = self.get_active_epoch(run_id)
        current_tier = str(
            (active or {}).get("escalation_level")
            or (active or {}).get("workflow_id")
            or "normal"
        )
        decision = evaluate_escalation(EscalationInputs(
            current_tier=current_tier,
            minimum_tier=minimum_tier,
            changed_files=len(changed_files),
            changed_security_paths=changed_security,
            unresolved_findings=sum(
                item.get("disposition") == "accepted"
                and item.get("verification_status") != "verified"
                for item in findings
            ),
            missing_requirements=len(coverage.get("missing_mandatory", [])),
            evidence_incomplete=not bool(coverage.get("complete", True)),
            repeated_repairs=sum(
                item.get("status") == "completed"
                and ("repair" in str(item.get("phase_id", "")).lower()
                     or "repair" in str(item.get("phase_template_id", "")).lower())
                for item in phases
            ),
        ))
        result: dict[str, Any] = {
            "changeset_id": changeset_id,
            "changed_files": changed_files,
            "minimum_tier": minimum_tier,
            "decision": decision.__dict__,
            "applied": False,
        }
        if not decision.should_escalate:
            return result
        result["proposal"] = self.propose_escalation(run_id, epoch_id, decision)
        if apply:
            result["state"] = self.escalate_epoch(run_id, epoch_id, decision)
            result["applied"] = True
        return result

    def reclassify_before_gate(
        self,
        run_id: str,
        epoch_id: str,
        *,
        gate: str,
    ) -> dict[str, Any]:
        """Re-evaluate the last integrated workspace change before a gate.

        Worker and integration hooks normally evaluate a changeset when it is
        produced or applied.  A restart, a controller-owned integration, or a
        late repair can otherwise leave a verification/completion gate using
        an old tier decision.  The canonical workspace records its latest
        applied changeset, so this check is deterministic and idempotent: an
        already-promoted epoch simply returns ``applied=False``.
        """
        main_workspaces = self.get_workspaces(
            run_id=run_id, epoch_id=epoch_id, kind="main",
        )
        if not main_workspaces:
            return {"gate": gate, "applied": False, "reason": "no canonical workspace"}
        changeset_id = str(main_workspaces[0].get("last_changeset_id") or "")
        if not changeset_id:
            return {"gate": gate, "applied": False, "reason": "no integrated changeset"}
        result = self.auto_escalate_after_changeset(
            run_id, epoch_id, changeset_id, apply=True,
        )
        result["gate"] = gate
        return result

    def escalate_epoch(
        self, run_id: str, epoch_id: str, decision: EscalationDecision, *, applied_by: str = "controller"
    ) -> dict[str, Any]:
        """Apply one escalation with one SQLite transaction.

        Existing work is not silently downgraded or rewritten.  The epoch is
        marked escalated, pending claims are invalidated, and a durable event
        records the new tier.  The controller then re-queries runnable actions
        and may schedule the compensating phases defined by the workflow.
        """
        if not decision.should_escalate:
            raise ValueError("decision does not request escalation")
        tiers = ("trivial", "normal", "cross-cutting", "high-risk")
        if decision.from_tier not in tiers or decision.to_tier not in tiers:
            raise ValueError("escalation tiers must be trivial, normal, cross-cutting, or high-risk")
        if tiers.index(decision.to_tier) <= tiers.index(decision.from_tier):
            raise ValueError("escalation must move to a strictly stronger tier")
        escalation_id = f"esc:{uuid.uuid4()}"
        now = _utcnow()
        conn = self._new_conn()
        reservations: list[str] = []
        token_reservations: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            epoch = conn.execute(
                "SELECT workflow_id, minimum_tier, escalation_level, escalation_generation "
                "FROM epochs WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if epoch is None:
                raise ValueError("active epoch not found")
            current_tier = str(epoch["escalation_level"] or epoch["workflow_id"] or "normal")
            minimum_tier = str(epoch["minimum_tier"] or "trivial")
            if current_tier not in tiers:
                current_tier = "normal"
            if tiers.index(decision.to_tier) < max(
                tiers.index(current_tier),
                tiers.index(minimum_tier) if minimum_tier in tiers else 0,
            ):
                raise ValueError("escalation target is below the epoch's committed tier")
            claims = conn.execute(
                "SELECT reservation_id, token_reservation_id FROM runnable_action_claims "
                "WHERE run_id=? AND epoch_id=? AND status IN ('claimed','consumed')",
                (run_id, epoch_id),
            ).fetchall()
            reservations = [str(row[0]) for row in claims if row[0]]
            token_reservations = [str(row[1]) for row in claims if row[1]]
            conn.execute(
                "UPDATE epochs SET workflow_id=?, escalation_level=?, escalation_state='escalated', "
                "escalation_reason=?, escalated_at=?, mutation_paused=1, "
                "escalation_generation=COALESCE(escalation_generation, 0)+1 "
                "WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (decision.to_tier, decision.to_tier, "; ".join(decision.reasons)[:2000], now, run_id, epoch_id),
            )
            conn.execute(
                "UPDATE runnable_action_claims SET status='cancelled', consumed_at=? "
                "WHERE run_id=? AND epoch_id=? AND status IN ('claimed','consumed')",
                (now, run_id, epoch_id),
            )
            # Preserve completed evidence, but prevent lower-tier mutating
            # phases from becoming runnable after the escalation. Active
            # workers are stopped by the native guard on their next mutation.
            conn.execute(
                "UPDATE workflow_phases SET status='rolled_back', error=COALESCE(error, ?) "
                "WHERE run_id=? AND epoch_id=? AND mutating=1 "
                "AND status IN ('pending','active')",
                (f"mutation paused for escalation {escalation_id}", run_id, epoch_id),
            )
            conn.execute(
                "INSERT INTO escalation_events "
                "(escalation_id,run_id,epoch_id,from_tier,to_tier,reason,signals_json,policy_digest,status,created_at,applied_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (escalation_id, run_id, epoch_id, decision.from_tier, decision.to_tier,
                 "; ".join(decision.reasons), json.dumps(list(decision.reasons), separators=(",", ":")),
                 decision.policy_digest, "applied", now, now),
            )
            self._append_compensating_phases(
                conn, run_id, epoch_id, escalation_id, decision.to_tier, now,
            )
            # Any completion token issued before the graph changed is stale.
            conn.execute(
                "UPDATE completion_tokens SET consumed_at=COALESCE(consumed_at, ?) "
                "WHERE run_id=? AND epoch_id=? AND consumed_at IS NULL",
                (now, run_id, epoch_id),
            )
            conn.commit()
            for reservation_id in reservations:
                try:
                    self.release_provider_reservation(reservation_id, "escalated")
                except Exception:
                    pass
            for reservation_id in token_reservations:
                try:
                    self.release_token_reservation(reservation_id, "escalated")
                except Exception:
                    pass
            return self.get_escalation_state(run_id, epoch_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _append_compensating_phases(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        epoch_id: str,
        escalation_id: str,
        tier: str,
        now: str,
    ) -> None:
        """Append the mandatory review/repair closure for an escalation."""
        suffix = escalation_id.replace(":", "-")
        phases = [
            (f"{suffix}-recon", "recon", 0, ["recon"], [], "escalation-recon"),
            (f"{suffix}-retroactive-design-review", "adversary", 0, ["adversary"], [f"{suffix}-recon"], "retroactive-design-review"),
            (f"{suffix}-implementation-risk-review", "adversary", 0, ["adversary"], [f"{suffix}-retroactive-design-review"], "implementation-risk-review"),
            (f"{suffix}-repair-or-replan", "repairer", 1, ["repairer"], [f"{suffix}-implementation-risk-review"], "repair-or-replan"),
            (f"{suffix}-fresh-adversarial-review", "adversary", 0, ["adversary"], [f"{suffix}-repair-or-replan"], "fresh-adversarial-review"),
            (f"{suffix}-verification", "controller", 0, [], [f"{suffix}-fresh-adversarial-review"], "verification"),
        ]
        ordinal_row = conn.execute(
            "SELECT COALESCE(MAX(ordinal), 0) FROM workflow_phases WHERE run_id=? AND epoch_id=?",
            (run_id, epoch_id),
        ).fetchone()
        ordinal = int(ordinal_row[0] or 0)
        for phase_id, actor, mutating, roles, dependencies, template in phases:
            ordinal += 1
            execution_kind = "controller_action" if actor == "controller" else "native_agent"
            conn.execute(
                """INSERT OR IGNORE INTO workflow_phases
                (run_id, epoch_id, phase_id, status, actor, required, mutating,
                 allowed_roles_json, dependencies_json, condition_json, parallel_group,
                 specification_hash, ordinal, distinct_agent_from_json, max_duration_seconds,
                 turn_budget, provider_requirements_json, min_fanout, max_fanout, result_schema,
                 quality_quorum, fallback_policy, execution_kind, required_actor,
                 max_parallelism, required_successes, max_attempts, max_attempts_per_model,
                 sidecar_id, phase_template_id, iteration, supersedes_phase_id, trigger_event,
                 completion_mode, requires_controller_acceptance)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, epoch_id, phase_id, "pending", actor, 1, mutating,
                    json.dumps(roles), json.dumps(dependencies), None, None,
                    f"{escalation_id}:{template}:{tier}", ordinal, json.dumps(["same-execution"]),
                    None, None, json.dumps([]), 1, 1, None, 1, None, execution_kind,
                    actor, 1, 1, 1, None, None, template, 0, None, escalation_id,
                    "all" if template in {"escalation-recon", "fresh-adversarial-review"} else "controller"
                    if actor == "controller" else "quorum",
                    1,
                ),
            )

    def acknowledge_escalation(
        self, run_id: str, epoch_id: str, *, acknowledged_by: str = "controller",
    ) -> dict[str, Any]:
        """Resume mutation only after the controller acknowledges the new graph."""
        now = _utcnow()
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT escalation_state FROM epochs WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            ).fetchone()
            if row is None:
                raise ValueError("active epoch not found")
            if str(row[0]) != "escalated":
                raise ValueError("epoch has no pending escalation")
            # Acknowledgment is the controller's decision to resume the
            # stronger graph, not a bypass around it.  The compensating
            # read-only review must finish before mutation can resume.
            required_templates = {
                "escalation-recon",
                "retroactive-design-review",
                "implementation-risk-review",
            }
            completed_templates = {
                str(item[0])
                for item in conn.execute(
                    "SELECT phase_template_id FROM workflow_phases "
                    "WHERE run_id=? AND epoch_id=? AND status='completed' "
                    "AND phase_template_id IS NOT NULL",
                    (run_id, epoch_id),
                ).fetchall()
                if item[0]
            }
            missing = sorted(required_templates - completed_templates)
            if missing:
                raise ValueError(
                    "cannot acknowledge escalation before compensating review completes: "
                    + ", ".join(missing)
                )
            conn.execute(
                "UPDATE epochs SET mutation_paused=0, escalation_state='acknowledged' "
                "WHERE run_id=? AND epoch_id=? AND closed_at IS NULL",
                (run_id, epoch_id),
            )
            conn.execute(
                "UPDATE escalation_events SET status='accepted', acknowledged_at=?, acknowledged_by=? "
                "WHERE run_id=? AND epoch_id=? AND status='applied' "
                "AND acknowledged_at IS NULL",
                (now, acknowledged_by[:256], run_id, epoch_id),
            )
            conn.commit()
            return self.get_escalation_state(run_id, epoch_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
