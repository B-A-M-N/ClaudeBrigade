"""Stable, human-readable orchestration projections.

Hooks and MCP return the same semantic states; this module keeps presentation
from reimplementing scheduler policy or exposing raw SQLite rows directly.
"""

from __future__ import annotations

import json
from typing import Any


def _safe_call(state: Any, method: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Read an optional projection without making presentation authoritative."""
    function = getattr(state, method, None)
    if not callable(function):
        return default
    try:
        return function(*args, **kwargs)
    except Exception:
        return default


def _phase_progress(phases: list[dict[str, Any]], executions: list[dict[str, Any]]) -> dict[str, Any]:
    active = next((item for item in phases if item.get("status") == "active"), None)
    if active is None:
        active = next((item for item in phases if item.get("status") == "pending"), None)
    if active is None:
        return {"phase_id": None, "completed": 0, "required": 0, "status": "idle"}
    phase_id = str(active.get("phase_id"))
    relevant = [item for item in executions if item.get("phase_id") == phase_id]
    completed = sum(item.get("status") == "completed" for item in relevant)
    required = int(active.get("required_successes") or active.get("quality_quorum") or 1)
    return {
        "phase_id": phase_id,
        "status": active.get("status"),
        "completed": completed,
        "required": required,
        "iteration": active.get("iteration", 0),
        "actor": active.get("actor") or active.get("required_actor") or "worker",
    }


def orchestration_snapshot(state: Any, run_id: str, epoch_id: str) -> dict[str, Any]:
    phases = state.get_workflow_phases(run_id, epoch_id)
    executions = state.get_agent_executions(run_id, epoch_id=epoch_id)
    packages = state.get_work_packages(run_id, epoch_id)
    coverage = state.get_requirement_coverage(run_id, epoch_id)
    findings = _safe_call(state, "get_findings", run_id, epoch_id=epoch_id, default=[])
    candidates = _safe_call(
        state, "get_integration_candidates", run_id=run_id, epoch_id=epoch_id, default=[]
    )
    workspaces = _safe_call(
        state, "get_workspaces", run_id=run_id, epoch_id=epoch_id, default=[]
    )
    run = _safe_call(state, "get_run", run_id, default={}) or {}
    token_reservations = _safe_call(
        state, "get_token_reservations", run_id, epoch_id=epoch_id, default=[]
    )
    provider_reservations = _safe_call(
        state, "get_provider_reservations", run_id=run_id, epoch_id=epoch_id,
        active_only=True, default=[]
    )
    run_capacity = _safe_call(
        state, "get_run_resource_capacity", run_id, epoch_id, default={}
    ) or {}
    phase_counts = {
        "completed": sum(item.get("status") == "completed" for item in phases),
        "active": sum(item.get("status") == "active" for item in phases),
        "pending": sum(item.get("status") == "pending" for item in phases),
        "failed": sum(item.get("status") == "failed" for item in phases),
    }
    required_phases = [item for item in phases if item.get("required")]
    required_incomplete = [
        item for item in required_phases
        if item.get("status") != "completed"
    ]
    package_counts = {
        status: sum(item.get("status") == status for item in packages)
        for status in (
            "ready", "claimed", "running", "completed", "integrated",
            "blocked", "retry", "cancelled",
        )
    }
    worker_states = []
    for item in executions:
        worker_states.append({
            "execution_id": item.get("execution_id"),
            "phase_id": item.get("phase_id"),
            "package_id": item.get("package_id"),
            "worker": item.get("native_agent_name") or item.get("worker_id") or item.get("role"),
            "kind": "coprocessor" if item.get("execution_kind") in {"coprocessor_call", "sidecar_call"}
            else "native",
            "status": item.get("status"),
            "orphaned": bool(item.get("orphaned_at")) or item.get("error_class") == "orphaned_after_restart",
            "orphaned_at": item.get("orphaned_at"),
            "accepted": item.get("accepted_by_controller"),
            "disposition": item.get("result_disposition"),
            "quality_score": item.get("quality_score"),
            "integrated": item.get("integrated"),
            "verified": item.get("verified"),
            "worker_kind": item.get("worker_kind"),
            "model_id": item.get("model_id"),
            "provider_id": item.get("provider_id"),
        })
    finding_counts = {
        "open": sum(item.get("status") not in {"resolved", "closed"} for item in findings),
        "accepted": sum(item.get("disposition") == "accepted" for item in findings),
        "unresolved_accepted": sum(
            item.get("disposition") == "accepted"
            and item.get("verification_status") != "verified"
            for item in findings
        ),
    }
    candidate_counts = {
        disposition: sum(item.get("disposition") == disposition for item in candidates)
        for disposition in ("green", "yellow", "red", "pending", "applied")
    }
    blockers: list[str] = []
    if not phases:
        blockers.append("workflow phases have not been initialized")
    missing = coverage.get("missing_mandatory") or coverage.get("missing") or []
    if missing:
        blockers.append(f"missing requirements: {len(missing)}")
    if required_incomplete:
        blockers.append(f"required phases incomplete: {len(required_incomplete)}")
    wait_reasons: list[str] = []
    phase_by_id = {str(item.get("phase_id")): item for item in phases}
    for phase in phases:
        if phase.get("status") not in {"pending", "active"}:
            continue
        phase_id = str(phase.get("phase_id"))
        try:
            dependencies = json.loads(str(phase.get("dependencies_json") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            dependencies = []
        waiting_dependencies = [
            str(dependency) for dependency in dependencies
            if (phase_by_id.get(str(dependency)) or {}).get("status") not in {"completed", "skipped"}
        ]
        if waiting_dependencies:
            wait_reasons.append(
                f"{phase_id} waiting for {', '.join(waiting_dependencies)}"
            )
        if (
            phase.get("mutating")
            and str(phase.get("fanout_from") or "") == "work_packages"
            and not any(item.get("phase_id") == phase_id for item in packages)
        ):
            wait_reasons.append(f"{phase_id} waiting for controller-published work packages")
    if not wait_reasons and not executions and required_incomplete:
        wait_reasons.append("waiting for the next admitted action")
    if finding_counts["unresolved_accepted"]:
        blockers.append(f"accepted findings: {finding_counts['unresolved_accepted']}")
    pending_candidates = candidate_counts["yellow"] + candidate_counts["red"] + candidate_counts["pending"]
    if pending_candidates:
        blockers.append(f"integration candidates: {pending_candidates}")
    active_shadows = sum(
        item.get("kind") == "shadow" and item.get("status") in {"active", "ready"}
        for item in workspaces
    )
    if active_shadows:
        blockers.append(f"shadow worktrees: {active_shadows}")
    escalation = _safe_call(state, "get_escalation_state", run_id, epoch_id, default={}) or {}
    epoch = escalation.get("epoch") or {}
    if epoch.get("mutation_paused") or epoch.get("escalation_state") == "escalated":
        blockers.append("mutation paused for escalation acknowledgment")
    for reason in run_capacity.get("blocked") or []:
        if reason not in blockers:
            blockers.append(str(reason))
    phase_progress = _phase_progress(phases, executions)
    token_spent = sum(int(item.get("total_tokens") or 0) for item in executions)
    token_reserved = sum(int(item.get("estimated_tokens") or 0) for item in token_reservations)
    provider_capacity: dict[str, dict[str, int | None]] = {}
    try:
        from enhanced_router.registry import get_registry
        registry = get_registry()
        for provider_id, provider in registry.providers.items():
            rows = [item for item in provider_reservations if item.get("provider_id") == provider_id]
            provider_capacity[provider_id] = {
                "active": sum(item.get("state") == "reserved" for item in rows),
                "queued": sum(item.get("state") == "queued" for item in rows),
                "limit": provider.limits.max_active_agents,
            }
    except Exception:
        for provider_id in sorted({str(item.get("provider_id")) for item in provider_reservations}):
            rows = [item for item in provider_reservations if item.get("provider_id") == provider_id]
            provider_capacity[provider_id] = {
                "active": sum(item.get("state") == "reserved" for item in rows),
                "queued": sum(item.get("state") == "queued" for item in rows),
                "limit": None,
            }
    return {
        "run_id": run_id,
        "epoch_id": epoch_id,
        "phase_counts": phase_counts,
        "phases": [
            {
                "id": item.get("phase_id"),
                "status": item.get("status"),
                "actor": item.get("actor") or item.get("required_actor") or "worker",
                "required": bool(item.get("required")),
                "iteration": item.get("iteration", 0),
                "produces": item.get("produces"),
                "fanout_from": item.get("fanout_from"),
            }
            for item in phases
        ],
        "packages": [
            {
                "id": item.get("package_id"),
                "objective": item.get("objective"),
                "status": item.get("status"),
                "risk": item.get("risk"),
            }
            for item in packages
        ],
        "package_counts": package_counts,
        "package_progress": {
            "total": len(packages),
            "ready": package_counts["ready"],
            "active": package_counts["claimed"] + package_counts["running"],
            "completed": package_counts["completed"] + package_counts["integrated"],
            "blocked": package_counts["blocked"] + package_counts["retry"],
        },
        "coverage": {
            "covered": coverage.get("covered", 0),
            "total": coverage.get("total", 0),
            "complete": coverage.get("complete", True),
            "missing": coverage.get("missing_mandatory", []),
        },
        "ambiguities": state.get_ambiguities(run_id, epoch_id),
        "workers": worker_states,
        "findings": finding_counts,
        "integration": {
            "counts": candidate_counts,
            "pending": pending_candidates,
        },
        "workspaces": {
            "active_shadows": active_shadows,
            "active": sum(item.get("status") == "active" for item in workspaces),
            "ready": sum(item.get("status") == "ready" for item in workspaces),
        },
        "phase_progress": phase_progress,
        "blockers": blockers,
        "wait_reasons": wait_reasons,
        "resources": {
            "providers": provider_capacity,
            "run_policy": run_capacity.get("policy") or run.get("resource_policy"),
            "run_capacity": run_capacity,
            "token_budget": run.get("token_budget"),
            "token_spent": token_spent,
            "token_reserved": token_reserved,
        },
        "completion_state": {
            "required_phases_complete": not required_incomplete,
            "agents_completed": sum(item.get("status") == "completed" for item in executions),
            "accepted": sum(bool(item.get("accepted_by_controller")) for item in executions),
            "integrated": sum(bool(item.get("integrated")) for item in executions),
            "verified": sum(bool(item.get("verified")) for item in executions),
            "project_complete": (
                bool(phases)
                and
                not required_incomplete
                and not blockers
                and bool(coverage.get("complete", True))
            ),
        },
        "escalation": escalation,
    }


def compact_status(snapshot: dict[str, Any]) -> str:
    counts = snapshot.get("phase_counts") or {}
    coverage = snapshot.get("coverage") or {}
    phase = f"phases {counts.get('completed', 0)}/{sum(counts.get(key, 0) for key in ('completed', 'active', 'pending', 'failed'))}"
    requirements = f"req {coverage.get('covered', 0)}/{coverage.get('total', 0)}"
    packages = snapshot.get("packages") or []
    active_packages = sum(item.get("status") in {"claimed", "running"} for item in packages)
    progress = snapshot.get("phase_progress") or {}
    phase_name = progress.get("phase_id") or "idle"
    findings = snapshot.get("findings") or {}
    blockers = snapshot.get("blockers") or []
    wait_reasons = snapshot.get("wait_reasons") or []
    suffix = f" | phase {phase_name}"
    if findings.get("unresolved_accepted"):
        suffix += f" | findings {findings['unresolved_accepted']}"
    if blockers:
        suffix += f" | blocked {len(blockers)}"
    elif wait_reasons:
        suffix += f" | waiting {len(wait_reasons)}"
    return f"{phase} | {requirements} | packages {active_packages}/{len(packages)}{suffix}"
