#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import logging
import os
import pathlib
import sys
from datetime import datetime, timezone

LOGGER = logging.getLogger(__name__)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workspace_fingerprint import fingerprint  # noqa: E402
from ledger_io import append_jsonl  # noqa: E402
from enhanced_router.base import MUTATORS  # noqa: E402


def record_ledger(session_dir: pathlib.Path, record: dict) -> None:
    append_jsonl(session_dir / "ledger.jsonl", record)


def _role(agent_type: str) -> str:
    if agent_type == "controller-direct":
        return "controller"
    return next((candidate for candidate in ("recon", "implementer", "adversary", "repairer") if candidate in agent_type), "recon")


class WorkspaceIsolationError(RuntimeError):
    """Raised when a mutating native agent is not in an owned shadow."""


def _execution_workspace(state: object, run_id: str, epoch_id: str, execution_id: str,
                         cwd: pathlib.Path, agent_type: str) -> str | None:
    """Register the actual Claude worktree used by this execution.

    Claude Code owns native ``isolation: worktree`` creation.  The hook does
    not create a second worktree; it records the child worktree and lets the
    router integrate its changeset when the child terminates.
    """
    if agent_type not in MUTATORS:
        return None
    from enhanced_router.shadow_worktree import ShadowWorktreeManager

    run = state.get_run(run_id)  # type: ignore[attr-defined]
    if not run or not run.get("cwd"):
        raise WorkspaceIsolationError("run has no registered canonical workspace")
    main_path = pathlib.Path(str(run["cwd"])).resolve()
    main_manager = ShadowWorktreeManager(main_path)
    main_baseline = main_manager.baseline()
    main_row = state.register_main_workspace(  # type: ignore[attr-defined]
        workspace_id=main_baseline.workspace_id,
        run_id=run_id,
        epoch_id=epoch_id,
        path=str(main_path),
        base_sha=main_baseline.base_sha,
        dirty_patch_hash=main_baseline.dirty_patch_hash,
        baseline_untracked_files=list(main_baseline.untracked_files),
    )
    child_root = ShadowWorktreeManager.repository_root(cwd)
    if child_root == main_path:
        raise WorkspaceIsolationError(
            "mutating native agent resolved to the canonical checkout"
        )
    if not main_manager.is_registered_worktree(child_root):
        raise WorkspaceIsolationError(
            "mutating native agent is not a Git worktree registered by the canonical checkout"
        )

    child_baseline = ShadowWorktreeManager(child_root).baseline()
    expected_base = str(main_row.get("current_base_sha") or main_row.get("base_sha") or "")
    expected_dirty = str(
        main_row.get("current_dirty_hash")
        or main_row.get("dirty_patch_hash")
        or ""
    )
    if child_baseline.base_sha != expected_base:
        raise WorkspaceIsolationError(
            "mutating native worktree is based on a different canonical revision"
        )
    if child_baseline.dirty_patch_hash != expected_dirty:
        raise WorkspaceIsolationError(
            "mutating native worktree does not contain the canonical dirty baseline"
        )
    expected_untracked = set(
        json.loads(str(main_row.get("baseline_untracked_json") or "[]"))
    )
    if not expected_untracked.issubset(set(child_baseline.untracked_files)):
        raise WorkspaceIsolationError(
            "mutating native worktree is missing files from the canonical baseline"
        )
    workspace_id = "shadow-native-" + hashlib.sha256(
        str(child_root).encode("utf-8")
    ).hexdigest()[:24]
    existing = state.get_workspace(workspace_id)  # type: ignore[attr-defined]
    if existing is None:
        state.create_workspace(  # type: ignore[attr-defined]
            workspace_id=workspace_id,
            run_id=run_id,
            epoch_id=epoch_id,
            kind="shadow",
            path=str(child_root),
            base_sha=child_baseline.base_sha,
            dirty_patch_hash=child_baseline.dirty_patch_hash,
            status="active",
            owner_execution_id=execution_id,
            baseline_untracked_files=list(child_baseline.untracked_files),
            parent_canonical_generation=int(main_row.get("canonical_generation") or 0),
            parent_dirty_patch_hash=str(
                main_row.get("current_dirty_hash") or main_row.get("dirty_patch_hash") or ""
            ),
        )
        existing = state.get_workspace(workspace_id)  # type: ignore[attr-defined]
    if (
        existing is None
        or existing.get("kind") != "shadow"
        or existing.get("status") != "active"
        or existing.get("owner_execution_id") != execution_id
        or pathlib.Path(str(existing.get("path"))).resolve() != child_root
    ):
        raise WorkspaceIsolationError(
            "shadow workspace is missing, inactive, or owned by another execution"
        )
    return workspace_id


def _finalize_changeset(state: object, run_id: str, epoch_id: str, execution: dict,
                        status: str) -> str | None:
    """Extract and, for a green completed child, integrate its changeset."""
    workspace_id = execution.get("workspace_id")
    if not workspace_id:
        return None
    workspace = state.get_workspace(str(workspace_id))  # type: ignore[attr-defined]
    if not workspace or workspace.get("kind") != "shadow":
        return None
    run = state.get_run(run_id)  # type: ignore[attr-defined]
    if not run or not run.get("cwd"):
        raise RuntimeError("run has no canonical workspace path")
    from enhanced_router.shadow_worktree import ShadowWorktreeManager, ShadowWorkspace

    manager = ShadowWorktreeManager(str(run["cwd"]))
    shadow = ShadowWorkspace(
        workspace_id=str(workspace["workspace_id"]),
        path=pathlib.Path(str(workspace["path"])),
        base_sha=str(workspace["base_sha"]),
        dirty_patch_hash=str(workspace.get("dirty_patch_hash") or ""),
        run_id=run_id,
        epoch_id=epoch_id,
        execution_id=str(execution["execution_id"]),
        baseline_untracked_files=tuple(
            json.loads(workspace.get("baseline_untracked_json") or "[]")
        ),
        parent_canonical_generation=(
            int(workspace["parent_canonical_generation"])
            if workspace.get("parent_canonical_generation") is not None else None
        ),
    )
    try:
        changeset = manager.extract_changeset(state=state, workspace=shadow)
        classification = manager.classify_overlap(
            state=state, run_id=run_id, epoch_id=epoch_id, changeset=changeset,
        )
        if status == "completed" and classification["disposition"] == "green":
            advisory = _fastpath_merge_advice(
                run_id=run_id,
                epoch_id=epoch_id,
                changeset=changeset,
                overlap=classification["overlap"],
            )
            if advisory in {"fail", "escalate"}:
                state.mark_integration_candidate(  # type: ignore[attr-defined]
                    changeset.changeset_id,
                    disposition="yellow",
                    validation={
                        **changeset.validation,
                        "fastpath_decision": advisory,
                    },
                )
                state.update_workspace_status(str(workspace_id), "ready")  # type: ignore[attr-defined]
                return str(changeset.changeset_id)
            main_rows = state.get_workspaces(  # type: ignore[attr-defined]
                run_id=run_id, epoch_id=epoch_id, kind="main", status="active"
            )
            if not main_rows:
                raise RuntimeError("canonical workspace is not registered")
            manager.integrate_green(
                state=state, run_id=run_id, epoch_id=epoch_id, changeset=changeset,
                expected_dirty_patch_hash=str(
                    main_rows[0].get("current_dirty_hash")
                    or main_rows[0].get("dirty_patch_hash")
                    or ""
                ),
            )
            state.update_workspace_status(str(workspace_id), "merged")  # type: ignore[attr-defined]
        elif status == "completed":
            state.update_workspace_status(str(workspace_id), "ready")  # type: ignore[attr-defined]
        else:
            state.update_workspace_status(str(workspace_id), "discarded")  # type: ignore[attr-defined]
        return str(changeset.changeset_id)
    finally:
        try:
            manager.remove_shadow(shadow.path)
        except Exception as exc:
            LOGGER.warning("unable to remove completed shadow worktree: %s", exc)


def _fastpath_merge_advice(
    *, run_id: str, epoch_id: str, changeset: object, overlap: dict,
) -> str | None:
    """Ask the optional DiffusionGemma verifier for merge-risk advice.

    This is deliberately advisory.  A PASS does not authorize integration;
    the deterministic ``git apply --check`` in ``integrate_green`` remains the
    authority.  FAIL/ESCALATE only turns an otherwise green candidate into a
    controller-review yellow candidate.
    """
    import urllib.error
    import urllib.request

    patch = getattr(changeset, "patch", b"")
    if not isinstance(patch, bytes):
        return None
    max_diff_bytes = 24_000
    packet = {
        "run_id": run_id,
        "epoch_id": epoch_id,
        "verification_id": f"merge:{getattr(changeset, 'changeset_id', 'unknown')}",
        "packet": {
            "contract": {"purpose": "shadow changeset integration"},
            "deterministic_checks": {
                "changeset_valid": bool(getattr(changeset, "validation", {}).get("valid")),
                "overlap_free": not bool(overlap.get("overlaps")),
                "base_identity": True,
            },
            "changed_files": list(getattr(changeset, "changed_files", ())),
            "diff": patch[:max_diff_bytes].decode("utf-8", "replace"),
            "diff_truncated": len(patch) > max_diff_bytes,
            "claims": ["worker changeset is ready for deterministic preflight"],
            "findings": [],
        },
    }
    base = os.environ.get("BRIGADE_ROUTER_URL", "http://127.0.0.1:8787").rstrip("/")
    request = urllib.request.Request(
        f"{base}/internal/fastpath/verify",
        data=json.dumps(packet, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Enhanced-Token": os.environ.get("ENHANCED_ROUTER_TOKEN", ""),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=0.35) as response:
            payload = json.loads(response.read(16_384).decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    decision = payload.get("decision") if isinstance(payload, dict) else None
    return decision if decision in {"pass", "fail", "escalate"} else None


def _close_execution(run_id: str, agent_id: str, status: str, summary: str) -> None:
    from enhanced_router.registry import get_registry
    from enhanced_router.state import get_state

    state = get_state()
    executions = state.get_agent_executions(run_id, status="started") + state.get_agent_executions(run_id, status="running")
    execution = next((item for item in executions if item.get("claude_agent_id") == agent_id), None)
    if execution:
        try:
            changeset_id = _finalize_changeset(state, run_id, str(execution["epoch_id"]), execution, status)
            if changeset_id:
                summary = f"{summary}\nchangeset={changeset_id}".strip()
        except Exception as exc:
            LOGGER.warning("unable to finalize execution changeset: %s", exc)
            if status == "completed":
                status = "failed"
                summary = f"{summary}\nchangeset integration failed: {exc}".strip()
        state.update_agent_execution(execution["execution_id"], status=status, result_summary=summary[:2_000])
        try:
            state.finish_spawn_assignment(
                run_id, str(execution["epoch_id"]), agent_id, status,
            )
        except (AttributeError, ValueError):
            LOGGER.warning("unable to close native action assignment for %s", agent_id)
    binding = state.get_agent_binding(run_id, agent_id)
    if binding:
        state.release_binding(run_id, agent_id)
    # Mutation authority is independent from route binding.  Always release it
    # on every terminal lifecycle outcome so a failed or cancelled mutator
    # cannot strand the workspace for the remainder of the run.
    state.release_mutation_lease(run_id, agent_id)
    if execution and execution.get("phase_id"):
        state.complete_phase_if_ready(
            run_id, str(execution["epoch_id"]), str(execution["phase_id"])
        )
    for reservation in state.get_provider_reservations(active_only=True):
        if reservation.get("execution_id") in {execution.get("execution_id") if execution else None, f"pending:{agent_id}"}:
            state.release_provider_reservation(reservation["reservation_id"], "released")
            provider = get_registry().providers.get(str(reservation["provider_id"]))
            if provider:
                state.admit_provider_agents(str(reservation["provider_id"]), provider.limits.max_active_agents)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session = str(data.get("session_id", "unknown"))
    session_dir = cache / "sessions" / session
    active_dir = session_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)
    epoch_file = session_dir / "active_epoch_id.txt"
    epoch_id = epoch_file.read_text(encoding="utf-8").strip() if epoch_file.exists() else "ep_unknown"
    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    try:
        current_fp = fingerprint(cwd, session_id=session, epoch_id=epoch_id)
    except Exception:
        current_fp = None
    event = str(data.get("hook_event_name") or "")
    agent_id = str(data.get("agent_id") or "unknown")
    agent_type = str(data.get("agent_type") or data.get("subagent_type") or "unknown")
    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID", "")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "event": event,
        "session_id": session, "epoch_id": epoch_id, "agent_id": agent_id,
        "agent_type": agent_type, "transcript": data.get("agent_transcript_path"),
        "fingerprint": current_fp, "run_id": run_id, "role": _role(agent_type),
        "public_model_alias": data.get("model") or "",
        "pinned_backing_model": data.get("resolved_model") or "",
    }
    if event == "SubagentStart" and run_id:
        try:
            from enhanced_router.registry import get_registry
            from enhanced_router.state import get_state
            from enhanced_router.backends import ROLE_MODEL_BINDINGS
            state = get_state()
            epoch = state.get_active_epoch(run_id)
            if epoch is None:
                raise RuntimeError("SubagentStart has no active Brigade epoch")
            role = _role(agent_type)
            phase = state.prepare_agent_phase(run_id, epoch["epoch_id"], role, agent_id)
            route = state.get_role_route(run_id, epoch["epoch_id"], role) if role != "controller" else None
            model_id = ROLE_MODEL_BINDINGS.get(
                agent_type,
                str(route.get("model_id")) if route else str(data.get("resolved_model") or "controller"),
            )
            spawn_intent: dict | None = None
            claim = state.get_pending_spawn_claim(  # type: ignore[attr-defined]
                run_id, str(epoch["epoch_id"]), agent_type, role,
            )
            if claim is None:
                raise RuntimeError(
                    "SubagentStart has no uniquely claimed native action; refusing to bind"
                )
            model_id = str(claim["model_id"])
            spawn_intent = {
                "action_id": claim["action_id"],
                "intent_id": claim.get("intent_id"),
                "reservation_id": claim.get("reservation_id"),
                "claim_token": claim.get("claim_token"),
                "spawn_call_id": claim.get("spawn_call_id"),
            }
            execution_id = f"exec:{agent_id}"
            workspace_id = _execution_workspace(
                state, run_id, epoch_id, execution_id, cwd, agent_type,
            )
            attached = state.attach_spawned_agent(  # type: ignore[attr-defined]
                run_id=run_id,
                epoch_id=str(epoch["epoch_id"]),
                native_agent_name=agent_type,
                role=role,
                model_id=model_id,
                claude_agent_id=agent_id,
                execution_id=execution_id,
                workspace_id=workspace_id,
                phase_id=phase.get("phase_id") if phase else None,
                provider_id=(
                    get_registry().get_model(model_id).provider_id
                    if model_id != "unknown" else None
                ),
                spawn_call_id=str(claim.get("spawn_call_id") or "") or None,
                claim_token=str(claim.get("claim_token") or "") or None,
            )
            execution = attached["execution"]
            record.update({"execution_id": execution_id, "role": role, "pinned_backing_model": model_id})
            for marker in active_dir.glob("spawn_*.json"):
                try:
                    candidate = json.loads(marker.read_text(encoding="utf-8"))
                    if candidate.get("action_id") == claim.get("action_id"):
                        marker.unlink(missing_ok=True)
                        break
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        except Exception as exc:
            LOGGER.warning("unable to attach authoritative agent execution: %s", exc)
            if run_id:
                try:
                    if "claim" in locals() and claim:
                        state.fail_spawn_claim(
                            str(claim["action_id"]),
                            status="failed",
                            reason=str(exc),
                        )
                except Exception:
                    LOGGER.exception("unable to close failed spawn claim")
    append_jsonl(session_dir / "agents.jsonl", record)
    record_ledger(session_dir, record)
    marker = active_dir / f"{agent_id}.json"
    if event == "SubagentStart":
        try:
            tmp = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
            tmp.replace(marker)
        except OSError:
            LOGGER.exception("Failed to write agent marker for %s", agent_id)
    elif event in {"SubagentStop", "StopFailure"}:
        status = "failed" if event == "StopFailure" else str(data.get("status") or "completed")
        if status not in {"completed", "failed", "timeout", "cancelled"}:
            status = "completed"
        if run_id:
            try:
                _close_execution(run_id, agent_id, status, str(data.get("summary") or ""))
            except Exception as exc:
                LOGGER.warning("unable to close authoritative execution: %s", exc)
        marker.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
