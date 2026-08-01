#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Ensure enhanced_router is importable (supports both pip-installed and dev modes)
_bracket = os.environ.get("CLAUDE_BRIGADE_PYTHON")
if _bracket:
    _site = str(pathlib.Path(_bracket).parent / "site-packages")
    if _site not in sys.path:
        sys.path.insert(0, _site)
from workspace_fingerprint import fingerprint  # noqa: E402
from ledger_io import append_jsonl  # noqa: E402


def record_ledger(session_dir: pathlib.Path, record: dict) -> None:
    append_jsonl(session_dir / "ledger.jsonl", record)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session = str(data.get("session_id", "unknown"))
    session_dir = cache / "sessions" / session
    active_dir = session_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    current_fp = None
    try:
        current_fp = fingerprint(cwd, session_id=session)
    except Exception:
        pass

    source = str(data.get("source") or data.get("hook_event_name") or "startup")
    epoch_file = session_dir / "active_epoch_id.txt"
    baseline_file = session_dir / "active_epoch_baseline.txt"

    # --- Run/session registration only --------------------------------------
    # Task classification belongs to UserPromptSubmit/controller intake. A
    # session start must never freeze a default workflow before a task exists.
    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    sqlite_state_handled = False

    if run_id:
        try:
            from enhanced_router.state import get_state  # type: ignore
        except ImportError:
            pass  # package not available (e.g. development without pip install)
        else:
            state = get_state()
            session_id = str(data.get("session_id", "unknown"))
            cwd = str(pathlib.Path(data.get("cwd", ".")).resolve())
            try:
                state.create_run(run_id, session_id=session_id, cwd=cwd)
                # Register the canonical Git checkout before any Agent can
                # receive mutation authority.  This is a collision guard for
                # two runs sharing one checkout; it does not create a shadow
                # worktree or alter the user's dirty state.
                try:
                    from enhanced_router.shadow_worktree import ShadowWorktreeManager

                    manager = ShadowWorktreeManager(cwd)
                    baseline = manager.baseline()
                    state.register_main_workspace(
                        workspace_id=baseline.workspace_id,
                        run_id=str(run_id),
                        epoch_id="session-intake",
                        path=str(manager.repo_path),
                        base_sha=baseline.base_sha,
                        dirty_patch_hash=baseline.dirty_patch_hash,
                        baseline_untracked_files=list(baseline.untracked_files),
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    record_ledger(session_dir, {
                        "event": "WorkspaceRegistrationFailed",
                        "session_id": session_id,
                        "run_id": run_id,
                        "reason": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    sqlite_state_handled = False
                    return 0
                sqlite_state_handled = True
                active = state.get_active_epoch(run_id)
                if active:
                    active_epoch_id = str(active["epoch_id"])
                    epoch_id = active_epoch_id
                    if not epoch_file.exists() or epoch_file.read_text().strip() != epoch_id:
                        epoch_file.write_text(epoch_id, encoding="utf-8")
                        if current_fp and not baseline_file.exists():
                            baseline_file.write_text(current_fp, encoding="utf-8")
                    # Marker files are a projection of authoritative SQLite
                    # lifecycle state.  Resume/reconnect must preserve live
                    # workers and rehydrate missing projections instead of
                    # deleting every marker unconditionally.
                    live_statuses = {"started", "running", "streaming", "verifying"}
                    executions = state.get_agent_executions(
                        str(run_id), epoch_id=active_epoch_id,
                    )
                    live = {
                        str(item.get("claude_agent_id")): item
                        for item in executions
                        if item.get("claude_agent_id") and item.get("status") in live_statuses
                    }
                    for marker in active_dir.glob("*.json"):
                        if marker.name.startswith("spawn_"):
                            continue
                        try:
                            marker_agent_id = str(
                                json.loads(marker.read_text(encoding="utf-8")).get("agent_id")
                            )
                        except (OSError, ValueError, json.JSONDecodeError):
                            marker_agent_id = marker.stem
                        if marker_agent_id not in live:
                            marker.unlink(missing_ok=True)
                    for marker_agent_id, execution in live.items():
                        marker = active_dir / f"{marker_agent_id}.json"
                        if marker.exists():
                            try:
                                existing_marker = json.loads(
                                    marker.read_text(encoding="utf-8")
                                )
                                if (
                                    existing_marker.get("agent_id") == marker_agent_id
                                    and existing_marker.get("execution_id") == execution.get("execution_id")
                                ):
                                    continue
                            except (OSError, ValueError, json.JSONDecodeError):
                                pass
                        marker.write_text(json.dumps({
                            "event": "SessionResume",
                            "agent_id": marker_agent_id,
                            "agent_type": f"brigade-{execution.get('role', 'unknown')}",
                            "role": execution.get("role"),
                            "model": execution.get("model_id"),
                            "execution_id": execution.get("execution_id"),
                            "run_id": run_id,
                            "epoch_id": active_epoch_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }, separators=(",", ":")), encoding="utf-8")
                record_ledger(session_dir, {
                    "event": "SessionStart",
                    "session_id": session_id,
                    "epoch_id": active["epoch_id"] if active else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "workspace_fingerprint": current_fp,
                    "source": source,
                })
            except (OSError, RuntimeError):
                # State initialization failures are observable, but do not
                # pretend that an epoch exists.
                sqlite_state_handled = False

    # Legacy fallback records a session baseline only; it does not create a
    # workflow epoch before a user task is submitted.
    if not sqlite_state_handled:
        record_ledger(session_dir, {
                "event": "SessionStartUntracked",
                "session_id": session,
                "epoch_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "workspace_fingerprint": current_fp,
                "source": source,
            })

    return 0


if __name__ == "__main__":
    from _shared import fail_open_main
    raise SystemExit(fail_open_main(main))
