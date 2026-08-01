#!/usr/bin/env python3
"""SessionEnd hook — closes the active run/epoch and releases bindings.

Called when a Claude Code session ends. Cleans up Brigade state while
preserving route history and evidence for audit.

Idempotent: safe to call multiple times.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys

LOGGER = logging.getLogger(__name__)

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Ensure enhanced_router is importable (supports both pip-installed and dev modes)
_bracket = os.environ.get("CLAUDE_BRIGADE_PYTHON")
if _bracket:
    _site = str(pathlib.Path(_bracket).parent / "site-packages")
    if _site not in sys.path:
        sys.path.insert(0, _site)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    if not run_id:
        return 0  # not a Brigade session, skip

    try:
        from enhanced_router.state import get_state
    except ImportError:
        return 0  # sidecar not installed, skip

    state = get_state()

    # Close active epoch and run — wrapped to prevent hook crash
    try:
        # Close active epoch
        active_epoch = state.get_active_epoch(run_id)
        if active_epoch:
            epoch_id = active_epoch["epoch_id"]
            state.create_route_snapshot(run_id, epoch_id, "session_end")
            state.close_epoch(run_id, epoch_id)

        # Close run (idempotent)
        state.close_run(run_id)
        try:
            state.cancel_action_claims(str(run_id))
        except AttributeError:
            pass
        # Release canonical ownership and discard any abandoned shadow
        # worktrees.  Completed changesets are already integrated; no
        # unreviewed shadow is silently copied during session teardown.
        for workspace in state.get_workspaces(run_id=str(run_id)):
            if workspace.get("status") not in {"active", "ready"}:
                continue
            if workspace.get("kind") == "shadow":
                try:
                    from enhanced_router.shadow_worktree import ShadowWorktreeManager

                    run = state.get_run(str(run_id))
                    if run and run.get("cwd"):
                        ShadowWorktreeManager(str(run["cwd"])).remove_shadow(workspace["path"])
                except Exception:
                    LOGGER.warning("unable to remove abandoned shadow %s", workspace.get("path"))
            state.update_workspace_status(str(workspace["workspace_id"]), "discarded")
        try:
            from enhanced_router.registry import get_registry

            registry = get_registry()
            for reservation in state.get_provider_reservations(active_only=True):
                if reservation.get("run_id") != str(run_id):
                    continue
                state.release_provider_reservation(str(reservation["reservation_id"]), "cancelled")
                provider = registry.providers.get(str(reservation.get("provider_id")))
                if provider:
                    state.admit_provider_agents(
                        str(reservation["provider_id"]), provider.limits.max_active_agents
                    )
        except Exception:
            LOGGER.warning("unable to release all provider reservations for run %s", run_id)
    except Exception:
        LOGGER.exception("Failed to close state in session_end")
        # Don't re-raise — allow session to end gracefully

    # Clean up session compatibility files
    session = str(data.get("session_id", "unknown"))
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session_dir = cache / "sessions" / session

    # Remove ephemeral marker files but keep history (ledger.jsonl, agents.jsonl)
    for name in ("active_epoch_id.txt", "active_epoch_baseline.txt", "stop_hook_retry_count.txt"):
        marker = session_dir / name
        if marker.exists():
            marker.unlink()

    # Remove active agent markers
    active_dir = session_dir / "active"
    if active_dir.exists():
        for stale in active_dir.glob("*.json"):
            stale.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    from _shared import fail_open_main
    raise SystemExit(fail_open_main(main))
