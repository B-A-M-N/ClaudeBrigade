#!/usr/bin/env python3
"""SessionEnd hook — closes the active run/epoch and releases bindings.

Called when a Claude Code session ends. Cleans up Brigade state while
preserving route history and evidence for audit.

Idempotent: safe to call multiple times.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))


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

    # Close active epoch
    active_epoch = state.get_active_epoch(run_id)
    if active_epoch:
        epoch_id = active_epoch["epoch_id"]
        state.create_route_snapshot(run_id, epoch_id, "session_end")
        state.close_epoch(run_id, epoch_id)

    # Close run (idempotent)
    state.close_run(run_id)

    # Clean up session compatibility files
    session = str(data.get("session_id", "unknown"))
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
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
    raise SystemExit(main())