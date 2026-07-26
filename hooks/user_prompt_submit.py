#!/usr/bin/env python3
"""UserPromptSubmit hook — creates a new epoch when the previous one is closed.

This hook is invoked by Claude Code on every user prompt submission. It ensures
that when a coding task completes and the epoch is closed, the next user prompt
automatically starts a fresh epoch with the default workflow/profile.

Idempotent: never creates a second open epoch when one already exists.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workspace_fingerprint import fingerprint


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    if not run_id:
        return 0  # not a Brigade session, skip

    session = str(data.get("session_id", "unknown"))
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
    session_dir = cache / "sessions" / session
    epoch_file = session_dir / "active_epoch_id.txt"
    baseline_file = session_dir / "active_epoch_baseline.txt"

    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    current_fp = None
    try:
        current_fp = fingerprint(cwd)
    except Exception:
        pass

    try:
        from enhanced_router.state import get_state
    except ImportError:
        return 0  # sidecar not installed, skip

    state = get_state()
    active = state.get_active_epoch(run_id)

    # If no active epoch exists (previous was closed), create one
    if active is None:
        epoch_id = f"ep_{uuid.uuid4().hex[:12]}"
        state.create_epoch_from_profile(run_id, epoch_id, "normal", "hybrid")

        # Write compatibility files
        session_dir.mkdir(parents=True, exist_ok=True)
        epoch_file.write_text(epoch_id, encoding="utf-8")
        if current_fp:
            baseline_file.write_text(current_fp, encoding="utf-8")

        # Record EpochStart event in JSONL ledger
        ledger_path = session_dir / "ledger.jsonl"
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "EpochStart",
                        "session_id": session,
                        "epoch_id": epoch_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "baseline_fingerprint": current_fp,
                        "source": "user_prompt_submit",
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())