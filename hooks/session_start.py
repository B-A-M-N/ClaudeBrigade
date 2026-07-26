#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workspace_fingerprint import fingerprint


def record_ledger(session_dir: pathlib.Path, record: dict) -> None:
    ledger_path = session_dir / "ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
    session = str(data.get("session_id", "unknown"))
    session_dir = cache / "sessions" / session
    active_dir = session_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale markers from any prior crashed session
    for stale in active_dir.glob("*.json"):
        stale.unlink(missing_ok=True)

    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    current_fp = None
    try:
        current_fp = fingerprint(cwd)
    except Exception:
        pass

    source = str(data.get("source") or data.get("hook_event_name") or "startup")
    epoch_file = session_dir / "active_epoch_id.txt"
    baseline_file = session_dir / "active_epoch_baseline.txt"

    # --- Run / Epoch bootstrap via SQLite RouteState ----------------
    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    if run_id:
        try:
            from enhanced_router.state import RouteState, get_state  # type: ignore
        except ImportError:
            pass  # package not installed yet, skip SQLite bootstrap
        else:
            state = get_state()
            state.create_run(run_id, session, str(cwd))  # idempotent
            active = state.get_active_epoch(run_id)
            if active is None:
                epoch_id = f"ep_{uuid.uuid4().hex[:12]}"
                state.create_epoch_from_profile(
                    run_id, epoch_id, "normal", "hybrid"
                )
                # Use the epoch_id just created for backward-compat file
                epoch_file.write_text(epoch_id, encoding="utf-8")
                if current_fp:
                    baseline_file.write_text(current_fp, encoding="utf-8")
                    record_ledger(session_dir, {
                        "event": "EpochStart",
                        "session_id": session,
                        "epoch_id": epoch_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "baseline_fingerprint": current_fp,
                        "source": source,
                    })

    # Start a new epoch on startup or clear, or if no active epoch exists
    should_start_new_epoch = (
        source in {"startup", "clear"} or
        not epoch_file.exists() or
        not baseline_file.exists()
    )

    if source == "compact":
        # Compaction must never reset active epoch baseline
        should_start_new_epoch = False

    if should_start_new_epoch and current_fp:
        epoch_id = f"ep_{uuid.uuid4().hex[:12]}"
        epoch_file.write_text(epoch_id, encoding="utf-8")
        baseline_file.write_text(current_fp, encoding="utf-8")

        record_ledger(session_dir, {
            "event": "EpochStart",
            "session_id": session,
            "epoch_id": epoch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_fingerprint": current_fp,
            "source": source,
        })
    elif source == "resume" and epoch_file.exists():
        # Record resume event under existing epoch
        epoch_id = epoch_file.read_text(encoding="utf-8").strip()
        record_ledger(session_dir, {
            "event": "SessionResume",
            "session_id": session,
            "epoch_id": epoch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_fingerprint": current_fp,
            "source": source,
        })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
