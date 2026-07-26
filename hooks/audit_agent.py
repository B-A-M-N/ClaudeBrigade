#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone


sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workspace_fingerprint import fingerprint


def _clear_stale_markers(active_dir: pathlib.Path, log: pathlib.Path) -> None:
    """Remove leftover active-agent markers from a previously crashed session."""
    if log.exists():
        return
    for stale in active_dir.glob("*.json"):
        stale.unlink(missing_ok=True)


def record_ledger(session_dir: pathlib.Path, record: dict) -> None:
    ledger_path = session_dir / "ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    data = json.load(sys.stdin)
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
    session = str(data.get("session_id", "unknown"))
    session_dir = cache / "sessions" / session
    active_dir = session_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)

    epoch_file = session_dir / "active_epoch_id.txt"
    epoch_id = epoch_file.read_text(encoding="utf-8").strip() if epoch_file.exists() else "ep_unknown"

    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    current_fp = None
    try:
        current_fp = fingerprint(cwd)
    except Exception:
        pass

    event = data.get("hook_event_name")
    agent_id = str(data.get("agent_id", "unknown"))
    agent_type = str(data.get("agent_type", "unknown"))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "session_id": session,
        "epoch_id": epoch_id,
        "agent_id": agent_id,
        "agent_type": agent_type,
        "transcript": data.get("agent_transcript_path"),
        "fingerprint": current_fp,
    }

    log = session_dir / "agents.jsonl"

    if event == "SubagentStart":
        _clear_stale_markers(active_dir, log)

    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    record_ledger(session_dir, record)

    marker = active_dir / f"{agent_id}.json"
    if event == "SubagentStart":
        marker.write_text(json.dumps(record), encoding="utf-8")
    elif event == "SubagentStop":
        marker.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
