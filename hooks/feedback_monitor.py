#!/usr/bin/env python3
"""PostToolBatch monitor that injects bounded coprocessor feedback."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _shared import read_active_epoch_id, resolve_session_dir  # noqa: E402
from ledger_io import read_jsonl_cached  # noqa: E402


def _names(data: dict) -> set[str]:
    names: set[str] = set()
    value = data.get("tool_name")
    if isinstance(value, str) and value:
        names.add(value)
    for key in ("tool_uses", "tool_results"):
        values = data.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            for name_key in ("tool_name", "name"):
                name = item.get(name_key)
                if isinstance(name, str) and name:
                    names.add(name)
    return names


def _watched_tools() -> set[str]:
    """Read the projected monitor policy without querying the router hot path."""
    raw = os.environ.get("BRIGADE_FEEDBACK_WATCHED_TOOLS", "")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                values = {str(item) for item in parsed if str(item)}
                if values:
                    return values
        except ValueError:
            values = {item.strip() for item in raw.split(",") if item.strip()}
            if values:
                return values
    return {"Edit", "Write", "NotebookEdit", "Bash"}


def _checkpoint_name(data: dict) -> str:
    """Name semantic mutation checkpoints separately from generic batches."""
    if _names(data).intersection({"Edit", "Write", "NotebookEdit"}):
        return "post_edit_checkpoint"
    return "post_tool_batch"


def _request(data: dict, run_id: str, epoch_id: str, session_id: str) -> dict | None:
    base = (
        os.environ.get("BRIGADE_ROUTER_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "http://127.0.0.1:8787"
    ).rstrip("/")
    body = {
        "run_id": run_id,
        "epoch_id": epoch_id,
        "session_id": session_id,
        "agent_id": data.get("agent_id"),
        "hook_event_name": data.get("hook_event_name", "PostToolBatch"),
        "checkpoint": _checkpoint_name(data),
        "cwd": data.get("cwd"),
        "tool_name": data.get("tool_name"),
        "tool_input": data.get("tool_input"),
        "tool_response": data.get("tool_response"),
        "tool_uses": data.get("tool_uses"),
        "tool_results": data.get("tool_results"),
        "workspace_digest": _last_workspace_digest(resolve_session_dir(data)),
    }
    encoded = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/internal/feedback/checkpoint",
        data=encoded,
        headers={
            "Content-Type": "application/json",
            "X-Enhanced-Token": os.environ.get(
                "ENHANCED_ROUTER_TOKEN", os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            ),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            payload = json.loads(response.read(128_000).decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _last_workspace_digest(session_dir: pathlib.Path) -> str | None:
    """Read the latest audit fingerprint without recalculating the repository."""
    ledger = session_dir / "ledger.jsonl"
    if not ledger.exists():
        return None
    latest: str | None = None
    for event in read_jsonl_cached(ledger):
        if event.get("event") == "Mutation" and event.get("fingerprint"):
            latest = str(event["fingerprint"])
    return latest


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    if not isinstance(data, dict) or data.get("hook_event_name") not in {None, "PostToolBatch"}:
        return 0
    if not _names(data).intersection(_watched_tools()):
        # This is the hot path: no router call, SQLite access, or workspace
        # fingerprint when the batch cannot affect feedback relevance.
        return 0
    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    if not run_id:
        return 0
    session_id = str(data.get("session_id", "unknown"))
    session_dir = resolve_session_dir(data)
    epoch_id = read_active_epoch_id(session_dir)
    result = _request(data, str(run_id), epoch_id, session_id)
    if not result or result.get("status") != "completed" or not result.get("feedback"):
        return 0
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolBatch",
            "additionalContext": str(result["feedback"]),
        }
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
