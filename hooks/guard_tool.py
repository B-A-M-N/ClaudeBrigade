#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

MUTATORS = {"brigade-implementer", "brigade-repairer", "sonnet-direct"}
ALLOWED_SUBAGENTS = {"brigade-recon", "brigade-implementer", "brigade-adversary", "brigade-repairer", "sonnet-direct"}

MUTATING_BASH = re.compile(
    r"(?:^|[;&|]\s*)(?:rm|mv|cp|touch|mkdir|rmdir|truncate|install|patch|dd|ln)\b"
    r"|(?:^|\s)(?:sed\s+-i|perl\s+-pi|tee\b)"
    r"|(?:^|\s)git\s+(?:add|apply|am|checkout|clean|commit|merge|rebase|reset|restore|switch)\b"
    r"|(?:^|\s)(?:npm|pnpm|yarn|pip|pip3|uv|cargo)\s+(?:install|add|remove|update|fmt)\b"
    r"|(?:^|\s)(?:go\s+fmt|gofmt|rustfmt|prettier|black|ruff\s+format)\b"
    r"|(?:python|python3|node|perl|ruby)\s+-[ce]\s+.*(?:open|write|unlink|remove)"
    r"|(?<![<])>(?![>&])|>>",
    re.IGNORECASE,
)


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def record_ledger_event(session_id: str, event_type: str, details: dict) -> None:
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
    session_dir = cache / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    epoch_file = session_dir / "active_epoch_id.txt"
    epoch_id = epoch_file.read_text(encoding="utf-8").strip() if epoch_file.exists() else "ep_unknown"

    ledger_path = session_dir / "ledger.jsonl"
    record = {
        "event": event_type,
        "session_id": session_id,
        "epoch_id": epoch_id,
        "details": details,
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def lookup_agent_type(data: dict) -> str:
    agent_id = data.get("agent_id")
    if not agent_id:
        return "unknown"
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
    marker = cache / "sessions" / str(data.get("session_id", "unknown")) / "active" / f"{agent_id}.json"
    try:
        return str(json.loads(marker.read_text(encoding="utf-8")).get("agent_type", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


def main() -> int:
    data = json.load(sys.stdin)
    tool = str(data.get("tool_name", ""))
    agent_type = lookup_agent_type(data)
    session_id = str(data.get("session_id", "unknown"))

    if tool == "Agent":
        subagent_type = str((data.get("tool_input") or {}).get("subagent_type") or (data.get("tool_input") or {}).get("agent_type") or "")
        if subagent_type and subagent_type not in ALLOWED_SUBAGENTS:
            deny(f"Subagent role '{subagent_type}' is not in the authorized enhanced subagent allowlist.")
            return 0

    if tool in {"Write", "Edit", "NotebookEdit"}:
        if agent_type not in MUTATORS:
            deny(f"{agent_type} is read-only. Delegate source mutation to an authorized implementation agent.")
            return 0
        record_ledger_event(session_id, "MutationIntent", {"tool": tool, "agent_type": agent_type})

    if tool == "Bash":
        command = str((data.get("tool_input") or {}).get("command", ""))
        if MUTATING_BASH.search(command):
            if agent_type not in MUTATORS:
                deny(f"Mutating shell command blocked for read-only role {agent_type}.")
                return 0
            record_ledger_event(session_id, "MutationIntent", {"tool": "Bash", "command": command, "agent_type": agent_type})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
