#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workspace_fingerprint import fingerprint

MUTATORS = {"longcat-implementer", "longcat-repairer", "sonnet-direct"}
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

TEST_BASH = re.compile(
    r"\b(?:pytest|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|cargo\s+test|go\s+test|make\s+test|python3?\s+-m\s+pytest)\b",
    re.IGNORECASE,
)

# Detect shell composition that overrides failure exit codes
SHELL_MASKING = re.compile(r"\|\|\s*(?:true|echo|exit\s+0)\b|;\s*echo\b", re.IGNORECASE)


def record_ledger(session_dir: pathlib.Path, record: dict) -> None:
    ledger_path = session_dir / "ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def lookup_agent_type(session_dir: pathlib.Path, agent_id: str | None) -> str:
    if not agent_id:
        return "unknown"
    marker = session_dir / "active" / f"{agent_id}.json"
    try:
        return str(json.loads(marker.read_text(encoding="utf-8")).get("agent_type", "unknown"))
    except Exception:
        return "unknown"


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
    session = str(data.get("session_id", "unknown"))
    session_dir = cache / "sessions" / session
    session_dir.mkdir(parents=True, exist_ok=True)

    epoch_file = session_dir / "active_epoch_id.txt"
    epoch_id = epoch_file.read_text(encoding="utf-8").strip() if epoch_file.exists() else "ep_unknown"

    event_name = data.get("hook_event_name", "PostToolUse")
    tool_name = str(data.get("tool_name", ""))
    tool_input = data.get("tool_input") or {}
    tool_response = data.get("tool_response") or {}
    is_failure = (event_name == "PostToolUseFailure")
    agent_id = data.get("agent_id")
    agent_type = lookup_agent_type(session_dir, agent_id)

    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    current_fp = None
    try:
        current_fp = fingerprint(cwd)
    except Exception:
        pass

    # Confirmed Mutation logging on PostToolUse for successful file edits or mutating commands
    if not is_failure:
        if tool_name in {"Write", "Edit", "NotebookEdit"}:
            record_ledger(session_dir, {
                "event": "Mutation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session,
                "epoch_id": epoch_id,
                "tool": tool_name,
                "agent_type": agent_type,
                "fingerprint": current_fp,
            })
        elif tool_name == "Bash":
            command = str(tool_input.get("command", ""))
            if MUTATING_BASH.search(command):
                record_ledger(session_dir, {
                    "event": "Mutation",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": session,
                    "epoch_id": epoch_id,
                    "tool": "Bash",
                    "command": command,
                    "agent_type": agent_type,
                    "fingerprint": current_fp,
                })

    if tool_name == "Bash":
        command = str(tool_input.get("command", ""))
        if TEST_BASH.search(command):
            exit_code = tool_response.get("exit_code") if isinstance(tool_response, dict) else None
            if exit_code is None and is_failure:
                exit_code = 1
            elif exit_code is None:
                exit_code = 0

            # Reject shell composition masking failure
            if SHELL_MASKING.search(command):
                masked = True
                status_event = "TestExecutionFailure"
            else:
                masked = False
                status_event = "TestExecutionFailure" if (is_failure or exit_code != 0) else "TestExecutionSuccess"

            record_ledger(session_dir, {
                "event": status_event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session,
                "epoch_id": epoch_id,
                "command": command,
                "exit_code": exit_code,
                "masked": masked,
                "agent_id": agent_id,
                "fingerprint": current_fp,
            })

    elif tool_name == "Agent":
        subagent_type = str(tool_input.get("subagent_type") or tool_input.get("agent_type") or "unknown")

        # Claude Code documents successful foreground agent tool response status as "completed"
        raw_status = str(tool_response.get("status") or "")
        if is_failure:
            status = "error"
        elif raw_status in {"completed", "success"}:
            status = "completed"
        else:
            status = raw_status or "completed"

        resolved_model = str(tool_response.get("resolvedModel") or tool_response.get("model") or "")

        record_ledger(session_dir, {
            "event": "AgentResult",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session,
            "epoch_id": epoch_id,
            "subagent_type": subagent_type,
            "agent_id": agent_id,
            "status": status,
            "resolved_model": resolved_model,
            "fingerprint": current_fp,
        })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
