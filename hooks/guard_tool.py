#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import sys
import uuid

from ledger_io import append_jsonl

from enhanced_router.base import authorized_subagents, mutating_agents

# ---------------------------------------------------------------------------
# Shell-structure-aware mutation detector
# ---------------------------------------------------------------------------
# Strategy:
#   1. Strip heredoc bodies so embedded content is not scanned as shell syntax.
#   2. Remove stderr-to-devnull and devnull redirections before checking for
#      output redirection (these are observational, not mutating).
#   3. Scan command tokens — not raw strings — for mutating constructs.
# ---------------------------------------------------------------------------

# Heredoc body: from the opening marker to the closing marker on its own line.
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1.*?\n.*?\n\2", re.DOTALL)

# Safe redirections that are observational only:
#   2>/dev/null  2>&1  >/dev/null  &>/dev/null  2>>/dev/null
_SAFE_REDIR_RE = re.compile(
    r"2>/dev/null"
    r"|2>>/dev/null"
    r"|>/dev/null"
    r"|&>/dev/null"
    r"|2>&1"
    r"|1>&2",
    re.IGNORECASE,
)

# Mutating shell constructs (command-token level, after safe redirections removed):
#   - File-mutating builtins/utilities
#   - In-place editors
#   - Mutating git sub-commands
#   - Package-manager mutation sub-commands
#   - Formatters that rewrite files
#   - Inline Python/Node/Perl scripts that open/write/unlink files
#   - Any remaining > or >> that is NOT already handled (i.e. real redirections)
_MUTATING_BASH = re.compile(
    r"(?:^|[;&|]\s*)(?:rm|mv|cp|touch|mkdir|rmdir|truncate|install|patch|dd|ln)\b"
    r"|(?:^|\s)(?:sed\s+-i|perl\s+-pi|tee\b)"
    r"|(?:^|\s)find\s+.*(?:-delete|-exec\b|-execdir\b|-ok\b|-okdir\b|-fprint\b|-fprintf\b|-fls\b)"
    r"|(?:^|\s)git\s+(?:add|apply|am|checkout|clean|commit|merge|rebase|reset|restore|switch)\b"
    r"|(?:^|\s)git\s+.*--(?:output|output-directory)\b"
    r"|(?:^|\s)(?:npm|pnpm|yarn|pip|pip3|uv|cargo)\s+(?:install|add|remove|update|fmt)\b"
    r"|(?:^|\s)(?:go\s+fmt|gofmt|rustfmt|prettier|black|ruff\s+(?:format|check\s+.*--fix|check\s+.*-fix|--fix|-fix))\b"
    r"|(?:^|\s)pytest\s+.*--(?:junitxml|resultlog|html|cov-report)\b"
    r"|(?:python|python3|node|perl|ruby)\s+-[ce]\s+.*(?:open|write|unlink|remove)"
    r"|(?<!<)>(?![>&])|>>",
    re.IGNORECASE,
)


def _is_mutating(command: str) -> bool:
    """Return True only when the command actually mutates the workspace."""
    # 1. Strip heredoc bodies so embedded content is invisible to the scanner.
    stripped = _HEREDOC_RE.sub("", command)
    # 2. Remove safe/observational redirections before checking for real ones.
    stripped = _SAFE_REDIR_RE.sub("", stripped)
    # 3. Scan the cleaned shell structure.
    return bool(_MUTATING_BASH.search(stripped))


# Public alias kept for backward-compatibility with tests and external tooling.
MUTATING_BASH = _MUTATING_BASH

_READ_ONLY_COMMANDS = frozenset({
    "pwd", "ls", "find", "rg", "grep", "git", "cat", "head", "tail",
    "sed", "awk", "sort", "uniq", "wc", "file", "stat", "which", "command",
})
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "ls-files", "rev-parse", "branch",
    "describe", "grep", "rev-list", "worktree",
})


def _is_read_only_shell(command: str) -> bool:
    """Allow only simple observational commands for read-only agents.

    Syntax scanning remains useful for diagnostics, but authorization is based
    on a small command allowlist and rejects shell composition/redirection.
    This prevents an existing script or interpreter from becoming an implicit
    write capability.
    """
    if not command.strip() or _is_mutating(command):
        return False
    if any(token in command for token in (">", "<", "`", "$(", ";", "&&", "||")):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    # Pipelines are permitted only when every command is independently
    # allowlisted; no shell control operator is allowed above.
    commands = command.split("|")
    for raw in commands:
        try:
            part = shlex.split(raw.strip(), posix=True)
        except ValueError:
            return False
        if not part or part[0] not in _READ_ONLY_COMMANDS:
            return False
        if part[0] == "git":
            if len(part) < 2 or part[1] not in _READ_ONLY_GIT_SUBCOMMANDS:
                return False
            if part[1] == "worktree" and len(part) > 2 and part[2] != "list":
                return False
    return True


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
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session_dir = cache / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    epoch_file = session_dir / "active_epoch_id.txt"
    epoch_id = epoch_file.read_text(encoding="utf-8").strip() if epoch_file.exists() else "ep_unknown"

    record = {
        "event": event_type,
        "session_id": session_id,
        "epoch_id": epoch_id,
        "details": details,
    }
    append_jsonl(session_dir / "ledger.jsonl", record)


def lookup_agent_type(data: dict) -> str:
    agent_id = data.get("agent_id")
    if not agent_id:
        return "unknown"
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    marker = cache / "sessions" / str(data.get("session_id", "unknown")) / "active" / f"{agent_id}.json"
    try:
        return str(json.loads(marker.read_text(encoding="utf-8")).get("agent_type", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


def _role_for_agent(agent_type: str, subagent_type: str) -> str:
    value = subagent_type or agent_type
    for role in ("recon", "implementer", "adversary", "repairer"):
        if role in value:
            return role
    return "implementer" if "code" in value or "test" in value else "recon"


def _ensure_mutation_lease(data: dict, agent_type: str, cwd: pathlib.Path) -> bool:
    """Require the current execution to own the workspace writer lease."""
    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    if not run_id:
        deny(
            "Mutation blocked: no authenticated Brigade run is attached to this "
            "execution. Start the session through ClaudeBrigade."
        )
        return False
    agent_id = str(data.get("agent_id") or f"controller:{data.get('session_id', 'unknown')}")
    try:
        from enhanced_router.state import get_state

        state = get_state()
        epoch = state.get_active_epoch(str(run_id))
        if not epoch:
            deny("Mutation blocked: no active Brigade task epoch.")
            return False
        role = "controller" if agent_type in {"unknown", "controller-direct"} else _role_for_agent(agent_type, "")
        active_phases = [
            phase for phase in state.get_active_phases(str(run_id), epoch["epoch_id"])
            if phase.get("mutating")
        ]
        if not active_phases:
            deny("Mutation blocked: no active mutating workflow phase.")
            return False
        permitted = False
        for phase in active_phases:
            allowed_roles = json.loads(phase.get("allowed_roles_json") or "[]")
            actor = str(phase.get("actor") or "")
            if role in allowed_roles or (role == "controller" and actor == "controller"):
                permitted = True
                break
        if not permitted:
            deny(f"Mutation blocked: agent role '{role}' is not allowed in the active phase.")
            return False
        workspace_id = str(cwd.resolve())
        execution = next(
            (
                item for item in state.get_agent_executions(str(run_id), epoch_id=epoch["epoch_id"])
                if item.get("claude_agent_id") == agent_id
                and item.get("status") in {"started", "running"}
            ),
            None,
        )
        shadows = state.get_workspaces(
            run_id=str(run_id), epoch_id=epoch["epoch_id"], kind="shadow", status="active",
        )
        owned_shadow = next(
            (
                item for item in shadows
                if pathlib.Path(str(item.get("path"))).resolve() == cwd.resolve()
                and item.get("owner_execution_id") == (execution or {}).get("execution_id")
            ),
            None,
        )
        if agent_type in mutating_agents() and owned_shadow is None:
            deny(
                "Mutation blocked: mutating executions must own an active shadow worktree; "
                "canonical or unregistered workspaces are not writable."
            )
            return False
        if not state.acquire_mutation_lease(
            str(run_id), epoch["epoch_id"], agent_id, role, workspace_id=workspace_id
        ):
            deny("Mutation blocked: another execution owns the workspace mutation lease.")
            return False
        return True
    except Exception as exc:
        deny(f"Mutation blocked: unable to validate workspace lease ({exc}).")
        return False


def _reserve_agent_slot(data: dict, subagent_type: str, session_id: str) -> bool:
    """Consume a controller-planned action before Claude Code spawns a child.

    SQLite cannot replay a denied native ``Agent`` tool call.  Capacity is
    therefore reserved by the controller's explicit MCP claim, and this hook
    only consumes that claim.  It never creates a queued reservation of its
    own.
    """
    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    if not run_id:
        return True
    try:
        from enhanced_router.state import get_state
        state = get_state()
        epoch = state.get_active_epoch(str(run_id))
        if not epoch:
            deny("Agent spawn blocked: no active Brigade epoch.")
            return False
        tool_input = data.get("tool_input") or {}
        action_id = tool_input.get("brigade_action_id") or tool_input.get("action_id")
        call_id = str(data.get("tool_use_id") or data.get("request_id") or uuid.uuid4().hex)
        try:
            claim = state.consume_runnable_action_for_spawn(
                str(run_id), str(epoch["epoch_id"]), subagent_type,
                action_id=str(action_id) if action_id else None,
                spawn_call_id=call_id,
            )
        except TypeError as exc:
            # Keep older test doubles and embedded callers source-compatible;
            # the real RouteState records the spawn correlation key above.
            if "spawn_call_id" not in str(exc):
                raise
            claim = state.consume_runnable_action_for_spawn(
                str(run_id), str(epoch["epoch_id"]), subagent_type,
                action_id=str(action_id) if action_id else None,
            )
        if claim is None:
            deny(
                "Agent spawn blocked: call get_runnable_actions, claim the returned "
                "action, and invoke the matching native agent."
            )
            return False
        marker_dir = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade" / "sessions" / session_id / "active"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker = marker_dir / f"spawn_{call_id}.json"
        tmp = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({
            "call_id": call_id,
            "action_id": claim["action_id"],
            "native_agent_name": subagent_type,
            "reservation_id": claim.get("reservation_id"),
            "intent_id": claim.get("intent_id"),
            "role": claim["role"],
            "provider_id": claim.get("provider_id"),
            "model_id": claim["model_id"],
            "state": claim["status"],
        }, separators=(",", ":")), encoding="utf-8")
        tmp.replace(marker)
        return True
    except Exception as exc:
        deny(f"Agent admission failed: {exc}")
        return False
def main() -> int:
    data = json.load(sys.stdin)
    tool = str(data.get("tool_name", ""))
    agent_type = lookup_agent_type(data)
    session_id = str(data.get("session_id", "unknown"))

    if tool == "Agent":
        subagent_type = str((data.get("tool_input") or {}).get("subagent_type") or (data.get("tool_input") or {}).get("agent_type") or "")
        if subagent_type and subagent_type not in authorized_subagents():
            deny(f"Subagent role '{subagent_type}' is not in the authorized enhanced subagent allowlist.")
            return 0
        if not _reserve_agent_slot(data, subagent_type, session_id):
            return 0

    if tool in {"Write", "Edit", "NotebookEdit"}:
        if agent_type not in mutating_agents():
            deny(f"{agent_type} is read-only. Delegate source mutation to an authorized implementation agent.")
            return 0
        cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
        if not _ensure_mutation_lease(data, agent_type, cwd):
            return 0
        record_ledger_event(session_id, "MutationIntent", {"tool": tool, "agent_type": agent_type})

    if tool == "Bash":
        command = str((data.get("tool_input") or {}).get("command", ""))
        cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
        # Snapshot pre-execution fingerprint so audit_tool can detect real mutations
        # by fingerprint diff rather than syntax scanning. Written atomically so
        # a crashed hook doesn't leave a stale file from a previous invocation.
        try:
            sys.path.insert(0, str(pathlib.Path(__file__).parent))
            from workspace_fingerprint import fingerprint as _fingerprint
            pre_fp = _fingerprint(cwd, session_id=session_id)
            cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
            session_dir = cache / "sessions" / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            tool_use_id = str(data.get("tool_use_id") or uuid.uuid4().hex)
            pre_fp_file = session_dir / f"bash_pre_fp.{tool_use_id}.txt"
            tmp_pre_fp_file = session_dir / f"bash_pre_fp.{tool_use_id}.tmp.{os.getpid()}"
            tmp_pre_fp_file.write_text(pre_fp, encoding="utf-8")
            tmp_pre_fp_file.replace(pre_fp_file)
        except Exception:
            pass

        if agent_type not in mutating_agents():
            if not _is_read_only_shell(command):
                deny(
                    f"Bash blocked for read-only role {agent_type}: command is not in "
                    "the Brigade observational allowlist."
                )
                return 0
        elif _is_mutating(command):
            if not _ensure_mutation_lease(data, agent_type, cwd):
                return 0
            record_ledger_event(session_id, "MutationIntent", {"tool": "Bash", "command": command, "agent_type": agent_type})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
