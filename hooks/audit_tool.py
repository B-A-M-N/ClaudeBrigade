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
from ledger_io import read_jsonl_cached
from _shared import record_ledger, resolve_session_dir, read_active_epoch_id


TEST_BASH = re.compile(
    r"\b(?:pytest|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|cargo\s+test|go\s+test|make\s+test|python3?\s+-m\s+pytest)\b",
    re.IGNORECASE,
)

# Constructs that actually suppress exit codes (masking real failures).
# Compound verification pipelines like `pytest; echo; ruff; mypy` are NOT masking.
_SHELL_MASKING = re.compile(
    r"\|\|\s*(?:true|:)\b"        # || true  || :
    r"|\|\|\s*exit\s+0\b"         # || exit 0
    r"|;\s*exit\s+0\b"            # ; exit 0
    r"|set\s+\+e\b",              # set +e
    re.IGNORECASE,
)


def _last_ledger_fingerprint(session_dir: pathlib.Path) -> str | None:
    """Return the fingerprint from the most recent Mutation entry in the ledger."""
    ledger_path = session_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return None
    last_fp: str | None = None
    for evt in read_jsonl_cached(ledger_path):
        if evt.get("event") == "Mutation" and evt.get("fingerprint"):
            last_fp = str(evt["fingerprint"])
    return last_fp


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

    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")

    session = str(data.get("session_id", "unknown"))
    session_dir = resolve_session_dir(data)
    session_dir.mkdir(parents=True, exist_ok=True)
    epoch_id = read_active_epoch_id(session_dir)

    event_name = data.get("hook_event_name", "PostToolUse")
    tool_name = str(data.get("tool_name", ""))
    tool_input = data.get("tool_input") or {}
    tool_response = data.get("tool_response") or {}
    is_failure = (event_name == "PostToolUseFailure")
    agent_id = data.get("agent_id")
    agent_type = lookup_agent_type(session_dir, agent_id)

    # Tool budgets are authoritative SQLite counters, not values supplied by
    # Claude Code.  Correlate the hook event with the active child execution
    # and increment atomically once per PostToolUse/PostToolUseFailure event.
    if agent_id and run_id:
        try:
            from enhanced_router.state import get_state
            execution = next(
                (
                    item for item in get_state().get_agent_executions(
                        str(run_id), epoch_id=epoch_id,
                    )
                    if item.get("claude_agent_id") == agent_id
                    and item.get("status") in {"started", "running"}
                ),
                None,
            )
            if execution is not None:
                get_state().increment_execution_tool_calls(
                    str(execution["execution_id"]),
                    run_id=str(run_id), epoch_id=epoch_id,
                )
        except Exception:
            # Evidence hooks must not make a completed tool call disappear;
            # the missing counter remains visible to completion diagnostics.
            pass

    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    current_fp = None
    try:
        current_fp = fingerprint(cwd, session_id=session, epoch_id=epoch_id)
    except Exception:
        pass

    # ---------------------------------------------------------------------------
    # Mutation logging: record only when the workspace fingerprint actually
    # changed after the tool ran. This catches heredoc writes, python -c, and
    # any other construct that bypassed the syntax scanner in guard_tool.
    # Runs regardless of is_failure because a failed command may still mutate disk.
    # ---------------------------------------------------------------------------
    if tool_name in {"Write", "Edit", "NotebookEdit"} and not is_failure:
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
        tool_use_id = str(data.get("tool_use_id") or "")
        pre_fp_file = session_dir / f"bash_pre_fp.{tool_use_id}.txt" if tool_use_id else session_dir / "bash_pre_fp.txt"
        pre_fp = None
        if pre_fp_file.exists():
            try:
                pre_fp = pre_fp_file.read_text(encoding="utf-8").strip() or None
            except Exception:
                pass
            finally:
                pre_fp_file.unlink(missing_ok=True)
        else:
            pre_fp = _last_ledger_fingerprint(session_dir)

        if current_fp and pre_fp and current_fp != pre_fp:
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
            if _SHELL_MASKING.search(command):
                sys.stderr.write(
                    f"WARNING: Agent {agent_id} attempted to mask test failure with "
                    f"exit-code suppression in command: {command[:200]}\n"
                )
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

        # The spawned agent's ID is in tool_response.agentId, NOT data.agent_id
        # (data.agent_id identifies the containing context, not the spawned subagent)
        spawned_agent_id = str(tool_response.get("agentId") or "")

        # Claude Code documents successful foreground agent tool response status as "completed"
        raw_status = str(tool_response.get("status") or "")
        if is_failure:
            status = "error"
        elif raw_status in {"completed", "success"}:
            status = "completed"
        else:
            status = raw_status or "unknown"

        resolved_model = str(tool_response.get("resolvedModel") or tool_response.get("model") or "")

        record_ledger(session_dir, {
            "event": "AgentResult",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session,
            "epoch_id": epoch_id,
            "subagent_type": subagent_type,
            "agent_id": agent_id,
            "spawned_agent_id": spawned_agent_id,
            "status": status,
            "resolved_model": resolved_model,
            "fingerprint": current_fp,
        })

        # Release bindings and close executions for every terminal outcome.
        if spawned_agent_id and run_id:
            try:
                from enhanced_router.state import get_state
                from enhanced_router.registry import get_registry
                state = get_state()
                terminal_status = "failed" if status == "error" else status
                try:
                    execution_epoch = next(
                        (
                            item["epoch_id"] for item in state.get_agent_executions(run_id)
                            if item.get("claude_agent_id") == spawned_agent_id
                        ),
                        None,
                    )
                    if execution_epoch:
                        state.finish_spawn_assignment(
                            str(run_id), str(execution_epoch), spawned_agent_id,
                            terminal_status,
                        )
                except (AttributeError, ValueError):
                    pass
                for execution in state.get_agent_executions(run_id):
                    if execution.get("claude_agent_id") == spawned_agent_id and execution.get("status") in {"started", "running"}:
                        state.update_agent_execution(execution["execution_id"], status=terminal_status)
                        for reservation in state.get_provider_reservations(active_only=True):
                            if reservation.get("execution_id") != execution.get("execution_id"):
                                continue
                            state.release_provider_reservation(reservation["reservation_id"], "released")
                            provider = get_registry().providers.get(str(reservation.get("provider_id")))
                            if provider:
                                state.admit_provider_agents(
                                    str(reservation["provider_id"]),
                                    provider.limits.max_active_agents,
                                )
                binding = state.get_agent_binding(run_id, spawned_agent_id)
                if binding is not None:
                    state.release_binding(run_id, spawned_agent_id)
                state.release_mutation_lease(str(run_id), spawned_agent_id)
            except ImportError:
                pass  # sidecar not available

    return 0


if __name__ == "__main__":
    from _shared import fail_open_main
    raise SystemExit(fail_open_main(main))
