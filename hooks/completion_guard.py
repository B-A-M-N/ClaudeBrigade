#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

# Explicitly anchor the import so this script works regardless of the working
# directory from which Claude Code invokes it.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workspace_fingerprint import fingerprint, repository_root, clear_fingerprint_cache
from ledger_io import append_jsonl, read_jsonl
from enhanced_router.base import implementation_agents

LOGGER = logging.getLogger(__name__)

REQUIRED = {
    "Workflow-Tier",
    "Implementation-Agent",
    "Controller-Diff-Review",
    "Adversarial-Review",
    "Accepted-Findings",
    "Verification",
    "Route-Snapshot-SHA256",
    "Verified-Workspace-SHA256",
}

def block(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))


def block_with_retry_guard(reason: str, session_dir: pathlib.Path, stop_hook_active: bool, message: str = "") -> int:
    retry_file = session_dir / "stop_hook_retry_count.txt"
    try:
        current_retries = int(retry_file.read_text().strip()) if retry_file.exists() else 0
    except (ValueError, OSError):
        current_retries = 0
        retry_file.unlink(missing_ok=True)

    if stop_hook_active:
        current_retries += 1
        retry_file.write_text(str(current_retries), encoding="utf-8")
        if current_retries >= 3:
            # Only allow exit when the assistant replaces its completion claim
            # with an explicit failure acknowledgment ("Enhanced-Completion: failed"),
            # NOT a false success ("Enhanced-Completion: yes").
            if "Enhanced-Completion: failed" in message:
                LOGGER.warning(
                    "Stop hook retries exhausted and failure acknowledged. Allowing exit."
                )
                retry_file.unlink(missing_ok=True)
                return 0
            LOGGER.warning(
                "Max stop_hook retries exceeded (%d) but message still claims success. "
                "Requiring explicit failure acknowledgment.",
                current_retries,
            )
            sys.stderr.write(
                f"ERROR: Maximum stop_hook retries exceeded ({reason}).\n"
                f"The assistant must acknowledge failure with 'Enhanced-Completion: failed' "
                f"before the session can exit.\n"
            )
            block(reason)
            return 0

    block(reason)
    return 0


def fields(message: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in message.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z-]+):\s*(.+?)\s*$", line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def read_epoch_ledger(session_dir: pathlib.Path, active_epoch_id: str) -> list[dict]:
    ledger_path = session_dir / "ledger.jsonl"
    return [
        event for event in read_jsonl(ledger_path)
        if event.get("epoch_id") == active_epoch_id
    ]


def completed_agents(session_dir: pathlib.Path, active_epoch_id: str) -> Counter[str]:
    starts: dict[str, str] = {}
    completed: Counter[str] = Counter()
    log = session_dir / "agents.jsonl"
    for record in read_jsonl(log):
        agent_id = str(record.get("agent_id", ""))
        agent_type = str(record.get("agent_type", "unknown"))
        event = record.get("event")
        epoch_id = str(record.get("epoch_id", ""))
        # Only count agents that belong to the active epoch
        if epoch_id != active_epoch_id:
            continue
        if event == "SubagentStart":
            starts[agent_id] = agent_type
        elif event == "SubagentStop" and starts.get(agent_id) == agent_type:
            completed[agent_type] += 1
            starts.pop(agent_id, None)
    return completed


def validate_ledger_sequence(parsed: dict[str, str], session_dir: pathlib.Path, active_epoch_id: str) -> str | None:
    tier = parsed["Workflow-Tier"]
    implementation_agent = parsed["Implementation-Agent"]
    if implementation_agent not in implementation_agents():
        return f"Unknown implementation agent: {implementation_agent}"

    completed = completed_agents(session_dir, active_epoch_id)
    if completed[implementation_agent] < 1:
        return f"No completed {implementation_agent} lifecycle is recorded for this session"

    if tier == "trivial" and implementation_agent != "controller-direct":
        return "Trivial tier must use the controlled controller-direct mutation path"
    if tier in {"normal", "cross-cutting", "high-risk"} and implementation_agent == "brigade-repairer":
        if completed["brigade-implementer"] < 1:
            return "A repairer cannot be the only implementation lifecycle; initial implementer evidence is missing"

    if tier in {"cross-cutting", "high-risk"} and completed["brigade-recon"] < 1:
        return f"{tier} work requires a completed brigade-recon lifecycle"

    # Read events from the active epoch only
    ledger_events = read_epoch_ledger(session_dir, active_epoch_id)

    recon_stops = []
    impl_starts = []
    adv_design_stops = []
    adv_impl_stops = []
    repair_stops = []
    last_mutation_idx = -1
    last_successful_test_idx = -1

    for idx, evt in enumerate(ledger_events):
        ev_type = evt.get("event")
        agent_type = (
            evt.get("subagent_type") or
            evt.get("agent_type") or
            (evt.get("details", {}) if isinstance(evt.get("details"), dict) else {}).get("agent_type")
        )
        status = evt.get("status")

        if ev_type == "Mutation":
            last_mutation_idx = idx
        elif ev_type == "TestExecutionSuccess":
            last_successful_test_idx = idx
        elif ev_type == "AgentResult":
            # Require status == 'completed'
            if status not in {"completed", "success"}:
                return f"Subagent {agent_type} outcome failed with status '{status}'; task cannot be accepted."

        # Agent type already guarantees correct model routing; skip resolved_model checks.

        elif ev_type == "SubagentStop":
            if agent_type == "brigade-recon":
                recon_stops.append(idx)
            elif agent_type == "brigade-adversary":
                if not impl_starts:
                    adv_design_stops.append(idx)
                else:
                    adv_impl_stops.append(idx)
            elif agent_type in {"brigade-repairer", "controller-direct"} and impl_starts:
                repair_stops.append(idx)
        elif ev_type == "SubagentStart":
            if agent_type in {"brigade-implementer", "controller-direct"}:
                impl_starts.append(idx)

    # 1. Recon phase sequence check
    if tier in {"cross-cutting", "high-risk"}:
        if not recon_stops or (impl_starts and recon_stops[0] > impl_starts[0]):
            return "Sequence violation: brigade-recon must complete before implementation begins"

    # 2. Adversary phase sequence check
    if tier == "high-risk":
        if not adv_design_stops or (impl_starts and adv_design_stops[0] > impl_starts[0]):
            return "Sequence violation: design adversary must complete before implementation begins"
        if not adv_impl_stops or (impl_starts and adv_impl_stops[-1] < impl_starts[0]):
            return "Sequence violation: implementation adversary must review after implementation finishes"
    elif tier == "cross-cutting":
        if not adv_impl_stops:
            return "Cross-cutting work requires an implementation adversary review after implementation finishes"
        if impl_starts and adv_impl_stops[-1] < impl_starts[0]:
            return "Sequence violation: adversarial review must review after implementation finishes"

    # 3. Accepted findings repair validation (supports brigade-repairer and controller-direct)
    if parsed["Accepted-Findings"] == "resolved":
        if not repair_stops:
            return "Accepted findings are marked resolved, but no repair lifecycle (brigade-repairer or controller-direct) is recorded after adversary review"
        if adv_impl_stops and repair_stops[-1] < adv_impl_stops[0]:
            return "Sequence violation: repair must execute after adversarial findings were reported"

    # 4. Successful post-mutation test verification requirement
    if last_mutation_idx != -1:
        if last_successful_test_idx == -1:
            return "Workspace was mutated, but no successful test execution (exit code 0) is recorded in the active epoch."
        if last_successful_test_idx < last_mutation_idx:
            return "Workspace was mutated after the last successful test execution. Final tests must be executed against the final workspace state."

    return None


def record_epoch_close(session_dir: pathlib.Path, active_epoch_id: str) -> None:
    record = {
        "event": "EpochClose",
        "session_id": session_dir.name,
        "epoch_id": active_epoch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(session_dir / "ledger.jsonl", record)

    # Clear active epoch files so next task starts a fresh epoch
    (session_dir / "active_epoch_id.txt").unlink(missing_ok=True)
    (session_dir / "active_epoch_baseline.txt").unlink(missing_ok=True)


def _extract_run_id(parsed: dict[str, str], data: dict[str, Any]) -> str | None:
    return os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")


def validate_completion_via_state(
    parsed: dict[str, str],
    run_id: str,
    epoch_id: str,
    session_dir: pathlib.Path,
) -> tuple[bool, str | None]:
    """Delegate completion validation to the authoritative RouteState.validate_completion().

    Returns (valid, reason).
    """
    try:
        from enhanced_router.state import get_state
    except ImportError:
        return False, "Authoritative workflow state is unavailable"

    state = get_state()
    result = state.validate_completion(
        run_id=run_id,
        epoch_id=epoch_id,
        parsed=parsed,
        session_dir=session_dir,
    )
    return result["valid"], result.get("reason")


def main() -> int:
    data = json.load(sys.stdin)
    message = str(data.get("last_assistant_message", ""))
    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    stop_hook_active = bool(data.get("stop_hook_active", False))

    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session_id = str(data.get("session_id", "unknown"))
    session_dir = cache / "sessions" / session_id

    epoch_file = session_dir / "active_epoch_id.txt"
    active_epoch_id = epoch_file.read_text(encoding="utf-8").strip() if epoch_file.exists() else "ep_unknown"

    baseline_file = session_dir / "active_epoch_baseline.txt"
    baseline_fp = baseline_file.read_text(encoding="utf-8").strip() if baseline_file.exists() else None

    # Check workspace mutation status in active epoch
    ledger_events = read_epoch_ledger(session_dir, active_epoch_id)
    has_mutation_events = any(evt.get("event") == "Mutation" for evt in ledger_events)

    has_fingerprint_change = False
    if baseline_fp:
        try:
            current_fp = fingerprint(
                cwd,
                session_id=session_id,
                epoch_id=active_epoch_id,
            )
            if baseline_fp != current_fp:
                has_fingerprint_change = True
        except Exception:
            pass

    has_mutated = has_mutation_events or has_fingerprint_change
    has_completion_report = "Enhanced-Completion: yes" in message

    # Fail-closed mutation-triggered gate:
    # If workspace was mutated but no completion report was provided, block completion!
    if has_mutated and not has_completion_report:
        return block_with_retry_guard("Workspace was mutated during this session epoch. A verified completion report (Enhanced-Completion: yes ...) is required before stopping.", session_dir, stop_hook_active, message)

    # If workspace was not mutated and no completion report is present, allow clean exit (Q&A / read-only).
    if not has_completion_report:
        return 0

    parsed = fields(message)
    missing = sorted(REQUIRED - parsed.keys())
    if missing:
        return block_with_retry_guard("Completion evidence is incomplete: missing " + ", ".join(missing), session_dir, stop_hook_active, message)

    tier = parsed["Workflow-Tier"]
    if tier not in {"trivial", "normal", "cross-cutting", "high-risk"}:
        return block_with_retry_guard(f"Unknown workflow tier: {tier}", session_dir, stop_hook_active, message)

    # Validate evidence against authoritative state via validate_completion MCP op
    run_id = _extract_run_id(parsed, data)
    if run_id:
        valid, reason = validate_completion_via_state(parsed, run_id, active_epoch_id, session_dir)
        if not valid:
            return block_with_retry_guard(reason or "Completion validation failed", session_dir, stop_hook_active, message)

    if tier in {"cross-cutting", "high-risk"} and parsed["Adversarial-Review"] != "passed":
        return block_with_retry_guard(f"{tier} work requires a passed adversarial review", session_dir, stop_hook_active, message)
    if parsed["Controller-Diff-Review"] != "passed" or parsed["Verification"] != "passed":
        return block_with_retry_guard("Controller diff review and final verification must both pass", session_dir, stop_hook_active, message)
    if parsed["Accepted-Findings"] not in {"none", "resolved"}:
        return block_with_retry_guard("Accepted findings remain unresolved", session_dir, stop_hook_active, message)

    active = session_dir / "active"
    if active.exists() and any(f for f in active.iterdir() if f.suffix == ".json"):
        return block_with_retry_guard("A subagent is still active; wait for it before accepting completion", session_dir, stop_hook_active, message)

    agent_error = validate_ledger_sequence(parsed, session_dir, active_epoch_id)
    if agent_error:
        return block_with_retry_guard(agent_error, session_dir, stop_hook_active, message)

    try:
        root = repository_root(cwd)

        # Guard against repos with no commits yet where HEAD does not exist.
        head_exists = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0

        if head_exists:
            subprocess.run(
                ["git", "-C", str(root), "diff", "--check", "HEAD"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            LOGGER.info("Skipping git diff --check: repository has no commits yet")

        actual = fingerprint(
            cwd,
            session_id=session_id,
            epoch_id=active_epoch_id,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return block_with_retry_guard(f"Final deterministic workspace check failed: {exc}", session_dir, stop_hook_active, message)

    expected = parsed["Verified-Workspace-SHA256"].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return block_with_retry_guard("Verified workspace hash must be exactly 64 lowercase hexadecimal characters", session_dir, stop_hook_active, message)
    if expected != actual:
        return block_with_retry_guard(f"Verified workspace hash does not match current workspace. expected={expected} current={actual}", session_dir, stop_hook_active, message)

    if run_id:
        try:
            from enhanced_router.state import get_state
            state = get_state()
            attestation = state.prepare_completion_token(
                run_id, active_epoch_id, actual,
            )
            consumed = state.consume_completion_token(
                run_id,
                active_epoch_id,
                attestation["token"],
                actual,
                parsed["Route-Snapshot-SHA256"].lower(),
            )
            if not consumed.get("valid"):
                return block_with_retry_guard(
                    str(consumed.get("reason") or "Completion attestation failed"),
                    session_dir,
                    stop_hook_active,
                    message,
                )
        except Exception as exc:
            return block_with_retry_guard(
                f"State-generated completion attestation failed: {exc}",
                session_dir,
                stop_hook_active,
                message,
            )

    # Close SQLite epoch (authoritative state), then clean up file-based markers.
    # This order ensures that if a crash occurs between the two, the SQLite state
    # correctly reflects the closed epoch (file markers are only used for bootstrapping).
    closed_sqlite = False
    if run_id:
        try:
            from enhanced_router.state import get_state
            state = get_state()
            run = state.get_run(run_id)
            if run and not run.get("closed_at"):
                active_sqlite = state.get_active_epoch(run_id)
                if active_sqlite:
                    state.close_epoch(run_id, active_sqlite["epoch_id"])
                    closed_sqlite = True
        except ImportError:
            pass  # package not available, skip SQLite state management

    # Clean up file-based epoch markers
    if closed_sqlite or not run_id:
        # Only close file-based if SQLite close succeeded, or if SQLite wasn't used
        record_epoch_close(session_dir, active_epoch_id)
        # Clear fingerprint cache for next epoch
        clear_fingerprint_cache()

    (session_dir / "stop_hook_retry_count.txt").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
