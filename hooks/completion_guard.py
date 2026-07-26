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

# Explicitly anchor the import so this script works regardless of the working
# directory from which Claude Code invokes it.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from workspace_fingerprint import fingerprint, repository_root

LOGGER = logging.getLogger(__name__)

REQUIRED = {
    "Workflow-Tier",
    "Implementation-Agent",
    "Sonnet-Diff-Review",
    "Adversarial-Review",
    "Accepted-Findings",
    "Verification",
    "Verified-Workspace-SHA256",
}
IMPLEMENTATION_AGENTS = {"longcat-implementer", "longcat-repairer", "sonnet-direct"}


def block(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))


def block_with_retry_guard(reason: str, session_dir: pathlib.Path, stop_hook_active: bool) -> int:
    retry_file = session_dir / "stop_hook_retry_count.txt"
    current_retries = int(retry_file.read_text().strip()) if retry_file.exists() else 0

    if stop_hook_active:
        current_retries += 1
        retry_file.write_text(str(current_retries), encoding="utf-8")
        if current_retries >= 3:
            LOGGER.warning("Max stop_hook retry limit reached (%d attempts). Reason: %s", current_retries, reason)
            retry_file.unlink(missing_ok=True)
            sys.stderr.write(f"WARNING: Maximum stop_hook retries exceeded ({reason}). Allowing exit.\n")
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
    events = []
    if not ledger_path.exists():
        return events
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                evt = json.loads(line)
                # Inspect only events belonging to the active epoch
                if evt.get("epoch_id") == active_epoch_id:
                    events.append(evt)
    except Exception:
        pass
    return events


def completed_agents(session_dir: pathlib.Path) -> Counter[str]:
    starts: dict[str, str] = {}
    completed: Counter[str] = Counter()
    log = session_dir / "agents.jsonl"
    try:
        lines = log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return completed
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        agent_id = str(record.get("agent_id", ""))
        agent_type = str(record.get("agent_type", "unknown"))
        event = record.get("event")
        if event == "SubagentStart":
            starts[agent_id] = agent_type
        elif event == "SubagentStop" and starts.get(agent_id) == agent_type:
            completed[agent_type] += 1
            starts.pop(agent_id, None)
    return completed


def validate_ledger_sequence(parsed: dict[str, str], session_dir: pathlib.Path, active_epoch_id: str) -> str | None:
    tier = parsed["Workflow-Tier"]
    implementation_agent = parsed["Implementation-Agent"]
    if implementation_agent not in IMPLEMENTATION_AGENTS:
        return f"Unknown implementation agent: {implementation_agent}"

    completed = completed_agents(session_dir)
    if completed[implementation_agent] < 1:
        return f"No completed {implementation_agent} lifecycle is recorded for this session"

    if tier == "trivial" and implementation_agent != "sonnet-direct":
        return "Trivial tier must use the controlled sonnet-direct mutation path"
    if tier in {"normal", "cross-cutting", "high-risk"} and implementation_agent == "longcat-repairer":
        if completed["longcat-implementer"] < 1:
            return "A repairer cannot be the only implementation lifecycle; initial implementer evidence is missing"

    if tier in {"cross-cutting", "high-risk"} and completed["longcat-recon"] < 1:
        return f"{tier} work requires a completed longcat-recon lifecycle"

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
        resolved_model = str(evt.get("resolved_model") or "").lower()

        if ev_type == "Mutation":
            last_mutation_idx = idx
        elif ev_type == "TestExecutionSuccess":
            last_successful_test_idx = idx
        elif ev_type == "AgentResult":
            # Require status == 'completed'
            if status not in {"completed", "success"}:
                return f"Subagent {agent_type} outcome failed with status '{status}'; task cannot be accepted."

            # Verify model resolution for LongCat vs Sonnet
            if agent_type and agent_type.startswith("longcat-"):
                if resolved_model and not any(k in resolved_model for k in ("longcat", "anthropic-longcat-2-0")):
                    return f"Model resolution mismatch: {agent_type} resolved to '{resolved_model}' instead of LongCat."
            elif agent_type == "sonnet-direct":
                if resolved_model and "sonnet" not in resolved_model:
                    return f"Model resolution mismatch: sonnet-direct resolved to '{resolved_model}' instead of Sonnet."

        elif ev_type == "SubagentStop":
            if agent_type == "longcat-recon":
                recon_stops.append(idx)
            elif agent_type == "longcat-adversary":
                if not impl_starts:
                    adv_design_stops.append(idx)
                else:
                    adv_impl_stops.append(idx)
            elif agent_type in {"longcat-repairer", "sonnet-direct"} and impl_starts:
                repair_stops.append(idx)
        elif ev_type == "SubagentStart":
            if agent_type in {"longcat-implementer", "sonnet-direct"}:
                impl_starts.append(idx)

    # 1. Recon phase sequence check
    if tier in {"cross-cutting", "high-risk"}:
        if not recon_stops or (impl_starts and recon_stops[0] > impl_starts[0]):
            return "Sequence violation: longcat-recon must complete before implementation begins"

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

    # 3. Accepted findings repair validation (supports longcat-repairer and sonnet-direct)
    if parsed["Accepted-Findings"] == "resolved":
        if not repair_stops:
            return "Accepted findings are marked resolved, but no repair lifecycle (longcat-repairer or sonnet-direct) is recorded after adversary review"
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
    ledger_path = session_dir / "ledger.jsonl"
    record = {
        "event": "EpochClose",
        "session_id": session_dir.name,
        "epoch_id": active_epoch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    # Clear active epoch files so next task starts a fresh epoch
    (session_dir / "active_epoch_id.txt").unlink(missing_ok=True)
    (session_dir / "active_epoch_baseline.txt").unlink(missing_ok=True)


def main() -> int:
    data = json.load(sys.stdin)
    message = str(data.get("last_assistant_message", ""))
    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    stop_hook_active = bool(data.get("stop_hook_active", False))

    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
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
            current_fp = fingerprint(cwd)
            if baseline_fp != current_fp:
                has_fingerprint_change = True
        except Exception:
            pass

    has_mutated = has_mutation_events or has_fingerprint_change
    has_completion_report = "Enhanced-Completion: yes" in message

    # Fail-closed mutation-triggered gate:
    # If workspace was mutated but no completion report was provided, block completion!
    if has_mutated and not has_completion_report:
        return block_with_retry_guard("Workspace was mutated during this session epoch. A verified completion report (Enhanced-Completion: yes ...) is required before stopping.", session_dir, stop_hook_active)

    # If workspace was not mutated and no completion report is present, allow clean exit (Q&A / read-only).
    if not has_completion_report:
        return 0

    parsed = fields(message)
    missing = sorted(REQUIRED - parsed.keys())
    if missing:
        return block_with_retry_guard("Completion evidence is incomplete: missing " + ", ".join(missing), session_dir, stop_hook_active)

    tier = parsed["Workflow-Tier"]
    if tier not in {"trivial", "normal", "cross-cutting", "high-risk"}:
        return block_with_retry_guard(f"Unknown workflow tier: {tier}", session_dir, stop_hook_active)
    if tier in {"cross-cutting", "high-risk"} and parsed["Adversarial-Review"] != "passed":
        return block_with_retry_guard(f"{tier} work requires a passed adversarial review", session_dir, stop_hook_active)
    if parsed["Sonnet-Diff-Review"] != "passed" or parsed["Verification"] != "passed":
        return block_with_retry_guard("Sonnet diff review and final verification must both pass", session_dir, stop_hook_active)
    if parsed["Accepted-Findings"] not in {"none", "resolved"}:
        return block_with_retry_guard("Accepted findings remain unresolved", session_dir, stop_hook_active)

    active = session_dir / "active"
    if active.exists() and any(active.iterdir()):
        return block_with_retry_guard("A subagent is still active; wait for it before accepting completion", session_dir, stop_hook_active)

    agent_error = validate_ledger_sequence(parsed, session_dir, active_epoch_id)
    if agent_error:
        return block_with_retry_guard(agent_error, session_dir, stop_hook_active)

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

        actual = fingerprint(cwd)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return block_with_retry_guard(f"Final deterministic workspace check failed: {exc}", session_dir, stop_hook_active)

    expected = parsed["Verified-Workspace-SHA256"].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        return block_with_retry_guard("Verified workspace hash must be exactly 64 lowercase hexadecimal characters", session_dir, stop_hook_active)
    if expected != actual:
        return block_with_retry_guard(f"Verified workspace hash does not match current workspace. expected={expected} current={actual}", session_dir, stop_hook_active)

    # Verification passed cleanly: close active epoch
    record_epoch_close(session_dir, active_epoch_id)
    (session_dir / "stop_hook_retry_count.txt").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
