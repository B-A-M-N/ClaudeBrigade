import json
import pathlib
import io
import pytest
from completion_guard import main as completion_guard_main, validate_ledger_sequence
from guard_tool import main as guard_tool_main, MUTATING_BASH


def test_guard_tool_mutating_bash_regex():
    assert MUTATING_BASH.search("rm -rf foo") is not None
    assert MUTATING_BASH.search("git commit -m 'test'") is not None
    assert MUTATING_BASH.search("echo hello > file.txt") is not None
    assert MUTATING_BASH.search("python3 -c 'open(\"file\", \"w\").write(\"a\")'") is not None
    assert MUTATING_BASH.search("dd if=/dev/zero of=file") is not None
    assert MUTATING_BASH.search("ln -s a b") is not None

    # Read-only commands should not match MUTATING_BASH
    assert MUTATING_BASH.search("git status") is None
    assert MUTATING_BASH.search("ls -la") is None
    assert MUTATING_BASH.search("grep -rn 'foo' .") is None
    assert MUTATING_BASH.search("pytest") is None


def test_subagent_allowlist_denies_unauthorized_role(monkeypatch):
    payload = {
        "tool_name": "Agent",
        "tool_input": {"subagent_type": "malicious-unauthorized-agent"},
        "session_id": "test-allowlist"
    }
    stdin_str = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io_string_stream(stdin_str))

    denied_reasons = []
    monkeypatch.setattr("guard_tool.deny", lambda reason: denied_reasons.append(reason))

    guard_tool_main()
    assert len(denied_reasons) == 1
    assert "not in the authorized enhanced subagent allowlist" in denied_reasons[0]


def test_completion_guard_mutation_triggered_gate(tmp_path, monkeypatch):
    session_id = "test-session-mutation"
    cache_dir = tmp_path / ".cache" / "claude-enhanced"
    session_dir = cache_dir / "sessions" / session_id
    session_dir.mkdir(parents=True)

    epoch_file = session_dir / "active_epoch_id.txt"
    epoch_file.write_text("ep_123", encoding="utf-8")

    ledger_path = session_dir / "ledger.jsonl"
    ledger_path.write_text(json.dumps({"event": "Mutation", "epoch_id": "ep_123", "tool": "Write", "agent_type": "longcat-implementer"}) + "\n")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))

    payload = {
        "session_id": session_id,
        "cwd": str(tmp_path),
        "last_assistant_message": "I edited the file for you."
    }

    stdin_str = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io_string_stream(stdin_str))

    blocked_reasons = []
    monkeypatch.setattr("completion_guard.block", lambda reason: blocked_reasons.append(reason))

    completion_guard_main()
    assert len(blocked_reasons) == 1
    assert "Workspace was mutated during this session epoch" in blocked_reasons[0]


def test_cross_cutting_requires_impl_adversary(tmp_path):
    session_dir = tmp_path / "session_cross_cutting"
    session_dir.mkdir()

    active_epoch_id = "ep_cross"

    # Sequence: Recon -> Impl (no adversary after impl)
    ledger_events = [
        {"event": "SubagentStart", "epoch_id": active_epoch_id, "agent_type": "longcat-recon"},
        {"event": "SubagentStop", "epoch_id": active_epoch_id, "agent_type": "longcat-recon"},
        {"event": "SubagentStart", "epoch_id": active_epoch_id, "agent_type": "longcat-implementer"},
        {"event": "Mutation", "epoch_id": active_epoch_id, "tool": "Write", "agent_type": "longcat-implementer"},
        {"event": "SubagentStop", "epoch_id": active_epoch_id, "agent_type": "longcat-implementer"},
        {"event": "TestExecutionSuccess", "epoch_id": active_epoch_id, "command": "pytest", "exit_code": 0},
    ]
    with (session_dir / "ledger.jsonl").open("w") as f:
        for e in ledger_events:
            f.write(json.dumps(e) + "\n")

    records = [
        {"event": "SubagentStart", "agent_id": "1", "agent_type": "longcat-recon"},
        {"event": "SubagentStop", "agent_id": "1", "agent_type": "longcat-recon"},
        {"event": "SubagentStart", "agent_id": "2", "agent_type": "longcat-implementer"},
        {"event": "SubagentStop", "agent_id": "2", "agent_type": "longcat-implementer"},
    ]
    (session_dir / "agents.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    parsed = {
        "Workflow-Tier": "cross-cutting",
        "Implementation-Agent": "longcat-implementer",
        "Sonnet-Diff-Review": "passed",
        "Adversarial-Review": "passed",
        "Accepted-Findings": "none",
        "Verification": "passed",
        "Verified-Workspace-SHA256": "0" * 64
    }

    err = validate_ledger_sequence(parsed, session_dir, active_epoch_id)
    assert err is not None
    assert "Cross-cutting work requires an implementation adversary review" in err


def test_model_resolution_mismatch_is_rejected(tmp_path):
    session_dir = tmp_path / "session_model_mismatch"
    session_dir.mkdir()
    active_epoch_id = "ep_model"

    ledger_events = [
        {"event": "AgentResult", "epoch_id": active_epoch_id, "subagent_type": "longcat-implementer", "status": "completed", "resolved_model": "gpt-4-oops"},
        {"event": "SubagentStart", "epoch_id": active_epoch_id, "agent_type": "longcat-implementer"},
        {"event": "SubagentStop", "epoch_id": active_epoch_id, "agent_type": "longcat-implementer"},
    ]
    with (session_dir / "ledger.jsonl").open("w") as f:
        for e in ledger_events:
            f.write(json.dumps(e) + "\n")

    records = [
        {"event": "SubagentStart", "agent_id": "1", "agent_type": "longcat-implementer"},
        {"event": "SubagentStop", "agent_id": "1", "agent_type": "longcat-implementer"},
    ]
    (session_dir / "agents.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    parsed = {
        "Workflow-Tier": "normal",
        "Implementation-Agent": "longcat-implementer",
        "Sonnet-Diff-Review": "passed",
        "Adversarial-Review": "not-required",
        "Accepted-Findings": "none",
        "Verification": "passed",
        "Verified-Workspace-SHA256": "0" * 64
    }

    err = validate_ledger_sequence(parsed, session_dir, active_epoch_id)
    assert err is not None
    assert "Model resolution mismatch" in err


def test_bounded_stop_hook_retries(tmp_path, monkeypatch):
    session_id = "test-stop-retry"
    cache_dir = tmp_path / ".cache" / "claude-enhanced"
    session_dir = cache_dir / "sessions" / session_id
    session_dir.mkdir(parents=True)

    (session_dir / "active_epoch_id.txt").write_text("ep_retry", encoding="utf-8")

    # Force a validation error (missing required fields)
    payload = {
        "session_id": session_id,
        "cwd": str(tmp_path),
        "last_assistant_message": "Enhanced-Completion: yes",
        "stop_hook_active": True
    }

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))

    blocked_reasons = []
    monkeypatch.setattr("completion_guard.block", lambda reason: blocked_reasons.append(reason))

    # Attempt 1 (stop_hook_active=True)
    stdin_str = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io_string_stream(stdin_str))
    completion_guard_main()

    # Attempt 2
    stdin_str = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io_string_stream(stdin_str))
    completion_guard_main()

    # Attempt 3 - Max retries reached, should allow exit (return 0) without blocking
    stdin_str = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io_string_stream(stdin_str))
    res = completion_guard_main()

    assert res == 0
    assert len(blocked_reasons) == 2


class io_string_stream:
    def __init__(self, text):
        self.text = text
    def read(self):
        return self.text
