import json
from completion_guard import main as completion_guard_main, validate_ledger_sequence
from guard_tool import _is_read_only_shell, main as guard_tool_main, MUTATING_BASH


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


def test_read_only_shell_boundary_rejects_hidden_writers():
    assert _is_read_only_shell("git status") is True
    assert _is_read_only_shell("rg -n TODO router") is True
    assert _is_read_only_shell("./existing-script.sh") is False
    assert _is_read_only_shell("python3 -c 'open(\\\"x\\\", \\\"w\\\")'") is False
    assert _is_read_only_shell("find . -exec touch marker 'x'") is False


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


def test_agent_spawn_requires_a_cooperative_action_claim(tmp_path, monkeypatch):
    class FakeState:
        def get_active_epoch(self, run_id):
            return {"epoch_id": "ep-1"}

        def consume_runnable_action_for_spawn(self, run_id, epoch_id, native_agent_name, *, action_id=None):
            return None

    monkeypatch.setenv("CLAUDE_BRIGADE_RUN_ID", "run-1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    monkeypatch.setattr("enhanced_router.state.get_state", lambda: FakeState())
    monkeypatch.setattr("guard_tool.deny", lambda reason: denied_reasons.append(reason))
    denied_reasons = []
    monkeypatch.setattr(
        "sys.stdin",
        io_string_stream(json.dumps({
            "tool_name": "Agent",
            "tool_input": {"subagent_type": "brigade-recon"},
            "session_id": "claimed-agent-test",
        })),
    )

    guard_tool_main()

    assert len(denied_reasons) == 1
    assert "get_runnable_actions" in denied_reasons[0]


def test_completion_guard_mutation_triggered_gate(tmp_path, monkeypatch):
    session_id = "test-session-mutation"
    cache_dir = tmp_path / ".cache" / "claude-brigade"
    session_dir = cache_dir / "sessions" / session_id
    session_dir.mkdir(parents=True)

    epoch_file = session_dir / "active_epoch_id.txt"
    epoch_file.write_text("ep_123", encoding="utf-8")

    ledger_path = session_dir / "ledger.jsonl"
    ledger_path.write_text(json.dumps({"event": "Mutation", "epoch_id": "ep_123", "tool": "Write", "agent_type": "brigade-implementer"}) + "\n")

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
        {"event": "SubagentStart", "epoch_id": active_epoch_id, "agent_type": "brigade-recon"},
        {"event": "SubagentStop", "epoch_id": active_epoch_id, "agent_type": "brigade-recon"},
        {"event": "SubagentStart", "epoch_id": active_epoch_id, "agent_type": "brigade-implementer"},
        {"event": "Mutation", "epoch_id": active_epoch_id, "tool": "Write", "agent_type": "brigade-implementer"},
        {"event": "SubagentStop", "epoch_id": active_epoch_id, "agent_type": "brigade-implementer"},
        {"event": "TestExecutionSuccess", "epoch_id": active_epoch_id, "command": "pytest", "exit_code": 0},
    ]
    with (session_dir / "ledger.jsonl").open("w") as f:
        for e in ledger_events:
            f.write(json.dumps(e) + "\n")

    records = [
        {"event": "SubagentStart", "agent_id": "1", "agent_type": "brigade-recon", "epoch_id": active_epoch_id},
        {"event": "SubagentStop", "agent_id": "1", "agent_type": "brigade-recon", "epoch_id": active_epoch_id},
        {"event": "SubagentStart", "agent_id": "2", "agent_type": "brigade-implementer", "epoch_id": active_epoch_id},
        {"event": "SubagentStop", "agent_id": "2", "agent_type": "brigade-implementer", "epoch_id": active_epoch_id},
    ]
    (session_dir / "agents.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    parsed = {
        "Workflow-Tier": "cross-cutting",
        "Implementation-Agent": "brigade-implementer",
        "Controller-Diff-Review": "passed",
        "Adversarial-Review": "passed",
        "Accepted-Findings": "none",
        "Verification": "passed",
        "Verified-Workspace-SHA256": "0" * 64
    }

    err = validate_ledger_sequence(parsed, session_dir, active_epoch_id)
    assert err is not None
    assert "Cross-cutting work requires an implementation adversary review" in err


def test_model_resolution_accepts_legitimate_models(tmp_path):
    """After removing the broken model-resolution validation, legitimate
    upstream model names (e.g. litellm or direct-anthropic model IDs) must
    no longer cause false-positive rejections.

    The agent type itself is the guarantee of correct routing — resolved_model
    can be any upstream model name.
    """
    session_dir = tmp_path / "session_model_ok"
    session_dir.mkdir()
    active_epoch_id = "ep_model_ok"

    ledger_events = [
        {"event": "AgentResult", "epoch_id": active_epoch_id, "subagent_type": "brigade-implementer", "status": "completed", "resolved_model": "LongCat-2.0"},
        {"event": "SubagentStart", "epoch_id": active_epoch_id, "agent_type": "brigade-implementer"},
        {"event": "SubagentStop", "epoch_id": active_epoch_id, "agent_type": "brigade-implementer"},
    ]
    with (session_dir / "ledger.jsonl").open("w") as f:
        for e in ledger_events:
            f.write(json.dumps(e) + "\n")

    records = [
        {"event": "SubagentStart", "agent_id": "1", "agent_type": "brigade-implementer", "epoch_id": active_epoch_id},
        {"event": "SubagentStop", "agent_id": "1", "agent_type": "brigade-implementer", "epoch_id": active_epoch_id},
    ]
    (session_dir / "agents.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    parsed = {
        "Workflow-Tier": "normal",
        "Implementation-Agent": "brigade-implementer",
        "Controller-Diff-Review": "passed",
        "Adversarial-Review": "not-required",
        "Accepted-Findings": "none",
        "Verification": "passed",
        "Verified-Workspace-SHA256": "0" * 64
    }

    err = validate_ledger_sequence(parsed, session_dir, active_epoch_id)
    assert err is None, f"Expected no error for legitimate upstream model, got: {err}"


def test_bounded_stop_hook_retries(tmp_path, monkeypatch):
    session_id = "test-stop-retry"
    cache_dir = tmp_path / ".cache" / "claude-brigade"
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

    # Attempt 3 - Max retries reached, assistant still claims success
    stdin_str = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io_string_stream(stdin_str))
    res = completion_guard_main()

    # Should still block (message says "Enhanced-Completion: yes", not "failed")
    assert len(blocked_reasons) == 3

    # Attempt 4 - Assistant acknowledges failure, should allow exit
    payload["last_assistant_message"] = "Enhanced-Completion: failed"
    stdin_str = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io_string_stream(stdin_str))
    res = completion_guard_main()

    assert res == 0
    assert len(blocked_reasons) == 3  # No additional block


class io_string_stream:
    def __init__(self, text):
        self.text = text
    def read(self):
        return self.text
