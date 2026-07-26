from pathlib import Path

from enhanced_router.agents_json import _split_top_level_csv, render_agents


def test_agent_tool_allowlist_is_not_split_inside_parentheses():
    value = "Agent(worker, reviewer), Read, Bash"
    assert _split_top_level_csv(value) == ["Agent(worker, reviewer)", "Read", "Bash"]


def test_bundle_agents_render_for_native_agents_flag():
    directory = Path(__file__).resolve().parents[1] / "agents"
    agents = render_agents(directory)
    assert agents["enhanced-controller"]["model"] == "sonnet[1m]"
    assert agents["longcat-implementer"]["model"] == "anthropic-longcat-2-0"
    assert "Agent(longcat-recon, longcat-implementer" in agents["enhanced-controller"]["tools"][0]
    assert agents["longcat-adversary"]["background"] is False
    assert "prompt" in agents["longcat-repairer"]
