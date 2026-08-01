from pathlib import Path

import pytest
from enhanced_router.agents_json import _split_top_level_csv, render_agents


def test_agent_tool_allowlist_is_not_split_inside_parentheses():
    value = "Agent(worker, reviewer), Read, Bash"
    assert _split_top_level_csv(value) == ["Agent(worker, reviewer)", "Read", "Bash"]


def test_bundle_agents_render_for_native_agents_flag():
    directory = Path(__file__).resolve().parents[1] / "agents"
    agents = render_agents(directory)
    assert agents["brigade-implementer"]["model"] == "anthropic-brigade-implementer"
    assert agents["brigade-adversary"]["background"] is True
    assert "prompt" in agents["brigade-repairer"]
    # Verify tools field is properly parsed as a list
    assert isinstance(agents["brigade-recon"]["tools"], list)


def test_registry_manifest_can_generate_a_model_qualified_agent(tmp_path):
    source = Path(__file__).resolve().parents[1] / "agents"
    for name in ("brigade-recon.md", "brigade-implementer.md", "brigade-adversary.md", "brigade-repairer.md", "controller-direct.md"):
        (tmp_path / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")

    class RegistryStub:
        def specialist_manifest(self):
            return {
                "brigade-custom-scout": {
                    "native_agent_name": "brigade-custom-scout",
                    "public_model_alias": "anthropic-brigade-custom-scout",
                    "model_id": "custom-model",
                    "role": "recon",
                    "profile_id": "custom",
                }
            }

    agents = render_agents(tmp_path, RegistryStub())
    assert agents["brigade-custom-scout"]["model"] == "anthropic-brigade-custom-scout"
    assert "custom-model" in agents["brigade-custom-scout"]["description"]


def test_controller_append_has_brigade_references():
    """Verify the controller-append.md body references brigade agent names."""
    directory = Path(__file__).resolve().parents[1] / "agents"
    append_path = directory / "controller-append.md"
    text = append_path.read_text(encoding="utf-8")
    assert "brigade-implementer" in text
    assert "brigade-repairer" in text
    assert "brigade-recon" in text
    assert "brigade-adversary" in text
    assert "brigade" in text.lower()
    # Should not reference old agent names or enhanced-controller
    assert "longcat-" not in text
    assert "enhanced-controller" not in text


def test_controller_direct_uses_active_binding():
    directory = Path(__file__).resolve().parents[1] / "agents"
    agents = render_agents(directory)
    body = agents["controller-direct"]["prompt"]
    assert "active-controller mutation path" in body
    assert "LongCat" not in body


def test_required_roles_validation():
    directory = Path(__file__).resolve().parents[1] / "agents"
    agents = render_agents(directory)
    assert "brigade-recon" in agents
    assert "brigade-implementer" in agents
    assert "brigade-adversary" in agents
    assert "brigade-repairer" in agents
    assert "controller-direct" in agents


def test_expected_models_validation():
    directory = Path(__file__).resolve().parents[1] / "agents"
    agents = render_agents(directory)
    assert agents["brigade-recon"]["model"] == "anthropic-brigade-recon"
    assert agents["brigade-implementer"]["model"] == "anthropic-brigade-implementer"
    assert agents["brigade-adversary"]["model"] == "anthropic-brigade-adversary"
    assert agents["brigade-repairer"]["model"] == "anthropic-brigade-repairer"


def test_missing_role_raises_value_error(tmp_path):
    # Write a minimal valid agent that's NOT one of the required roles
    agent_file = tmp_path / "dummy.md"
    agent_file.write_text(
        "---\n"
        "name: dummy\n"
        "description: A dummy agent\n"
        "---\n\n"
        "prompt content\n"
    )
    from enhanced_router.agents_json import render_agents as _ra
    with pytest.raises(ValueError, match="Missing required agent roles"):
        _ra(tmp_path)


def test_wrong_model_raises_value_error(tmp_path):
    # Write all required agents, but give one a wrong model
    for name, model in [
        ("brigade-recon", "anthropic-brigade-recon"),
        ("brigade-adversary", "anthropic-brigade-adversary"),
        ("brigade-repairer", "anthropic-brigade-repairer"),
        ("controller-direct", None),
        ("brigade-implementer", "wrong-model"),
    ]:
        (tmp_path / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: Agent {name}\nmodel: {model}\n"
            "---\n\nprompt content\n"
        )
    from enhanced_router.agents_json import render_agents as _ra
    with pytest.raises(ValueError, match="has model 'wrong-model' but expected 'anthropic-brigade-implementer'"):
        _ra(tmp_path)
