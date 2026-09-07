"""Regression coverage for additive native slots and sidecar workers."""

from pathlib import Path

import pytest

from enhanced_router.agents_json import render_agents
from enhanced_router.config_models import ProfileSpec, SpecialistSpec
from enhanced_router.registry import ModelRegistry
from enhanced_router.state import RouteState


def bundled_registry() -> ModelRegistry:
    registry = ModelRegistry(Path(__file__).resolve().parents[1] / "config")
    registry.load_models()
    registry.load_profiles()
    registry.load_workflows()
    registry.load_providers()
    registry.load_fastpath()
    registry.load_sidecars()
    registry.load_sidecar_profiles()
    registry.load_launch_presets()
    registry._validate_cross_refs()
    return registry


def test_native_slots_and_sidecar_workers_are_additive():
    registry = bundled_registry()
    limits = registry.providers["freeinference"].limits
    assert limits.max_concurrency == 4
    assert limits.controller_reserve == 1
    assert limits.max_worker_concurrency == 3
    slots = registry.slot_alias_manifest("freeinference")
    assert set(slots) == {"main", "background", "sonnet", "haiku", "opus", "fable"}
    assert slots["background"]["environment_variable"] == "ANTHROPIC_SMALL_FAST_MODEL"
    assert slots["background"]["model_alias"] == (
        "anthropic-brigade-slot-background"
    )
    assert all(item["model_alias"] != "custom" for item in slots.values())

    workers = registry.native_worker_manifest("freeinference")
    assert workers["brigade-implementer"]["model_alias"] == "sonnet"
    assert workers["brigade-grounder"]["model_alias"] == "haiku"
    assert workers["brigade-critical-architect"]["model_alias"] == "fable"

    # The independently configured native sidecar remains present and keeps
    # its own public identity/model instead of collapsing into a role slot.
    sidecar = workers["brigade-sidecar-secondary-implementation-worker"]
    assert sidecar["source_kind"] == "sidecar_agent"
    assert sidecar["model_id"] == "minimax-m3"
    assert sidecar["worker_id"] == "secondary-implementation-worker"
    assert sidecar["model_alias"] == "anthropic-brigade-sidecar-secondary-implementation-worker"
    assert sidecar.get("slot") is None
    assert sidecar["public_model_alias"] not in {
        entry["public_model_alias"] for entry in slots.values()
    }
    aliases = registry.role_model_aliases()
    assert aliases["anthropic-brigade-sidecar-secondary-implementation-worker"] == "implementer"
    assert registry.role_model_bindings()["anthropic-brigade-sidecar-secondary-implementation-worker"] == "minimax-m3"


def test_worker_identities_are_model_neutral():
    registry = bundled_registry()
    manifest = registry.native_worker_manifest("freeinference")
    model_tokens = ("deepseek", "qwen", "kimi", "minimax", "glm", "longcat")

    for entry in manifest.values():
        identity = f"{entry['native_agent_name']} {entry['public_model_alias']}".lower()
        assert not any(token in identity for token in model_tokens), identity


def test_selected_profile_renders_native_slot_models_and_sidecars():
    registry = bundled_registry()
    agents = render_agents(
        Path(__file__).resolve().parents[1] / "agents",
        registry,
        "freeinference",
    )
    assert agents["brigade-implementer"]["model"] == "sonnet"
    assert agents["brigade-grounder"]["model"] == "haiku"
    assert agents["brigade-reviewer"]["model"] == "opus"
    assert agents["brigade-critical-architect"]["model"] == "fable"
    assert agents["brigade-sidecar-secondary-implementation-worker"]["model"] == (
        "anthropic-brigade-sidecar-secondary-implementation-worker"
    )
    assert "Edit" not in agents["brigade-reviewer"]["tools"]
    assert agents["brigade-implementer"]["isolation"] == "worktree"


def test_fi_flow_bundle_renders_explicit_sidecar_capabilities():
    registry = bundled_registry()
    agents = render_agents(
        Path(__file__).resolve().parents[1] / "agents",
        registry,
        "freeinference",
        "fi-flow-proven",
    )

    assert agents["brigade-sidecar-implementer"]["model"] == (
        "anthropic-brigade-sidecar-implementer"
    )
    assert agents["brigade-sidecar-implementer"]["isolation"] == "worktree"
    assert "Edit" in agents["brigade-sidecar-implementer"]["tools"]
    assert agents["brigade-sidecar-completion-controller"]["model"] == (
        "anthropic-brigade-sidecar-completion-controller"
    )
    assert "Edit" not in agents["brigade-sidecar-completion-controller"]["tools"]
    assert agents["brigade-sidecar-critical-gate"]["model"] == (
        "anthropic-brigade-sidecar-critical-gate"
    )


def test_sidecar_profile_does_not_rebind_native_slots():
    registry = bundled_registry()
    from enhanced_router.config_models import SidecarProfileSpec

    registry._sidecar_profiles["review-stack"] = SidecarProfileSpec(
        sidecar_agent_ids=["minimax-builder"],
    )
    before = registry.slot_alias_manifest("freeinference")
    workers = registry.native_worker_manifest("freeinference", "review-stack")
    after = registry.slot_alias_manifest("freeinference")

    assert set(after) == {"main", "background", "sonnet", "haiku", "opus", "fable"}
    assert {
        slot: (entry["model_id"], entry["provider_id"], entry["endpoint"])
        for slot, entry in after.items()
    } == {
        slot: (entry["model_id"], entry["provider_id"], entry["endpoint"])
        for slot, entry in before.items()
    }
    assert any(entry["source_kind"] == "sidecar_agent" for entry in workers.values())


def test_fi_flow_sidecar_bundle_is_separate_from_native_slots():
    registry = bundled_registry()
    preset = registry.launch_presets["fi-flow-proven"]
    assert preset.inference_profile_id == "freeinference"
    assert preset.sidecar_profile_id == "fi-flow-proven"
    assert preset.workflow_id == "normal"

    slots = registry.slot_alias_manifest(preset.inference_profile_id)
    workers = registry.native_worker_manifest(
        preset.inference_profile_id, preset.sidecar_profile_id,
    )
    sidecars = {
        entry["source_id"]: entry
        for entry in workers.values()
        if entry.get("source_kind") == "sidecar_agent"
    }
    assert sidecars["deepseek-implementer"]["model_id"] == "deepseek-v4-flash"
    assert sidecars["deepseek-implementer"]["worker_id"] == "implementation-worker"
    assert sidecars["glm-critical-gate"]["model_id"] == "glm-5.2"
    assert sidecars["glm-critical-gate"]["worker_id"] == "critical-gate"
    assert sidecars["qwen-grounder"]["template"] == "brigade-grounder"
    assert sidecars["minimax-senior-reviewer"]["template"] == "brigade-sidecar-senior-reviewer"
    assert all(entry.get("slot") is None for entry in sidecars.values())
    assert all(
        entry["public_model_alias"] not in {
            slot["public_model_alias"] for slot in slots.values()
        }
        for entry in sidecars.values()
    )

    phases = registry.workflows["normal"].phases
    assert next(item for item in phases if item.id == "implementation").sidecar_agent == "implementation-worker"
    assert next(item for item in phases if item.id == "critical-gate").sidecar_agent == "critical-gate"
    assert registry.get_sidecar_agent("implementation-worker").model_id == "deepseek-v4-flash"
    assert registry.get_sidecar_agent("critical-gate").model_id == "glm-5.2"


def test_claude_augmented_preset_composes_native_and_sidecar_workers():
    registry = bundled_registry()
    preset = registry.launch_presets["claude-augmented"]
    assert preset.workflow_id == "sidecar-augmented-normal"
    workflow = registry.get_workflow(preset.workflow_id)
    assert workflow is not None
    implementation = next(item for item in workflow.phases if item.id == "implementation")
    post_grounding = next(item for item in workflow.phases if item.id == "post-grounding")
    critical_gate = next(item for item in workflow.phases if item.id == "critical-gate")
    assert implementation.agent_id == "implementer"
    assert implementation.sidecar_agent is None
    assert post_grounding.sidecar_agent == "grounder"
    assert critical_gate.sidecar_agent == "critical-gate"
    assert any(item.agent_id == "critical-verifier" for item in workflow.phases)


def test_epoch_persists_immutable_slot_bindings(tmp_path, monkeypatch):
    registry = bundled_registry()
    import enhanced_router.registry as registry_module

    monkeypatch.setattr(registry_module, "get_registry", lambda: registry)
    state = RouteState(tmp_path / "state.db")
    state.create_run("run-slots")
    state.create_epoch_from_profile("run-slots", "ep-1", "normal", "freeinference")
    assert state.get_active_epoch("run-slots")["composition_mode"] == "sidecar-only"

    bindings = state.get_slot_bindings("run-slots", "ep-1")
    assert {item["slot_name"] for item in bindings} == {
        "main", "background", "sonnet", "haiku", "opus", "fable"
    }
    sonnet = next(item for item in bindings if item["slot_name"] == "sonnet")
    assert sonnet["model_alias"] == "sonnet"
    assert sonnet["logical_model_id"] == "kimi-k2.7-code"
    assert sonnet["public_alias"] == "anthropic-brigade-slot-sonnet"


def test_custom_lane_is_optional_and_never_emits_a_custom_native_alias():
    registry = bundled_registry()
    profile = registry.profiles["freeinference"]
    from enhanced_router.config_models import SlotRouteSpec

    profile.slots["custom"] = SlotRouteSpec(
        model="glm-5.1",
        reserve_class="optional",
    )
    slots = registry.slot_alias_manifest("freeinference")
    custom = slots["custom"]
    assert custom["model_alias"] == "anthropic-brigade-slot-custom"
    assert custom["native_alias"] is None
    assert custom["environment_variable"] == "ANTHROPIC_CUSTOM_MODEL_OPTION"


def test_action_ids_use_stable_package_slots(tmp_path, monkeypatch):
    """Concurrent action reads use persisted package identity, not row count."""
    registry = bundled_registry()
    import enhanced_router.registry as registry_module

    monkeypatch.setattr(registry_module, "get_registry", lambda: registry)
    state = RouteState(tmp_path / "state-actions.db")
    state.create_run("run-actions")
    state.create_epoch("run-actions", "ep-1", "cross-cutting", "freeinference")
    state.set_role_route("run-actions", "ep-1", "recon", "qwen3.6-35b", "test")
    state.initialize_workflow_phases(
        "run-actions", "ep-1", [{"id": "recon", "roles": ["recon"], "max_fanout": 2}],
    )
    state.start_phase("run-actions", "ep-1", "recon")

    first = state.get_runnable_action_wave("run-actions", "ep-1", limit=2)
    native = [item for item in first if item.get("action_kind") == "native_agent"]
    assert native
    assert all(":slot-" in item["package_id"] for item in native)
    assert all(":attempt-1" in item["action_id"] for item in native)
    assert all("len(" not in item["action_id"] for item in native)


def test_catalog_only_specialist_is_excluded_explicitly():
    specialist = SpecialistSpec(
        model="minimax-m3",
        roles=["architect"],
        launchable=False,
    )
    profile = ProfileSpec(
        recon="qwen3.6-35b",
        implementer="kimi-k2.7-code",
        adversary="glm-5.1",
        repairer="glm-5-turbo",
        controller_model="glm-5.1",
        specialists={"catalog-architect": specialist},
    )
    registry = bundled_registry()
    registry.profiles["catalog-test"] = profile

    manifest = registry.native_worker_manifest("catalog-test")

    assert not any(
        entry.get("source_id") == "catalog-architect"
        for entry in manifest.values()
    )


def test_launchable_specialist_without_identity_fails_closed():
    with pytest.raises(ValueError, match="launchable specialists"):
        SpecialistSpec(
            model="minimax-m3",
            roles=["architect"],
        )
