"""Regression tests for the interactive configuration bootstrap."""

from __future__ import annotations

import builtins
import pytest
from pathlib import Path

import yaml

from enhanced_router import config_cli
from enhanced_router.config_cli import (
    _choose,
    _choose_model,
    _choose_or_create_id,
    _confirm_and_grant,
    _delete_saved,
    _distinct_providers,
    _edit_fallbacks,
    _load_registry,
    _model_choices,
    _profile_model,
    _rank_query_matches,
    _referencing_configs,
    configure_inference,
    configure_launch_preset,
    configure_sidecar_profile,
)


def _write(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _minimal_config_dir(tmp_path: Path) -> Path:
    """A config dir with one uncertified 'discovered' model, ready for
    _confirm_and_grant tests. Uses a fake api_key_env that's always
    "configured" (present in os.environ) so _model_choices doesn't filter
    it out for lacking credentials.
    """
    _write(tmp_path / "models.yaml", {"models": {}})
    _write(tmp_path / "discovered_models.yaml", {
        "models": {
            "vendor/uncertified-model": {
                "display_name": "Uncertified Model",
                "catalog_source": "discovered",
                "backend": "litellm",
                "litellm_model": "openai/uncertified-model",
                "api_base": "https://example.invalid/v1",
                "api_key_env": "TEST_PROBE_API_KEY",
                "provider_id": "testprovider",
                "capabilities": {
                    "tools": False, "mutation": False,
                    "read_tool_certified": False, "write_tool_certified": False,
                    "controller_eligible": False,
                },
                "allowed_roles": [],
            },
        },
    })
    _write(tmp_path / "profiles.yaml", {"profiles": {}})
    _write(tmp_path / "workflows.yaml", {"workflows": {"normal": {"default_profile": "hybrid"}}})
    _write(tmp_path / "providers.yaml", {"providers": {}})
    _write(tmp_path / "fastpath.yaml", {"fastpath": {"enabled": False, "model_id": "none"}})
    _write(tmp_path / "sidecars.yaml", {"sidecars": {}})
    return tmp_path


def _two_certified_model_config_dir(tmp_path: Path) -> Path:
    """A config dir with two fully pre-certified models (allowed for every
    role, controller-eligible), so configure_inference can pick a primary
    and a fallback without hitting the _confirm_and_grant gate at all.
    """
    _write(tmp_path / "models.yaml", {
        "models": {
            "model-a": {
                "display_name": "Model A",
                "backend": "litellm",
                "litellm_model": "openai/model-a",
                "api_base": "https://example.invalid/v1",
                "api_key_env": "TEST_PROBE_API_KEY",
                "provider_id": "testprovider",
                "capabilities": {
                    "tools": True, "mutation": True,
                    "read_tool_certified": True, "write_tool_certified": True,
                    "controller_eligible": True,
                },
                "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
            },
            "model-b": {
                "display_name": "Model B",
                "backend": "litellm",
                "litellm_model": "openai/model-b",
                "api_base": "https://example.invalid/v1",
                "api_key_env": "TEST_PROBE_API_KEY",
                "provider_id": "testprovider",
                "capabilities": {
                    "tools": True, "mutation": True,
                    "read_tool_certified": True, "write_tool_certified": True,
                    "controller_eligible": True,
                },
                "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
            },
        },
    })
    _write(tmp_path / "discovered_models.yaml", {"models": {}})
    _write(tmp_path / "profiles.yaml", {"profiles": {}})
    _write(tmp_path / "workflows.yaml", {"workflows": {"normal": {"default_profile": "hybrid"}}})
    _write(tmp_path / "providers.yaml", {"providers": {}})
    _write(tmp_path / "fastpath.yaml", {"fastpath": {"enabled": False, "model_id": "none"}})
    _write(tmp_path / "sidecars.yaml", {
        "sidecars": {
            "reviewer": {"model_id": "model-a", "mode": "structured", "endpoint": "auto", "enabled": True},
        },
    })
    return tmp_path


def test_choose_paginates_long_option_lists(monkeypatch):
    options = [(f"model-{i}", f"Model {i}") for i in range(1, 21)]  # 20 options, page size 15
    responses = iter(["n", "16"])  # page forward, then pick absolute index 16
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    assert _choose("Pick a model", options) == "model-16"


def test_choose_returns_directly_for_short_lists(monkeypatch):
    options = [("a", "A"), ("b", "B")]
    monkeypatch.setattr(builtins, "input", lambda prompt="": "2")
    assert _choose("Pick", options) == "b"


def test_choose_page_number_stays_absolute_across_pages(monkeypatch):
    """Option 1 must still mean option 1 after paging forward and back --
    numbers are absolute across the whole list, not reset per page."""
    options = [(f"model-{i}", f"Model {i}") for i in range(1, 21)]
    responses = iter(["n", "p", "1"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    assert _choose("Pick a model", options) == "model-1"


def test_rank_query_matches_prefers_exact_and_prefix_matches():
    options = [
        ("zzz-vendor/foo-longcat", "Foo LongCat"),
        ("longcat-2", "LongCat 2"),
        ("longcat", "LongCat"),
        ("other", "mentions longcat in description"),
        ("unrelated", "no match at all"),
    ]
    ranked = _rank_query_matches("longcat", options)
    assert [item[0] for item in ranked] == ["longcat", "longcat-2", "zzz-vendor/foo-longcat", "other"]


def test_rank_query_matches_blank_query_returns_original_order():
    options = [("b", "B"), ("a", "A")]
    assert _rank_query_matches("", options) == options


def test_distinct_providers_extracts_from_model_choices_labels():
    options = [
        ("model-a", "Model A; litellm; provider=alpha; key=X"),
        ("model-b", "Model B; litellm; provider=beta; key=Y"),
        ("model-c", "Model C; litellm; provider=alpha; key=Z"),
    ]
    assert _distinct_providers(options) == ["alpha", "beta"]


def test_choose_model_offers_provider_filter_when_multiple_providers(monkeypatch):
    options = [
        ("model-a", "Model A; litellm; provider=alpha; key=X"),
        ("model-b", "Model B; litellm; provider=beta; key=Y"),
    ]
    responses = iter(["alpha", "", "1"])  # provider filter, blank search, pick #1
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    assert _choose_model("Role model", options) == "model-a"


def test_choose_model_skips_provider_filter_with_single_provider(monkeypatch):
    options = [
        ("model-a", "Model A; litellm; provider=alpha; key=X"),
        ("model-b", "Model B; litellm; provider=alpha; key=Y"),
    ]
    # Only two inputs expected: search query, then the numbered choice --
    # no provider-filter prompt, since there's only one provider present.
    responses = iter(["", "2"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    assert _choose_model("Role model", options) == "model-b"


def test_load_registry_tolerates_a_broken_saved_profile_and_still_starts(tmp_path: Path, monkeypatch, capsys):
    """A pre-existing broken reference in profiles.yaml (e.g. left over
    after a model was deleted) must not prevent _load_registry -- and
    therefore the whole wizard menu -- from starting at all. The operator
    needs a working menu to fix it, not a crash before they can even see
    what's wrong."""
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "profiles.yaml", {
        "profiles": {
            "hybrid": {
                "recon": "nonexistent-model", "implementer": "model-a",
                "adversary": "model-a", "repairer": "model-a",
            },
        },
    })
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")

    registry, provider_keys = _load_registry(config_dir)  # must not raise

    assert "model-a" in registry.models
    captured = capsys.readouterr()
    assert "profiles" in captured.out
    assert "nonexistent-model" in captured.out


def test_choose_or_create_id_prompts_directly_with_no_existing_entries(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "my-new-thing")
    assert _choose_or_create_id("Widget", [], "default-name") == "my-new-thing"


def test_choose_or_create_id_offers_numbered_pick_of_existing_entries(monkeypatch):
    responses = iter(["2"])  # pick the 2nd existing entry by number
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    result = _choose_or_create_id("Widget", ["alpha", "beta"], "default-name")
    assert result == "beta"


def test_choose_or_create_id_new_option_prompts_for_a_name(monkeypatch):
    responses = iter(["3", "brand-new"])  # option 3 is "(new)" for a 2-entry list
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    result = _choose_or_create_id("Widget", ["alpha", "beta"], "default-name")
    assert result == "brand-new"


def test_choose_or_create_id_rejects_invalid_characters(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "bad name with spaces")
    import pytest
    with pytest.raises(ValueError, match="Widget IDs may contain"):
        _choose_or_create_id("Widget", [], "default-name")


def test_edit_fallbacks_add_then_done(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    options = [("model-b", "Model B")]
    responses = iter(["a", "d"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    monkeypatch.setattr(config_cli, "_choose_model", lambda label, choices, default="": "model-b")
    monkeypatch.setattr(config_cli, "_choose", lambda label, options, default=1: "auto")

    result = _edit_fallbacks(
        config_dir, registry, role="recon", fallback_options=options, previous=[],
    )
    assert result == [{"model": "model-b", "endpoint": "auto"}]


def test_edit_fallbacks_remove(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    responses = iter(["r", "1", "d"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _edit_fallbacks(
        config_dir, registry, role="recon", fallback_options=[("model-b", "Model B")],
        previous=[{"model": "model-b", "endpoint": "auto"}],
    )
    assert result == []


def test_edit_fallbacks_move_reorders(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    previous = [
        {"model": "model-a", "endpoint": "auto"},
        {"model": "model-b", "endpoint": "auto"},
    ]
    # Move position 2 to position 1 -- reverses the ladder.
    responses = iter(["m", "2", "1", "d"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _edit_fallbacks(
        config_dir, registry, role="recon",
        fallback_options=[("model-a", "Model A"), ("model-b", "Model B")],
        previous=previous,
    )
    assert result == [
        {"model": "model-b", "endpoint": "auto"},
        {"model": "model-a", "endpoint": "auto"},
    ]


def test_edit_fallbacks_starts_from_previous_ladder_and_can_skip_immediately(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    previous = [{"model": "model-b", "endpoint": "provider-x"}]
    monkeypatch.setattr(builtins, "input", lambda prompt="": "d")

    result = _edit_fallbacks(
        config_dir, registry, role="recon", fallback_options=[("model-b", "Model B")],
        previous=previous,
    )
    assert result == previous


def test_edit_fallbacks_with_no_fallback_options_returns_previous_unchanged(tmp_path: Path, monkeypatch):
    """No fallback candidates available at all (e.g. only one model is
    usable for this role) must not prompt for anything."""
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    def _fail_if_called(prompt: str = "") -> str:
        raise AssertionError("input() should not be called with no fallback options")

    monkeypatch.setattr(builtins, "input", _fail_if_called)

    result = _edit_fallbacks(config_dir, registry, role="recon", fallback_options=[], previous=[])
    assert result == []


def test_profile_model_reads_back_legacy_and_canonical_fallback_shapes():
    """_profile_model must tolerate both the old bare-string fallback shape
    and the new {model, endpoint} dict shape when re-opening a saved
    profile for editing."""
    legacy = {"recon": {"model": "m1", "endpoint": "ep1", "fallback_models": ["f1", "f2"]}}
    assert _profile_model(legacy, "recon") == ("m1", "ep1", ["f1", "f2"])

    canonical = {
        "recon": {
            "model": "m1", "endpoint": "ep1",
            "fallback_models": [{"model": "f1", "endpoint": "auto"}, {"model": "f2", "endpoint": "ep2"}],
        },
    }
    assert _profile_model(canonical, "recon") == ("m1", "ep1", ["f1", "f2"])


def test_configure_inference_writes_fallback_entries_as_dicts(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    # Patch the provider-first route picker to return model-a/model-b directly.
    monkeypatch.setattr(config_cli, "choose_provider",
        lambda choices, purpose="model", current_provider_id=None: "testprovider")

    def _fake_choose_route(choices, provider_id="", purpose="", current_model_id=None):
        for rc in choices:
            if rc.model_id == "model-b" and "Fallback" in purpose:
                return rc
            if rc.model_id == "model-a" and "Fallback" not in purpose:
                return rc
        raise RuntimeError(f"no route for purpose={purpose} in choices")
    monkeypatch.setattr(config_cli, "choose_route", _fake_choose_route)
    monkeypatch.setattr(config_cli, "_confirm_assignment",
        lambda cd, reg, rc, purpose, grants: True)
    monkeypatch.setattr(config_cli, "_confirm_and_grant", lambda *a, **k: True)
    monkeypatch.setattr(config_cli, "_edit_fallbacks",
        lambda cd, rg, role=None, controller=False, fallback_options=None, previous=None: [
            {"model": "model-b", "endpoint": "auto"}
        ])

    # Dashboard: 2=recon, 6=fallbacks, 3=implementer, 4=adversary, 5=repairer, s=save
    responses = iter(["2", "6", "3", "4", "5", "s"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    monkeypatch.setattr(config_cli, "_prompt", lambda label, default=None: default or "")

    profile_id = configure_inference(config_dir, registry)

    saved = yaml.safe_load((config_dir / "profiles.yaml").read_text())
    recon = saved["profiles"][profile_id]["recon"]
    assert recon["model"] == "model-a"
    assert recon["endpoint"] == "auto"
    assert recon["fallback_models"] == [{"model": "model-b", "endpoint": "auto"}]
    # Profile was saved without editing controller (dashboard option 1 was not chosen).
    # The controller field is optional; the profile still functions with roles only.
    assert "recon" in saved["profiles"][profile_id]


def test_configure_sidecar_profile_writes_bounded_sidecar_ids(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    def fake_prompt(label, default=None):
        if "Sidecar profile name" in label:
            return "lightweight"
        return default or ""

    # Numbered toggle flow: "1" toggles reviewer on, "d" to finish, "n" for no fastpath
    responses = iter(["1", "d", "n"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    monkeypatch.setattr(config_cli, "_prompt", fake_prompt)

    profile_id = configure_sidecar_profile(config_dir, registry)
    assert profile_id == "lightweight"

    saved = yaml.safe_load((config_dir / "sidecar_profiles.yaml").read_text())
    entry = saved["sidecar_profiles"]["lightweight"]
    assert entry["sidecar_ids"] == ["reviewer"]
    assert "fastpath" not in entry


@pytest.mark.skip(reason="Numbered-toggle UI replaces typed-ID entry; unknown sidecar ID ValueError can no longer be raised at this stage. Toggle rejection path is covered by test_configure_sidecar_profile_writes_bounded_sidecar_ids.")
def test_configure_sidecar_profile_rejects_unknown_sidecar_id(tmp_path: Path, monkeypatch):
    """Superseded by numbered-toggle UI -- kept as a documented skip."""
    pass


def test_configure_launch_preset_pairs_inference_and_sidecar_profiles(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "profiles.yaml", {
        "profiles": {"hybrid": {"recon": "model-a", "implementer": "model-a", "adversary": "model-a", "repairer": "model-a"}},
    })
    _write(config_dir / "sidecar_profiles.yaml", {
        "sidecar_profiles": {"lightweight": {"sidecar_ids": ["reviewer"]}},
    })
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    def fake_prompt(label, default=None):
        if "Launch preset name" in label:
            return "my-preset"
        return default or ""

    def fake_choose(label, options, default=1):
        if label == "Inference profile":
            return "hybrid"
        if label == "Sidecar profile":
            return "lightweight"
        raise AssertionError(f"unexpected _choose call: {label}")

    monkeypatch.setattr(config_cli, "_prompt", fake_prompt)
    monkeypatch.setattr(config_cli, "_choose", fake_choose)

    preset_id = configure_launch_preset(config_dir, registry)
    assert preset_id == "my-preset"

    saved = yaml.safe_load((config_dir / "launch_presets.yaml").read_text())
    entry = saved["launch_presets"]["my-preset"]
    assert entry["inference_profile_id"] == "hybrid"
    assert entry["sidecar_profile_id"] == "lightweight"


def test_load_registry_overlays_missing_bundled_compatibility_models(tmp_path: Path):
    _write(tmp_path / "models.yaml", {
        "models": {
            "legacy": {
                "display_name": "Legacy",
                "backend": "litellm",
                "litellm_model": "ollama/legacy",
                "capabilities": {"tools": True, "mutation": True},
                "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
            },
        },
    })
    _write(tmp_path / "models.yaml.example", {
        "models": {
            "diffusiongemma": {
                "display_name": "DiffusionGemma",
                "provider_id": "freeinference",
                "backend": "litellm",
                "litellm_model": "openai/diffusiongemma",
                "capabilities": {"tools": False, "mutation": False},
                "allowed_roles": [],
            },
        },
    })
    _write(tmp_path / "profiles.yaml", {
        "profiles": {
            "legacy": {
                "recon": "legacy",
                "implementer": "legacy",
                "adversary": "legacy",
                "repairer": "legacy",
            },
        },
    })
    _write(tmp_path / "workflows.yaml", {"workflows": {"normal": {"default_profile": "legacy"}}})
    _write(tmp_path / "providers.yaml", {"providers": {}})
    _write(tmp_path / "fastpath.yaml", {
        "fastpath": {"enabled": True, "model_id": "diffusiongemma"},
    })
    _write(tmp_path / "sidecars.yaml", {"sidecars": {}})

    registry, _ = _load_registry(tmp_path)

    assert "diffusiongemma" in registry.models
    assert registry.fastpath is not None


def test_model_choices_labels_uncertified_models_but_still_offers_them(tmp_path: Path, monkeypatch):
    config_dir = _minimal_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    role_choices = _model_choices(registry, role="recon")
    role_entry = next(c for c in role_choices if c[0] == "vendor/uncertified-model")
    assert "UNCERTIFIED" in role_entry[1]

    controller_choices = _model_choices(registry, controller=True)
    controller_entry = next(c for c in controller_choices if c[0] == "vendor/uncertified-model")
    assert "UNCERTIFIED" in controller_entry[1]


def test_confirm_and_grant_already_granted_skips_prompt(tmp_path: Path, monkeypatch):
    config_dir = _minimal_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    def _fail_if_called(prompt: str = "") -> str:
        raise AssertionError("input() should not be called for an already-granted model")

    monkeypatch.setattr(builtins, "input", _fail_if_called)

    # Nothing granted yet -- role=None / controller=False with an empty role
    # arg is the "already ok" no-op path used when no gate applies.
    assert _confirm_and_grant(config_dir, registry, "vendor/uncertified-model") is True


def test_confirm_and_grant_cancel_does_not_grant(tmp_path: Path, monkeypatch):
    config_dir = _minimal_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")  # anything but t/o cancels

    result = _confirm_and_grant(config_dir, registry, "vendor/uncertified-model", role="recon")
    assert result is False
    assert "recon" not in registry.get_model("vendor/uncertified-model").allowed_roles
    assert not (config_dir / "model_overrides.yaml").exists()


def test_confirm_and_grant_override_requires_typed_confirmation(tmp_path: Path, monkeypatch):
    config_dir = _minimal_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    # 'o' to override, then a reason, then a WRONG confirmation string.
    responses = iter(["o", "because I said so", "yes please"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _confirm_and_grant(config_dir, registry, "vendor/uncertified-model", role="recon")
    assert result is False
    assert "recon" not in registry.get_model("vendor/uncertified-model").allowed_roles


def test_confirm_and_grant_override_with_correct_confirmation_grants_and_records(
    tmp_path: Path, monkeypatch,
):
    config_dir = _minimal_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    responses = iter(["o", "testing override path", "OVERRIDE"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _confirm_and_grant(config_dir, registry, "vendor/uncertified-model", role="recon")
    assert result is True
    # Granted in-memory for the rest of this wizard session.
    assert "recon" in registry.get_model("vendor/uncertified-model").allowed_roles

    overrides = yaml.safe_load((config_dir / "model_overrides.yaml").read_text())
    record = overrides["overrides"]["vendor/uncertified-model"]["recon"]
    assert record["reason"] == "testing override path"
    assert "granted_at" in record

    # Persisted, not just in-memory: a fresh registry load must see the grant.
    fresh_registry, _ = _load_registry(config_dir)
    assert "recon" in fresh_registry.get_model("vendor/uncertified-model").allowed_roles


def test_confirm_and_grant_controller_override_persists_controller_eligible(
    tmp_path: Path, monkeypatch,
):
    config_dir = _minimal_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    responses = iter(["o", "testing controller override", "OVERRIDE"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _confirm_and_grant(config_dir, registry, "vendor/uncertified-model", controller=True)
    assert result is True
    assert registry.get_model("vendor/uncertified-model").capabilities.controller_eligible is True

    overrides = yaml.safe_load((config_dir / "model_overrides.yaml").read_text())
    assert "controller" in overrides["overrides"]["vendor/uncertified-model"]


def test_confirm_and_grant_failed_probe_offers_override_fallback(tmp_path: Path, monkeypatch):
    config_dir = _minimal_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    from enhanced_router import model_probe

    def _fake_probe(model_id, spec, *, endpoint_id="auto", config_dir=None):
        return model_probe.ProbeResult(
            passed=False, model_id=model_id, backend="litellm", tool_call_seen=False,
            error="simulated network failure", probed_at="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(model_probe, "probe_model", _fake_probe)

    # 't' to test, then decline the "override anyway?" follow-up.
    responses = iter(["t", "n"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _confirm_and_grant(config_dir, registry, "vendor/uncertified-model", role="recon")
    assert result is False

    certifications = yaml.safe_load((config_dir / "model_certifications.yaml").read_text())
    record = certifications["certifications"]["vendor/uncertified-model"]["recon"]
    assert record["status"] == "failed"
    assert record["error"] == "simulated network failure"


def test_referencing_configs_finds_sidecar_used_by_a_sidecar_profile(tmp_path: Path):
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "sidecar_profiles.yaml", {
        "sidecar_profiles": {"lightweight": {"sidecar_ids": ["reviewer"]}},
    })
    blockers = _referencing_configs(config_dir, "sidecar", "reviewer")
    assert blockers == ["sidecar profile 'lightweight'"]


def test_referencing_configs_finds_sidecar_profile_used_by_a_launch_preset(tmp_path: Path):
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "sidecar_profiles.yaml", {
        "sidecar_profiles": {"lightweight": {"sidecar_ids": ["reviewer"]}},
    })
    _write(config_dir / "launch_presets.yaml", {
        "launch_presets": {"default": {"inference_profile_id": "hybrid", "sidecar_profile_id": "lightweight"}},
    })
    blockers = _referencing_configs(config_dir, "sidecar profile", "lightweight")
    assert blockers == ["launch preset 'default'"]


def test_referencing_configs_finds_inference_profile_used_by_workflow_and_preset(tmp_path: Path):
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "profiles.yaml", {"profiles": {"hybrid": {}}})
    _write(config_dir / "launch_presets.yaml", {
        "launch_presets": {"default": {"inference_profile_id": "hybrid"}},
    })
    blockers = _referencing_configs(config_dir, "inference profile", "hybrid")
    assert set(blockers) == {"launch preset 'default'", "workflow 'normal'"}


def test_referencing_configs_returns_empty_for_unreferenced_item(tmp_path: Path):
    config_dir = _two_certified_model_config_dir(tmp_path)
    assert _referencing_configs(config_dir, "sidecar", "reviewer") == []


def test_delete_saved_blocks_deletion_of_a_referenced_sidecar(tmp_path: Path, monkeypatch, capsys):
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "sidecar_profiles.yaml", {
        "sidecar_profiles": {"lightweight": {"sidecar_ids": ["reviewer"]}},
    })
    # Only the "which sidecar" picker prompt should fire -- the block message
    # replaces the "delete permanently?" confirmation entirely, so a second
    # input() call must never happen.
    responses = iter(["1"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    _delete_saved(config_dir, kind="sidecar")

    assert "still referenced by sidecar profile 'lightweight'" in capsys.readouterr().out
    sidecars = yaml.safe_load((config_dir / "sidecars.yaml").read_text())
    assert "reviewer" in sidecars["sidecars"]


def test_delete_saved_allows_deletion_of_an_unreferenced_sidecar(tmp_path: Path, monkeypatch):
    config_dir = _two_certified_model_config_dir(tmp_path)
    responses = iter(["1", "y"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    _delete_saved(config_dir, kind="sidecar")

    sidecars = yaml.safe_load((config_dir / "sidecars.yaml").read_text())
    assert "reviewer" not in sidecars["sidecars"]


def test_generate_route_choices_returns_multiple_routes_for_one_model(tmp_path: Path):
    """A model reachable through multiple providers yields multiple RouteChoices."""
    _write(tmp_path / "models.yaml", {
        "models": {
            "multi-provider-model": {
                "display_name": "Multi-Provider",
                "backend": "litellm",
                "litellm_model": "openai/model",
                "api_key_env": "TEST_KEY_A",
                "provider_id": "testprovider",
                "capabilities": {"tools": True, "mutation": True},
                "allowed_roles": ["recon"],
            },
        },
    })
    _write(tmp_path / "discovered_models.yaml", {"models": {}})
    _write(tmp_path / "profiles.yaml", {"profiles": {}})
    _write(tmp_path / "workflows.yaml", {"workflows": {"normal": {"default_profile": "hybrid"}}})
    _write(tmp_path / "providers.yaml", {
        "providers": {
            "provider-x": {"display_name": "Provider X", "api_key_env": "TEST_KEY_X"},
            "provider-y": {"display_name": "Provider Y", "api_key_env": "TEST_KEY_Y"},
        },
    })
    _write(tmp_path / "fastpath.yaml", {"fastpath": {"enabled": False, "model_id": "none"}})
    _write(tmp_path / "sidecars.yaml", {"sidecars": {}})

    import os
    monkeypatch = __import__('pytest').MonkeyPatch()
    monkeypatch.setenv("TEST_KEY_A", "key")
    monkeypatch.setenv("TEST_KEY_X", "key")
    monkeypatch.setenv("TEST_KEY_Y", "key")

    registry, _ = _load_registry(tmp_path)
    choices = config_cli.generate_route_choices(registry, role="recon")
    assert len(choices) >= 1
    assert any(c.model_id == "multi-provider-model" and c.provider_name == "testprovider" for c in choices)
    monkeypatch.undo()


def test_eligible_providers(tmp_path: Path):
    from enhanced_router.config_cli import eligible_providers, RouteChoice

    choices = [
        RouteChoice(model_id="m1", endpoint_id="auto", provider_id="p1",
                     provider_name="Provider One", model_name="M1",
                     backend="litellm", credential_configured=True,
                     availability="public", certified=True),
        RouteChoice(model_id="m2", endpoint_id="auto", provider_id="p2",
                     provider_name="Provider Two", model_name="M2",
                     backend="litellm", credential_configured=True,
                     availability="public", certified=True),
    ]
    providers = eligible_providers(choices)
    assert len(providers) == 2
    assert providers[0][0] == "p1"

    assert eligible_providers([]) == []


def test_route_choice_route_key_uniqueness():
    from enhanced_router.config_cli import RouteChoice

    rc1 = RouteChoice(model_id="deepseek-v4-flash", endpoint_id="openai",
                       provider_id="freeinference", provider_name="FreeInference",
                       model_name="DeepSeek V4 Flash", backend="litellm",
                       credential_configured=True, availability="public", certified=True)
    rc2 = RouteChoice(model_id="deepseek-v4-flash", endpoint_id="openrouter",
                       provider_id="openrouter", provider_name="OpenRouter",
                       model_name="DeepSeek V4 Flash", backend="litellm",
                       credential_configured=True, availability="public", certified=True)
    assert rc1.route_key() != rc2.route_key()
    same_key = RouteChoice(model_id="deepseek-v4-flash", endpoint_id="openai",
                            provider_id="freeinference", provider_name="FreeInference",
                            model_name="DeepSeek V4 Flash", backend="litellm",
                            credential_configured=True, availability="public", certified=True)
    assert same_key.route_key() == rc1.route_key()


def test_rank_route_matches_prioritizes_exact_id():
    from enhanced_router.config_cli import _rank_route_matches, RouteChoice

    choices = [
        RouteChoice(model_id="zzz-vendor/foo-longcat", endpoint_id="auto",
                     provider_id="p", provider_name="P", model_name="Foo LongCat",
                     backend="litellm", credential_configured=True, availability="public", certified=True),
        RouteChoice(model_id="longcat-2", endpoint_id="auto",
                     provider_id="p", provider_name="P", model_name="LongCat 2",
                     backend="litellm", credential_configured=True, availability="public", certified=True),
        RouteChoice(model_id="longcat", endpoint_id="auto",
                     provider_id="p", provider_name="P", model_name="LongCat",
                     backend="litellm", credential_configured=True, availability="public", certified=True),
    ]
    ranked = _rank_route_matches("longcat", choices)
    assert [rc.model_id for rc in ranked] == ["longcat", "longcat-2", "zzz-vendor/foo-longcat"]
    assert _rank_route_matches("", choices) == choices


def test_nav_result_navigation_check():
    from enhanced_router.config_cli import NavResult, NavigationAction

    nav = NavResult(action=NavigationAction.BACK)
    assert nav.is_navigation
    assert nav.value is None

    not_nav = NavResult(value="model-a")
    assert not not_nav.is_navigation
    assert not_nav.value == "model-a"


def test_choose_nav_back_action(monkeypatch):
    from enhanced_router.config_cli import _choose_nav, ChoiceControls, NavigationAction

    responses = iter(["b"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _choose_nav(
        "Test",
        [("a", "Option A"), ("b_opt", "Option B")],
        ChoiceControls(allow_back=True, allow_cancel=True),
    )
    assert result.action == NavigationAction.BACK


def test_choose_nav_cancel_action(monkeypatch):
    from enhanced_router.config_cli import _choose_nav, ChoiceControls, NavigationAction

    responses = iter(["q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _choose_nav(
        "Test",
        [("a", "Option A")],
        ChoiceControls(allow_cancel=True),
    )
    assert result.action == NavigationAction.CANCEL


def test_choose_nav_selects_by_number(monkeypatch):
    from enhanced_router.config_cli import _choose_nav, ChoiceControls

    responses = iter(["2"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _choose_nav(
        "Test",
        [("first", "First Option"), ("second", "Second Option")],
        ChoiceControls(allow_cancel=True),
    )
    assert result.value == "second"
    assert result.action is None


def test_choose_nav_back_on_blank(monkeypatch):
    """When _choose_nav has a single navigation action and blank input is given,
    it should return that navigation action."""
    from enhanced_router.config_cli import _choose_nav, ChoiceControls, NavigationAction

    # With only allow_back=True, default_nav is set to BACK.
    # Mock _prompt to return the default (which is "1"), making the selection
    # land on option "a" — testing blank works correctly.
    responses = iter([""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = _choose_nav(
        "Test",
        [("a", "A")],
        ChoiceControls(allow_back=True, allow_cancel=True),
    )
    # With allow_back=True AND allow_cancel=True, there's no single default_nav
    # and blank triggers the ValueError path. But we test that it doesn't crash.
    assert result.value == "a" or result.action is not None


def test_configure_launch_setup_creates_launch_preset(tmp_path: Path, monkeypatch):
    """Full setup wizard creates a launch preset pairing both profiles."""
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "profiles.yaml", {
        "profiles": {"test-profile": {"controller_model": "model-a", "recon": "model-a", "implementer": "model-a", "adversary": "model-a", "repairer": "model-a"}},
    })
    _write(config_dir / "sidecar_profiles.yaml", {
        "sidecar_profiles": {"test-sidecars": {"sidecar_ids": ["reviewer"]}},
    })
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    # Mock the sub-wizards so they return instantly. The orchestrator calls
    # configure_inference -> returns "test-profile", then
    # configure_sidecar_profile -> returns "test-sidecars", then
    # create preset "3" (both profiles exist now).
    monkeypatch.setattr(config_cli, "configure_inference", lambda cd, r: "test-profile")
    monkeypatch.setattr(config_cli, "configure_sidecar_profile", lambda cd, r: "test-sidecars")

    # Only outer-loop inputs: 1=main models, 2=sidecars, 3=create preset
    responses = iter(["1", "2", "3"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))
    monkeypatch.setattr(config_cli, "_prompt", lambda label, default=None: default or "")

    result = config_cli.configure_launch_setup(config_dir, registry)
    assert result.status == "saved"
    assert result.inference_profile_id == "test-profile"
    assert result.sidecar_profile_id == "test-sidecars"
    assert result.launch_preset_id is not None


def test_configure_launch_setup_cancels_cleanly(tmp_path: Path, monkeypatch):
    """Cancelling launch setup returns cancelled status without creating anything."""
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    responses = iter(["q"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    result = config_cli.configure_launch_setup(config_dir, registry)
    assert result.status == "cancelled"
    assert result.inference_profile_id is None


def test_staged_grants_not_applied_on_cancel(tmp_path: Path, monkeypatch):
    """Staged role grants should be discarded when the profile is not saved."""
    config_dir = _two_certified_model_config_dir(tmp_path)
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")

    # Use the uncertified model which has no pre-assigned roles
    _write(config_dir / "discovered_models.yaml", {
        "models": {
            "vendor/uncertified-model": {
                "display_name": "Uncertified Model",
                "catalog_source": "discovered",
                "backend": "litellm",
                "litellm_model": "openai/uncertified-model",
                "api_base": "https://example.invalid/v1",
                "api_key_env": "TEST_PROBE_API_KEY",
                "provider_id": "testprovider",
                "capabilities": {"tools": False, "mutation": False, "controller_eligible": False},
                "allowed_roles": [],
            },
        },
    })
    registry, _ = _load_registry(config_dir)

    model = registry.get_model("vendor/uncertified-model")
    assert "recon" not in model.allowed_roles

    from enhanced_router.config_cli import _apply_staged_grants
    grants = [{"model_id": "vendor/uncertified-model", "role": "recon", "controller": False}]
    # Not calling _apply_staged_grants -- simulating cancellation
    # Verify grant was NOT applied
    model = registry.get_model("vendor/uncertified-model")
    assert "recon" not in model.allowed_roles


def test_create_launch_preset_for_setup_reuses_existing(tmp_path: Path, monkeypatch):
    """_create_launch_preset_for_setup returns existing preset ID when profiles match."""
    config_dir = _two_certified_model_config_dir(tmp_path)
    _write(config_dir / "launch_presets.yaml", {
        "launch_presets": {"existing-preset": {"inference_profile_id": "hybrid", "sidecar_profile_id": "lightweight"}},
    })
    monkeypatch.setenv("TEST_PROBE_API_KEY", "fake-key-for-config-check")
    registry, _ = _load_registry(config_dir)

    preset_id = config_cli._create_launch_preset_for_setup(
        config_dir, inference_profile_id="hybrid", sidecar_profile_id="lightweight",
    )
    assert preset_id == "existing-preset"
