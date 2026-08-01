"""Regression tests for the interactive configuration bootstrap."""

from __future__ import annotations

import builtins
from pathlib import Path

import yaml

from enhanced_router.config_cli import (
    _confirm_and_grant,
    _load_registry,
    _model_choices,
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
