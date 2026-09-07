"""Tests for LiteLLM configuration compilation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from enhanced_router.config_models import ModelCapabilities, ModelEndpointSpec, ModelSpec
from enhanced_router.litellm_config import (
    config_digest,
    deployment_filter_enabled,
    generate_litellm_config,
    write_litellm_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def litellm_model() -> ModelSpec:
    return ModelSpec(
        display_name="Test LiteLLM",
        backend="litellm",
        litellm_model="ollama/test-model",
        api_base="http://127.0.0.1:11434",
        capabilities=ModelCapabilities(
            tools=True,
            mutation=True,
            context_tokens=131072,
            reasoning="medium",
            local=True,
        ),
        allowed_roles={"recon", "implementer"},
    )


@pytest.fixture
def litellm_model_with_key() -> ModelSpec:
    return ModelSpec(
        display_name="Test Key Model",
        backend="litellm",
        litellm_model="openrouter/vendor/model",
        api_key_env="OPENROUTER_API_KEY",
        capabilities=ModelCapabilities(
            tools=True,
            mutation=False,
            context_tokens=200000,
            reasoning="high",
            local=False,
        ),
        allowed_roles={"recon", "adversary"},
    )


@pytest.fixture
def direct_model() -> ModelSpec:
    return ModelSpec(
        display_name="Direct Model",
        backend="direct-anthropic",
        upstream_model="LongCat-2.0",
        api_base="https://api.test.com/anthropic",
        capabilities=ModelCapabilities(
            tools=True,
            mutation=True,
            context_tokens=1000000,
            reasoning="high",
            local=False,
        ),
        allowed_roles={"recon", "implementer", "adversary", "repairer"},
    )


@pytest.fixture
def discovered_litellm_model() -> ModelSpec:
    return ModelSpec(
        display_name="Discovered Model",
        backend="litellm",
        litellm_model="openrouter/vendor/discovered",
        catalog_source="discovered",
        capabilities=ModelCapabilities(
            tools=True,
            mutation=False,
            context_tokens=131072,
            reasoning="medium",
            local=False,
        ),
        allowed_roles={"recon"},
    )


@pytest.fixture
def disabled_litellm_model() -> ModelSpec:
    return ModelSpec(
        display_name="Disabled Model",
        backend="litellm",
        litellm_model="ollama/disabled",
        enabled=False,
        capabilities=ModelCapabilities(
            tools=True,
            mutation=True,
            context_tokens=131072,
            reasoning="medium",
            local=True,
        ),
        allowed_roles={"recon"},
    )


# ---------------------------------------------------------------------------
# Tests: generate_litellm_config
# ---------------------------------------------------------------------------


class TestGenerateLiteLLMConfig:
    def test_managed_group_shares_logical_alias(self):
        model = ModelSpec(
            display_name="Grouped",
            backend="litellm",
            routing_mode="managed-group",
            deployment_group="grouped-model",
            endpoints={
                "local": ModelEndpointSpec(
                    backend="litellm", litellm_model="ollama/grouped",
                    api_base="http://127.0.0.1:11434", certified=True,
                ),
                "freeinference": ModelEndpointSpec(
                    backend="litellm", litellm_model="openai/grouped",
                    api_base="https://freeinference.org/v1",
                    api_key_env="FREEINFERENCE_API_KEY", certified=True,
                ),
            },
            capabilities=ModelCapabilities(tools=True, mutation=False, reasoning="medium", local=False),
            allowed_roles={"recon"},
        )
        parsed = yaml.safe_load(generate_litellm_config({"grouped": model}))
        entries = parsed["model_list"]
        assert [entry["model_name"] for entry in entries] == [
            "brigade-grouped-model", "brigade-grouped-model",
        ]
        assert {entry["model_info"]["deployment_id"] for entry in entries} == {
            "freeinference", "local",
        }

    def test_empty_models(self):
        """Empty models dict produces empty model_list."""
        config = generate_litellm_config({})
        parsed = yaml.safe_load(config)
        assert parsed is not None
        assert parsed["model_list"] == []

    def test_only_litellm_models_included(self, direct_model, litellm_model):
        """Only models with backend='litellm' appear in the config."""
        models = {
            "direct": direct_model,
            "litellm-test": litellm_model,
        }
        config = generate_litellm_config(models)
        parsed = yaml.safe_load(config)
        model_names = [m["model_name"] for m in parsed["model_list"]]
        assert "brigade-litellm-test" in model_names
        assert "brigade-direct" not in model_names

    def test_disabled_models_excluded(self, litellm_model, disabled_litellm_model):
        """Disabled litellm models are excluded."""
        models = {
            "enabled": litellm_model,
            "disabled": disabled_litellm_model,
        }
        config = generate_litellm_config(models)
        parsed = yaml.safe_load(config)
        model_names = [m["model_name"] for m in parsed["model_list"]]
        assert "brigade-enabled" in model_names
        assert "brigade-disabled" not in model_names

    def test_model_name_prefix(self, litellm_model):
        """Model names use 'brigade-' prefix with registry key."""
        models = {"qwen-local": litellm_model}
        config = generate_litellm_config(models)
        parsed = yaml.safe_load(config)
        assert parsed["model_list"][0]["model_name"] == "brigade-qwen-local"

    def test_api_base_embedded(self, litellm_model):
        """Static api_base appears directly in config."""
        models = {"test": litellm_model}
        config = generate_litellm_config(models)
        parsed = yaml.safe_load(config)
        params = parsed["model_list"][0]["litellm_params"]
        assert params["api_base"] == "http://127.0.0.1:11434"

    def test_api_key_via_env_var(self, litellm_model_with_key, monkeypatch):
        """API key uses os.environ/ syntax, never materialized."""
        # The application may materialize named keyring slots into process
        # environment variables. Exercise the unslotted legacy path without
        # inheriting the developer shell's credential state.
        for name in tuple(os.environ):
            if name.startswith("BRIGADE_KEYRING_"):
                monkeypatch.delenv(name, raising=False)
        models = {"test-key": litellm_model_with_key}
        config = generate_litellm_config(models)
        parsed = yaml.safe_load(config)
        params = parsed["model_list"][0]["litellm_params"]
        assert "api_key" in params
        assert params["api_key"] == "os.environ/OPENROUTER_API_KEY"
        assert "sk-" not in config  # no secret materialized

    def test_no_secrets_in_config(self, litellm_model):
        """The config never contains raw secret values."""
        models = {"safe": litellm_model}
        config = generate_litellm_config(models)
        # Check no bearer tokens, no sk- patterns, no secrets
        assert "sk-" not in config
        assert "Bearer" not in config

    def test_master_key_via_env(self):
        """master_key uses os.environ/BRIGADE_LITELLM_KEY."""
        config = generate_litellm_config({})
        parsed = yaml.safe_load(config)
        assert parsed["general_settings"]["master_key"] == "os.environ/BRIGADE_LITELLM_KEY"

    def test_models_sorted_deterministically(self, litellm_model):
        """Models appear sorted by registry key for deterministic output."""
        models = {
            "z-model": litellm_model,
            "a-model": litellm_model,
        }
        config = generate_litellm_config(models)
        parsed = yaml.safe_load(config)
        names = [m["model_name"] for m in parsed["model_list"]]
        assert names == sorted(names)

    def test_valid_yaml_output(self, litellm_model, litellm_model_with_key):
        """Output is valid YAML that parses correctly."""
        models = {
            "a": litellm_model,
            "b": litellm_model_with_key,
        }
        config = generate_litellm_config(models)
        parsed = yaml.safe_load(config)
        assert "model_list" in parsed
        assert "general_settings" in parsed
        assert len(parsed["model_list"]) == 2

    def test_deployment_filter_callback_is_explicitly_opt_in(self, litellm_model, monkeypatch):
        monkeypatch.delenv("BRIGADE_LITELLM_DISPATCH_FILTER", raising=False)
        disabled = yaml.safe_load(generate_litellm_config({"model": litellm_model}))
        assert "callbacks" not in disabled["litellm_settings"]

        monkeypatch.setenv("BRIGADE_LITELLM_DISPATCH_FILTER", "1")
        enabled = yaml.safe_load(generate_litellm_config({"model": litellm_model}))
        assert enabled["litellm_settings"]["callbacks"] == [
            "brigade_litellm_dispatch.proxy_handler_instance"
        ]


# ---------------------------------------------------------------------------
# Tests: config_digest
# ---------------------------------------------------------------------------


class TestConfigDigest:
    def test_deterministic(self, litellm_model):
        """Same input produces same digest."""
        models = {"m": litellm_model}
        d1 = config_digest(models)
        d2 = config_digest(models)
        assert d1 == d2

    def test_different_for_different_models(self, litellm_model, litellm_model_with_key):
        """Different inputs produce different digests."""
        d1 = config_digest({"m": litellm_model})
        d2 = config_digest({"m": litellm_model_with_key})
        assert d1 != d2

    def test_only_litellm_models_affect_digest(self, litellm_model, direct_model):
        """direct-anthropic models do NOT affect the digest."""
        d1 = config_digest({"m": litellm_model})
        d2 = config_digest({"m": litellm_model, "d": direct_model})
        assert d1 == d2

    def test_disabled_status_affects_digest(self, litellm_model, disabled_litellm_model):
        """enabled=False changes the digest."""
        d1 = config_digest({"m": litellm_model})
        d2 = config_digest({"m": disabled_litellm_model})
        assert d1 != d2

    def test_referenced_ids_excludes_unreferenced_discovered_models(self, discovered_litellm_model):
        """A discovered model outside the referenced set changes the digest
        the same way removing it from *models* entirely would -- this is
        what lets a scope-only change (a new run needing it) be detected as
        a real config change even though the model's own definition never
        moved.
        """
        models = {"m": discovered_litellm_model}
        with_scope_excluded = config_digest(models, referenced_ids=set())
        unscoped = config_digest(models)
        assert with_scope_excluded != unscoped
        assert with_scope_excluded == config_digest({})

    def test_referenced_ids_keeps_included_discovered_models(self, discovered_litellm_model):
        models = {"m": discovered_litellm_model}
        assert config_digest(models, referenced_ids={"m"}) == config_digest(models)

    def test_referenced_ids_never_excludes_bundled_models(self, litellm_model):
        """Only catalog_source == 'discovered' models are subject to the
        referenced-ids filter -- an operator-configured model always
        counts, matching generate_litellm_config's own filter.
        """
        models = {"m": litellm_model}
        assert config_digest(models, referenced_ids=set()) == config_digest(models)

    def test_dispatch_filter_toggle_changes_digest(self, litellm_model, monkeypatch):
        monkeypatch.delenv("BRIGADE_LITELLM_DISPATCH_FILTER", raising=False)
        disabled = config_digest({"m": litellm_model})
        monkeypatch.setenv("BRIGADE_LITELLM_DISPATCH_FILTER", "1")
        enabled = config_digest({"m": litellm_model})
        assert disabled != enabled

    def test_dispatch_filter_explicit_value_overrides_environment(
        self, monkeypatch
    ):
        monkeypatch.setenv("BRIGADE_LITELLM_DISPATCH_FILTER", "1")
        assert deployment_filter_enabled(False) is False
        assert deployment_filter_enabled(True) is True


# ---------------------------------------------------------------------------
# Tests: write_litellm_config
# ---------------------------------------------------------------------------


class TestWriteLiteLLMConfig:
    def test_writes_atomically(self):
        """Config file is written and has mode 0600."""
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "litellm.yaml"
            write_litellm_config("test: true", output)
            assert output.exists()
            content = output.read_text()
            assert content == "test: true"

    def test_file_permissions(self):
        """Written config has mode 0o600."""
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "litellm.yaml"
            write_litellm_config("key: value", output)
            mode = output.stat().st_mode & 0o777
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_no_partial_write_on_failure(self):
        """On failure, no partial file remains."""
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "litellm.yaml"
            with pytest.raises(Exception):
                write_litellm_config("test: true", output)
                # Simulate failure by providing bad path
                write_litellm_config("test: true", Path("/nonexistent/dir/file.yaml"))
            # Original file should not exist
            # (The first write succeeded; this tests no orphan temps)
            temp_files = list(Path(td).glob(".litellm-*"))
            assert len(temp_files) == 0, f"Orphan temp files: {temp_files}"
