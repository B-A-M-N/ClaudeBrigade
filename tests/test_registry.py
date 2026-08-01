"""Tests for registry.py — ModelRegistry YAML loading and recommendation."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from enhanced_router.registry import ModelRegistry


@pytest.fixture
def config_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary config directory with minimal valid YAML files."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    return cfg


def _write_config(cfg: pathlib.Path, filename: str, data: dict) -> None:
    (cfg / filename).write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def valid_registry(config_dir: pathlib.Path) -> ModelRegistry:
    """Return a ModelRegistry preloaded with valid test config."""
    _write_config(config_dir, "models.yaml", {
        "models": {
            "model-a": {
                "display_name": "Model A",
                "backend": "direct-anthropic",
                "upstream_model": "ModelA-v1",
                    "api_base": "https://api.test.com/anthropic",
                "capabilities": {"tools": True, "mutation": True, "context_tokens": 100000, "reasoning": "high", "local": False},
                "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
            },
            "model-b": {
                "display_name": "Model B",
                "backend": "direct-anthropic",
                "upstream_model": "ModelB-v1",
                    "api_base": "https://api.test.com/anthropic",
                "capabilities": {"tools": True, "mutation": False, "context_tokens": 50000, "reasoning": "medium", "local": True},
                "allowed_roles": ["recon", "adversary"],
            },
            "model-c": {
                "display_name": "Model C",
                "backend": "direct-anthropic",
                "upstream_model": "ModelC-v1",
                    "api_base": "https://api.test.com/anthropic",
                "capabilities": {"tools": False, "mutation": False, "context_tokens": 10000, "reasoning": "low", "local": False},
                "allowed_roles": ["recon"],
            },
        }
    })
    _write_config(config_dir, "profiles.yaml", {
        "profiles": {
            "default": {
                "recon": "model-a",
                "implementer": "model-a",
                "adversary": "model-b",
                "repairer": "model-a",
            },
        }
    })
    _write_config(config_dir, "workflows.yaml", {
        "workflows": {
            "normal": {"default_profile": "default"},
        }
    })
    reg = ModelRegistry(config_dir)
    reg.load_models()
    reg.load_profiles()
    reg.load_workflows()
    return reg


class TestModelRegistryLoad:
    def test_load_models(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "m1": {
                    "display_name": "M1",
                    "backend": "direct-anthropic",
                    "upstream_model": "M1Up",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon"],
                },
            }
        })
        reg = ModelRegistry(config_dir)
        models = reg.load_models()
        assert "m1" in models
        assert models["m1"].display_name == "M1"

    def test_load_profiles(self, config_dir, valid_registry):
        profiles = valid_registry.profiles
        assert "default" in profiles
        assert profiles["default"].implementer == "model-a"

    def test_load_workflows(self, config_dir, valid_registry):
        workflows = valid_registry.workflows
        assert "normal" in workflows
        assert workflows["normal"].default_profile == "default"

    def test_missing_models_file(self, tmp_path):
        reg = ModelRegistry(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            reg.load_models()

    def test_reload(self, config_dir, valid_registry):
        valid_registry.reload()
        assert len(valid_registry.models) == 3

    def test_registry_hash(self, config_dir, valid_registry):
        h = valid_registry.registry_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
        # Same config = same hash
        h2 = valid_registry.registry_hash()
        assert h == h2


class TestModelRegistryLookup:
    def test_get_model(self, valid_registry):
        spec = valid_registry.get_model("model-a")
        assert spec.display_name == "Model A"
        assert spec.backend == "direct-anthropic"

    def test_get_model_unknown(self, valid_registry):
        with pytest.raises(KeyError, match="nonexistent"):
            valid_registry.get_model("nonexistent")

    def test_get_profile(self, valid_registry):
        profile = valid_registry.get_profile("default")
        assert profile.recon == "model-a"

    def test_get_profile_unknown(self, valid_registry):
        with pytest.raises(KeyError, match="unknown"):
            valid_registry.get_profile("unknown")

    def test_models_for_role_filters_by_allowed_roles(self, valid_registry):
        implementer_models = valid_registry.models_for_role("implementer")
        model_ids = {mid for mid, _ in implementer_models}
        assert "model-a" in model_ids
        assert "model-b" not in model_ids  # model-b only allows recon, adversary
        assert "model-c" not in model_ids  # model-c only allows recon

    def test_models_for_role_excludes_disabled(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "enabled-model": {
                    "display_name": "Enabled",
                    "backend": "direct-anthropic",
                    "upstream_model": "E",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon"],
                    "enabled": True,
                },
                "disabled-model": {
                    "display_name": "Disabled",
                    "backend": "direct-anthropic",
                    "upstream_model": "D",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon"],
                    "enabled": False,
                },
            }
        })
        reg = ModelRegistry(config_dir)
        reg.load_models()
        models = reg.models_for_role("recon")
        model_ids = {mid for mid, _ in models}
        assert "enabled-model" in model_ids
        assert "disabled-model" not in model_ids


class TestModelRegistryRecommend:
    def test_recommend_returns_ranked(self, valid_registry):
        results = valid_registry.recommend("recon")
        assert len(results) >= 1
        assert results[0].score > 0
        assert isinstance(results[0].reason, str)

    def test_recommend_prefers_profile_default(self, valid_registry):
        results = valid_registry.recommend("recon", profile_id="default")
        assert results[0].model_id == "model-a"  # profile default

    def test_recommend_excludes_no_tools_when_required(self, valid_registry):
        results = valid_registry.recommend("recon", constraints=None)
        # model-c has tools=False but requires_tools defaults to True
        result_ids = [r.model_id for r in results]
        assert "model-c" not in result_ids

    def test_recommend_deterministic(self, valid_registry):
        r1 = valid_registry.recommend("recon", profile_id="default")
        r2 = valid_registry.recommend("recon", profile_id="default")
        assert [r.model_id for r in r1] == [r.model_id for r in r2]

    def test_recommend_respects_local_only(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "remote-model": {
                    "display_name": "Remote",
                    "backend": "direct-anthropic",
                    "upstream_model": "R",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon"],
                },
                "local-model": {
                    "display_name": "Local",
                    "backend": "direct-anthropic",
                    "upstream_model": "L",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": True},
                    "allowed_roles": ["recon"],
                },
            }
        })
        reg = ModelRegistry(config_dir)
        reg.load_models()
        from enhanced_router.config_models import RecommendationConstraints
        results = reg.recommend("recon", constraints=RecommendationConstraints(role="recon", local_only=True))
        assert results[0].model_id == "local-model"


class TestProfileCrossReference:
    def test_unknown_model_in_profile_rejected(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "valid-model": {
                    "display_name": "Valid",
                    "backend": "direct-anthropic",
                    "upstream_model": "V",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon"],
                },
            }
        })
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "bad": {
                    "recon": "nonexistent-model",
                    "implementer": "valid-model",
                    "adversary": "valid-model",
                    "repairer": "valid-model",
                },
            }
        })
        reg = ModelRegistry(config_dir)
        reg.load_models()
        reg.load_profiles()
        with pytest.raises(ValueError, match="unknown model"):
            reg._validate_cross_refs()