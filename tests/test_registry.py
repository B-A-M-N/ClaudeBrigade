"""Tests for registry.py — ModelRegistry YAML loading and recommendation."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from enhanced_router.registry import ConfigDiagnostic, ModelRegistry


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
    def test_bundled_free_controller_profiles(self):
        repo_config = pathlib.Path(__file__).resolve().parents[1] / "config"
        registry = ModelRegistry(repo_config)
        registry.load_models()
        registry.load_profiles()
        registry.load_workflows()
        registry.load_providers()
        registry.load_fastpath()
        registry.load_sidecars()
        registry._validate_cross_refs()

        expected = {
            "nvidia-nim": "nvidia-nim-free",
            "opencode-zen": "opencode-zen-free",
            "kilocode-free": "kilocode-auto-free",
            "openrouter-free": "openrouter-free",
        }
        for profile_id, model_id in expected.items():
            profile = registry.get_profile(profile_id)
            assert profile.controller_model == model_id
            assert registry.get_model(model_id).capabilities.cost_class == "free"

    def test_specialist_manifest_is_registry_backed(self):
        repo_config = pathlib.Path(__file__).resolve().parents[1] / "config"
        registry = ModelRegistry(repo_config)
        registry.load_models()
        registry.load_profiles()

        aliases = registry.role_model_aliases()
        bindings = registry.role_model_bindings()

        assert aliases["anthropic-brigade-fi-qwen-scout"] == "recon"
        assert bindings["brigade-fi-qwen-scout"] == "qwen3.6-35b"
        assert registry.native_agent_name("glm-5.1", "adversary") == "brigade-fi-glm-adversary"

    def test_specialist_manifest_profile_id_scopes_to_a_single_profile(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "model-a": {
                    "display_name": "Model A",
                    "backend": "direct-anthropic",
                    "upstream_model": "ModelA-v1",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 100000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon"],
                },
                "model-b": {
                    "display_name": "Model B",
                    "backend": "direct-anthropic",
                    "upstream_model": "ModelB-v1",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 100000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon"],
                },
            }
        })
        specialist = {
            "model": "model-a",
            "roles": ["recon"],
            "native_agent_name": "brigade-scout",
            "public_model_alias": "anthropic-brigade-scout",
        }
        conflicting_specialist = dict(specialist, model="model-b")
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "profile-a": {
                    "recon": "model-a", "implementer": "model-a",
                    "adversary": "model-a", "repairer": "model-a",
                    "specialists": {"scout": specialist},
                },
                "profile-b": {
                    "recon": "model-b", "implementer": "model-b",
                    "adversary": "model-b", "repairer": "model-b",
                    "specialists": {"scout": conflicting_specialist},
                },
            }
        })
        registry = ModelRegistry(config_dir)
        registry.load_models()
        registry.load_profiles()

        # Two saved profiles disagree on what "brigade-scout" backs -- an
        # unrelated profile's specialist naming choice must not block a
        # launch that only ever selected the other one.
        with pytest.raises(ValueError, match="conflicting registry definitions"):
            registry.specialist_manifest()

        scoped = registry.specialist_manifest(profile_id="profile-a")
        assert scoped["brigade-scout"]["model_id"] == "model-a"

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

    def test_recommend_prefer_low_cost_ranks_free_model_first(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "pricey-model": {
                    "display_name": "Pricey",
                    "backend": "direct-anthropic",
                    "upstream_model": "P",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {
                        "tools": True, "mutation": True, "context_tokens": 1000,
                        "reasoning": "high", "local": False, "cost_class": "high",
                    },
                    "allowed_roles": ["recon"],
                },
                "free-model": {
                    "display_name": "Free",
                    "backend": "direct-anthropic",
                    "upstream_model": "F",
                    "api_base": "https://api.test.com/anthropic",
                    "capabilities": {
                        "tools": True, "mutation": True, "context_tokens": 1000,
                        "reasoning": "high", "local": False, "cost_class": "free",
                    },
                    "allowed_roles": ["recon"],
                },
            }
        })
        reg = ModelRegistry(config_dir)
        reg.load_models()
        from enhanced_router.config_models import RecommendationConstraints
        default_results = reg.recommend("recon")
        # Without prefer_low_cost, both models tie on every other axis --
        # cost must not silently affect the default ranking.
        default_scores = {r.model_id: r.score for r in default_results}
        assert default_scores["pricey-model"] == default_scores["free-model"]

        cost_results = reg.recommend(
            "recon", constraints=RecommendationConstraints(role="recon", prefer_low_cost=True),
        )
        assert cost_results[0].model_id == "free-model"


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


class TestProfileReadinessPerFallbackEndpoint:
    """profile_readiness must check each fallback candidate against its own
    pinned endpoint, not always 'auto' -- a fallback that pins a specific
    (broken) endpoint must be skipped in favor of the next candidate, not
    reported as ready because some *other* endpoint of that model happens
    to work under 'auto' resolution."""

    def test_fallback_pinned_to_broken_endpoint_is_skipped(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "model-broken-primary": {
                    "display_name": "Broken Primary",
                    "backend": "direct-anthropic",
                    "api_key_env": "MISSING_PRIMARY_KEY_ENV",
                    "api_base_env": "MISSING_PRIMARY_BASE_ENV",
                    "upstream_model": "P",
                    "capabilities": {"tools": True, "mutation": False, "reasoning": "medium", "local": False},
                    "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
                },
                "model-x": {
                    "display_name": "Model X",
                    "backend": "direct-anthropic",
                    "capabilities": {"tools": True, "mutation": False, "reasoning": "medium", "local": False},
                    "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
                    "endpoints": {
                        "good-ep": {
                            "backend": "direct-anthropic",
                            "upstream_model": "X",
                            "api_base": "https://good.example.invalid/v1",
                        },
                        "broken-ep": {
                            "backend": "direct-anthropic",
                            "upstream_model": "X",
                            "api_base_env": "MISSING_BROKEN_EP_ENV",
                        },
                    },
                },
                "model-y": {
                    "display_name": "Model Y",
                    "backend": "direct-anthropic",
                    "upstream_model": "Y",
                    "api_base": "https://good.example.invalid/v1",
                    "capabilities": {"tools": True, "mutation": False, "reasoning": "medium", "local": False},
                    "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
                },
            },
        })
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "test-profile": {
                    "recon": {
                        "model": "model-broken-primary",
                        "endpoint": "auto",
                        "fallback_models": [
                            {"model": "model-x", "endpoint": "broken-ep"},
                            {"model": "model-y", "endpoint": "auto"},
                        ],
                    },
                    "implementer": "model-y",
                    "adversary": "model-y",
                    "repairer": "model-y",
                },
            }
        })
        reg = ModelRegistry(config_dir)
        reg.load_models()
        reg.load_profiles()

        readiness = reg.profile_readiness("test-profile")
        recon = readiness["roles"]["recon"]
        # model-x must be skipped (its pinned 'broken-ep' has no available
        # credential/api_base) in favor of the next fallback, model-y --
        # NOT selected via a mistaken 'auto' resolution of model-x's other,
        # unrelated 'good-ep' endpoint.
        assert recon["selected_candidate"] == "model-y"
        assert recon["ready"] is True


class TestSidecarProfilesAndLaunchPresets:
    def _base_registry(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "model-a": {
                    "display_name": "Model A",
                    "backend": "direct-anthropic",
                    "upstream_model": "A",
                    "api_base": "https://a.example.invalid/v1",
                    "capabilities": {
                        "tools": True, "mutation": True, "write_tool_certified": True,
                        "reasoning": "medium", "local": False,
                    },
                    "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
                },
                "model-b": {
                    "display_name": "Model B",
                    "backend": "direct-anthropic",
                    "upstream_model": "B",
                    "api_base": "https://b.example.invalid/v1",
                    "capabilities": {
                        "tools": True, "mutation": False, "write_tool_certified": False,
                        "reasoning": "medium", "local": False,
                    },
                    "allowed_roles": [],
                },
            },
        })
        _write_config(config_dir, "sidecars.yaml", {
            "sidecars": {
                "reviewer": {"model_id": "model-b", "mode": "structured", "endpoint": "auto", "enabled": True},
                "verifier": {"model_id": "model-b", "mode": "verify", "endpoint": "auto", "enabled": True},
            },
        })
        _write_config(config_dir, "fastpath.yaml", {
            "fastpath": {"enabled": True, "model_id": "model-b"},
        })
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "hybrid": {
                    "recon": "model-a", "implementer": "model-a",
                    "adversary": "model-a", "repairer": "model-a",
                },
            },
        })
        reg = ModelRegistry(config_dir)
        reg.load_models()
        reg.load_sidecars()
        reg.load_fastpath()
        reg.load_profiles()
        return reg

    def test_resolve_sidecars_with_no_profile_returns_all_global_sidecars(self, config_dir):
        reg = self._base_registry(config_dir)
        assert set(reg.resolve_sidecars(None)) == {"reviewer", "verifier"}

    def test_resolve_sidecars_with_profile_bounds_to_listed_ids(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "sidecar_profiles.yaml", {
            "sidecar_profiles": {"lightweight": {"sidecar_ids": ["reviewer"]}},
        })
        reg.load_sidecar_profiles()
        assert set(reg.resolve_sidecars("lightweight")) == {"reviewer"}

    def test_resolve_fastpath_prefers_profile_own_fastpath_over_global(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "sidecar_profiles.yaml", {
            "sidecar_profiles": {
                "custom": {
                    "sidecar_ids": [],
                    "fastpath": {"enabled": True, "model_id": "model-b", "timeout_seconds": 11},
                },
            },
        })
        reg.load_sidecar_profiles()
        resolved = reg.resolve_fastpath("custom")
        assert resolved is not None
        assert resolved.timeout_seconds == 11
        # The global fastpath.yaml singleton is unaffected -- still its own value.
        assert reg.fastpath is not None
        assert reg.fastpath.timeout_seconds != 11

    def test_resolve_fastpath_falls_back_to_global_when_profile_has_none(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "sidecar_profiles.yaml", {
            "sidecar_profiles": {"lightweight": {"sidecar_ids": ["reviewer"]}},
        })
        reg.load_sidecar_profiles()
        resolved = reg.resolve_fastpath("lightweight")
        assert resolved is reg.fastpath

    def test_resolve_fastpath_with_no_profile_returns_global_singleton(self, config_dir):
        reg = self._base_registry(config_dir)
        assert reg.resolve_fastpath(None) is reg.fastpath

    def test_validate_cross_refs_rejects_sidecar_profile_with_unknown_sidecar(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "sidecar_profiles.yaml", {
            "sidecar_profiles": {"broken": {"sidecar_ids": ["nonexistent-sidecar"]}},
        })
        reg.load_sidecar_profiles()
        with pytest.raises(ValueError, match="unknown sidecar"):
            reg._validate_cross_refs()

    def test_validate_cross_refs_rejects_launch_preset_with_unknown_inference_profile(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "launch_presets.yaml", {
            "launch_presets": {"broken": {"inference_profile_id": "nonexistent-profile"}},
        })
        reg.load_launch_presets()
        with pytest.raises(ValueError, match="unknown inference"):
            reg._validate_cross_refs()

    def test_validate_cross_refs_rejects_launch_preset_with_unknown_sidecar_profile(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "launch_presets.yaml", {
            "launch_presets": {
                "broken": {"inference_profile_id": "hybrid", "sidecar_profile_id": "nonexistent-sidecar-profile"},
            },
        })
        reg.load_launch_presets()
        with pytest.raises(ValueError, match="unknown sidecar"):
            reg._validate_cross_refs()

    def test_validate_cross_refs_accepts_valid_sidecar_profile_and_launch_preset(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "sidecar_profiles.yaml", {
            "sidecar_profiles": {"lightweight": {"sidecar_ids": ["reviewer"]}},
        })
        _write_config(config_dir, "launch_presets.yaml", {
            "launch_presets": {
                "default": {"inference_profile_id": "hybrid", "sidecar_profile_id": "lightweight"},
            },
        })
        reg.load_sidecar_profiles()
        reg.load_launch_presets()
        reg._validate_cross_refs()  # must not raise

    def test_profile_model_diversity_warnings_flags_same_model_for_every_role(self, config_dir):
        # _base_registry's "hybrid" profile assigns model-a to all four roles.
        reg = self._base_registry(config_dir)
        warnings = reg.profile_model_diversity_warnings()
        assert len(warnings) == 1
        assert "hybrid" in warnings[0]
        assert "model-a" in warnings[0]

    def test_profile_model_diversity_warnings_is_silent_for_differentiated_roles(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "hybrid": {
                    "recon": "model-a", "implementer": "model-a",
                    "adversary": "model-b", "repairer": "model-a",
                },
            },
        })
        reg.load_profiles()
        assert reg.profile_model_diversity_warnings() == []

    def test_profile_model_diversity_warnings_does_not_block_loading_or_cross_refs(self, config_dir):
        # Same-model-for-every-role is advisory, not fail-closed: it must
        # still load and pass _validate_cross_refs.
        reg = self._base_registry(config_dir)
        reg._validate_cross_refs()  # must not raise


class TestReferencedModelIdsScoping:
    def _registry(self, config_dir):
        _write_config(config_dir, "models.yaml", {
            "models": {
                "model-a": {
                    "display_name": "Model A", "backend": "direct-anthropic",
                    "upstream_model": "A", "api_base": "https://a.example.invalid/v1",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
                },
                "model-b": {
                    "display_name": "Model B", "backend": "direct-anthropic",
                    "upstream_model": "B", "api_base": "https://b.example.invalid/v1",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
                },
                "model-c": {
                    "display_name": "Model C", "backend": "direct-anthropic",
                    "upstream_model": "C", "api_base": "https://c.example.invalid/v1",
                    "capabilities": {"tools": True, "mutation": True, "context_tokens": 1000, "reasoning": "high", "local": False},
                    "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
                },
            },
        })
        _write_config(config_dir, "sidecars.yaml", {
            "sidecars": {
                "reviewer": {"model_id": "model-a", "mode": "structured", "endpoint": "auto", "enabled": True},
                "verifier": {"model_id": "model-c", "mode": "verify", "endpoint": "auto", "enabled": True},
            },
        })
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "profile-1": {
                    "recon": "model-a", "implementer": "model-a",
                    "adversary": "model-a", "repairer": "model-a",
                },
                "profile-2": {
                    "recon": "model-b", "implementer": "model-b",
                    "adversary": "model-b", "repairer": "model-b",
                },
            },
        })
        _write_config(config_dir, "sidecar_profiles.yaml", {
            "sidecar_profiles": {"reviewer-only": {"sidecar_ids": ["reviewer"]}},
        })
        reg = ModelRegistry(config_dir)
        reg.load_models()
        reg.load_profiles()
        reg.load_sidecars()
        reg.load_sidecar_profiles()
        return reg

    def test_no_scope_includes_every_profile_and_sidecar(self, config_dir):
        reg = self._registry(config_dir)
        ids = reg.referenced_model_ids()
        assert {"model-a", "model-b", "model-c"} <= ids

    def test_profile_ids_excludes_unselected_profiles(self, config_dir):
        reg = self._registry(config_dir)
        ids = reg.referenced_model_ids(profile_ids={"profile-1"})
        assert "model-a" in ids
        assert "model-b" not in ids

    def test_sidecar_profile_ids_bounds_to_listed_sidecars(self, config_dir):
        reg = self._registry(config_dir)
        ids = reg.referenced_model_ids(
            profile_ids=set(), sidecar_profile_ids={"reviewer-only"},
        )
        assert "model-a" in ids  # reviewer's model
        assert "model-c" not in ids  # verifier excluded by the sidecar profile

    class _FakeState:
        def __init__(self, selections):
            self._selections = selections

        def active_run_selections(self):
            return self._selections

    def test_for_active_runs_falls_back_to_unscoped_with_no_open_runs(self, config_dir):
        reg = self._registry(config_dir)
        ids = reg.referenced_model_ids_for_active_runs(self._FakeState([]))
        assert {"model-a", "model-b", "model-c"} <= ids

    def test_for_active_runs_scopes_to_the_union_of_open_run_selections(self, config_dir):
        reg = self._registry(config_dir)
        state = self._FakeState([
            {"inference_profile_id": "profile-1", "sidecar_profile_id": "reviewer-only"},
        ])
        ids = reg.referenced_model_ids_for_active_runs(state)
        assert "model-a" in ids
        assert "model-b" not in ids
        assert "model-c" not in ids

    def test_for_active_runs_unions_two_concurrent_runs_on_different_profiles(self, config_dir):
        """Two runs open at once on different profiles must each keep their
        own models covered -- the shared daemon serves both, so scoping to
        only one run's selection would starve the other's LiteLLM config.
        """
        reg = self._registry(config_dir)
        state = self._FakeState([
            {"inference_profile_id": "profile-1", "sidecar_profile_id": "reviewer-only"},
            {"inference_profile_id": "profile-2", "sidecar_profile_id": "reviewer-only"},
        ])
        ids = reg.referenced_model_ids_for_active_runs(state)
        assert "model-a" in ids  # profile-1's model
        assert "model-b" in ids  # profile-2's model
        assert "model-c" not in ids  # verifier still excluded -- neither run's sidecar profile includes it

    def test_for_active_runs_falls_back_when_any_run_is_unscoped(self, config_dir):
        reg = self._registry(config_dir)
        state = self._FakeState([
            {"inference_profile_id": "profile-1", "sidecar_profile_id": "reviewer-only"},
            {"inference_profile_id": None, "sidecar_profile_id": None},
        ])
        ids = reg.referenced_model_ids_for_active_runs(state)
        # A bare --model launch has no bounded selection to scope to, so the
        # whole scope must widen back out rather than silently exclude
        # models the unscoped run might need.
        assert {"model-a", "model-b", "model-c"} <= ids


class TestCollectConfigDiagnostics:
    """collect_config_diagnostics() is the wizard-facing, tolerant sibling
    of _validate_cross_refs(): it must never raise, and must report every
    independently-broken section rather than stopping at the first one --
    that's what lets the config wizard start (and let an operator fix
    things) even when the saved config already has a problem somewhere."""

    def _base_registry(self, config_dir):
        return TestSidecarProfilesAndLaunchPresets()._base_registry(config_dir)

    def test_returns_empty_list_for_valid_config(self, config_dir):
        reg = self._base_registry(config_dir)
        assert reg.collect_config_diagnostics() == []

    def test_never_raises_on_a_broken_profile_reference(self, config_dir):
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "hybrid": {
                    "recon": "nonexistent-model", "implementer": "model-a",
                    "adversary": "model-a", "repairer": "model-a",
                },
            },
        })
        reg.load_profiles()
        diagnostics = reg.collect_config_diagnostics()  # must not raise
        assert any(d.section == "profiles" for d in diagnostics)

    def test_reports_one_diagnostic_per_independently_broken_section(self, config_dir):
        """A broken profile AND a broken sidecar profile at the same time
        must both be reported -- one section's failure must not hide
        another, unlike _validate_cross_refs which stops at the first."""
        reg = self._base_registry(config_dir)
        _write_config(config_dir, "profiles.yaml", {
            "profiles": {
                "hybrid": {
                    "recon": "nonexistent-model", "implementer": "model-a",
                    "adversary": "model-a", "repairer": "model-a",
                },
            },
        })
        _write_config(config_dir, "sidecar_profiles.yaml", {
            "sidecar_profiles": {"broken": {"sidecar_ids": ["nonexistent-sidecar"]}},
        })
        reg.load_profiles()
        reg.load_sidecar_profiles()

        diagnostics = reg.collect_config_diagnostics()
        sections = {d.section for d in diagnostics}
        assert "profiles" in sections
        assert "sidecar_profiles" in sections
        assert all(isinstance(d, ConfigDiagnostic) for d in diagnostics)

        # _validate_cross_refs, by contrast, still fails closed on the
        # first violation it finds -- runtime startup must keep this
        # behavior unchanged.
        with pytest.raises(ValueError):
            reg._validate_cross_refs()
