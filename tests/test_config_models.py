"""Tests for config_models.py Pydantic models."""

import pytest
from pydantic import ValidationError

from enhanced_router.config_models import (
    ModelCapabilities,
    ModelSpec,
    ProfileSpec,
    RecommendationConstraints,
    RankedModel,
)


class TestModelCapabilities:
    def test_valid_capabilities(self):
        cap = ModelCapabilities(
            tools=True,
            mutation=True,
            context_tokens=1000000,
            reasoning="high",
            local=False,
        )
        assert cap.tools is True
        assert cap.mutation is True
        assert cap.context_tokens == 1000000
        assert cap.reasoning == "high"
        assert cap.local is False

    @pytest.mark.parametrize(
        "reasoning", ["low", "medium", "high"]
    )
    def test_reasoning_literal(self, reasoning):
        cap = ModelCapabilities(
            tools=False,
            mutation=False,
            context_tokens=4096,
            reasoning=reasoning,
            local=False,
        )
        assert cap.reasoning == reasoning

    def test_invalid_reasoning_rejected(self):
        with pytest.raises(ValidationError, match="reasoning"):
            ModelCapabilities(
                tools=False,
                mutation=False,
                context_tokens=4096,
                reasoning="extreme",
                local=False,
            )


class TestModelSpec:
    def test_valid_direct_anthropic(self):
        spec = ModelSpec(
            display_name="Test Model",
            backend="direct-anthropic",
            upstream_model="TestUpstream",
            capabilities=ModelCapabilities(
                tools=True,
                mutation=False,
                context_tokens=8192,
                reasoning="low",
                local=True,
            ),
            allowed_roles={"recon"},
        )
        assert spec.display_name == "Test Model"
        assert spec.backend == "direct-anthropic"
        assert spec.upstream_model == "TestUpstream"
        assert spec.enabled is True

    def test_valid_litellm(self):
        spec = ModelSpec(
            display_name="LiteLLM Model",
            backend="litellm",
            litellm_model="ollama/qwen",
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
        assert spec.litellm_model == "ollama/qwen"
        assert "recon" in spec.allowed_roles
        assert "implementer" in spec.allowed_roles

    def test_litellm_without_litellm_model_rejected(self):
        with pytest.raises(ValidationError, match="litellm_model"):
            ModelSpec(
                display_name="Bad",
                backend="litellm",
                capabilities=ModelCapabilities(
                    tools=False,
                    mutation=False,
                    context_tokens=1,
                    reasoning="low",
                    local=False,
                ),
            )

    def test_direct_anthropic_without_upstream_rejected(self):
        with pytest.raises(ValidationError, match="upstream_model"):
            ModelSpec(
                display_name="Bad",
                backend="direct-anthropic",
                capabilities=ModelCapabilities(
                    tools=False,
                    mutation=False,
                    context_tokens=1,
                    reasoning="low",
                    local=False,
                ),
            )

    def test_missing_api_key_env_not_rejected_at_construction(self, monkeypatch):
        """API key env var checks happen at load time, not ModelSpec construction."""
        monkeypatch.delenv("NONEXISTENT_KEY_XYZ", raising=False)
        spec = ModelSpec(
            display_name="Bad",
            backend="direct-anthropic",
            upstream_model="Test",
            api_key_env="NONEXISTENT_KEY_XYZ",
            capabilities=ModelCapabilities(
                tools=False,
                mutation=False,
                context_tokens=1,
                reasoning="low",
                local=False,
            ),
        )
        assert spec.api_key_env == "NONEXISTENT_KEY_XYZ"
        assert spec.enabled is True

    def test_allowed_roles_as_set(self):
        spec = ModelSpec(
            display_name="Set roles",
            backend="direct-anthropic",
            upstream_model="Test",
            capabilities=ModelCapabilities(
                tools=False,
                mutation=False,
                context_tokens=1,
                reasoning="low",
                local=False,
            ),
            allowed_roles={"recon", "implementer", "adversary", "repairer"},
        )
        assert spec.allowed_roles == {"recon", "implementer", "adversary", "repairer"}


class TestProfileSpec:
    def test_valid_profile(self):
        profile = ProfileSpec(
            recon="longcat-2",
            implementer="longcat-2",
            adversary="longcat-2",
            repairer="longcat-2",
        )
        assert profile.recon == "longcat-2"
        assert profile.implementer == "longcat-2"

    def test_different_roles(self):
        profile = ProfileSpec(
            recon="qwen-local",
            implementer="longcat-2",
            adversary="glm-review",
            repairer="longcat-2",
        )
        assert profile.recon == "qwen-local"
        assert profile.adversary == "glm-review"

    def test_role_name_used_as_model_id_rejected(self):
        """Profile values must not be role names themselves."""
        with pytest.raises(ValidationError, match="role name"):
            ProfileSpec(
                recon="recon",  # role name used as model ID
                implementer="longcat-2",
                adversary="longcat-2",
                repairer="longcat-2",
            )

    def test_multiple_roles_can_share_same_model(self):
        """Same model ID for multiple roles is valid and expected."""
        profile = ProfileSpec(
            recon="longcat-2",
            implementer="longcat-2",
            adversary="longcat-2",
            repairer="longcat-2",
        )
        assert profile.recon == "longcat-2"
        assert profile.implementer == "longcat-2"

    def test_no_controller_field(self):
        """ProfileSpec must not have a controller field."""
        profile = ProfileSpec(
            recon="a",
            implementer="b",
            adversary="c",
            repairer="d",
        )
        assert not hasattr(profile, "controller")


class TestRecommendationConstraints:
    def test_defaults(self):
        c = RecommendationConstraints(role="recon")
        assert c.role == "recon"
        assert c.required_context_tokens is None
        assert c.local_only is False
        assert c.requires_tools is True

    def test_with_constraints(self):
        c = RecommendationConstraints(
            role="implementer",
            required_context_tokens=100000,
            local_only=True,
            requires_tools=True,
        )
        assert c.required_context_tokens == 100000
        assert c.local_only is True


class TestRankedModel:
    def test_basic(self):
        r = RankedModel(model_id="longcat-2", score=25, reason="role-match,tool-support")
        assert r.model_id == "longcat-2"
        assert r.score == 25
        assert r.reason == "role-match,tool-support"
