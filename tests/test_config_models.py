"""Tests for config_models.py Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from enhanced_router.config_models import (
    FastpathConfigSpec,
    LaunchPresetSpec,
    ModelCapabilities,
    ModelSpec,
    ProfileSpec,
    RecommendationConstraints,
    RankedModel,
    SidecarProfileSpec,
    TierPolicy,
    WorkflowPhase,
    WorkflowSpec,
    determine_tier,
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
    def test_managed_group_requires_litellm_endpoints(self):
        with pytest.raises(ValidationError, match="managed-group"):
            ModelSpec(
                display_name="Bad group",
                backend="litellm",
                routing_mode="managed-group",
                litellm_model="openai/model",
                capabilities=ModelCapabilities(tools=True, mutation=False, reasoning="low", local=False),
            )

    def test_valid_direct_anthropic(self):
        spec = ModelSpec(
            display_name="Test Model",
            api_base="https://api.test.com/anthropic",
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
            api_base="https://api.test.com/anthropic",
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
            api_base="https://api.test.com/anthropic",
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

    def test_controller_route_defaults_to_none(self):
        """With neither controller_model nor controller set, controller_route() is None."""
        profile = ProfileSpec(
            recon="a",
            implementer="b",
            adversary="c",
            repairer="d",
        )
        assert profile.controller is None
        assert profile.controller_route() is None

    def test_controller_route_prefers_new_field_over_legacy_string(self):
        profile = ProfileSpec(
            recon="a",
            implementer="b",
            adversary="c",
            repairer="d",
            controller_model="legacy-model",
            controller={"model": "new-model", "endpoint": "provider-x"},
        )
        route = profile.controller_route()
        assert route is not None
        assert route.model == "new-model"
        assert route.endpoint == "provider-x"

    def test_controller_route_synthesizes_from_legacy_controller_model(self):
        profile = ProfileSpec(
            recon="a",
            implementer="b",
            adversary="c",
            repairer="d",
            controller_model="legacy-model",
        )
        route = profile.controller_route()
        assert route is not None
        assert route.model == "legacy-model"
        assert route.endpoint == "auto"


class TestSidecarProfileSpec:
    def test_defaults_have_no_sidecars_and_no_dedicated_fastpath(self):
        spec = SidecarProfileSpec()
        assert spec.sidecar_ids == []
        assert spec.fastpath is None

    def test_can_bound_sidecars_and_carry_its_own_fastpath(self):
        spec = SidecarProfileSpec(
            sidecar_ids=["reviewer", "verifier"],
            fastpath={"enabled": True, "model_id": "diffusiongemma"},
        )
        assert spec.sidecar_ids == ["reviewer", "verifier"]
        assert isinstance(spec.fastpath, FastpathConfigSpec)
        assert spec.fastpath.model_id == "diffusiongemma"


class TestLaunchPresetSpec:
    def test_requires_inference_profile_id(self):
        with pytest.raises(ValidationError):
            LaunchPresetSpec()

    def test_sidecar_profile_id_is_optional(self):
        spec = LaunchPresetSpec(inference_profile_id="hybrid")
        assert spec.sidecar_profile_id is None

    def test_pairs_both_ids(self):
        spec = LaunchPresetSpec(inference_profile_id="hybrid", sidecar_profile_id="lightweight")
        assert spec.inference_profile_id == "hybrid"
        assert spec.sidecar_profile_id == "lightweight"


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


# ------------------------------------------------------------------
# WorkflowPhase tests
# ------------------------------------------------------------------


class TestWorkflowPhase:
    def test_defaults(self):
        """WorkflowPhase should apply sensible defaults."""
        phase = WorkflowPhase(id="impl")
        assert phase.id == "impl"
        assert phase.roles == []
        assert phase.required is True
        assert phase.mutation is False
        assert phase.depends_on == []
        assert phase.conditional is None
        assert phase.actor is None

    def test_full_specification(self):
        """WorkflowPhase accepts all fields."""
        phase = WorkflowPhase(
            id="audit",
            roles=["recon", "adversary"],
            required=False,
            mutation=False,
            depends_on=["setup"],
            conditional="found_issue",
            actor="reviewer",
        )
        assert phase.id == "audit"
        assert phase.roles == ["recon", "adversary"]
        assert phase.required is False
        assert phase.mutation is False
        assert phase.depends_on == ["setup"]
        assert phase.conditional == "found_issue"
        assert phase.actor == "reviewer"


# ------------------------------------------------------------------
# WorkflowSpec tests
# ------------------------------------------------------------------


class TestWorkflowSpec:
    def test_with_phases(self):
        """WorkflowSpec with a list of phases parses correctly."""
        spec = WorkflowSpec(
            default_profile="hybrid",
            phases=[
                WorkflowPhase(id="setup", roles=["recon"], required=True),
                WorkflowPhase(
                    id="implement",
                    roles=["implementer"],
                    depends_on=["setup"],
                    mutation=True,
                ),
            ],
        )
        assert spec.default_profile == "hybrid"
        assert len(spec.phases) == 2
        assert spec.phases[0].id == "setup"
        assert spec.phases[1].id == "implement"
        assert spec.phases[1].depends_on == ["setup"]

    def test_empty_phases(self):
        """WorkflowSpec with no phases defaults to an empty list."""
        spec = WorkflowSpec(default_profile="longcat")
        assert spec.default_profile == "longcat"
        assert spec.phases == []

    def test_cross_cutting_phase_ordering(self):
        """cross-cutting workflow must have phases in correct dependency order."""
        spec = WorkflowSpec(
            default_profile="hybrid",
            phases=[
                WorkflowPhase(id="recon", roles=["recon"], required=True),
                WorkflowPhase(
                    id="implementation",
                    roles=["implementer"],
                    depends_on=["recon"],
                ),
                WorkflowPhase(
                    id="adversarial-review",
                    roles=["adversary"],
                    depends_on=["implementation"],
                ),
                WorkflowPhase(
                    id="repair",
                    roles=["repairer"],
                    depends_on=["adversarial-review"],
                    conditional="accepted_findings",
                ),
                WorkflowPhase(
                    id="verification",
                    actor="controller",
                    depends_on=["repair"],
                ),
            ],
        )
        ids = [p.id for p in spec.phases]
        assert ids == ["recon", "implementation", "adversarial-review", "repair", "verification"]
        # Verify dependencies point to earlier phases
        dep_map = {p.id: p.depends_on for p in spec.phases}
        assert dep_map["implementation"] == ["recon"]
        assert dep_map["adversarial-review"] == ["implementation"]
        assert dep_map["repair"] == ["adversarial-review"]
        assert dep_map["verification"] == ["repair"]

    def test_trivial_workflow_has_no_phases(self):
        """trivial workflow should have an empty phases list."""
        spec = WorkflowSpec(default_profile="longcat", phases=[])
        assert spec.default_profile == "longcat"
        assert spec.phases == []


# ------------------------------------------------------------------
# TierPolicy and determine_tier tests
# ------------------------------------------------------------------


class TestTierPolicy:
    def test_defaults(self):
        policy = TierPolicy()
        assert policy.min_tier == "normal"
        assert policy.escalation_allowed is True

    def test_custom_values(self):
        policy = TierPolicy(min_tier="high-risk", escalation_allowed=False)
        assert policy.min_tier == "high-risk"
        assert policy.escalation_allowed is False


class TestDetermineTier:
    def test_trivial_signal(self):
        assert determine_tier(["typo"]) == "trivial"
        assert determine_tier(["comment"]) == "trivial"
        assert determine_tier(["format"]) == "trivial"
        assert determine_tier(["rename"]) == "trivial"

    def test_normal_signal(self):
        assert determine_tier(["feature"]) == "normal"
        assert determine_tier(["fix"]) == "normal"
        assert determine_tier(["refactor"]) in ("normal", "cross-cutting")

    def test_cross_cutting_signal(self):
        assert determine_tier(["migration"]) == "cross-cutting"
        assert determine_tier(["multi-file"]) == "cross-cutting"

    def test_high_risk_signal(self):
        assert determine_tier(["security"]) == "high-risk"
        assert determine_tier(["auth"]) == "high-risk"
        assert determine_tier(["credential"]) == "high-risk"
        assert determine_tier(["payment"]) == "high-risk"
        assert determine_tier(["database"]) == "high-risk"

    def test_no_matching_signal_returns_normal(self):
        assert determine_tier(["unknown-task"]) == "normal"
        assert determine_tier([]) == "normal"

    def test_highest_tier_wins(self):
        """When multiple tiers match, the highest-risk tier wins."""
        assert determine_tier(["typo", "security"]) == "high-risk"
        assert determine_tier(["feature", "migration"]) == "cross-cutting"
        assert determine_tier(["typo", "feature"]) == "normal"

    def test_refactor_ambiguity(self):
        """'refactor' appears in both normal and cross-cutting — should yield cross-cutting."""
        assert determine_tier(["refactor"]) == "cross-cutting"
