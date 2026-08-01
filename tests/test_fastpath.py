from __future__ import annotations

from types import SimpleNamespace

import pytest

from enhanced_router.fastpath import (
    FastpathPacketBuilder,
    FastpathPolicyValidator,
    FastpathRouteCandidate,
    FastpathRouteProposal,
    build_route_candidates,
)


class FakeRegistry:
    def __init__(self, models):
        self.models = models

    def get_model(self, model_id):
        return self.models[model_id]


def _fake_model(*, enabled=True, allowed_roles=frozenset({"implementer"}), write_tool_certified=True):
    return SimpleNamespace(
        enabled=enabled,
        allowed_roles=allowed_roles,
        capabilities=SimpleNamespace(write_tool_certified=write_tool_certified),
    )


class FakeState:
    def get_endpoint_observations(self, *args, **kwargs):
        return {}


def test_route_schema_rejects_unknown_keys_and_bad_confidence():
    with pytest.raises(ValueError):
        FastpathRouteProposal.model_validate({
            "workflow_tier": "normal",
            "recommended_roles": ["implementer"],
            "routes": {},
            "confidence": 1.2,
            "escalate_to_controller": False,
            "unexpected": True,
        })


def test_packet_builder_bounds_diff_and_marks_truncation():
    builder = FastpathPacketBuilder()
    packet = builder.verification_packet(
        contract={}, deterministic_checks={}, allowed_files=[], changed_files=[],
        diff="\n".join(f"line-{i}" for i in range(500)), commands=[], claims=[], findings=[],
    )
    assert packet["diff_truncated"] is True
    assert len(packet["diff"].splitlines()) == 400


def test_fastpath_cannot_lower_deterministic_tier():
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"slot": "work", "preferred_logical_model": "worker", "fanout": 1}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    with pytest.raises(ValueError, match="lower"):
        FastpathPolicyValidator().validate_route(
            proposal,
            minimum_tier="cross-cutting",
            registry=FakeRegistry({}),
            state=FakeState(),
            configuration_hash="cfg",
        )


def test_route_schema_accepts_logical_model_shape_but_only_auto_endpoint():
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {
            "implementer": {
                "model": "worker",
                "endpoint": "auto",
            },
        },
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    target = proposal.routes["implementer"]
    assert target.logical_model == "worker"
    assert target.endpoint == "auto"

    with pytest.raises(ValueError, match="auto"):
        FastpathRouteProposal.model_validate({
            "workflow_tier": "normal",
            "recommended_roles": ["implementer"],
            "routes": {
                "implementer": {
                    "model": "worker",
                    "endpoint": "openai",
                },
            },
            "confidence": 0.99,
            "escalate_to_controller": False,
        })


def test_validate_route_resolves_model_from_offered_candidate_id():
    """P0-1/P0-2: DiffusionGemma selects a candidate_id, never a bare model
    string -- validate_route resolves model/endpoint from the offered
    candidate set itself, not from anything the model claims."""
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"candidate_id": "cand-implementer-worker"}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    candidates = [
        FastpathRouteCandidate(candidate_id="cand-implementer-worker", role="implementer", model_id="worker"),
    ]
    validated = FastpathPolicyValidator().validate_route(
        proposal, minimum_tier="normal",
        registry=FakeRegistry({"worker": _fake_model()}),
        state=FakeState(), configuration_hash="cfg",
        candidates=candidates,
    )
    target = validated.routes["implementer"]
    assert target.model == "worker"
    assert target.endpoint == "auto"


def test_validate_route_rejects_a_candidate_id_that_was_not_offered():
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"candidate_id": "cand-invented"}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    with pytest.raises(ValueError, match="not offered"):
        FastpathPolicyValidator().validate_route(
            proposal, minimum_tier="normal",
            registry=FakeRegistry({"worker": _fake_model()}),
            state=FakeState(), configuration_hash="cfg",
            candidates=[
                FastpathRouteCandidate(candidate_id="cand-implementer-worker", role="implementer", model_id="worker"),
            ],
        )


def test_validate_route_rejects_a_candidate_id_offered_for_a_different_role():
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"candidate_id": "cand-recon-worker"}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    with pytest.raises(ValueError, match="not offered"):
        FastpathPolicyValidator().validate_route(
            proposal, minimum_tier="normal",
            registry=FakeRegistry({"worker": _fake_model()}),
            state=FakeState(), configuration_hash="cfg",
            candidates=[
                FastpathRouteCandidate(candidate_id="cand-recon-worker", role="recon", model_id="worker"),
            ],
        )


def test_validate_route_without_candidates_falls_back_to_legacy_model_field():
    """Backward compatibility: a caller that doesn't supply a candidate set
    (candidates=None, the default) still validates the model's own
    model/preferred_logical_model string exactly as before Phase 2."""
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"model": "worker", "endpoint": "auto"}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    validated = FastpathPolicyValidator().validate_route(
        proposal, minimum_tier="normal",
        registry=FakeRegistry({"worker": _fake_model()}),
        state=FakeState(), configuration_hash="cfg",
    )
    assert validated.routes["implementer"].model == "worker"


def test_build_route_candidates_uses_registry_recommend_not_invention():
    """Candidates come from ModelRegistry.recommend() -- real, role-eligible,
    scored models -- never anything the fastpath model could have invented."""
    from pathlib import Path

    from enhanced_router.registry import ModelRegistry

    registry = ModelRegistry(Path(__file__).resolve().parents[1] / "config")
    registry.load_models()
    candidates = build_route_candidates(registry=registry, roles=["implementer"], per_role=2)
    assert candidates
    assert all(c.role == "implementer" for c in candidates)
    assert len(candidates) <= 2
    for candidate in candidates:
        spec = registry.get_model(candidate.model_id)
        assert spec.enabled
        assert "implementer" in spec.allowed_roles
        assert candidate.candidate_id == f"cand-implementer-{candidate.model_id}"
