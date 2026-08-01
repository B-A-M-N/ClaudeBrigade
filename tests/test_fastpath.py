from __future__ import annotations

import pytest

from enhanced_router.fastpath import (
    FastpathPacketBuilder,
    FastpathPolicyValidator,
    FastpathRouteProposal,
)


class FakeRegistry:
    def __init__(self, models):
        self.models = models

    def get_model(self, model_id):
        return self.models[model_id]


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
