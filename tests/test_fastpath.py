from __future__ import annotations

import pytest

from enhanced_router.fastpath import (
    FastpathClient,
    FastpathPacketBuilder,
    FastpathPolicyValidator,
    FastpathRouteProposal,
    FastpathVerification,
    build_route_candidates,
    route_template,
    verification_template,
    _strip_thought_envelope,
)


class FakeRegistry:
    def __init__(self, models):
        self.models = models

    def get_model(self, model_id):
        return self.models[model_id]


class FakeState:
    def get_endpoint_observations(self, *args, **kwargs):
        return {}


class _RankedModel:
    def __init__(self, model_id: str, score: int):
        self.model_id = model_id
        self.score = score


class FakeRecommendingRegistry(FakeRegistry):
    """FakeRegistry plus a deterministic .recommend(role) for candidate tests."""

    def __init__(self, models, ranked_by_role: dict[str, list[_RankedModel]]):
        super().__init__(models)
        self.ranked_by_role = ranked_by_role

    def recommend(self, role):
        return self.ranked_by_role.get(role, [])


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


def test_strip_thought_envelope_removes_leading_think_block():
    content = "<think>reasoning about the task</think>\n{\"ok\": true}"
    assert _strip_thought_envelope(content) == '{"ok": true}'


def test_strip_thought_envelope_is_a_noop_without_a_think_block():
    content = '{"ok": true}'
    assert _strip_thought_envelope(content) == content


def _fake_client(**overrides) -> FastpathClient:
    kwargs = dict(
        api_base="https://fastpath.example.invalid",
        model="diffusiongemma",
        api_key_env="FREEINFERENCE_API_KEY",
        provider_id="freeinference",
    )
    kwargs.update(overrides)
    return FastpathClient(**kwargs)


def _fake_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.asyncio
async def test_request_defaults_to_json_object_and_no_reasoning_effort(monkeypatch):
    captured: dict = {}

    async def fake_post(**kwargs):
        captured.update(kwargs)
        return _fake_response('{"decision": "escalate", "checks": {}, '
                               '"violations": [], "requires_full_adversary": true, '
                               '"confidence": 0.1}')

    monkeypatch.setattr("enhanced_router.backends.post_openai_compatible_json", fake_post)
    client = _fake_client()
    result = await client.request("verify", {"contract": {}})

    assert result["decision"] == "escalate"
    payload = captured["payload"]
    assert payload["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in payload
    assert payload["max_tokens"] == 256


@pytest.mark.asyncio
async def test_request_strict_schema_sends_json_schema_response_format(monkeypatch):
    captured: dict = {}

    async def fake_post(**kwargs):
        captured.update(kwargs)
        return _fake_response('{"workflow_tier": "normal", "recommended_roles": [], '
                               '"routes": {}, "confidence": 0.1, '
                               '"escalate_to_controller": true}')

    monkeypatch.setattr("enhanced_router.backends.post_openai_compatible_json", fake_post)
    client = _fake_client(strict_schema=True)
    await client.request("route", {"task": "x"})

    response_format = captured["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == FastpathRouteProposal.model_json_schema()


@pytest.mark.asyncio
async def test_request_strict_schema_uses_the_right_schema_per_mode(monkeypatch):
    captured: dict = {}

    async def fake_post(**kwargs):
        captured.update(kwargs)
        return _fake_response('{"decision": "escalate", "checks": {}, '
                               '"violations": [], "requires_full_adversary": true, '
                               '"confidence": 0.1}')

    monkeypatch.setattr("enhanced_router.backends.post_openai_compatible_json", fake_post)
    client = _fake_client(strict_schema=True)
    await client.request("verify", {"contract": {}})

    schema = captured["payload"]["response_format"]["json_schema"]["schema"]
    assert schema == FastpathVerification.model_json_schema()


@pytest.mark.asyncio
async def test_request_disable_thinking_sends_reasoning_effort_none(monkeypatch):
    captured: dict = {}

    async def fake_post(**kwargs):
        captured.update(kwargs)
        return _fake_response('{"decision": "escalate", "checks": {}, '
                               '"violations": [], "requires_full_adversary": true, '
                               '"confidence": 0.1}')

    monkeypatch.setattr("enhanced_router.backends.post_openai_compatible_json", fake_post)
    client = _fake_client(disable_thinking=True)
    await client.request("verify", {"contract": {}})

    assert captured["payload"]["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_request_strips_thought_envelope_before_parsing(monkeypatch):
    async def fake_post(**kwargs):
        return _fake_response(
            '<think>let me think</think>{"decision": "escalate", "checks": {}, '
            '"violations": [], "requires_full_adversary": true, "confidence": 0.1}'
        )

    monkeypatch.setattr("enhanced_router.backends.post_openai_compatible_json", fake_post)
    client = _fake_client()
    result = await client.request("verify", {"contract": {}})
    assert result["decision"] == "escalate"


@pytest.mark.asyncio
async def test_request_raises_clean_error_on_invalid_json(monkeypatch):
    async def fake_post(**kwargs):
        return _fake_response("not json at all")

    monkeypatch.setattr("enhanced_router.backends.post_openai_compatible_json", fake_post)
    client = _fake_client()
    with pytest.raises(ValueError, match="not valid JSON"):
        await client.request("verify", {"contract": {}})


def test_route_template_defaults_to_escalation():
    template = route_template("normal", ["recon", "implementer"])
    assert template["confidence"] == 0.0
    assert template["escalate_to_controller"] is True
    assert template["recommended_roles"] == []
    assert template["routes"] == {
        "recon": {"candidate_id": None},
        "implementer": {"candidate_id": None},
    }


def test_verification_template_defaults_to_escalation():
    template = verification_template(["lint", "tests"])
    assert template["decision"] == "escalate"
    assert template["confidence"] == 0.0
    assert template["requires_full_adversary"] is True
    assert template["checks"] == {"lint": "unknown", "tests": "unknown"}


def test_build_route_candidates_produces_compact_deterministic_aliases():
    registry = FakeRecommendingRegistry({}, {
        "implementer": [
            _RankedModel("provider/big-model-name", 10),
            _RankedModel("provider/other-model", 8),
        ],
        "recon": [_RankedModel("provider/scout", 9)],
    })
    packet_candidates, candidate_map = build_route_candidates(
        registry, ["recon", "implementer", "adversary"],
    )
    assert packet_candidates["implementer"] == [
        {"candidate_id": "i0", "score": 10},
        {"candidate_id": "i1", "score": 8},
    ]
    assert packet_candidates["recon"] == [{"candidate_id": "r0", "score": 9}]
    assert "adversary" not in packet_candidates  # no ranked models -> omitted

    # No raw model_id leaks into the packet-safe candidate list.
    for role_candidates in packet_candidates.values():
        for entry in role_candidates:
            assert "model_id" not in entry

    assert candidate_map["implementer"] == {
        "i0": "provider/big-model-name", "i1": "provider/other-model",
    }
    assert candidate_map["recon"] == {"r0": "provider/scout"}


def test_build_route_candidates_respects_limit():
    registry = FakeRecommendingRegistry({}, {
        "recon": [_RankedModel(f"model-{i}", 10 - i) for i in range(5)],
    })
    packet_candidates, candidate_map = build_route_candidates(registry, ["recon"], limit=2)
    assert len(packet_candidates["recon"]) == 2
    assert set(candidate_map["recon"]) == {"r0", "r1"}


def _write_certified_model(model_id: str):
    from types import SimpleNamespace
    return SimpleNamespace(
        enabled=True,
        allowed_roles=["implementer"],
        capabilities=SimpleNamespace(write_tool_certified=True),
    )


def test_validate_route_resolves_candidate_id_from_offered_set():
    registry = FakeRegistry({"provider/big-model-name": _write_certified_model("provider/big-model-name")})
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"candidate_id": "i0"}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    validated = FastpathPolicyValidator().validate_route(
        proposal, minimum_tier="normal", registry=registry, state=FakeState(),
        configuration_hash="cfg",
        candidate_map={"implementer": {"i0": "provider/big-model-name"}},
    )
    assert validated.routes["implementer"].candidate_id == "i0"


def test_validate_route_rejects_a_candidate_id_not_in_the_offered_set():
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"candidate_id": "i7"}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    with pytest.raises(ValueError, match="not in the offered set"):
        FastpathPolicyValidator().validate_route(
            proposal, minimum_tier="normal", registry=FakeRegistry({}), state=FakeState(),
            configuration_hash="cfg",
            candidate_map={"implementer": {"i0": "provider/big-model-name"}},
        )


def test_validate_route_rejects_candidate_id_when_no_candidates_were_offered():
    proposal = FastpathRouteProposal.model_validate({
        "workflow_tier": "normal",
        "recommended_roles": ["implementer"],
        "routes": {"implementer": {"candidate_id": "i0"}},
        "confidence": 0.99,
        "escalate_to_controller": False,
    })
    with pytest.raises(ValueError, match="not in the offered set"):
        FastpathPolicyValidator().validate_route(
            proposal, minimum_tier="normal", registry=FakeRegistry({}), state=FakeState(),
            configuration_hash="cfg", candidate_map=None,
        )
