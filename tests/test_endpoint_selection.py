from __future__ import annotations

import pytest

from enhanced_router.config_models import ModelCapabilities, ModelEndpointSpec, ModelSpec
from enhanced_router.endpoint_selection import select_endpoint


class FakeState:
    def __init__(self, observations):
        self.observations = observations

    def get_endpoint_observations(self, model_id):
        return self.observations


def test_qwen_selects_openai_when_cache_rate_is_higher():
    spec = ModelSpec(
        display_name="Qwen",
        provider_id="freeinference",
        backend="litellm",
        litellm_model="openai/qwen3.6-35b",
        endpoints={
            "openai": ModelEndpointSpec(backend="litellm", litellm_model="openai/qwen3.6-35b", api_base="https://freeinference.org/v1", certified=True),
            "anthropic": ModelEndpointSpec(backend="direct-anthropic", upstream_model="qwen3.6-35b", api_base="https://freeinference.org/anthropic", certified=True),
        },
        capabilities=ModelCapabilities(tools=True, mutation=False, context_tokens=262144, reasoning="medium"),
    )
    selected = select_endpoint(
        "qwen3.6-35b",
        spec,
        FakeState({
            "openai": {"sample_count": 5, "input_tokens_total": 500_000, "cache_rate": 0.88, "observed_at": "2026-07-31T15:00:00+00:00"},
            "anthropic": {"sample_count": 5, "input_tokens_total": 500_000, "cache_rate": 0.02, "observed_at": "2026-07-31T15:00:00+00:00"},
        }),
    )
    assert selected.endpoint_id == "openai"


def test_explicit_endpoint_is_pinned_over_cache_rate():
    spec = ModelSpec(
        display_name="Qwen",
        backend="litellm",
        litellm_model="openai/qwen3.6-35b",
        endpoints={
            "openai": ModelEndpointSpec(backend="litellm", litellm_model="openai/qwen3.6-35b", certified=True),
            "anthropic": ModelEndpointSpec(backend="direct-anthropic", upstream_model="qwen3.6-35b", api_base="https://freeinference.org/anthropic", certified=True),
        },
        capabilities=ModelCapabilities(tools=True, mutation=False, context_tokens=262144, reasoning="medium"),
    )
    selected = select_endpoint("qwen3.6-35b", spec, FakeState({}), explicit_endpoint="anthropic")
    assert selected.endpoint_id == "anthropic"


def test_provider_override_filters_auto_endpoint_selection():
    """A route candidate's provider cannot be silently ignored."""
    spec = ModelSpec(
        display_name="Shared model",
        backend="litellm",
        litellm_model="openai/shared-model",
        endpoints={
            "openai": ModelEndpointSpec(
                backend="litellm", litellm_model="openai/shared-model",
                provider_id="provider-a", certified=True,
            ),
            "anthropic": ModelEndpointSpec(
                backend="direct-anthropic", upstream_model="shared-model",
                api_base="https://provider-b.example/anthropic",
                provider_id="provider-b", certified=True,
            ),
        },
        capabilities=ModelCapabilities(
            tools=True, mutation=False, context_tokens=262144, reasoning="medium",
        ),
    )
    selected = select_endpoint(
        "shared-model", spec, FakeState({}), provider_id="provider-b",
    )
    assert selected.endpoint_id == "anthropic"


def test_uncertified_endpoints_are_not_production_candidates():
    spec = ModelSpec(
        display_name="Qwen",
        provider_id="freeinference",
        backend="litellm",
        litellm_model="openai/qwen3.6-35b",
        endpoints={
            "openai": ModelEndpointSpec(backend="litellm", litellm_model="openai/qwen3.6-35b"),
            "anthropic": ModelEndpointSpec(backend="direct-anthropic", upstream_model="qwen3.6-35b", api_base="https://freeinference.org/anthropic"),
        },
        capabilities=ModelCapabilities(tools=True, mutation=False, context_tokens=262144, reasoning="medium"),
    )
    with pytest.raises(ValueError, match="no certified endpoint"):
        select_endpoint("qwen3.6-35b", spec, FakeState({}))
