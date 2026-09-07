from __future__ import annotations

import asyncio

from enhanced_router.litellm_dispatch_filter import BrigadeDeploymentFilter


def _deployment(endpoint_id: str, provider_id: str = "freeinference") -> dict:
    return {
        "model_name": "brigade-model",
        "model_info": {
            "deployment_id": endpoint_id,
            "provider_id": provider_id,
        },
    }


def test_filter_returns_healthy_deployments_without_brigade_policy():
    healthy = [_deployment("openai"), _deployment("anthropic")]
    result = asyncio.run(
        BrigadeDeploymentFilter().async_filter_deployments(
            "brigade-model", healthy, None, {}, None
        )
    )
    assert result == healthy


def test_filter_enforces_allowed_deployments_and_preference():
    openai = _deployment("openai")
    anthropic = _deployment("anthropic")
    result = asyncio.run(
        BrigadeDeploymentFilter().async_filter_deployments(
            "brigade-model",
            [openai, anthropic],
            None,
            {
                "metadata": {
                    "brigade_route_policy": {
                        "allowed_deployments": ["openai", "anthropic"],
                        "provider_ids": ["freeinference"],
                        "preferred_deployment": "anthropic",
                    }
                }
            },
            None,
        )
    )
    assert result == [anthropic]


def test_filter_fails_closed_when_policy_has_no_matching_deployment():
    result = asyncio.run(
        BrigadeDeploymentFilter().async_filter_deployments(
            "brigade-model",
            [_deployment("openai")],
            None,
            {
                "metadata": {
                    "brigade_route_policy": {
                        "allowed_deployments": ["missing"],
                        "provider_ids": ["freeinference"],
                    }
                }
            },
            None,
        )
    )
    assert result == []

