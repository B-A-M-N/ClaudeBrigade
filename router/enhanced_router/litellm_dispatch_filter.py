"""LiteLLM 1.93 deployment filter loaded inside the proxy child.

This file is copied beside each generated LiteLLM config because LiteLLM's
config loader imports callback modules relative to that config file.  Keep it
self-contained: it must not import ClaudeBrigade state or secrets from the
parent process.
"""

from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger


_POLICY_KEY = "brigade_route_policy"


def _policy(request_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    metadata = (request_kwargs or {}).get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(_POLICY_KEY)
    return value if isinstance(value, dict) else None


def _deployment_identity(deployment: dict[str, Any]) -> tuple[str | None, str | None]:
    model_info = deployment.get("model_info")
    if not isinstance(model_info, dict):
        model_info = {}
    deployment_id = model_info.get("deployment_id") or deployment.get("deployment_id")
    provider_id = model_info.get("provider_id") or deployment.get("provider_id")
    return (
        str(deployment_id) if deployment_id is not None else None,
        str(provider_id) if provider_id is not None else None,
    )


class BrigadeDeploymentFilter(CustomLogger):
    """Fail-closed filter for an immutable Brigade managed-group policy."""

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, Any]],
        messages: list[Any] | None,
        request_kwargs: dict[str, Any] | None = None,
        parent_otel_span: Any = None,
    ) -> list[dict[str, Any]]:
        del model, messages, parent_otel_span
        policy = _policy(request_kwargs)
        if policy is None:
            return healthy_deployments

        allowed_deployments = policy.get("allowed_deployments")
        allowed_providers = policy.get("provider_ids")
        if not isinstance(allowed_deployments, list) or not isinstance(allowed_providers, list):
            return []
        allowed_deployment_set = {str(value) for value in allowed_deployments}
        allowed_provider_set = {str(value) for value in allowed_providers}
        if not allowed_deployment_set:
            return []

        candidates = []
        for deployment in healthy_deployments:
            if not isinstance(deployment, dict):
                continue
            deployment_id, provider_id = _deployment_identity(deployment)
            if deployment_id in allowed_deployment_set and (
                not allowed_provider_set or provider_id in allowed_provider_set
            ):
                candidates.append(deployment)

        preferred = policy.get("preferred_deployment")
        if preferred is not None:
            preferred_matches = [
                deployment
                for deployment in candidates
                if _deployment_identity(deployment)[0] == str(preferred)
            ]
            if preferred_matches:
                return preferred_matches
        return candidates


proxy_handler_instance = BrigadeDeploymentFilter()
