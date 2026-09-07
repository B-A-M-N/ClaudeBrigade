"""Cache-aware selection among certified deployments of a logical model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from enhanced_router.config_models import ModelEndpointSpec, ModelSpec


@dataclass(frozen=True)
class SelectedEndpoint:
    endpoint_id: str
    spec: ModelEndpointSpec
    reason: str


def _legacy_endpoint(spec: ModelSpec) -> ModelEndpointSpec:
    return ModelEndpointSpec(
        backend=spec.backend,
        upstream_model=spec.upstream_model,
        litellm_model=spec.litellm_model,
        api_base=spec.api_base,
        api_base_env=spec.api_base_env,
        api_key_env=spec.api_key_env,
        auth=spec.auth,
        provider_id=spec.provider_id,
        protocol=("anthropic-messages" if spec.backend != "litellm" else "openai-chat"),
        certified=spec.provider_id is None and spec.backend != "anthropic-passthrough",
        tools=spec.capabilities.tools,
        streaming=spec.capabilities.streaming,
        reasoning=spec.capabilities.reasoning != "low",
        max_context_tokens=spec.capabilities.max_context_tokens,
        max_output_tokens=spec.capabilities.max_output_tokens,
    )


def select_endpoint(
    model_id: str,
    spec: ModelSpec,
    state: object,
    *,
    explicit_endpoint: str | None = None,
    require_certified: bool = True,
    provider_id: str | None = None,
    configuration_hash: str | None = None,
    required_capabilities: tuple[str, ...] = ("messages", "streaming"),
) -> SelectedEndpoint:
    """Select an endpoint using token-weighted cache evidence.

    Existing catalogs with one top-level backend are treated as a single
    deployment. New catalogs should define ``endpoints`` and certification
    status explicitly.
    """
    if spec.availability == "unavailable":
        raise ValueError(f"Model '{model_id}' is unavailable in the provider catalog")
    endpoints = spec.endpoints or {"default": _legacy_endpoint(spec)}
    if provider_id:
        provider_endpoints = {
            endpoint_id: endpoint
            for endpoint_id, endpoint in endpoints.items()
            if (endpoint.provider_id or spec.provider_id) == provider_id
        }
        if not provider_endpoints:
            raise ValueError(
                f"Provider '{provider_id}' has no endpoint for model '{model_id}'"
            )
        endpoints = provider_endpoints
    candidates: dict[str, ModelEndpointSpec] = {}
    for endpoint_id, endpoint in endpoints.items():
        if endpoint.availability == "unavailable":
            continue
        if not require_certified:
            candidates[endpoint_id] = endpoint
            continue
        certified = endpoint.certified
        checker = getattr(state, "is_endpoint_certified", None)
        if checker is not None and provider_id and configuration_hash:
            certified = checker(
                provider_id=provider_id,
                model_id=model_id,
                endpoint_id=endpoint_id,
                configuration_hash=configuration_hash,
                capabilities=required_capabilities,
            )
        if certified:
            candidates[endpoint_id] = endpoint
    if not candidates:
        raise ValueError(f"Model '{model_id}' has no certified endpoint")
    if explicit_endpoint and explicit_endpoint != "auto":
        endpoint = candidates.get(explicit_endpoint)
        if endpoint is None:
            raise ValueError(f"Endpoint '{explicit_endpoint}' is unavailable for model '{model_id}'")
        return SelectedEndpoint(explicit_endpoint, endpoint, "explicit endpoint override")

    get_observations = getattr(state, "get_endpoint_observations")
    try:
        observations = get_observations(
            provider_id=provider_id,
            model_id=model_id,
            configuration_hash=configuration_hash,
            maximum_age_seconds=spec.endpoint_policy.maximum_metric_age_seconds,
        )
    except TypeError:
        observations = get_observations(model_id)
    default_id = spec.endpoint_policy.default_endpoint
    if default_id not in candidates:
        default_id = sorted(candidates)[0]
    eligible: list[tuple[float, int, str]] = []
    for endpoint_id in candidates:
        observation = observations.get(endpoint_id)
        if observation is None:
            continue
        if observation.get("sample_count", 0) < spec.endpoint_policy.minimum_requests:
            continue
        if observation.get("input_tokens_total", 0) < spec.endpoint_policy.minimum_input_tokens:
            continue
        observed_at = observation.get("observed_at")
        if not observed_at:
            continue
        try:
            observed = datetime.fromisoformat(str(observed_at)).astimezone(timezone.utc)
            if (datetime.now(timezone.utc) - observed).total_seconds() > spec.endpoint_policy.maximum_metric_age_seconds:
                continue
        except ValueError:
            continue
        eligible.append((
            float(observation.get("cache_rate", 0.0)),
            int(observation.get("input_tokens_total", 0)),
            endpoint_id,
        ))
    if not eligible or spec.endpoint_policy.mode != "highest-cache-rate":
        return SelectedEndpoint(default_id, candidates[default_id], "insufficient cache evidence; configured default")
    selected = max(eligible, key=lambda item: (item[0], item[1], item[2]))
    evidence = ", ".join(
        f"{endpoint}={rate:.3f} over {tokens} input tokens"
        for rate, tokens, endpoint in sorted(eligible, key=lambda item: item[2])
    )
    return SelectedEndpoint(selected[2], candidates[selected[2]], f"highest token-weighted cache rate ({evidence})")
