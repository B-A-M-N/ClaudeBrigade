from __future__ import annotations

from fi_litellm.sync import build_config
from fi_litellm.endpoint_policy import EndpointObservation, EndpointPolicy, normalize_usage, select_endpoint


def test_build_config_uses_openai_provider_namespace() -> None:
    config = build_config([{"id": "glm-5.1"}, {"id": "qwen3.6-35b"}])
    entries = config["model_list"]
    assert [entry["model_name"] for entry in entries] == ["glm-5.1", "qwen3.6-35b"]
    assert entries[0]["litellm_params"]["model"] == "openai/glm-5.1"
    assert entries[1]["litellm_params"]["api_base"].endswith("/v1")
    assert config["litellm_settings"]["drop_params"] is False
    assert config["litellm_settings"]["num_retries"] == 0


def test_build_config_deduplicates_and_sorts_catalog() -> None:
    config = build_config([{"id": "z-model"}, {"id": "a-model"}, {"id": "z-model"}])
    assert [entry["model_name"] for entry in config["model_list"]] == ["a-model", "z-model"]


def test_auto_endpoint_selection_uses_token_weighted_cache_rate() -> None:
    openai = EndpointObservation()
    anthropic = EndpointObservation()
    for _ in range(5):
        openai.add(input_tokens=100_000, cache_read_tokens=90_000)
        anthropic.add(input_tokens=100_000, cache_read_tokens=5_000)
    selected = select_endpoint(
        ["openai", "anthropic"],
        policy=EndpointPolicy(minimum_requests=5, minimum_input_tokens=500_000),
        observations={"openai": openai, "anthropic": anthropic},
    )
    assert selected.endpoint_id == "openai"
    assert "token-weighted cache rate" in selected.reason


def test_explicit_endpoint_wins_over_cache_evidence() -> None:
    selected = select_endpoint(
        ["openai", "anthropic"],
        explicit_endpoint="anthropic",
        observations={"openai": EndpointObservation()},
    )
    assert selected.endpoint_id == "anthropic"


def test_usage_normalization_handles_openai_and_anthropic_shapes() -> None:
    assert normalize_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}, "completion_tokens": 5})["cache_read_tokens"] == 80
    assert normalize_usage({"input_tokens": 100, "cache_read_input_tokens": 80, "output_tokens": 5})["cache_read_tokens"] == 80
