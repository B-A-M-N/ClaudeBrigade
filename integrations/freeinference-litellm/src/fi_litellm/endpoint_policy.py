"""Endpoint selection for logical FreeInference models.

Selection is deliberately conservative: compatibility eligibility is checked
by the caller, then ``auto`` ranks endpoints by token-weighted cache reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass(frozen=True)
class EndpointPolicy:
    mode: str = "highest-cache-rate"
    default_endpoint: str = "openai"
    minimum_requests: int = 5
    minimum_input_tokens: int = 50_000
    maximum_metric_age_seconds: int = 86_400


@dataclass
class EndpointObservation:
    input_tokens_total: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    sample_count: int = 0
    latency_ms_total: float = 0.0
    succeeded: int = 0
    observed_at: float = field(default_factory=time)

    def add(
        self,
        *,
        input_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        succeeded: bool = True,
    ) -> None:
        self.input_tokens_total += max(0, input_tokens)
        self.cache_read_tokens += max(0, cache_read_tokens)
        self.cache_write_tokens += max(0, cache_write_tokens)
        self.output_tokens += max(0, output_tokens)
        self.sample_count += 1
        self.latency_ms_total += max(0.0, latency_ms)
        self.succeeded += int(succeeded)
        self.observed_at = time()

    @property
    def cache_rate(self) -> float:
        if self.input_tokens_total <= 0:
            return 0.0
        return min(1.0, self.cache_read_tokens / self.input_tokens_total)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.sample_count if self.sample_count else 0.0


@dataclass(frozen=True)
class EndpointSelection:
    endpoint_id: str
    reason: str


def select_endpoint(
    endpoints: set[str] | list[str] | tuple[str, ...],
    *,
    policy: EndpointPolicy | None = None,
    observations: dict[str, EndpointObservation] | None = None,
    explicit_endpoint: str | None = None,
    now: float | None = None,
) -> EndpointSelection:
    """Select one eligible endpoint without moving a pinned caller.

    Explicit selection always wins. For ``auto``, an endpoint only competes
    using cache evidence after the configured sample and token thresholds;
    otherwise the configured default endpoint wins deterministically.
    """
    available = sorted(set(endpoints))
    if not available:
        raise ValueError("at least one endpoint is required")
    policy = policy or EndpointPolicy()
    observations = observations or {}
    if explicit_endpoint and explicit_endpoint != "auto":
        if explicit_endpoint not in available:
            raise ValueError(f"endpoint '{explicit_endpoint}' is not available")
        return EndpointSelection(explicit_endpoint, "explicit endpoint override")

    baseline = policy.default_endpoint if policy.default_endpoint in available else available[0]
    if policy.mode != "highest-cache-rate":
        return EndpointSelection(baseline, f"configured default endpoint ({policy.mode})")

    current_time = time() if now is None else now
    eligible: list[tuple[float, int, str]] = []
    for endpoint_id in available:
        observation = observations.get(endpoint_id)
        if observation is None:
            continue
        if current_time - observation.observed_at > policy.maximum_metric_age_seconds:
            continue
        if observation.sample_count < policy.minimum_requests:
            continue
        if observation.input_tokens_total < policy.minimum_input_tokens:
            continue
        eligible.append((observation.cache_rate, observation.input_tokens_total, endpoint_id))

    if not eligible:
        return EndpointSelection(baseline, "insufficient fresh cache evidence; configured default")
    selected = max(eligible, key=lambda item: (item[0], item[1], item[2]))
    evidence = ", ".join(
        f"{endpoint} cache rate {rate:.3f} over {tokens} input tokens"
        for rate, tokens, endpoint in sorted(eligible, key=lambda item: item[2])
    )
    return EndpointSelection(selected[2], f"highest token-weighted cache rate: {evidence}")


def normalize_usage(payload: dict) -> dict[str, int]:
    """Normalize common OpenAI and Anthropic usage payloads."""
    usage = payload.get("usage", payload)
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0}
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    return {
        "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "cache_read_tokens": int(usage.get("cache_read_input_tokens", prompt_details.get("cached_tokens", 0)) or 0),
        "cache_write_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
    }
