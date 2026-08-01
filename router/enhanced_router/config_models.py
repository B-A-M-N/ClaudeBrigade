"""Pydantic models for ClaudeBrigade configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    """Base class for operator configuration.

    Configuration typos must fail at load time.  Runtime request payloads have
    their own compatibility models; these models describe the immutable
    policy graph and therefore must not silently discard fields.
    """

    model_config = ConfigDict(extra="forbid")


class ModelCapabilities(StrictConfigModel):
    """Capabilities for a given model."""

    tools: bool = False
    mutation: bool = False
    # Provider-owned capability.  ``None`` means the provider catalog has not
    # supplied a limit yet; zero is not a fabricated fallback limit.
    context_tokens: int | None = None
    reasoning: Literal["low", "medium", "high"] = "medium"
    local: bool = False
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    anthropic_messages: bool = False
    openai_chat_completions: bool = False
    streaming: bool = True
    tool_calls: bool = False
    parallel_tool_calls: bool = False
    structured_output: bool = False
    thinking: bool = False
    tool_streaming: bool = False
    prompt_cache: bool = False
    controller_eligible: bool = False
    latency_class: Literal["fast", "standard", "slow"] = "standard"
    cost_class: Literal["free", "low", "standard", "high"] = "standard"
    write_tool_certified: bool | None = None
    read_tool_certified: bool | None = None

    @model_validator(mode="after")
    def _normalize_compatibility_fields(self) -> "ModelCapabilities":
        """Normalize only structural aliases; never infer authority."""
        # ``mutation`` was the pre-certification configuration field.  Keep a
        # one-way migration for existing user YAML while making the new field
        # authoritative for all routing decisions.  The field still does not
        # grant filesystem authority; the native agent and mutation lease do.
        if self.write_tool_certified is None:
            self.write_tool_certified = self.mutation
        if self.max_context_tokens is None and self.context_tokens is not None:
            self.max_context_tokens = self.context_tokens
        elif self.context_tokens is None and self.max_context_tokens is not None:
            self.context_tokens = self.max_context_tokens
        if self.tool_calls is False:
            self.tool_calls = self.tools
        return self


class EndpointPolicySpec(StrictConfigModel):
    """Selection policy for multiple protocol deployments of one model."""

    mode: Literal["highest-cache-rate", "configured-default"] = "highest-cache-rate"
    default_endpoint: str = "openai"
    minimum_requests: int = Field(default=5, ge=1)
    minimum_input_tokens: int = Field(default=50_000, ge=0)
    maximum_metric_age_seconds: int = Field(default=86_400, ge=1)


class ModelEndpointSpec(StrictConfigModel):
    """One protocol deployment of a logical model."""

    backend: Literal["direct-anthropic", "litellm", "anthropic-passthrough"]
    upstream_model: str | None = None
    litellm_model: str | None = None
    api_base: str | None = None
    api_base_env: str | None = None
    api_key_env: str | None = None
    auth: "ModelAuthSpec | None" = None
    provider_id: str | None = None
    protocol: Literal["anthropic-messages", "openai-chat", "unknown"] = "unknown"
    availability: Literal["public", "restricted", "account-specific", "unavailable", "unknown"] = "unknown"
    pricing_class: Literal["free", "low", "standard", "high", "unknown"] = "unknown"
    priority: int = 0
    routing_owner: Literal["brigade", "provider"] = "brigade"
    max_context_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    tools: bool = False
    streaming: bool = True
    reasoning: bool = False
    data_policy: Literal["public", "private", "secret"] = "private"
    certified: bool = False
    certification_id: str | None = None

    @model_validator(mode="after")
    def _validate_endpoint(self) -> "ModelEndpointSpec":
        if self.backend == "litellm" and not self.litellm_model:
            raise ValueError("litellm endpoint requires litellm_model")
        if self.backend in {"direct-anthropic", "anthropic-passthrough"} and not self.upstream_model:
            raise ValueError(f"{self.backend} endpoint requires upstream_model")
        if self.backend == "direct-anthropic" and not (self.api_base or self.api_base_env):
            raise ValueError("direct-anthropic endpoint requires api_base or api_base_env")
        return self


class ProviderLimitsSpec(StrictConfigModel):
    # ``max_concurrency`` is the operator-facing single knob.  The two
    # underlying limits remain separate so admission can distinguish native
    # agent slots from upstream request streams.
    max_concurrency: int | None = Field(default=None, ge=1)
    max_active_agents: int = Field(default=4, ge=1)
    max_inflight_requests: int = Field(default=4, ge=1)
    max_queued_agents: int = Field(default=8, ge=0)
    queue_timeout_seconds: float = Field(default=20.0, gt=0)

    @model_validator(mode="after")
    def _apply_shared_concurrency(self) -> "ProviderLimitsSpec":
        if self.max_concurrency is not None:
            self.max_active_agents = self.max_concurrency
            self.max_inflight_requests = self.max_concurrency
        return self


class ProviderDeadlineSpec(StrictConfigModel):
    time_to_first_token_seconds: float = Field(default=45.0, gt=0)
    stream_idle_seconds: float = Field(default=90.0, gt=0)
    request_wall_seconds: float = Field(default=600.0, gt=0)
    agent_wall_seconds: float = Field(default=900.0, gt=0)


class ProviderRetrySpec(StrictConfigModel):
    max_attempts: int = Field(default=0, ge=0)
    retryable_statuses: list[int] = Field(default_factory=list)
    max_backoff_seconds: float = Field(default=15.0, ge=0)


class ProviderSpec(StrictConfigModel):
    display_name: str
    api_key_env: str | None = None
    max_concurrency_env: str | None = None
    endpoints: dict[str, str] = Field(default_factory=dict)
    limits: ProviderLimitsSpec = Field(default_factory=ProviderLimitsSpec)
    deadlines: ProviderDeadlineSpec = Field(default_factory=ProviderDeadlineSpec)
    retry: ProviderRetrySpec = Field(default_factory=ProviderRetrySpec)
    routing_owner: Literal["brigade", "provider"] = "brigade"
    requests_per_minute: int | None = Field(default=None, ge=1)
    requests_per_day: int | None = Field(default=None, ge=1)
    reserve_requests: int = Field(default=0, ge=0)
    discovery: dict = Field(default_factory=dict)


class FastpathConfigSpec(StrictConfigModel):
    """Bounded, advisory DiffusionGemma service policy."""

    enabled: bool = False
    model_id: str
    endpoint: str = "auto"
    modes: list[Literal["route", "verify"]] = Field(default_factory=lambda: ["route"])
    read_only: bool = True
    authoritative: bool = False
    timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    max_packet_tokens: int = Field(default=12_000, ge=256)
    max_packet_bytes: int = Field(default=64_000, ge=1024)
    max_output_tokens: int = Field(default=1_024, ge=64)
    route_confidence_threshold: float = Field(default=0.88, ge=0, le=1)
    failure_policy: Literal["bypass", "fail"] = "bypass"
    system_prompts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _advisory_only(self) -> "FastpathConfigSpec":
        if self.authoritative:
            raise ValueError("fastpath must remain advisory")
        if not self.read_only:
            raise ValueError("fastpath must be read-only")
        return self


class SidecarSpec(StrictConfigModel):
    """Independent, bounded inference policy for one router sidecar."""

    model_id: str
    mode: Literal["route", "verify", "structured"] = "structured"
    endpoint: str = "auto"
    enabled: bool = True
    timeout_seconds: float = Field(default=45.0, gt=0, le=600)
    max_packet_bytes: int = Field(default=64_000, ge=1_024, le=256_000)
    max_output_tokens: int = Field(default=2_048, ge=64, le=131_072)
    system_prompt: str = ""

    @model_validator(mode="after")
    def _validate_sidecar_policy(self) -> "SidecarSpec":
        if not self.model_id.strip():
            raise ValueError("sidecar model_id must not be empty")
        if self.endpoint != "auto" and not self.endpoint.strip():
            raise ValueError("sidecar endpoint must be 'auto' or a named endpoint")
        return self


class SidecarProfileSpec(StrictConfigModel):
    """A named, reusable bundle limiting which globally-defined sidecars
    (sidecars.yaml) are available for a launch, plus that launch's own
    fastpath coprocessor config.

    Fastpath is scoped per sidecar-profile here rather than being a single
    global singleton: two launch presets can run different fastpath
    coprocessors (or none) side by side. When ``fastpath`` is not set, the
    caller falls back to the bare global ``fastpath.yaml`` singleton for
    backward compatibility with configs that predate sidecar profiles.
    """

    sidecar_ids: list[str] = Field(default_factory=list)
    fastpath: FastpathConfigSpec | None = None


class LaunchPresetSpec(StrictConfigModel):
    """Pairs a saved inference profile with a saved sidecar profile so an
    operator can switch both together with one named choice at launch."""

    inference_profile_id: str
    sidecar_profile_id: str | None = None


class ModelAuthSpec(StrictConfigModel):
    """Explicit transport authentication for a provider backend.

    Supported types:
      - bearer:       ``Authorization: Bearer <key>``
      - x-api-key:    ``x-api-key: <key>``
      - custom-header: ``<header>: <key>``
      - none:         no credential injection
    """

    type: Literal["bearer", "x-api-key", "custom-header", "none"] = "bearer"
    header: str | None = None
    prefix: str = ""

    def effective_header(self) -> str:
        """Return the header name to use for this auth spec."""
        if self.type == "bearer":
            return self.header or "Authorization"
        if self.type == "x-api-key":
            return self.header or "x-api-key"
        return self.header or "Authorization"

    def format_value(self, key: str) -> str:
        """Format a credential value for injection into the header."""
        if self.type == "bearer":
            prefix = self.prefix if self.prefix else "Bearer "
            return f"{prefix}{key}"
        return key


class ModelSpec(StrictConfigModel):
    """Full specification for a single model."""

    display_name: str
    backend: Literal["direct-anthropic", "litellm", "anthropic-passthrough"]
    upstream_model: str | None = None
    litellm_model: str | None = None
    api_base: str | None = None
    api_base_env: str | None = None
    api_key_env: str | None = None
    auth: ModelAuthSpec | None = None
    provider_id: str | None = None
    catalog_source: Literal["discovered", "bundled", "operator"] = "bundled"
    availability: Literal["public", "restricted", "account-specific", "unavailable", "unknown"] = "unknown"
    # ``fixed`` selects one certified endpoint when a binding is created.
    # ``managed-group`` binds a logical LiteLLM group and lets the provider
    # router choose among its certified equivalent deployments per request.
    routing_mode: Literal["fixed", "managed-group"] = "fixed"
    deployment_group: str | None = None
    endpoints: dict[str, ModelEndpointSpec] = Field(default_factory=dict)
    endpoint_policy: EndpointPolicySpec = Field(default_factory=EndpointPolicySpec)
    capabilities: ModelCapabilities
    allowed_roles: set[str] = Field(default_factory=set)
    enabled: bool = True

    def has_litellm_endpoint(self) -> bool:
        """Whether any enabled logical deployment uses the LiteLLM backend."""
        return self.backend == "litellm" or any(
            endpoint.backend == "litellm" for endpoint in self.endpoints.values()
        )

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> "ModelSpec":
        if self.backend == "litellm" and not self.litellm_model and not self.endpoints:
            raise ValueError(
                "backend='litellm' requires a non-empty litellm_model field"
            )
        if self.backend == "direct-anthropic" and not self.upstream_model and not self.endpoints:
            raise ValueError(
                "backend='direct-anthropic' requires a non-empty upstream_model field"
            )
        if self.backend == "anthropic-passthrough" and not self.upstream_model and not self.endpoints:
            raise ValueError(
                "backend='anthropic-passthrough' requires a non-empty upstream_model field"
            )
        # P0-1: direct-anthropic MUST have an explicit endpoint — no Anthropic fallback
        if self.backend == "direct-anthropic" and not self.endpoints:
            if not self.api_base and not self.api_base_env:
                raise ValueError(
                    "backend='direct-anthropic' requires an explicit endpoint "
                    "(api_base or api_base_env). The Anthropic fallback was removed "
                    "to prevent credential leakage to the wrong host."
                )
        if self.routing_mode == "managed-group":
            if not self.endpoints:
                raise ValueError("routing_mode='managed-group' requires endpoint deployments")
            if any(endpoint.backend != "litellm" for endpoint in self.endpoints.values()):
                raise ValueError(
                    "managed-group routes require LiteLLM-backed equivalent deployments"
                )
        return self


class ProfileSpec(StrictConfigModel):
    """Profile mapping role names to model IDs.

    ``controller`` (a ``RouteTargetSpec``) and the legacy ``controller_model``
    string are both optional overrides for the profile's controller route;
    they do not make the controller a statically routed role like recon/
    implementer/adversary/repairer. Use ``controller_route()`` to read
    whichever is set.
    """

    recon: "str | RouteTargetSpec"
    implementer: "str | RouteTargetSpec"
    adversary: "str | RouteTargetSpec"
    repairer: "str | RouteTargetSpec"
    controller_model: str | None = None
    controller: "RouteTargetSpec | None" = None
    specialists: dict[str, "SpecialistSpec"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_role_name_as_model_id(self) -> "ProfileSpec":
        """Reject profiles that use role names as model IDs.

        Profile values must reference models from models.yaml
        (e.g. 'longcat-2', 'qwen-local'), not role names themselves.
        """
        role_names = {"recon", "implementer", "adversary", "repairer"}
        for field_name in ("recon", "implementer", "adversary", "repairer"):
            value = getattr(self, field_name)
            if isinstance(value, RouteTargetSpec):
                value = value.model
            if value in role_names:
                raise ValueError(
                    f"Profile field '{field_name}' uses role name '{value}' "
                    f"as model ID; must reference a model from models.yaml"
                )
        return self

    def route_target(self, role: str) -> "RouteTargetSpec":
        value = getattr(self, role)
        return value if isinstance(value, RouteTargetSpec) else RouteTargetSpec(model=value)

    def controller_route(self) -> "RouteTargetSpec | None":
        """Return the controller's route, from the new ``controller`` field
        if present, else synthesized from the legacy ``controller_model``
        string, else ``None`` if this profile configures no controller."""
        if self.controller is not None:
            return self.controller
        if self.controller_model:
            return RouteTargetSpec(model=self.controller_model)
        return None


class RouteCandidateSpec(StrictConfigModel):
    """One concrete model+endpoint choice for a role or controller route."""

    model: str
    endpoint: str = "auto"


class RouteTargetSpec(StrictConfigModel):
    """A role's (or controller's) primary route plus ordered fallback candidates.

    Each fallback is a full (model, endpoint) pair, not just a model ID, so a
    fallback can pin a different provider/endpoint for the same model or an
    entirely different model+endpoint combination.
    """

    primary: RouteCandidateSpec
    fallbacks: list[RouteCandidateSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_shape(cls, data: Any) -> Any:
        """Parse the old flat {model, endpoint, fallback_models: [str]} shape
        (and bare fallback dicts already shaped like {model, endpoint}) into
        the new primary/fallbacks shape. Existing profiles.yaml files and
        ``RouteTargetSpec(model=...)`` call sites must keep working unchanged."""
        if not isinstance(data, dict) or "primary" in data:
            return data
        if "model" not in data:
            return data
        fallback_raw = data.get("fallback_models") or []
        fallbacks = []
        for item in fallback_raw:
            if isinstance(item, str):
                fallbacks.append({"model": item})
            elif isinstance(item, dict):
                fallbacks.append(item)
        return {
            "primary": {"model": data["model"], "endpoint": data.get("endpoint", "auto")},
            "fallbacks": fallbacks,
        }

    # Backward-compatible read accessors -- every existing caller reads
    # .model / .endpoint / .fallback_models directly; keep those working so
    # this migration does not require touching every consumer.
    @property
    def model(self) -> str:
        return self.primary.model

    @property
    def endpoint(self) -> str:
        return self.primary.endpoint

    @property
    def fallback_models(self) -> list[str]:
        return [c.model for c in self.fallbacks]


class SpecialistSpec(StrictConfigModel):
    """Named visible agent assignment within a profile."""

    model: str
    roles: list[str] = Field(default_factory=list)
    latency_class: Literal["fast", "standard", "slow"] = "standard"
    activation: str | None = None
    endpoint: str = "auto"
    native_agent_name: str | None = None
    public_model_alias: str | None = None

    @model_validator(mode="after")
    def _validate_launch_identity(self) -> "SpecialistSpec":
        if bool(self.native_agent_name) != bool(self.public_model_alias):
            raise ValueError(
                "native_agent_name and public_model_alias must be supplied together"
            )
        if self.native_agent_name and not self.native_agent_name.startswith("brigade-"):
            raise ValueError("native_agent_name must use the 'brigade-' namespace")
        if self.public_model_alias and not self.public_model_alias.startswith("anthropic-brigade-"):
            raise ValueError(
                "public_model_alias must use the 'anthropic-brigade-' namespace"
            )
        return self


ProfileSpec.model_rebuild()


class RecommendationConstraints(StrictConfigModel):
    """Filters passed to ``ModelRegistry.recommend``."""

    role: str
    required_context_tokens: int | None = None
    local_only: bool = False
    requires_tools: bool = True
    prefer_low_cost: bool = False


class RankedModel(StrictConfigModel):
    """A single ranked result from ``recommend``."""

    model_id: str
    score: int
    reason: str


# ------------------------------------------------------------------
# Workflow definitions
# ------------------------------------------------------------------


class WorkflowPhase(StrictConfigModel):
    """A single phase in a typed workflow definition."""

    id: str
    ordinal: int | None = None
    roles: list[str] = Field(default_factory=list)
    required: bool = True
    mutation: bool = False
    depends_on: list[str] = Field(default_factory=list)
    conditional: str | None = None
    actor: str | None = None
    distinct_agent_from: list[str] = Field(default_factory=list)
    parallel_group: str | None = None
    max_duration_seconds: int | None = Field(default=None, ge=1)
    turn_budget: int | None = Field(default=None, ge=1)
    provider_requirements: list[str] = Field(default_factory=list)
    min_fanout: int = Field(default=1, ge=1)
    max_fanout: int = Field(default=1, ge=1)
    result_schema: str | None = None
    quality_quorum: int = Field(default=1, ge=1)
    fallback_policy: str | None = None
    execution_kind: Literal["native_agent", "sidecar_call", "controller_action"] = "native_agent"
    max_parallelism: int | None = Field(default=None, ge=1)
    required_successes: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    max_attempts_per_model: int | None = Field(default=None, ge=1)
    sidecar: str | None = None

    @model_validator(mode="after")
    def _validate_sidecar_phase(self) -> "WorkflowPhase":
        if self.sidecar and self.execution_kind != "sidecar_call":
            raise ValueError("workflow phase sidecar requires execution_kind='sidecar_call'")
        if self.sidecar and self.mutation:
            raise ValueError("sidecar phases must remain read-only")
        if self.execution_kind == "sidecar_call" and self.mutation:
            raise ValueError("sidecar_call phases must remain read-only")
        return self


class WorkflowSpec(StrictConfigModel):
    """Typed workflow specification loaded from workflows.yaml."""

    default_profile: str
    phases: list[WorkflowPhase] = Field(default_factory=list)


# ------------------------------------------------------------------
# Tier policy — deterministic task-classification
# ------------------------------------------------------------------

TIER_SIGNALS: dict[str, list[str]] = {
    "trivial": ["typo", "comment", "format", "rename"],
    "normal": ["feature", "fix", "refactor"],
    "cross-cutting": ["refactor", "migration", "multi-file"],
    "high-risk": ["security", "auth", "credential", "payment", "database"],
}


class TierPolicy(StrictConfigModel):
    """Policy governing tier escalation behavior."""

    min_tier: str = "normal"
    escalation_allowed: bool = True


def determine_tier(signals: list[str]) -> str:
    """Determine the workflow tier from task signals.

    Returns the *highest-risk* tier matched by any of the provided signals.
    Tier ordering: trivial < normal < cross-cutting < high-risk.
    If no signal matches, returns "normal".
    """
    # Build a reverse map: signal_word -> tier
    word_to_tier: dict[str, str] = {}
    for tier_name, tier_words in TIER_SIGNALS.items():
        for word in tier_words:
            word_to_tier[word] = tier_name

    # Find the highest tier matched
    tier_order = ["trivial", "normal", "cross-cutting", "high-risk"]
    matched: list[str] = []
    for signal in signals:
        tier = word_to_tier.get(signal)
        if tier is not None:
            matched.append(tier)

    if not matched:
        return "normal"

    # Return the highest tier found
    for tier in reversed(tier_order):
        if tier in matched:
            return tier

    return "normal"
