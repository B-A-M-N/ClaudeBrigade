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


# Claude Code's native frontmatter aliases.  ``background`` and ``custom``
# are ClaudeBrigade execution lanes, not values that should be emitted as a
# Claude Code ``model:`` alias.  Background requests use the small-fast
# environment route; custom uses the configured concrete router alias.
ClaudeNativeAlias = Literal["main", "sonnet", "haiku", "opus", "fable"]
NativeSlotName = Literal[
    "main", "sonnet", "haiku", "opus", "fable", "background", "custom", "small-fast"
]


class SlotRouteSpec(StrictConfigModel):
    """Backing route for one Claude Code native model slot."""

    model: str
    endpoint: str = "auto"
    provider_id: str | None = None
    fallbacks: list["RouteCandidateSpec"] = Field(default_factory=list)
    reserve_class: Literal[
        "controller", "critical", "verification", "worker", "optional"
    ] = "worker"


class NativeAgentSpec(StrictConfigModel):
    """Named native worker projected from a profile's model slot."""

    template: str
    slot: NativeSlotName | None = None
    explicit_model: str | None = None
    roles: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    can_mutate: bool = False
    counts_as_implementation: bool = False
    may_spawn_agents: bool = False
    may_integrate: bool = False
    may_adjudicate: bool = False
    isolation: Literal["none", "worktree"] = "none"
    background: bool = True
    permission_mode: str | None = None
    max_turns: int = Field(default=80, ge=1)
    effort: Literal["low", "medium", "high"] = "medium"

    @model_validator(mode="after")
    def _validate_native_agent(self) -> "NativeAgentSpec":
        if bool(self.slot) == bool(self.explicit_model):
            raise ValueError(
                "native agent must select exactly one slot or explicit_model"
            )
        if self.can_mutate and self.isolation != "worktree":
            raise ValueError("mutating native agents require worktree isolation")
        if not self.may_spawn_agents and "Agent" in self.tools:
            raise ValueError("non-controller native agents cannot receive Agent")
        if self.can_mutate and not (self.counts_as_implementation or "Edit" in self.tools or "Write" in self.tools):
            raise ValueError("mutating native agent has no mutation tools")
        if self.can_mutate:
            self.counts_as_implementation = True
        return self


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
    controller_reserve: int = Field(default=0, ge=0)
    max_worker_concurrency: int | None = Field(default=None, ge=1)
    priority_policy: Literal["fifo", "strict"] = "fifo"

    @model_validator(mode="after")
    def _apply_shared_concurrency(self) -> "ProviderLimitsSpec":
        if self.max_concurrency is not None:
            self.max_active_agents = self.max_concurrency
            self.max_inflight_requests = self.max_concurrency
        if self.controller_reserve >= self.max_active_agents:
            raise ValueError("controller_reserve must be below max_active_agents")
        if self.max_worker_concurrency is None:
            self.max_worker_concurrency = max(
                1, self.max_active_agents - self.controller_reserve
            )
        if self.max_worker_concurrency > self.max_active_agents:
            raise ValueError("max_worker_concurrency cannot exceed max_active_agents")
        return self


class RunResourcePolicy(StrictConfigModel):
    """Run-wide resource ceilings applied across every provider and lane.

    Provider admission remains authoritative for provider-specific limits.  This
    policy is the complementary run-level budget: it prevents one workflow
    from consuming all of the run's worker, mutation, review, coprocessor, or
    shadow-worktree capacity even when those requests are spread across
    multiple providers.
    """

    max_active_native_agents: int | None = Field(default=6, ge=1)
    max_active_mutators: int | None = Field(default=2, ge=1)
    max_active_reviewers: int | None = Field(default=3, ge=1)
    max_active_coprocessors: int | None = Field(default=2, ge=1)
    max_active_worktrees: int | None = Field(default=3, ge=1)
    max_reserved_tokens: int | None = Field(default=None, ge=1)
    max_estimated_cost: float | None = Field(default=None, ge=0)
    deadline_seconds: float | None = Field(default=None, gt=0)


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
    # Health is an admission input, not merely a display/recommendation hint.
    # Untested models remain usable by default for development and discovery,
    # while a stale or confirmed-failed check cannot be admitted silently.
    health_max_age_seconds: float = Field(default=900.0, gt=0)
    allow_untested_models: bool = True
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
    provider_id: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    fallback_routes: list["RouteCandidateSpec"] = Field(default_factory=list)
    modes: list[Literal["route", "verify"]] = Field(default_factory=lambda: ["route"])
    read_only: bool = True
    authoritative: bool = False
    timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    max_packet_tokens: int = Field(default=12_000, ge=256)
    max_packet_bytes: int = Field(default=64_000, ge=1024)
    # DiffusionGemma generates one 256-token canvas in parallel; a longer
    # response requires a second sequential canvas, giving up most of its
    # latency advantage over an ordinary autoregressive call for what should
    # be a small classification result.
    max_output_tokens: int = Field(default=256, ge=64, le=256)
    route_confidence_threshold: float = Field(default=0.88, ge=0, le=1)
    failure_policy: Literal["bypass", "fail"] = "bypass"
    system_prompts: dict[str, str] = Field(default_factory=dict)
    # Off by default: forces response_format to the exact Pydantic schema
    # instead of a bare json_object. Constrained decoding support varies by
    # deployment -- verify against the actual endpoint before enabling.
    strict_schema: bool = False
    # Off by default: sends reasoning_effort="none". Only meaningful for a
    # model whose deployment actually recognizes that field -- verify before
    # enabling, since an unrecognized field's handling is provider-specific.
    disable_thinking: bool = False

    @model_validator(mode="after")
    def _advisory_only(self) -> "FastpathConfigSpec":
        if self.authoritative:
            raise ValueError("fastpath must remain advisory")
        if not self.read_only:
            raise ValueError("fastpath must be read-only")
        if not self.fallback_routes and self.fallback_models:
            self.fallback_routes = [
                RouteCandidateSpec(model=model) for model in self.fallback_models
            ]
        return self

    @property
    def route(self) -> "RouteTargetSpec":
        """Expose the canonical primary-plus-fallback route ladder.

        ``model_id``/``endpoint`` remain readable for legacy callers and
        YAML migration, but new routing code should consume this exact route
        identity instead of rebuilding candidates ad hoc.
        """
        return RouteTargetSpec(
            primary=RouteCandidateSpec(
                model=self.model_id,
                endpoint=self.endpoint,
                provider_id=self.provider_id,
            ),
            fallbacks=list(self.fallback_routes),
        )


class SidecarAgentSpec(StrictConfigModel):
    """A full Claude Code worker backed by an independently selected model.

    Sidecar agents execute through Claude Code's native ``Agent`` tool.  This
    model describes the projected worker identity and capability policy; it
    does not grant authority by itself.  Sidecars carry a concrete route of
    their own; they are not assignments to the native Claude Code model
    slots. Claims, lifecycle attachment, and mutation leases remain
    authoritative at runtime.
    """

    # Stable semantic identity used by workflow/action/lifecycle records.
    # The YAML mapping key remains the migration/configuration identifier, but
    # it must not be the authority for the worker's role or backing model.
    worker_id: str | None = None
    model_id: str
    native_agent_name: str
    public_model_alias: str
    roles: list[str] = Field(default_factory=list)
    description: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=lambda: ["Read", "Grep", "Glob", "Bash"])
    disallowed_tools: list[str] = Field(default_factory=list)
    can_mutate: bool = False
    counts_as_implementation: bool = False
    isolation: Literal["none", "worktree"] = "none"
    background: bool = True
    max_turns: int = Field(default=80, ge=1)
    effort: Literal["low", "medium", "high"] = "medium"
    permission_mode: str | None = None
    template: str | None = None
    endpoint: str = "auto"
    provider_id: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    fallback_routes: list["RouteCandidateSpec"] = Field(default_factory=list)
    max_parallelism: int = Field(default=1, ge=1)
    may_spawn_agents: bool = False
    may_integrate: bool = False
    may_adjudicate: bool = False
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_sidecar_agent_policy(self) -> "SidecarAgentSpec":
        if self.worker_id is not None:
            self.worker_id = self.worker_id.strip()
            if not self.worker_id:
                raise ValueError("sidecar worker_id must not be empty")
            if any(char.isspace() for char in self.worker_id):
                raise ValueError("sidecar worker_id must not contain whitespace")
        if not self.model_id.strip():
            raise ValueError("sidecar agent model_id must not be empty")
        if not self.native_agent_name.startswith("brigade-"):
            raise ValueError("native_agent_name must use the 'brigade-' namespace")
        if not self.public_model_alias.startswith("anthropic-brigade-"):
            raise ValueError(
                "public_model_alias must use the 'anthropic-brigade-' namespace"
            )
        if self.can_mutate and self.isolation != "worktree":
            raise ValueError("mutating sidecar agents require worktree isolation")
        if self.can_mutate and not ({"Edit", "Write"} & set(self.tools)):
            raise ValueError("mutating sidecar agent has no mutation tools")
        if self.endpoint != "auto" and not self.endpoint.strip():
            raise ValueError("sidecar endpoint must be 'auto' or a named endpoint")
        if self.can_mutate and not self.counts_as_implementation:
            # A mutating reviewer/repair worker is still an implementation
            # lifecycle for completion evidence unless explicitly configured
            # otherwise by a future policy layer.
            self.counts_as_implementation = True
        if not self.fallback_routes and self.fallback_models:
            self.fallback_routes = [
                RouteCandidateSpec(model=model)
                for model in self.fallback_models
            ]
        return self

    @property
    def route(self) -> "RouteTargetSpec":
        """Return the worker's complete ordered route ladder."""
        return RouteTargetSpec(
            primary=RouteCandidateSpec(
                model=self.model_id,
                endpoint=self.endpoint,
                provider_id=self.provider_id,
            ),
            fallbacks=list(self.fallback_routes),
        )


class CoprocessorSpec(StrictConfigModel):
    """Independent, bounded structured inference policy.

    Coprocessors never receive Claude Code tools or filesystem authority.
    DiffusionGemma route/verify calls use this lane.
    """

    model_id: str
    mode: Literal["route", "verify", "sentinel", "structured"] = "structured"
    endpoint: str = "auto"
    provider_id: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    fallback_routes: list["RouteCandidateSpec"] = Field(default_factory=list)
    enabled: bool = True
    timeout_seconds: float = Field(default=45.0, gt=0, le=600)
    # Keep policy bounds aligned with the executor's hard transport limits.
    max_packet_bytes: int = Field(default=64_000, ge=1_024, le=64_000)
    max_output_tokens: int = Field(default=2_048, ge=64, le=32_000)
    system_prompt: str = ""
    input_schema_id: str | None = None
    output_schema_id: str | None = None
    prompt_version: str = "v1"
    capability_contract_version: str = "v1"
    minimum_confidence: float | None = Field(default=None, ge=0, le=1)
    supported_task_classes: list[str] = Field(default_factory=list)
    supported_languages: list[str] = Field(default_factory=list)
    supported_artifact_types: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    maximum_diff_lines: int | None = Field(default=None, ge=1)
    maximum_context_tokens: int | None = Field(default=None, ge=1)
    abstention_required_when_incomplete: bool = True
    requires_independent_model: bool = False
    independence: dict[str, bool] = Field(default_factory=dict)
    automatic_mode: Literal[
        "automatic", "advisory-on-request", "shadow-only", "disabled"
    ] = "automatic"
    minimum_evaluated_outcomes: int = Field(default=20, ge=1)
    minimum_adoption_rate: float = Field(default=0.0, ge=0, le=1)
    maximum_harm_rate: float = Field(default=0.25, ge=0, le=1)
    degraded_mode: Literal[
        "advisory-on-request", "shadow-only", "disabled"
    ] = "shadow-only"

    @model_validator(mode="after")
    def _validate_coprocessor_policy(self) -> "CoprocessorSpec":
        if not self.model_id.strip():
            raise ValueError("coprocessor model_id must not be empty")
        if self.endpoint != "auto" and not self.endpoint.strip():
            raise ValueError("coprocessor endpoint must be 'auto' or a named endpoint")
        if not self.fallback_routes and self.fallback_models:
            self.fallback_routes = [
                RouteCandidateSpec(model=model) for model in self.fallback_models
            ]
        if self.automatic_mode == "automatic" and not self.output_schema_id:
            # Legacy calls remain usable, but cannot be promoted to automatic
            # feedback until they declare an output contract.
            self.automatic_mode = "advisory-on-request"
        return self

    @property
    def route(self) -> "RouteTargetSpec":
        """Return the bounded call's complete ordered route ladder."""
        return RouteTargetSpec(
            primary=RouteCandidateSpec(
                model=self.model_id,
                endpoint=self.endpoint,
                provider_id=self.provider_id,
            ),
            fallbacks=list(self.fallback_routes),
        )


class FeedbackMonitorSpec(StrictConfigModel):
    """Policy for automatically injecting bounded coprocessor feedback."""

    enabled: bool = False
    coprocessor_id: str = "verification-reviewer"
    checkpoints: list[Literal["post_tool_batch", "post_edit_checkpoint"]] = Field(
        default_factory=lambda: ["post_tool_batch"]
    )
    watched_tools: list[str] = Field(
        default_factory=lambda: ["Edit", "Write", "NotebookEdit", "Bash"]
    )
    cooldown_seconds: float = Field(default=15.0, ge=0, le=3600)
    max_calls_per_execution: int = Field(default=3, ge=1, le=32)
    max_parallelism: int = Field(default=1, ge=1, le=8)
    wait_seconds: float = Field(default=6.0, ge=0, le=120)
    required: bool = False

    @model_validator(mode="after")
    def _validate_feedback_monitor(self) -> "FeedbackMonitorSpec":
        if not self.coprocessor_id.strip():
            raise ValueError("feedback monitor coprocessor_id must not be empty")
        if not self.checkpoints:
            raise ValueError("feedback monitor requires at least one checkpoint")
        if not self.watched_tools:
            raise ValueError("feedback monitor requires at least one watched tool")
        return self


# Source compatibility for installed callers and pre-migration fixtures. New
# code should use CoprocessorSpec and coprocessor_call terminology.
SidecarSpec = CoprocessorSpec


class SidecarProfileSpec(StrictConfigModel):
    """A named launch bundle for native sidecar agents and coprocessors.

    ``sidecar_ids`` remains a legacy alias for bounded coprocessors in
    pre-migration profiles. New configuration should use the explicit fields.

    The profile also carries that launch's own
    fastpath coprocessor config.

    Fastpath is scoped per sidecar-profile here rather than being a single
    global singleton: two launch presets can run different fastpath
    coprocessors (or none) side by side. When ``fastpath`` is not set, the
    caller falls back to the bare global ``fastpath.yaml`` singleton for
    backward compatibility with configs that predate sidecar profiles.
    """

    sidecar_agent_ids: list[str] = Field(default_factory=list)
    # Launch-level kill switch for bounded router-owned calls. Native Claude
    # Code sidecar agents and the separately configured fastpath keep their
    # own policies and are not affected by this flag.
    coprocessors_enabled: bool = True
    # Per-launch route ladders for native workers.  This lets one named
    # sidecar profile route the same worker identity through a different
    # provider/model/endpoint set without mutating the global worker spec.
    agent_route_overrides: dict[str, RouteTargetSpec] = Field(default_factory=dict)
    coprocessor_ids: list[str] = Field(default_factory=list)
    sidecar_ids: list[str] = Field(default_factory=list)
    fastpath: FastpathConfigSpec | None = None
    feedback_monitor: FeedbackMonitorSpec | None = None


class LaunchPresetSpec(StrictConfigModel):
    """Pairs a saved inference profile with a saved sidecar profile so an
    operator can switch both together with one named choice at launch.

    ``workflow_id`` is optional for backwards compatibility.  When present,
    it selects the orchestration composition for this launch; it never
    changes the inference or sidecar model lanes themselves.
    """

    inference_profile_id: str
    sidecar_profile_id: str | None = None
    workflow_id: str | None = None


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
    # Native Claude Code model slots are an additional projection of the
    # ordinary role routes. They do not replace the four durable role routes.
    slots: dict[str, SlotRouteSpec] = Field(default_factory=dict)
    agents: dict[str, NativeAgentSpec] = Field(default_factory=dict)

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
        invalid_slots = set(self.slots) - {
            "main", "sonnet", "haiku", "opus", "fable", "background", "custom", "small-fast"
        }
        if invalid_slots:
            raise ValueError(
                "Profile contains unknown native slots: "
                + ", ".join(sorted(invalid_slots))
            )
        return self

    def route_target(self, role: str) -> "RouteTargetSpec":
        value = getattr(self, role)
        return value if isinstance(value, RouteTargetSpec) else RouteTargetSpec(
            primary=RouteCandidateSpec(model=value)
        )

    def controller_route(self) -> "RouteTargetSpec | None":
        """Return the controller's route, from the new ``controller`` field
        if present, else synthesized from the legacy ``controller_model``
        string, else ``None`` if this profile configures no controller."""
        if self.controller is not None:
            return self.controller
        if self.controller_model:
            return RouteTargetSpec(
                primary=RouteCandidateSpec(model=self.controller_model)
            )
        return None


class RouteCandidateSpec(StrictConfigModel):
    """One concrete provider+model+endpoint choice for a route."""

    model: str
    endpoint: str = "auto"
    provider_id: str | None = None


class RouteTargetSpec(StrictConfigModel):
    """A role's (or controller's) primary route plus ordered fallback candidates.

    Each fallback is a full (model, endpoint) pair, not just a model ID, so a
    fallback can pin a different provider/endpoint for the same model or an
    entirely different model+endpoint combination.
    """

    primary: RouteCandidateSpec
    fallbacks: list[RouteCandidateSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_route_ladder(self) -> "RouteTargetSpec":
        """Reject duplicate exact routes before they reach runtime state."""
        seen: set[tuple[str, str, str | None]] = set()
        for index, candidate in enumerate([self.primary, *self.fallbacks]):
            key = (candidate.model, candidate.endpoint, candidate.provider_id)
            if key in seen:
                label = "primary" if index == 0 else f"fallback #{index}"
                raise ValueError(f"route ladder contains duplicate candidate at {label}")
            seen.add(key)
        return self

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
        fallback_raw = data.get("fallback_routes") or data.get("fallback_models") or []
        fallbacks = []
        for item in fallback_raw:
            if isinstance(item, str):
                fallbacks.append({"model": item})
            elif isinstance(item, dict):
                fallbacks.append(item)
        return {
            "primary": {
                "model": data["model"],
                "endpoint": data.get("endpoint", "auto"),
                "provider_id": data.get("provider_id"),
            },
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
    # Prompt/tool contract template. The template is a role surface; the
    # backing model remains a separate route property.
    template: str | None = None
    roles: list[str] = Field(default_factory=list)
    latency_class: Literal["fast", "standard", "slow"] = "standard"
    activation: str | None = None
    endpoint: str = "auto"
    # Catalog/recommendation-only specialists must say so explicitly.  A
    # launchable specialist without a native identity is a configuration
    # error, not something the manifest may silently omit.
    launchable: bool = True
    native_agent_name: str | None = None
    public_model_alias: str | None = None

    @model_validator(mode="after")
    def _validate_launch_identity(self) -> "SpecialistSpec":
        if self.launchable and not self.native_agent_name:
            raise ValueError(
                "launchable specialists must declare native_agent_name and public_model_alias"
            )
        if not self.launchable and (self.native_agent_name or self.public_model_alias):
            raise ValueError(
                "catalog-only specialists must not declare a native launch identity"
            )
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


class PhaseConditionSpec(StrictConfigModel):
    """Deterministic condition used to branch or skip a workflow phase."""

    kind: Literal[
        "accepted_findings",
        "phase_result_equals",
        "phase_result_in",
        "workflow_tier_at_least",
        "explicit_escalation",
        "workspace_generation_changed",
    ]
    phase_id: str | None = None
    field: str | None = None
    value: str | None = None
    values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_condition(self) -> "PhaseConditionSpec":
        if self.kind in {"phase_result_equals", "phase_result_in"}:
            if not self.phase_id or not self.field:
                raise ValueError(
                    f"{self.kind} requires phase_id and field"
                )
            if self.kind == "phase_result_equals" and self.value is None:
                raise ValueError("phase_result_equals requires value")
            if self.kind == "phase_result_in" and not self.values:
                raise ValueError("phase_result_in requires values")
        return self


class ResultContractSpec(StrictConfigModel):
    """Machine-readable verdict contract for a native worker phase."""

    schema_id: str
    success_verdicts: list[str] = Field(default_factory=list)
    branch_verdicts: list[str] = Field(default_factory=list)
    failure_verdicts: list[str] = Field(default_factory=list)
    blocked_verdicts: list[str] = Field(default_factory=list)
    branch_field: str | None = None
    branches: dict[str, str] = Field(default_factory=dict)
    requires_current_generation: bool = False
    requires_current_digest: bool = False

    @model_validator(mode="after")
    def _validate_contract(self) -> "ResultContractSpec":
        verdicts = (
            set(self.success_verdicts)
            | set(self.branch_verdicts)
            | set(self.failure_verdicts)
            | set(self.blocked_verdicts)
        )
        if len(verdicts) != (
            len(self.success_verdicts)
            + len(self.branch_verdicts)
            + len(self.failure_verdicts)
            + len(self.blocked_verdicts)
        ):
            raise ValueError("result verdicts must be unique across contract classes")
        if self.branches and not self.branch_field:
            raise ValueError("result branches require branch_field")
        return self


ProfileSpec.model_rebuild()
FastpathConfigSpec.model_rebuild()
SidecarAgentSpec.model_rebuild()
CoprocessorSpec.model_rebuild()


class RecommendationConstraints(StrictConfigModel):
    """Filters passed to ``ModelRegistry.recommend``."""

    role: str
    required_context_tokens: int | None = None
    local_only: bool = False
    requires_tools: bool = True
    prefer_low_cost: bool = False
    max_cost_class: Literal["free", "low", "standard", "high"] | None = None
    minimum_health_score: int = Field(default=0, ge=0, le=5)
    exclude_model_ids: set[str] = Field(default_factory=set)


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
    condition: PhaseConditionSpec | None = None
    actor: str | None = None
    agent_id: str | None = None
    distinct_agent_from: list[str] = Field(default_factory=list)
    parallel_group: str | None = None
    max_duration_seconds: int | None = Field(default=None, ge=1)
    turn_budget: int | None = Field(default=None, ge=1)
    provider_requirements: list[str] = Field(default_factory=list)
    min_fanout: int = Field(default=1, ge=1)
    max_fanout: int = Field(default=1, ge=1)
    result_schema: str | None = None
    result_contract: ResultContractSpec | None = None
    quality_quorum: int = Field(default=1, ge=1)
    fallback_policy: str | None = None
    execution_kind: Literal[
        "native_agent", "coprocessor_call", "controller_action", "sidecar_call"
    ] = "native_agent"
    max_parallelism: int | None = Field(default=None, ge=1)
    required_successes: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    max_attempts_per_model: int | None = Field(default=None, ge=1)
    completion_mode: Literal["quorum", "all", "all_packages", "controller"] = "quorum"
    launch_policy: Literal[
        "minimum_first", "all_packages", "fill_capacity", "hedged"
    ] = "minimum_first"
    minimum_quality_score: float | None = Field(default=None, ge=0, le=1)
    # A terminal worker lifecycle is not quality evidence.  Native workers
    # and bounded calls must be explicitly adjudicated before a phase can
    # satisfy its quorum.
    requires_controller_acceptance: bool = True
    initial_fanout: int | None = Field(default=None, ge=1)
    maximum_replicas: int | None = Field(default=None, ge=1)
    hedge_delay_seconds: float | None = Field(default=None, ge=0)
    produces: str | None = None
    fanout_from: str | None = None
    sidecar_agent: str | None = None
    coprocessor: str | None = None
    # Legacy bounded-sidecar field. Retained for reading old workflow YAML;
    # new definitions must use ``coprocessor`` or ``sidecar_agent``.
    sidecar: str | None = None

    @model_validator(mode="after")
    def _validate_sidecar_phase(self) -> "WorkflowPhase":
        if self.sidecar and self.execution_kind != "sidecar_call":
            raise ValueError("workflow phase sidecar requires execution_kind='sidecar_call'")
        if self.sidecar and self.mutation:
            raise ValueError("sidecar phases must remain read-only")
        if self.execution_kind == "sidecar_call" and self.mutation:
            raise ValueError("sidecar_call phases must remain read-only")
        if self.sidecar_agent and self.execution_kind != "native_agent":
            raise ValueError("sidecar agents execute as native agents")
        if self.agent_id and self.execution_kind != "native_agent":
            raise ValueError("named native agents require execution_kind='native_agent'")
        if self.agent_id and self.sidecar_agent:
            raise ValueError("a workflow phase cannot name both agent_id and sidecar_agent")
        if self.coprocessor and self.execution_kind != "coprocessor_call":
            raise ValueError("coprocessors require coprocessor_call")
        if self.execution_kind == "coprocessor_call" and self.mutation:
            raise ValueError("coprocessor calls cannot mutate")
        if self.sidecar_agent and self.coprocessor:
            raise ValueError("a workflow phase cannot name both a sidecar agent and coprocessor")
        if self.sidecar_agent and self.sidecar:
            raise ValueError("sidecar_agent cannot be combined with legacy sidecar")
        if self.initial_fanout is not None and self.maximum_replicas is not None:
            if self.initial_fanout > self.maximum_replicas:
                raise ValueError("initial_fanout cannot exceed maximum_replicas")
        if self.condition is not None and self.conditional is not None:
            raise ValueError("use condition instead of legacy conditional, not both")
        return self


class WorkflowSpec(StrictConfigModel):
    """Typed workflow specification loaded from workflows.yaml."""

    default_profile: str
    # Composition is an execution-plane contract, not a model preference.
    # ``adaptive`` preserves compatibility for operator-authored workflows;
    # bundled workflows declare their intended native/sidecar composition.
    composition_mode: Literal[
        "native-only", "sidecar-only", "native-augmented", "adaptive"
    ] = "adaptive"
    # Custom workflow IDs are not ordered by name.  A declared tier lets the
    # task intake gate prove that a forced/custom workflow still satisfies an
    # already committed minimum tier instead of silently downgrading it.
    tier: Literal["trivial", "normal", "cross-cutting", "high-risk"] | None = None
    phases: list[WorkflowPhase] = Field(default_factory=list)
    resource_policy: RunResourcePolicy = Field(default_factory=RunResourcePolicy)

    @model_validator(mode="after")
    def _validate_composition_mode(self) -> "WorkflowSpec":
        has_native_worker = any(
            phase.agent_id is not None for phase in self.phases
        )
        has_sidecar_worker = any(
            phase.sidecar_agent is not None for phase in self.phases
        )
        if self.composition_mode == "native-only" and has_sidecar_worker:
            raise ValueError(
                "native-only workflows cannot contain sidecar_agent phases"
            )
        if self.composition_mode == "sidecar-only" and has_native_worker:
            raise ValueError(
                "sidecar-only workflows cannot contain agent_id worker phases"
            )
        if self.composition_mode == "native-augmented" and not (
            has_native_worker and has_sidecar_worker
        ):
            raise ValueError(
                "native-augmented workflows require both native and sidecar worker phases"
            )
        return self


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
