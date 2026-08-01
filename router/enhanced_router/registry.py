"""ModelRegistry — loads YAML config and provides lookup + deterministic ranking."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import yaml

from enhanced_router.config_models import (
    ModelAuthSpec,
    ModelCapabilities,
    ModelEndpointSpec,
    EndpointPolicySpec,
    FastpathConfigSpec,
    ModelSpec,
    ProfileSpec,
    ProviderSpec,
    RecommendationConstraints,
    RankedModel,
    SpecialistSpec,
    WorkflowSpec,
)

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))

# One-time migration aliases for the pre-logical-model FreeInference catalog.
# They are accepted at input boundaries but are never published as separate
# operational model identities.
MODEL_ID_ALIASES: dict[str, str] = {
    "freeinf-flash": "glm-5-turbo",
    "freeinf-qwen": "qwen3.6-35b",
    "freeinf-glm-5.1": "glm-5.1",
    "freeinf-minimax": "minimax-m2.7",
    "freeinf-minimax-m3": "minimax-m3",
    "freeinf-kimi-k2.7-code": "kimi-k2.7-code",
    "freeinf-diffusiongemma": "diffusiongemma",
}


class ModelRegistry:
    """Central registry backed by YAML configuration files."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self._config_dir = Path(config_dir) if config_dir else None
        self._models: dict[str, ModelSpec] = {}
        self._profiles: dict[str, ProfileSpec] = {}
        self._workflows: dict[str, WorkflowSpec] = {}
        self._providers: dict[str, ProviderSpec] = {}
        self._fastpath: FastpathConfigSpec | None = None
        self._discovered_digest: str = ""
        self._raw_yaml: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public loading
    # ------------------------------------------------------------------

    def load_models(self, path: str | Path | None = None) -> dict[str, ModelSpec]:
        """Load model definitions from *path* (or the default config location)."""
        target = Path(path) if path else self._config_path("models.yaml")
        if not target.exists():
            raise FileNotFoundError(f"models config not found: {target}")
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["models"] = raw
        data = yaml.safe_load(raw)
        models: dict[str, ModelSpec] = {}
        for model_id, raw_model in (data or {}).get("models", {}).items():
            caps = raw_model.get("capabilities", {})
            if isinstance(caps, str):
                caps = yaml.safe_load(caps)
            capabilities = ModelCapabilities(**caps)
            auth_raw = raw_model.get("auth")
            auth_spec = ModelAuthSpec(**auth_raw) if auth_raw else None
            endpoints = {
                endpoint_id: ModelEndpointSpec(**endpoint)
                for endpoint_id, endpoint in (raw_model.get("endpoints") or {}).items()
            }
            endpoint_policy = EndpointPolicySpec(**(raw_model.get("endpoint_policy") or {}))
            spec = ModelSpec(
                display_name=raw_model["display_name"],
                backend=raw_model["backend"],
                upstream_model=raw_model.get("upstream_model"),
                litellm_model=raw_model.get("litellm_model"),
                api_base=raw_model.get("api_base"),
                api_base_env=raw_model.get("api_base_env"),
                api_key_env=raw_model.get("api_key_env"),
                auth=auth_spec,
                provider_id=raw_model.get("provider_id"),
                catalog_source=raw_model.get("catalog_source", "bundled"),
                availability=raw_model.get("availability", "unknown"),
                endpoints=endpoints,
                endpoint_policy=endpoint_policy,
                capabilities=capabilities,
                allowed_roles=set(raw_model.get("allowed_roles", [])),
                enabled=raw_model.get("enabled", True),
            )
            # Endpoint declarations are deployments of the logical model. A
            # legacy top-level provider remains a migration fallback, but new
            # multi-provider routes must carry identity on each endpoint.
            if spec.provider_id:
                spec.endpoints = {
                    endpoint_id: endpoint.model_copy(update={"provider_id": endpoint.provider_id or spec.provider_id})
                    for endpoint_id, endpoint in spec.endpoints.items()
                }
            models[model_id] = spec
        self._models = models
        return models

    def load_profiles(
        self, path: str | Path | None = None
    ) -> dict[str, ProfileSpec]:
        """Load profile definitions from *path*."""
        target = Path(path) if path else self._config_path("profiles.yaml")
        if not target.exists():
            raise FileNotFoundError(f"profiles config not found: {target}")
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["profiles"] = raw
        data = yaml.safe_load(raw)
        profiles: dict[str, ProfileSpec] = {}
        for profile_id, raw_profile_value in (data or {}).get("profiles", {}).items():
            raw_profile: dict[str, Any] = dict(raw_profile_value)

            def canonical(value: Any) -> Any:
                if isinstance(value, str):
                    candidate = MODEL_ID_ALIASES.get(value, value)
                    return candidate if candidate in self._models else value
                if isinstance(value, dict):
                    value = dict(value)
                    if isinstance(value.get("model"), str):
                        value["model"] = MODEL_ID_ALIASES.get(value["model"], value["model"])
                    return value
                return value

            specialists: dict[str, SpecialistSpec] = {}
            raw_specialists = raw_profile.get("specialists", {}) or {}
            if isinstance(raw_specialists, dict):
                for name, raw_specialist in raw_specialists.items():
                    if not isinstance(raw_specialist, dict):
                        continue
                    specialist: dict[str, Any] = {
                        str(key): value for key, value in raw_specialist.items()
                    }
                    model_value = specialist.get("model", "")
                    if isinstance(model_value, str):
                        specialist["model"] = MODEL_ID_ALIASES.get(model_value, model_value)
                    specialists[str(name)] = SpecialistSpec(**specialist)
            spec = ProfileSpec(
                recon=canonical(raw_profile["recon"]),
                implementer=canonical(raw_profile["implementer"]),
                adversary=canonical(raw_profile["adversary"]),
                repairer=canonical(raw_profile["repairer"]),
                specialists=specialists,
            )
            profiles[profile_id] = spec
        self._profiles = profiles
        return profiles

    def load_workflows(
        self, path: str | Path | None = None
    ) -> dict[str, WorkflowSpec]:
        """Load workflow metadata from *path*."""
        target = Path(path) if path else self._config_path("workflows.yaml")
        if not target.exists():
            raise FileNotFoundError(f"workflows config not found: {target}")
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["workflows"] = raw
        data = yaml.safe_load(raw)
        workflows: dict[str, WorkflowSpec] = {}
        for wf_id, raw_wf in (data or {}).get("workflows", {}).items():
            workflows[wf_id] = WorkflowSpec(**raw_wf)
        self._workflows = workflows
        return workflows

    def load_providers(self, path: str | Path | None = None) -> dict[str, ProviderSpec]:
        """Load provider limits and deadlines when a provider catalog exists."""
        target = Path(path) if path else self._config_path("providers.yaml")
        if not target.exists():
            self._providers = {}
            self._raw_yaml.pop("providers", None)
            return self._providers
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["providers"] = raw
        data = yaml.safe_load(raw) or {}
        self._providers = {
            provider_id: ProviderSpec(**provider)
            for provider_id, provider in (data.get("providers") or {}).items()
        }
        return self._providers

    def load_fastpath(self, path: str | Path | None = None) -> dict:
        """Load optional fastpath service configuration into the registry hash."""
        target = Path(path) if path else self._config_path("fastpath.yaml")
        if not target.exists():
            self._raw_yaml.pop("fastpath", None)
            return {}
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["fastpath"] = raw
        data = yaml.safe_load(raw) or {}
        self._fastpath = FastpathConfigSpec(**(data.get("fastpath") or {}))
        return data

    def apply_discovered_catalog(self, entries: Sequence[Any], response_digest: str = "") -> None:
        """Merge authenticated provider metadata into the active generation.

        Discovery can add logical models, but it never grants roles,
        controller eligibility, mutation, or certification. Context/output
        limits come from the provider response when present.
        """
        seen_by_provider_endpoint: dict[tuple[str, str], set[str]] = {}
        for entry in entries:
            model_id = str(getattr(entry, "model_id"))
            provider_id = str(getattr(entry, "provider_id"))
            endpoint_id = str(getattr(entry, "endpoint_id", "openai"))
            context = getattr(entry, "context_tokens", None)
            output = getattr(entry, "max_output_tokens", None)
            provider = self._providers.get(provider_id)
            api_base = provider.endpoints.get(endpoint_id) if provider else None
            seen_by_provider_endpoint.setdefault((provider_id, endpoint_id), set()).add(model_id)
            if model_id not in self._models:
                endpoint_backend = "direct-anthropic" if endpoint_id == "anthropic" else "litellm"
                discovered_endpoint = ModelEndpointSpec(
                    backend=endpoint_backend,
                    provider_id=provider_id,
                    protocol=("anthropic-messages" if endpoint_backend == "direct-anthropic" else "openai-chat"),
                    upstream_model=model_id if endpoint_backend == "direct-anthropic" else None,
                    litellm_model=f"openai/{model_id}" if endpoint_backend == "litellm" else None,
                    api_base=api_base,
                    api_key_env=provider.api_key_env if provider else None,
                    auth=(ModelAuthSpec(type="bearer", header="Authorization", prefix="Bearer ")
                          if endpoint_backend == "direct-anthropic" else None),
                    max_context_tokens=context,
                    max_output_tokens=output,
                    tools=bool(getattr(entry, "capabilities", {}).get("tools", False)),
                    certified=False,
                    availability="public",
                )
                self._models[model_id] = ModelSpec(
                    display_name=str(getattr(entry, "display_name", None) or model_id),
                    provider_id=provider_id,
                    backend="litellm",
                    litellm_model=f"openai/{model_id}",
                    api_base=api_base,
                    api_key_env=provider.api_key_env if provider else None,
                    endpoints={endpoint_id: discovered_endpoint},
                    capabilities=ModelCapabilities(
                        tools=False, mutation=False, context_tokens=context,
                        max_context_tokens=context, max_output_tokens=output,
                        openai_chat_completions=True, streaming=True,
                        controller_eligible=False, read_tool_certified=False,
                        write_tool_certified=False,
                    ),
                    allowed_roles=set(), enabled=True,
                    catalog_source="discovered", availability="account-specific",
                )
                continue
            spec = self._models[model_id]
            caps = spec.capabilities.model_copy(update={
                "context_tokens": context if context is not None else spec.capabilities.context_tokens,
                "max_context_tokens": context if context is not None else spec.capabilities.max_context_tokens,
                "max_output_tokens": output if output is not None else spec.capabilities.max_output_tokens,
            })
            endpoint = spec.endpoints.get(endpoint_id)
            if endpoint is None:
                endpoint_backend = "direct-anthropic" if endpoint_id == "anthropic" else "litellm"
                endpoint = ModelEndpointSpec(
                    backend=endpoint_backend,
                    provider_id=provider_id,
                    protocol=("anthropic-messages" if endpoint_backend == "direct-anthropic" else "openai-chat"),
                    upstream_model=model_id if endpoint_backend == "direct-anthropic" else None,
                    litellm_model=f"openai/{model_id}" if endpoint_backend == "litellm" else None,
                    api_base=api_base,
                    api_key_env=provider.api_key_env if provider else None,
                    auth=(ModelAuthSpec(type="bearer", header="Authorization", prefix="Bearer ")
                          if endpoint_backend == "direct-anthropic" else None),
                    certified=False,
                )
            spec.endpoints[endpoint_id] = endpoint.model_copy(update={
                    "availability": "public",
                    "max_context_tokens": context if context is not None else endpoint.max_context_tokens,
                    "max_output_tokens": output if output is not None else endpoint.max_output_tokens,
                })
            self._models[model_id] = spec.model_copy(update={
                "capabilities": caps,
                "availability": "public" if spec.provider_id == provider_id else spec.availability,
            })
        # A model absent from a fresh provider catalog must not be selected for
        # a new call. Existing immutable bindings do not consult this registry
        # field and therefore remain pinned and inspectable.
        for model_id, spec in list(self._models.items()):
            if spec.provider_id is None:
                continue
            for endpoint_id, endpoint in list(spec.endpoints.items()):
                provider_id = endpoint.provider_id or spec.provider_id
                seen = seen_by_provider_endpoint.get((provider_id, endpoint_id))
                if seen is None or model_id not in seen:
                    spec.endpoints[endpoint_id] = endpoint.model_copy(update={"availability": "unavailable"})
        self._discovered_digest = response_digest

    def discover_provider_catalog(
        self,
        provider_id: str,
        *,
        endpoint_id: str = "openai",
        state: object | None = None,
        request_id: str | None = None,
        timeout: float = 30.0,
    ) -> tuple[str, int]:
        """Fetch and publish one authenticated provider catalog on demand.

        Discovery is explicit; startup and health checks never make inference
        or catalog calls.  The authenticated provider response supplies model
        context/output limits when it provides them.
        """
        import os

        from enhanced_router.provider_discovery import discover_openai_models, response_digest

        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"unknown provider '{provider_id}'")
        base_url = provider.endpoints.get(endpoint_id)
        if not base_url:
            raise ValueError(f"provider '{provider_id}' has no '{endpoint_id}' catalog endpoint")
        if not provider.api_key_env:
            raise ValueError(f"provider '{provider_id}' has no API key environment variable")
        api_key = os.environ.get(provider.api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"provider credential '{provider.api_key_env}' is not configured")
        entries = discover_openai_models(
            provider_id=provider_id,
            endpoint_id=endpoint_id,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        digest = response_digest(entries)
        self.apply_discovered_catalog(entries, digest)
        self._validate_cross_refs()
        if state is not None:
            persist = getattr(state, "store_provider_catalog_entries", None)
            if persist is not None:
                persist(
                    provider_id=provider_id,
                    response_digest=digest,
                    entries=entries,
                    request_id=request_id,
                )
        return digest, len(entries)

    # ------------------------------------------------------------------
    # Validation helpers (called after loading all sections)
    # ------------------------------------------------------------------

    def _validate_cross_refs(self) -> None:
        """Validate that profile references point to existing, enabled, compatible models.

        For each profile role reference, ensures:
        - model exists in registry
        - model is enabled
        - role is in model's allowed_roles
        - required capabilities satisfied (tools for all roles, mutation for implementer/repairer)
        - provider endpoint configured (for direct-anthropic) or litellm_model set (for litellm)
        - if auth required, api_key_env is set
        """
        for pid, profile in self._profiles.items():
            for role in ("recon", "implementer", "adversary", "repairer"):
                mid = profile.route_target(role).model
                if mid not in self._models:
                    raise ValueError(
                        f"Profile '{pid}' references unknown model '{mid}' "
                        f"for role '{role}'"
                    )
                spec = self._models[mid]

                if not spec.enabled:
                    raise ValueError(
                        f"Profile '{pid}' references disabled model '{mid}' "
                        f"for role '{role}'"
                    )

                if role not in spec.allowed_roles:
                    raise ValueError(
                        f"Profile '{pid}' references model '{mid}' for role '{role}' "
                        f"but model does not allow this role (allowed: {sorted(spec.allowed_roles)})"
                    )

                # Required capabilities per role
                if not spec.capabilities.tools:
                    raise ValueError(
                        f"Profile '{pid}' references model '{mid}' for role '{role}' "
                        f"but model has tools=false (required for all roles)"
                    )

                if role in ("implementer", "repairer") and spec.capabilities.write_tool_certified is not True:
                    raise ValueError(
                        f"Profile '{pid}' references model '{mid}' for role '{role}' "
                        f"but model is not write-tool certified (required for mutating roles)"
                    )

                # Backend-specific validation
                if spec.backend == "direct-anthropic":
                    if not spec.upstream_model:
                        raise ValueError(
                            f"Model '{mid}' (direct-anthropic) has no upstream_model"
                        )
                    if not spec.api_base and not spec.api_base_env:
                        raise ValueError(
                            f"Model '{mid}' (direct-anthropic) requires api_base or api_base_env"
                        )
                    # Credential presence is a runtime readiness concern. Keep
                    # structural registry validation usable for doctor/tests;
                    # endpoint selection and dispatch fail closed when the key
                    # is unavailable.
                elif spec.backend == "litellm":
                    if not spec.litellm_model:
                        raise ValueError(
                            f"Model '{mid}' (litellm) has no litellm_model"
                        )

            for specialist_id, specialist in profile.specialists.items():
                if not specialist_id.strip():
                    raise ValueError(f"Profile '{pid}' contains an empty specialist name")
                specialist_model = self._models.get(specialist.model)
                if specialist_model is None:
                    raise ValueError(
                        f"Profile '{pid}' specialist '{specialist_id}' references unknown model '{specialist.model}'"
                    )
                if not specialist_model.enabled:
                    raise ValueError(
                        f"Profile '{pid}' specialist '{specialist_id}' references disabled model '{specialist.model}'"
                    )
                for role in specialist.roles:
                    if role not in _VALID_ROLES:
                        raise ValueError(f"Profile '{pid}' specialist '{specialist_id}' has invalid role '{role}'")
                    if role not in specialist_model.allowed_roles:
                        raise ValueError(
                            f"Profile '{pid}' specialist '{specialist_id}' model '{specialist.model}' does not allow role '{role}'"
                        )

        if self._providers:
            for model_id, spec in self._models.items():
                if spec.provider_id and spec.provider_id not in self._providers:
                    raise ValueError(
                        f"Model '{model_id}' references unknown provider '{spec.provider_id}'"
                    )
                for endpoint_id, endpoint in spec.endpoints.items():
                    provider_id = endpoint.provider_id or spec.provider_id
                    if provider_id and provider_id not in self._providers:
                        raise ValueError(
                            f"Model '{model_id}' endpoint '{endpoint_id}' references unknown provider '{provider_id}'"
                        )
                    if endpoint.routing_owner == "provider" and endpoint.provider_id is None:
                        raise ValueError(
                            f"Model '{model_id}' endpoint '{endpoint_id}' must declare provider_id"
                        )

        if self._fastpath is not None:
            if self._fastpath.enabled:
                fastpath_model = self._models.get(self._fastpath.model_id)
                if fastpath_model is None:
                    raise ValueError(f"fastpath references unknown model '{self._fastpath.model_id}'")
                if not fastpath_model.enabled:
                    raise ValueError("fastpath model must be enabled when fastpath is enabled")
                if fastpath_model.capabilities.write_tool_certified:
                    raise ValueError("fastpath model cannot be write-tool certified")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> ModelSpec:
        """Return the ``ModelSpec`` for *model_id* or raise."""
        spec = self._models.get(MODEL_ID_ALIASES.get(model_id, model_id))
        if spec is None:
            raise KeyError(f"Unknown model: {model_id}")
        return spec

    def models_for_role(self, role: str) -> list[tuple[str, ModelSpec]]:
        """Return enabled models whose ``allowed_roles`` includes *role*."""
        result: list[tuple[str, ModelSpec]] = []
        for mid, spec in sorted(self._models.items()):
            if not spec.enabled:
                continue
            if role not in spec.allowed_roles:
                continue
            result.append((mid, spec))
        return result

    def controller_models(self) -> list[tuple[str, ModelSpec]]:
        """Return enabled models explicitly eligible to run the controller."""
        return [
            (model_id, spec)
            for model_id, spec in sorted(self._models.items())
            if spec.enabled and spec.capabilities.controller_eligible
        ]

    def get_profile(self, profile_id: str) -> ProfileSpec:
        """Return the ``ProfileSpec`` for *profile_id* or raise."""
        spec = self._profiles.get(profile_id)
        if spec is None:
            raise KeyError(f"Unknown profile: {profile_id}")
        return spec

    # ------------------------------------------------------------------
    # Deterministic recommendation (NOT LLM-based)
    # ------------------------------------------------------------------

    def recommend(
        self,
        role: str,
        constraints: RecommendationConstraints | None = None,
        profile_id: str | None = None,
        healthy_only: bool = False,
    ) -> list[RankedModel]:
        """Score and rank models for a role using a deterministic scoring function.

        Score components (each 0-10):
          1. **Role match** (0-10) — model allows this role.
          2. **Tool support** (0-5) — model has tools capability.
          3. **Context sufficiency** (0-5) — model context >= constraint.
          4. **Local/remote preference** (0-5) — matches local_only constraint.
          5. **Profile preference** (0-5) — model is the profile default for the role.
          6. **Health bonus** (0-5) — model is healthy (reachable, authenticated, compatible).

        Returns models sorted by total score descending, then by model_id for
        determinism.
        """
        if constraints is None:
            constraints = RecommendationConstraints(role=role)

        profile_default_model: str | None = None
        if profile_id:
            profile = self.get_profile(profile_id)
            profile_default_model = profile.route_target(role).model

        # Get state for health checks
        try:
            from enhanced_router.state import get_state
            state = get_state()
        except Exception:
            state = None

        candidates: list[tuple[str, ModelSpec, int, str]] = []
        for mid, spec in self.models_for_role(role):
            reasons: list[str] = []
            score = 0

            # 1. Role match — already filtered in models_for_role
            score += 10
            reasons.append("role-match")

            # 2. Tool support
            tool_score = 5 if spec.capabilities.tools else 0
            if tool_score == 0 and constraints.requires_tools:
                # Filter out if tools required but not available
                continue
            score += tool_score
            if tool_score > 0:
                reasons.append("tool-support")

            # 3. Context sufficiency
            context_score = 0
            if constraints.required_context_tokens is not None:
                known_context = spec.capabilities.max_context_tokens
                if known_context is None:
                    # Unknown is not evidence of sufficiency.  The provider
                    # catalog must supply a limit before a hard context
                    # requirement can be accepted.
                    continue
                if known_context >= constraints.required_context_tokens:
                    context_score = 5
                    reasons.append("context-sufficient")
                else:
                    # Partial score proportional to coverage
                    ratio = known_context / max(
                        constraints.required_context_tokens, 1
                    )
                    context_score = int(5 * ratio)
            else:
                context_score = 5  # no constraint
            score += context_score

            # 4. Local/remote preference
            local_score = 0
            if constraints.local_only:
                if spec.capabilities.local:
                    local_score = 5
                    reasons.append("local")
            else:
                local_score = 5  # no preference
            score += local_score

            # 5. Profile preference
            profile_score = 0
            if profile_default_model and mid == profile_default_model:
                profile_score = 5
                reasons.append("profile-pref")
            score += profile_score

            # 6. Health bonus (enabled + reachable + authenticated + compatible)
            health_score = 0
            if state is not None:
                health = state.get_model_health(mid)
                if health is not None:
                    if health.get("reachable") and health.get("authenticated") and health.get("compatible"):
                        health_score = 5
                        reasons.append("healthy")
                    elif health.get("status") == "degraded":
                        health_score = 2
                        reasons.append("degraded")
                    else:
                        reasons.append("unhealthy")
                else:
                    # No health record - untested
                    if healthy_only:
                        continue  # Skip untested models when healthy_only=True
                    reasons.append("untested")
            else:
                health_score = 5 if spec.enabled else 0
            score += health_score
            if not spec.enabled:
                reasons.append("disabled")

            candidates.append((mid, spec, score, ", ".join(reasons)))

        # Sort by score desc, then model_id asc for determinism
        candidates.sort(key=lambda c: (-c[2], c[0]))

        return [
            RankedModel(model_id=mid, score=score, reason=reason)
            for mid, _, score, reason in candidates
        ]

    # ------------------------------------------------------------------
    # Hashing and reloading
    # ------------------------------------------------------------------

    def registry_hash(self) -> str:
        """SHA-256 hash over combined YAML content of all three configs."""
        combined = ""
        for key in sorted(self._raw_yaml):
            combined += key + "=" + self._raw_yaml[key]
        combined += "discovered=" + self._discovered_digest
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def reload(self) -> None:
        """Re-load all YAML files from the config directory."""
        if self._config_dir is None:
            raise RuntimeError("No config directory set for reload")
        self.load_models()
        self.load_profiles()
        self.load_workflows()
        self.load_providers()
        self.load_fastpath()
        self._validate_cross_refs()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _config_path(self, filename: str) -> Path:
        if self._config_dir:
            return self._config_dir / filename
        # Default location relative to this file
        pkg_dir = Path(__file__).resolve().parent.parent.parent
        return pkg_dir / "config" / filename

    @property
    def models(self) -> dict[str, ModelSpec]:
        return self._models

    @property
    def profiles(self) -> dict[str, ProfileSpec]:
        return self._profiles

    def get_workflow(self, workflow_id: str) -> WorkflowSpec | None:
        """Return the ``WorkflowSpec`` for *workflow_id*, or None."""
        return self._workflows.get(workflow_id)

    @property
    def workflows(self) -> dict[str, WorkflowSpec]:
        return self._workflows

    @property
    def providers(self) -> dict[str, ProviderSpec]:
        return self._providers

    @property
    def fastpath(self) -> FastpathConfigSpec | None:
        return self._fastpath


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry_instance: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the module-level ModelRegistry singleton, initializing it if needed.

    Uses ``BRIGADE_CONFIG_DIR`` from the environment. Falls back to ``config/``
    relative to the repository root for development mode.
    """
    global _registry_instance
    if _registry_instance is None:
        from enhanced_router.base import BRIGADE_CONFIG_DIR

        cfg = BRIGADE_CONFIG_DIR
        # In development: fall back to repo-relative config/
        if not cfg.is_dir() or not any(cfg.glob("*.yaml")):
            import pathlib

            pkg_dir = pathlib.Path(__file__).resolve().parent.parent.parent
            repo_config = pkg_dir / "config"
            if repo_config.is_dir():
                cfg = repo_config
            else:
                raise RuntimeError(
                    f"No config directory found at {BRIGADE_CONFIG_DIR} or {repo_config}. "
                    "Set BRIGADE_CONFIG_DIR or run from the repository root."
                )

        _registry_instance = ModelRegistry(cfg)
        _registry_instance.load_models()
        # Keep bundled canonical entries available when an older user-owned
        # catalog predates logical FreeInference models or Claude passthrough
        # entries. Existing operator entries remain authoritative.
        bundled_dir = Path(__file__).resolve().parent.parent.parent / "config"
        if bundled_dir != cfg and (bundled_dir / "models.yaml").exists():
            bundled = ModelRegistry(bundled_dir)
            bundled.load_models()
            for model_id, spec in bundled.models.items():
                if spec.backend == "anthropic-passthrough" or spec.provider_id == "freeinference":
                    _registry_instance.models.setdefault(model_id, spec)
            for alias, canonical_id in MODEL_ID_ALIASES.items():
                if alias in _registry_instance.models and canonical_id in _registry_instance.models:
                    del _registry_instance.models[alias]
        _registry_instance.load_profiles()
        _registry_instance.load_workflows()
        _registry_instance.load_providers()
        _registry_instance.load_fastpath()
        _registry_instance._validate_cross_refs()
    return _registry_instance
