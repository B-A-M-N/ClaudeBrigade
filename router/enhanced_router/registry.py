"""ModelRegistry — loads YAML config and provides lookup + deterministic ranking."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import yaml

from enhanced_router.config_models import (
    ModelAuthSpec,
    ModelCapabilities,
    ModelEndpointSpec,
    EndpointPolicySpec,
    FastpathConfigSpec,
    LaunchPresetSpec,
    ModelSpec,
    ProfileSpec,
    ProviderSpec,
    RecommendationConstraints,
    RankedModel,
    SidecarProfileSpec,
    SidecarSpec,
    SpecialistSpec,
    WorkflowSpec,
)

_VALID_ROLES = frozenset(("recon", "implementer", "adversary", "repairer"))

# Used only when RecommendationConstraints.prefer_low_cost is set (recommend()'s
# cost-preference scoring component); free/cheap tiers score higher.
_COST_CLASS_SCORE = {"free": 5, "low": 3, "standard": 1, "high": 0}


@dataclass(frozen=True)
class ConfigDiagnostic:
    """One independently-failing validation section, from
    ModelRegistry.collect_config_diagnostics(). 'section' names which
    config concern failed (e.g. 'profiles', 'sidecars'); 'message' is the
    same text _validate_cross_refs would have raised for that section."""

    section: str
    message: str


def _number_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _credential_available(key_name: str | None) -> bool:
    if not key_name:
        return True
    try:
        from enhanced_router.credential_store import resolve_loaded

        if resolve_loaded(key_name):
            return True
    except Exception:
        pass
    return bool(os.environ.get(key_name))

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
        self._sidecars: dict[str, SidecarSpec] = {}
        self._fastpath: FastpathConfigSpec | None = None
        self._sidecar_profiles: dict[str, SidecarProfileSpec] = {}
        self._launch_presets: dict[str, LaunchPresetSpec] = {}
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
        discovered_target = self._config_path("discovered_models.yaml")
        if discovered_target.exists():
            discovered_raw = discovered_target.read_text(encoding="utf-8")
            self._raw_yaml["discovered_models"] = discovered_raw
            discovered_data = yaml.safe_load(discovered_raw) or {}
            merged = dict(data or {})
            merged_models = dict(merged.get("models") or {})
            for model_id, model in (discovered_data.get("models") or {}).items():
                merged_models.setdefault(model_id, model)
            merged["models"] = merged_models
            data = merged
        else:
            self._raw_yaml.pop("discovered_models", None)
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
                controller_model=canonical(raw_profile.get("controller_model"))
                if raw_profile.get("controller_model")
                else None,
                controller=canonical(raw_profile.get("controller"))
                if raw_profile.get("controller")
                else None,
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

    def load_sidecars(self, path: str | Path | None = None) -> dict[str, SidecarSpec]:
        """Load independent sidecar policies from the operator config."""
        target = Path(path) if path else self._config_path("sidecars.yaml")
        if not target.exists():
            self._sidecars = {}
            self._raw_yaml.pop("sidecars", None)
            return self._sidecars
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["sidecars"] = raw
        data = yaml.safe_load(raw) or {}
        self._sidecars = {
            str(sidecar_id): SidecarSpec(**sidecar)
            for sidecar_id, sidecar in (data.get("sidecars") or {}).items()
        }
        return self._sidecars

    def load_sidecar_profiles(self, path: str | Path | None = None) -> dict[str, SidecarProfileSpec]:
        """Load named sidecar-profile bundles from the operator config.

        Optional: a config directory that predates sidecar profiles simply
        has no sidecar_profiles.yaml, leaving this empty. ``resolve_sidecars``
        and ``resolve_fastpath`` fall back to the bare global sidecars.yaml /
        fastpath.yaml in that case.
        """
        target = Path(path) if path else self._config_path("sidecar_profiles.yaml")
        if not target.exists():
            self._sidecar_profiles = {}
            self._raw_yaml.pop("sidecar_profiles", None)
            return self._sidecar_profiles
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["sidecar_profiles"] = raw
        data = yaml.safe_load(raw) or {}
        self._sidecar_profiles = {
            str(profile_id): SidecarProfileSpec(**profile)
            for profile_id, profile in (data.get("sidecar_profiles") or {}).items()
        }
        return self._sidecar_profiles

    def load_launch_presets(self, path: str | Path | None = None) -> dict[str, LaunchPresetSpec]:
        """Load named launch presets (inference profile + sidecar profile pairs)."""
        target = Path(path) if path else self._config_path("launch_presets.yaml")
        if not target.exists():
            self._launch_presets = {}
            self._raw_yaml.pop("launch_presets", None)
            return self._launch_presets
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["launch_presets"] = raw
        data = yaml.safe_load(raw) or {}
        self._launch_presets = {
            str(preset_id): LaunchPresetSpec(**preset)
            for preset_id, preset in (data.get("launch_presets") or {}).items()
        }
        return self._launch_presets

    def apply_discovered_catalog(self, entries: Sequence[Any], response_digest: str = "") -> None:
        """Merge authenticated provider metadata into the active generation.

        Discovery can add logical models, but it never grants roles,
        controller eligibility, mutation, or certification. Context/output
        limits come from the provider response when present.
        """
        seen_by_provider_endpoint: dict[tuple[str, str], set[str]] = {}
        for entry in entries:
            model_id = str(getattr(entry, "model_id"))
            logical_model_id = str(getattr(entry, "logical_model_id", None) or model_id)
            provider_id = str(getattr(entry, "provider_id"))
            endpoint_id = str(getattr(entry, "endpoint_id", "openai"))
            context = getattr(entry, "context_tokens", None)
            output = getattr(entry, "max_output_tokens", None)
            provider = self._providers.get(provider_id)
            api_base = provider.endpoints.get(endpoint_id) if provider else None
            seen_models = seen_by_provider_endpoint.setdefault((provider_id, endpoint_id), set())
            seen_models.update({model_id, logical_model_id})
            if logical_model_id not in self._models:
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
                self._models[logical_model_id] = ModelSpec(
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
            spec = self._models[logical_model_id]
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
            self._models[logical_model_id] = spec.model_copy(update={
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

    def fetch_discovered_entries(
        self,
        provider_id: str,
        *,
        endpoint_id: str = "openai",
        timeout: float = 30.0,
    ) -> list[Any]:
        """Perform just the network fetch for one provider's catalog.

        Pure with respect to registry state -- makes no mutation, so callers
        (e.g. a bounded thread pool refreshing several providers at once) can
        run it concurrently across providers without any risk of a
        read-modify-write race on the shared discovered-catalog file. Pair
        with ``apply_discovered_entries`` to publish the result.
        """
        import os

        from enhanced_router.provider_discovery import discover_openai_models

        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"unknown provider '{provider_id}'")
        # An explicit discovery.url wins -- a provider's catalog listing
        # endpoint doesn't always match one of its configured inference
        # endpoints (e.g. a provider-wide /models route versus a
        # per-deployment chat completions URL).
        discovery_url = provider.discovery.get("url")
        base_url = discovery_url or provider.endpoints.get(endpoint_id)
        if not base_url:
            raise ValueError(f"provider '{provider_id}' has no '{endpoint_id}' catalog endpoint")
        if not provider.api_key_env:
            raise ValueError(f"provider '{provider_id}' has no API key environment variable")
        try:
            from enhanced_router.credential_store import resolve_loaded

            api_key = (resolve_loaded(provider.api_key_env) or "").strip()
        except Exception:
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
        if provider.discovery.get("free_only"):
            entries = [
                entry for entry in entries
                if self._catalog_entry_is_free(provider_id, entry)
            ]
            if provider.discovery.get("strip_free_suffix"):
                entries = [
                    replace(entry, model_id=entry.model_id.removesuffix(":free"))
                    for entry in entries
                ]
        # Provider catalogs frequently reuse the same upstream model ID. Keep
        # logical registry IDs provider-qualified while retaining the raw ID
        # for the actual OpenAI-compatible request.
        return [
            replace(entry, logical_model_id=f"{provider_id}/{entry.model_id}")
            for entry in entries
        ]

    def apply_discovered_entries(
        self,
        entries: Sequence[Any],
        *,
        provider_id: str,
        state: object | None = None,
        request_id: str | None = None,
    ) -> tuple[str, int]:
        """Publish a previously-fetched catalog to this registry.

        Mutates shared registry state (in-memory models, the persisted
        ``discovered_models.yaml`` file) -- callers refreshing multiple
        providers must serialize calls to this method even if the fetches
        that produced *entries* ran concurrently.
        """
        from enhanced_router.provider_discovery import response_digest

        digest = response_digest(entries)
        self.apply_discovered_catalog(entries, digest)
        self._persist_discovered_catalog(entries)
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
        entries = self.fetch_discovered_entries(
            provider_id, endpoint_id=endpoint_id, timeout=timeout,
        )
        return self.apply_discovered_entries(
            entries, provider_id=provider_id, state=state, request_id=request_id,
        )

    @staticmethod
    def _catalog_entry_is_free(provider_id: str, entry: Any) -> bool:
        """Apply provider-specific free-catalog rules without guessing price."""
        raw = getattr(entry, "raw", {})
        raw_id = str(raw.get("id", getattr(entry, "model_id", "")))
        if provider_id in {"kilocode", "kilo"}:
            if raw_id == "kilo-auto/free" or raw_id.endswith(":free"):
                return True
            pricing = raw.get("pricing")
            if isinstance(pricing, dict):
                return (
                    _number_value(pricing.get("input")) == 0
                    and _number_value(pricing.get("output")) == 0
                )
            return (
                _number_value(raw.get("input_price")) == 0
                and _number_value(raw.get("output_price")) == 0
            )
        if provider_id == "requesty":
            return _number_value(raw.get("input_price")) == 0 and _number_value(raw.get("output_price")) == 0
        pricing = raw.get("pricing")
        if isinstance(pricing, dict):
            return _number_value(pricing.get("input")) == 0 and _number_value(pricing.get("output")) == 0
        return False

    def _persist_discovered_catalog(self, entries: Sequence[Any]) -> None:
        """Persist provider-discovered model metadata without credentials."""
        if not entries:
            return
        target = self._config_path("discovered_models.yaml")
        existing: dict[str, Any] = {}
        if target.exists():
            existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        models = dict(existing.get("models") or {})
        for entry in entries:
            provider_id = str(getattr(entry, "provider_id"))
            endpoint_id = str(getattr(entry, "endpoint_id", "openai"))
            raw_model_id = str(getattr(entry, "model_id"))
            model_id = str(
                getattr(entry, "logical_model_id", None)
                or f"{provider_id}/{raw_model_id}"
            )
            provider = self._providers.get(provider_id)
            api_base = provider.endpoints.get(endpoint_id) if provider else None
            api_key_env = provider.api_key_env if provider else None
            # A prior session may have explicitly granted this model a role
            # (see config_cli.configure_inference's discovered-model grant
            # path). Re-discovery must not silently revoke that -- only the
            # catalog-sourced metadata (context/output limits, display name)
            # should refresh here.
            previous = models.get(model_id) or {}
            previous_caps = previous.get("capabilities") or {}
            models[model_id] = {
                "display_name": str(getattr(entry, "display_name", None) or model_id),
                "catalog_source": "discovered",
                "availability": "account-specific",
                "backend": "litellm",
                "litellm_model": f"openai/{raw_model_id}",
                "api_base": api_base,
                "api_key_env": api_key_env,
                "provider_id": provider_id,
                "capabilities": {
                    "tools": bool(previous_caps.get("tools", False)),
                    "mutation": bool(previous_caps.get("mutation", False)),
                    "read_tool_certified": bool(previous_caps.get("read_tool_certified", False)),
                    "write_tool_certified": bool(previous_caps.get("write_tool_certified", False)),
                    "context_tokens": getattr(entry, "context_tokens", None),
                    "max_output_tokens": getattr(entry, "max_output_tokens", None),
                    "openai_chat_completions": True,
                    "streaming": True,
                },
                "allowed_roles": list(previous.get("allowed_roles") or []),
                "endpoints": {
                    endpoint_id: {
                        "backend": "litellm",
                        "provider_id": provider_id,
                        "protocol": "openai-chat",
                        "litellm_model": f"openai/{raw_model_id}",
                        "api_base": api_base,
                        "api_key_env": api_key_env,
                        "max_context_tokens": getattr(entry, "context_tokens", None),
                        "max_output_tokens": getattr(entry, "max_output_tokens", None),
                        "availability": "account-specific",
                        "certified": False,
                    }
                },
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            yaml.safe_dump({"models": models}, sort_keys=False), encoding="utf-8"
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        self._raw_yaml["discovered_models"] = target.read_text(encoding="utf-8")

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

        Fail-closed: raises on the FIRST violation found, across all
        sections. This is what runtime startup (get_registry()) calls, and
        it must keep failing loudly and immediately on any invalid config.
        For the interactive wizard, which needs to stay usable even when
        part of an existing config is broken, see collect_config_diagnostics
        below -- it runs every section independently and collects each
        section's failure instead of aborting at the first one.
        """
        self._validate_profile_routes()
        self._validate_provider_refs()
        self._validate_fastpath_ref()
        self._validate_sidecar_refs()
        self._validate_workflow_sidecar_refs()
        self._validate_sidecar_profile_refs()
        self._validate_launch_preset_refs()

    def collect_config_diagnostics(self) -> list["ConfigDiagnostic"]:
        """Run every cross-ref validation section independently.

        Unlike _validate_cross_refs (fail-closed, stops at the first
        violation anywhere), this collects one diagnostic per FAILING
        SECTION and keeps going -- a broken sidecar reference doesn't hide
        a broken profile reference, and vice versa. This is what the
        interactive config wizard uses so an operator can see (and fix)
        everything wrong with their saved config in one pass, rather than
        having the wizard refuse to even start over a single bad reference
        somewhere in a file they weren't trying to touch. Only reports the
        first failure *within* each section, not every individual bad
        reference -- an exhaustive per-item scan would require rewriting
        each check below to not raise, which isn't worth the risk to the
        fail-closed runtime path both this and _validate_cross_refs share.
        """
        sections: list[tuple[str, Any]] = [
            ("profiles", self._validate_profile_routes),
            ("providers", self._validate_provider_refs),
            ("fastpath", self._validate_fastpath_ref),
            ("sidecars", self._validate_sidecar_refs),
            ("workflows", self._validate_workflow_sidecar_refs),
            ("sidecar_profiles", self._validate_sidecar_profile_refs),
            ("launch_presets", self._validate_launch_preset_refs),
        ]
        diagnostics: list[ConfigDiagnostic] = []
        for section, validator in sections:
            try:
                validator()
            except ValueError as exc:
                diagnostics.append(ConfigDiagnostic(section=section, message=str(exc)))
        return diagnostics

    def _validate_profile_routes(self) -> None:
        for pid, profile in self._profiles.items():
            controller_route = profile.controller_route()
            if controller_route is not None:
                controller_model_id = controller_route.model
                controller = self._models.get(controller_model_id)
                if controller is None:
                    raise ValueError(
                        f"Profile '{pid}' references unknown controller model "
                        f"'{controller_model_id}'"
                    )
                if not controller.enabled:
                    raise ValueError(
                        f"Profile '{pid}' references disabled controller model "
                        f"'{controller_model_id}'"
                    )
                if not (
                    controller.capabilities.controller_eligible
                    or controller.backend == "anthropic-passthrough"
                ):
                    raise ValueError(
                        f"Profile '{pid}' controller model '{controller_model_id}' "
                        "is not controller-compatible"
                    )
                for fallback_model_id in controller_route.fallback_models:
                    if fallback_model_id not in self._models:
                        raise ValueError(
                            f"Profile '{pid}' controller fallback model "
                            f"'{fallback_model_id}' is unknown"
                        )
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

                for fallback_model_id in profile.route_target(role).fallback_models:
                    fallback = self._models.get(fallback_model_id)
                    if fallback is None:
                        raise ValueError(
                            f"Profile '{pid}' fallback model '{fallback_model_id}' "
                            f"for role '{role}' is unknown"
                        )
                    if not fallback.enabled:
                        raise ValueError(
                            f"Profile '{pid}' fallback model '{fallback_model_id}' "
                            f"for role '{role}' is disabled"
                        )
                    if role not in fallback.allowed_roles:
                        raise ValueError(
                            f"Profile '{pid}' fallback model '{fallback_model_id}' "
                            f"does not allow role '{role}'"
                        )
                    if role in {"implementer", "repairer"} and fallback.capabilities.write_tool_certified is not True:
                        raise ValueError(
                            f"Profile '{pid}' fallback model '{fallback_model_id}' "
                            f"is not write-tool certified for role '{role}'"
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

    def _validate_provider_refs(self) -> None:
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

    def _validate_fastpath_ref(self) -> None:
        if self._fastpath is not None:
            if self._fastpath.enabled:
                fastpath_model = self._models.get(self._fastpath.model_id)
                if fastpath_model is None:
                    raise ValueError(f"fastpath references unknown model '{self._fastpath.model_id}'")
                if not fastpath_model.enabled:
                    raise ValueError("fastpath model must be enabled when fastpath is enabled")
                if fastpath_model.capabilities.write_tool_certified:
                    raise ValueError("fastpath model cannot be write-tool certified")

    def _validate_sidecar_refs(self) -> None:
        for sidecar_id, sidecar in self._sidecars.items():
            if not sidecar_id.strip():
                raise ValueError("sidecar IDs must not be empty")
            model = self._models.get(sidecar.model_id)
            if model is None:
                raise ValueError(
                    f"sidecar '{sidecar_id}' references unknown model '{sidecar.model_id}'"
                )
            if not model.enabled:
                raise ValueError(
                    f"sidecar '{sidecar_id}' references disabled model '{sidecar.model_id}'"
                )
            if model.backend == "anthropic-passthrough":
                raise ValueError(
                    f"sidecar '{sidecar_id}' cannot use the controller passthrough backend"
                )
            if sidecar.endpoint != "auto" and sidecar.endpoint not in model.endpoints:
                raise ValueError(
                    f"sidecar '{sidecar_id}' references unknown endpoint "
                    f"'{sidecar.endpoint}' on model '{sidecar.model_id}'"
                )

    def _validate_workflow_sidecar_refs(self) -> None:
        for workflow_id, workflow in self._workflows.items():
            for phase in workflow.phases:
                if phase.sidecar and phase.sidecar not in self._sidecars:
                    raise ValueError(
                        f"workflow '{workflow_id}' phase '{phase.id}' references "
                        f"unknown sidecar '{phase.sidecar}'"
                    )

    def _validate_sidecar_profile_refs(self) -> None:
        for profile_id, sidecar_profile in self._sidecar_profiles.items():
            for sidecar_id in sidecar_profile.sidecar_ids:
                if sidecar_id not in self._sidecars:
                    raise ValueError(
                        f"sidecar profile '{profile_id}' references unknown sidecar '{sidecar_id}'"
                    )
            if sidecar_profile.fastpath is not None and sidecar_profile.fastpath.enabled:
                fastpath_model = self._models.get(sidecar_profile.fastpath.model_id)
                if fastpath_model is None:
                    raise ValueError(
                        f"sidecar profile '{profile_id}' fastpath references "
                        f"unknown model '{sidecar_profile.fastpath.model_id}'"
                    )
                if fastpath_model.capabilities.write_tool_certified:
                    raise ValueError(
                        f"sidecar profile '{profile_id}' fastpath model cannot be write-tool certified"
                    )

    def _validate_launch_preset_refs(self) -> None:
        for preset_id, preset in self._launch_presets.items():
            if preset.inference_profile_id not in self._profiles:
                raise ValueError(
                    f"launch preset '{preset_id}' references unknown inference "
                    f"profile '{preset.inference_profile_id}'"
                )
            if preset.sidecar_profile_id is not None and preset.sidecar_profile_id not in self._sidecar_profiles:
                raise ValueError(
                    f"launch preset '{preset_id}' references unknown sidecar "
                    f"profile '{preset.sidecar_profile_id}'"
                )

    def profile_model_diversity_warnings(self) -> list[str]:
        """Flag profiles where recon/implementer/adversary/repairer all
        resolve to the same model.

        Not part of _validate_cross_refs / collect_config_diagnostics --
        same-model-for-every-role is a valid, loadable configuration (an
        operator may deliberately accept it for cost or licensing reasons),
        just one where the adversary role reviews the implementer role's
        output under the identical underlying model, which weakens
        adversarial review into self-review. Advisory only.
        """
        warnings: list[str] = []
        for profile_id, profile in self._profiles.items():
            models = {
                profile.route_target(role).model
                for role in ("recon", "implementer", "adversary", "repairer")
            }
            if len(models) == 1:
                warnings.append(
                    f"profile '{profile_id}' assigns the same model "
                    f"('{next(iter(models))}') to recon, implementer, adversary, "
                    f"and repairer -- adversarial review will not be independent "
                    f"of the implementation it's reviewing"
                )
        return warnings

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

    def referenced_model_ids(
        self,
        *,
        profile_ids: set[str] | None = None,
        sidecar_profile_ids: set[str] | None = None,
    ) -> set[str]:
        """Return every model_id actually assigned somewhere (profile/sidecar/fastpath).

        Provider discovery can add hundreds of models to the registry, but
        only a handful are ever actually assigned to a role, the controller,
        a sidecar, or fastpath. Registering all of them with the LiteLLM
        child process anyway means minutes of unnecessary model
        registration at every child startup for models nothing will ever
        call. This is used to filter the LiteLLM config down to what's
        actually in use; it is not a general-purpose "is this model good"
        check, so callers that want the full catalog (e.g. tests) simply
        don't apply it.

        ``profile_ids`` / ``sidecar_profile_ids`` further bound the result to
        specific saved profiles (e.g. the ones currently selected by runs in
        flight) instead of every profile the config has ever saved. ``None``
        (the default) for either scans everything, matching prior behavior
        -- required at cold daemon startup, before any run has made a
        selection to scope to.
        """
        ids: set[str] = set()
        profiles = (
            self._profiles.values()
            if profile_ids is None
            else (self._profiles[pid] for pid in profile_ids if pid in self._profiles)
        )
        for profile in profiles:
            controller_route = profile.controller_route()
            if controller_route is not None:
                ids.add(controller_route.model)
                ids.update(controller_route.fallback_models)
            for role in ("recon", "implementer", "adversary", "repairer"):
                target = getattr(profile, role)
                if isinstance(target, str):
                    ids.add(target)
                else:
                    ids.add(target.model)
                    ids.update(target.fallback_models)
            for specialist in profile.specialists.values():
                ids.add(specialist.model)

        if sidecar_profile_ids is None:
            for sidecar in self._sidecars.values():
                ids.add(sidecar.model_id)
            if self._fastpath is not None:
                ids.add(self._fastpath.model_id)
            for sidecar_profile in self._sidecar_profiles.values():
                if sidecar_profile.fastpath is not None:
                    ids.add(sidecar_profile.fastpath.model_id)
        else:
            for sp_id in sidecar_profile_ids:
                for sidecar in self.resolve_sidecars(sp_id).values():
                    ids.add(sidecar.model_id)
                fastpath = self.resolve_fastpath(sp_id)
                if fastpath is not None:
                    ids.add(fastpath.model_id)
        return ids

    def referenced_model_ids_for_active_runs(self, state: Any) -> set[str]:
        """Scope ``referenced_model_ids`` to what runs currently in flight selected.

        Falls back to the unscoped (every saved profile) result when there
        are no open runs yet, or when any open run launched without a
        profile selection (a bare ``--model`` override) -- in either case
        there is no bounded selection set to scope to safely.
        """
        selections = state.active_run_selections()
        if not selections:
            return self.referenced_model_ids()

        profile_ids: set[str] = set()
        sidecar_profile_ids: set[str] = set()
        profile_scope_bounded = True
        sidecar_scope_bounded = True
        for selection in selections:
            profile_id = selection.get("inference_profile_id")
            sidecar_profile_id = selection.get("sidecar_profile_id")
            if profile_id:
                profile_ids.add(profile_id)
            else:
                profile_scope_bounded = False
            if sidecar_profile_id:
                sidecar_profile_ids.add(sidecar_profile_id)
            else:
                sidecar_scope_bounded = False

        return self.referenced_model_ids(
            profile_ids=profile_ids if profile_scope_bounded else None,
            sidecar_profile_ids=sidecar_profile_ids if sidecar_scope_bounded else None,
        )

    def get_profile(self, profile_id: str) -> ProfileSpec:
        """Return the ``ProfileSpec`` for *profile_id* or raise."""
        spec = self._profiles.get(profile_id)
        if spec is None:
            raise KeyError(f"Unknown profile: {profile_id}")
        return spec

    def specialist_manifest(self, profile_id: str | None = None) -> dict[str, dict[str, str]]:
        from enhanced_router.agent_manifest import specialist_manifest

        return specialist_manifest(self, profile_id)

    def role_model_aliases(self, profile_id: str | None = None) -> dict[str, str]:
        from enhanced_router.agent_manifest import role_model_aliases

        return role_model_aliases(self, profile_id)

    def role_model_bindings(self, profile_id: str | None = None) -> dict[str, str]:
        from enhanced_router.agent_manifest import role_model_bindings

        return role_model_bindings(self, profile_id)

    def native_agent_name(self, model_id: str, role: str) -> str:
        from enhanced_router.agent_manifest import native_agent_name

        return native_agent_name(self, model_id, role)

    def _model_readiness_reasons(
        self, model_id: str, role: str, endpoint: str = "auto",
    ) -> list[str]:
        """Return concrete reasons why one model route cannot start."""
        try:
            spec = self.get_model(model_id)
        except KeyError:
            return ["model is not in the active registry"]
        reasons: list[str] = []
        if not spec.enabled:
            reasons.append("model is disabled")
        if role in {"implementer", "repairer"} and not spec.capabilities.write_tool_certified:
            reasons.append("write-tool certification is missing")
        if spec.provider_id and spec.provider_id not in self._providers:
            reasons.append(f"provider '{spec.provider_id}' is not configured")
        endpoint_spec = spec.endpoints.get(endpoint) if endpoint != "auto" else None
        candidates = [endpoint_spec] if endpoint_spec is not None else list(spec.endpoints.values())
        if not candidates:
            candidates = [spec]
        configured_transport = False
        for candidate in candidates:
            provider_id = getattr(candidate, "provider_id", None) or spec.provider_id
            provider = self._providers.get(provider_id) if provider_id else None
            backend = getattr(candidate, "backend", spec.backend)
            api_base = getattr(candidate, "api_base", None) or spec.api_base
            api_base_env = getattr(candidate, "api_base_env", None) or spec.api_base_env
            api_key_env = getattr(candidate, "api_key_env", None) or spec.api_key_env
            if provider is not None:
                api_base = api_base or provider.endpoints.get(endpoint)
                api_key_env = api_key_env or provider.api_key_env
            if backend == "direct-anthropic" and not (
                api_base or (api_base_env and os.environ.get(api_base_env))
            ):
                continue
            if api_key_env and not _credential_available(api_key_env):
                continue
            if backend == "litellm" and not (
                getattr(candidate, "litellm_model", None) or spec.litellm_model
            ):
                continue
            configured_transport = True
            break
        if not configured_transport:
            reasons.append("no configured endpoint with available credentials")
        return reasons

    def profile_readiness(self, profile_id: str) -> dict[str, Any]:
        """Report whether every role has a usable primary or fallback route."""
        profile = self.get_profile(profile_id)
        roles: dict[str, dict[str, Any]] = {}
        for role in _VALID_ROLES:
            target = profile.route_target(role)
            fallback_models = list(target.fallback_models)
            candidate_specs = [target.primary, *target.fallbacks]
            candidate_reasons = [
                self._model_readiness_reasons(spec.model, role, spec.endpoint)
                for spec in candidate_specs
            ]
            selected = next(
                (spec.model for spec, reasons in zip(candidate_specs, candidate_reasons) if not reasons),
                None,
            )
            roles[role] = {
                "model_id": target.model,
                "fallback_models": fallback_models,
                "selected_candidate": selected,
                "ready": selected is not None,
                "reasons": [] if selected is not None else candidate_reasons[0],
            }
        controller_reasons: list[str] = []
        controller_model = profile.controller_model
        if controller_model:
            controller_reasons = self._model_readiness_reasons(controller_model, "controller")
        controller_ready = not controller_model or not controller_reasons
        return {
            "profile_id": profile_id,
            "controller": {
                "model_id": controller_model,
                "ready": controller_ready,
                "reasons": controller_reasons,
            },
            "ready": controller_ready and all(item["ready"] for item in roles.values()),
            "roles": dict(sorted(roles.items())),
        }

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
          7. **Cost preference** (0-5) — only scored when
             ``constraints.prefer_low_cost`` is set; cheaper ``cost_class``
             tiers score higher. Opt-in and additive so it never changes the
             ranking for existing callers that don't ask for it.

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

            # 7. Cost preference (opt-in; additive, never applied by default)
            if constraints.prefer_low_cost:
                cost_score = _COST_CLASS_SCORE.get(spec.capabilities.cost_class, 0)
                score += cost_score
                if cost_score >= 3:
                    reasons.append(f"cost-{spec.capabilities.cost_class}")

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

    @property
    def config_dir(self) -> Path | None:
        return self._config_dir

    def reload(self) -> None:
        """Re-load all YAML files from the config directory."""
        if self._config_dir is None:
            raise RuntimeError("No config directory set for reload")
        self.load_models()
        self.load_profiles()
        self.load_workflows()
        self.load_providers()
        self.load_fastpath()
        self.load_sidecars()
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

    @property
    def sidecars(self) -> dict[str, SidecarSpec]:
        return self._sidecars

    def get_sidecar(self, sidecar_id: str) -> SidecarSpec:
        """Return one configured sidecar or raise a stable lookup error."""
        spec = self._sidecars.get(sidecar_id)
        if spec is None:
            raise KeyError(f"Unknown sidecar: {sidecar_id}")
        return spec

    @property
    def sidecar_profiles(self) -> dict[str, SidecarProfileSpec]:
        return self._sidecar_profiles

    def get_sidecar_profile(self, sidecar_profile_id: str) -> SidecarProfileSpec:
        spec = self._sidecar_profiles.get(sidecar_profile_id)
        if spec is None:
            raise KeyError(f"Unknown sidecar profile: {sidecar_profile_id}")
        return spec

    @property
    def launch_presets(self) -> dict[str, LaunchPresetSpec]:
        return self._launch_presets

    def get_launch_preset(self, launch_preset_id: str) -> LaunchPresetSpec:
        spec = self._launch_presets.get(launch_preset_id)
        if spec is None:
            raise KeyError(f"Unknown launch preset: {launch_preset_id}")
        return spec

    def resolve_sidecars(self, sidecar_profile_id: str | None) -> dict[str, SidecarSpec]:
        """Return the sidecars available for *sidecar_profile_id*.

        With no profile selected, every globally-defined sidecar remains
        available -- the pre-sidecar-profile behavior, unchanged for any
        config that hasn't adopted profiles. With a profile selected, only
        the sidecar IDs it lists are available, so a launch can bound which
        (possibly resource-costly) sidecars a run may invoke.
        """
        if sidecar_profile_id is None:
            return self._sidecars
        profile = self.get_sidecar_profile(sidecar_profile_id)
        return {
            sidecar_id: self._sidecars[sidecar_id]
            for sidecar_id in profile.sidecar_ids
            if sidecar_id in self._sidecars
        }

    def resolve_fastpath(self, sidecar_profile_id: str | None) -> FastpathConfigSpec | None:
        """Return the fastpath config for *sidecar_profile_id*.

        A sidecar profile's own ``fastpath`` takes precedence; falling back
        to the bare global fastpath.yaml singleton keeps configs that
        predate sidecar profiles working unchanged.
        """
        if sidecar_profile_id is not None:
            profile = self.get_sidecar_profile(sidecar_profile_id)
            if profile.fastpath is not None:
                return profile.fastpath
        return self._fastpath

    def sidecar_readiness(self, sidecar_id: str) -> dict[str, Any]:
        """Report static transport readiness without making an inference call."""
        try:
            sidecar = self.get_sidecar(sidecar_id)
        except KeyError:
            return {"sidecar_id": sidecar_id, "ready": False, "reasons": ["sidecar is not configured"]}
        reasons: list[str] = []
        if not sidecar.enabled:
            reasons.append("sidecar is disabled")
        try:
            model = self.get_model(sidecar.model_id)
        except KeyError:
            reasons.append("model is not in the active registry")
            return {"sidecar_id": sidecar_id, "model_id": sidecar.model_id, "ready": False, "reasons": reasons}
        if not model.enabled:
            reasons.append("model is disabled")
        candidates = (
            [model.endpoints[sidecar.endpoint]]
            if sidecar.endpoint != "auto" and sidecar.endpoint in model.endpoints
            else list(model.endpoints.values())
        )
        if not candidates:
            candidates = [model]
        configured_transport = False
        provider_ids: set[str] = set()
        for candidate in candidates:
            provider_id = getattr(candidate, "provider_id", None) or model.provider_id
            if provider_id:
                provider_ids.add(provider_id)
            provider = self._providers.get(provider_id) if provider_id else None
            api_base = getattr(candidate, "api_base", None) or model.api_base
            api_base_env = getattr(candidate, "api_base_env", None) or model.api_base_env
            key_env = getattr(candidate, "api_key_env", None) or model.api_key_env
            if provider is not None:
                key_env = key_env or provider.api_key_env
            if getattr(candidate, "backend", model.backend) == "direct-anthropic":
                if not (api_base or (api_base_env and os.environ.get(api_base_env))):
                    continue
            if key_env and not _credential_available(key_env):
                continue
            if getattr(candidate, "backend", model.backend) == "litellm":
                if not (getattr(candidate, "litellm_model", None) or model.litellm_model):
                    continue
            configured_transport = True
            break
        if not configured_transport:
            reasons.append("no configured endpoint with an available credential")
        return {
            "sidecar_id": sidecar_id,
            "model_id": sidecar.model_id,
            "mode": sidecar.mode,
            "provider_ids": sorted(provider_ids),
            "ready": not reasons,
            "reasons": reasons,
        }


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
        _registry_instance.load_sidecars()
        _registry_instance.load_sidecar_profiles()
        _registry_instance.load_launch_presets()
        _registry_instance._validate_cross_refs()
    return _registry_instance
