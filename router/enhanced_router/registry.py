"""ModelRegistry — loads YAML config and provides lookup + deterministic ranking."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from enhanced_router.config_models import (
    ModelCapabilities,
    ModelSpec,
    ProfileSpec,
    RecommendationConstraints,
    RankedModel,
)


class ModelRegistry:
    """Central registry backed by YAML configuration files."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self._config_dir = Path(config_dir) if config_dir else None
        self._models: dict[str, ModelSpec] = {}
        self._profiles: dict[str, ProfileSpec] = {}
        self._workflows: dict[str, dict[str, Any]] = {}
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
            spec = ModelSpec(
                display_name=raw_model["display_name"],
                backend=raw_model["backend"],
                upstream_model=raw_model.get("upstream_model"),
                litellm_model=raw_model.get("litellm_model"),
                api_base=raw_model.get("api_base"),
                api_base_env=raw_model.get("api_base_env"),
                api_key_env=raw_model.get("api_key_env"),
                capabilities=capabilities,
                allowed_roles=set(raw_model.get("allowed_roles", [])),
                enabled=raw_model.get("enabled", True),
            )
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
        for profile_id, raw_profile in (data or {}).get("profiles", {}).items():
            spec = ProfileSpec(
                recon=raw_profile["recon"],
                implementer=raw_profile["implementer"],
                adversary=raw_profile["adversary"],
                repairer=raw_profile["repairer"],
            )
            profiles[profile_id] = spec
        self._profiles = profiles
        return profiles

    def load_workflows(
        self, path: str | Path | None = None
    ) -> dict[str, dict[str, Any]]:
        """Load workflow metadata from *path*."""
        target = Path(path) if path else self._config_path("workflows.yaml")
        if not target.exists():
            raise FileNotFoundError(f"workflows config not found: {target}")
        raw = target.read_text(encoding="utf-8")
        self._raw_yaml["workflows"] = raw
        data = yaml.safe_load(raw)
        workflows: dict[str, dict[str, Any]] = {}
        for wf_id, raw_wf in (data or {}).get("workflows", {}).items():
            workflows[wf_id] = dict(raw_wf)
        self._workflows = workflows
        return workflows

    # ------------------------------------------------------------------
    # Validation helpers (called after loading all sections)
    # ------------------------------------------------------------------

    def _validate_cross_refs(self) -> None:
        """Validate that profile references point to existing models."""
        for pid, profile in self._profiles.items():
            for role in ("recon", "implementer", "adversary", "repairer"):
                mid = getattr(profile, role)
                if mid not in self._models:
                    raise ValueError(
                        f"Profile '{pid}' references unknown model '{mid}' "
                        f"for role '{role}'"
                    )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> ModelSpec:
        """Return the ``ModelSpec`` for *model_id* or raise."""
        spec = self._models.get(model_id)
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
    ) -> list[RankedModel]:
        """Score and rank models for a role using a deterministic scoring function.

        Score components (each 0-10):
          1. **Role match** (0-10) — model allows this role.
          2. **Tool support** (0-5) — model has tools capability.
          3. **Context sufficiency** (0-5) — model context >= constraint.
          4. **Local/remote preference** (0-5) — matches local_only constraint.
          5. **Profile preference** (0-5) — model is the profile default for the role.
          6. **Health bonus** (0-5) — model is enabled.

        Returns models sorted by total score descending, then by model_id for
        determinism.
        """
        if constraints is None:
            constraints = RecommendationConstraints(role=role)

        profile_default_model: str | None = None
        if profile_id:
            profile = self.get_profile(profile_id)
            profile_default_model = getattr(profile, role)

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
                if spec.capabilities.context_tokens >= constraints.required_context_tokens:
                    context_score = 5
                    reasons.append("context-sufficient")
                else:
                    # Partial score proportional to coverage
                    ratio = spec.capabilities.context_tokens / max(
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

            # 6. Health (enabled) bonus
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
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def reload(self) -> None:
        """Re-load all YAML files from the config directory."""
        if self._config_dir is None:
            raise RuntimeError("No config directory set for reload")
        self.load_models()
        self.load_profiles()
        self.load_workflows()
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

    @property
    def workflows(self) -> dict[str, dict[str, Any]]:
        return self._workflows


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
        _registry_instance.load_profiles()
        _registry_instance.load_workflows()
    return _registry_instance
