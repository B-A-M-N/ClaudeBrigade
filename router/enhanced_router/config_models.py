"""Pydantic models for ClaudeBrigade configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ModelCapabilities(BaseModel):
    """Capabilities for a given model."""

    tools: bool
    mutation: bool
    context_tokens: int
    reasoning: Literal["low", "medium", "high"]
    local: bool


class ModelSpec(BaseModel):
    """Full specification for a single model."""

    display_name: str
    backend: Literal["direct-anthropic", "litellm"]
    upstream_model: str | None = None
    litellm_model: str | None = None
    api_base: str | None = None
    api_base_env: str | None = None
    api_key_env: str | None = None
    capabilities: ModelCapabilities
    allowed_roles: set[str] = Field(default_factory=set)
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> "ModelSpec":
        if self.backend == "litellm" and not self.litellm_model:
            raise ValueError(
                "backend='litellm' requires a non-empty litellm_model field"
            )
        if self.backend == "direct-anthropic" and not self.upstream_model:
            raise ValueError(
                "backend='direct-anthropic' requires a non-empty upstream_model field"
            )
        return self


class ProfileSpec(BaseModel):
    """Profile mapping role names to model IDs.

    No ``controller`` field — only dynamically routed roles.
    """

    recon: str
    implementer: str
    adversary: str
    repairer: str

    @model_validator(mode="after")
    def _no_role_name_as_model_id(self) -> "ProfileSpec":
        """Reject profiles that use role names as model IDs.

        Profile values must reference models from models.yaml
        (e.g. 'longcat-2', 'qwen-local'), not role names themselves.
        """
        role_names = {"recon", "implementer", "adversary", "repairer"}
        for field_name in ("recon", "implementer", "adversary", "repairer"):
            value = getattr(self, field_name)
            if value in role_names:
                raise ValueError(
                    f"Profile field '{field_name}' uses role name '{value}' "
                    f"as model ID; must reference a model from models.yaml"
                )
        return self


class RecommendationConstraints(BaseModel):
    """Filters passed to ``ModelRegistry.recommend``."""

    role: str
    required_context_tokens: int | None = None
    local_only: bool = False
    requires_tools: bool = True


class RankedModel(BaseModel):
    """A single ranked result from ``recommend``."""

    model_id: str
    score: int
    reason: str