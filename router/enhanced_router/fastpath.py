"""Router-owned DiffusionGemma fastpath contracts.

Fastpath is deliberately an internal service boundary. It may recommend a
workflow or flag verification concerns, but it cannot mutate RouteState or
replace mandatory controller/adversary decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

class FastpathRouteCandidate(BaseModel):
    """One router-computed, pre-vetted routing option offered to DiffusionGemma.

    Built from ModelRegistry.recommend() plus live model-health state --
    never invented by the fastpath model. DiffusionGemma selects a
    candidate_id; it never emits a bare model string, so an unbound-role
    proposal can only ever reference a model the router already confirmed
    is enabled, role-compatible, and (for a mutating role) write-tool
    certified at packet-build time.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    role: Literal["recon", "implementer", "adversary", "repairer"]
    model_id: str
    endpoint: Literal["auto"] = "auto"
    cost_class: str = "standard"
    max_context_tokens: int | None = None
    write_certified: bool = False
    score: int = 0


class FastpathRoleProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # candidate_id selects one of the packet's offered FastpathRouteCandidate
    # entries; model/endpoint are resolved from that candidate server-side
    # by FastpathPolicyValidator, never trusted verbatim from the model's
    # own output. model/preferred_logical_model/slot remain accepted so an
    # older or misbehaving fastpath prompt degrades to "no route for this
    # role" (skipped below) instead of a hard parse failure.
    candidate_id: str | None = None
    model: str | None = None
    endpoint: Literal["auto"] = "auto"
    slot: Literal["fast", "work", "deep"] = "work"
    preferred_logical_model: str | None = None
    fanout: int = Field(default=1, ge=1, le=4)

    @property
    def logical_model(self) -> str | None:
        return self.model or self.preferred_logical_model


class FastpathRouteProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_tier: Literal["trivial", "normal", "cross-cutting", "high-risk"]
    recommended_roles: list[Literal["recon", "implementer", "adversary", "repairer"]]
    routes: dict[str, FastpathRoleProposal]
    parallel_groups: list[list[str]] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    confidence: float
    escalate_to_controller: bool

    @field_validator("confidence")
    @classmethod
    def confidence_is_bounded(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


class FastpathVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["pass", "fail", "escalate"]
    checks: dict[str, Literal["pass", "fail", "unknown"]]
    violations: list[str] = Field(default_factory=list)
    requires_full_adversary: bool
    confidence: float

    @field_validator("confidence")
    @classmethod
    def verification_confidence_is_bounded(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


def build_route_candidates(
    *, registry: Any, roles: list[str], profile_id: str | None = None, per_role: int = 3,
) -> list[FastpathRouteCandidate]:
    """Build the router-authoritative candidate set fastpath may choose from.

    Uses ModelRegistry.recommend() (role-match, tool support, context,
    health) rather than letting the fastpath model invent a model ID from
    nothing -- see FastpathRouteCandidate's docstring. Deterministic:
    recommend() already sorts by score desc then model_id asc, so the same
    registry state always produces the same candidate_id set.
    """
    candidates: list[FastpathRouteCandidate] = []
    for role in roles:
        ranked = registry.recommend(role, profile_id=profile_id, healthy_only=False)
        for ranked_model in ranked[:per_role]:
            spec = registry.get_model(ranked_model.model_id)
            candidates.append(FastpathRouteCandidate(
                candidate_id=f"cand-{role}-{ranked_model.model_id}",
                role=role,
                model_id=ranked_model.model_id,
                cost_class=spec.capabilities.cost_class,
                max_context_tokens=spec.capabilities.max_context_tokens,
                write_certified=bool(spec.capabilities.write_tool_certified),
                score=ranked_model.score,
            ))
    return candidates


@dataclass(frozen=True)
class FastpathLimits:
    timeout_seconds: float = 5.0
    max_packet_bytes: int = 64_000
    max_diff_lines: int = 400
    max_commands: int = 32
    max_claims: int = 64


class FastpathPacketBuilder:
    """Build bounded packets from deterministic facts, never worker prose."""

    def __init__(self, limits: FastpathLimits | None = None) -> None:
        self.limits = limits or FastpathLimits()

    def route_packet(
        self, *, task: str, repository: dict[str, Any], deterministic_minimum_tier: str,
        risk_signals: list[str], required_capabilities: list[str],
        candidates: list["FastpathRouteCandidate"], unbound_roles: list[str],
        current_routes: dict[str, str],
    ) -> dict[str, Any]:
        """Build the route packet actually sent to the fastpath model.

        candidates/unbound_roles/current_routes are the router-authoritative
        facts DiffusionGemma was previously never given (P0-1): it can only
        select a candidate_id from *candidates*, never invent a model
        string, and it can see which roles are already bound so it doesn't
        propose replacing one.
        """
        packet = {
            "task": task[:8_000],
            "repository": repository,
            "deterministic_minimum_tier": deterministic_minimum_tier,
            "risk_signals": risk_signals[:32],
            "required_capabilities": required_capabilities[:32],
            "candidates": [c.model_dump() for c in candidates[:64]],
            "unbound_roles": unbound_roles,
            "current_routes": current_routes,
        }
        return self._bounded(packet)

    def verification_packet(self, *, contract: dict[str, Any], deterministic_checks: dict[str, Any],
                            allowed_files: list[str], changed_files: list[str], diff: str,
                            commands: list[str], claims: list[str], findings: list[dict[str, Any]]) -> dict[str, Any]:
        lines = diff.splitlines()
        packet = {
            "contract": contract,
            "deterministic_checks": deterministic_checks,
            "allowed_files": allowed_files[:256],
            "changed_files": changed_files[:256],
            "diff": "\n".join(lines[: self.limits.max_diff_lines]),
            "commands": commands[: self.limits.max_commands],
            "claims": claims[: self.limits.max_claims],
            "findings": findings[: self.limits.max_claims],
        }
        if len(lines) > self.limits.max_diff_lines:
            packet["diff_truncated"] = True
        return self._bounded(packet)

    def _bounded(self, packet: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        if len(encoded.encode("utf-8")) <= self.limits.max_packet_bytes:
            return packet
        packet["truncation_marker"] = "fastpath packet exceeded byte limit"
        packet["task"] = str(packet.get("task", ""))[:2_000]
        packet["diff"] = str(packet.get("diff", ""))[:8_000]
        while len(json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) > self.limits.max_packet_bytes:
            for key in ("findings", "claims", "commands", "changed_files", "allowed_files"):
                values = packet.get(key)
                if isinstance(values, list) and values:
                    packet[key] = values[:-1]
                    break
            else:
                break
        return packet


class FastpathPolicyValidator:
    """Validate fastpath output against deterministic policy and registry state."""

    _tier_order = {"trivial": 0, "normal": 1, "cross-cutting": 2, "high-risk": 3}

    def validate_route(
        self,
        proposal: FastpathRouteProposal,
        *,
        minimum_tier: str,
        registry: Any,
        state: Any,
        configuration_hash: str,
        confidence_threshold: float = 0.88,
        candidates: list[FastpathRouteCandidate] | None = None,
    ) -> FastpathRouteProposal:
        if self._tier_order[proposal.workflow_tier] < self._tier_order.get(minimum_tier, 1):
            raise ValueError("fastpath attempted to lower the deterministic workflow tier")
        if proposal.confidence < confidence_threshold:
            raise ValueError("fastpath confidence is below policy threshold")
        required_roles = set(proposal.recommended_roles)
        if proposal.workflow_tier in {"cross-cutting", "high-risk"}:
            required_roles.update({"recon", "implementer", "adversary"})
        if not required_roles.issubset(proposal.routes):
            raise ValueError("fastpath proposal omits a mandatory role")

        by_id = {c.candidate_id: c for c in candidates} if candidates is not None else None

        for role, target in proposal.routes.items():
            if role not in {"recon", "implementer", "adversary", "repairer"}:
                raise ValueError(f"fastpath proposed unknown role '{role}'")
            if target.endpoint != "auto":
                raise ValueError("fastpath cannot select a physical endpoint")

            if by_id is not None:
                # Candidate-bounded mode (Phase 2): the model may only select
                # candidate_id values the router itself offered for this
                # role -- it never gets to supply a bare model string.
                if target.candidate_id is None:
                    continue
                candidate = by_id.get(target.candidate_id)
                if candidate is None or candidate.role != role:
                    raise ValueError(
                        f"fastpath selected candidate_id '{target.candidate_id}' which "
                        f"was not offered for role '{role}'"
                    )
                target.model = candidate.model_id
                target.endpoint = candidate.endpoint
                model_id = candidate.model_id
            else:
                # Legacy mode (no candidate set supplied): the model's own
                # model/preferred_logical_model string, checked against the
                # registry same as before Phase 2.
                model_id = target.logical_model
                if model_id is None:
                    continue

            spec = registry.get_model(model_id)
            if not spec.enabled or role not in spec.allowed_roles:
                raise ValueError(f"fastpath route is unavailable for role '{role}'")
            if role in {"implementer", "repairer"} and not spec.capabilities.write_tool_certified:
                raise ValueError(f"fastpath assigned a non-mutating model to '{role}'")
        return proposal

    def validate_verification(
        self,
        verification: FastpathVerification,
        *,
        packet: dict[str, Any],
        deterministic_failed: bool = False,
    ) -> FastpathVerification:
        """Validate advisory verification output before it reaches state."""
        if "truncation_marker" in packet or packet.get("diff_truncated"):
            raise ValueError("truncated fastpath evidence cannot produce a passing decision")
        if deterministic_failed and verification.decision == "pass":
            raise ValueError("fastpath cannot override a failed deterministic check")
        if any(value == "unknown" for value in verification.checks.values()) and verification.decision == "pass":
            raise ValueError("unknown deterministic checks require escalation")
        if any(value == "fail" for value in verification.checks.values()) and verification.decision == "pass":
            raise ValueError("fastpath cannot mark failed checks as pass")
        if verification.requires_full_adversary and verification.decision == "pass":
            raise ValueError("verification requiring a full adversary cannot pass")
        if packet.get("security_sensitive") or packet.get("persistence_changed"):
            if verification.decision == "pass":
                raise ValueError("security or persistence changes require escalation")
        return verification


class FastpathClient:
    """Bounded, credential-owning client for internal route/verify calls."""

    def __init__(self, *, api_base: str, model: str, api_key_env: str,
                 provider_id: str, endpoint_id: str | None = None,
                 limits: FastpathLimits | None = None,
                 system_prompts: dict[str, str] | None = None,
                 max_output_tokens: int = 1024) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.provider_id = provider_id
        self.endpoint_id = endpoint_id
        self.limits = limits or FastpathLimits()
        self.system_prompts = dict(system_prompts or {})
        self.max_output_tokens = max_output_tokens

    async def request(self, mode: Literal["route", "verify"], packet: dict[str, Any]) -> dict[str, Any]:
        prompt = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        messages: list[dict[str, str]] = []
        system = self.system_prompts.get(mode)
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        from enhanced_router.backends import post_openai_compatible_json

        payload = await post_openai_compatible_json(
            api_base=self.api_base,
            model=self.model,
            api_key_env=self.api_key_env,
            provider_id=self.provider_id,
            endpoint_id=self.endpoint_id,
            payload={
                "messages": messages,
                "max_tokens": self.max_output_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            request_id=f"fastpath:{mode}",
            extra_headers={"X-Brigade-Fastpath": mode},
            timeout_seconds=self.limits.timeout_seconds,
            # Route requests block a fresh task's initial materialization
            # (see hooks/user_prompt_submit.py's bounded wait), so they get
            # priority within the provider's existing concurrency cap --
            # queue ordering only, never extra capacity. Verify requests
            # aren't latency-critical the same way and stay FIFO.
            priority=(mode == "route"),
        )
        choices = payload.get("choices")
        content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices else None
        if not isinstance(content, str):
            raise ValueError("fastpath response did not contain JSON message content")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("fastpath response must be a JSON object")
        return parsed
