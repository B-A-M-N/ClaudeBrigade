"""Router-owned DiffusionGemma fastpath contracts.

Fastpath is deliberately an internal service boundary. It may recommend a
workflow or flag verification concerns, but it cannot mutate RouteState or
replace mandatory controller/adversary decisions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Some diffusion-serving stacks can emit an empty thought-channel envelope
# even when thinking is disabled (see FastpathConfigSpec.disable_thinking).
# Stripped defensively before json.loads rather than trusted to be absent.
_THOUGHT_ENVELOPE_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _strip_thought_envelope(content: str) -> str:
    """Strip a leading <think>...</think>-style envelope, if present."""
    return _THOUGHT_ENVELOPE_PATTERN.sub("", content, count=1)


class FastpathRoleProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ``candidate_id`` is the preferred shape: a compact alias (e.g. "i0")
    # into the per-role candidate list the router offered in the packet,
    # resolved back to a real model_id by FastpathPolicyValidator against
    # that same offered set -- the model never has to spell out (or invent)
    # a full model_id. ``model``/``preferred_logical_model`` remain accepted
    # for older local fastpath prompts that still propose a model directly;
    # ``endpoint`` stays fixed at "auto" either way, since the sidecar never
    # chooses a physical endpoint.
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


# Role -> compact alias prefix. "p" for repairer since "r" is taken by recon.
_ROLE_ALIAS_PREFIX = {"recon": "r", "implementer": "i", "adversary": "a", "repairer": "p"}


def build_route_candidates(
    registry: Any, roles: list[str], *, limit: int = 3,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """Build a compact, packet-safe candidate list per role plus its
    server-side alias -> model_id resolution map.

    DiffusionGemma only ever sees the alias (e.g. "i0"), never the raw
    model_id -- the router owns the mapping and FastpathPolicyValidator
    resolves the chosen alias back to a real model_id from this same
    offered set, so a route output can never introduce an arbitrary
    candidate the router didn't actually offer.
    """
    packet_candidates: dict[str, list[dict[str, Any]]] = {}
    candidate_map: dict[str, dict[str, str]] = {}
    for role in roles:
        prefix = _ROLE_ALIAS_PREFIX.get(role)
        if prefix is None:
            continue
        ranked = registry.recommend(role)[:limit]
        role_candidates = []
        role_map: dict[str, str] = {}
        for index, entry in enumerate(ranked):
            alias = f"{prefix}{index}"
            role_map[alias] = entry.model_id
            role_candidates.append({"candidate_id": alias, "score": entry.score})
        if role_candidates:
            packet_candidates[role] = role_candidates
            candidate_map[role] = role_map
    return packet_candidates, candidate_map


def route_template(minimum_tier: str, roles: list[str]) -> dict[str, Any]:
    """Fail-safe default route output: escalates and proposes nothing.

    DiffusionGemma must positively alter this state (select a real offered
    candidate_id, raise confidence, or clear escalate_to_controller with
    justification) to produce an accepted recommendation -- silence or an
    unparseable/invalid response never defaults to an accepted route.
    """
    return {
        "workflow_tier": minimum_tier,
        "recommended_roles": [],
        "routes": {role: {"candidate_id": None} for role in roles},
        "parallel_groups": [],
        "signals": [],
        "confidence": 0.0,
        "escalate_to_controller": True,
    }


def verification_template(check_names: list[str]) -> dict[str, Any]:
    """Fail-safe default verification output: never initializes to pass."""
    return {
        "decision": "escalate",
        "checks": {name: "unknown" for name in check_names},
        "violations": [],
        "requires_full_adversary": True,
        "confidence": 0.0,
    }


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

    def route_packet(self, *, task: str, repository: dict[str, Any], deterministic_minimum_tier: str,
                     risk_signals: list[str], required_capabilities: list[str], available_routes: list[str]) -> dict[str, Any]:
        packet = {
            "task": task[:8_000],
            "repository": repository,
            "deterministic_minimum_tier": deterministic_minimum_tier,
            "risk_signals": risk_signals[:32],
            "required_capabilities": required_capabilities[:32],
            "available_routes": available_routes[:64],
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
        candidate_map: dict[str, dict[str, str]] | None = None,
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

        for role, target in proposal.routes.items():
            if role not in {"recon", "implementer", "adversary", "repairer"}:
                raise ValueError(f"fastpath proposed unknown role '{role}'")
            if target.endpoint != "auto":
                raise ValueError("fastpath cannot select a physical endpoint")
            if target.candidate_id is not None:
                # The compact-alias path: the model never named a real
                # model_id, only an alias into the candidate set the router
                # itself offered for this role -- resolving it here means a
                # route output can never introduce a candidate that wasn't
                # actually offered, regardless of what the model returns.
                offered = (candidate_map or {}).get(role, {})
                model_id = offered.get(target.candidate_id)
                if model_id is None:
                    raise ValueError(
                        f"fastpath selected candidate '{target.candidate_id}' for "
                        f"role '{role}', which was not in the offered set"
                    )
            else:
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

    # Which schema each mode's strict response_format is generated from.
    _SCHEMA_MODELS: dict[str, type[BaseModel]] = {
        "route": FastpathRouteProposal,
        "verify": FastpathVerification,
    }

    def __init__(self, *, api_base: str, model: str, api_key_env: str,
                 provider_id: str, endpoint_id: str | None = None,
                 limits: FastpathLimits | None = None,
                 system_prompts: dict[str, str] | None = None,
                 max_output_tokens: int = 256,
                 strict_schema: bool = False,
                 disable_thinking: bool = False) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.provider_id = provider_id
        self.endpoint_id = endpoint_id
        self.limits = limits or FastpathLimits()
        self.system_prompts = dict(system_prompts or {})
        self.max_output_tokens = max_output_tokens
        self.strict_schema = strict_schema
        self.disable_thinking = disable_thinking

    async def request(self, mode: Literal["route", "verify"], packet: dict[str, Any]) -> dict[str, Any]:
        prompt = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        messages: list[dict[str, str]] = []
        system = self.system_prompts.get(mode)
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        from enhanced_router.backends import post_openai_compatible_json

        if self.strict_schema:
            schema_model = self._SCHEMA_MODELS[mode]
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"brigade_fastpath_{mode}",
                    "strict": True,
                    "schema": schema_model.model_json_schema(),
                },
            }
        else:
            response_format = {"type": "json_object"}

        request_payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "response_format": response_format,
            "stream": False,
        }
        if self.disable_thinking:
            request_payload["reasoning_effort"] = "none"

        payload = await post_openai_compatible_json(
            api_base=self.api_base,
            model=self.model,
            api_key_env=self.api_key_env,
            provider_id=self.provider_id,
            endpoint_id=self.endpoint_id,
            payload=request_payload,
            request_id=f"fastpath:{mode}",
            extra_headers={"X-Brigade-Fastpath": mode},
            timeout_seconds=self.limits.timeout_seconds,
        )
        choices = payload.get("choices")
        content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices else None
        if not isinstance(content, str):
            raise ValueError("fastpath response did not contain JSON message content")
        try:
            parsed = json.loads(_strip_thought_envelope(content))
        except json.JSONDecodeError as exc:
            raise ValueError(f"fastpath response was not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("fastpath response must be a JSON object")
        return parsed
