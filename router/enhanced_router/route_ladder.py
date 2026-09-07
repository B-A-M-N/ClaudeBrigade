"""Small, shared helpers for exact provider/model/endpoint route ladders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from typing import Any


def candidate_dict(candidate: Any) -> dict[str, Any]:
    """Return a JSON-safe route candidate without losing provider identity."""
    if isinstance(candidate, dict):
        raw = candidate
    elif hasattr(candidate, "model_dump"):
        raw = candidate.model_dump(exclude_none=True)
    else:
        raw = {
            "model": getattr(candidate, "model", ""),
            "endpoint": getattr(candidate, "endpoint", "auto"),
            "provider_id": getattr(candidate, "provider_id", None),
        }
    return {
        "model": str(raw.get("model") or ""),
        "endpoint": str(raw.get("endpoint") or "auto"),
        **({"provider_id": str(raw["provider_id"])} if raw.get("provider_id") else {}),
    }


def target_candidates(target: Any) -> list[dict[str, Any]]:
    """Serialize a RouteTargetSpec in primary-first order."""
    if target is None:
        return []
    primary = getattr(target, "primary", target)
    fallbacks = getattr(target, "fallbacks", None)
    if fallbacks is None and isinstance(target, dict):
        fallbacks = target.get("fallbacks", [])
    return [candidate_dict(primary), *(candidate_dict(item) for item in (fallbacks or []))]


def dedupe_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate route identities while preserving order."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for candidate in candidates:
        normalized = candidate_dict(candidate)
        key = (
            normalized["model"],
            normalized["endpoint"],
            normalized.get("provider_id"),
        )
        if not normalized["model"] or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def route_key(route: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return the identity used for attempt caps and diagnostics."""
    normalized = candidate_dict(route)
    return normalized["model"], normalized["endpoint"], normalized.get("provider_id")


@dataclass(frozen=True, slots=True)
class RouteKey:
    """The immutable identity of one provider/model/endpoint candidate."""

    model_id: str
    endpoint_id: str
    provider_id: str | None = None

    @classmethod
    def from_candidate(cls, candidate: Any) -> "RouteKey":
        normalized = candidate_dict(candidate)
        return cls(
            model_id=normalized["model"],
            endpoint_id=normalized["endpoint"],
            provider_id=normalized.get("provider_id"),
        )

    def as_dict(self) -> dict[str, str]:
        result = {
            "model": self.model_id,
            "endpoint": self.endpoint_id,
        }
        if self.provider_id:
            result["provider_id"] = self.provider_id
        return result

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def route_digest(candidate: Any) -> str:
    """Return a stable digest for one exact route identity."""
    return RouteKey.from_candidate(candidate).digest()


def route_ladder_digest(candidates: Iterable[Any]) -> str:
    """Return a stable, order-sensitive digest for a route ladder."""
    normalized = [RouteKey.from_candidate(item).as_dict() for item in candidates]
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
