"""Normalize explicit compatibility reports into route certification evidence.

The FreeInference/LiteLLM harness deliberately lives in a separately usable
integration package.  This module is the narrow boundary between that
sanitized report and ClaudeBrigade's authoritative endpoint-certification
state.  It never runs inference and it never treats catalog discovery as
certification; callers must explicitly publish a completed report.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


HARNESS_VERSION = "fi-contract-v2"
PROTOCOL_VERSION = "openai-chat-anthropic-messages-v1"

# These are intentionally status-level capabilities.  A result is certified
# only when the exact harness check reports ``pass``; ``partial`` and
# ``not_observed`` remain failures for admission purposes.
_CAPABILITY_CHECKS: dict[str, tuple[str, ...]] = {
    "messages": ("openai_nonstream", "anthropic_messages"),
    "streaming": ("openai_stream",),
    "tools": ("tool_call_single", "tool_result_continuation"),
    "parallel_tools": ("tool_call_parallel",),
    "structured_output": ("structured_output",),
    "usage_accounting": ("usage_accounting",),
    "cancellation": ("cancellation",),
}


@dataclass(frozen=True)
class ContractCertification:
    """Sanitized, route-independent interpretation of a harness report."""

    model_id: str
    tested_at: str | None
    capabilities: dict[str, bool]
    checks: dict[str, str]
    evidence_digest: str
    harness_version: str
    protocol_version: str


def _status(value: Any) -> str:
    return value if isinstance(value, str) else "not_observed"


def _evidence_payload(report: Mapping[str, Any], checks: Mapping[str, str]) -> dict[str, Any]:
    """Return only non-secret contract facts used for the evidence digest."""
    return {
        "model": str(report.get("model", "")),
        "tested_at": report.get("tested_at"),
        "checks": dict(sorted(checks.items())),
        "harness_version": str(report.get("harness_version") or HARNESS_VERSION),
        "protocol_version": str(report.get("protocol_version") or PROTOCOL_VERSION),
    }


def normalize_contract_report(
    report: Mapping[str, Any],
    *,
    model_id: str | None = None,
) -> ContractCertification:
    """Validate and normalize a sanitized compatibility report.

    This accepts old reports that predate explicit version fields, but it
    rejects reports without a model identity or with non-object input.  It
    preserves every recognized check as a boolean certification capability;
    callers can therefore see exactly why an endpoint is not eligible.
    """
    if not isinstance(report, Mapping):
        raise ValueError("compatibility report must be an object")
    report_model = report.get("model")
    resolved_model = model_id or (str(report_model) if report_model else "")
    if not resolved_model:
        raise ValueError("compatibility report is missing model")
    if model_id is not None and report_model is not None and str(report_model) != model_id:
        raise ValueError(
            f"compatibility report model {report_model!r} does not match {model_id!r}"
        )

    checks: dict[str, str] = {}
    capabilities: dict[str, bool] = {}
    for capability, fields in _CAPABILITY_CHECKS.items():
        values = tuple(_status(report.get(field)) for field in fields)
        checks[capability] = "pass" if all(value == "pass" for value in values) else (
            "partial" if any(value == "partial" for value in values) else "fail"
        )
        capabilities[capability] = checks[capability] == "pass"

    payload = _evidence_payload(report, checks)
    evidence_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ContractCertification(
        model_id=resolved_model,
        tested_at=str(report["tested_at"]) if report.get("tested_at") is not None else None,
        capabilities=capabilities,
        checks=checks,
        evidence_digest=evidence_digest,
        harness_version=str(report.get("harness_version") or HARNESS_VERSION),
        protocol_version=str(report.get("protocol_version") or PROTOCOL_VERSION),
    )


def publish_contract_report(
    state: Any,
    *,
    provider_id: str,
    model_id: str,
    endpoint_id: str,
    configuration_hash: str,
    report: Mapping[str, Any],
    certification_id: str | None = None,
    litellm_version: str | None = None,
    expires_at: str | None = None,
) -> list[dict[str, Any]]:
    """Publish an explicitly supplied report through RouteState.

    ``state`` is intentionally protocol-shaped instead of importing the
    application singleton.  This keeps the operation usable by the CLI,
    MCP, tests, and offline certification tooling without creating a hidden
    network or global-state dependency.
    """
    normalized = normalize_contract_report(report, model_id=model_id)
    return state.record_endpoint_certifications(
        provider_id=provider_id,
        model_id=model_id,
        endpoint_id=endpoint_id,
        configuration_hash=configuration_hash,
        harness_version=normalized.harness_version,
        protocol_version=normalized.protocol_version,
        evidence_digest=normalized.evidence_digest,
        capabilities=normalized.capabilities,
        certification_id=certification_id,
        litellm_version=litellm_version,
        expires_at=expires_at,
    )
