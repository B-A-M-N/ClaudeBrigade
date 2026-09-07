from __future__ import annotations

from enhanced_router.certification import (
    HARNESS_VERSION,
    PROTOCOL_VERSION,
    normalize_contract_report,
    publish_contract_report,
)
from enhanced_router.state import RouteState


def _report(**overrides):
    report = {
        "model": "deepseek-v4-flash",
        "tested_at": "2026-08-03T00:00:00+00:00",
        "openai_nonstream": "pass",
        "openai_stream": "pass",
        "anthropic_messages": "pass",
        "tool_call_single": "pass",
        "tool_call_parallel": "pass",
        "tool_result_continuation": "pass",
        "structured_output": "pass",
        "usage_accounting": "pass",
        "cancellation": "pass",
        "harness_version": HARNESS_VERSION,
        "protocol_version": PROTOCOL_VERSION,
    }
    report.update(overrides)
    return report


def test_normalize_contract_report_requires_all_checks_for_capability():
    normalized = normalize_contract_report(
        _report(tool_result_continuation="not_observed")
    )

    assert normalized.capabilities["messages"] is True
    assert normalized.capabilities["streaming"] is True
    assert normalized.capabilities["tools"] is False
    assert normalized.capabilities["parallel_tools"] is True
    assert normalized.checks["tools"] == "fail"
    assert len(normalized.evidence_digest) == 64


def test_normalize_contract_report_rejects_route_identity_mismatch():
    try:
        normalize_contract_report(_report(), model_id="qwen3.6-35b")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("route identity mismatch was accepted")


def test_publish_contract_report_persists_route_specific_capabilities(tmp_path):
    state = RouteState(str(tmp_path / "state.db"))
    rows = publish_contract_report(
        state,
        provider_id="freeinference",
        model_id="deepseek-v4-flash",
        endpoint_id="openai",
        configuration_hash="registry-1",
        report=_report(),
        litellm_version="1.93.0",
    )

    assert {row["capability"] for row in rows} == {
        "cancellation",
        "messages",
        "parallel_tools",
        "streaming",
        "structured_output",
        "tools",
        "usage_accounting",
    }
    assert state.is_endpoint_certified(
        provider_id="freeinference",
        model_id="deepseek-v4-flash",
        endpoint_id="openai",
        configuration_hash="registry-1",
        capabilities=("messages", "streaming", "tools"),
    ) is True

