from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from enhanced_router import config_cli


def _report() -> dict[str, str]:
    return {
        "model": "model-a",
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
    }


def test_certify_report_command_validates_route_and_publishes(tmp_path, monkeypatch, capsys):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    endpoint = SimpleNamespace(provider_id="provider-a")
    spec = SimpleNamespace(provider_id="provider-a", endpoints={"openai": endpoint})
    calls: dict = {}

    class Registry:
        def get_model(self, model_id):
            assert model_id == "model-a"
            return spec

        def registry_hash(self):
            return "registry-a"

    class State:
        def record_endpoint_certifications(self, **kwargs):
            calls.update(kwargs)
            return [{"capability": "messages", "status": "pass"}]

    monkeypatch.setattr(config_cli, "_load_registry", lambda _path: (Registry(), ()))
    monkeypatch.setattr(config_cli, "get_state", lambda: State())

    result = config_cli.certify_report_command(
        tmp_path,
        report_path=report_path,
        provider_id="provider-a",
        model_id="model-a",
        endpoint_id="openai",
    )

    assert result == 0
    assert calls["configuration_hash"] == "registry-a"
    assert calls["endpoint_id"] == "openai"
    assert calls["capabilities"]["messages"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "published"


def test_certify_report_command_rejects_wrong_provider(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    spec = SimpleNamespace(
        provider_id="provider-a",
        endpoints={"openai": SimpleNamespace(provider_id="provider-a")},
    )

    class Registry:
        def get_model(self, _model_id):
            return spec

        def registry_hash(self):
            return "registry-a"

    monkeypatch.setattr(config_cli, "_load_registry", lambda _path: (Registry(), ()))

    with pytest.raises(ValueError, match="belongs to provider"):
        config_cli.certify_report_command(
            tmp_path,
            report_path=report_path,
            provider_id="provider-b",
            model_id="model-a",
            endpoint_id="openai",
        )

