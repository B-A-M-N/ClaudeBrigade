from __future__ import annotations

import io
import json

from hooks import statusline


class _State:
    def get_controller_binding(self, run_id, session_id):
        return {"registry_model_id": "controller-model"}

    def get_active_epoch(self, run_id):
        return {"epoch_id": "ep-1"}

    def get_agent_executions(self, run_id, epoch_id=None):
        return [{
            "status": "running",
            "execution_kind": "native_agent",
            "worker_kind": "sidecar_agent",
            "worker_id": "minimax-builder",
            "role": "implementer",
            "model_id": "minimax-m3",
            "provider_id": "freeinference",
            "endpoint_id": "openai",
        }]

    def get_provider_reservations(self, **kwargs):
        return []

    def get_integration_candidates(self, **kwargs):
        return []


def test_statusline_uses_authoritative_native_sidecar_execution(monkeypatch):
    monkeypatch.setenv("CLAUDE_BRIGADE_RUN_ID", "run-1")
    monkeypatch.setattr(statusline.sys, "stdin", io.StringIO(json.dumps({
        "model": {"display_name": "Controller"},
        "session_id": "session-1",
        "workspace": {"current_dir": "/tmp"},
    })))

    import enhanced_router.state as state_module

    monkeypatch.setattr(state_module, "get_state", lambda: _State())

    rendered = statusline._compute_status()

    assert "[N:S] minimax-builder" in rendered
    assert "freeinference/minimax-m3@openai" in rendered
