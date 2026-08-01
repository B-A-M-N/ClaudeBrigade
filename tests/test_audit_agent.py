"""Tests for hooks/audit_agent.py's fastpath merge-advice timeout.

Only the timeout-resolution logic is covered here -- the rest of
audit_agent.py's SubagentStop/StopFailure flow needs a full session/state/
worktree environment and isn't exercised at the unit level.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hooks.audit_agent import _fastpath_merge_advice


def _fake_changeset():
    return SimpleNamespace(
        changeset_id="cs-1", patch=b"--- a\n+++ b\n",
        changed_files=("a.py",), validation={"valid": True},
    )


def test_fastpath_merge_advice_uses_configured_timeout_not_a_fixed_guess():
    """P1-2: a fail/escalate decision here actually reclassifies an
    otherwise green changeset to yellow, so the wait must genuinely match
    the configured fastpath budget instead of a hardcoded 0.35s that almost
    always times out before real inference finishes."""
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        response = MagicMock()
        response.read.return_value = json.dumps({"decision": "pass"}).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    fake_registry = SimpleNamespace(
        resolve_fastpath=lambda sidecar_profile_id: SimpleNamespace(timeout_seconds=5.0),
    )
    fake_state = SimpleNamespace(get_run=lambda run_id: {"sidecar_profile_id": None})

    with patch("enhanced_router.registry.get_registry", return_value=fake_registry), \
         patch("enhanced_router.state.get_state", return_value=fake_state), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _fastpath_merge_advice(
            run_id="r1", epoch_id="ep-1", changeset=_fake_changeset(), overlap={},
        )
    assert result == "pass"
    assert captured["timeout"] == 5.5  # configured 5.0 + the 0.5s transport margin


def test_fastpath_merge_advice_falls_back_to_default_timeout_when_config_unavailable():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        response = MagicMock()
        response.read.return_value = json.dumps({"decision": "escalate"}).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    with patch("enhanced_router.registry.get_registry", side_effect=RuntimeError("no registry")), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _fastpath_merge_advice(
            run_id="r1", epoch_id="ep-1", changeset=_fake_changeset(), overlap={},
        )
    assert result == "escalate"
    assert captured["timeout"] == 0.35


def test_fastpath_merge_advice_returns_none_on_timeout():
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    fake_registry = SimpleNamespace(
        resolve_fastpath=lambda sidecar_profile_id: SimpleNamespace(timeout_seconds=5.0),
    )
    fake_state = SimpleNamespace(get_run=lambda run_id: None)

    with patch("enhanced_router.registry.get_registry", return_value=fake_registry), \
         patch("enhanced_router.state.get_state", return_value=fake_state), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _fastpath_merge_advice(
            run_id="r1", epoch_id="ep-1", changeset=_fake_changeset(), overlap={},
        )
    assert result is None
