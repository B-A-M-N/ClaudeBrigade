"""Tests for the UserPromptSubmit hook's two-stage task-start pieces.

Only the testable, extracted logic is covered here (route-override parsing,
sync-vs-async request framing) -- main() itself needs a full session/state/
ledger environment and is exercised indirectly through tests/test_e2e.py's
router-level fastpath coverage instead.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from hooks.user_prompt_submit import (
    _fastpath_route_request,
    _route_overrides_from_fastpath_result,
)


def test_route_overrides_none_result_applies_nothing():
    overrides, proposal_id, applied = _route_overrides_from_fastpath_result(None)
    assert overrides == {}
    assert proposal_id is None
    assert applied is False


def test_route_overrides_accepted_proposal_extracts_model_per_role():
    fastpath = {
        "validation_status": "accepted_for_controller_review",
        "proposal_id": "proposal-1",
        "routes": {
            "implementer": {"model": "longcat-2", "endpoint": "auto"},
            "recon": {"model": "glm-5.1", "endpoint": "auto"},
        },
    }
    overrides, proposal_id, applied = _route_overrides_from_fastpath_result(fastpath)
    assert overrides == {"implementer": "longcat-2", "recon": "glm-5.1"}
    assert proposal_id == "proposal-1"
    assert applied is True


def test_route_overrides_accepted_proposal_with_no_usable_routes_is_not_applied():
    """A skip-only proposal (every role's target has no model) validates but
    contributes nothing -- must not be reported as applied."""
    fastpath = {
        "validation_status": "accepted_for_controller_review",
        "proposal_id": "proposal-1",
        "routes": {"implementer": {"endpoint": "auto"}},
    }
    overrides, proposal_id, applied = _route_overrides_from_fastpath_result(fastpath)
    assert overrides == {}
    assert proposal_id == "proposal-1"
    assert applied is False


def test_route_overrides_queued_proposal_reports_id_but_does_not_apply():
    """Missed the materialization window -- surfaced for later review, but
    contributes no overrides to this materialization."""
    fastpath = {"validation_status": "queued", "proposal_id": "proposal-1"}
    overrides, proposal_id, applied = _route_overrides_from_fastpath_result(fastpath)
    assert overrides == {}
    assert proposal_id == "proposal-1"
    assert applied is False


def test_route_overrides_bypassed_proposal_applies_nothing():
    fastpath = {"validation_status": "bypassed", "validation_reason": "confidence too low"}
    overrides, proposal_id, applied = _route_overrides_from_fastpath_result(fastpath)
    assert overrides == {}
    assert proposal_id is None
    assert applied is False


def test_route_overrides_ignores_malformed_routes_shape():
    fastpath = {
        "validation_status": "accepted_for_controller_review",
        "proposal_id": "proposal-1",
        "routes": {"implementer": "not-an-object", "recon": {"model": 123}},
    }
    overrides, proposal_id, applied = _route_overrides_from_fastpath_result(fastpath)
    assert overrides == {}
    assert applied is False


def test_fastpath_route_request_sync_omits_async_header():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.headers)
        captured["timeout"] = timeout
        response = MagicMock()
        response.read.return_value = json.dumps({"validation_status": "queued"}).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _fastpath_route_request({"run_id": "r1"}, asynchronous=False, timeout_seconds=0.9)
    assert result == {"validation_status": "queued"}
    assert "X-brigade-fastpath-async" not in captured["headers"]
    assert captured["timeout"] == 0.9


def test_fastpath_route_request_async_sets_header():
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.headers)
        response = MagicMock()
        response.read.return_value = json.dumps({"validation_status": "queued"}).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _fastpath_route_request({"run_id": "r1"}, asynchronous=True, timeout_seconds=0.35)
    assert captured["headers"].get("X-brigade-fastpath-async") == "1"


def test_fastpath_route_request_returns_none_on_timeout():
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = _fastpath_route_request({"run_id": "r1"}, asynchronous=False, timeout_seconds=0.1)
    assert result is None
