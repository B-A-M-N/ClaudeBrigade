#!/usr/bin/env python3
"""UserPromptSubmit hook — deterministic task intake and epoch creation."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_bracket = os.environ.get("CLAUDE_BRIGADE_PYTHON")
if _bracket:
    _site = str(pathlib.Path(_bracket).parent / "site-packages")
    if _site not in sys.path:
        sys.path.insert(0, _site)

from workspace_fingerprint import fingerprint  # noqa: E402
from ledger_io import append_jsonl  # noqa: E402


def _fastpath_route(packet: dict) -> dict | None:
    """Ask the optional loopback fastpath without blocking task intake."""
    return _fastpath_route_request(packet, asynchronous=True, timeout_seconds=0.35)


def _fastpath_route_wait(packet: dict, *, timeout_seconds: float) -> dict | None:
    """Ask the fastpath and wait up to *timeout_seconds* for a real result.

    A "very small relevance budget" for the two-stage task-start flow: the
    router keeps running the inference call to completion in the background
    regardless of whether this HTTP call is still waiting, so a timeout here
    just means materialization goes ahead with deterministic defaults --
    the proposal still completes and becomes reviewable later (Phase 1's
    controller_route_review action) for any role that's still unbound.
    """
    return _fastpath_route_request(packet, asynchronous=False, timeout_seconds=timeout_seconds)


def _fastpath_route_request(packet: dict, *, asynchronous: bool, timeout_seconds: float) -> dict | None:
    import urllib.error
    import urllib.request

    base = os.environ.get("BRIGADE_ROUTER_URL", "http://127.0.0.1:8787").rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "X-Enhanced-Token": os.environ.get("ENHANCED_ROUTER_TOKEN", ""),
    }
    if asynchronous:
        headers["X-Brigade-Fastpath-Async"] = "1"
    request = urllib.request.Request(
        f"{base}/internal/fastpath/route",
        data=json.dumps(packet, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read(128_000).decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


# How long a fresh task's materialization waits for DiffusionGemma before
# falling back to deterministic profile defaults. Deliberately much shorter
# than fastpath.yaml's full inference timeout_seconds (5s default) -- this
# runs synchronously in a hook on every new task, so it must stay small
# enough not to make ordinary interactive latency noticeably worse; a
# proposal that misses this window is still fully usable afterward via the
# controller_route_review runnable action for any role still unbound.
_ROUTE_MATERIALIZATION_BUDGET_SECONDS = 0.9


def _route_overrides_from_fastpath_result(
    fastpath: dict | None,
) -> tuple[dict[str, str], str | None, bool]:
    """Parse a fastpath route response into materialize_task_epoch's
    route_overrides, plus (proposal_id, proposal_applied) for the hook's own
    additionalContext/disposition bookkeeping.

    Only a validation_status of "accepted_for_controller_review" is ever
    trusted for overrides -- the router already ran FastpathPolicyValidator
    against the registry (candidate-bounded, enabled, role-allowed,
    write-certified) before returning that status, so nothing here needs to
    re-validate the model/endpoint shape, only extract it defensively in
    case of a malformed/unexpected response.
    """
    route_overrides: dict[str, str] = {}
    proposal_id: str | None = None
    if not fastpath:
        return route_overrides, proposal_id, False

    status = fastpath.get("validation_status")
    if status == "accepted_for_controller_review":
        proposal_id = str(fastpath.get("proposal_id") or "") or None
        routes = fastpath.get("routes")
        if isinstance(routes, dict):
            for role, target in routes.items():
                if isinstance(target, dict) and isinstance(target.get("model"), str):
                    route_overrides[role] = target["model"]
        return route_overrides, proposal_id, bool(route_overrides)

    if status == "queued":
        # Still running server-side; missed this window -- not applied, but
        # still worth surfacing proposal_id so it's discoverable later.
        proposal_id = str(fastpath.get("proposal_id") or "") or None

    return route_overrides, proposal_id, False


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or data.get("run_id")
    if not run_id:
        return 0

    session = str(data.get("session_id", "unknown"))
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session_dir = cache / "sessions" / session
    epoch_file = session_dir / "active_epoch_id.txt"
    baseline_file = session_dir / "active_epoch_baseline.txt"
    cwd = pathlib.Path(str(data.get("cwd", "."))).resolve()
    prompt = str(data.get("prompt") or data.get("user_prompt") or "")

    current_fp = None
    try:
        current_fp = fingerprint(cwd, session_id=session)
    except Exception:
        pass

    try:
        from enhanced_router.policy import determine_minimum_workflow, extract_task_features
        from enhanced_router.state import get_state
    except ImportError:
        return 0

    state = get_state()
    features = extract_task_features(prompt, cwd)
    minimum_tier = determine_minimum_workflow(features)
    intake_id = f"intake_{uuid.uuid4().hex}"
    state.create_task_intake(
        intake_id=intake_id,
        run_id=run_id,
        session_id=session,
        prompt=prompt,
        request_kind=features.request_kind,
        repository_features={
            "languages": features.languages,
            "likely_files": features.explicit_files,
            "subsystems": features.subsystems,
            "estimated_context_tokens": features.estimated_context_tokens,
        },
        deterministic_signals=features.risk_signals,
        minimum_tier=minimum_tier,
    )

    active = state.get_active_epoch(run_id)
    if active is None:
        # The run row is the authoritative source: a shared router daemon can
        # serve multiple concurrent runs, each launched with its own
        # selection, and CLAUDE_BRIGADE_PROFILE only reaches the single
        # process tree that inherited it from whichever launcher invocation
        # started this particular session. The env var remains a fallback
        # for a run that was never given an explicit selection (e.g. created
        # directly rather than through the launcher's wizard).
        run_row = state.get_run(run_id)
        profile_id = (
            (run_row.get("inference_profile_id") if run_row else None)
            or os.environ.get("CLAUDE_BRIGADE_PROFILE")
            or "hybrid"
        )
        # Two-stage task start: plan (pure, pre-generates epoch_id) -> wait a
        # small bounded budget for DiffusionGemma -> materialize once, using
        # a validated accepted proposal if one arrived in time or
        # deterministic profile defaults otherwise. A proposal that misses
        # the window keeps running server-side and surfaces later via the
        # controller_route_review runnable action for any role still
        # unbound -- it is never discarded, only too late to shape the
        # *initial* materialization.
        plan = state.prepare_task_plan(
            run_id=run_id,
            session_id=session,
            cwd=str(cwd),
            workflow_id=minimum_tier,
            profile_id=profile_id,
            signals=features.risk_signals,
            prompt=prompt,
            intake_id=intake_id,
            minimum_tier=minimum_tier,
            baseline_fingerprint=current_fp,
            contract={
                "request_kind": features.request_kind,
                "required_capabilities": features.required_capabilities,
                "proposal_id": None,
            },
        )
        proposal_request_id = f"proposal_{uuid.uuid4().hex}"
        fastpath = _fastpath_route_wait({
            "intake_id": intake_id,
            "run_id": run_id,
            "epoch_id": plan["epoch_id"],
            "proposal_id": proposal_request_id,
            "task": prompt[:8_000],
            "repository": {
                "languages": features.languages,
                "likely_files": features.explicit_files,
                "subsystems": features.subsystems,
                "estimated_context_tokens": features.estimated_context_tokens,
            },
            "deterministic_minimum_tier": minimum_tier,
            "risk_signals": features.risk_signals,
            "required_capabilities": features.required_capabilities,
        }, timeout_seconds=_ROUTE_MATERIALIZATION_BUDGET_SECONDS)

        route_overrides, proposal_id, proposal_applied = _route_overrides_from_fastpath_result(fastpath)
        contract = state.materialize_task_epoch(plan, route_overrides=route_overrides)
        epoch_id = contract["epoch_id"]
        if proposal_applied and proposal_id:
            # Already resolved by materialization -- close the loop so it
            # doesn't also surface as a pending controller_route_review
            # action for a decision that's already been made.
            state.set_route_proposal_disposition(
                proposal_id, run_id, "accepted",
                reason="auto-applied at task materialization", epoch_id=epoch_id,
            )
        session_dir.mkdir(parents=True, exist_ok=True)
        epoch_file.write_text(epoch_id, encoding="utf-8")
        if current_fp:
            baseline_file.write_text(current_fp, encoding="utf-8")
        append_jsonl(session_dir / "ledger.jsonl", {
            "event": "EpochStart",
            "session_id": session,
            "epoch_id": epoch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_fingerprint": current_fp,
            "source": "user_prompt_submit",
        })
    else:
        epoch_id = str(active.get("epoch_id"))
        proposal_request_id = f"proposal_{uuid.uuid4().hex}"
        fastpath = _fastpath_route({
            "intake_id": intake_id,
            "run_id": run_id,
            "epoch_id": epoch_id,
            "proposal_id": proposal_request_id,
            "task": prompt[:8_000],
            "repository": {
                "languages": features.languages,
                "likely_files": features.explicit_files,
                "subsystems": features.subsystems,
                "estimated_context_tokens": features.estimated_context_tokens,
            },
            "deterministic_minimum_tier": minimum_tier,
            "risk_signals": features.risk_signals,
            "required_capabilities": features.required_capabilities,
        })
        proposal_id = None
        proposal_applied = False
        if fastpath and fastpath.get("validation_status") == "queued":
            proposal_id = str(fastpath.get("proposal_id") or "") or None

    context = {
        "intake_id": intake_id,
        "minimum_tier": minimum_tier,
        "signals": features.risk_signals,
        "likely_files": features.explicit_files,
        "proposal_id": proposal_id,
        "controller_disposition_required": bool(proposal_id) and not proposal_applied,
    }
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "BRIGADE TASK INTAKE: " + json.dumps(context, separators=(",", ":")),
        }
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
