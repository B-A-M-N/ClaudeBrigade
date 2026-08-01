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
    import urllib.error
    import urllib.request

    base = os.environ.get("BRIGADE_ROUTER_URL", "http://127.0.0.1:8787").rstrip("/")
    request = urllib.request.Request(
        f"{base}/internal/fastpath/route",
        data=json.dumps(packet, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Enhanced-Token": os.environ.get("ENHANCED_ROUTER_TOKEN", ""),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=0.35) as response:
            payload = json.loads(response.read(128_000).decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


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

    fastpath = _fastpath_route({
        "intake_id": intake_id,
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
    if fastpath:
        proposal_id = f"proposal_{uuid.uuid4().hex}"
        state.create_route_proposal(
            proposal_id=proposal_id,
            intake_id=intake_id,
            source="fastpath",
            parsed_proposal=fastpath,
            validation_status=str(fastpath.get("validation_status", "pending")),
            validation_reason=str(fastpath.get("validation_reason", "")),
            fastpath_model_id=str(fastpath.get("fastpath_model_id", "diffusiongemma")),
            fastpath_endpoint_id=str(fastpath.get("fastpath_endpoint_id", "")) or None,
            confidence=float(fastpath.get("confidence", 0.0) or 0.0),
        )

    active = state.get_active_epoch(run_id)
    if active is None:
        profile_id = os.environ.get("BRIGADE_DEFAULT_PROFILE", "hybrid")
        contract = state.begin_task(
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
                "proposal_id": proposal_id,
            },
        )
        epoch_id = contract["epoch_id"]
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

    context = {
        "intake_id": intake_id,
        "minimum_tier": minimum_tier,
        "signals": features.risk_signals,
        "likely_files": features.explicit_files,
        "proposal_id": proposal_id,
        "controller_disposition_required": bool(proposal_id),
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
