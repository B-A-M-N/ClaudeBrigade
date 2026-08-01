#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time


_cache: dict = {"data": None, "mtime": 0.0}


def _cached_status() -> str:
    # Cache ~1 second — no disk I/O on every render
    now = time.monotonic()
    if _cache["data"] is not None and now < _cache.get("deadline", 0):
        return _cache["data"]

    status = _compute_status()
    _cache["data"] = status
    _cache["deadline"] = now + 1.0
    return status


def _compute_status() -> str:
    data = json.load(sys.stdin)
    model = ((data.get("model") or {}).get("display_name") or (data.get("model") or {}).get("id") or "Claude")
    pct = int(((data.get("context_window") or {}).get("used_percentage") or 0))
    cwd = pathlib.Path((data.get("workspace") or {}).get("current_dir") or data.get("cwd") or ".")
    session = str(data.get("session_id", "unknown"))

    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    active_dir = cache / "sessions" / session / "active"
    agents = []
    if active_dir.exists():
        for marker in active_dir.glob("*.json"):
            try:
                record = json.loads(marker.read_text())
                name = str(record.get("agent_type", "agent"))
                backing = str(record.get("pinned_backing_model") or record.get("resolved_model") or "?")
                agents.append(f"{name}:{backing}")
            except Exception:
                pass
    active = ",".join(sorted(agents)) if agents else "main"
    provider_summary = ""
    run_id = os.environ.get("CLAUDE_BRIGADE_RUN_ID") or str(data.get("run_id") or "")
    if run_id:
        try:
            sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "router"))
            from enhanced_router.state import get_state
            from enhanced_router.registry import get_registry
            state = get_state()
            session_binding = state.get_controller_binding(run_id, session)
            if session_binding:
                model = session_binding.get("registry_model_id") or model
            active_epoch = state.get_active_epoch(run_id)
            reservations = state.get_provider_reservations(active_only=True)
            if reservations:
                by_provider: dict[str, tuple[int, int]] = {}
                for reservation in reservations:
                    provider = str(reservation.get("provider_id") or "unknown")
                    active_count, queued_count = by_provider.get(provider, (0, 0))
                    if reservation.get("state") == "reserved":
                        active_count += 1
                    else:
                        queued_count += 1
                    by_provider[provider] = (active_count, queued_count)
                provider_summary = " | " + ",".join(
                    f"{provider} {active_count}/{get_registry().providers[provider].limits.max_active_agents}"
                    + (f" queued {queued_count}" if queued_count else "")
                    for provider, (active_count, queued_count) in sorted(by_provider.items())
                    if provider in get_registry().providers
                )
            pending_candidates = state.get_integration_candidates(
                run_id=run_id,
                epoch_id=str(active_epoch.get("epoch_id")) if active_epoch else None,
            )
            pending_merge = sum(
                1 for item in pending_candidates
                if item.get("disposition") in {"yellow", "red", "pending"}
            )
            if pending_merge:
                provider_summary += f" | controller-merge {pending_merge}"
        except Exception:
            provider_summary = ""

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(cwd), "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "detached"
    except Exception:
        branch = "no-git"

    return f"BRIGADE | controller: {model} | {active}{provider_summary} | {branch} | ctx {pct}%"


def main() -> int:
    print(_cached_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
