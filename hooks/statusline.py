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
                name = str(record.get("native_agent_name") or record.get("agent_type", "agent"))
                backing = str(record.get("pinned_backing_model") or record.get("resolved_model") or "?")
                agents.append(f"{name}:{backing}")
            except Exception:
                pass
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
            if active_epoch is None:
                raise RuntimeError("no active Brigade epoch")
            epoch_id = str(active_epoch["epoch_id"])
            executions = state.get_agent_executions(run_id, epoch_id=epoch_id)
            live_statuses = {"started", "running", "streaming", "verifying"}
            state_agents = []
            for item in executions:
                if item.get("status") not in live_statuses:
                    continue
                execution_kind = str(item.get("execution_kind") or "")
                if execution_kind in {"coprocessor_call", "sidecar_call"}:
                    prefix = "[C]"
                elif item.get("worker_kind") == "sidecar_agent":
                    prefix = "[N:S]"
                else:
                    prefix = "[N]"
                provider = str(item.get("provider_id") or "")
                model_id = str(item.get("model_id") or "?")
                endpoint = str(item.get("endpoint_id") or "auto")
                route = f"{provider}/{model_id}@{endpoint}" if provider else f"{model_id}@{endpoint}"
                state_agents.append(
                    f"{prefix} {item.get('native_agent_name') or item.get('worker_id') or item.get('role') or 'agent'}"
                    f" — {route} — {item.get('status') or 'unknown'}"
                )
            if state_agents:
                agents = state_agents
            recent_failures = [
                item for item in executions
                if item.get("status") in {"failed", "timeout", "cancelled"}
            ]
            if recent_failures:
                latest = recent_failures[0]
                provider_summary += " | last-fail " + ":".join(
                    str(value or "?")
                    for value in (
                        latest.get("role"),
                        latest.get("error_class") or latest.get("status"),
                    )
                )
            reservations = state.get_provider_reservations(
                run_id=run_id, epoch_id=epoch_id, active_only=True,
            )
            if reservations:
                by_provider: dict[str, tuple[int, int]] = {}
                registry = get_registry()
                for reservation in reservations:
                    provider = str(reservation.get("provider_id") or "unknown")
                    active_count, queued_count = by_provider.get(provider, (0, 0))
                    if reservation.get("state") == "reserved":
                        active_count += 1
                    else:
                        queued_count += 1
                    by_provider[provider] = (active_count, queued_count)
                provider_summary += " | " + ",".join(
                    f"{provider} {active_count}/{registry.providers[provider].limits.max_active_agents}"
                    + (f" queued {queued_count}" if queued_count else "")
                    for provider, (active_count, queued_count) in sorted(by_provider.items())
                    if provider in registry.providers
                )
            pending_candidates = state.get_integration_candidates(
                run_id=run_id,
                epoch_id=epoch_id,
            )
            pending_merge = sum(
                1 for item in pending_candidates
                if item.get("disposition") in {"yellow", "red", "pending"}
            )
            if pending_merge:
                provider_summary += f" | controller-merge {pending_merge}"
            from enhanced_router.presentation import compact_status, orchestration_snapshot
            provider_summary += " | " + compact_status(orchestration_snapshot(state, run_id, epoch_id))
        except Exception:
            provider_summary = ""

    # SQLite is authoritative when available. Recompute this after replacing
    # marker projections so native sidecars and coprocessors actually reach
    # the rendered status line.
    active = ",".join(sorted(agents)) if agents else "main"

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
