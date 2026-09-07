"""Read-only Rich status view for the active ClaudeBrigade run."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _load_snapshot(run_id: str | None, epoch_id: str | None) -> dict[str, Any]:
    from enhanced_router.presentation import orchestration_snapshot
    from enhanced_router.state import get_state

    state = get_state()
    selected_run = state.get_run(run_id) if run_id else state.get_latest_active_run()
    if selected_run is None:
        return {"ready": False, "error": "No active ClaudeBrigade run found"}
    selected_run_id = str(selected_run["run_id"])
    active = state.get_active_epoch(selected_run_id)
    selected_epoch_id = epoch_id or (str(active["epoch_id"]) if active else "")
    if not selected_epoch_id:
        return {"ready": False, "run_id": selected_run_id, "error": "No active epoch found"}
    snapshot = orchestration_snapshot(state, selected_run_id, selected_epoch_id)
    snapshot["ready"] = True
    snapshot["workflow_id"] = (active or {}).get("workflow_id")
    return snapshot


def _plain(snapshot: dict[str, Any]) -> str:
    if not snapshot.get("ready"):
        return f"ClaudeBrigade: {snapshot.get('error', 'unavailable')}"
    counts = snapshot.get("phase_counts") or {}
    coverage = snapshot.get("coverage") or {}
    progress = snapshot.get("phase_progress") or {}
    packages = snapshot.get("packages") or []
    lines = [
        f"Workflow: {snapshot.get('workflow_id') or 'unknown'}",
        f"Phase: {progress.get('phase_id') or 'idle'} ({counts.get('completed', 0)}/{sum(counts.values())} phases)",
        f"Requirements: {coverage.get('covered', 0)}/{coverage.get('total', 0)} verified",
        "Packages:",
    ]
    for package in packages:
        lines.append(f"  {package.get('status', '?'):10} {package.get('objective') or package.get('id')}")
    if not packages:
        lines.append("  none")
    workers = snapshot.get("workers") or []
    lines.append(f"Workers: {len(workers)}")
    for worker in workers:
        lines.append(
            f"  {worker.get('status', '?'):10} {worker.get('worker') or worker.get('execution_id')}"
            f" [{worker.get('disposition') or 'awaiting adjudication'}]"
        )
    wait_reasons = snapshot.get("wait_reasons") or []
    blockers = snapshot.get("blockers") or []
    if wait_reasons:
        lines.append("Waiting:")
        lines.extend(f"  {item}" for item in wait_reasons)
    if blockers:
        lines.append("Blockers:")
        lines.extend(f"  {item}" for item in blockers)
    return "\n".join(lines)


def _rich(snapshot: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    if not snapshot.get("ready"):
        console.print(Panel(str(snapshot.get("error", "unavailable")), title="ClaudeBrigade", border_style="red"))
        return
    counts = snapshot.get("phase_counts") or {}
    coverage = snapshot.get("coverage") or {}
    progress = snapshot.get("phase_progress") or {}
    total_phases = sum(int(counts.get(key, 0)) for key in ("completed", "active", "pending", "failed"))
    console.print(
        Panel(
            f"[bold]{snapshot.get('workflow_id') or 'unknown'}[/bold]  "
            f"phase [cyan]{progress.get('phase_id') or 'idle'}[/cyan]  "
            f"{counts.get('completed', 0)}/{total_phases} phases  "
            f"requirements {coverage.get('covered', 0)}/{coverage.get('total', 0)}",
            title="ClaudeBrigade",
            border_style="cyan",
        )
    )

    packages = Table(title="Work packages")
    packages.add_column("Status", style="green")
    packages.add_column("Package")
    packages.add_column("Risk")
    for package in snapshot.get("packages") or []:
        packages.add_row(
            str(package.get("status") or "?"),
            str(package.get("objective") or package.get("id") or "?"),
            str(package.get("risk") or "normal"),
        )
    if not snapshot.get("packages"):
        packages.add_row("-", "none", "-")
    console.print(packages)

    workers = Table(title="Workers")
    workers.add_column("State")
    workers.add_column("Identity")
    workers.add_column("Package")
    workers.add_column("Disposition")
    for worker in snapshot.get("workers") or []:
        workers.add_row(
            str(worker.get("status") or "?"),
            str(worker.get("worker") or worker.get("execution_id") or "?"),
            str(worker.get("package_id") or "-"),
            str(worker.get("disposition") or "awaiting adjudication"),
        )
    if not snapshot.get("workers"):
        workers.add_row("-", "none", "-", "-")
    console.print(workers)

    wait_reasons = snapshot.get("wait_reasons") or []
    blockers = snapshot.get("blockers") or []
    if wait_reasons:
        console.print(Panel("\n".join(str(item) for item in wait_reasons), title="Waiting", border_style="yellow"))
    if blockers:
        console.print(Panel("\n".join(str(item) for item in blockers), title="Blockers", border_style="red"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claude-brigade status")
    parser.add_argument("--run-id")
    parser.add_argument("--epoch-id")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        snapshot = _load_snapshot(args.run_id or os.environ.get("CLAUDE_BRIGADE_RUN_ID"), args.epoch_id)
    except Exception as exc:
        snapshot = {"ready": False, "error": str(exc)}
    if args.as_json:
        print(json.dumps(snapshot, indent=2, sort_keys=True, default=str))
    else:
        try:
            _rich(snapshot)
        except ImportError:
            print(_plain(snapshot))
    return 0 if snapshot.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
