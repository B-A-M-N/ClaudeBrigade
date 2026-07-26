#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys


def main() -> int:
    data = json.load(sys.stdin)
    model = ((data.get("model") or {}).get("display_name") or (data.get("model") or {}).get("id") or "Claude")
    pct = int(((data.get("context_window") or {}).get("used_percentage") or 0))
    cwd = pathlib.Path((data.get("workspace") or {}).get("current_dir") or data.get("cwd") or ".")
    session = str(data.get("session_id", "unknown"))

    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-enhanced"
    active_dir = cache / "sessions" / session / "active"
    roles = []
    if active_dir.exists():
        for marker in active_dir.glob("*.json"):
            try:
                roles.append(str(json.loads(marker.read_text()).get("agent_type", "agent")))
            except Exception:
                pass
    active = ",".join(sorted(roles)) if roles else "controller"

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(cwd), "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "detached"
    except Exception:
        branch = "no-git"

    print(f"ENHANCED | {model} | {active} | {branch} | ctx {pct}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
