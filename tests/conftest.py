import sys
from pathlib import Path

# Ensure router and hooks directories are on sys.path for pytest
root_dir = Path(__file__).resolve().parents[1]
router_dir = root_dir / "router"
hooks_dir = root_dir / "hooks"

if str(router_dir) not in sys.path:
    sys.path.insert(0, str(router_dir))

if str(hooks_dir) not in sys.path:
    sys.path.insert(0, str(hooks_dir))
