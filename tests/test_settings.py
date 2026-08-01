import json
from pathlib import Path


def test_bundled_settings_do_not_pin_dynamic_router_url():
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))

    # The active Claude Code model is the runtime controller.  The shared
    # profile must not pin it to a vendor-specific model.
    assert "model" not in settings
    assert "ANTHROPIC_BASE_URL" not in settings.get("env", {})
