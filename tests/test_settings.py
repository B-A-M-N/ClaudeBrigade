import json
from pathlib import Path


def test_bundled_settings_do_not_pin_dynamic_router_url():
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))

    assert settings.get("model") == "sonnet[1m]"
    assert "ANTHROPIC_BASE_URL" not in settings.get("env", {})
