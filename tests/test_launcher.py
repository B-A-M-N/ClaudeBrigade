import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


def _setup_launcher_env(
    tmp_path: Path, config_dir_name: str = "claude-brigade", cache_dir_name: str = "claude-brigade"
) -> dict:
    """Set up a mock environment for launcher testing.

    Returns os.environ-like dict with paths pointing into tmp_path.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    config_base = tmp_path / "config"
    config_base.mkdir(parents=True, exist_ok=True)
    config_dir = config_base / config_dir_name
    config_dir.mkdir(parents=True, exist_ok=True)
    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    state_base = tmp_path / "state"
    state_base.mkdir(parents=True, exist_ok=True)
    cache_base = tmp_path / "cache"
    cache_base.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_base / cache_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_dir = state_base / "claude-brigade"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Router token
    (config_dir / "router.token").write_text("test-token")

    # providers.env
    providers = config_dir / "providers.env"
    providers.write_text("LONGCAT_API_KEY=test-key")
    providers.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # Fake python3 that intercepts commands
    fake_python = bin_dir / "python3"
    real_python = sys.executable
    fake_python.write_text(f"""#!/bin/bash
if [[ "$1" == "-m" && "$2" == "enhanced_router.agents_json" ]]; then
    echo '{{}}'
    exit 0
elif [[ "$1" == "-m" && "$2" == "uvicorn" ]]; then
    exit 0
elif [[ "$1" == "-" && "$2" == "8787" ]]; then
    # find_free_port mock: say 8788
    echo "8788"
    exit 0
elif [[ "$1" == "-m" && "$2" == "enhanced_router.state" ]]; then
    # Pre-registration mock
    exit 0
elif [[ "$1" == "-m" && "$2" == "enhanced_router.bootstrap_env" ]]; then
    # Exercise the real bootstrap_env module and its NUL-delimited contract.
    exec "{real_python}" "$@"
    exit 0
else
    exec "{real_python}" "$@"
fi
""")
    fake_python.chmod(0o755)

    # Fake curl for health checks
    fake_curl = bin_dir / "curl"
    fake_curl.write_text("""#!/bin/bash
# If port is 8787, fail (occupied); otherwise succeed
if [[ "$*" == *"127.0.0.1:8787"* ]]; then
    exit 1
fi
exit 0
""")
    fake_curl.chmod(0o755)

    # Fake claude — dumps args
    fake_claude = bin_dir / "claude"
    fake_claude.write_text("""#!/bin/bash
echo "$@"
""")
    fake_claude.chmod(0o755)

    # Fake command for the virtual environment Python
    venv_bin = app_dir / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(fake_python)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["CLAUDE_BRIGADE_APP_DIR"] = str(app_dir)
    env["CLAUDE_BRIGADE_PROFILE_DIR"] = str(config_dir)
    env["BRIGADE_CONFIG_DIR"] = str(config_dir)
    env["BRIGADE_STATE_DIR"] = str(state_dir)
    env["BRIGADE_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "router")
    env["XDG_CONFIG_HOME"] = str(config_base)
    env["XDG_CACHE_HOME"] = str(cache_base)
    env.pop("CLAUDE_ENHANCED_PORT", None)
    env.pop("CLAUDE_ENHANCED_CONFIG_DIR", None)
    env.pop("CLAUDE_ENHANCED_APP_DIR", None)

    return env


def test_launcher_durable_port_override(tmp_path: Path):
    env = _setup_launcher_env(tmp_path)
    launcher = Path(__file__).resolve().parents[1] / "bin" / "claude-brigade"
    result = subprocess.run([str(launcher), "hello"], env=env, capture_output=True, text=True)
    stdout = result.stdout
    stderr = result.stderr

    expected_settings = '{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"'
    assert "--settings" in stdout, f"stdout={stdout!r} stderr={stderr!r}"
    assert expected_settings in stdout, f"settings not found in stdout={stdout!r}"
    assert "CLAUDE_BRIGADE_PYTHON" in stdout
    assert "CLAUDE_BRIGADE_RUN_ID" in stdout
    assert "BRIGADE_CONFIG_DIR" in stdout
    assert "BRIGADE_STATE_DIR" in stdout
    assert "BRIGADE_CACHE_DIR" in stdout
    assert "BRIGADE_LITELLM_KEY" not in stdout, (
        "BRIGADE_LITELLM_KEY must NOT appear in Claude Code session settings"
    )

    # Verify the launcher script contains expected env vars
    launcher_text = launcher.read_text()
    assert "CLAUDE_BRIGADE_RUN_ID" in launcher_text
    assert "BRIGADE_LITELLM_KEY" in launcher_text

    # Settings include code-enablement env vars
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    assert settings.get("env", {}).get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") == "1"


def test_provider_keys_not_in_session_settings(tmp_path: Path):
    """Provider keys loaded from providers.env must NOT leak into SESSION_SETTINGS_JSON."""
    env = _setup_launcher_env(tmp_path)
    launcher = Path(__file__).resolve().parents[1] / "bin" / "claude-brigade"
    result = subprocess.run([str(launcher), "hello"], env=env, capture_output=True, text=True)
    stdout = result.stdout
    # Check that no provider key name appears in the JSON portion
    start = stdout.find('{"env":')
    if start >= 0:
        end = stdout.find("}", start) + 1
        settings_part = stdout[start:end]
        assert "LONGCAT_API_KEY" not in settings_part, (
            "Provider keys must not appear in session settings"
        )


def test_real_bootstrap_env_contract(tmp_path: Path):
    from enhanced_router.bootstrap_env import BootstrapResult, load_providers_env

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    assert load_providers_env(config_dir) == BootstrapResult({}, ())

    providers = config_dir / "providers.env"
    providers.write_text('FREEINFERENCE_API_KEY="hyi:a=b!"\n', encoding="utf-8")
    providers.chmod(stat.S_IRUSR | stat.S_IWUSR)
    result = load_providers_env(config_dir)
    assert result.provider_env == {"FREEINFERENCE_API_KEY": "hyi:a=b!"}
    assert result.provider_keys == ("FREEINFERENCE_API_KEY",)


def test_bootstrap_loads_freeinference_concurrency_override(tmp_path: Path):
    from enhanced_router.bootstrap_env import load_providers_env

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    providers = config_dir / "providers.env"
    providers.write_text(
        "FREEINFERENCE_MAX_CONCURRENCY=4\nFREEINFERENCE_API_KEY=key\n",
        encoding="utf-8",
    )
    providers.chmod(stat.S_IRUSR | stat.S_IWUSR)

    result = load_providers_env(config_dir)

    assert result.provider_env["FREEINFERENCE_MAX_CONCURRENCY"] == "4"
    assert "FREEINFERENCE_MAX_CONCURRENCY" in result.provider_keys


def test_bootstrap_accepts_freeinference_kit_endpoint_aliases(tmp_path: Path):
    from enhanced_router.bootstrap_env import load_providers_env

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    providers = config_dir / "providers.env"
    providers.write_text(
        "FREEINFERENCE_API_KEY=key\n"
        "FREEINFERENCE_OPENAI_BASE=https://freeinference.org/v1\n"
        "FREEINFERENCE_ANTHROPIC_BASE=https://freeinference.org/anthropic\n",
        encoding="utf-8",
    )
    providers.chmod(stat.S_IRUSR | stat.S_IWUSR)

    result = load_providers_env(config_dir)

    assert result.provider_env["FREEINFERENCE_OPENAI_BASE"] == "https://freeinference.org/v1"
    assert result.provider_env["FREEINFERENCE_ANTHROPIC_BASE"] == "https://freeinference.org/anthropic"


def test_bootstrap_accepts_legacy_provider_catalog_settings(tmp_path: Path):
    from enhanced_router.bootstrap_env import load_providers_env

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    providers = config_dir / "providers.env"
    providers.write_text(
        "OPENROUTER_API_KEY=key\n"
        "OPENROUTER_API_BASE=https://openrouter.ai/api/v1\n"
        "NVIDIA_API_KEY=key\n"
        "NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1\n"
        "FREEMODEL_CC_API_KEY=key\n"
        "FREEMODEL_CC_API_BASE=https://api.freemodel.dev/v1\n"
        "MODELSCOPE_API_KEY=key\n"
        "MODELSCOPE_API_BASE=https://api-inference.modelscope.cn/v1\n"
        "UNOROUTER_API_KEY=key\n"
        "UNOROUTER_API_BASE=https://unorouter.ai/v1\n"
        "LOGFLARE_API_KEY=key\n"
        "LOGFLARE_API_BASE=https://api.logflare.app\n",
        encoding="utf-8",
    )
    providers.chmod(stat.S_IRUSR | stat.S_IWUSR)

    result = load_providers_env(config_dir)

    assert result.provider_env["OPENROUTER_API_BASE"] == "https://openrouter.ai/api/v1"
    assert result.provider_env["NVIDIA_API_BASE"] == "https://integrate.api.nvidia.com/v1"
    # Unsupported legacy providers are accepted for migration but are not
    # exported into the router process until a Brigade provider definition
    # explicitly consumes them.
    assert "MODELSCOPE_API_KEY" not in result.provider_env


@pytest.mark.parametrize("contents", ["BROKEN\n", "UNSUPPORTED=value\n"])
def test_bootstrap_rejects_malformed_or_unsupported_provider_assignments(tmp_path: Path, contents: str):
    from enhanced_router.bootstrap_env import load_providers_env
    from enhanced_router.env_parser import EnvParseError

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    providers = config_dir / "providers.env"
    providers.write_text(contents, encoding="utf-8")
    providers.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(EnvParseError):
        load_providers_env(config_dir)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_launcher_durable_port_override(Path(td))
        test_provider_keys_not_in_session_settings(Path(td))
