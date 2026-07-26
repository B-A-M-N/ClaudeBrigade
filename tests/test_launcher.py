import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

def test_launcher_durable_port_override(tmp_path: Path):
    # Setup the mock environment
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    config_base = tmp_path / "config"
    config_base.mkdir()
    config_dir = config_base / "claude-enhanced"
    config_dir.mkdir()
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    cache_base = tmp_path / "cache"
    cache_base.mkdir()
    cache_dir = cache_base / "claude-enhanced"
    cache_dir.mkdir()

    # Create fake settings.json with stale 8787
    settings_file = config_dir / "settings.json"
    settings_file.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"
        }
    }))

    # Create fake tokens
    (config_dir / "router.token").write_text("test-token")
    env_file = config_dir / "longcat.env"
    env_file.write_text("LONGCAT_API_KEY='test-key'")
    # chmod 600
    env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # Create fake python3 that intercepts commands to simulate environment
    fake_python = bin_dir / "python3"
    real_python = sys.executable
    real_python = sys.executable
    fake_python.write_text(f"""#!/bin/bash
if [[ "$1" == "-m" && "$2" == "enhanced_router.agents_json" ]]; then
    echo "{{}}"
    exit 0
elif [[ "$1" == "-m" && "$2" == "uvicorn" ]]; then
    exit 0
elif [[ "$1" == "-" && "$2" == "8787" ]]; then
    # find_free_port mock
    echo "8788"
    exit 0
else
    exec "{real_python}" "$@"
fi
""")
    fake_python.chmod(0o755)

    # Create fake curl to simulate health check behavior
    fake_curl = bin_dir / "curl"
    fake_curl.write_text("""#!/bin/bash
# If port is 8787, fail (simulate occupied by other process)
if [[ "$*" == *"127.0.0.1:8787"* ]]; then
    exit 1
fi
# Otherwise succeed
exit 0
""")
    fake_curl.chmod(0o755)

    # Create fake claude that dumps arguments
    fake_claude = bin_dir / "claude"
    fake_claude.write_text("""#!/bin/bash
echo "$@"
""")
    fake_claude.chmod(0o755)

    # Run the launcher
    launcher = Path(__file__).resolve().parents[1] / "bin" / "claude-enhanced"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["CLAUDE_ENHANCED_CONFIG_DIR"] = str(config_dir)
    env["CLAUDE_ENHANCED_APP_DIR"] = str(app_dir)
    env["XDG_CONFIG_HOME"] = str(config_base)
    env["XDG_CACHE_HOME"] = str(cache_base)
    
    # We remove CLAUDE_ENHANCED_PORT to let it default to 8787
    env.pop("CLAUDE_ENHANCED_PORT", None)

    # Mock python venv
    venv_bin = app_dir / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(fake_python)

    result = subprocess.run([str(launcher), "hello"], env=env, capture_output=True, text=True)
    stdout = result.stdout
    stderr = result.stderr
    
    expected_settings = '{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"'
    assert "--settings" in stdout
    assert expected_settings in stdout
    # Verify new Brigade env vars are included
    assert "CLAUDE_BRIGADE_PYTHON" in stdout
    assert "CLAUDE_BRIGADE_RUN_ID" in stdout
    assert "BRIGADE_CONFIG_DIR" in stdout
    assert "BRIGADE_STATE_DIR" in stdout
    assert "BRIGADE_CACHE_DIR" in stdout

    # Launcher exports CLAUDE_BRIGADE_RUN_ID in its script
    launcher = Path(__file__).resolve().parents[1] / "bin" / "claude-enhanced"
    launcher_text = launcher.read_text()
    assert "CLAUDE_BRIGADE_RUN_ID" in launcher_text

    # Settings include CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    assert settings.get("env", {}).get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") == "1"

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_launcher_durable_port_override(Path(td))
