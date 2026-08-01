"""Small local CLI; all network operations are explicit."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
from pathlib import Path

import typer

from .sync import DEFAULT_CONFIG_PATH, build_config, fetch_models, live_probe, require_environment, write_config
from .harness import run_contract

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _config_path() -> Path:
    return Path(os.environ.get("FREEINFERENCE_LITELLM_CONFIG", str(DEFAULT_CONFIG_PATH)))


@app.command()
def init() -> None:
    """Create a local env file with a generated proxy key."""
    env_path = Path(os.environ.get("FREEINFERENCE_LITELLM_ENV", ".env"))
    if env_path.exists():
        raise typer.BadParameter(f"Refusing to overwrite existing {env_path}")
    free_key = typer.prompt("FreeInference API key", hide_input=True)
    local_key = "sk-local-" + secrets.token_urlsafe(24)
    env_path.write_text(
        "FREEINFERENCE_API_KEY=" + free_key.strip() + "\n"
        + "LITELLM_MASTER_KEY=" + local_key + "\n"
        + "FREEINFERENCE_BASE_URL=" + os.environ.get("FREEINFERENCE_BASE_URL", "https://freeinference.org/v1") + "\n",
        encoding="utf-8",
    )
    env_path.chmod(0o600)
    typer.echo(f"Wrote {env_path} with mode 0600")


@app.command()
def sync() -> None:
    """Import the models accessible to the configured BYOK key."""
    models = fetch_models(require_environment("FREEINFERENCE_API_KEY"))
    path = write_config(build_config(models), _config_path())
    typer.echo(f"Wrote {len(models)} models to {path}")


@app.command()
def run(
    host: str = typer.Option(os.environ.get("LITELLM_HOST", "127.0.0.1")),
    port: int = typer.Option(int(os.environ.get("LITELLM_PORT", "4000"))),
) -> None:
    """Run the pinned local LiteLLM proxy."""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise typer.BadParameter("The prototype only permits loopback binding")
    require_environment("FREEINFERENCE_API_KEY")
    require_environment("LITELLM_MASTER_KEY")
    subprocess.run(
        ["litellm", "--config", str(_config_path()), "--host", host, "--port", str(port)],
        check=True,
    )


@app.command()
def doctor(live: bool = typer.Option(False, "--live")) -> None:
    """Check model discovery; optionally run an explicit synthetic probe."""
    models = fetch_models(require_environment("FREEINFERENCE_API_KEY"))
    typer.echo(json.dumps({"status": "ok", "model_count": len(models)}, indent=2))
    if live:
        local_key = require_environment("LITELLM_MASTER_KEY")
        model_id = str(models[0]["id"])
        typer.echo(json.dumps(live_probe(model_id, local_key=local_key), indent=2))


@app.command(name="test")
def test_model(model: str) -> None:
    """Run the contract harness for one model through the local proxy."""
    local_key = require_environment("LITELLM_MASTER_KEY")
    report = run_contract(
        model,
        base_url=f"http://{os.environ.get('LITELLM_HOST', '127.0.0.1')}:{os.environ.get('LITELLM_PORT', '4000')}",
        local_key=local_key,
    )
    output = Path(os.environ.get("FREEINFERENCE_REPORT_DIR", "reports"))
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{model.replace('/', '_')}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote sanitized compatibility report to {path}")


@app.command()
def report() -> None:
    """Print the newest sanitized compatibility report."""
    reports = sorted(Path("reports").glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise typer.BadParameter("No reports found")
    typer.echo(reports[-1].read_text(encoding="utf-8"))
