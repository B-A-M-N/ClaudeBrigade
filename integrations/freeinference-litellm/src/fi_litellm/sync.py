"""Discover FreeInference models and render a deterministic LiteLLM config."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
import yaml


DEFAULT_BASE_URL = "https://freeinference.org/v1"
DEFAULT_CONFIG_PATH = Path("config/generated.yaml")


def require_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def base_url() -> str:
    return os.environ.get("FREEINFERENCE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def fetch_models(api_key: str, *, endpoint: str | None = None) -> list[dict[str, Any]]:
    """Fetch and validate the user's accessible model catalog."""
    response = httpx.get(
        f"{(endpoint or base_url()).rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=30.0,
        follow_redirects=False,
    )
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("FreeInference /models returned no data array")

    valid: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if isinstance(model_id, str) and model_id.strip():
            valid.append(entry)
    if not valid:
        raise RuntimeError("FreeInference returned no usable model IDs")
    return valid


def build_config(models: list[dict[str, Any]], *, upstream: str | None = None) -> dict[str, Any]:
    """Build a deterministic config using explicit OpenAI provider names."""
    upstream_url = (upstream or base_url()).rstrip("/")
    model_ids = sorted({entry["id"].strip() for entry in models})
    return {
        "model_list": [
            {
                "model_name": model_id,
                "litellm_params": {
                    "model": f"openai/{model_id}",
                    "api_base": upstream_url,
                    "api_key": "os.environ/FREEINFERENCE_API_KEY",
                },
                "model_info": {
                    "source": "freeinference",
                    "upstream_model_id": model_id,
                },
            }
            for model_id in model_ids
        ],
        "litellm_settings": {
            "drop_params": False,
            "num_retries": 0,
            "request_timeout": 600,
        },
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
        "model_group_settings": {
            "forward_client_headers_to_llm_api": model_ids,
        },
    }


def write_config(config: dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False, default_flow_style=False).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def live_probe(model_id: str, *, base_url: str = "http://127.0.0.1:4000", local_key: str) -> dict[str, Any]:
    """Run one explicit synthetic request through the local proxy."""
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {local_key}",
            "Content-Type": "application/json",
            "X-Probe": "synthetic",
        },
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly: FI_PROXY_PROBE_OK"}],
            "max_tokens": 16,
            "temperature": 0,
        },
        timeout=60.0,
        follow_redirects=False,
    )
    response.raise_for_status()
    return {"model": model_id, "status": "ok", "request_id": response.headers.get("x-request-id")}
