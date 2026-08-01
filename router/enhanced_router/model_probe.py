"""Real, bounded compatibility probes for uncertified models.

Selecting a model in the interactive config wizard must never itself imply
that it can reliably serve as the controller or a mutating native role --
discovery supplies no evidence of tool-call support, streaming, or
tool-result continuation. This module makes one real, minimal round-trip
API call to the model's actual endpoint (bypassing the LiteLLM proxy
entirely, since it isn't running yet at config time) and checks whether the
model can accept a tool definition and either invoke it or respond
sensibly. That is real evidence, not a guess -- but it is deliberately
narrow: it does not test streaming, multi-turn tool-result continuation, or
long-running agentic loops. A pass here is evidence of basic
transport/tool-call compatibility, not a substitute for the certification
Anthropic's own models carry.

This performs a real network call using the operator's own credentials and
therefore has a real cost; callers should only invoke it on explicit
request, never automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from enhanced_router.config_models import ModelSpec
from enhanced_router.credential_store import resolve as resolve_credential

_PROBE_TIMEOUT_SECONDS = 20.0

_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "report_probe_result",
        "description": "Call this tool with the word 'ok' to confirm tool-call support.",
        "parameters": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
            "required": ["status"],
        },
    },
}

_ANTHROPIC_TOOL = {
    "name": "report_probe_result",
    "description": "Call this tool with the word 'ok' to confirm tool-call support.",
    "input_schema": {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
    },
}

_PROMPT = "Call the report_probe_result tool with status='ok'. Do not respond with plain text."


@dataclass
class ProbeResult:
    passed: bool
    model_id: str
    backend: str
    tool_call_seen: bool
    http_status: int | None = None
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    probed_at: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "status": "certified" if self.passed else "failed",
            "backend": self.backend,
            "tool_call_seen": self.tool_call_seen,
            "http_status": self.http_status,
            "error": self.error,
            "evidence": self.evidence,
            "probed_at": self.probed_at,
        }


def _missing_credential(model_id: str, backend: str, key_env: str | None) -> ProbeResult:
    return ProbeResult(
        passed=False, model_id=model_id, backend=backend, tool_call_seen=False,
        error=f"credential '{key_env}' is not configured" if key_env else "model has no api_key_env",
    )


def probe_model(
    model_id: str, spec: ModelSpec, *, endpoint_id: str = "auto", config_dir: str | None = None,
) -> ProbeResult:
    """Make one real, minimal tool-call round-trip against a model's endpoint.

    Bypasses LiteLLM (it may not be running yet); calls the underlying
    provider endpoint directly using the same credential/api_base the
    router itself would eventually use.
    """
    import httpx

    endpoint = None
    if endpoint_id != "auto" and endpoint_id in spec.endpoints:
        endpoint = spec.endpoints[endpoint_id]
    elif spec.endpoints:
        endpoint = next(iter(spec.endpoints.values()))

    backend = (endpoint.backend if endpoint else spec.backend)
    api_base = (endpoint.api_base if endpoint else None) or spec.api_base
    key_env = (endpoint.api_key_env if endpoint else None) or spec.api_key_env
    upstream_model = (endpoint.upstream_model if endpoint else None) or spec.upstream_model
    litellm_model = (endpoint.litellm_model if endpoint else None) or spec.litellm_model

    if not api_base:
        return ProbeResult(
            passed=False, model_id=model_id, backend=backend, tool_call_seen=False,
            error="model has no api_base to probe",
        )

    api_key = (resolve_credential(key_env, config_dir) if key_env else None) or ""
    if key_env and not api_key:
        return _missing_credential(model_id, backend, key_env)

    started = time.monotonic()
    try:
        if backend == "direct-anthropic":
            result = _probe_anthropic(api_base, api_key, upstream_model or model_id)
        else:
            raw_model = litellm_model.split("/", 1)[-1] if litellm_model else model_id
            result = _probe_openai_chat(api_base, api_key, raw_model)
    except httpx.HTTPError as exc:
        result = ProbeResult(
            passed=False, model_id=model_id, backend=backend, tool_call_seen=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    result.model_id = model_id
    result.backend = backend
    result.evidence["elapsed_seconds"] = round(time.monotonic() - started, 2)
    result.probed_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    return result


def _probe_openai_chat(api_base: str, api_key: str, raw_model: str) -> ProbeResult:
    import httpx

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": raw_model,
        "messages": [{"role": "user", "content": _PROMPT}],
        "tools": [_OPENAI_TOOL],
        "tool_choice": "auto",
        "max_tokens": 64,
    }
    with httpx.Client(timeout=_PROBE_TIMEOUT_SECONDS) as client:
        resp = client.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        return ProbeResult(
            passed=False, model_id=raw_model, backend="litellm", tool_call_seen=False,
            http_status=resp.status_code, error=resp.text[:400],
        )
    try:
        data = resp.json()
        choice = data["choices"][0]["message"]
    except (KeyError, IndexError, ValueError) as exc:
        return ProbeResult(
            passed=False, model_id=raw_model, backend="litellm", tool_call_seen=False,
            http_status=resp.status_code, error=f"malformed response: {exc}",
            evidence={"raw": resp.text[:400]},
        )
    tool_calls = choice.get("tool_calls") or []
    tool_call_seen = any(
        tc.get("function", {}).get("name") == "report_probe_result" for tc in tool_calls
    )
    return ProbeResult(
        passed=tool_call_seen, model_id=raw_model, backend="litellm",
        tool_call_seen=tool_call_seen, http_status=resp.status_code,
        error=None if tool_call_seen else "model responded but did not invoke the requested tool",
        evidence={"finish_reason": data.get("choices", [{}])[0].get("finish_reason")},
    )


def _probe_anthropic(api_base: str, api_key: str, upstream_model: str) -> ProbeResult:
    import httpx

    url = api_base.rstrip("/") + "/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": upstream_model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": _PROMPT}],
        "tools": [_ANTHROPIC_TOOL],
    }
    with httpx.Client(timeout=_PROBE_TIMEOUT_SECONDS) as client:
        resp = client.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        return ProbeResult(
            passed=False, model_id=upstream_model, backend="direct-anthropic", tool_call_seen=False,
            http_status=resp.status_code, error=resp.text[:400],
        )
    try:
        data = resp.json()
        blocks = data.get("content", [])
    except ValueError as exc:
        return ProbeResult(
            passed=False, model_id=upstream_model, backend="direct-anthropic", tool_call_seen=False,
            http_status=resp.status_code, error=f"malformed response: {exc}",
            evidence={"raw": resp.text[:400]},
        )
    tool_call_seen = any(
        block.get("type") == "tool_use" and block.get("name") == "report_probe_result"
        for block in blocks
    )
    return ProbeResult(
        passed=tool_call_seen, model_id=upstream_model, backend="direct-anthropic",
        tool_call_seen=tool_call_seen, http_status=resp.status_code,
        error=None if tool_call_seen else "model responded but did not invoke the requested tool",
        evidence={"stop_reason": data.get("stop_reason")},
    )
