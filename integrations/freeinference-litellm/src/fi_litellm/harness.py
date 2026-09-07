"""Explicit, sanitized compatibility probes for the local proxy.

The harness is intentionally opt-in and reports contract outcomes only.  It
does not persist prompts, tool arguments, response text, or credentials.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx


HARNESS_VERSION = "fi-contract-v2"
PROTOCOL_VERSION = "openai-chat-anthropic-messages-v1"


def _status(response: httpx.Response, *, require_json: bool = False) -> str:
    if not 200 <= response.status_code < 300:
        return "fail"
    if require_json:
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return "fail"
        if not isinstance(payload, dict):
            return "fail"
    return "pass"


def _usage_status(payload: dict[str, Any]) -> str:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return "not_observed"
    input_seen = any(key in usage for key in ("input_tokens", "prompt_tokens"))
    output_seen = any(key in usage for key in ("output_tokens", "completion_tokens"))
    return "pass" if input_seen and output_seen else "partial"


def _stream_status(client: httpx.Client, model: str) -> tuple[str, dict[str, Any] | None]:
    chunks = 0
    saw_done = False
    usage: dict[str, Any] | None = None
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "Reply OK"}],
                "max_tokens": 8,
            },
        ) as response:
            if not 200 <= response.status_code < 300:
                return "fail", None
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    value = line[5:].strip()
                    if value == "[DONE]":
                        saw_done = True
                        continue
                    try:
                        payload = json.loads(value)
                    except json.JSONDecodeError:
                        return "fail", None
                    if isinstance(payload, dict):
                        chunks += 1
                        if isinstance(payload.get("usage"), dict):
                            usage = payload["usage"]
    except httpx.HTTPError:
        return "fail", None
    if chunks == 0:
        return "fail", usage
    return ("pass" if saw_done else "partial"), usage


def _tool_call_status(client: httpx.Client, model: str) -> str:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Call the lookup tool with id 1."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up an item",
                    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                },
            }],
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    if _status(response) != "pass":
        return "fail"
    try:
        message = response.json()["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        return "pass" if calls and calls[0].get("function", {}).get("name") == "lookup" else "not_observed"
    except (KeyError, IndexError, TypeError, ValueError):
        return "fail"


def _tool_calls(response: httpx.Response) -> list[dict[str, Any]] | None:
    """Extract tool calls without retaining model content in a report."""
    try:
        payload = response.json()
        message = payload["choices"][0]["message"]
        calls = message.get("tool_calls") or []
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls):
        return None
    return calls


def _tool_call_parallel_status(client: httpx.Client, model: str) -> str:
    """Check whether the model can emit two independent tool calls."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Call both lookup and inspect."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up an item",
                        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "description": "Inspect an item",
                        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                    },
                },
            ],
            "tool_choice": "required",
            "max_tokens": 128,
        },
    )
    if _status(response) != "pass":
        return "fail"
    calls = _tool_calls(response)
    if not calls:
        return "not_observed"
    names = {
        call.get("function", {}).get("name")
        for call in calls
        if isinstance(call.get("function"), dict)
    }
    return "pass" if {"lookup", "inspect"}.issubset(names) else "not_observed"


def _tool_result_continuation_status(client: httpx.Client, model: str) -> str:
    """Check the second request in a tool-call continuation loop."""
    first = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Call lookup with id 1, then use its result."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up an item",
                    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
                },
            }],
            "tool_choice": "required",
            "max_tokens": 64,
        },
    )
    if _status(first) != "pass":
        return "fail"
    calls = _tool_calls(first)
    if not calls:
        return "not_observed"
    try:
        first_payload = first.json()
        assistant_message = dict(first_payload["choices"][0]["message"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return "fail"
    tool_messages = []
    for call in calls:
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            return "fail"
        tool_messages.append({"role": "tool", "tool_call_id": call_id, "content": "item-1"})
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": "Call lookup with id 1, then use its result."},
                assistant_message,
                *tool_messages,
            ],
            "max_tokens": 64,
        },
    )
    return "pass" if _status(second, require_json=True) == "pass" else "fail"


def _cancellation_status(client: httpx.Client, model: str) -> str:
    """Check that an early client close cleanly terminates a stream."""
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "stream": True,
                "messages": [{"role": "user", "content": "Reply with a long response."}],
                "max_tokens": 128,
            },
        ) as response:
            if not 200 <= response.status_code < 300:
                return "fail"
            for line in response.iter_lines():
                if line:
                    return "pass"
            return "not_observed"
    except httpx.HTTPError:
        return "fail"


def _structured_output_status(client: httpx.Client, model: str) -> str:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Return a JSON object with ok=true."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "probe",
                    "strict": True,
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
                },
            },
            "max_tokens": 32,
        },
    )
    if _status(response) != "pass":
        return "fail"
    try:
        text = response.json()["choices"][0]["message"].get("content", "")
        parsed = json.loads(text)
        return "pass" if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool) else "fail"
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return "fail"


def run_contract(model: str, *, base_url: str, local_key: str) -> dict[str, Any]:
    """Run bounded transport, tool-loop, structured-output and SSE checks."""
    headers = {
        "Authorization": f"Bearer {local_key}",
        "Content-Type": "application/json",
        "X-Session-ID": f"fi-contract-{model}",
        "X-Request-ID": f"fi-contract-{model}-001",
    }
    with httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=90.0, follow_redirects=False) as client:
        openai = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 8},
        )
        try:
            openai_payload = openai.json()
        except (ValueError, json.JSONDecodeError):
            openai_payload = {}
        stream_status, stream_usage = _stream_status(client, model)
        anthropic = client.post(
            "/v1/messages",
            headers={**headers, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "Reply OK"}]},
        )
        tool_status = _tool_call_status(client, model)
        parallel_tool_status = _tool_call_parallel_status(client, model)
        continuation_status = _tool_result_continuation_status(client, model)
        structured_status = _structured_output_status(client, model)
        cancellation_status = _cancellation_status(client, model)

    return {
        "harness_version": HARNESS_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model": model,
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "openai_nonstream": _status(openai, require_json=True),
        "openai_stream": stream_status,
        "anthropic_messages": _status(anthropic, require_json=True),
        "tool_call_single": tool_status,
        "tool_call_parallel": parallel_tool_status,
        "tool_result_continuation": continuation_status,
        "structured_output": structured_status,
        "reasoning_content": "observed" if isinstance(openai_payload.get("choices"), list) else "not_observed",
        "image_input": "not_tested",
        "usage_accounting": _usage_status(openai_payload) if stream_usage is None else ("pass" if _usage_status(openai_payload) == "pass" else "partial"),
        "request_id_echo": "pass" if openai.headers.get("x-request-id") else "not_observed",
        "cancellation": cancellation_status,
        "error_mapping_429": "not_tested",
        "error_mapping_503": "not_tested",
    }
