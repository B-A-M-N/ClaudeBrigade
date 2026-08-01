"""Bounded, router-owned execution lane for read-only specialist calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any

from enhanced_router.backends import (
    BackendType,
    proxy_direct_anthropic,
    proxy_litellm_messages,
)
from enhanced_router.routing import RequestIdentity, resolve_request
from enhanced_router.state import RouteState, WorkflowStateError

LOGGER = logging.getLogger("claude-enhanced-router.sidecar")

_MAX_PACKET_BYTES = 64_000
_MAX_RESULT_BYTES = 128_000


@dataclass(frozen=True)
class _URL:
    path: str = "/v1/messages"
    query: str = ""


@dataclass(frozen=True)
class _InternalRequest:
    """Small request adapter accepted by the existing backend adapters."""

    method: str
    headers: dict[str, str]
    url: _URL = _URL()


class SidecarExecutor:
    """Run structured, read-only specialist calls under router supervision."""

    def __init__(self, state: RouteState) -> None:
        self.state = state
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def invoke(
        self,
        *,
        run_id: str,
        epoch_id: str,
        action_id: str,
        claim_token: str,
        packet: dict[str, Any],
    ) -> dict[str, Any]:
        encoded = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
        action = next(
            (
                item for item in self.state.get_runnable_actions(
                    run_id, epoch_id, include_claimed=True
                )
                if item.get("action_id") == action_id
            ),
            None,
        )
        max_packet_bytes = int(
            (action or {}).get("max_packet_bytes") or _MAX_PACKET_BYTES
        )
        if len(encoded.encode("utf-8")) > min(max_packet_bytes, _MAX_PACKET_BYTES):
            raise WorkflowStateError(
                f"sidecar packet exceeds the {min(max_packet_bytes, _MAX_PACKET_BYTES)} byte bound"
            )
        execution_id = f"scx_{uuid.uuid4().hex}"
        execution = self.state.start_sidecar_execution(
            run_id=run_id,
            epoch_id=epoch_id,
            action_id=action_id,
            claim_token=claim_token,
            execution_id=execution_id,
            packet=packet,
        )
        task = asyncio.create_task(
            self._run(execution_id, run_id, epoch_id, packet),
            name=f"brigade-sidecar-{execution_id}",
        )
        async with self._lock:
            self._tasks[execution_id] = task
        task.add_done_callback(lambda finished: self._forget(execution_id, finished))
        return execution

    def _forget(self, execution_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(execution_id, None)
        if not task.cancelled() and task.exception() is not None:
            LOGGER.error("sidecar task crashed execution=%s: %s", execution_id, task.exception())

    async def cancel(self, execution_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(execution_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # A task can be cancelled before its coroutine gets a chance
                # to enter the lifecycle handler. Reconcile that edge below.
                pass
            execution = self.state.get_agent_execution(execution_id)
            if execution is not None and execution.get("status") not in {
                "completed", "failed", "timeout", "cancelled",
            }:
                updated = self.state.update_agent_execution(
                    execution_id,
                    status="cancelled",
                    error="cancelled by controller",
                    error_class="cancelled_by_controller",
                )
                if updated is not None:
                    execution = updated
                    self._event(execution_id, execution, "cancelled", {"reason": "controller"})
            return execution
        execution = self.state.get_agent_execution(execution_id)
        if execution is None:
            return None
        if execution.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
            updated = self.state.update_agent_execution(
                execution_id, status="cancelled", error="cancelled by controller",
                error_class="cancelled_by_controller",
            )
            if updated is not None:
                execution = updated
                self._event(execution_id, execution, "cancelled", {"reason": "controller"})
        return execution

    async def retry(self, execution_id: str) -> dict[str, Any]:
        retry = self.state.prepare_sidecar_retry(execution_id)
        return await self.invoke(
            run_id=str(retry["run_id"]),
            epoch_id=str(retry["epoch_id"]),
            action_id=str(retry["action_id"]),
            claim_token=str(retry["claim_token"]),
            packet=dict(retry["packet"]),
        )

    async def invoke_detached(
        self,
        *,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        phase_id: str,
        role: str,
        model_id: str,
        provider_id: str | None,
        packet: dict[str, Any],
        timeout_seconds: float,
        runner: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run a router-owned advisory job under persisted sidecar lifecycle."""
        execution = self.state.start_detached_sidecar_execution(
            run_id=run_id,
            epoch_id=epoch_id,
            execution_id=execution_id,
            phase_id=phase_id,
            role=role,
            model_id=model_id,
            provider_id=provider_id,
            packet=packet,
        )
        task = asyncio.create_task(
            self._run_detached(
                execution_id=execution_id,
                run_id=run_id,
                epoch_id=epoch_id,
                timeout_seconds=timeout_seconds,
                runner=runner,
            ),
            name=f"brigade-sidecar-{execution_id}",
        )
        async with self._lock:
            self._tasks[execution_id] = task
        task.add_done_callback(lambda finished: self._forget(execution_id, finished))
        return execution

    async def wait(self, execution_id: str) -> dict[str, Any] | None:
        """Wait for a locally owned detached job and return durable state."""
        task = self._tasks.get(execution_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
        return self.state.get_agent_execution(execution_id)

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run(
        self,
        execution_id: str,
        run_id: str,
        epoch_id: str,
        packet: dict[str, Any],
    ) -> None:
        execution = self.state.get_agent_execution_scoped(run_id, epoch_id, execution_id)
        if execution is None:
            return
        agent_id = str(execution["claude_agent_id"])
        try:
            self.state.update_agent_execution(execution_id, status="running")
            self._event(execution_id, execution, "running", {})
            timeout_seconds = self._timeout_seconds(execution)
            response_payload, status_code, usage = await asyncio.wait_for(
                self._request(
                    run_id=run_id,
                    epoch_id=epoch_id,
                    execution=execution,
                    packet=packet,
                ),
                timeout=timeout_seconds,
            )
            if status_code < 200 or status_code >= 300:
                raise RuntimeError(f"sidecar provider returned HTTP {status_code}")
            result = self._extract_result(response_payload)
            self._validate_result_schema(run_id, epoch_id, execution, result)
            result_json = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            if len(result_json.encode("utf-8")) > _MAX_RESULT_BYTES:
                raise ValueError("sidecar result exceeds the 128 KiB bound")
            completed = self.state.update_agent_execution(
                execution_id,
                status="completed",
                result_type="structured_json",
                result_summary=self._summary(result),
                output_hash=hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                result_json=result_json,
                schema_valid=True,
                evidence_valid=True,
                accepted_by_controller=True,
                quality_score=1.0,
                verdict=(result.get("verdict") if isinstance(result, dict)
                         and isinstance(result.get("verdict"), str) else None),
                confidence=(float(result["confidence"]) if isinstance(result, dict)
                            and isinstance(result.get("confidence"), (int, float)) else None),
                request_count=1,
                total_tokens=usage.get("total_tokens") if usage else None,
                input_tokens=usage.get("input_tokens") if usage else None,
                output_tokens=usage.get("output_tokens") if usage else None,
            )
            self._event(execution_id, completed or execution, "completed", {
                "result_type": "structured_json",
                "status_code": status_code,
            })
            self.state.finish_spawn_assignment(run_id, epoch_id, agent_id, "completed")
        except asyncio.CancelledError:
            cancelled = self.state.update_agent_execution(
                execution_id, status="cancelled", error="cancelled by controller",
                error_class="cancelled_by_controller",
            )
            self._event(execution_id, cancelled or execution, "cancelled", {})
            self.state.finish_spawn_assignment(run_id, epoch_id, agent_id, "cancelled")
        except Exception as exc:
            error_class = self._error_class(exc)
            failed = self.state.update_agent_execution(
                execution_id,
                status="failed",
                error=str(exc)[:500],
                error_class=error_class,
                request_count=1,
            )
            self._event(execution_id, failed or execution, "failed", {
                "error_class": error_class,
                "reason": str(exc)[:500],
            })
            self.state.finish_spawn_assignment(run_id, epoch_id, agent_id, "failed")

    async def _run_detached(
        self,
        *,
        execution_id: str,
        run_id: str,
        epoch_id: str,
        timeout_seconds: float,
        runner: Callable[[], Awaitable[dict[str, Any]]],
    ) -> None:
        execution = self.state.get_agent_execution_scoped(run_id, epoch_id, execution_id)
        if execution is None:
            return
        try:
            running = self.state.update_agent_execution(execution_id, status="running")
            self._event(execution_id, running or execution, "running", {})
            result = await asyncio.wait_for(runner(), timeout=timeout_seconds)
            if not isinstance(result, dict):
                raise ValueError("detached sidecar result must be an object")
            result_json = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            if len(result_json.encode("utf-8")) > _MAX_RESULT_BYTES:
                raise ValueError("detached sidecar result exceeds the 128 KiB bound")
            completed = self.state.update_agent_execution(
                execution_id,
                status="completed",
                result_type="structured_json",
                result_summary=self._summary(result),
                output_hash=hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                result_json=result_json,
                schema_valid=True,
                evidence_valid=True,
                accepted_by_controller=False,
                quality_score=1.0,
            )
            self._event(execution_id, completed or execution, "completed", {
                "result_type": "structured_json",
            })
        except asyncio.CancelledError:
            cancelled = self.state.update_agent_execution(
                execution_id,
                status="cancelled",
                error="cancelled by controller",
                error_class="cancelled_by_controller",
            )
            self._event(execution_id, cancelled or execution, "cancelled", {})
        except Exception as exc:
            error_class = self._error_class(exc)
            failed = self.state.update_agent_execution(
                execution_id,
                status="timed_out" if isinstance(exc, asyncio.TimeoutError) else "failed",
                error=str(exc)[:500],
                error_class=error_class,
            )
            self._event(execution_id, failed or execution, "failed", {
                "error_class": error_class,
                "reason": str(exc)[:500],
            })

    async def _request(
        self,
        *,
        run_id: str,
        epoch_id: str,
        execution: dict[str, Any],
        packet: dict[str, Any],
    ) -> tuple[dict[str, Any], int, dict[str, int]]:
        role = str(execution["role"])
        public_model = f"anthropic-brigade-{role}"
        sidecar = self._sidecar_spec(execution)
        identity = RequestIdentity(
            run_id=run_id,
            claude_session_id=None,
            claude_agent_id=str(execution["claude_agent_id"]),
            claude_parent_agent_id=None,
            endpoint_kind="/v1/messages",
        )
        resolved = resolve_request(
            identity=identity,
            public_model=public_model,
            explicit_model_id=sidecar.model_id if sidecar is not None else None,
            explicit_endpoint=(
                sidecar.endpoint
                if sidecar is not None and sidecar.endpoint != "auto"
                else None
            ),
            sidecar=sidecar is not None,
        )
        request = _InternalRequest(
            method="POST",
            headers={
                "content-type": "application/json",
                "x-request-id": f"sidecar:{execution['execution_id']}",
                "x-brigade-run-id": run_id,
                "x-claude-code-agent-id": str(execution["claude_agent_id"]),
            },
        )
        messages: list[dict[str, str]] = []
        if sidecar is not None and sidecar.system_prompt.strip():
            messages.append({"role": "system", "content": sidecar.system_prompt})
        messages.append({
            "role": "user",
            "content": json.dumps(packet, separators=(",", ":"), ensure_ascii=False),
        })
        payload = {
            "model": public_model,
            "messages": messages,
            "max_tokens": sidecar.max_output_tokens if sidecar is not None else 2_048,
            "stream": False,
        }
        if resolved.kind is BackendType.DIRECT_ANTHROPIC:
            response = await proxy_direct_anthropic(request, payload, resolved)
        elif resolved.kind is BackendType.LITELLM:
            response = await proxy_litellm_messages(request, payload, resolved)
        else:
            raise RuntimeError("sidecar calls cannot use the controller passthrough")
        body = getattr(response, "body", None)
        if body is None and hasattr(response, "body_iterator"):
            chunks = [chunk async for chunk in response.body_iterator]
            body = b"".join(chunks)
        if not isinstance(body, (bytes, bytearray)):
            raise RuntimeError("sidecar backend returned no response body")
        try:
            parsed = json.loads(bytes(body))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("sidecar backend returned malformed JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("sidecar backend returned a non-object response")
        raw_usage = parsed.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        return parsed, int(getattr(response, "status_code", 500)), {
            "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    def _sidecar_spec(self, execution: dict[str, Any]) -> Any | None:
        phase = next(
            (
                item for item in self.state.get_workflow_phases(
                    str(execution["run_id"]), str(execution["epoch_id"])
                )
                if item.get("phase_id") == execution.get("phase_id")
            ),
            None,
        )
        sidecar_id = str((phase or {}).get("sidecar_id") or "").strip()
        if not sidecar_id:
            return None
        from enhanced_router.registry import get_registry

        registry = get_registry()
        getter = getattr(registry, "get_sidecar", None)
        if getter is None:
            return None
        return getter(sidecar_id)

    def _timeout_seconds(self, execution: dict[str, Any]) -> float:
        sidecar = self._sidecar_spec(execution)
        return float(sidecar.timeout_seconds) if sidecar is not None else 45.0

    @staticmethod
    def _extract_result(response: dict[str, Any]) -> Any:
        content = response.get("content")
        if isinstance(content, list):
            text = "".join(
                str(item.get("text", "")) for item in content
                if isinstance(item, dict) and item.get("type", "text") == "text"
            )
        elif isinstance(response.get("choices"), list) and response["choices"]:
            message = response["choices"][0].get("message", {})
            text = message.get("content", "") if isinstance(message, dict) else ""
        else:
            text = response.get("output", response.get("text", ""))
        if isinstance(text, (dict, list)):
            return text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("sidecar response contained no structured result")
        try:
            return json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"text": text[:32_000]}

    def _validate_result_schema(
        self, run_id: str, epoch_id: str, execution: dict[str, Any], result: Any,
    ) -> None:
        phase = next(
            (item for item in self.state.get_workflow_phases(run_id, epoch_id)
             if item.get("phase_id") == execution.get("phase_id")),
            None,
        )
        schema_raw = phase.get("result_schema") if phase else None
        if not schema_raw:
            return
        schema = json.loads(str(schema_raw))
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if not isinstance(result, dict) or not isinstance(required, list):
            raise ValueError("sidecar result does not match its phase schema")
        missing = [str(key) for key in required if key not in result]
        if missing:
            raise ValueError(f"sidecar result is missing: {', '.join(missing)}")

    def _event(self, execution_id: str, execution: dict[str, Any], event_type: str, payload: dict) -> None:
        try:
            self.state.append_execution_event(
                str(execution["run_id"]), str(execution["epoch_id"]), execution_id,
                event_type, payload,
            )
        except Exception:
            LOGGER.exception("failed to persist sidecar event execution=%s", execution_id)

    @staticmethod
    def _summary(result: Any) -> str:
        text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        return text[:1_000]

    @staticmethod
    def _error_class(exc: Exception) -> str:
        if isinstance(exc, asyncio.TimeoutError):
            return "wall_timeout"
        if isinstance(exc, WorkflowStateError):
            return "workflow_state_error"
        if isinstance(exc, ValueError):
            return "malformed_result"
        return "sidecar_runtime_error"


_executor: SidecarExecutor | None = None


def get_sidecar_executor(state: RouteState | None = None) -> SidecarExecutor:
    global _executor
    if _executor is None:
        if state is None:
            from enhanced_router.state import get_state
            state = get_state()
        _executor = SidecarExecutor(state)
    return _executor


async def shutdown_sidecar_executor() -> None:
    global _executor
    if _executor is not None:
        await _executor.shutdown()
        _executor = None
