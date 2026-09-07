"""Compatibility implementation for bounded coprocessor calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any

from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate as validate_json_schema

from enhanced_router.backends import (
    BackendType,
    proxy_direct_anthropic,
    proxy_litellm_messages,
)
from enhanced_router.routing import RequestIdentity, resolve_request
from enhanced_router.route_ladder import candidate_dict, dedupe_candidates, route_digest
from enhanced_router.state import RouteState, WorkflowStateError

LOGGER = logging.getLogger("claude-enhanced-router.sidecar")

_MAX_PACKET_BYTES = 64_000
_MAX_RESULT_BYTES = 128_000
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


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


class CoprocessorExecutor:
    """Run bounded structured calls under router supervision."""

    def __init__(self, state: RouteState) -> None:
        self.state = state
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._detached_jobs: dict[str, dict[str, Any]] = {}
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
        execution = self.state.get_agent_execution(execution_id)
        if (
            execution is not None
            and execution.get("execution_kind") in {"sidecar_call", "coprocessor_call"}
            and str(execution.get("phase_id") or "").startswith("fastpath:")
        ):
            if execution.get("status") not in {"failed", "timeout"}:
                raise WorkflowStateError("detached fastpath execution is not retryable")
            job = self._detached_jobs.get(execution_id)
            if job is None:
                job = self._recover_detached_job(execution)
            attempt = int(execution.get("retry_count") or 0) + 2
            if attempt > 2:
                raise WorkflowStateError("detached fastpath retry budget exhausted")
            retry_id = f"{execution_id}:retry:{uuid.uuid4().hex[:8]}"
            return await self.invoke_detached(
                run_id=str(job["run_id"]),
                epoch_id=str(job["epoch_id"]),
                execution_id=retry_id,
                phase_id=str(job["phase_id"]),
                role=str(job["role"]),
                model_id=str(job["model_id"]),
                provider_id=job.get("provider_id"),
                packet=dict(job["packet"]),
                timeout_seconds=float(job["timeout_seconds"]),
                runner=job["runner"],
                attempt=attempt,
                parent_execution_id=execution_id,
            )
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
        attempt: int = 1,
        parent_execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a router-owned advisory job under persisted sidecar lifecycle."""
        execution = self.state.start_detached_coprocessor_execution(
            run_id=run_id,
            epoch_id=epoch_id,
            execution_id=execution_id,
            phase_id=phase_id,
            role=role,
            model_id=model_id,
            provider_id=provider_id,
            packet=packet,
            parent_execution_id=parent_execution_id,
            retry_count=max(0, attempt - 1),
        )
        self._detached_jobs[execution_id] = {
            "run_id": run_id,
            "epoch_id": epoch_id,
            "phase_id": phase_id,
            "role": role,
            "model_id": model_id,
            "provider_id": provider_id,
            "packet": dict(packet),
            "timeout_seconds": timeout_seconds,
            "runner": runner,
            "attempt": attempt,
        }
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

    async def invoke_feedback(
        self,
        *,
        run_id: str,
        epoch_id: str,
        coprocessor_id: str,
        packet: dict[str, Any],
        parent_execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Start one automatic feedback call under normal provider admission."""
        from enhanced_router.registry import get_registry

        registry = get_registry()
        spec = registry.get_coprocessor(coprocessor_id)
        if not spec.enabled:
            raise WorkflowStateError(f"coprocessor '{coprocessor_id}' is disabled")
        model = registry.get_model(spec.model_id)
        provider_id = spec.provider_id or model.provider_id
        if not provider_id:
            raise WorkflowStateError(
                f"coprocessor '{coprocessor_id}' has no provider route"
            )
        execution_id = f"cfb_{uuid.uuid4().hex}"

        async def runner() -> dict[str, Any]:
            return await self._run_feedback_request(
                execution_id=execution_id,
                run_id=run_id,
                epoch_id=epoch_id,
                spec=spec,
                packet=packet,
            )

        return await self.invoke_detached(
            run_id=run_id,
            epoch_id=epoch_id,
            execution_id=execution_id,
            phase_id=f"feedback:{coprocessor_id}",
            role=f"coprocessor:{coprocessor_id}",
            model_id=spec.model_id,
            provider_id=provider_id,
            packet=packet,
            timeout_seconds=spec.timeout_seconds,
            runner=runner,
            parent_execution_id=parent_execution_id,
        )

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
        self._detached_jobs.clear()

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
        try:
            self.state.update_agent_execution(execution_id, status="running")
            self._event(execution_id, execution, "running", {})
            timeout_seconds = self._timeout_seconds(execution)
            sidecar = self._sidecar_spec(execution)
            candidates = dedupe_candidates([
                {
                    "model": sidecar.model_id,
                    "endpoint": sidecar.endpoint,
                    "provider_id": getattr(sidecar, "provider_id", None),
                },
                *[
                    candidate_dict(item)
                    for item in getattr(sidecar, "fallback_routes", [])
                ],
            ]) if sidecar is not None else [{}]
            response_payload: dict[str, Any] | None = None
            status_code = 500
            usage: dict[str, int] = {}
            selected_attempt_id: str | None = None
            for index, candidate in enumerate(candidates or [{}]):
                candidate = candidate_dict(candidate)
                attempt_id = f"{execution_id}:route:{index}"
                self.state.start_route_attempt(
                    attempt_id=attempt_id,
                    run_id=run_id,
                    epoch_id=epoch_id,
                    execution_id=execution_id,
                    candidate_index=index,
                    model_id=candidate["model"],
                    provider_id=candidate.get("provider_id"),
                    endpoint_id=candidate.get("endpoint", "auto"),
                    route_digest=route_digest(candidate),
                )
                attempt_finished = False
                try:
                    response_payload, status_code, usage = await asyncio.wait_for(
                        self._request(
                            run_id=run_id,
                            epoch_id=epoch_id,
                            execution=execution,
                            packet=packet,
                            candidate=candidate or None,
                        ),
                        timeout=timeout_seconds,
                    )
                    if 200 <= status_code < 300:
                        # Do not call a provider response successful until
                        # extraction and the declared output schema have also
                        # passed.  Transport success with malformed JSON is a
                        # failed structured attempt, not quality evidence.
                        selected_attempt_id = attempt_id
                        attempt_finished = True
                        break
                    if status_code not in _RETRYABLE_HTTP_STATUSES:
                        raise RuntimeError(f"sidecar provider returned HTTP {status_code}")
                    raise RuntimeError(f"sidecar provider returned HTTP {status_code}")
                except asyncio.CancelledError:
                    if not attempt_finished:
                        self.state.finish_route_attempt(
                            attempt_id, status="cancelled", status_code=status_code,
                        )
                    raise
                except ValueError:
                    # A malformed structured result is not a transport
                    # failure and must not silently move the same packet to a
                    # second provider.
                    self.state.finish_route_attempt(
                        attempt_id, status="failed", status_code=status_code,
                        error_class="malformed_result",
                    )
                    raise
                except Exception as exc:
                    self.state.finish_route_attempt(
                        attempt_id, status="failed", status_code=status_code,
                        error_class=self._error_class(exc), error=str(exc),
                    )
                    if status_code not in _RETRYABLE_HTTP_STATUSES:
                        raise
                    if index + 1 >= len(candidates):
                        raise
                    # Bounded calls may release their failed candidate before
                    # selecting the next route. Native Claude workers never
                    # use this executor and remain immutably bound.
                    self.state.release_binding(
                        run_id, str(execution["claude_agent_id"]),
                    )
            if response_payload is None:
                raise RuntimeError("sidecar route ladder returned no response")
            try:
                result = self._extract_result(response_payload)
                self._validate_result_schema(run_id, epoch_id, execution, result)
            except Exception as exc:
                if selected_attempt_id is not None:
                    self.state.finish_route_attempt(
                        selected_attempt_id,
                        status="failed",
                        status_code=status_code,
                        error_class="malformed_result",
                        error=str(exc),
                    )
                    selected_attempt_id = None
                raise
            result_json = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            if len(result_json.encode("utf-8")) > _MAX_RESULT_BYTES:
                if selected_attempt_id is not None:
                    self.state.finish_route_attempt(
                        selected_attempt_id,
                        status="failed",
                        status_code=status_code,
                        error_class="malformed_result",
                        error="sidecar result exceeds the 128 KiB bound",
                    )
                    selected_attempt_id = None
                raise ValueError("sidecar result exceeds the 128 KiB bound")
            if selected_attempt_id is not None:
                self.state.finish_route_attempt(
                    selected_attempt_id,
                    status="succeeded",
                    status_code=status_code,
                    usage=usage,
                )
            self.state.finalize_router_owned_execution(
                execution_id,
                status="completed",
                result_type="structured_json",
                result_summary=self._summary(result),
                output_hash=hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                result_json=result_json,
                schema_valid=True,
                evidence_valid=None,
                accepted_by_controller=False,
                quality_score=None,
                verdict=(result.get("verdict") if isinstance(result, dict)
                         and isinstance(result.get("verdict"), str) else None),
                confidence=(float(result["confidence"]) if isinstance(result, dict)
                            and isinstance(result.get("confidence"), (int, float)) else None),
                request_count=1,
                total_tokens=usage.get("total_tokens") if usage else None,
                input_tokens=usage.get("input_tokens") if usage else None,
                output_tokens=usage.get("output_tokens") if usage else None,
                estimated_cost=usage.get("estimated_cost") if usage else None,
            )
        except asyncio.CancelledError:
            self.state.finalize_router_owned_execution(
                execution_id, status="cancelled", error="cancelled by controller",
                error_class="cancelled_by_controller",
            )
        except Exception as exc:
            error_class = self._error_class(exc)
            self.state.finalize_router_owned_execution(
                execution_id,
                status="failed",
                error=str(exc)[:500],
                error_class=error_class,
                request_count=1,
            )

    def _recover_detached_job(self, execution: dict[str, Any]) -> dict[str, Any]:
        """Rebuild a supported fastpath runner from durable execution evidence."""
        run_id = str(execution["run_id"])
        epoch_id = str(execution["epoch_id"])
        execution_id = str(execution["execution_id"])
        events = self.state.get_execution_events(
            run_id, epoch_id, execution_id, limit=1,
        )
        if not events or not isinstance(events[0].get("payload"), dict):
            raise WorkflowStateError("detached fastpath input packet is unavailable")
        packet = dict(events[0]["payload"])
        phase_id = str(execution.get("phase_id") or "")
        if phase_id == "fastpath:route":
            from enhanced_router.app import _run_fastpath_route

            async def runner() -> dict[str, Any]:
                return await _run_fastpath_route(packet)
        elif phase_id == "fastpath:verify":
            from enhanced_router.app import _run_fastpath_verify

            async def runner() -> dict[str, Any]:
                return await _run_fastpath_verify(packet)
        else:
            raise WorkflowStateError("detached execution type cannot be recovered")
        timeout_seconds = 5.0
        try:
            from enhanced_router.registry import get_registry

            config = get_registry().fastpath
            if config is not None:
                timeout_seconds = float(config.timeout_seconds)
        except (ImportError, AttributeError, RuntimeError):
            pass
        return {
            "run_id": run_id,
            "epoch_id": epoch_id,
            "phase_id": phase_id,
            "role": str(execution["role"]),
            "model_id": str(execution["model_id"]),
            "provider_id": execution.get("provider_id"),
            "packet": packet,
            "timeout_seconds": timeout_seconds,
            "runner": runner,
        }
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
            self.state.finalize_router_owned_execution(
                execution_id,
                status="completed",
                result_type="structured_json",
                result_summary=self._summary(result),
                output_hash=hashlib.sha256(result_json.encode("utf-8")).hexdigest(),
                result_json=result_json,
                schema_valid=True,
                evidence_valid=None,
                accepted_by_controller=False,
                quality_score=None,
            )
            self._detached_jobs.pop(execution_id, None)
        except asyncio.CancelledError:
            self.state.finalize_router_owned_execution(
                execution_id,
                status="cancelled",
                error="cancelled by controller",
                error_class="cancelled_by_controller",
            )
        except Exception as exc:
            error_class = self._error_class(exc)
            self.state.finalize_router_owned_execution(
                execution_id,
                status="timed_out" if isinstance(exc, asyncio.TimeoutError) else "failed",
                error=str(exc)[:500],
                error_class=error_class,
            )

    async def _request(
        self,
        *,
        run_id: str,
        epoch_id: str,
        execution: dict[str, Any],
        packet: dict[str, Any],
        coprocessor: Any | None = None,
        candidate: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        role = "adversary" if coprocessor is not None else str(execution["role"])
        public_model = f"anthropic-brigade-{role}"
        sidecar = coprocessor or self._sidecar_spec(execution)
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
            explicit_model_id=(
                candidate.get("model") if candidate else None
            ) or (sidecar.model_id if sidecar is not None else None),
            explicit_endpoint=(
                candidate.get("endpoint")
                if candidate is not None and candidate.get("endpoint") not in {None, "auto"}
                else sidecar.endpoint
                if sidecar is not None and sidecar.endpoint != "auto"
                else None
            ),
            sidecar=sidecar is not None,
            _route_candidate=candidate,
        )
        request_lane = (
            "feedback"
            if str(execution.get("role") or "").startswith("coprocessor:")
            else "fastpath"
            if str(execution.get("phase_id") or "").startswith("fastpath:")
            else "worker"
        )
        resolved = type(resolved)(
            **{**resolved.__dict__, "request_lane": request_lane}
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
        raw_cost = usage.get("cost", usage.get("estimated_cost"))
        try:
            estimated_cost = float(raw_cost) if raw_cost is not None else None
        except (TypeError, ValueError):
            estimated_cost = None
        return parsed, int(getattr(response, "status_code", 500)), {
            "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "estimated_cost": estimated_cost,
        }

    async def _run_feedback_request(
        self,
        *,
        execution_id: str,
        run_id: str,
        epoch_id: str,
        spec: Any,
        packet: dict[str, Any],
    ) -> dict[str, Any]:
        execution = {
            "execution_id": execution_id,
            "run_id": run_id,
            "epoch_id": epoch_id,
            "claude_agent_id": f"sidecar:{execution_id}",
            "role": f"coprocessor:{spec.model_id}",
        }
        candidates = dedupe_candidates([
            {
                "model": spec.model_id,
                "endpoint": spec.endpoint,
                "provider_id": spec.provider_id,
            },
            *[
                candidate_dict(item)
                for item in getattr(spec, "fallback_routes", [])
            ],
        ])
        response: dict[str, Any] | None = None
        selected_attempt_id: str | None = None
        selected_status_code: int | None = None
        selected_usage: dict[str, Any] = {}
        for index, candidate in enumerate(candidates or [{}]):
            candidate = candidate_dict(candidate)
            status_code: int | None = None
            attempt_id = f"{execution_id}:route:{index}"
            self.state.start_route_attempt(
                attempt_id=attempt_id,
                run_id=run_id,
                epoch_id=epoch_id,
                execution_id=execution_id,
                candidate_index=index,
                model_id=candidate["model"],
                provider_id=candidate.get("provider_id"),
                endpoint_id=candidate.get("endpoint", "auto"),
                route_digest=route_digest(candidate),
            )
            try:
                response, status_code, usage = await self._request(
                    run_id=run_id,
                    epoch_id=epoch_id,
                    execution=execution,
                    packet=packet,
                    coprocessor=spec,
                    candidate=candidate or None,
                )
                if 200 <= status_code < 300:
                    self.state.update_agent_execution(
                        execution_id,
                        request_count=1,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        estimated_cost=usage.get("estimated_cost"),
                    )
                    selected_attempt_id = attempt_id
                    selected_status_code = status_code
                    selected_usage = usage
                    break
                raise RuntimeError(f"coprocessor provider returned HTTP {status_code}")
            except asyncio.CancelledError:
                self.state.finish_route_attempt(
                    attempt_id, status="cancelled", status_code=status_code,
                )
                raise
            except ValueError:
                self.state.finish_route_attempt(
                    attempt_id, status="failed", status_code=status_code,
                    error_class="malformed_result",
                )
                raise
            except Exception as exc:
                self.state.finish_route_attempt(
                    attempt_id, status="failed", status_code=status_code,
                    error_class=self._error_class(exc), error=str(exc),
                )
                if status_code is not None and status_code not in _RETRYABLE_HTTP_STATUSES:
                    raise
                if index + 1 >= len(candidates):
                    raise
                self.state.release_binding(
                    run_id, str(execution["claude_agent_id"]),
                )
        if response is None:
            raise RuntimeError("coprocessor route ladder returned no response")
        try:
            result = self._extract_result(response)
            self._validate_named_output_schema(spec, result, packet)
        except Exception as exc:
            if selected_attempt_id is not None:
                self.state.finish_route_attempt(
                    selected_attempt_id,
                    status="failed",
                    status_code=selected_status_code,
                    error_class="malformed_result",
                    error=str(exc),
                )
                selected_attempt_id = None
            raise
        if selected_attempt_id is not None:
            self.state.finish_route_attempt(
                selected_attempt_id,
                status="succeeded",
                status_code=selected_status_code,
                usage=selected_usage,
            )
        return {"coprocessor_result": result}

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
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("coprocessor response was not valid JSON") from exc

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
        try:
            schema = json.loads(str(schema_raw))
            if not isinstance(schema, dict):
                raise ValueError("sidecar phase schema must be a JSON object")
            validate_json_schema(instance=result, schema=schema)
        except (TypeError, ValueError, json.JSONDecodeError, JSONSchemaValidationError) as exc:
            raise ValueError("sidecar result does not match its phase schema") from exc

    @staticmethod
    def _validate_named_output_schema(
        spec: Any, result: Any, packet: dict[str, Any],
    ) -> None:
        """Validate named coprocessor contracts before persistence."""
        schema_id = str(getattr(spec, "output_schema_id", None) or "").strip()
        if not schema_id:
            if not isinstance(result, (dict, list, str, int, float, bool)) and result is not None:
                raise ValueError("coprocessor result is not JSON-compatible")
            return
        if schema_id == "json_object":
            if not isinstance(result, dict):
                raise ValueError("coprocessor output contract requires a JSON object")
            return
        if schema_id == "sentinel_v1":
            schema = {
                "type": "object",
                "required": [
                    "alert", "why", "suggested_next", "confidence", "evidence_refs",
                ],
                "properties": {
                    "alert": {"enum": ["none", "watch", "escalate"]},
                    "why": {"type": "string", "minLength": 1, "maxLength": 500},
                    "suggested_next": {
                        "enum": [
                            "re_ground", "invoke_minimax", "invoke_controller",
                            "invoke_glm", "proceed", "none",
                        ]
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
            }
            try:
                validate_json_schema(instance=result, schema=schema)
            except JSONSchemaValidationError as exc:
                raise ValueError("coprocessor sentinel does not match sentinel_v1") from exc
            if not isinstance(result, dict):
                raise ValueError("coprocessor sentinel must be a JSON object")
            evidence_manifest = packet.get("evidence_manifest")
            refs = {str(ref) for ref in result.get("evidence_refs", [])}
            if not isinstance(evidence_manifest, dict):
                if refs:
                    raise ValueError(
                        "coprocessor sentinel contains findings without an evidence manifest"
                    )
            else:
                unknown = sorted(refs - set(evidence_manifest))
                if unknown:
                    raise ValueError(
                        "coprocessor sentinel references unknown evidence: "
                        + ", ".join(unknown[:8])
                    )
            if getattr(spec, "minimum_confidence", None) is not None:
                if float(result["confidence"]) < float(spec.minimum_confidence):
                    raise ValueError("coprocessor sentinel confidence is below the configured minimum")
            return
        if schema_id == "feedback_v1":
            schema = {
                "type": "object",
                "required": ["decision", "findings", "unknowns", "needs_more_evidence"],
                "properties": {
                    "decision": {
                        "enum": ["continue", "repair", "escalate", "insufficient_evidence"],
                    },
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "severity", "claim", "evidence_refs", "confidence"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "severity": {"enum": ["blocker", "high", "medium", "low"]},
                                "claim": {"type": "string"},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                "required_action": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "unknowns": {"type": "array", "items": {"type": "string"}},
                    "needs_more_evidence": {"type": "boolean"},
                },
                "additionalProperties": False,
            }
            try:
                validate_json_schema(instance=result, schema=schema)
            except JSONSchemaValidationError as exc:
                raise ValueError("coprocessor feedback does not match feedback_v1") from exc
            if not isinstance(result, dict):
                raise ValueError("coprocessor feedback must be a JSON object")
            evidence_manifest = packet.get("evidence_manifest")
            refs = {
                str(ref)
                for finding in result.get("findings", [])
                if isinstance(finding, dict)
                for ref in finding.get("evidence_refs", [])
            }
            if isinstance(evidence_manifest, dict):
                allowed = set(evidence_manifest)
                unknown = sorted(refs - allowed)
                if unknown:
                    raise ValueError(
                        "coprocessor feedback references unknown evidence: "
                        + ", ".join(unknown[:8])
                    )
            elif refs:
                raise ValueError("coprocessor feedback contains findings without an evidence manifest")
            if getattr(spec, "minimum_confidence", None) is not None:
                threshold = float(spec.minimum_confidence)
                if any(float(item["confidence"]) < threshold for item in result["findings"]):
                    raise ValueError("coprocessor feedback contains a finding below minimum confidence")
            return
        raise ValueError(f"unknown coprocessor output schema '{schema_id}'")

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


_executor: CoprocessorExecutor | None = None


def get_coprocessor_executor(state: RouteState | None = None) -> CoprocessorExecutor:
    global _executor
    if _executor is None:
        if state is None:
            from enhanced_router.state import get_state
            state = get_state()
        _executor = CoprocessorExecutor(state)
    return _executor


def get_sidecar_executor(state: RouteState | None = None) -> CoprocessorExecutor:
    """Compatibility alias for the pre-migration executor name."""
    return get_coprocessor_executor(state)


async def shutdown_sidecar_executor() -> None:
    global _executor
    if _executor is not None:
        await _executor.shutdown()
        _executor = None


async def shutdown_coprocessor_executor() -> None:
    """Stop the bounded coprocessor lane."""
    await shutdown_sidecar_executor()


# Compatibility class name for existing integrations and tests.
SidecarExecutor = CoprocessorExecutor
