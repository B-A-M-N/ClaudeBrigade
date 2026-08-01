"""LiteLLM child process supervisor with blue-green catalog replacement.

Manages the lifecycle of one or more LiteLLM proxy child processes.
A *generation* is a versioned catalog of model definitions compiled from
the registry.  Each generation may have one active deployment (a running
LiteLLM proxy process on a specific port).

Blue-green replacement:

1. A new generation is created in *staging* state.
2. A new LiteLLM child is started on a free port.
3. Health probes verify the child is ready.
4. The generation is *activated* (previous generation → *retired*).
5. The old child is *drained* and eventually killed once all its bindings
   are released.

Safety properties:

- Only one generation is *active* at a time.
- A generation cannot be activated unless at least one deployment is healthy.
- Retired generations are not killed until all pinned agent bindings release.
- Crash detection reaps dead children and updates deployment state.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import signal
import sys
import time
from typing import Any

from enhanced_router.state import RouteState

LOGGER = logging.getLogger("claude-enhanced-router")

# Environment variable for the internal LiteLLM master key
_LITELLM_KEY_ENV = "BRIGADE_LITELLM_KEY"

# Resolve litellm binary from the active venv (same directory as the interpreter)
_LITELLM_BIN = str(pathlib.Path(sys.executable).parent / "litellm")

# Verify it exists at import time so failures are loud and early
if not pathlib.Path(_LITELLM_BIN).exists():
    LOGGER.warning(
        "LiteLLM executable not found at %s. "
        "Ensure litellm[proxy] is installed in the active venv.",
        _LITELLM_BIN,
    )


class LiteLLMUnhealthyError(RuntimeError):
    """Raised when a LiteLLM child process fails health probes."""


class LiteLLMCrashError(RuntimeError):
    """Raised when a LiteLLM child process exits unexpectedly."""


def _find_free_port(start: int = 18000) -> int:
    """Return the first available TCP port >= *start*.

    Tries up to 400 ports across two ranges (18000-18199, 18200-18399)
    before giving up.  Raises ``OSError`` with a descriptive message
    including the number of concurrent generations to help narrow down
    port exhaustion.
    """
    import socket

    for port in range(start, start + 200):
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            pass

    # Fallback range — allows up to 400 concurrent generations
    for port in range(start + 200, start + 400):
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
            return port
        except OSError:
            pass

    raise OSError(
        f"No free port found after scanning {start}-{start + 399}. "
        f"Too many concurrent LiteLLM generations ({400} ports exhausted)."
    )


async def _health_probe(
    port: int,
    *,
    timeout: float = 5.0,
    retries: int = 30,
    interval: float = 0.2,
) -> bool:
    """Probe a LiteLLM child's ``/health`` endpoint.

    Returns ``True`` if the child responds with HTTP 200 within the retry
    budget.  ``False`` if all retries are exhausted.
    """
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        for attempt in range(retries):
            try:
                resp = await client.get(
                    f"http://127.0.0.1:{port}/health",
                )
                if resp.status_code == 200:
                    return True
            except (httpx.HTTPError, OSError):
                pass
            if attempt < retries - 1:
                await asyncio.sleep(interval)
    return False


async def _kill_process(pid: int, sig: int = signal.SIGTERM) -> None:
    """Send a signal to a process, ignoring ``ProcessLookupError`` and ``PermissionError``."""
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass  # already dead or not ours


async def _wait_process(
    pid: int, timeout: float = 10.0
) -> int | None:
    """Wait for a child process to exit within *timeout* seconds.

    Returns the exit code, or ``None`` if the process did not exit in time.
    """
    import asyncio

    def _waiter() -> int | None:
        try:
            wpid, status = os.waitpid(pid, os.WNOHANG)
            if wpid == pid:
                return status
        except ChildProcessError:
            return 0  # already reaped
        return None

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        code = _waiter()
        if code is not None:
            return _exit_code_or_none(code)
        await asyncio.sleep(0.1)
    return None


def _exit_code_or_none(status: int) -> int | None:
    """Return the exit code from ``os.waitpid`` status, or ``None``."""
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return None


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class LiteLLMSupervisor:
    """Manages LiteLLM child processes with generation-based lifecycle.

    Usage::

        supervisor = LiteLLMSupervisor(state, config_dir)
        await supervisor.start_generation(registry_hash, models, reason)
        # ... supervisor is now running one active LiteLLM child
        await supervisor.reload(registry_hash, models, reason)
        # ... old child is draining, new child is active
    """

    def __init__(
        self,
        state: RouteState,
        config_dir: str | pathlib.Path,
        litellm_key: str | None = None,
    ) -> None:
        self._state = state
        self._config_dir = pathlib.Path(config_dir)
        self._litellm_key = litellm_key or os.environ.get(_LITELLM_KEY_ENV, "")
        self._active_generation: int | None = None
        self._active_port: int | None = None
        self._active_pid: int | None = None
        self._draining_generation: int | None = None
        self._draining_pid: int | None = None
        self._processes: dict[int, asyncio.subprocess.Process] = {}
        self._crash_count: int = 0
        self._last_crash_at: float = 0.0
        self._crash_cooldown: float = 0.0  # current backoff duration
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_state = "disabled"
        self._shutting_down = False
        self._request_activity: dict[int, dict[str, tuple[bool, float]]] = {}
        self._last_start_args: tuple[str, dict[str, Any], str] | None = None
        self._restart_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def crash_count(self) -> int:
        """Number of consecutive LiteLLM child crashes since last successful start."""
        return self._crash_count

    @property
    def active_port(self) -> int | None:
        """The port of the currently active LiteLLM child, or None."""
        return self._active_port

    @property
    def active_generation(self) -> int | None:
        """The generation number of the currently active catalog, or None."""
        return self._active_generation

    @property
    def lifecycle_state(self) -> str:
        """Current serialized supervisor lifecycle state."""
        return self._lifecycle_state

    def track_request_started(
        self, generation: int, request_id: str, *, streaming: bool,
    ) -> None:
        """Record a request pinned to a generation before it is sent."""
        self._request_activity.setdefault(int(generation), {})[request_id] = (
            streaming, time.monotonic(),
        )

    def track_request_finished(self, generation: int, request_id: str) -> None:
        """Remove a completed or cancelled request from generation activity."""
        requests = self._request_activity.get(int(generation))
        if requests is None:
            return
        requests.pop(request_id, None)
        if not requests:
            self._request_activity.pop(int(generation), None)

    def generation_activity(self, generation: int) -> dict[str, int | float | None]:
        """Return request/stream counts used by blue-green draining."""
        requests = self._request_activity.get(int(generation), {})
        starts = [started for _, started in requests.values()]
        return {
            "active_requests": len(requests),
            "active_streams": sum(streaming for streaming, _ in requests.values()),
            "oldest_request_started_at": min(starts) if starts else None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_generation(
        self,
        registry_hash: str,
        models: dict[str, Any],
        config_text: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Serialize generation startup with all other lifecycle changes."""
        async with self._lifecycle_lock:
            return await self._start_generation_unlocked(
                registry_hash, models, config_text, reason,
            )

    async def _start_generation_unlocked(
        self,
        registry_hash: str,
        models: dict[str, Any],
        config_text: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Create a new generation and start its LiteLLM child.

        The new generation is *activated* immediately (first generation
        skips staging).  Returns metadata about the generation and
        deployment.
        """
        from enhanced_router.litellm_config import config_digest

        self._lifecycle_state = "starting"
        self._shutting_down = False

        digest = config_digest(models)

        # Create generation row (staging)
        generation = self._state.create_litellm_generation(
            registry_hash=registry_hash,
            model_count=sum(
                1
                for s in models.values()
                if getattr(s, "backend", None) == "litellm"
                and getattr(s, "enabled", True)
            ),
            config_digest=digest,
            reason=reason or "initial",
        )

        # Write config file
        config_path = self._config_dir / f"litellm.generated.{generation}.yaml"
        from enhanced_router.litellm_config import write_litellm_config

        write_litellm_config(config_text, config_path)

        # Find a free port
        port = _find_free_port(18000 if not self._active_port else self._active_port + 1)

        # Start the child process
        pid, child_proc = await self._spawn_child(config_path, port, generation)

        # Register deployment
        dep_id = self._state.register_litellm_deployment(
            generation, port, pid
        )

        # Health probe
        healthy = await _health_probe(port)
        if not healthy:
            self._state.update_litellm_deployment(dep_id, status="failed")
            # Clean up the process tracking entry
            self._processes.pop(generation, None)
            # Terminate the failed child
            try:
                await _kill_process(pid, signal.SIGTERM)
                await _wait_process(pid, timeout=3.0)
                await _kill_process(pid, signal.SIGKILL)
                await _wait_process(pid, timeout=2.0)
            except Exception:
                pass
            self._lifecycle_state = "degraded"
            raise LiteLLMUnhealthyError(
                f"LiteLLM generation {generation} failed health probe on port {port}"
            )

        self._state.update_litellm_deployment(dep_id, status="active")

        # Activate — retire previous active generation
        old_active = self._state.get_active_litellm_generation()
        self._state.activate_litellm_generation(generation)

        # Track active/deployment state
        old_pid = self._active_pid
        old_port = self._active_port

        self._active_generation = generation
        self._active_port = port
        self._active_pid = pid
        self._processes[generation] = child_proc
        self._lifecycle_state = "active"
        self._last_start_args = (registry_hash, dict(models), config_text)

        # Start crash monitor for the new child
        asyncio.create_task(self._monitor_child(pid, generation, dep_id))

        # If there was a previous active, move it to draining
        if old_active:
            self._draining_generation = old_active["generation"]
            self._draining_pid = old_pid
            asyncio.create_task(
                self._drain_old(
                    old_gen=old_active["generation"],
                    old_pid=old_pid,
                    old_port=old_port,
                )
            )

        return {
            "generation": generation,
            "deployment": dep_id,
            "port": port,
            "pid": pid,
            "previous_generation": old_active["generation"] if old_active else None,
        }

    async def reload(
        self,
        registry_hash: str,
        models: dict[str, Any],
        config_text: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Reload the LiteLLM catalog with a new generation.

        If the config digest hasn't changed, returns immediately with
        ``changed=False``.  Otherwise creates a new generation and
        activates it.
        """
        from enhanced_router.litellm_config import config_digest

        async with self._lifecycle_lock:
            self._lifecycle_state = "reloading"
            # Skip if nothing changed
            current_active = self._state.get_active_litellm_generation()
            if current_active:
                new_digest = config_digest(models)
                if current_active["config_digest"] == new_digest:
                    self._lifecycle_state = "active"
                    return {
                        "generation": current_active["generation"],
                        "changed": False,
                        "reason": "config unchanged",
                    }

            result = await self._start_generation_unlocked(
                registry_hash=registry_hash,
                models=models,
                config_text=config_text,
                reason=reason,
            )
            result["changed"] = True
            return result

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Kill all child processes and clean up state."""
        async with self._lifecycle_lock:
            self._lifecycle_state = "stopping"
            self._shutting_down = True
            if self._restart_task is not None:
                self._restart_task.cancel()
                self._restart_task = None
            if self._active_pid:
                await _kill_process(self._active_pid)
                await _wait_process(self._active_pid, timeout)
            if self._draining_pid:
                await _kill_process(self._draining_pid)
                await _wait_process(self._draining_pid, timeout)

            # Kill any remaining processes not covered above
            for gen, proc in list(self._processes.items()):
                pid = proc.pid
                if pid and pid not in (self._active_pid, self._draining_pid):
                    await _kill_process(pid)
                    await _wait_process(pid, timeout)
            self._processes.clear()
            self._request_activity.clear()
            self._active_generation = None
            self._active_port = None
            self._active_pid = None
            self._draining_generation = None
            self._draining_pid = None
            self._lifecycle_state = "disabled"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _spawn_child(
        self,
        config_path: pathlib.Path,
        port: int,
        generation: int,
    ) -> tuple[int, asyncio.subprocess.Process]:
        """Start a LiteLLM child process.

        Uses ``asyncio.create_subprocess_exec`` for proper async process
        lifecycle.  Returns ``(pid, process)``.
        """
        env = os.environ.copy()
        env["BRIGADE_LITELLM_KEY"] = self._litellm_key

        log_path = self._config_dir / f"litellm.{generation}.log"
        log_file = open(log_path, "w")
        try:
            proc = await asyncio.create_subprocess_exec(
                _LITELLM_BIN,
                "--config",
                str(config_path),
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        finally:
            log_file.close()

        if proc.pid is None:
            raise LiteLLMCrashError(
                f"Failed to spawn LiteLLM for generation {generation}"
            )

        LOGGER.info(
            "litellm spawned gen=%s pid=%s port=%s",
            generation,
            proc.pid,
            port,
        )
        return (proc.pid, proc)

    async def _monitor_child(
        self,
        pid: int,
        generation: int,
        dep_id: int,
    ) -> None:
        """Monitor a child process for unexpected exit.

        Deployment state machine::

            starting -> active       (health probe succeeds)
            starting -> failed       (health probe fails)
            active     -> draining   (new generation activated)
            active     -> dead       (graceful exit, return code 0)
            active     -> failed     (crash, non-zero exit code)
            draining   -> dead       (drain complete, killed)

        If the child exits without being explicitly killed by the
        drain/supervisor, the deployment is marked dead and new
        LiteLLM bindings will fail with 503.
        """
        proc = self._processes.get(generation)
        if proc is None:
            LOGGER.warning("no process handle for gen=%s, monitoring disabled", generation)
            return

        try:
            returncode = await proc.wait()
        except Exception:
            LOGGER.exception("crash monitor wait error for gen=%s", generation)
            return

        # Process exited -- classify by return code
        if returncode == 0:
            # Clean exit: either intentional drain or unexpected graceful shutdown.
            # Use 'dead' (not 'stopped') because it is a valid CHECK constraint value.
            reason = "drained" if self._draining_pid == pid else "clean_exit"
            self._state.update_litellm_deployment(
                dep_id, status="dead", termination_reason=reason
            )
            # Clear tracking state if this was the active child
            if self._active_pid == pid:
                self._active_generation = None
                self._active_port = None
                self._active_pid = None
            # Remove from process tracking
            self._processes.pop(generation, None)
            # Reset crash tracking on clean exit
            self._crash_count = 0
            self._crash_cooldown = 0.0
            return

        # Non-zero exit code: crash
        self._crash_count += 1
        self._last_crash_at = __import__("time").monotonic()
        # Exponential backoff: 2s, 4s, 8s, 16s, ... capped at 120s
        self._crash_cooldown = min(2.0 * (2 ** (self._crash_count - 1)), 120.0)

        LOGGER.error(
            "litellm child gen=%s pid=%s exited unexpectedly with code=%s",
            generation,
            pid,
            returncode,
        )
        self._state.update_litellm_deployment(
            dep_id, status="failed", termination_reason=f"exit_code_{returncode}"
        )
        # Clear active tracking so bindings fail with 503 instead of stale state
        if self._active_pid == pid:
            self._active_generation = None
            self._active_port = None
            self._active_pid = None
            self._lifecycle_state = "degraded"
            if not self._shutting_down and self._last_start_args is not None:
                if self._restart_task is None or self._restart_task.done():
                    self._restart_task = asyncio.create_task(self._restart_after_crash())

    async def _restart_after_crash(self) -> None:
        """Restart the last known-good catalog with bounded backoff."""
        max_restarts = 5
        while not self._shutting_down and self._crash_count <= max_restarts:
            self._lifecycle_state = "backing_off"
            await asyncio.sleep(max(0.0, self._crash_cooldown))
            if self._shutting_down or not self.can_restart() or self._last_start_args is None:
                return
            registry_hash, models, config_text = self._last_start_args
            try:
                await self.start_generation(
                    registry_hash=registry_hash,
                    models=models,
                    config_text=config_text,
                    reason="automatic crash recovery",
                )
                return
            except Exception as exc:
                self._crash_count += 1
                self._last_crash_at = time.monotonic()
                self._crash_cooldown = min(
                    2.0 * (2 ** max(0, self._crash_count - 1)), 120.0,
                )
                self._lifecycle_state = "degraded"
                LOGGER.error("LiteLLM automatic restart failed: %s", exc)
        self._lifecycle_state = "failed"

    def can_restart(self) -> bool:
        """Check if enough time has passed since the last crash to allow restart.

        Returns ``True`` immediately if there have been no crashes.
        Returns ``True`` if the cooldown period has elapsed since the last crash.
        """
        if self._crash_count == 0:
            return True
        elapsed = time.monotonic() - self._last_crash_at
        return elapsed >= self._crash_cooldown

    async def _drain_old(
        self,
        old_gen: int | None,
        old_pid: int | None,
        old_port: int | None,
        drain_timeout: float = 600.0,
    ) -> None:
        """Drain and kill the old generation.

        Waits for all pinned agent bindings on this generation to be
        released (checked via ``count_active_bindings_for_generation``),
        with a hard *drain_timeout* deadline.  If bindings haven't released
        within the timeout the generation is force-killed regardless.

        The loop checks every 2 seconds up to the deadline.
        """
        if old_pid is None:
            return

        LOGGER.info(
            "draining litellm gen=%s pid=%s port=%s",
            old_gen,
            old_pid,
            old_port,
        )

        # Hard deadline: wait up to drain_timeout for bindings to release
        deadline = asyncio.get_event_loop().time() + drain_timeout
        warned = False

        # Allow a brief initial grace period for in-flight requests
        await asyncio.sleep(min(2.0, drain_timeout / 4))

        while asyncio.get_event_loop().time() < deadline:
            if old_gen is not None:
                bindings = self._state.count_active_bindings_for_generation(old_gen)
                activity = self.generation_activity(old_gen)
                raw_requests = activity["active_requests"]
                raw_streams = activity["active_streams"]
                active_requests = int(raw_requests if isinstance(raw_requests, (int, float)) else 0)
                active_streams = int(raw_streams if isinstance(raw_streams, (int, float)) else 0)
            else:
                bindings = active_requests = active_streams = 0

            if bindings == 0 and active_requests == 0 and active_streams == 0:
                break

            if not warned:
                LOGGER.info(
                    "waiting for gen=%s activity to drain bindings=%s requests=%s streams=%s",
                    old_gen, bindings, active_requests, active_streams,
                )
                warned = True

            await asyncio.sleep(2.0)
        else:
            LOGGER.warning(
                "drain timed out for gen=%s after %.1fs, force-killing; activity=%s",
                old_gen,
                drain_timeout,
                self.generation_activity(old_gen) if old_gen is not None else {},
            )
            if old_gen is not None:
                self._state.fail_litellm_generation_executions(old_gen)

        LOGGER.info(
            "all bindings released for gen=%s, shutting down pid=%s",
            old_gen,
            old_pid,
        )

        # SIGTERM
        await _kill_process(old_pid, signal.SIGTERM)
        code = await _wait_process(old_pid, timeout=5.0)

        if code is None:
            # Force kill
            await _kill_process(old_pid, signal.SIGKILL)
            code = await _wait_process(old_pid, timeout=5.0)

        LOGGER.info(
            "drained litellm gen=%s pid=%s exit=%s",
            old_gen,
            old_pid,
            code,
        )

        self._draining_generation = None
        self._draining_pid = None
