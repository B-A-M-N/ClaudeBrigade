"""Comprehensive tests for LiteLLM child process supervisor with blue-green catalog replacement."""

from __future__ import annotations

import asyncio
import signal
import time
from unittest import mock

import pytest

from enhanced_router.litellm_supervisor import (
    LiteLLMUnhealthyError,
    LiteLLMSupervisor,
    _exit_code_or_none,
    _find_free_port,
    _health_probe,
    _kill_process,
    _wait_process,
)


# =====================================================================
# Helpers
# =====================================================================


def _mock_create_task(captures: list) -> mock.MagicMock:
    """Return a mock for ``asyncio.create_task`` that stores the Future.

    The stored Futures are returned by the mock so callers that ``await``
    them don't trigger \"coroutine was never awaited\" warnings.
    """
    m = mock.MagicMock()
    m.side_effect = lambda coro, **kwargs: (captures.append(coro), mock.Mock(abort=lambda: None))[1]
    return m


def _make_mock_state() -> mock.MagicMock:
    """Build a MagicMock that pretends to be a RouteState.

    All _state calls return harmless defaults so the supervisor's
    internal logic can proceed without touching a database.
    """
    state = mock.MagicMock()
    # Default: no active generation
    state.get_active_litellm_generation.return_value = None
    # count_active_bindings_for_generation: 0 by default
    state.count_active_bindings_for_generation.return_value = 0
    return state


# =====================================================================
# TestLiteLLMHelpers -- standalone helper functions
# =====================================================================


class TestLiteLLMHelpers:
    """Tests for _find_free_port, _health_probe, _kill_process,
    _wait_process, _exit_code_or_none."""

    # ------------------------------------------------------------------
    # _find_free_port
    # ------------------------------------------------------------------

    def test_find_free_port_returns_int(self):
        """_find_free_port returns an int when a socket binds successfully."""
        fake_sock = mock.MagicMock()
        with mock.patch("socket.socket", return_value=fake_sock):
            port = _find_free_port(19000)
        assert isinstance(port, int)
        assert port == 19000

    def test_find_free_port_exhaustion(self):
        """_find_free_port raises OSError when all 200 ports are unavailable."""

        def make_socket(*_args, **_kwargs):
            s = mock.MagicMock()
            s.__enter__ = mock.MagicMock(side_effect=OSError("port in use"))
            s.__exit__ = mock.MagicMock(return_value=None)
            return s

        with mock.patch("socket.socket", make_socket):
            with pytest.raises(OSError, match="No free port found"):
                _find_free_port(20000)

    # ------------------------------------------------------------------
    # _health_probe
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_probe_success(self):
        """_health_probe returns True when the first request gets 200."""
        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_client = mock.AsyncMock()
        fake_client.get.return_value = fake_response
        fake_client.__aenter__ = mock.AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = mock.AsyncMock(return_value=None)

        with mock.patch("httpx.AsyncClient", return_value=fake_client):
            result = await _health_probe(19000, timeout=1.0, retries=2, interval=0.01)

        assert result is True

    @pytest.mark.asyncio
    async def test_health_probe_failure(self):
        """_health_probe returns False when all retries raise HTTPError."""
        fake_client = mock.AsyncMock()
        # Make the mock raise an actual httpx.HTTPError
        import httpx

        fake_client.get.side_effect = httpx.HTTPError("fail")
        fake_client.__aenter__ = mock.AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = mock.AsyncMock(return_value=None)

        with mock.patch("httpx.AsyncClient", return_value=fake_client):
            result = await _health_probe(19000, timeout=1.0, retries=2, interval=0.01)

        assert result is False

    @pytest.mark.asyncio
    async def test_health_probe_sends_master_key_as_bearer_auth(self):
        """An unauthenticated /health request hits LiteLLM's
        user_api_key_auth dependency's "no api key" error branch, which
        unconditionally imports the optional `prisma` package to classify
        the error -- turning a clean 401 into an uncaught ModuleNotFoundError
        (HTTP 500) when prisma isn't installed, masking a healthy child as a
        startup failure. Sending the configured master key must route the
        request around that branch entirely.
        """
        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_client = mock.AsyncMock()
        fake_client.get.return_value = fake_response
        fake_client.__aenter__ = mock.AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = mock.AsyncMock(return_value=None)

        with mock.patch("httpx.AsyncClient", return_value=fake_client):
            await _health_probe(19000, api_key="secret-master-key", timeout=1.0, retries=2, interval=0.01)

        fake_client.get.assert_called_once()
        _, kwargs = fake_client.get.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer secret-master-key"}

    @pytest.mark.asyncio
    async def test_health_probe_without_api_key_sends_no_auth_header(self):
        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_client = mock.AsyncMock()
        fake_client.get.return_value = fake_response
        fake_client.__aenter__ = mock.AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = mock.AsyncMock(return_value=None)

        with mock.patch("httpx.AsyncClient", return_value=fake_client):
            await _health_probe(19000, timeout=1.0, retries=2, interval=0.01)

        _, kwargs = fake_client.get.call_args
        assert kwargs["headers"] == {}

    # ------------------------------------------------------------------
    # _kill_process
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_kill_process_already_dead(self):
        """_kill_process ignores ProcessLookupError (process already gone)."""
        with mock.patch("os.kill", side_effect=ProcessLookupError):
            await _kill_process(99999)
        # No exception raised -- verified by the test not crashing

    @pytest.mark.asyncio
    async def test_kill_process_success(self):
        """_kill_process calls os.kill with the right args."""
        with mock.patch("os.kill") as mock_kill:
            await _kill_process(12345, signal.SIGTERM)
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    # ------------------------------------------------------------------
    # _wait_process
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_wait_process_exits(self):
        """_wait_process returns exit code when waitpid succeeds."""
        # status 0 means child exited with code 0
        exit_status = 0  # WIFEXITED true, WEXITSTATUS == 0
        with mock.patch("os.waitpid", return_value=(12345, exit_status)):
            result = await _wait_process(12345, timeout=1.0)
        assert result == 0

    @pytest.mark.asyncio
    async def test_wait_process_timeout(self):
        """_wait_process returns None when WNOHANG returns zero."""
        # os.waitpid returns (0, 0) meaning no child ready yet -- but
        # we need (0, 0) which has wpid=0 != pid so _waiter returns None.
        # Actually os.waitpid(pid, WNOHANG) returns (0, 0) when no child
        # is ready. The _waiter checks wpid == pid, so 0 != 12345 -> None.
        # But we also need to make sure the event loop time advances so
        # the timeout fires quickly. We'll mock it to return (0, 0) every time.
        with mock.patch("os.waitpid", return_value=(0, 0)):
            with mock.patch("asyncio.get_event_loop") as mock_loop:
                loop_inst = mock.MagicMock()
                # Start high enough to immediately exceed deadline
                loop_inst.time.side_effect = [100.0, 101.0]
                mock_loop.return_value = loop_inst
                result = await _wait_process(12345, timeout=0.1)
        assert result is None

    # ------------------------------------------------------------------
    # _exit_code_or_none
    # ------------------------------------------------------------------

    def test_exit_code_or_none_normal(self):
        """WIFEXITED / WEXITSTATUS path returns the exit code."""
        # Simulate WIFEXITED(status) == True, WEXITSTATUS(status) == 42
        fake_status = 0x001A  # 26 decimal: WIFEXITED=True, WEXITSTATUS=26
        # On Linux: status & 0xFF = exit code when WIFEXITED
        # 0x0010 = WIFEXITED, 0x007F = WEXITSTATUS mask
        # 0x001A = 26: WIFEXITED true, exit code = 26 & 0xFF = 26
        with mock.patch("os.WIFEXITED", return_value=True):
            with mock.patch("os.WEXITSTATUS", return_value=26):
                result = _exit_code_or_none(fake_status)
        assert result == 26

    def test_exit_code_or_none_signaled(self):
        """WIFSIGNALED / WTERMSIG path returns negative signal number."""
        with mock.patch("os.WIFSIGNALED", return_value=True):
            with mock.patch("os.WTERMSIG", return_value=9):
                result = _exit_code_or_none(0x0009)
        assert result == -9  # negative of signal number

    def test_exit_code_or_none_neither(self):
        """Neither WIFEXITED nor WIFSIGNALED -> None."""
        with mock.patch("os.WIFEXITED", return_value=False):
            with mock.patch("os.WIFSIGNALED", return_value=False):
                result = _exit_code_or_none(0x0000)
        assert result is None


# =====================================================================
# TestLiteLLMSupervisor
# =====================================================================


class TestLiteLLMSupervisor:
    """Tests for the LiteLLMSupervisor class."""

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------

    def test_init_sets_defaults(self):
        """All init fields start as None / empty dict."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")
        assert sup._active_generation is None
        assert sup._active_port is None
        assert sup._active_pid is None
        assert sup._draining_generation is None
        assert sup._draining_pid is None
        assert sup._processes == {}
        # active_port / active_generation properties
        assert sup.active_port is None
        assert sup.active_generation is None

    def test_generation_activity_tracks_requests_and_streams(self):
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        sup.track_request_started(7, "request-1", streaming=True)
        sup.track_request_started(7, "request-2", streaming=False)
        activity = sup.generation_activity(7)
        assert activity["active_requests"] == 2
        assert activity["active_streams"] == 1
        assert activity["oldest_request_started_at"] is not None

        sup.track_request_finished(7, "request-1")
        assert sup.generation_activity(7)["active_streams"] == 0
        sup.track_request_finished(7, "request-2")
        assert sup.generation_activity(7)["active_requests"] == 0

    # ------------------------------------------------------------------
    # start_generation -- happy path
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_generation_creates_generation(self):
        """First start_generation creates a generation, spawns child,
        probes health, activates, and returns metadata."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        # _spawn_child returns (pid, proc)
        fake_proc = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc.pid = 54321
        fake_health_probe = mock.AsyncMock(return_value=True)
        # create_litellm_generation returns the generation number
        state.create_litellm_generation.return_value = 1
        # register_litellm_deployment returns deployment id
        state.register_litellm_deployment.return_value = 1

        tasks: list[mock.Mock] = []
        with (
            mock.patch.object(sup, "_spawn_child", new=mock.AsyncMock(return_value=(54321, fake_proc))),
            mock.patch(
                "enhanced_router.litellm_supervisor._health_probe", fake_health_probe
            ),
            mock.patch("enhanced_router.litellm_supervisor._find_free_port", return_value=18000),
            mock.patch(
                "enhanced_router.litellm_config.config_digest", return_value="digest-abc"
            ),
            _mock_create_task(tasks),
        ):
            result = await sup.start_generation(
                registry_hash="hash-1",
                models={"m1": mock.MagicMock(backend="litellm", enabled=True)},
                config_text="model_list:",
                reason="initial",
            )

        # Verify generation was created
        state.create_litellm_generation.assert_called_once()
        # Verify deployment was registered
        state.register_litellm_deployment.assert_called_once()
        # Health probe was called
        fake_health_probe.assert_called_once()
        # Generation was activated
        state.activate_litellm_generation.assert_called_once()
        # Supervisor tracks state
        assert sup._active_generation == 1
        assert sup._active_port == 18000
        assert sup._active_pid == 54321
        # Return structure
        assert result["generation"] == 1
        assert result["port"] == 18000
        assert result["pid"] == 54321
        assert result["previous_generation"] is None
        # Await captured background tasks to avoid "coroutine was never awaited"
        for task in tasks:
            try:
                task.result()  # Futures from our mock don't actually run coroutines
            except Exception:
                pass

    # ------------------------------------------------------------------
    # start_generation -- health failure
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_generation_health_fail(self):
        """When health probe fails, deployment status is 'failed' and
        LiteLLMUnhealthyError is raised."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        fake_proc = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc.pid = 54321

        tasks: list[mock.Mock] = []
        with (
            mock.patch.object(sup, "_spawn_child", new=mock.AsyncMock(return_value=(54321, fake_proc))),
            mock.patch("enhanced_router.litellm_supervisor._health_probe", new=mock.AsyncMock(return_value=False)),
            mock.patch("enhanced_router.litellm_supervisor._kill_process", new=mock.AsyncMock()),
            mock.patch("enhanced_router.litellm_supervisor._wait_process", new=mock.AsyncMock(return_value=None)),
            mock.patch("enhanced_router.litellm_supervisor._find_free_port", return_value=18000),
            mock.patch(
                "enhanced_router.litellm_config.config_digest", return_value="digest-abc"
            ),
            _mock_create_task(tasks),
        ):
            with pytest.raises(LiteLLMUnhealthyError, match="failed health probe"):
                await sup.start_generation(
                    registry_hash="hash-1",
                    models={"m1": mock.MagicMock(backend="litellm", enabled=True)},
                    config_text="model_list:",
                    reason="initial",
                )

        # Deployment status should be "failed"
        # dep_id is the return value of register_litellm_deployment (a MagicMock int)
        call_args = state.update_litellm_deployment.call_args
        assert call_args[1]["status"] == "failed"
        # Generation was created
        state.create_litellm_generation.assert_called_once()

    # ------------------------------------------------------------------
    # start_generation -- blue-green (second activation retires first)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_generation_blue_green(self):
        """A second start_generation retires the first generation and
        starts draining the old child."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        fake_proc1 = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc1.pid = 10001
        fake_proc2 = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc2.pid = 10002

        # First call: no active generation
        state.get_active_litellm_generation.return_value = None

        call_count = [0]

        async def spawn_side_effect(*args, **kwargs):
            call_count[0] += 1
            pid = 10001 if call_count[0] == 1 else 10002
            proc = fake_proc1 if call_count[0] == 1 else fake_proc2
            return (pid, proc)

        # The second call needs get_active_litellm_generation to return the first gen
        state.get_active_litellm_generation.side_effect = [
            None,  # first call
            {"generation": 1, "config_digest": "d1", "status": "active"},  # second call
        ]
        # create_litellm_generation returns the generation number
        state.create_litellm_generation.side_effect = [1, 2]
        # register_litellm_deployment returns deployment ids
        state.register_litellm_deployment.side_effect = [10, 20]

        tasks: list[mock.Mock] = []
        with (
            mock.patch.object(sup, "_spawn_child", new=mock.AsyncMock(side_effect=spawn_side_effect)),
            mock.patch("enhanced_router.litellm_supervisor._health_probe", new=mock.AsyncMock(return_value=True)),
            mock.patch("enhanced_router.litellm_supervisor._find_free_port", side_effect=[18000, 18001]),
            mock.patch(
                "enhanced_router.litellm_config.config_digest", return_value="digest-abc"
            ),
            _mock_create_task(tasks),
        ):
            # First start
            result1 = await sup.start_generation(
                registry_hash="hash-1",
                models={"m1": mock.MagicMock(backend="litellm", enabled=True)},
                config_text="model_list:",
                reason="initial",
            )
            assert result1["generation"] == 1
            assert result1["previous_generation"] is None

            # Second start
            result2 = await sup.start_generation(
                registry_hash="hash-2",
                models={"m2": mock.MagicMock(backend="litellm", enabled=True)},
                config_text="model_list:",
                reason="reload",
            )
            assert result2["generation"] == 2
            assert result2["previous_generation"] == 1

        # Verify old generation was moved to draining
        assert sup._draining_generation == 1
        assert sup._draining_pid == 10001
        # Cleanup captured background tasks
        for task in tasks:
            try:
                task.result()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # reload -- unchanged
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reload_unchanged(self):
        """When config_digest hasn't changed, reload returns changed=False."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        state.get_active_litellm_generation.return_value = {
            "generation": 1,
            "config_digest": "same-digest",
            "status": "active",
        }

        with mock.patch(
            "enhanced_router.litellm_config.config_digest", return_value="same-digest"
        ):
            result = await sup.reload(
                registry_hash="hash-1",
                models={"m1": mock.MagicMock(backend="litellm", enabled=True)},
                config_text="model_list:",
                reason="reload",
            )

        assert result["changed"] is False
        assert result["generation"] == 1
        assert result["reason"] == "config unchanged"
        # start_generation must NOT have been called
        state.create_litellm_generation.assert_not_called()

    # ------------------------------------------------------------------
    # reload -- changed
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reload_changed(self):
        """When config_digest changed, reload starts a new generation."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        state.get_active_litellm_generation.return_value = {
            "generation": 1,
            "config_digest": "old-digest",
            "status": "active",
        }

        fake_proc = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc.pid = 20000

        tasks: list[mock.Mock] = []
        with (
            mock.patch.object(sup, "_spawn_child", new=mock.AsyncMock(return_value=(20000, fake_proc))),
            mock.patch("enhanced_router.litellm_supervisor._health_probe", new=mock.AsyncMock(return_value=True)),
            mock.patch("enhanced_router.litellm_supervisor._find_free_port", return_value=18001),
            mock.patch(
                "enhanced_router.litellm_config.config_digest", return_value="new-digest"
            ),
            _mock_create_task(tasks),
        ):
            result = await sup.reload(
                registry_hash="hash-2",
                models={"m2": mock.MagicMock(backend="litellm", enabled=True)},
                config_text="new_model_list:",
                reason="config changed",
            )

        assert result["changed"] is True
        state.create_litellm_generation.assert_called_once()
        # Cleanup captured background tasks
        for task in tasks:
            try:
                task.result()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_reload_passes_referenced_ids_through_to_config_digest(self):
        """A scope-only change (same models, narrower/wider referenced_ids)
        must still be detectable -- reload must feed the caller's
        referenced_ids into config_digest rather than always computing an
        unscoped digest, or a new run's previously-excluded models would
        never trigger a fresh generation.
        """
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        state.get_active_litellm_generation.return_value = {
            "generation": 1,
            "config_digest": "old-digest",
            "status": "active",
        }

        with mock.patch(
            "enhanced_router.litellm_config.config_digest", return_value="old-digest"
        ) as mock_digest:
            result = await sup.reload(
                registry_hash="hash-1",
                models={"m1": mock.MagicMock(backend="litellm", enabled=True)},
                config_text="model_list:",
                reason="reload",
                referenced_ids={"m1"},
            )

        mock_digest.assert_called_once_with(
            {"m1": mock.ANY}, referenced_ids={"m1"},
        )
        assert result["changed"] is False

    @pytest.mark.asyncio
    async def test_restart_after_crash_preserves_referenced_ids(self):
        """Crash recovery replays the last known-good generation via
        start_generation directly (bypassing reload's digest check), so it
        must carry the same referenced_ids scope forward too -- otherwise a
        restarted child would silently widen back out to every model.
        """
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")
        sup._last_start_args = ("hash-1", {"m1": mock.MagicMock()}, "model_list:", {"m1"})

        with mock.patch.object(
            sup, "start_generation", new=mock.AsyncMock(return_value={"generation": 2}),
        ) as mock_start:
            await sup._restart_after_crash()

        mock_start.assert_called_once_with(
            registry_hash="hash-1",
            models={"m1": mock.ANY},
            config_text="model_list:",
            reason="automatic crash recovery",
            referenced_ids={"m1"},
        )

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_shutdown_kills_processes(self):
        """shutdown kills active_pid and draining_pid."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")
        sup._active_pid = 10001
        sup._draining_pid = 10002

        with (
            mock.patch("enhanced_router.litellm_supervisor._kill_process", new=mock.AsyncMock()) as mock_kill,
            mock.patch("enhanced_router.litellm_supervisor._wait_process", new=mock.AsyncMock(return_value=0)) as mock_wait,
        ):
            await sup.shutdown(timeout=5.0)

        # active_pid killed
        assert mock_kill.call_count == 2
        assert mock_wait.call_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_remaining_processes(self):
        """shutdown also kills processes in _processes that are not
        active_pid or draining_pid."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")
        sup._active_pid = 10001
        sup._draining_pid = 10002

        extra_proc = mock.MagicMock(spec=asyncio.subprocess.Process)
        extra_proc.pid = 10003
        sup._processes[99] = extra_proc

        with (
            mock.patch("enhanced_router.litellm_supervisor._kill_process", new=mock.AsyncMock()) as mock_kill,
            mock.patch("enhanced_router.litellm_supervisor._wait_process", new=mock.AsyncMock(return_value=0)),
        ):
            await sup.shutdown(timeout=5.0)

        # Kill calls: active + draining + extra
        assert mock_kill.call_count == 3
        # Processes should be cleared
        assert sup._processes == {}

    # ------------------------------------------------------------------
    # _monitor_child
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_monitor_child_clean_exit(self):
        """When child exits with code 0 and is NOT the draining pid,
        deployment is marked 'dead' with reason 'clean_exit'."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        fake_proc = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc.wait = mock.AsyncMock(return_value=0)

        sup._processes[1] = fake_proc
        await sup._monitor_child(pid=9999, generation=1, dep_id=42)

        state.update_litellm_deployment.assert_called_with(
            42, status="dead", termination_reason="clean_exit"
        )

    @pytest.mark.asyncio
    async def test_monitor_child_drained_exit(self):
        """When child exits with code 0 AND matches draining_pid,
        deployment is marked 'dead' with reason 'drained'."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        fake_proc = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc.wait = mock.AsyncMock(return_value=0)

        sup._processes[1] = fake_proc
        sup._draining_pid = 9999
        await sup._monitor_child(pid=9999, generation=1, dep_id=42)

        state.update_litellm_deployment.assert_called_with(
            42, status="dead", termination_reason="drained"
        )

    @pytest.mark.asyncio
    async def test_monitor_child_crash(self):
        """When child exits with non-zero code, deployment is marked 'failed'."""
        state = _make_mock_state()
        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        fake_proc = mock.MagicMock(spec=asyncio.subprocess.Process)
        fake_proc.wait = mock.AsyncMock(return_value=1)

        sup._processes[1] = fake_proc
        await sup._monitor_child(pid=9999, generation=1, dep_id=42)

        state.update_litellm_deployment.assert_called_with(
            42, status="failed", termination_reason="exit_code_1"
        )

    # ------------------------------------------------------------------
    # _drain_old -- bindings release quickly
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_drain_old_no_bindings(self):
        """When count_active_bindings_for_generation returns 0, drain
        proceeds to kill immediately (after min drain sleep)."""
        state = _make_mock_state()
        state.count_active_bindings_for_generation.return_value = 0

        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        with (
            mock.patch("enhanced_router.litellm_supervisor._kill_process", new=mock.AsyncMock()) as mock_kill,
            mock.patch("enhanced_router.litellm_supervisor._wait_process", new=mock.AsyncMock(return_value=0)) as mock_wait,
            mock.patch("asyncio.sleep", new=mock.AsyncMock()),
        ):
            await sup._drain_old(old_gen=1, old_pid=10001, old_port=18000)

        # Kill and wait called once each (SIGTERM)
        mock_kill.assert_called_once()
        mock_wait.assert_called_once()
        # Draining state cleared
        assert sup._draining_generation is None
        assert sup._draining_pid is None

    @pytest.mark.asyncio
    async def test_drain_old_timeout(self):
        """When bindings never release past max_iterations, force-kill is triggered."""
        state = _make_mock_state()
        # Always return 1 binding -- drain loop never finds 0
        state.count_active_bindings_for_generation.return_value = 1

        sup = LiteLLMSupervisor(state, "/tmp/litellm-test-config")

        async def wait_side_effect(pid, timeout=10.0):
            # First call (SIGTERM path): timeout -> triggers SIGKILL
            # Second call (after SIGKILL): success
            wait_side_effect.call_count += 1
            if wait_side_effect.call_count == 1:
                return None
            return 0

        wait_side_effect.call_count = 0

        with (
            mock.patch("enhanced_router.litellm_supervisor._kill_process", new=mock.AsyncMock()) as mock_kill,
            mock.patch("enhanced_router.litellm_supervisor._wait_process", new=mock.AsyncMock(side_effect=wait_side_effect)),
            mock.patch("asyncio.sleep", new=mock.AsyncMock()),
            mock.patch("logging.Logger.warning", mock.MagicMock()),
        ):
            await sup._drain_old(old_gen=5, old_pid=20000, old_port=18000, drain_timeout=30.0)

        # SIGKILL must have been called because SIGTERM timed out
        assert mock_kill.call_count >= 2
        # Verify SIGKILL was among the signals sent
        kill_signals = [call[0][1] for call in mock_kill.call_args_list]
        assert signal.SIGKILL in kill_signals

    # ------------------------------------------------------------------
    # start_generation -- health failure cleans up processes
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_generation_health_fail_cleans_up_processes(self):
        """When health probe fails, _processes should not have a stale entry."""
        state = mock.MagicMock()
        gen = 42
        state.create_litellm_generation.return_value = gen
        state.register_litellm_deployment.return_value = 100

        sup = LiteLLMSupervisor(state, "/tmp")

        async def fake_spawn(*args, **kwargs):
            return (99999, mock.MagicMock())

        with (
            mock.patch.object(sup, "_spawn_child", side_effect=fake_spawn),
            mock.patch(
                "enhanced_router.litellm_supervisor._health_probe",
                return_value=False,
            ),
        ):
            with pytest.raises(LiteLLMUnhealthyError):
                await sup.start_generation(
                    registry_hash="h1",
                    models={},
                    config_text="",
                    reason="health-fail-test",
                )

        # _processes should NOT have the failed generation
        assert gen not in sup._processes

    # ------------------------------------------------------------------
    # _kill_process -- permission error handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_kill_process_permission_error_does_not_crash(self):
        """_kill_process should handle PermissionError gracefully."""
        # Should not raise for a nonexistent process
        await _kill_process(999999, signal.SIGTERM)

        # Also should not raise for PermissionError
        with mock.patch("os.kill", side_effect=PermissionError):
            await _kill_process(12345, signal.SIGTERM)

    # ------------------------------------------------------------------
    # Crash tracking
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_crash_tracking_increments_on_monitor(self):
        """_monitor_child with non-zero exit increments crash_count."""
        state = mock.MagicMock()
        sup = LiteLLMSupervisor(state, "/tmp")

        # Create a mock process that returns non-zero
        proc = mock.AsyncMock()
        proc.wait.return_value = 1
        sup._processes[1] = proc

        await sup._monitor_child(pid=9999, generation=1, dep_id=100)

        assert sup._crash_count == 1
        assert sup._crash_cooldown > 0

    @pytest.mark.asyncio
    async def test_clean_exit_resets_crash_count(self):
        """A successful (zero) exit resets crash_count to 0."""
        state = mock.MagicMock()
        sup = LiteLLMSupervisor(state, "/tmp")
        sup._crash_count = 3
        sup._crash_cooldown = 16.0

        proc = mock.AsyncMock()
        proc.wait.return_value = 0
        sup._processes[1] = proc

        await sup._monitor_child(pid=9999, generation=1, dep_id=100)

        assert sup._crash_count == 0
        assert sup._crash_cooldown == 0.0

    def test_can_restart_no_crashes(self):
        """can_restart returns True when there have been no crashes."""
        state = mock.MagicMock()
        sup = LiteLLMSupervisor(state, "/tmp")
        assert sup.can_restart() is True

    def test_can_restart_during_cooldown(self):
        """can_restart returns False during cooldown period."""
        state = mock.MagicMock()
        sup = LiteLLMSupervisor(state, "/tmp")
        sup._crash_count = 2
        sup._crash_cooldown = 60.0
        # _last_crash_at defaults to 0.0, so elapsed is large --
        # set it to now to simulate recent crash
        sup._last_crash_at = time.monotonic()
        assert sup.can_restart() is False
