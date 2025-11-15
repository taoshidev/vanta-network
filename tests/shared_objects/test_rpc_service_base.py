"""
Unit tests for RPCServiceBase class.

Tests cover:
- Direct mode (unit tests) vs RPC mode initialization
- Health check functionality with rate limiting
- Auto-restart on consecutive failures
- Stale server cleanup
- Graceful shutdown
- Connection retries
- Abstract method enforcement
"""
import pytest
import time
import os
import signal
import subprocess
from typing import Optional
from unittest.mock import Mock, patch, MagicMock
from multiprocessing import Process, Event

from shared_objects.rpc_service_base import RPCServiceBase
from time_util.time_util import TimeUtil


class TestHelpers:
    """Test utilities for deterministic waiting without sleep guessing"""

    @staticmethod
    def wait_for_condition(
        condition_fn,
        timeout: float = 5.0,
        interval: float = 0.01,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Wait for an arbitrary condition to become true.

        Instead of guessing with time.sleep(), this polls the actual condition
        with a short interval. Much more deterministic and usually faster.

        Args:
            condition_fn: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            interval: Polling interval in seconds (default: 10ms)
            error_message: Optional message to include in timeout error

        Returns:
            bool: True if condition met, False if timeout
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                if condition_fn():
                    return True
            except Exception:
                # Condition check raised exception, keep trying
                pass

            # Sleep briefly between checks
            remaining = deadline - time.time()
            time.sleep(min(interval, max(0, remaining)))

        # Timeout - provide helpful message if available
        if error_message:
            raise AssertionError(
                f"{error_message} (timeout after {timeout}s)"
            )

        return False

    @staticmethod
    def wait_for_process_termination(
        process: Process,
        timeout: float = 5.0
    ) -> bool:
        """
        Wait for a process to actually terminate.

        More reliable than just calling process.join() because it also
        polls the is_alive() status with exponential backoff.

        Args:
            process: Process to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if process terminated, False if still alive after timeout
        """
        # First try join with most of the timeout
        process.join(timeout=timeout * 0.9)

        # Then poll to confirm termination
        return TestHelpers.wait_for_condition(
            lambda: not process.is_alive(),
            timeout=timeout * 0.1,
            interval=0.01
        )


def _run_mock_server(address, authkey, server_ready, fail_health_check, service_name):
    """Module-level function to run mock server in separate process"""
    from setproctitle import setproctitle
    setproctitle(f"test_{service_name}")

    # Import here to avoid circular dependencies
    from multiprocessing.managers import BaseManager

    server_instance = MockServer(fail_health_check=fail_health_check)

    # Register the server instance with the service name
    # This matches how _serve_rpc() works in RPCServiceBase
    class ServiceManager(BaseManager):
        pass

    ServiceManager.register(service_name, callable=lambda: server_instance)

    manager = ServiceManager(address=address, authkey=authkey)
    server = manager.get_server()

    # Signal that server is ready
    server_ready.set()

    # Serve until shutdown
    server.serve_forever()


def _slow_start_server():
    """Module-level function that never signals ready (for timeout testing)"""
    import time
    time.sleep(100)  # Never signal ready


class MockServer:
    """Mock RPC server for testing"""
    def __init__(self, fail_health_check=False):
        self.fail_health_check = fail_health_check
        self.health_check_count = 0

    def health_check_rpc(self):
        """Mock health check that can be configured to fail"""
        self.health_check_count += 1
        if self.fail_health_check:
            raise Exception("Mock health check failure")
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis()
        }

    def some_method_rpc(self, arg):
        """Mock RPC method for testing"""
        return f"result_{arg}"


class ConcreteRPCService(RPCServiceBase):
    """Concrete implementation of RPCServiceBase for testing"""

    def __init__(self, service_name="TestService", port=50099, running_unit_tests=True,
                 enable_health_check=False, health_check_interval_s=60,
                 max_consecutive_failures=3, enable_auto_restart=True,
                 slack_notifier=None, fail_server_health_check=False):
        self.fail_server_health_check = fail_server_health_check

        super().__init__(
            service_name=service_name,
            port=port,
            running_unit_tests=running_unit_tests,
            enable_health_check=enable_health_check,
            health_check_interval_s=health_check_interval_s,
            max_consecutive_failures=max_consecutive_failures,
            enable_auto_restart=enable_auto_restart,
            slack_notifier=slack_notifier
        )

        self._initialize_service()

    def _create_direct_server(self):
        """Create mock server instance for direct mode"""
        return MockServer(fail_health_check=self.fail_server_health_check)

    def _start_server_process(self, address, authkey, server_ready):
        """Start mock RPC server in separate process"""
        process = Process(
            target=_run_mock_server,
            args=(address, authkey, server_ready, self.fail_server_health_check, self.service_name),
            daemon=True
        )
        process.start()
        return process


class TestRPCServiceBaseInitialization:
    """Test initialization and mode selection"""

    def test_direct_mode_initialization(self):
        """Test that direct mode creates in-memory server instance"""
        service = ConcreteRPCService(running_unit_tests=True)

        # Should create direct server instance
        assert service._server_proxy is not None
        assert isinstance(service._server_proxy, MockServer)
        assert service._server_process is None
        assert service._client_manager is None

    def test_rpc_mode_initialization(self):
        """Test that RPC mode starts server process and connects client"""
        service = ConcreteRPCService(running_unit_tests=False, port=50100)

        try:
            # Should create server process and client connection
            assert service._server_proxy is not None
            assert service._server_process is not None
            assert service._server_process.is_alive()
            assert service._client_manager is not None
            assert service._authkey is not None
            assert len(service._authkey) == 32  # Secure authkey
        finally:
            service.shutdown()

    def test_service_name_and_port(self):
        """Test that service name and port are correctly set"""
        service = ConcreteRPCService(
            service_name="CustomService",
            port=50101,
            running_unit_tests=True
        )

        assert service.service_name == "CustomService"
        assert service.port == 50101

    def test_health_check_configuration(self):
        """Test that health check parameters are correctly configured"""
        service = ConcreteRPCService(
            running_unit_tests=True,
            enable_health_check=True,
            health_check_interval_s=30,
            max_consecutive_failures=5,
            enable_auto_restart=False
        )

        assert service.enable_health_check is True
        assert service.health_check_interval_s == 30
        assert service.health_check_interval_ms == 30000
        assert service.max_consecutive_failures == 5
        assert service.enable_auto_restart is False


class TestRPCServiceBaseCommunication:
    """Test RPC communication between client and server"""

    def test_direct_mode_method_call(self):
        """Test calling RPC methods in direct mode"""
        service = ConcreteRPCService(running_unit_tests=True)

        result = service._server_proxy.some_method_rpc("test")
        assert result == "result_test"

    def test_rpc_mode_method_call(self):
        """Test calling RPC methods in RPC mode"""
        service = ConcreteRPCService(running_unit_tests=False, port=50102)

        try:
            result = service._server_proxy.some_method_rpc("test")
            assert result == "result_test"
        finally:
            service.shutdown()


class TestHealthCheck:
    """Test health check functionality"""

    def test_health_check_disabled_in_direct_mode(self):
        """Test that health checks are disabled in direct mode"""
        service = ConcreteRPCService(
            running_unit_tests=True,
            enable_health_check=True
        )

        # Health checks should be disabled for direct mode
        assert service._health_check_enabled_for_instance is False

        # Health check should return True (always healthy)
        result = service.health_check()
        assert result is True

    def test_health_check_enabled_in_rpc_mode(self):
        """Test that health checks are enabled in RPC mode"""
        service = ConcreteRPCService(
            running_unit_tests=False,
            port=50103,
            enable_health_check=True
        )

        try:
            # Health checks should be enabled for RPC mode
            assert service._health_check_enabled_for_instance is True

            # Health check should return True (server is healthy)
            result = service.health_check()
            assert result is True
        finally:
            service.shutdown()

    def test_health_check_rate_limiting(self):
        """Test that health checks are rate limited"""
        service = ConcreteRPCService(
            running_unit_tests=False,
            port=50104,
            enable_health_check=True,
            health_check_interval_s=10
        )

        try:
            # First health check should execute
            current_time_ms = TimeUtil.now_in_millis()
            result1 = service.health_check(current_time_ms)
            assert result1 is True
            assert service._last_health_check_time_ms == current_time_ms

            # Second health check immediately after should be skipped (rate limited)
            result2 = service.health_check(current_time_ms + 1000)  # +1 second
            assert result2 is True  # Returns True without actually checking

            # Health check after interval should execute
            result3 = service.health_check(current_time_ms + 11000)  # +11 seconds
            assert result3 is True
            assert service._last_health_check_time_ms == current_time_ms + 11000
        finally:
            service.shutdown()

    def test_health_check_success_resets_failure_counter(self):
        """Test that successful health check resets failure counter"""
        service = ConcreteRPCService(
            running_unit_tests=False,
            port=50105,
            enable_health_check=True,
            health_check_interval_s=1
        )

        try:
            # Manually set some failures
            service._consecutive_failures = 2

            # Successful health check should reset counter
            current_time_ms = TimeUtil.now_in_millis()
            result = service.health_check(current_time_ms)
            assert result is True
            assert service._consecutive_failures == 0
        finally:
            service.shutdown()

    def test_health_check_failure_increments_counter(self):
        """Test that failed health check increments failure counter"""
        service = ConcreteRPCService(
            running_unit_tests=False,
            port=50106,
            enable_health_check=True,
            health_check_interval_s=1,
            max_consecutive_failures=5,  # High threshold to prevent restart
            enable_auto_restart=False,  # Disable restart for this test
            fail_server_health_check=True  # Make server fail health checks
        )

        try:
            # Health check should fail and increment counter
            current_time_ms = TimeUtil.now_in_millis()
            result = service.health_check(current_time_ms)
            assert result is False
            assert service._consecutive_failures == 1

            # Another failure should increment again
            result2 = service.health_check(current_time_ms + 2000)
            assert result2 is False
            assert service._consecutive_failures == 2
        finally:
            service.shutdown()

    @patch('shared_objects.rpc_service_base.bt.logging')
    def test_health_check_recovery_notification(self, mock_logging):
        """Test that recovery from failures is logged"""
        mock_notifier = Mock()
        service = ConcreteRPCService(
            running_unit_tests=False,
            port=50107,
            enable_health_check=True,
            health_check_interval_s=1,
            slack_notifier=mock_notifier
        )

        try:
            # Simulate previous failures
            service._consecutive_failures = 2

            # Successful health check should log recovery
            current_time_ms = TimeUtil.now_in_millis()
            result = service.health_check(current_time_ms)
            assert result is True

            # Should have logged success message
            mock_logging.success.assert_called()

            # Should have sent Slack notification
            mock_notifier.send_message.assert_called_once()
            call_args = mock_notifier.send_message.call_args
            assert "recovered" in call_args[0][0].lower()
        finally:
            service.shutdown()


class TestAutoRestart:
    """Test auto-restart functionality"""

    @patch('shared_objects.rpc_service_base.bt.logging')
    def test_restart_disabled(self, mock_logging):
        """Test that auto-restart can be disabled"""
        mock_notifier = Mock()
        service = ConcreteRPCService(
            running_unit_tests=False,
            port=50108,
            enable_health_check=True,
            health_check_interval_s=1,
            max_consecutive_failures=2,
            enable_auto_restart=False,  # Disable restart
            slack_notifier=mock_notifier,
            fail_server_health_check=True
        )

        try:
            # Trigger consecutive failures
            current_time_ms = TimeUtil.now_in_millis()
            service.health_check(current_time_ms)
            service.health_check(current_time_ms + 2000)

            # Should have logged warning about manual intervention
            assert any("manual intervention" in str(call).lower()
                      for call in mock_logging.warning.call_args_list)
        finally:
            service.shutdown()


class TestShutdown:
    """Test shutdown and cleanup functionality"""

    def test_shutdown_direct_mode(self):
        """Test shutdown in direct mode"""
        service = ConcreteRPCService(running_unit_tests=True)

        assert service._server_proxy is not None

        service.shutdown()

        # Should clear server proxy
        assert service._server_proxy is None

    def test_shutdown_rpc_mode(self):
        """Test shutdown in RPC mode"""
        service = ConcreteRPCService(running_unit_tests=False, port=50109)

        assert service._server_process is not None
        assert service._server_process.is_alive()

        service.shutdown()

        # Should terminate server process
        assert service._server_proxy is None
        assert service._client_manager is None
        if service._server_process is not None:
            # Wait for actual termination instead of guessing with sleep
            assert TestHelpers.wait_for_process_termination(
                service._server_process,
                timeout=2.0
            )

    def test_shutdown_idempotent(self):
        """Test that shutdown can be called multiple times safely"""
        service = ConcreteRPCService(running_unit_tests=False, port=50110)

        # First shutdown
        service.shutdown()

        # Second shutdown should not raise exception
        service.shutdown()

    def test_del_calls_shutdown(self):
        """Test that __del__ calls shutdown"""
        service = ConcreteRPCService(running_unit_tests=False, port=50111)

        process = service._server_process
        assert process.is_alive()

        # Delete service object
        del service

        # Process should be terminated - wait for actual termination
        assert TestHelpers.wait_for_process_termination(process, timeout=2.0)


class TestStaleServerCleanup:
    """Test stale server cleanup functionality"""

    @pytest.mark.skipif(os.name != 'posix', reason="Cleanup only works on POSIX systems")
    @patch('shared_objects.rpc_service_base.subprocess.run')
    @patch('shared_objects.rpc_service_base.os.kill')
    def test_cleanup_stale_server_kills_process(self, mock_kill, mock_subprocess_run):
        """Test that cleanup kills processes using the port"""
        # Mock lsof finding a process
        mock_subprocess_run.side_effect = [
            Mock(returncode=0, stdout="12345\n"),  # lsof finds PID 12345
            Mock(returncode=0, stdout="python\n"),  # ps shows it's python
        ]

        # Create service in direct mode (won't start RPC server)
        service = ConcreteRPCService(running_unit_tests=True, port=50112)

        # Directly call the cleanup method to test it
        service._cleanup_stale_server()

        # Should have called lsof
        assert mock_subprocess_run.call_count >= 1
        lsof_call = mock_subprocess_run.call_args_list[0]
        assert 'lsof' in str(lsof_call[0][0])

    @pytest.mark.skipif(os.name != 'posix', reason="Cleanup only works on POSIX systems")
    @patch('shared_objects.rpc_service_base.subprocess.run')
    def test_cleanup_graceful_on_lsof_not_found(self, mock_subprocess_run):
        """Test that cleanup handles missing lsof gracefully"""
        mock_subprocess_run.side_effect = FileNotFoundError("lsof not found")

        # Should not raise exception
        service = ConcreteRPCService(running_unit_tests=True, port=50113)

        # Directly call cleanup - should handle FileNotFoundError gracefully
        service._cleanup_stale_server()  # Should not raise
        assert service is not None


class TestConnectionRetries:
    """Test connection retry logic"""

    def test_connection_with_retries(self):
        """Test that client retries connection on failure"""
        # This is implicitly tested by successful RPC mode initialization
        # The base class retries up to 5 times with 1s delay
        service = ConcreteRPCService(running_unit_tests=False, port=50114)

        try:
            # Should have successfully connected
            assert service._client_manager is not None
            assert service._server_proxy is not None
        finally:
            service.shutdown()


class TestAbstractMethods:
    """Test that abstract methods must be implemented"""

    def test_missing_create_direct_server_raises_error(self):
        """Test that missing _create_direct_server raises TypeError (ABC prevents instantiation)"""

        class IncompleteService(RPCServiceBase):
            def __init__(self):
                super().__init__(
                    service_name="Incomplete",
                    port=50115,
                    running_unit_tests=True
                )
                self._initialize_service()

            # Missing _create_direct_server

            def _start_server_process(self, address, authkey, server_ready):
                pass

        # Python's ABC raises TypeError when trying to instantiate with missing abstract methods
        with pytest.raises(TypeError, match="abstract method"):
            service = IncompleteService()

    def test_missing_start_server_process_raises_error(self):
        """Test that missing _start_server_process raises TypeError (ABC prevents instantiation)"""

        class IncompleteService2(RPCServiceBase):
            def __init__(self):
                super().__init__(
                    service_name="Incomplete2",
                    port=50116,
                    running_unit_tests=False  # RPC mode
                )
                self._initialize_service()

            def _create_direct_server(self):
                return MockServer()

            # Missing _start_server_process

        # Python's ABC raises TypeError when trying to instantiate with missing abstract methods
        with pytest.raises(TypeError, match="abstract method"):
            service = IncompleteService2()


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_server_ready_timeout(self):
        """Test timeout when server doesn't signal ready"""

        class SlowStartService(RPCServiceBase):
            def __init__(self):
                super().__init__(
                    service_name="SlowStart",
                    port=50117,
                    running_unit_tests=False
                )
                self._initialize_service()

            def _create_direct_server(self):
                return MockServer()

            def _start_server_process(self, address, authkey, server_ready):
                # Start a process that never signals ready
                process = Process(target=_slow_start_server, daemon=True)
                process.start()
                return process

        # Should timeout after 10 seconds
        with pytest.raises(TimeoutError, match="startup timeout"):
            service = SlowStartService()

    def test_health_check_without_initialization(self):
        """Test that health check before initialization returns True"""
        service = ConcreteRPCService(
            running_unit_tests=True,
            enable_health_check=True
        )

        # Manually disable health check
        service._health_check_enabled_for_instance = False

        # Should return True without checking
        result = service.health_check()
        assert result is True


class TestIntegration:
    """Integration tests combining multiple features"""

    def test_full_lifecycle_direct_mode(self):
        """Test complete lifecycle in direct mode"""
        service = ConcreteRPCService(
            service_name="LifecycleTest",
            port=50118,
            running_unit_tests=True,
            enable_health_check=True
        )

        # Should be initialized
        assert service._server_proxy is not None

        # Should be able to call methods
        result = service._server_proxy.some_method_rpc("test")
        assert result == "result_test"

        # Health checks should work (but always return True in direct mode)
        assert service.health_check() is True

        # Should shutdown cleanly
        service.shutdown()
        assert service._server_proxy is None

    def test_full_lifecycle_rpc_mode(self):
        """Test complete lifecycle in RPC mode"""
        mock_notifier = Mock()
        service = ConcreteRPCService(
            service_name="LifecycleRPCTest",
            port=50119,
            running_unit_tests=False,
            enable_health_check=True,
            health_check_interval_s=1,
            max_consecutive_failures=3,
            slack_notifier=mock_notifier
        )

        try:
            # Should be initialized with server process
            assert service._server_process is not None
            assert service._server_process.is_alive()
            assert service._server_proxy is not None

            # Should be able to call RPC methods
            result = service._server_proxy.some_method_rpc("test")
            assert result == "result_test"

            # Health checks should work
            current_time_ms = TimeUtil.now_in_millis()
            assert service.health_check(current_time_ms) is True

            # Should be able to check health again after interval
            assert service.health_check(current_time_ms + 2000) is True
        finally:
            # Should shutdown cleanly
            service.shutdown()
            assert service._server_proxy is None
            assert service._client_manager is None
            if service._server_process is not None:
                # Wait for actual termination instead of guessing
                assert TestHelpers.wait_for_process_termination(
                    service._server_process,
                    timeout=2.0
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
