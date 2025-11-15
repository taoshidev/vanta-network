"""
Port Management Utility - Explicit port availability checking without sleep guessing.

This module provides utilities to:
- Check if a port is actually free (not just hoping after sleep)
- Wait for port release with exponential backoff polling
- Wait for services to start listening on a port

Eliminates the anti-pattern:
    process.terminate()
    time.sleep(1.5)  # Hope port is released

Replaced with:
    process.terminate()
    PortManager.wait_for_port_release(port)  # Know when it's released
"""
import socket
import time


class PortManager:
    """Manages port availability with explicit checking instead of sleep guessing"""

    @staticmethod
    def is_port_free(port: int, host: str = 'localhost') -> bool:
        """
        Check if a port is actually available for binding.

        Args:
            port: Port number to check
            host: Hostname to check (default: localhost)

        Returns:
            bool: True if port is free, False if in use

        Example:
            if PortManager.is_port_free(50000):
                # Safe to start server on port 50000
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except OSError:
            return False

    @staticmethod
    def is_port_listening(port: int, host: str = 'localhost', timeout: float = 0.1) -> bool:
        """
        Check if something is actively listening on a port.

        Args:
            port: Port number to check
            host: Hostname to check (default: localhost)
            timeout: Connection timeout in seconds

        Returns:
            bool: True if something is listening, False otherwise

        Example:
            if PortManager.is_port_listening(50000):
                # Server is up and accepting connections
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    @staticmethod
    def wait_for_port_release(
        port: int,
        host: str = 'localhost',
        timeout: float = 5.0,
        initial_delay: float = 0.01
    ) -> bool:
        """
        Wait for a port to be released with exponential backoff polling.

        Instead of blindly sleeping for a fixed duration, this polls the actual
        port state with increasing delays. Usually completes in <50ms.

        Args:
            port: Port number to wait for
            host: Hostname to check (default: localhost)
            timeout: Maximum time to wait in seconds
            initial_delay: Initial polling interval (doubles each iteration)

        Returns:
            bool: True if port was released, False if timeout

        Example:
            process.terminate()
            if PortManager.wait_for_port_release(50000, timeout=3.0):
                # Port released in <50ms typically
            else:
                # Port still in use after 3 seconds
        """
        deadline = time.time() + timeout
        backoff = initial_delay

        while time.time() < deadline:
            if PortManager.is_port_free(port, host):
                return True

            # Exponential backoff: 10ms, 20ms, 40ms, 80ms, 160ms, ...
            # Prevents busy-waiting while staying responsive
            remaining = deadline - time.time()
            time.sleep(min(backoff, remaining))
            backoff *= 2

        return False

    @staticmethod
    def wait_for_port_listen(
        port: int,
        host: str = 'localhost',
        timeout: float = 10.0,
        initial_delay: float = 0.01
    ) -> bool:
        """
        Wait for a service to start listening on a port.

        Polls until something is accepting connections on the port.
        Useful for waiting for servers to be ready.

        Args:
            port: Port number to wait for
            host: Hostname to check (default: localhost)
            timeout: Maximum time to wait in seconds
            initial_delay: Initial polling interval (doubles each iteration)

        Returns:
            bool: True if service is listening, False if timeout

        Example:
            process.start()
            if PortManager.wait_for_port_listen(50000, timeout=10.0):
                # Server is ready and accepting connections
            else:
                # Server failed to start within 10 seconds
        """
        deadline = time.time() + timeout
        backoff = initial_delay

        while time.time() < deadline:
            if PortManager.is_port_listening(port, host):
                return True

            # Exponential backoff
            remaining = deadline - time.time()
            time.sleep(min(backoff, remaining))
            backoff *= 2

        return False
