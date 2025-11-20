# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Generic RPC client for manager-to-manager communication.

This client allows one manager to make RPC calls to another manager
without importing the manager class itself.

Key principle: Only import shared protocol, never actual manager classes.
"""
from multiprocessing.managers import BaseManager
import bittensor as bt
from vali_objects.vali_config import ValiConfig
import time


class ManagerRPCClient:
    """
    Generic RPC client for calling methods on a remote manager.

    This client connects to an existing RPC server (EliminationManager or
    ChallengePeriodManager) and allows making RPC calls without importing
    the actual manager class.

    Example:
        # Connect to ChallengePeriodManager
        cp_client = ManagerRPCClient("localhost", 50003, "ChallengePeriodManagerServer")

        # Make RPC calls
        has_reasons = cp_client.call("has_elimination_reasons_rpc")
        reasons = cp_client.call("get_all_elimination_reasons_rpc")
        result = cp_client.call("pop_elimination_reason_rpc", hotkey="5abc...")
    """

    def __init__(self, host: str, port: int, service_name: str):
        """
        Initialize RPC client to connect to a remote manager.

        Args:
            host: Host address (e.g., "localhost")
            port: Port number (e.g., 50003 for CP, 50004 for Elim)
            service_name: Name of the service (e.g., "ChallengePeriodManagerServer")
        """
        self.host = host
        self.port = port
        self.service_name = service_name
        self.address = (host, port)

        # Generate authkey (must match server's authkey generation)
        self._authkey = ValiConfig.get_rpc_authkey(service_name, port)

        # Connection state
        self._manager = None
        self._proxy = None
        self._connected = False

        bt.logging.debug(
            f"[ManagerRPCClient] Created client for {service_name} at {host}:{port}"
        )

    def connect(self, max_retries: int = 60, retry_delay: float = 1.0) -> bool:
        """
        Connect to the remote RPC server.

        Args:
            max_retries: Maximum number of connection attempts (default: 60)
            retry_delay: Delay between retries in seconds (default: 1.0)

        Returns:
            bool: True if connected successfully, False otherwise
        """
        # Threshold for when to start logging warnings (avoid spam during startup)
        # Default: warn after 30 seconds (30 attempts with 1s delay)
        warning_threshold = 30

        for attempt in range(1, max_retries + 1):
            try:
                bt.logging.trace(
                    f"[ManagerRPCClient] Attempting to connect to {self.service_name} "
                    f"at {self.address} (attempt {attempt}/{max_retries})"
                )

                # Create manager and connect
                class TempManager(BaseManager):
                    pass

                # Register the proxy class (we don't need to know actual methods)
                TempManager.register(self.service_name)

                manager = TempManager(address=self.address, authkey=self._authkey)
                manager.connect()

                # Get proxy to remote service
                self._proxy = getattr(manager, self.service_name)()
                self._manager = manager
                self._connected = True

                # If it took multiple attempts, log that we eventually succeeded
                if attempt > 1:
                    bt.logging.success(
                        f"[ManagerRPCClient] Connected to {self.service_name} at {self.address} "
                        f"after {attempt} attempts"
                    )
                else:
                    bt.logging.success(
                        f"[ManagerRPCClient] Connected to {self.service_name} at {self.address}"
                    )
                return True

            except Exception as e:
                if attempt < max_retries:
                    # Only log warnings after several failed attempts to reduce startup noise
                    if attempt >= warning_threshold:
                        bt.logging.warning(
                            f"[ManagerRPCClient] Connection failed (attempt {attempt}/{max_retries}): {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                    else:
                        bt.logging.trace(
                            f"[ManagerRPCClient] Connection failed (attempt {attempt}/{max_retries}): {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                    time.sleep(retry_delay)
                else:
                    bt.logging.error(
                        f"[ManagerRPCClient] Failed to connect after {max_retries} attempts: {e}"
                    )
                    return False

        return False

    def call(self, method_name: str, *args, **kwargs):
        """
        Call a method on the remote RPC server.

        Auto-reconnects if not connected or connection was lost.

        Args:
            method_name: Name of the RPC method to call (e.g., "has_elimination_reasons_rpc")
            *args: Positional arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            The result from the RPC call

        Raises:
            RuntimeError: If not connected and reconnection fails
            AttributeError: If method doesn't exist on remote service
        """
        # Auto-reconnect if not connected
        if not self._connected or self._proxy is None:
            bt.logging.debug(
                f"[ManagerRPCClient] Not connected to {self.service_name}, attempting to reconnect..."
            )
            if not self.connect(max_retries=60, retry_delay=1.0):
                raise RuntimeError(
                    f"Not connected to {self.service_name}. Reconnection failed."
                )

        try:
            method = getattr(self._proxy, method_name)
            result = method(*args, **kwargs)

            bt.logging.trace(
                f"[ManagerRPCClient] Called {self.service_name}.{method_name}(*{args}, **{kwargs}) → {result}"
            )

            return result

        except AttributeError as e:
            bt.logging.error(
                f"[ManagerRPCClient] Method '{method_name}' not found on {self.service_name}: {e}"
            )
            raise
        except Exception as e:
            bt.logging.error(
                f"[ManagerRPCClient] RPC call failed: {self.service_name}.{method_name}: {e}"
            )
            raise

    def is_connected(self) -> bool:
        """Check if client is connected to remote service"""
        return self._connected and self._proxy is not None

    def disconnect(self):
        """Disconnect from remote service"""
        if self._manager:
            try:
                self._manager.shutdown()
            except:
                pass

        self._manager = None
        self._proxy = None
        self._connected = False

        bt.logging.debug(f"[ManagerRPCClient] Disconnected from {self.service_name}")
