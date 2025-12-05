from shared_objects.locks.position_lock_server import PositionLockProxy
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig


class PositionLockClient(RPCClientBase):
    """
    Lightweight RPC client for PositionLockServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_POSITIONLOCK_PORT.

    In test mode (running_unit_tests=True), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct PositionLockServer instance.

    Usage:
        # Production mode - connects to existing server
        client = PositionLockClient()
        with client.get_lock(hotkey, pair_id):
            # Critical section
            pass

        # Test mode - direct server access
        client = PositionLockClient(running_unit_tests=True)
        client.set_direct_server(server_instance)
    """

    def __init__(self, port: int = None, running_unit_tests: bool = False):
        """
        Initialize position lock client.

        Args:
            port: Port number of the position lock server (default: ValiConfig.RPC_POSITIONLOCK_PORT)
            running_unit_tests: If True, don't connect via RPC (use set_direct_server() instead)
        """
        super().__init__(
            service_name=ValiConfig.RPC_POSITIONLOCK_SERVICE_NAME,
            port=port or ValiConfig.RPC_POSITIONLOCK_PORT,
            connect_immediately=False
        )

    # ==================== Lock Methods ====================

    def get_lock(self, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0) -> PositionLockProxy:
        """
        Get a lock proxy for the given key.

        Returns a context manager that acquires/releases the lock via RPC.

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            timeout: Lock acquisition timeout in seconds

        Returns:
            PositionLockProxy: Context manager for the lock

        Usage:
            with client.get_lock(hotkey, pair_id):
                # Critical section
                pass
        """
        return PositionLockProxy(self._server, miner_hotkey, trade_pair_id, timeout)

    def acquire(self, miner_hotkey: str, trade_pair_id: str, timeout: float = 10.0) -> bool:
        """
        Acquire lock directly (without context manager).

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID
            timeout: Lock acquisition timeout in seconds

        Returns:
            bool: True if lock was acquired, False if timeout
        """
        return self._server.acquire_rpc(miner_hotkey, trade_pair_id, timeout)

    def release(self, miner_hotkey: str, trade_pair_id: str) -> bool:
        """
        Release lock directly (without context manager).

        Args:
            miner_hotkey: Miner's hotkey
            trade_pair_id: Trade pair ID

        Returns:
            bool: True if released successfully, False if error
        """
        return self._server.release_rpc(miner_hotkey, trade_pair_id)

    # ==================== Health Check ====================

    def health_check(self) -> dict:
        """Get health status from server."""
        return self._server.health_check_rpc()

    def get_lock_count(self) -> int:
        """Get the number of locks currently tracked."""
        return self._server.get_lock_count_rpc()
