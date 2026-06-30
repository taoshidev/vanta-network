# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ChallengePeriodClient - Lightweight RPC client for challenge period management.

This client connects to the ChallengePeriodServer via RPC.
Can be created in ANY process - just needs the server to be running.

Usage:
    from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient

    # Connect to server (uses ValiConfig.RPC_CHALLENGEPERIOD_PORT by default)
    client = ChallengePeriodClient()

    if client.has_miner(hotkey):
        bucket = client.get_miner_bucket(hotkey)

    # In child processes - same pattern, port from ValiConfig
    def child_func():
        client = ChallengePeriodClient()
        client.get_testing_miners()
"""
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class ChallengePeriodClient(RPCClientBase):
    """
    Lightweight RPC client for ChallengePeriodServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_CHALLENGEPERIOD_PORT.

    In LOCAL mode (connection_mode=RPCConnectionMode.LOCAL), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct ChallengePeriodServer instance.
    """

    def __init__(
        self,
        port: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False
    ):
        """
        Initialize challenge period client.

        Args:
            port: Port number of the challenge period server (default: ValiConfig.RPC_CHALLENGEPERIOD_PORT)
            connection_mode: RPCConnectionMode.LOCAL for tests (use set_direct_server()), RPCConnectionMode.RPC for production
        """
        self._direct_server = None
        self.running_unit_tests = running_unit_tests

        # In LOCAL mode, don't connect via RPC - tests will set direct server
        super().__init__(
            service_name=ValiConfig.RPC_CHALLENGEPERIOD_SERVICE_NAME,
            port=port or ValiConfig.RPC_CHALLENGEPERIOD_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=False,
            connection_mode=connection_mode
        )

    # ==================== Active Miners Methods ====================

    def has_miner(self, hotkey: str) -> bool:
        """Fast check if a miner is in active_miners (O(1))."""
        return self._server.has_miner_rpc(hotkey)

    def get_miner_bucket(self, hotkey: str, timestamp_ms: int | None = None) -> MinerBucket | None:
        """Get the bucket of a miner, optionally at a specific timestamp."""
        return self._server.get_miner_bucket_rpc(hotkey, timestamp_ms)

    def get_miner_start_time(self, hotkey: str) -> int | None:
        """Get the start time of a miner's current bucket."""
        return self._server.get_miner_start_time_rpc(hotkey)

    def get_hotkeys_by_bucket(self, buckets: MinerBucket | list[MinerBucket]) -> list[str]:
        """Get all hotkeys in a specific bucket."""
        return self._server.get_hotkeys_by_bucket_rpc(buckets)

    def get_all_miner_hotkeys(self) -> list[str]:
        """Get list of all active miner hotkeys."""
        return self._server.get_all_miner_hotkeys_rpc()

    def get_dashboard(self, hotkey) -> dict | None:
        return self._server.get_dashboard_rpc(hotkey)

    def pop_bucket_entry(self, hotkey: str, top_bucket: MinerBucket) -> bool:
        """Pop the top bucket entry if it matches top_bucket. Returns True if popped."""
        return self._server.pop_bucket_entry_rpc(hotkey, top_bucket)

    def reset_drawdown_cache(self, hotkey: str) -> None:
        """Reset drawdown stats to neutral defaults so stale values can't trigger elimination."""
        return self._server.reset_drawdown_cache_rpc(hotkey)

    def save_to_disk(self) -> None:
        """Persist in-memory challenge period state to disk."""
        return self._server.save_to_disk_rpc()

    def set_miner_bucket(self, hotkey: str, bucket: MinerBucket, start_time_ms: int) -> bool:
        """Set or update a miner's bucket information."""
        return self._server.set_miner_bucket_rpc(hotkey, bucket, start_time_ms)

    def remove_miners(self, hotkeys: str) -> bool:
        """Remove a miner from active_miners."""
        return self._server.remove_miners_rpc(hotkeys)

    def get_testing_miners(self) -> dict[str, int]:
        """Get all CHALLENGE bucket miners as dict {hotkey: start_time}."""
        return self._server.get_miners_rpc([MinerBucket.CHALLENGE, MinerBucket.SUBACCOUNT_CHALLENGE])

    def get_success_miners(self) -> dict[str, int]:
        """Get all MAINCOMP bucket miners as dict {hotkey: start_time}."""
        return self._server.get_miners_rpc([MinerBucket.MAINCOMP, MinerBucket.SUBACCOUNT_FUNDED])

    def get_probation_miners(self) -> dict[str, int]:
        """Get all PROBATION bucket miners as dict {hotkey: start_time}."""
        return self._server.get_miners_rpc(MinerBucket.PROBATION)

    def get_plagiarism_miners(self) -> dict[str, int]:
        """Get all PLAGIARISM bucket miners as dict {hotkey: start_time}."""
        return self._server.get_miners_rpc(MinerBucket.PLAGIARISM)

    # ==================== Daemon Methods ====================

    def get_daemon_info(self) -> dict:
        """
        Get daemon information for testing/debugging.

        Returns:
            dict: {
                "daemon_started": bool,
                "daemon_alive": bool,
                "daemon_ident": int (thread ID),
                "server_pid": int (process ID),
                "daemon_is_thread": bool
            }
        """
        return self._server.get_daemon_info_rpc()

    # ==================== Management Methods ====================

    def sync_challenge_period_data(self, active_miners_sync):
        """Sync challenge period data from another validator."""
        self._server.sync_challenge_period_data_rpc(active_miners_sync)

    def clear_test_state(self) -> None:
        """Clear all miner states for test isolation."""
        self._server.clear_test_state_rpc()

    def to_checkpoint_dict(self) -> dict:
        """Get challenge period data as a checkpoint dict for serialization."""
        return self._server.to_checkpoint_dict_rpc()

    def get_drawdown_stats(self, synthetic_hotkey: str) -> dict | None:
        """Get drawdown statistics for a synthetic hotkey for dashboard display."""
        return self._server.get_drawdown_stats_rpc(synthetic_hotkey)

    def get_miner_scores(self) -> tuple:
        """ Get cached miner scores for MinerStatisticsManager. """
        return self._server.get_miner_scores_rpc()

