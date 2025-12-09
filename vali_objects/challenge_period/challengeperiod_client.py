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
from typing import Optional, List

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

    # ==================== Elimination Reasons Methods ====================

    def get_all_elimination_reasons(self) -> dict:
        """Get all elimination reasons as a dict."""
        return self._server.get_all_elimination_reasons_rpc()

    def has_elimination_reasons(self) -> bool:
        """Check if there are any elimination reasons."""
        return self._server.has_elimination_reasons_rpc()

    def clear_elimination_reasons(self) -> None:
        """Clear all elimination reasons."""
        self._server.clear_elimination_reasons_rpc()

    def update_elimination_reasons(self, reasons_dict: dict) -> int:
        """Bulk update elimination reasons from a dict."""
        return self._server.update_elimination_reasons_rpc(reasons_dict)

    def pop_elimination_reason(self, hotkey: str):
        """Atomically get and remove an elimination reason for a single hotkey."""
        return self._server.pop_elimination_reason_rpc(hotkey)

    # ==================== Active Miners Methods ====================

    def has_miner(self, hotkey: str) -> bool:
        """Fast check if a miner is in active_miners (O(1))."""
        return self._server.has_miner_rpc(hotkey)

    def get_miner_bucket(self, hotkey: str) -> Optional[MinerBucket]:
        """Get the bucket of a miner."""
        bucket_value = self._server.get_miner_bucket_rpc(hotkey)
        return MinerBucket(bucket_value) if bucket_value else None

    def get_miner_start_time(self, hotkey: str) -> Optional[int]:
        """Get the start time of a miner's current bucket."""
        return self._server.get_miner_start_time_rpc(hotkey)

    def get_miner_previous_bucket(self, hotkey: str) -> Optional[MinerBucket]:
        """Get the previous bucket of a miner (used for plagiarism demotions)."""
        prev_bucket_value = self._server.get_miner_previous_bucket_rpc(hotkey)
        return MinerBucket(prev_bucket_value) if prev_bucket_value else None

    def get_miner_previous_time(self, hotkey: str) -> Optional[int]:
        """Get the start time of a miner's previous bucket."""
        return self._server.get_miner_previous_time_rpc(hotkey)

    def get_hotkeys_by_bucket(self, bucket: MinerBucket) -> List[str]:
        """Get all hotkeys in a specific bucket."""
        return self._server.get_hotkeys_by_bucket_rpc(bucket.value)

    def get_all_miner_hotkeys(self) -> List[str]:
        """Get list of all active miner hotkeys."""
        return self._server.get_all_miner_hotkeys_rpc()

    def set_miner_bucket(
        self,
        hotkey: str,
        bucket: MinerBucket,
        start_time: int,
        prev_bucket: Optional[MinerBucket] = None,
        prev_time: Optional[int] = None
    ) -> bool:
        """Set or update a miner's bucket information."""
        return self._server.set_miner_bucket_rpc(
            hotkey,
            bucket.value,
            start_time,
            prev_bucket.value if prev_bucket else None,
            prev_time
        )

    def remove_miner(self, hotkey: str) -> bool:
        """Remove a miner from active_miners."""
        return self._server.remove_miner_rpc(hotkey)

    def write_challengeperiod_from_memory_to_disk(self):
        return self._server.write_challengeperiod_from_memory_to_disk_rpc()

    def clear_all_miners(self) -> None:
        """Clear all miners from active_miners."""
        self._server.clear_all_miners_rpc()

    def update_miners(self, miners_dict: dict) -> int:
        """Bulk update active_miners from a dict."""
        # Convert tuples to dicts for RPC serialization
        miners_rpc_dict = {}
        for hotkey, (bucket, start_time, prev_bucket, prev_time) in miners_dict.items():
            miners_rpc_dict[hotkey] = {
                "bucket": bucket.value,
                "start_time": start_time,
                "prev_bucket": prev_bucket.value if prev_bucket else None,
                "prev_time": prev_time
            }

        return self._server.update_miners_rpc(miners_rpc_dict)

    def iter_active_miners(self):
        """
        Iterate over active miners.
        Note: This fetches ALL miners and iterates locally.
        """

        for hotkey, start_time in self.get_testing_miners().items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.CHALLENGE, start_time, prev_bucket, prev_time

        for hotkey, start_time in self.get_success_miners().items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.MAINCOMP, start_time, prev_bucket, prev_time

        for hotkey, start_time in self.get_probation_miners().items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.PROBATION, start_time, prev_bucket, prev_time

        for hotkey, start_time in self.get_plagiarism_miners().items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.PLAGIARISM, start_time, prev_bucket, prev_time

    def get_testing_miners(self) -> dict:
        """Get all CHALLENGE bucket miners as dict {hotkey: start_time}."""
        return self._server.get_testing_miners_rpc()

    def get_success_miners(self) -> dict:
        """Get all MAINCOMP bucket miners as dict {hotkey: start_time}."""
        return self._server.get_success_miners_rpc()

    def get_probation_miners(self) -> dict:
        """Get all PROBATION bucket miners as dict {hotkey: start_time}."""
        return self._server.get_probation_miners_rpc()

    def get_plagiarism_miners(self) -> dict:
        """Get all PLAGIARISM bucket miners as dict {hotkey: start_time}."""
        return self._server.get_plagiarism_miners_rpc()

    def get_miner_scores(self) -> tuple:
        """
        Get cached miner scores for MinerStatisticsManager.

        Returns:
            tuple containing:
            - asset_softmaxed_scores: dict[asset_class, dict[hotkey, score]]
            - asset_competitiveness: dict[asset_class, competitiveness_score]
        """
        return self._server.get_miner_scores_rpc()

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

    def _clear_challengeperiod_in_memory_and_disk(self):
        """Clear all challenge period data (memory and disk)."""
        self._server.clear_challengeperiod_in_memory_and_disk_rpc()

    def clear_test_state(self) -> None:
        """
        Clear ALL test-sensitive state (comprehensive reset for test isolation).

        This resets:
        - Challenge period data (active_miners, elimination_reasons)
        - refreshed_challengeperiod_start_time flag (prevents test contamination)
        - Any other stateful flags

        Should be called by ServerOrchestrator.clear_all_test_data() to ensure
        complete test isolation when servers are shared across tests.

        Use this instead of _clear_challengeperiod_in_memory_and_disk() alone to prevent test contamination.
        """
        self._server.clear_test_state_rpc()

    def _write_challengeperiod_from_memory_to_disk(self):
        """Write challenge period data from memory to disk."""
        self._server.write_challengeperiod_from_memory_to_disk_rpc()

    def sync_challenge_period_data(self, active_miners_sync):
        """Sync challenge period data from another validator."""
        self._server.sync_challenge_period_data_rpc(active_miners_sync)

    def refresh(self, current_time: int = None, iteration_epoch=None):
        """Refresh the challenge period manager."""
        self._server.refresh_rpc(current_time=current_time, iteration_epoch=iteration_epoch)

    def meets_time_criteria(self, current_time, bucket_start_time, bucket):
        """Check if a miner meets time criteria for their bucket."""
        return self._server.meets_time_criteria_rpc(current_time, bucket_start_time, bucket.value)

    def remove_eliminated(self, eliminations=None):
        """Remove eliminated miners from active_miners."""
        self._server.remove_eliminated_rpc(eliminations=eliminations)

    def update_plagiarism_miners(self, current_time, plagiarism_miners):
        """Update plagiarism miners."""
        self._server.update_plagiarism_miners_rpc(current_time, plagiarism_miners)

    def prepare_plagiarism_elimination_miners(self, current_time):
        """Prepare plagiarism miners for elimination."""
        return self._server.prepare_plagiarism_elimination_miners_rpc(current_time)

    def _demote_plagiarism_in_memory(self, hotkeys, current_time):
        """Demote miners to plagiarism bucket (exposed for testing)."""
        self._server.demote_plagiarism_in_memory_rpc(hotkeys, current_time)

    def promote_plagiarism_to_previous_bucket_in_memory(self, hotkeys, current_time):
        """Promote plagiarism miners to their previous bucket (exposed for testing)."""
        self._server.promote_plagiarism_to_previous_bucket_in_memory_rpc(hotkeys, current_time)

    def eliminate_challenge_period_in_memory(self, eliminations_with_reasons):
        """Eliminate miners from challenge period (exposed for testing)."""
        self._server.eliminate_challengeperiod_in_memory_rpc(eliminations_with_reasons)

    def add_challenge_period_testing_in_memory_and_disk(
        self,
        new_hotkeys,
        eliminations,
        hk_to_first_order_time,
        default_time
    ):
        """Add miners to challenge period (exposed for testing)."""
        self._server.add_challengeperiod_testing_in_memory_and_disk_rpc(
            new_hotkeys=new_hotkeys,
            eliminations=eliminations,
            hk_to_first_order_time=hk_to_first_order_time,
            default_time=default_time
        )

    def promote_challengeperiod_in_memory(self, hotkeys, current_time):
        """Promote miners to main competition (exposed for testing)."""
        self._server.promote_challengeperiod_in_memory_rpc(hotkeys, current_time)

    def inspect(
        self,
        positions,
        ledger,
        success_hotkeys,
        probation_hotkeys,
        inspection_hotkeys,
        current_time,
        hk_to_first_order_time=None,
        asset_softmaxed_scores=None
    ):
        """Run challenge period inspection (exposed for testing)."""
        return self._server.inspect_rpc(
            positions=positions,
            ledger=ledger,
            success_hotkeys=success_hotkeys,
            probation_hotkeys=probation_hotkeys,
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
            asset_softmaxed_scores=asset_softmaxed_scores
        )

    def to_checkpoint_dict(self) -> dict:
        """Get challenge period data as a checkpoint dict for serialization."""
        return self._server.to_checkpoint_dict_rpc()

    def set_last_update_time(self, timestamp_ms: int = 0) -> None:
        """Set the last update time (for testing - to force-allow refresh)."""
        self._server.set_last_update_time_rpc(timestamp_ms)
