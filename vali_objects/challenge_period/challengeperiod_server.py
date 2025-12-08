# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ChallengePeriodServer - RPC server for challenge period management.

This server runs in its own process and exposes challenge period management via RPC.
Clients connect using ChallengePeriodClient.

"""
import bittensor as bt
from typing import List, Optional, Tuple
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.rpc_server_base import RPCServerBase


# ==================== Server Implementation ====================
# Note: ChallengePeriodClient is in challengeperiod_client.py

class ChallengePeriodServer(RPCServerBase):
    """
    RPC server for challenge period management.

    Wraps ChallengePeriodManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to ChallengePeriodClient.

    This follows the same pattern as PerfLedgerServer and EliminationServer.
    """
    service_name = ValiConfig.RPC_CHALLENGEPERIOD_SERVICE_NAME
    service_port = ValiConfig.RPC_CHALLENGEPERIOD_PORT

    def __init__(
        self,
        *,
        is_backtesting=False,
        slack_notifier=None,
        start_server=True,
        start_daemon=False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize ChallengePeriodServer IN-PROCESS (never spawns).

        Args:
            is_backtesting: Whether running in backtesting mode
            slack_notifier: Slack notifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            running_unit_tests: Whether running in test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests

        # Always create in-process - constructor NEVER spawns
        bt.logging.info("[CP_SERVER] Creating ChallengePeriodServer in-process")

        # Create own CommonDataClient (forward compatibility - no parameter passing)
        self._common_data_client = CommonDataClient(
            connect_immediately=(connection_mode == RPCConnectionMode.RPC),
            connection_mode=connection_mode
        )

        # Create the actual ChallengePeriodManager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        self._manager = ChallengePeriodManager(
            is_backtesting=is_backtesting,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )

        bt.logging.info("[CP_SERVER] ChallengePeriodManager initialized")

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        # daemon_interval_s: 5 minutes (challenge period checks)
        # hang_timeout_s: Dynamically set to 2x interval to prevent false alarms during normal sleep
        daemon_interval_s = ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS / 1000.0  # 5 minutes (300s)
        hang_timeout_s = daemon_interval_s * 2.0  # 10 minutes (2x interval)

        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_CHALLENGEPERIOD_SERVICE_NAME,
            port=ValiConfig.RPC_CHALLENGEPERIOD_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s,
            connection_mode=connection_mode,
            daemon_stagger_s=10
        )

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Checks for sync in progress, then refreshes challenge period.
        """
        if self.sync_in_progress:
            bt.logging.debug("ChallengePeriodManager: Sync in progress, pausing...")
            return

        # Capture epoch at START of iteration
        iteration_epoch = self.sync_epoch

        # Run the challenge period refresh with captured epoch
        self._manager.refresh(current_time=None, iteration_epoch=iteration_epoch)

    @property
    def sync_in_progress(self):
        """Get sync_in_progress flag via CommonDataClient."""
        return self._common_data_client.get_sync_in_progress()

    @property
    def sync_epoch(self):
        """Get sync_epoch value via CommonDataClient."""
        return self._common_data_client.get_sync_epoch()

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "active_miners_count": len(self._manager.active_miners),
            "elimination_reasons_count": len(self._manager.eliminations_with_reasons)
        }

    # Note: Daemon control methods (start_daemon_rpc, stop_daemon_rpc, is_daemon_running_rpc, get_daemon_info_rpc)
    # are inherited from RPCServerBase

    # ==================== Query RPC Methods ====================

    def has_miner_rpc(self, hotkey: str) -> bool:
        """Fast check if a miner is in active_miners (O(1))."""
        return self._manager.has_miner(hotkey)

    def get_miner_bucket_rpc(self, hotkey: str) -> Optional[str]:
        """Get the bucket of a miner."""
        info = self._manager.active_miners.get(hotkey)
        if info and info[0]:
            return info[0].value
        return None

    def get_miner_start_time_rpc(self, hotkey: str) -> Optional[int]:
        """Get the start time of a miner's current bucket."""
        return self._manager.get_miner_start_time(hotkey)

    def get_miner_previous_bucket_rpc(self, hotkey: str) -> Optional[str]:
        """Get the previous bucket of a miner."""
        info = self._manager.active_miners.get(hotkey)
        if info and info[2]:
            return info[2].value
        return None

    def get_miner_previous_time_rpc(self, hotkey: str) -> Optional[int]:
        """Get the start time of a miner's previous bucket."""
        return self._manager.get_miner_previous_time(hotkey)

    def get_hotkeys_by_bucket_rpc(self, bucket_value: str) -> List[str]:
        """Get all hotkeys in a specific bucket."""
        from vali_objects.enums.miner_bucket_enum import MinerBucket
        bucket = MinerBucket(bucket_value)
        return self._manager.get_hotkeys_by_bucket(bucket)

    def get_all_miner_hotkeys_rpc(self) -> List[str]:
        """Get list of all active miner hotkeys."""
        return self._manager.get_all_miner_hotkeys()

    def get_testing_miners_rpc(self) -> dict:
        """Get all CHALLENGE bucket miners as dict {hotkey: start_time}."""
        return self._manager.get_testing_miners()

    def get_success_miners_rpc(self) -> dict:
        """Get all MAINCOMP bucket miners as dict {hotkey: start_time}."""
        return self._manager.get_success_miners()

    def get_probation_miners_rpc(self) -> dict:
        """Get all PROBATION bucket miners as dict {hotkey: start_time}."""
        return self._manager.get_probation_miners()

    def get_plagiarism_miners_rpc(self) -> dict:
        """Get all PLAGIARISM bucket miners as dict {hotkey: start_time}."""
        return self._manager.get_plagiarism_miners()

    def get_miner_scores_rpc(self) -> tuple:
        """Get cached miner scores for MinerStatisticsManager."""
        return self._manager.get_miner_scores()

    # ==================== Elimination Reasons RPC Methods ====================

    def get_all_elimination_reasons_rpc(self) -> dict:
        """Get all elimination reasons as a dict."""
        return self._manager.get_all_elimination_reasons()

    def has_elimination_reasons_rpc(self) -> bool:
        """Check if there are any elimination reasons."""
        return self._manager.has_elimination_reasons()

    def clear_elimination_reasons_rpc(self) -> None:
        """Clear all elimination reasons."""
        self._manager.clear_elimination_reasons()

    def pop_elimination_reason_rpc(self, hotkey: str) -> Optional[Tuple[str, float]]:
        """Atomically get and remove an elimination reason for a single hotkey."""
        return self._manager.pop_elimination_reason(hotkey)

    def update_elimination_reasons_rpc(self, reasons_dict: dict) -> int:
        """Accumulate elimination reasons from a dict."""
        return self._manager.update_elimination_reasons(reasons_dict)

    # ==================== Mutation RPC Methods ====================

    def set_miner_bucket_rpc(
        self,
        hotkey: str,
        bucket_value: str,
        start_time: int,
        prev_bucket_value: Optional[str] = None,
        prev_time: Optional[int] = None
    ) -> bool:
        """Set or update a miner's bucket information."""
        bucket = MinerBucket(bucket_value)
        prev_bucket = MinerBucket(prev_bucket_value) if prev_bucket_value else None
        return self._manager.set_miner_bucket(hotkey, bucket, start_time, prev_bucket, prev_time)

    def remove_miner_rpc(self, hotkey: str) -> bool:
        """Remove a miner from active_miners."""
        return self._manager.remove_miner(hotkey)

    def clear_all_miners_rpc(self) -> None:
        """Clear all miners from active_miners."""
        self._manager.clear_active_miners()

    def update_miners_rpc(self, miners_dict: dict) -> int:
        """
        Bulk update active_miners from a dict.

        Args:
            miners_dict: Dict mapping hotkey to dict with keys:
                - bucket: str (bucket value like "MAINCOMP")
                - start_time: int
                - prev_bucket: str or None
                - prev_time: int or None

        Returns:
            Number of miners updated
        """
        # Manager's update_miners now handles both tuple and dict formats
        return self._manager.update_active_miners(miners_dict)

    # ==================== Management RPC Methods ====================

    def refresh_rpc(self, current_time: int = None, iteration_epoch=None) -> None:
        """Trigger a challenge period refresh via RPC."""
        self._manager.refresh(current_time=current_time, iteration_epoch=iteration_epoch)

    def clear_challengeperiod_in_memory_and_disk_rpc(self) -> None:
        """Clear all challenge period data (memory and disk)."""
        self._manager._clear_challengeperiod_in_memory_and_disk()

    def clear_test_state_rpc(self) -> None:
        """
        Clear ALL test-sensitive state (for test isolation).

        This includes:
        - Challenge period data (active_miners, elimination_reasons)
        - refreshed_challengeperiod_start_time flag (prevents test contamination)
        - Any other stateful flags that affect test behavior

        Should be called by ServerOrchestrator.clear_all_test_data() to ensure
        complete test isolation when servers are shared across tests.
        """
        self._manager._clear_challengeperiod_in_memory_and_disk()
        self._manager.refreshed_challengeperiod_start_time = False  # Reset flag to allow refresh in each test
        # Future: Add any other stateful flags here

    def write_challengeperiod_from_memory_to_disk_rpc(self) -> None:
        """Write challenge period data from memory to disk."""
        self._manager._write_challengeperiod_from_memory_to_disk()

    def sync_challenge_period_data_rpc(self, active_miners_sync: dict) -> None:
        """Sync challenge period data from another validator."""
        self._manager.sync_challenge_period_data(active_miners_sync)

    def meets_time_criteria_rpc(self, current_time: int, bucket_start_time: int, bucket_value: str) -> bool:
        """Check if a miner meets time criteria for their bucket."""
        from vali_objects.enums.miner_bucket_enum import MinerBucket
        bucket = MinerBucket(bucket_value)
        return self._manager.meets_time_criteria(current_time, bucket_start_time, bucket)

    def remove_eliminated_rpc(self, eliminations: list = None) -> None:
        """Remove eliminated miners from active_miners."""
        self._manager.remove_eliminated(eliminations=eliminations)

    def update_plagiarism_miners_rpc(self, current_time: int, plagiarism_miners: dict) -> None:
        """Update plagiarism miners via RPC."""
        self._manager.update_plagiarism_miners(current_time, plagiarism_miners)

    def prepare_plagiarism_elimination_miners_rpc(self, current_time: int) -> dict:
        """Prepare plagiarism miners for elimination."""
        return self._manager.prepare_plagiarism_elimination_miners(current_time)

    def demote_plagiarism_in_memory_rpc(self, hotkeys: list, current_time: int) -> None:
        """Demote miners to plagiarism bucket (exposed for testing)."""
        self._manager._demote_plagiarism_in_memory(hotkeys, current_time)

    def promote_plagiarism_to_previous_bucket_in_memory_rpc(self, hotkeys: list, current_time: int) -> None:
        """Promote plagiarism miners to their previous bucket (exposed for testing)."""
        self._manager._promote_plagiarism_to_previous_bucket_in_memory(hotkeys, current_time)

    def eliminate_challengeperiod_in_memory_rpc(self, eliminations_with_reasons: dict) -> None:
        """Eliminate miners from challenge period (exposed for testing)."""
        self._manager._eliminate_challengeperiod_in_memory(eliminations_with_reasons)

    def add_challengeperiod_testing_in_memory_and_disk_rpc(
        self,
        new_hotkeys: list,
        eliminations: list,
        hk_to_first_order_time: dict,
        default_time: int
    ) -> None:
        """Add miners to challenge period (exposed for testing)."""
        self._manager._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys, eliminations, hk_to_first_order_time, default_time
        )

    def promote_challengeperiod_in_memory_rpc(self, hotkeys: list, current_time: int) -> None:
        """Promote miners to main competition (exposed for testing)."""
        self._manager._promote_challengeperiod_in_memory(hotkeys, current_time)

    def inspect_rpc(
        self,
        positions: dict,
        ledger: dict,
        success_hotkeys: list,
        probation_hotkeys: list,
        inspection_hotkeys: dict,
        current_time: int,
        hk_to_first_order_time: dict = None,
        asset_softmaxed_scores: dict = None
    ) -> tuple:
        """Run challenge period inspection (exposed for testing)."""
        return self._manager.inspect(
            positions,
            ledger,
            success_hotkeys,
            probation_hotkeys,
            inspection_hotkeys,
            current_time,
            hk_to_first_order_time,
            asset_softmaxed_scores
        )

    def to_checkpoint_dict_rpc(self) -> dict:
        """Get challenge period data as a checkpoint dict for serialization."""
        return self._manager.to_checkpoint_dict()

    def set_last_update_time_rpc(self, timestamp_ms: int = 0) -> None:
        """Set the last update time (for testing - to force-allow refresh)."""
        self._manager._last_update_time_ms = timestamp_ms
