# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ChallengePeriodServer - RPC server for challenge period management.

This server runs in its own process and exposes challenge period management via RPC.
Clients connect using ChallengePeriodClient.

"""
import time
import bittensor as bt
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.rpc_server_base import RPCServerBase


class ChallengePeriodServer(RPCServerBase):
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
            connect_immediately=False,  # Lazy connect on first use
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
        daemon_interval_s = ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS / 1000.0  # 1 minutes (60s)
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

    def run_daemon_iteration(self) -> str | None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Checks for sync in progress, then refreshes challenge period.
        """
        if self.sync_in_progress:
            bt.logging.warning("ChallengePeriodManager: Sync in progress, pausing...")
            time.sleep(1)
            return

        # Capture epoch at START of iteration
        iteration_epoch = self.sync_epoch

        # Run the challenge period refresh with captured epoch
        return self._manager.refresh(iteration_epoch=iteration_epoch)

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
        return {"active_miners_count": len(self._manager.miner_states)}

    def set_miner_bucket_rpc(self, hotkey: str, bucket: MinerBucket, start_time_ms: int) -> bool:
        return self._manager.set_miner_bucket(hotkey, bucket, start_time_ms)

    def remove_miners_rpc(self, hotkeys: str | list[str]) -> bool:
        return self._manager.remove_miners(hotkeys)

    def has_miner_rpc(self, hotkey: str) -> bool:
        return self._manager.has_miner(hotkey)

    def get_miner_bucket_rpc(self, hotkey: str, timestamp_ms: int | None = None) -> MinerBucket | None:
        return self._manager.get_miner_bucket(hotkey, timestamp_ms)

    def get_miner_start_time_rpc(self, hotkey: str) -> int | None:
        return self._manager.get_miner_start_time(hotkey)

    def get_hotkeys_by_bucket_rpc(self, buckets: MinerBucket | list[MinerBucket]) -> list[str]:
        return self._manager.get_hotkeys_by_bucket(buckets)

    def get_all_miner_hotkeys_rpc(self) -> list:
        return self._manager.get_all_miner_hotkeys()

    def get_miners_rpc(self, buckets: MinerBucket | list[MinerBucket]) -> dict[str, int]:
        return self._manager.get_miners(buckets)

    def get_miner_scores_rpc(self) -> tuple:
        return self._manager.get_miner_scores()

    def get_dashboard_rpc(self, hotkey: str) -> dict | None:
        return self._manager.get_dashboard(hotkey)

    def get_drawdown_stats_rpc(self, hotkey: str) -> dict | None:
        return self._manager.get_drawdown_stats(hotkey)

    def to_checkpoint_dict_rpc(self) -> dict:
        return self._manager.to_checkpoint_dict()

    def sync_challenge_period_data_rpc(self, miner_states_data: dict) -> None:
        return self._manager.sync_challenge_period_data(miner_states_data)

    def clear_test_state_rpc(self) -> None:
        """Clear all miner states for test isolation."""
        assert self.running_unit_tests, "clear_test_state_rpc should only be called in unit tests"
        with self._manager._buckets_lock:
            self._manager.miner_states.clear()
