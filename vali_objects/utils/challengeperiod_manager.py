# developer: jbonilla
# Copyright © 2024 Taoshi Inc
from typing import Optional, List
from multiprocessing import Process
import multiprocessing
import bittensor as bt
from shared_objects.rpc_service_base import RPCServiceBase
from shared_objects.cache_controller import CacheController
from vali_objects.utils.miner_bucket_enum import MinerBucket

class ChallengePeriodManager(RPCServiceBase, CacheController):
    """
    RPC Client for ChallengePeriodManager - manages challenge period state via RPC.

    This client connects to ChallengePeriodManagerServer running in a separate process.
    Much faster than IPC managerized dicts (50-200x improvement on batch operations).
    """

    def __init__(
            self,
            metagraph,
            perf_ledger_manager=None,
            position_manager=None,
            contract_manager=None,
            plagiarism_manager=None,
            asset_selection_manager=None,
            *,
            running_unit_tests=False,
            is_backtesting=False,
            shutdown_dict=None,
            sync_in_progress=None,
            slack_notifier=None,
            sync_epoch=None,
            start_daemon=False):

        # Initialize RPCServiceBase
        RPCServiceBase.__init__(
            self,
            service_name="ChallengePeriodManagerServer",
            port=50003,  # Unique port for ChallengePeriodManager
            running_unit_tests=running_unit_tests,
            enable_health_check=True,
            health_check_interval_s=60,
            max_consecutive_failures=3,
            enable_auto_restart=True,
            slack_notifier=slack_notifier
        )

        # Initialize CacheController
        CacheController.__init__(self, metagraph=metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)

        # Store dependencies needed for server creation (using private attributes for properties)
        self._perf_ledger_manager = perf_ledger_manager
        self._position_manager = position_manager
        self._contract_manager = contract_manager
        self._plagiarism_manager = plagiarism_manager
        self._asset_selection_manager = asset_selection_manager
        self.shutdown_dict = shutdown_dict
        self.is_backtesting = is_backtesting
        self.sync_in_progress = sync_in_progress
        self.sync_epoch = sync_epoch
        self.asset_selection_manager = asset_selection_manager

        # Start the RPC service (this replaces direct initialization)
        self._initialize_service()

        # Start daemon process if requested (for continuous challenge period updates)
        if start_daemon:
            self._start_daemon_process()

    def _start_daemon_process(self):
        """
        Start the daemon process for continuous challenge period updates.

        The daemon process runs run_update_loop() on the server, which continuously
        refreshes the challenge period manager state (updating buckets, handling
        promotions/demotions, etc.).

        This is separate from the RPC server process - the RPC server handles
        requests, while the daemon process performs continuous background updates.
        """

        # Call run_update_loop on the server (either direct or via RPC)
        if self.running_unit_tests:
            # In test mode with direct server, just call run_update_loop directly
            bt.logging.info("ChallengePeriodManager: Test mode - run_update_loop must be called manually")
            # We don't start a daemon in test mode because tests control the flow
        else:
            # In production with RPC server, the server already has run_update_loop
            # We need to start a daemon process that calls it
            daemon_process = multiprocessing.Process(
                target=self._server_proxy.run_update_loop,
                daemon=True
            )
            daemon_process.start()
            bt.logging.success(
                f"ChallengePeriodManager daemon process started with PID: {daemon_process.pid}"
            )

    # ==================== Dependency Properties (auto-sync to server in test mode) ====================

    @property
    def perf_ledger_manager(self):
        """Performance ledger manager dependency"""
        return self._perf_ledger_manager

    @perf_ledger_manager.setter
    def perf_ledger_manager(self, value):
        """Set perf ledger manager and auto-sync to server in test mode"""
        self._perf_ledger_manager = value
        # In test mode, also update server instance
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.perf_ledger_manager = value

    @property
    def position_manager(self):
        """Position manager dependency"""
        return self._position_manager

    @position_manager.setter
    def position_manager(self, value):
        """Set position manager and auto-sync to server in test mode"""
        self._position_manager = value
        # In test mode, also update server instance
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.position_manager = value

    @property
    def contract_manager(self):
        """Contract manager dependency"""
        return self._contract_manager

    @contract_manager.setter
    def contract_manager(self, value):
        """Set contract manager and auto-sync to server in test mode"""
        self._contract_manager = value
        # In test mode, also update server instance
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.contract_manager = value

    @property
    def plagiarism_manager(self):
        """Plagiarism manager dependency"""
        return self._plagiarism_manager

    @plagiarism_manager.setter
    def plagiarism_manager(self, value):
        """Set plagiarism manager and auto-sync to server in test mode"""
        self._plagiarism_manager = value
        # In test mode, also update server instance
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.plagiarism_manager = value

    @property
    def asset_selection_manager(self):
        """Asset selection manager dependency"""
        return self._asset_selection_manager

    @asset_selection_manager.setter
    def asset_selection_manager(self, value):
        """Set asset selection manager and auto-sync to server in test mode"""
        self._asset_selection_manager = value
        # In test mode, also update server instance
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.asset_selection_manager = value

    @property
    def active_miners(self):
        """
        Direct access to active_miners dict (test mode only).

        In test mode with direct server, provides access to the server's active_miners dict.
        DO NOT use in production - use RPC methods instead (set_miner_bucket, get_miner_bucket, etc.)

        Returns:
            dict: The server's active_miners dict (direct reference in test mode)
        """
        if not self.running_unit_tests:
            raise NotImplementedError(
                "active_miners direct access is only available in test mode. "
                "Use RPC methods like set_miner_bucket(), get_miner_bucket(), etc."
            )
        return self._server_proxy.active_miners

    @active_miners.setter
    def active_miners(self, value):
        """
        Set active_miners dict (test mode only).

        Args:
            value: Dict mapping hotkeys to (bucket, start_time, prev_bucket, prev_time) tuples
        """
        if not self.running_unit_tests:
            raise NotImplementedError(
                "active_miners direct access is only available in test mode. "
                "Use RPC methods like set_miner_bucket(), update_miners(), etc."
            )
        self._server_proxy.active_miners = value


    # ==================== Client Methods (proxy to RPC) ====================

    # ==================== Elimination Reasons Methods ====================

    def get_all_elimination_reasons(self) -> dict:
        """
        Get all elimination reasons as a dict.

        Returns:
            dict: Mapping hotkeys to (reason, drawdown) tuples
        """
        return self._server_proxy.get_all_elimination_reasons_rpc()

    def has_elimination_reasons(self) -> bool:
        """
        Check if there are any elimination reasons.

        Returns:
            bool: True if elimination reasons exist
        """
        return self._server_proxy.has_elimination_reasons_rpc()

    def clear_elimination_reasons(self) -> None:
        """Clear all elimination reasons."""
        self._server_proxy.clear_elimination_reasons_rpc()

    def update_elimination_reasons(self, reasons_dict: dict) -> int:
        """
        Bulk update elimination reasons from a dict.

        Args:
            reasons_dict: Dict mapping hotkeys to (reason, drawdown) tuples

        Returns:
            int: Number of elimination reasons set
        """
        return self._server_proxy.update_elimination_reasons_rpc(reasons_dict)

    # ==================== Active Miners Methods ====================

    def has_miner(self, hotkey: str) -> bool:
        """
        Fast check if a miner is in active_miners (O(1)).

        Args:
            hotkey: The miner hotkey to check

        Returns:
            bool: True if miner is active
        """
        return self._server_proxy.has_miner_rpc(hotkey)

    def get_miner_bucket(self, hotkey: str) -> Optional[MinerBucket]:
        """
        Get the bucket of a miner.

        Args:
            hotkey: The miner hotkey to look up

        Returns:
            MinerBucket: Bucket enum, or None if not found
        """
        bucket_value = self._server_proxy.get_miner_bucket_rpc(hotkey)
        return MinerBucket(bucket_value) if bucket_value else None

    def get_miner_start_time(self, hotkey: str) -> Optional[int]:
        """
        Get the start time of a miner's current bucket.

        Args:
            hotkey: The miner hotkey to look up

        Returns:
            int: Start time in milliseconds, or None if not found
        """
        return self._server_proxy.get_miner_start_time_rpc(hotkey)

    def get_miner_previous_bucket(self, hotkey: str) -> Optional[MinerBucket]:
        """
        Get the previous bucket of a miner (used for plagiarism demotions).

        Args:
            hotkey: The miner hotkey to look up

        Returns:
            MinerBucket: Previous bucket enum, or None if not found or not set
        """
        prev_bucket_value = self._server_proxy.get_miner_previous_bucket_rpc(hotkey)
        return MinerBucket(prev_bucket_value) if prev_bucket_value else None

    def get_miner_previous_time(self, hotkey: str) -> Optional[int]:
        """
        Get the start time of a miner's previous bucket.

        Args:
            hotkey: The miner hotkey to look up

        Returns:
            int: Previous bucket start time in milliseconds, or None if not found or not set
        """
        return self._server_proxy.get_miner_previous_time_rpc(hotkey)

    def get_hotkeys_by_bucket(self, bucket: MinerBucket) -> List[str]:
        """
        Get all hotkeys in a specific bucket.

        Args:
            bucket: Bucket enum (e.g., MinerBucket.CHALLENGE, MinerBucket.MAINCOMP)

        Returns:
            list: List of hotkeys in the bucket
        """
        return self._server_proxy.get_hotkeys_by_bucket_rpc(bucket.value)

    def get_all_miner_hotkeys(self) -> List[str]:
        """
        Get list of all active miner hotkeys.

        Returns:
            list: List of hotkeys
        """
        return self._server_proxy.get_all_miner_hotkeys_rpc()

    def set_miner_bucket(
        self,
        hotkey: str,
        bucket: MinerBucket,
        start_time: int,
        prev_bucket: Optional[MinerBucket] = None,
        prev_time: Optional[int] = None
    ) -> bool:
        """
        Set or update a miner's bucket information.

        Args:
            hotkey: The miner hotkey
            bucket: The current bucket
            start_time: Bucket start time in milliseconds
            prev_bucket: Previous bucket (for plagiarism demotions)
            prev_time: Previous bucket start time (for plagiarism demotions)

        Returns:
            bool: True if this is a new miner, False if updating existing
        """
        return self._server_proxy.set_miner_bucket_rpc(
            hotkey,
            bucket.value,
            start_time,
            prev_bucket.value if prev_bucket else None,
            prev_time
        )

    def remove_miner(self, hotkey: str) -> bool:
        """
        Remove a miner from active_miners.

        Args:
            hotkey: The miner hotkey to remove

        Returns:
            bool: True if removed, False if not found
        """
        return self._server_proxy.remove_miner_rpc(hotkey)

    def clear_all_miners(self) -> None:
        """Clear all miners from active_miners."""
        self._server_proxy.clear_all_miners_rpc()

    def update_miners(self, miners_dict: dict) -> int:
        """
        Bulk update active_miners from a dict.
        Used for syncing from another validator.

        Args:
            miners_dict: Dict mapping hotkeys to (bucket, start_time, prev_bucket, prev_time) tuples

        Returns:
            int: Number of miners updated
        """
        # Convert tuples to dicts for RPC serialization
        miners_rpc_dict = {}
        for hotkey, (bucket, start_time, prev_bucket, prev_time) in miners_dict.items():
            miners_rpc_dict[hotkey] = {
                "bucket": bucket.value,
                "start_time": start_time,
                "prev_bucket": prev_bucket.value if prev_bucket else None,
                "prev_time": prev_time
            }

        return self._server_proxy.update_miners_rpc(miners_rpc_dict)

    def iter_active_miners(self):
        """
        Iterate over active miners.
        Note: This fetches ALL miners and iterates locally.

        Yields:
            Tuples of (hotkey, bucket, start_time, prev_bucket, prev_time)
        """
        # Get all miners from server
        testing_miners = self.get_testing_miners()
        success_miners = self.get_success_miners()
        probation_miners = self.get_probation_miners()
        plagiarism_miners = self.get_plagiarism_miners()

        # Iterate over each bucket
        for hotkey, start_time in testing_miners.items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.CHALLENGE, start_time, prev_bucket, prev_time

        for hotkey, start_time in success_miners.items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.MAINCOMP, start_time, prev_bucket, prev_time

        for hotkey, start_time in probation_miners.items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.PROBATION, start_time, prev_bucket, prev_time

        for hotkey, start_time in plagiarism_miners.items():
            prev_bucket = self.get_miner_previous_bucket(hotkey)
            prev_time = self.get_miner_previous_time(hotkey)
            yield hotkey, MinerBucket.PLAGIARISM, start_time, prev_bucket, prev_time

    def get_testing_miners(self) -> dict:
        """Get all CHALLENGE bucket miners as dict {hotkey: start_time}."""
        return self._server_proxy.get_testing_miners_rpc()

    def get_success_miners(self) -> dict:
        """Get all MAINCOMP bucket miners as dict {hotkey: start_time}."""
        return self._server_proxy.get_success_miners_rpc()

    def get_probation_miners(self) -> dict:
        """Get all PROBATION bucket miners as dict {hotkey: start_time}."""
        return self._server_proxy.get_probation_miners_rpc()

    def get_plagiarism_miners(self) -> dict:
        """Get all PLAGIARISM bucket miners as dict {hotkey: start_time}."""
        return self._server_proxy.get_plagiarism_miners_rpc()

    # ==================== Management Methods (exposed via RPC) ====================

    def _clear_challengeperiod_in_memory_and_disk(self):
        """
        Clear all challenge period data (memory and disk).
        Uses RPC in both test and production modes.
        """
        self._server_proxy.clear_challengeperiod_in_memory_and_disk_rpc()

    def _write_challengeperiod_from_memory_to_disk(self):
        """
        Write challenge period data from memory to disk.
        Uses RPC in both test and production modes.
        """
        self._server_proxy.write_challengeperiod_from_memory_to_disk_rpc()

    def sync_challenge_period_data(self, active_miners_sync):
        """
        Sync challenge period data from another validator.
        Used for P2P synchronization.
        Uses RPC in both test and production modes.

        Args:
            active_miners_sync: Checkpoint dict from another validator
        """
        self._server_proxy.sync_challenge_period_data_rpc(active_miners_sync)

    def refresh(self, current_time: int = None, iteration_epoch=None):
        """
        Refresh the challenge period manager.
        Uses RPC in both test and production modes.

        Args:
            current_time: Current time in milliseconds
            iteration_epoch: Epoch captured at start of iteration
        """
        self._server_proxy.refresh_rpc(current_time=current_time, iteration_epoch=iteration_epoch)

    def meets_time_criteria(self, current_time, bucket_start_time, bucket):
        """
        Check if a miner meets time criteria for their bucket.
        Uses RPC in both test and production modes.

        Args:
            current_time: Current time in milliseconds
            bucket_start_time: Bucket start time in milliseconds
            bucket: MinerBucket enum

        Returns:
            bool: True if miner meets time criteria
        """
        return self._server_proxy.meets_time_criteria_rpc(current_time, bucket_start_time, bucket.value)

    def remove_eliminated(self, eliminations=None):
        """
        Remove eliminated miners from active_miners.
        Uses RPC in both test and production modes.

        Args:
            eliminations: Optional list of elimination dicts. If None, uses elimination_manager.
        """
        self._server_proxy.remove_eliminated_rpc(eliminations=eliminations)

    def update_plagiarism_miners(self, current_time, plagiarism_miners):
        """
        Update plagiarism miners.
        Uses RPC in both test and production modes.

        Args:
            current_time: Current time in milliseconds
            plagiarism_miners: Dict of plagiarism miners {hotkey: start_time}
        """
        self._server_proxy.update_plagiarism_miners_rpc(current_time, plagiarism_miners)

    def prepare_plagiarism_elimination_miners(self, current_time):
        """
        Prepare plagiarism miners for elimination.
        Uses RPC in both test and production modes.

        Args:
            current_time: Current time in milliseconds

        Returns:
            dict: Mapping hotkeys to (reason, drawdown) tuples
        """
        return self._server_proxy.prepare_plagiarism_elimination_miners_rpc(current_time)

    def _demote_plagiarism_in_memory(self, hotkeys, current_time):
        """
        Demote miners to plagiarism bucket (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            hotkeys: List of hotkeys to demote
            current_time: Current time in milliseconds
        """
        self._server_proxy._demote_plagiarism_in_memory_rpc(hotkeys, current_time)

    def _promote_plagiarism_to_previous_bucket_in_memory(self, hotkeys, current_time):
        """
        Promote plagiarism miners to their previous bucket (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            hotkeys: List of hotkeys to promote
            current_time: Current time in milliseconds
        """
        self._server_proxy._promote_plagiarism_to_previous_bucket_in_memory_rpc(hotkeys, current_time)

    def _eliminate_challengeperiod_in_memory(self, eliminations_with_reasons):
        """
        Eliminate miners from challenge period (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            eliminations_with_reasons: Dict mapping hotkeys to (reason, drawdown) tuples
        """
        self._server_proxy._eliminate_challengeperiod_in_memory_rpc(eliminations_with_reasons)

    def _add_challengeperiod_testing_in_memory_and_disk(
        self,
        new_hotkeys,
        eliminations,
        hk_to_first_order_time,
        default_time
    ):
        """
        Add miners to challenge period (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            new_hotkeys: List of hotkeys to potentially add
            eliminations: List of elimination dicts
            hk_to_first_order_time: Dict mapping hotkeys to first order timestamps
            default_time: Default time to use if no first order time
        """
        self._server_proxy._add_challengeperiod_testing_in_memory_and_disk_rpc(
            new_hotkeys=new_hotkeys,
            eliminations=eliminations,
            hk_to_first_order_time=hk_to_first_order_time,
            default_time=default_time
        )

    def _promote_challengeperiod_in_memory(self, hotkeys, current_time):
        """
        Promote miners to main competition (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            hotkeys: List of hotkeys to promote
            current_time: Current time in milliseconds
        """
        self._server_proxy._promote_challengeperiod_in_memory_rpc(hotkeys, current_time)

    def inspect(
        self,
        positions,
        ledger,
        success_hotkeys,
        probation_hotkeys,
        inspection_hotkeys,
        current_time,
        hk_to_first_order_time=None,
        combined_scores_dict=None
    ):
        """
        Run challenge period inspection (exposed for testing).
        Uses RPC in both test and production modes.

        Returns:
            tuple: (hotkeys_to_promote, hotkeys_to_demote, miners_to_eliminate)
        """
        return self._server_proxy.inspect_rpc(
            positions=positions,
            ledger=ledger,
            success_hotkeys=success_hotkeys,
            probation_hotkeys=probation_hotkeys,
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time,
            combined_scores_dict=combined_scores_dict
        )

    def to_checkpoint_dict(self) -> dict:
        """
        Get challenge period data as a checkpoint dict for serialization.
        Uses RPC in both test and production modes.

        Returns:
            dict: Mapping hotkeys to their bucket information
        """
        return self._server_proxy.to_checkpoint_dict_rpc()

    @staticmethod
    def parse_checkpoint_dict(json_dict):
        """
        Static method: Parse checkpoint dictionary from disk format.
        Available in both test and RPC modes.
        """
        from vali_objects.utils.challengeperiod_manager_server import ChallengePeriodManagerServer
        return ChallengePeriodManagerServer.parse_checkpoint_dict(json_dict)

    @staticmethod
    def screen_minimum_interaction(ledger_element) -> bool:
        """
        Static method: Check if miner has minimum number of trading days.
        Available in both test and RPC modes.

        Args:
            ledger_element: Performance ledger element (portfolio ledger)

        Returns:
            bool: True if miner has enough trading days, False otherwise
        """
        from vali_objects.utils.challengeperiod_manager_server import ChallengePeriodManagerServer
        return ChallengePeriodManagerServer.screen_minimum_interaction(ledger_element)
