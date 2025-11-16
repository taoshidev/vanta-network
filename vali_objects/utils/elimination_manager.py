# developer: jbonilla
# Copyright © 2024 Taoshi Inc
from enum import Enum
from typing import Dict, Set, List, Optional
from multiprocessing import Process
from shared_objects.rpc_service_base import RPCServiceBase
from shared_objects.cache_controller import CacheController

import bittensor as bt


class EliminationReason(Enum):
    ZOMBIE = "ZOMBIE"
    PLAGIARISM = "PLAGIARISM"
    MAX_TOTAL_DRAWDOWN = "MAX_TOTAL_DRAWDOWN"
    FAILED_CHALLENGE_PERIOD_TIME = "FAILED_CHALLENGE_PERIOD_TIME"
    FAILED_CHALLENGE_PERIOD_DRAWDOWN = "FAILED_CHALLENGE_PERIOD_DRAWDOWN"
    LIQUIDATED = "LIQUIDATED"


# Constants for departed hotkeys tracking
DEPARTED_HOTKEYS_KEY = "departed_hotkeys"


class EliminationManager(RPCServiceBase, CacheController):
    """"
    RPC Client for EliminationManager - manages elimination state via RPC.

    This client connects to EliminationManagerServer running in a separate process.
    Much faster than IPC managerized dicts (50-200x improvement on batch operations).

    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """

    def __init__(self, metagraph, position_manager, challengeperiod_manager,
                 running_unit_tests=False, shutdown_dict=None, use_ipc=False, is_backtesting=False,
                 shared_queue_websockets=None, contract_manager=None, position_locks=None,
                 sync_in_progress=None, slack_notifier=None, sync_epoch=None, limit_order_manager=None):

        # Initialize RPCServiceBase
        RPCServiceBase.__init__(
            self,
            service_name="EliminationManagerServer",
            port=50004,  # Unique port for EliminationManager
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
        self._position_manager = position_manager
        self._challengeperiod_manager = challengeperiod_manager
        self._contract_manager = contract_manager
        self.shutdown_dict = shutdown_dict
        self.is_backtesting = is_backtesting
        self.shared_queue_websockets = shared_queue_websockets
        self.position_locks = position_locks
        self.sync_in_progress = sync_in_progress
        self.sync_epoch = sync_epoch
        self.limit_order_manager = limit_order_manager

        # Start the RPC service (this replaces direct initialization)
        self._initialize_service()

    # ==================== Dependency Properties (auto-sync to server in test mode) ====================

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
    def challengeperiod_manager(self):
        """Challenge period manager dependency"""
        return self._challengeperiod_manager

    @challengeperiod_manager.setter
    def challengeperiod_manager(self, value):
        """Set challenge period manager and auto-sync to server in test mode"""
        self._challengeperiod_manager = value
        # In test mode, also update server instance
        if self.running_unit_tests and hasattr(self, '_server_proxy') and self._server_proxy:
            self._server_proxy.challengeperiod_manager = value

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

    # ==================== Client Methods (proxy to RPC) ====================

    def get_eliminations_lock(self):
        """
        Get the shared eliminations lock for cross-process synchronization.

        NOTE: This method should NOT be called on the RPC client. The lock is local
        to the server process. If you need synchronized access, make RPC calls which
        are automatically synchronized server-side.

        Raises:
            NotImplementedError: Always, because lock is server-side only
        """

        raise NotImplementedError(
            "get_eliminations_lock() is not available on RPC client. "
            "Locking happens automatically on server side for all RPC calls. "
            "If you need synchronized access, make RPC method calls instead."
        )

    def is_hotkey_eliminated(self, hotkey: str) -> bool:
        """
        Fast-path check if a hotkey is eliminated (O(1)).
        Use this in performance-critical paths like should_fail_early().

        Returns:
            bool: True if hotkey is eliminated, False otherwise
        """
        return self._server_proxy.is_hotkey_eliminated_rpc(hotkey)

    def hotkey_in_eliminations(self, hotkey: str) -> Optional[dict]:
        """
        Get full elimination details for a hotkey (O(1)).
        Returns the complete elimination dict with all metadata.

        Returns:
            dict or None: Elimination details if found, None otherwise
        """
        return self._server_proxy.get_elimination_rpc(hotkey)

    def get_elimination(self, hotkey: str) -> Optional[dict]:
        """
        Get elimination details for a hotkey.

        Args:
            hotkey: The hotkey to look up

        Returns:
            Elimination dict if found, None otherwise

        Example:
            elim = manager.get_elimination("miner_hotkey")
            if elim:
                print(f"Eliminated for: {elim['reason']}")
        """
        return self._server_proxy.get_elimination_rpc(hotkey)

    def get_eliminated_hotkeys(self) -> Set[str]:
        """Get all eliminated hotkeys as a set"""
        return self._server_proxy.get_eliminated_hotkeys_rpc()

    def get_eliminations_from_memory(self) -> List[dict]:
        """Get all eliminations as a list"""
        return self._server_proxy.get_eliminations_from_memory_rpc()

    def get_eliminations_from_disk(self) -> list:
        """Load eliminations from disk"""
        return self._server_proxy.get_eliminations_from_disk_rpc()

    def append_elimination_row(self, hotkey: str, current_dd: float, reason: str,
                                t_ms: int = None, price_info: dict = None, return_info: dict = None) -> None:
        """
        Add elimination row (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            hotkey: The hotkey to eliminate
            current_dd: Current drawdown
            reason: Elimination reason
            t_ms: Optional timestamp in milliseconds
            price_info: Optional price information
            return_info: Optional return information
        """
        self._server_proxy.append_elimination_row_rpc(hotkey, current_dd, reason,
                                                       t_ms=t_ms, price_info=price_info,
                                                       return_info=return_info)

    def add_elimination(self, hotkey: str, elimination_data: dict) -> bool:
        """
        Add or update an elimination record.

        Args:
            hotkey: The hotkey to eliminate
            elimination_data: Elimination dict with required fields

        Returns:
            True if added (new), False if already exists (updated)

        Raises:
            ValueError: If elimination_data is invalid

        Example:
            manager.add_elimination("miner_hotkey", {
                'hotkey': "miner_hotkey",
                'reason': EliminationReason.MAX_TOTAL_DRAWDOWN.value,
                'dd': 0.12,
                'elimination_initiated_time_ms': TimeUtil.now_in_millis()
            })
        """
        return self._server_proxy.add_elimination_rpc(hotkey, elimination_data)

    def remove_elimination(self, hotkey: str) -> bool:
        """
        Remove a single elimination.

        Args:
            hotkey: The hotkey to remove

        Returns:
            True if removed, False if not found

        Example:
            if manager.remove_elimination("miner_hotkey"):
                print("Elimination removed")
        """
        return self._server_proxy.remove_elimination_rpc(hotkey)

    def sync_eliminations(self, dat: list) -> list:
        """
        Sync eliminations from external source (batch update).

        Args:
            dat: List of elimination dicts to sync

        Returns:
            List of removed hotkeys
        """
        removed = self._server_proxy.sync_eliminations_rpc(dat)
        bt.logging.info(f'sync_eliminations: removed {len(removed)} hotkeys')
        return removed

    def clear_eliminations(self) -> None:
        """Clear all eliminations"""
        self._server_proxy.clear_eliminations_rpc()

    def is_hotkey_re_registered(self, hotkey: str) -> bool:
        """
        Check if a hotkey is re-registered (was previously de-registered and has re-registered).

        Args:
            hotkey: The hotkey to check

        Returns:
            True if the hotkey is in the metagraph AND in the departed_hotkeys dict, False otherwise
        """
        return self._server_proxy.is_hotkey_re_registered_rpc(hotkey)

    def get_departed_hotkeys(self) -> Dict[str, dict]:
        """Get all departed hotkeys"""
        return self._server_proxy.get_departed_hotkeys_rpc()

    def get_departed_hotkey_info(self, hotkey: str) -> Optional[dict]:
        """
        Get departed info for a single hotkey (O(1) RPC call).

        Args:
            hotkey: The hotkey to look up

        Returns:
            dict with 'detected_ms' if hotkey is departed, None otherwise
        """
        return self._server_proxy.get_departed_hotkey_info_rpc(hotkey)

    def delete_eliminations(self, deleted_hotkeys):
        """
        Delete multiple eliminations.

        Note: This is not exposed as RPC. Use remove_elimination() for single deletions
        or sync_eliminations() for batch updates.
        """
        for hotkey in deleted_hotkeys:
            self.remove_elimination(hotkey)

    def process_eliminations(self, position_locks=None, iteration_epoch=None):
        """
        Trigger elimination processing.
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager (optional, uses default if None)
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.process_eliminations_rpc(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def handle_perf_ledger_eliminations(self, position_locks=None, iteration_epoch=None):
        """
        Process performance ledger eliminations (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager (optional, uses default if None)
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.handle_perf_ledger_eliminations_rpc(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def handle_first_refresh(self, position_locks, iteration_epoch=None):
        """
        Handle first refresh on startup (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.handle_first_refresh_rpc(position_locks, iteration_epoch)

    @property
    def first_refresh_ran(self) -> bool:
        """
        Get the first_refresh_ran flag.
        Indicates whether the first refresh has been executed after validator startup.

        Returns:
            bool: True if first refresh has run, False otherwise
        """
        return self._server_proxy.get_first_refresh_ran_rpc()

    @first_refresh_ran.setter
    def first_refresh_ran(self, value: bool):
        """
        Set the first_refresh_ran flag.

        Args:
            value: Boolean value to set
        """
        self._server_proxy.set_first_refresh_ran_rpc(value)

    def is_zombie_hotkey(self, hotkey: str, all_hotkeys_set: set) -> bool:
        """
        Check if a hotkey is a zombie (not in metagraph).
        Uses RPC in both test and production modes.

        Args:
            hotkey: The hotkey to check
            all_hotkeys_set: Set of all current hotkeys in metagraph

        Returns:
            bool: True if hotkey is a zombie, False otherwise
        """
        return self._server_proxy.is_zombie_hotkey_rpc(hotkey, all_hotkeys_set)

    def handle_mdd_eliminations(self, position_locks=None, iteration_epoch=None):
        """
        Check for maximum drawdown eliminations (exposed for testing).
        Uses RPC in both test and production modes.

        Args:
            position_locks: Position locks manager (optional, uses default if None)
            iteration_epoch: Epoch captured at start of iteration (optional)
        """
        self._server_proxy.handle_mdd_eliminations_rpc(position_locks=position_locks, iteration_epoch=iteration_epoch)

    def save_eliminations(self):
        """
        Save eliminations to disk.
        Uses RPC in both test and production modes.
        """
        self._server_proxy.save_eliminations_rpc()

    def write_eliminations_to_disk(self, eliminations: list):
        """
        Write eliminations to disk.
        Uses RPC in both test and production modes.

        Args:
            eliminations: List of elimination dicts to write
        """
        self._server_proxy.write_eliminations_to_disk_rpc(eliminations)

    @property
    def eliminations(self) -> Dict[str, dict]:
        """
        Get eliminations dict (readonly copy).
        For test mode compatibility.

        Returns:
            dict: Copy of eliminations dict mapping hotkey to elimination data
        """
        return self._server_proxy.get_eliminations_dict_rpc()
