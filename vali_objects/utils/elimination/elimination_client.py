# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
EliminationClient - Lightweight RPC client for elimination management.

This client connects to the EliminationServer via RPC.
Can be created in ANY process - just needs the server to be running.

Usage:
    from vali_objects.utils.elimination_client import EliminationClient

    # Connect to server (uses ValiConfig.RPC_ELIMINATION_PORT by default)
    client = EliminationClient()

    if client.is_hotkey_eliminated(hotkey):
        print("Hotkey is eliminated")

"""
from typing import Dict, Set, List, Optional

import bittensor as bt

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from time_util.time_util import TimeUtil


class EliminationClient(RPCClientBase):
    """
    Lightweight RPC client for EliminationServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_ELIMINATION_PORT.

    Supports local caching for fast lookups without RPC calls:
        client = EliminationClient(local_cache_refresh_period_ms=5000)
        # Fast local lookup (no RPC):
        elim_info = client.get_elimination_local_cache(hotkey)

    """

    def __init__(
        self,
        port: int = None,
        local_cache_refresh_period_ms: int = None,
        connect_immediately: bool = False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize elimination client.

        Args:
            port: Port number of the elimination server (default: ValiConfig.RPC_ELIMINATION_PORT)
            local_cache_refresh_period_ms: If not None, spawn a daemon thread that refreshes
                a local cache at this interval for fast lookups without RPC.
            connection_mode: RPCConnectionMode.LOCAL for tests (use set_direct_server()), RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        # In LOCAL mode, don't connect via RPC - tests will set direct server
        super().__init__(
            service_name=ValiConfig.RPC_ELIMINATION_SERVICE_NAME,
            port=port or ValiConfig.RPC_ELIMINATION_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            local_cache_refresh_period_ms=local_cache_refresh_period_ms,
            connection_mode=connection_mode
        )

    # ==================== Query Methods ====================

    def is_hotkey_eliminated(self, hotkey: str) -> bool:
        """
        Fast-path check if a hotkey is eliminated (O(1)).

        Args:
            hotkey: The hotkey to check

        Returns:
            bool: True if hotkey is eliminated, False otherwise
        """
        return self._server.is_hotkey_eliminated_rpc(hotkey)

    def get_elimination(self, hotkey: str) -> Optional[dict]:
        """
        Get elimination details for a hotkey.

        Args:
            hotkey: The hotkey to look up

        Returns:
            Elimination dict if found, None otherwise
        """
        return self._server.get_elimination_rpc(hotkey)

    def hotkey_in_eliminations(self, hotkey: str) -> Optional[dict]:
        """Alias for get_elimination() for backward compatibility."""
        return self._server.get_elimination_rpc(hotkey)

    def get_eliminated_hotkeys(self) -> Set[str]:
        """Get all eliminated hotkeys as a set."""
        return self._server.get_eliminated_hotkeys_rpc()

    def get_eliminations_from_memory(self) -> List[dict]:
        """Get all eliminations as a list."""
        return self._server.get_eliminations_from_memory_rpc()

    def get_eliminations_from_disk(self) -> list:
        """Load eliminations from disk."""
        return self._server.get_eliminations_from_disk_rpc()

    def get_eliminations_dict(self) -> Dict[str, dict]:
        """Get eliminations dict (readonly copy)."""
        return self._server.get_eliminations_dict_rpc()

    @property
    def eliminations(self) -> Dict[str, dict]:
        """Get eliminations dict (readonly copy)."""
        return self._server.get_eliminations_dict_rpc()

    # ==================== Mutation Methods ====================

    def append_elimination_row(
        self,
        hotkey: str,
        current_dd: float,
        reason: str,
        t_ms: int = None,
        price_info: dict = None,
        return_info: dict = None
    ) -> None:
        """
        Add elimination row.

        Args:
            hotkey: The hotkey to eliminate
            current_dd: Current drawdown
            reason: Elimination reason
            t_ms: Optional timestamp in milliseconds
            price_info: Optional price information
            return_info: Optional return information
        """
        self._server.append_elimination_row_rpc(
            hotkey, current_dd, reason,
            t_ms=t_ms, price_info=price_info, return_info=return_info
        )

    def add_elimination(self, hotkey: str, elimination_data: dict) -> bool:
        """
        Add or update an elimination record.

        Args:
            hotkey: The hotkey to eliminate
            elimination_data: Elimination dict with required fields

        Returns:
            True if added (new), False if already exists (updated)
        """
        return self._server.add_elimination_rpc(hotkey, elimination_data)

    def remove_elimination(self, hotkey: str) -> bool:
        """
        Remove a single elimination.

        Args:
            hotkey: The hotkey to remove

        Returns:
            True if removed, False if not found
        """
        return self._server.remove_elimination_rpc(hotkey)

    def delete_eliminations(self, deleted_hotkeys) -> None:
        """Delete multiple eliminations."""
        for hotkey in deleted_hotkeys:
            self.remove_elimination(hotkey)

    def sync_eliminations(self, dat: list) -> list:
        """
        Sync eliminations from external source (batch update).

        Args:
            dat: List of elimination dicts to sync

        Returns:
            List of removed hotkeys
        """
        removed = self._server.sync_eliminations_rpc(dat)
        bt.logging.info(f'sync_eliminations: removed {len(removed)} hotkeys')
        return removed

    def clear_eliminations(self) -> None:
        """Clear all eliminations."""
        self._server.clear_eliminations_rpc()

    def clear_departed_hotkeys(self) -> None:
        """Clear all departed hotkeys."""
        self._server.clear_departed_hotkeys_rpc()

    def clear_test_state(self) -> None:
        """
        Clear ALL test-sensitive state (comprehensive reset for test isolation).

        This is a high-level cleanup method that resets:
        - Eliminations data
        - Departed hotkeys
        - first_refresh_ran flag
        - Any other stateful flags

        Should be called by ServerOrchestrator.clear_all_test_data() to ensure
        complete test isolation when servers are shared across tests.

        Use this instead of clear_eliminations() alone to prevent test contamination.
        """
        self._server.clear_test_state_rpc()

    def save_eliminations(self) -> None:
        """Save eliminations to disk."""
        self._server.save_eliminations_rpc()

    def write_eliminations_to_disk(self, eliminations: list) -> None:
        """Write eliminations to disk."""
        self._server.write_eliminations_to_disk_rpc(eliminations)

    def load_eliminations_from_disk(self) -> None:
        """Load eliminations from disk into memory (for testing recovery scenarios)."""
        self._server.load_eliminations_from_disk_rpc()

    def reload_from_disk(self) -> None:
        """Alias for load_eliminations_from_disk for backward compatibility."""
        self.load_eliminations_from_disk()

    # ==================== Cache Timing Methods ====================

    def refresh_allowed(self, interval_ms: int) -> bool:
        """
        Check if cache refresh is allowed based on time elapsed since last update.

        Args:
            interval_ms: Minimum interval in milliseconds between refreshes

        Returns:
            True if refresh is allowed, False otherwise
        """
        return self._server.refresh_allowed_rpc(interval_ms)

    def set_last_update_time(self) -> None:
        """Set the last update time to current time (for cache management)."""
        self._server.set_last_update_time_rpc()

    # ==================== Departed Hotkeys Methods ====================

    def is_hotkey_re_registered(self, hotkey: str) -> bool:
        """Check if a hotkey is re-registered (was departed, now back)."""
        return self._server.is_hotkey_re_registered_rpc(hotkey)

    def get_departed_hotkeys(self) -> Dict[str, dict]:
        """Get all departed hotkeys."""
        return self._server.get_departed_hotkeys_rpc()

    def get_departed_hotkey_info(self, hotkey: str) -> Optional[dict]:
        """Get departed info for a single hotkey."""
        return self._server.get_departed_hotkey_info_rpc(hotkey)

    def get_cached_elimination_data(self) -> tuple:
        """
        Get cached elimination data from server.

        Returns:
            Tuple of (eliminations_dict, departed_hotkeys_dict)
        """
        return self._server.get_cached_elimination_data_rpc()

    # ==================== Processing Methods ====================

    def process_eliminations(self, iteration_epoch=None) -> None:
        """Trigger elimination processing."""
        self._server.process_eliminations_rpc(
            iteration_epoch=iteration_epoch
        )

    def handle_perf_ledger_eliminations(self, iteration_epoch=None) -> None:
        """Process performance ledger eliminations."""
        self._server.handle_perf_ledger_eliminations_rpc(
            iteration_epoch=iteration_epoch
        )

    def handle_first_refresh(self, iteration_epoch=None) -> None:
        """Handle first refresh on startup."""
        self._server.handle_first_refresh_rpc(iteration_epoch)

    def handle_mdd_eliminations(self, iteration_epoch=None) -> None:
        """Check for maximum drawdown eliminations."""
        self._server.handle_mdd_eliminations_rpc(
            iteration_epoch=iteration_epoch
        )

    def handle_eliminated_miner(self, hotkey: str,
                               trade_pair_to_price_source_dict: dict = None,
                               iteration_epoch=None) -> None:
        """
        Handle cleanup for eliminated miner (deletes limit orders, closes positions).

        Args:
            hotkey: The hotkey to clean up
            trade_pair_to_price_source_dict: Dict mapping trade_pair_id (str) to price_source dict
            iteration_epoch: Optional iteration epoch for validation
        """
        self._server.handle_eliminated_miner_rpc(
            hotkey,
            trade_pair_to_price_source_dict=trade_pair_to_price_source_dict,
            iteration_epoch=iteration_epoch
        )

    def is_zombie_hotkey(self, hotkey: str, all_hotkeys_set: set) -> bool:
        """Check if a hotkey is a zombie (not in metagraph)."""
        return self._server.is_zombie_hotkey_rpc(hotkey, all_hotkeys_set)

    # ==================== State Properties ====================

    @property
    def first_refresh_ran(self) -> bool:
        """Get the first_refresh_ran flag."""
        return self._server.get_first_refresh_ran_rpc()

    @first_refresh_ran.setter
    def first_refresh_ran(self, value: bool):
        """Set the first_refresh_ran flag."""
        self._server.set_first_refresh_ran_rpc(value)

    def get_first_refresh_ran(self) -> bool:
        """Get the first_refresh_ran flag (method form for backward compatibility)."""
        return self._server.get_first_refresh_ran_rpc()

    def set_first_refresh_ran(self, value: bool) -> None:
        """Set the first_refresh_ran flag (method form for backward compatibility)."""
        self._server.set_first_refresh_ran_rpc(value)

    # ==================== Daemon Control ====================

    def start_daemon(self) -> None:
        """Request daemon start on server."""
        self._server.start_daemon_rpc()

    # ==================== Utility Methods ====================

    def generate_elimination_row(
        self,
        hotkey: str,
        current_dd: float,
        reason: str,
        t_ms: int = None,
        price_info: dict = None,
        return_info: dict = None
    ) -> dict:
        """
        Generate elimination row dict (client-side helper).

        Args:
            hotkey: The hotkey to eliminate
            current_dd: Current drawdown
            reason: Elimination reason
            t_ms: Optional timestamp in milliseconds
            price_info: Optional price information
            return_info: Optional return information

        Returns:
            Elimination row dict
        """
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()
        return {
            'hotkey': hotkey,
            'dd': current_dd,
            'reason': reason,
            'elimination_initiated_time_ms': t_ms,
            'price_info': price_info or {},
            'return_info': return_info or {}
        }

    # ==================== Local Cache Support ====================

    def populate_cache(self) -> Dict[str, any]:
        """
        Populate the local cache with elimination data from the server.

        Called periodically by the cache refresh daemon when
        local_cache_refresh_period_ms is configured.

        Returns:
            Dict with keys: 'eliminations', 'departed_hotkeys'
        """
        eliminations = self._server.get_eliminations_dict_rpc()
        departed_hotkeys = self._server.get_departed_hotkeys_rpc()
        return {
            "eliminations": eliminations,
            "departed_hotkeys": departed_hotkeys
        }

    def get_elimination_local_cache(self, hotkey: str) -> Optional[dict]:
        """
        Get elimination info for a hotkey from the local cache.

        This is a fast local lookup without any RPC call.
        Requires local_cache_refresh_period_ms to be configured.

        Args:
            hotkey: The hotkey to look up

        Returns:
            Elimination dict if found, None otherwise
        """
        with self._local_cache_lock:
            eliminations = self._local_cache.get("eliminations", {})
            return eliminations.get(hotkey)

    def get_departed_hotkey_info_local_cache(self, hotkey: str) -> Optional[dict]:
        """
        Get departed hotkey info from the local cache.

        This is a fast local lookup without any RPC call.
        Requires local_cache_refresh_period_ms to be configured.

        Args:
            hotkey: The hotkey to look up

        Returns:
            Departed hotkey info dict if found, None otherwise
        """
        with self._local_cache_lock:
            departed = self._local_cache.get("departed_hotkeys", {})
            return departed.get(hotkey)

    def is_hotkey_eliminated_local_cache(self, hotkey: str) -> bool:
        """
        Check if a hotkey is eliminated using local cache.

        This is a fast local lookup without any RPC call.
        Requires local_cache_refresh_period_ms to be configured.

        Args:
            hotkey: The hotkey to check

        Returns:
            True if hotkey is eliminated, False otherwise
        """
        with self._local_cache_lock:
            eliminations = self._local_cache.get("eliminations", {})
            return hotkey in eliminations

    def is_hotkey_re_registered_local_cache(self, hotkey: str) -> bool:
        """
        Check if a hotkey is re-registered using local cache.

        This is a fast local lookup without any RPC call.
        Requires local_cache_refresh_period_ms to be configured.

        Args:
            hotkey: The hotkey to check

        Returns:
            True if hotkey is in departed_hotkeys, False otherwise
        """
        with self._local_cache_lock:
            departed = self._local_cache.get("departed_hotkeys", {})
            return hotkey in departed
