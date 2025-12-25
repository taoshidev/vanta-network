# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
EliminationServer - RPC server for elimination management.

This server runs in its own process and exposes elimination management via RPC.
Clients connect using EliminationClient.

This follows the same pattern as PerfLedgerServer - the server wraps EliminationManager
and exposes its methods via RPC.

Usage:
    # Validator spawns the server at startup
    from vali_objects.utils.elimination_server import EliminationServer

    elimination_server = EliminationServer(
        start_server=True,
        start_daemon=True
    )

    # Other processes connect via EliminationClient
    from vali_objects.utils.elimination_client import EliminationClient
    client = EliminationClient()  # Uses ValiConfig.RPC_ELIMINATION_PORT
"""
import time
import threading

from vali_objects.utils.elimination.elimination_manager import EliminationManager
from typing import Dict, Set, List, Optional
from vali_objects.vali_config import ValiConfig
from setproctitle import setproctitle
from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.vali_config import RPCConnectionMode

import bittensor as bt


# ==================== Server Implementation ====================

class EliminationServer(RPCServerBase):
    """
    RPC server for elimination management.

    Wraps EliminationManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to EliminationClient.

    This follows the same pattern as PerfLedgerServer.
    """
    service_name = ValiConfig.RPC_ELIMINATION_SERVICE_NAME
    service_port = ValiConfig.RPC_ELIMINATION_PORT

    def __init__(
        self,
        is_backtesting=False,
        slack_notifier=None,
        start_server=True,
        start_daemon=False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False,
        serve: bool = False
    ):
        """
        Initialize EliminationServer.

        Args:
            is_backtesting: Whether running in backtesting mode
            position_locks: Position locks manager
            slack_notifier: Slack notifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
            running_unit_tests: Whether running in test mode
            serve: Whether to serve position updates via WebSocketNotifier
        """
        # Create own CommonDataClient (forward compatibility - no parameter passing)
        self.running_unit_tests = running_unit_tests
        self._common_data_client = CommonDataClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Create the actual EliminationManager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        self._manager = EliminationManager(
            is_backtesting=is_backtesting,
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests,
            serve=serve
        )

        bt.logging.info(f"[ELIM_SERVER] EliminationManager initialized")

        # Cache for fast fail-early checks (auto-refreshed by daemon)
        self._eliminations_cache = {}  # {hotkey: elimination_dict}
        self._departed_hotkeys_cache = {}  # {hotkey: departure_info_dict}
        self._cache_lock = threading.Lock()

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager and caches exist, so RPC calls won't fail
        # daemon_interval_s: 5 minutes (elimination checks are moderate frequency)
        # hang_timeout_s: 10 minutes (2x interval, prevents false alarms during startup)
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_ELIMINATION_SERVICE_NAME,
            port=ValiConfig.RPC_ELIMINATION_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=ValiConfig.ELIMINATION_CHECK_INTERVAL_MS // 1000,  # 5 minutes (300s)
            hang_timeout_s=600.0,  # 10 minutes (prevents false alarms during startup)
            connection_mode=connection_mode,
            daemon_stagger_s=20
        )

        # Initial cache population
        self._refresh_cache()

        # Start cache refresh daemon
        if connection_mode == RPCConnectionMode.RPC:
            self._start_cache_refresh_daemon()

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Checks for sync in progress, then processes eliminations via manager.
        """
        if self.sync_in_progress:
            bt.logging.debug("EliminationServer: Sync in progress, pausing...")
            return

        iteration_epoch = self.sync_epoch
        self._manager.process_eliminations(iteration_epoch=iteration_epoch)

    @property
    def sync_in_progress(self):
        """Get sync_in_progress flag via CommonDataClient."""
        return self._common_data_client.get_sync_in_progress()

    @property
    def sync_epoch(self):
        """Get sync_epoch value via CommonDataClient."""
        return self._common_data_client.get_sync_epoch()

    def _refresh_cache(self):
        """
        Refresh the fast-lookup caches from current state (thread-safe).

        Acquires MANAGER's lock first to get consistent snapshot, then updates cache.
        This prevents cache from seeing partial state during manager's sync operations.
        """
        # Get manager's lock for consistent snapshot
        manager_lock = self._manager.get_eliminations_lock()

        # Acquire manager lock to get consistent snapshot
        with manager_lock:
            # Get snapshots while holding manager lock
            eliminations_snapshot = dict(self._manager.eliminations)
            departed_snapshot = dict(self._manager.departed_hotkeys)

        # Update cache (release manager lock first to avoid nested locking)
        with self._cache_lock:
            self._eliminations_cache = eliminations_snapshot
            self._departed_hotkeys_cache = departed_snapshot
            bt.logging.debug(
                f"[CACHE_REFRESH] Refreshed: {len(self._eliminations_cache)} eliminated, "
                f"{len(self._departed_hotkeys_cache)} departed hotkeys"
            )

    def _cache_refresh_loop(self):
        """Background daemon that refreshes cache periodically."""
        setproctitle("vali_EliminationCacheRefresher")
        bt.logging.info(f"Elimination cache refresh daemon started ({ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S}-second interval)")

        while not self._is_shutdown():
            try:
                time.sleep(ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S)
                # Check shutdown again after sleep
                if self._is_shutdown():
                    break
                self._refresh_cache()
            except Exception as e:
                # If we're shutting down, exit gracefully without logging error
                if self._is_shutdown():
                    break
                bt.logging.error(f"Error in cache refresh daemon: {e}")
                time.sleep(ValiConfig.ELIMINATION_CACHE_REFRESH_INTERVAL_S)

        bt.logging.info("Elimination cache refresh daemon shutting down")

    def _start_cache_refresh_daemon(self):
        """Start the background cache refresh thread."""
        refresh_thread = threading.Thread(target=self._cache_refresh_loop, daemon=True)
        refresh_thread.start()
        bt.logging.info("Started cache refresh daemon")

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "num_eliminations": len(self._manager.eliminations),
            "num_departed_hotkeys": len(self._manager.departed_hotkeys)
        }

    def is_hotkey_eliminated_rpc(self, hotkey: str) -> bool:
        """Fast existence check (O(1))"""
        return self._manager.is_hotkey_eliminated(hotkey)

    def get_elimination_rpc(self, hotkey: str) -> Optional[dict]:
        """Get full elimination details"""
        return self._manager.get_elimination(hotkey)

    def get_eliminated_hotkeys_rpc(self) -> Set[str]:
        """Get all eliminated hotkeys"""
        return self._manager.get_eliminated_hotkeys()

    def get_eliminations_from_memory_rpc(self) -> List[dict]:
        """Get all eliminations as a list"""
        return self._manager.get_eliminations_from_memory()

    def get_eliminations_from_disk_rpc(self) -> list:
        """Load eliminations from disk"""
        return self._manager.get_eliminations_from_disk()

    def append_elimination_row_rpc(self, hotkey: str, current_dd: float, reason: str,
                                    t_ms: int = None, price_info: dict = None, return_info: dict = None) -> None:
        """Add elimination row."""
        self._manager.append_elimination_row(hotkey, current_dd, reason, t_ms=t_ms,
                                            price_info=price_info, return_info=return_info)

    def add_elimination_rpc(self, hotkey: str, elimination_data: dict) -> bool:
        """Add or update an elimination record. Returns True if new, False if updated."""
        return self._manager.add_elimination(hotkey, elimination_data)

    def remove_elimination_rpc(self, hotkey: str) -> bool:
        """Remove elimination. Returns True if removed, False if not found."""
        return self._manager.remove_elimination(hotkey)

    def sync_eliminations_rpc(self, eliminations_list: list) -> list:
        """Sync eliminations from external source (batch update). Returns list of removed hotkeys."""
        return self._manager.sync_eliminations(eliminations_list)

    def clear_eliminations_rpc(self) -> None:
        """Clear all eliminations for testing"""
        self._manager.clear_eliminations()

    def clear_departed_hotkeys_rpc(self) -> None:
        """Clear all departed hotkeys for testing"""
        self._manager.clear_departed_hotkeys()

    def clear_test_state_rpc(self) -> None:
        """
        Clear ALL test-sensitive state (for test isolation).

        This is a comprehensive reset that includes:
        - Eliminations data
        - Departed hotkeys
        - first_refresh_ran flag (prevents test contamination)
        - Any other stateful flags that affect test behavior

        Should be called by ServerOrchestrator.clear_all_test_data() to ensure
        complete test isolation when servers are shared across tests.
        """
        self._manager.clear_eliminations()
        self._manager.clear_departed_hotkeys()
        self._manager.first_refresh_ran = False  # Reset flag to allow handle_first_refresh() in each test
        # Future: Add any other stateful flags here

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def is_hotkey_eliminated(self, hotkey: str) -> bool:
        """Fast existence check (O(1))"""
        return self.is_hotkey_eliminated_rpc(hotkey)

    def get_elimination(self, hotkey: str) -> Optional[dict]:
        """Get full elimination details"""
        return self.get_elimination_rpc(hotkey)

    def hotkey_in_eliminations(self, hotkey: str) -> Optional[dict]:
        """Alias for get_elimination()"""
        return self.get_elimination_rpc(hotkey)

    def get_eliminated_hotkeys(self) -> Set[str]:
        """Get all eliminated hotkeys"""
        return self.get_eliminated_hotkeys_rpc()

    def get_eliminations_from_memory(self) -> List[dict]:
        """Get all eliminations as a list"""
        return self.get_eliminations_from_memory_rpc()

    def add_elimination(self, hotkey: str, elimination_data: dict) -> bool:
        """Add or update an elimination record."""
        return self.add_elimination_rpc(hotkey, elimination_data)

    def remove_elimination(self, hotkey: str) -> bool:
        """Remove elimination."""
        return self.remove_elimination_rpc(hotkey)

    def sync_eliminations(self, eliminations_list: list) -> list:
        """Sync eliminations from external source."""
        return self.sync_eliminations_rpc(eliminations_list)

    def clear_eliminations(self) -> None:
        """Clear all eliminations"""
        self.clear_eliminations_rpc()

    def is_hotkey_re_registered_rpc(self, hotkey: str) -> bool:
        """Check if hotkey is re-registered (was departed, now back)"""
        return self._manager.is_hotkey_re_registered(hotkey)

    def get_departed_hotkeys_rpc(self) -> Dict[str, dict]:
        """Get all departed hotkeys"""
        return self._manager.get_departed_hotkeys()

    def get_departed_hotkey_info_rpc(self, hotkey: str) -> Optional[dict]:
        """Get departed info for a single hotkey (O(1) lookup)"""
        return self._manager.get_departed_hotkey_info(hotkey)

    def get_cached_elimination_data_rpc(self) -> tuple:
        """Get cached elimination data."""
        with self._cache_lock:
            return (dict(self._eliminations_cache), dict(self._departed_hotkeys_cache))

    def get_eliminations_lock_rpc(self):
        """This method should not be called via RPC - lock is local to server"""
        raise NotImplementedError(
            "get_eliminations_lock() is not available via RPC. "
            "Locking happens automatically on server side."
        )

    def process_eliminations_rpc(self, iteration_epoch=None) -> None:
        """Trigger elimination processing via RPC."""
        self._manager.process_eliminations(iteration_epoch=iteration_epoch)

    def handle_perf_ledger_eliminations_rpc(self, iteration_epoch=None) -> None:
        """Process performance ledger eliminations."""
        self._manager.handle_perf_ledger_eliminations(iteration_epoch=iteration_epoch)

    def get_first_refresh_ran_rpc(self) -> bool:
        """Get the first_refresh_ran flag."""
        return self._manager.first_refresh_ran

    def set_first_refresh_ran_rpc(self, value: bool) -> None:
        """Set the first_refresh_ran flag."""
        self._manager.first_refresh_ran = value

    def is_zombie_hotkey_rpc(self, hotkey: str, all_hotkeys_set: set) -> bool:
        """Check if hotkey is a zombie."""
        return self._manager.is_zombie_hotkey(hotkey, all_hotkeys_set)

    def handle_mdd_eliminations_rpc(self, iteration_epoch=None) -> None:
        """Check for MDD eliminations."""
        self._manager.handle_mdd_eliminations(iteration_epoch=iteration_epoch)

    def handle_eliminated_miner_rpc(self, hotkey: str,
                                    trade_pair_to_price_source_dict: dict = None,
                                    iteration_epoch=None) -> None:
        """Handle cleanup for eliminated miner (deletes limit orders, closes positions)."""
        # Convert dict to TradePair objects (RPC can't serialize TradePair directly)
        from vali_objects.vali_dataclasses.price_source import PriceSource
        trade_pair_to_price_source = {}
        if trade_pair_to_price_source_dict:
            for trade_pair_id, ps_dict in trade_pair_to_price_source_dict.items():
                trade_pair = TradePair.from_trade_pair_id(trade_pair_id)
                price_source = PriceSource(**ps_dict) if isinstance(ps_dict, dict) else ps_dict
                trade_pair_to_price_source[trade_pair] = price_source

        self._manager.handle_eliminated_miner(hotkey, trade_pair_to_price_source, iteration_epoch)

    def save_eliminations_rpc(self) -> None:
        """Save eliminations to disk."""
        self._manager.save_eliminations()

    def write_eliminations_to_disk_rpc(self, eliminations: list) -> None:
        """Write eliminations to disk."""
        self._manager.write_eliminations_to_disk(eliminations)

    def load_eliminations_from_disk_rpc(self) -> None:
        """Load eliminations from disk into memory (for testing recovery scenarios)."""
        self._manager._load_eliminations_from_disk()

    def refresh_allowed_rpc(self, interval_ms: int) -> bool:
        """Check if cache refresh is allowed based on time elapsed since last update."""
        return self._manager.refresh_allowed(interval_ms)

    def set_last_update_time_rpc(self) -> None:
        """Set the last update time to current time (for cache management)."""
        self._manager.set_last_update_time()

    def get_eliminations_dict_rpc(self) -> Dict[str, dict]:
        """Get eliminations dict (copy)."""
        return self._manager.get_eliminations_dict()

    def handle_first_refresh_rpc(self, iteration_epoch=None) -> None:
        """Handle first refresh on startup."""
        self._manager.handle_first_refresh(iteration_epoch)

    # start_daemon_rpc() inherited from RPCServerBase

    # ==================== Internal Methods ====================

    def get_eliminations_lock(self):
        """Get the local eliminations lock (server-side only)"""
        return self._manager.get_eliminations_lock()

    def generate_elimination_row(self, hotkey, current_dd, reason, t_ms=None, price_info=None, return_info=None):
        """Generate elimination row dict."""
        return self._manager.generate_elimination_row(hotkey, current_dd, reason, t_ms=t_ms,
                                                      price_info=price_info, return_info=return_info)

    def append_elimination_row(self, hotkey, current_dd, reason, t_ms=None, price_info=None, return_info=None):
        """Add elimination row"""
        self._manager.append_elimination_row(hotkey, current_dd, reason, t_ms=t_ms,
                                             price_info=price_info, return_info=return_info)

    def delete_eliminations(self, deleted_hotkeys):
        """Delete multiple eliminations"""
        self._manager.delete_eliminations(deleted_hotkeys)

    def save_eliminations(self):
        """Save eliminations to disk"""
        self._manager.save_eliminations()

    def write_eliminations_to_disk(self, eliminations):
        """Write eliminations to disk"""
        self._manager.write_eliminations_to_disk(eliminations)

    def get_eliminations_from_disk(self) -> list:
        """Load eliminations from disk"""
        return self._manager.get_eliminations_from_disk()

    def _load_eliminations_from_disk(self):
        """Load eliminations from disk into memory (for testing recovery scenarios)"""
        self._manager._load_eliminations_from_disk()

    def refresh_allowed(self, interval_ms: int) -> bool:
        """Check if cache refresh is allowed"""
        return self._manager.refresh_allowed(interval_ms)

    def set_last_update_time(self):
        """Set the last update time"""
        self._manager.set_last_update_time()

    @property
    def first_refresh_ran(self):
        """Direct access to first_refresh_ran flag (for tests)."""
        return self._manager.first_refresh_ran

    @first_refresh_ran.setter
    def first_refresh_ran(self, value: bool):
        """Direct access to set first_refresh_ran flag (for tests)."""
        self._manager.first_refresh_ran = value
