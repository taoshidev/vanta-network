# developer: jbonilla
# Copyright 2024 Taoshi Inc
"""
PerfLedgerServer - RPC server for performance ledger management.

This server manages performance ledgers and exposes them via RPC.
Consumers create their own PerfLedgerClient to connect.
The server creates its own MetagraphClient internally (forward compatibility pattern).

Usage:
    # In validator.py
    perf_ledger_server = PerfLedgerServer(
        start_server=True,
        start_daemon=True
    )

    # In any consumer
    client = PerfLedgerClient()
    ledgers = client.get_perf_ledgers()
"""
import bittensor as bt
from typing import List
from shared_objects.common_data_server import CommonDataClient

from shared_objects.rpc_server_base import RPCServerBase
from shared_objects.rpc_client_base import RPCClientBase
from shared_objects.sn8_multiprocessing import ParallelizationMode
from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfLedger

class PerfLedgerClient(RPCClientBase):
    """
    Lightweight RPC client for PerfLedgerServer.

    Can be created in ANY process. No server ownership.
    Forward compatibility - consumers create their own client instance.

    Example:
        client = PerfLedgerClient()
        ledgers = client.get_perf_ledgers(portfolio_only=True)
    """

    def __init__(
        self,
        port: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        connect_immediately: bool = False,
        running_unit_tests: bool = False
    ):
        """
        Initialize PerfLedger client.

        Args:
            port: Port number of the PerfLedger server (default: ValiConfig.RPC_PERFLEDGER_PORT)
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - bypass RPC, use set_direct_server()
                - RPC (1): Normal RPC mode - connect via network
            connect_immediately: If True, connect in __init__. If False, call connect() later.
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_PERFLEDGER_SERVICE_NAME,
            port=port or ValiConfig.RPC_PERFLEDGER_PORT,
            max_retries=60,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Query Methods ====================

    def get_perf_ledgers(self, portfolio_only: bool = True, from_disk: bool = False) -> dict:
        """
        Get performance ledgers.

        Args:
            portfolio_only: If True, only return portfolio ledgers
            from_disk: If True, read from disk instead of memory

        Returns:
            Dict mapping hotkey to performance ledger(s)
        """
        # PerfLedger objects returned directly - BaseManager's pickle handles serialization
        return self._server.get_perf_ledgers_rpc(portfolio_only=portfolio_only, from_disk=from_disk)

    def generate_perf_ledgers_for_analysis(self, hotkey_to_positions, t_ms: int = None) -> dict:
        """Generate performance ledgers for analysis."""
        return self._server.generate_perf_ledgers_for_analysis_rpc(hotkey_to_positions, t_ms=t_ms)

    def filtered_ledger_for_scoring(
        self,
        portfolio_only: bool = False,
        hotkeys: List[str] = None
    ) -> dict[str, dict[str, PerfLedger]] | dict[str, PerfLedger]:
        """
        Get filtered ledger for scoring.

        Args:
            portfolio_only: If True, only return portfolio ledgers
            hotkeys: Optional list of hotkeys to filter

        Returns:
            Dict mapping hotkey to filtered performance ledger
        """
        # PerfLedger objects returned directly - BaseManager's pickle handles serialization
        return self._server.filtered_ledger_for_scoring_rpc(
            portfolio_only=portfolio_only,
            hotkeys=hotkeys
        )

    def get_perf_ledger_eliminations(self, first_fetch: bool = False) -> list:
        """
        Get performance ledger eliminations.

        Args:
            first_fetch: If True, load from disk instead of memory

        Returns:
            List of elimination dictionaries
        """
        return self._server.get_perf_ledger_eliminations_rpc(first_fetch=first_fetch)

    def write_perf_ledger_eliminations_to_disk(self, eliminations: list) -> None:
        """
        Write performance ledger eliminations to disk.

        Args:
            eliminations: List of elimination dictionaries to write
        """
        self._server.write_perf_ledger_eliminations_to_disk_rpc(eliminations)

    def clear_perf_ledger_eliminations(self) -> None:
        """Clear all perf ledger eliminations in memory (for testing)."""
        self._server.clear_perf_ledger_eliminations_rpc()

    def save_perf_ledgers(self, perf_ledgers: dict) -> None:
        """
        Save performance ledgers.

        Args:
            perf_ledgers: Dict mapping hotkey to performance ledger bundle
        """
        self._server.save_perf_ledgers_rpc(perf_ledgers)

    def wipe_miners_perf_ledgers(self, miners_to_wipe: List[str]) -> None:
        """
        Wipe performance ledgers for specified miners.

        Args:
            miners_to_wipe: List of miner hotkeys to wipe
        """
        self._server.wipe_miners_perf_ledgers_rpc(miners_to_wipe)

    def get_hotkey_to_perf_bundle(self) -> dict:
        """Get the in-memory hotkey to perf bundle dict."""
        # PerfLedger objects returned directly - BaseManager's pickle handles serialization
        return self._server.get_hotkey_to_perf_bundle_rpc()

    def get_perf_ledger_for_hotkey(self, hotkey: str) -> dict | None:
        """
        Get performance ledger for a specific hotkey.

        Args:
            hotkey: Miner hotkey

        Returns:
            Dict containing perf ledger bundle for the hotkey, or None if not found
        """
        return self._server.get_perf_ledger_for_hotkey_rpc(hotkey)

    def set_hotkey_perf_bundle(self, hotkey: str, bundle: dict) -> None:
        """Set perf bundle for a specific hotkey."""
        self._server.set_hotkey_perf_bundle_rpc(hotkey, bundle)

    def delete_hotkey_perf_bundle(self, hotkey: str) -> bool:
        """Delete perf bundle for a specific hotkey."""
        return self._server.delete_hotkey_perf_bundle_rpc(hotkey)

    def clear_all_ledger_data(self) -> None:
        """Clear all ledger data (unit tests only)."""
        self._server.clear_all_ledger_data_rpc()

    def re_init_perf_ledger_data(self) -> None:
        """Reinitialize perf ledger data by reloading from disk (unit tests only)."""
        self._server.re_init_perf_ledger_data_rpc()

    def get_perf_ledger_hks_to_invalidate(self) -> dict:
        """Get hotkeys to invalidate."""
        return self._server.get_perf_ledger_hks_to_invalidate_rpc()

    def set_perf_ledger_hks_to_invalidate(self, hks_to_invalidate: dict) -> None:
        """Set hotkeys to invalidate."""
        self._server.set_perf_ledger_hks_to_invalidate_rpc(hks_to_invalidate)

    def clear_perf_ledger_hks_to_invalidate(self) -> None:
        """Clear all hotkeys to invalidate."""
        self._server.clear_perf_ledger_hks_to_invalidate_rpc()

    def set_hotkey_to_invalidate(self, hotkey: str, timestamp_ms: int) -> None:
        """
        Set a single hotkey to invalidate.

        Args:
            hotkey: Hotkey to mark for invalidation
            timestamp_ms: Timestamp from which to invalidate (0 means invalidate all)
        """
        self._server.set_hotkey_to_invalidate_rpc(hotkey, timestamp_ms)

    def update_hotkey_to_invalidate(self, hotkey: str, timestamp_ms: int) -> None:
        """
        Update a hotkey's invalidation timestamp (uses min of existing and new).

        Args:
            hotkey: Hotkey to mark for invalidation
            timestamp_ms: Timestamp from which to invalidate
        """
        self._server.update_hotkey_to_invalidate_rpc(hotkey, timestamp_ms)

    def set_invalidation(self, hotkey: str, invalidate: bool) -> None:
        """
        Convenience method to invalidate or clear invalidation for a hotkey.

        Args:
            hotkey: Hotkey to mark for invalidation or clear
            invalidate: True to invalidate from timestamp 0 (all checkpoints), False to clear
        """
        if invalidate:
            # Invalidate from timestamp 0 (invalidate all checkpoints)
            self._server.set_hotkey_to_invalidate_rpc(hotkey, 0)
        else:
            # Clear invalidation by removing from dict
            hks_to_invalidate = self._server.get_perf_ledger_hks_to_invalidate_rpc()
            if hotkey in hks_to_invalidate:
                del hks_to_invalidate[hotkey]
                self._server.set_perf_ledger_hks_to_invalidate_rpc(hks_to_invalidate)

    def add_elimination_row(self, elimination_row: dict) -> None:
        """
        Add an elimination row to the perf ledger eliminations.

        This is used by tests to simulate performance ledger eliminations.

        Args:
            elimination_row: Elimination dict with hotkey, reason, dd, etc.
        """
        self._server.add_elimination_row_rpc(elimination_row)

    def health_check(self) -> dict:
        """Check server health."""
        return self._server.health_check_rpc()

    def update(self, t_ms=None):
        return self._server.update_rpc(t_ms=t_ms)


class PerfLedgerServer(RPCServerBase):
    """
    RPC server for performance ledger management.

    Wraps PerfLedgerManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to clients.
    """
    service_name = ValiConfig.RPC_PERFLEDGER_SERVICE_NAME
    service_port = ValiConfig.RPC_PERFLEDGER_PORT

    def __init__(
        self,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        is_backtesting: bool = False,
        parallel_mode: ParallelizationMode = ParallelizationMode.SERIAL
    ):
        """
        Initialize PerfLedgerServer.

        The server manages its own perf_ledger_hks_to_invalidate dict internally.
        Consumers use PerfLedgerClient to update invalidations via RPC.

        Args:
            slack_notifier: Slack notifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
            is_backtesting: Whether running in backtesting mode
        """

        # Create own CommonDataClient
        # Provides access to sync_in_progress, sync_epoch
        self.running_unit_tests = running_unit_tests
        self._common_data_client = CommonDataClient(
            connect_immediately=False,  # Lazy connect on first use
            connection_mode=connection_mode
        )

        # Create the actual PerfLedgerManager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        # Note: PerfLedgerManager will create its own perf_ledger_hks_to_invalidate dict internally.
        # Consumers use PerfLedgerClient to update invalidations via RPC.
        self._manager = PerfLedgerManager(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests,
            enable_rss=not (running_unit_tests or is_backtesting),
            is_backtesting=is_backtesting,
            parallel_mode=parallel_mode
        )

        bt.logging.info(f"[PERFLEDGER_SERVER] PerfLedgerManager initialized")

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        # daemon_interval_s: 5 minutes (perf ledger update frequency)
        # hang_timeout_s: 10 minutes (first iteration can take 5+ min processing large datasets)
        super().__init__(
            service_name=ValiConfig.RPC_PERFLEDGER_SERVICE_NAME,
            port=ValiConfig.RPC_PERFLEDGER_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=ValiConfig.PERF_LEDGER_REFRESH_TIME_MS / 1000.0,
            hang_timeout_s=1800,  # 30 minutes (heavy hotkey?)
            connection_mode=connection_mode
        )

        # Start daemon if requested (not in LOCAL mode)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """Single iteration of daemon work - delegates to manager's update loop logic."""
        if self.sync_in_progress:
            bt.logging.debug("[PERF_LEDGER_DAEMON] Sync in progress, pausing...")
            return

        if self._manager.refresh_allowed(ValiConfig.PERF_LEDGER_REFRESH_TIME_MS):
            bt.logging.info("[PERF_LEDGER_DAEMON] Starting perf ledger update...")
            self._manager.update()
            self._manager.set_last_update_time(skip_message=False)  # Enable logging to confirm updates
            bt.logging.success("[PERF_LEDGER_DAEMON] Perf ledger update completed")
        else:
            # Log when refresh is not allowed (helps diagnose silent daemon)
            time_since_last_update_ms = TimeUtil.now_in_millis() - self._manager.get_last_update_time_ms()
            time_until_next_update_ms = ValiConfig.PERF_LEDGER_REFRESH_TIME_MS - time_since_last_update_ms
            bt.logging.debug(
                f"[PERF_LEDGER_DAEMON] Refresh not allowed yet "
                f"(next update in {time_until_next_update_ms/1000:.1f}s)"
            )

    @property
    def sync_in_progress(self):
        """Get sync_in_progress flag via CommonDataClient."""
        return self._common_data_client.get_sync_in_progress()


    @property
    def sync_epoch(self):
        """Get sync_epoch value via CommonDataClient."""
        return self._common_data_client.get_sync_epoch()

    # ==================== RPC Methods (exposed to clients) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "num_ledgers": len(self._manager.hotkey_to_perf_bundle),
            "num_eliminations": len(self._manager.pl_elimination_rows)
        }

    def update_rpc(self, t_ms=None) -> dict:
        return self._manager.update(t_ms=t_ms)

    def get_perf_ledgers_rpc(self, portfolio_only: bool = True, from_disk: bool = False) -> dict:
        """Get performance ledgers via RPC."""
        # Return PerfLedger objects directly - BaseManager's pickle handles serialization
        return self._manager.get_perf_ledgers(portfolio_only=portfolio_only, from_disk=from_disk)

    def filtered_ledger_for_scoring_rpc(
        self,
        portfolio_only: bool = False,
        hotkeys: List[str] = None
    ) -> dict[str, dict[str, PerfLedger]] | dict[str, PerfLedger]:
        """Get filtered ledger for scoring via RPC."""
        # Return PerfLedger objects directly - BaseManager's pickle handles serialization
        return self._manager.filtered_ledger_for_scoring(
            portfolio_only=portfolio_only,
            hotkeys=hotkeys
        )

    def get_perf_ledger_eliminations_rpc(self, first_fetch: bool = False) -> list:
        """
        Get performance ledger eliminations via RPC.

        Args:
            first_fetch: If True, load from disk instead of memory

        Returns:
            List of elimination dictionaries
        """
        return list(self._manager.get_perf_ledger_eliminations(first_fetch=first_fetch))

    def write_perf_ledger_eliminations_to_disk_rpc(self, eliminations: list) -> None:
        """
        Write performance ledger eliminations to disk via RPC.

        Args:
            eliminations: List of elimination dictionaries to write
        """
        self._manager.write_perf_ledger_eliminations_to_disk(eliminations)

    def clear_perf_ledger_eliminations_rpc(self) -> None:
        """Clear all perf ledger eliminations in memory via RPC (for testing)."""
        self._manager.pl_elimination_rows.clear()

    def save_perf_ledgers_rpc(self, perf_ledgers: dict) -> None:
        """Save performance ledgers via RPC."""
        # Accept PerfLedger objects directly - BaseManager's pickle handles serialization
        self._manager.save_perf_ledgers(perf_ledgers)

    def wipe_miners_perf_ledgers_rpc(self, miners_to_wipe: List[str]) -> None:
        """
        Wipe performance ledgers for specified miners.

        This is called during pre_run_setup when order corrections reset miners.
        """
        if not miners_to_wipe:
            return

        bt.logging.info(f'[PERFLEDGER_SERVER] Wiping perf ledgers for {len(miners_to_wipe)} miners')

        # Get current ledgers
        perf_ledgers = self._manager.get_perf_ledgers(portfolio_only=False)
        n_before = len(perf_ledgers)

        # Filter out miners to wipe
        perf_ledgers_new = {k: v for k, v in perf_ledgers.items() if k not in miners_to_wipe}
        n_after = len(perf_ledgers_new)

        bt.logging.info(f'[PERFLEDGER_SERVER] Wiped perf ledgers: {n_before} -> {n_after}')

        # Save filtered ledgers
        self._manager.save_perf_ledgers(perf_ledgers_new)

        # Also update in-memory state
        for hotkey in miners_to_wipe:
            if hotkey in self._manager.hotkey_to_perf_bundle:
                del self._manager.hotkey_to_perf_bundle[hotkey]

    def get_hotkey_to_perf_bundle_rpc(self) -> dict:
        """Get the in-memory hotkey to perf bundle dict via RPC."""
        # Return PerfLedger objects directly - BaseManager's pickle handles serialization
        return dict(self._manager.hotkey_to_perf_bundle)

    def get_perf_ledger_for_hotkey_rpc(self, hotkey: str) -> dict | None:
        """
        Get performance ledger for a specific hotkey via RPC.

        Args:
            hotkey: Miner hotkey

        Returns:
            Dict containing perf ledger bundle for the hotkey, or None if not found
        """
        if hotkey in self._manager.hotkey_to_perf_bundle:
            # Return PerfLedger objects directly - BaseManager's pickle handles serialization
            return {hotkey: self._manager.hotkey_to_perf_bundle[hotkey]}
        return None

    def set_hotkey_perf_bundle_rpc(self, hotkey: str, bundle: dict) -> None:
        """Set perf bundle for a specific hotkey via RPC."""
        # Accept PerfLedger objects directly - BaseManager's pickle handles serialization
        self._manager.hotkey_to_perf_bundle[hotkey] = bundle

    def delete_hotkey_perf_bundle_rpc(self, hotkey: str) -> bool:
        """Delete perf bundle for a specific hotkey via RPC."""
        if hotkey in self._manager.hotkey_to_perf_bundle:
            del self._manager.hotkey_to_perf_bundle[hotkey]
            return True
        return False

    def generate_perf_ledgers_for_analysis_rpc(self, hotkey_to_positions: dict[str, List[Position]], t_ms: int = None) -> dict[str, dict[str, PerfLedger]]:
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()  # Time to build the perf ledgers up to. Goes back 30 days from this time.
        existing_perf_ledgers = {}
        return self._manager.update_all_perf_ledgers(hotkey_to_positions, existing_perf_ledgers, t_ms)

    def clear_all_ledger_data_rpc(self) -> None:
        """Clear all ledger data via RPC (unit tests only)."""
        self._manager.clear_all_ledger_data()

    def re_init_perf_ledger_data_rpc(self) -> None:
        """Reinitialize perf ledger data via RPC (unit tests only)."""
        self._manager.re_init_perf_ledger_data()

    def get_perf_ledger_hks_to_invalidate_rpc(self) -> dict:
        """Get hotkeys to invalidate via RPC."""
        return dict(self._manager.perf_ledger_hks_to_invalidate)

    def set_perf_ledger_hks_to_invalidate_rpc(self, hks_to_invalidate: dict) -> None:
        """Set hotkeys to invalidate via RPC."""
        self._manager.perf_ledger_hks_to_invalidate.clear()
        self._manager.perf_ledger_hks_to_invalidate.update(hks_to_invalidate)

    def clear_perf_ledger_hks_to_invalidate_rpc(self) -> None:
        """Clear all hotkeys to invalidate via RPC."""
        self._manager.perf_ledger_hks_to_invalidate.clear()

    def set_hotkey_to_invalidate_rpc(self, hotkey: str, timestamp_ms: int) -> None:
        """
        Set a single hotkey to invalidate via RPC.

        Args:
            hotkey: Hotkey to mark for invalidation
            timestamp_ms: Timestamp from which to invalidate (0 means invalidate all)
        """
        self._manager.perf_ledger_hks_to_invalidate[hotkey] = timestamp_ms

    def update_hotkey_to_invalidate_rpc(self, hotkey: str, timestamp_ms: int) -> None:
        """
        Update a hotkey's invalidation timestamp via RPC (uses min of existing and new).

        This method sets the timestamp to the minimum of the existing timestamp (if any)
        and the new timestamp. This ensures we invalidate from the earliest point of change.

        Args:
            hotkey: Hotkey to mark for invalidation
            timestamp_ms: Timestamp from which to invalidate
        """
        if hotkey in self._manager.perf_ledger_hks_to_invalidate:
            self._manager.perf_ledger_hks_to_invalidate[hotkey] = min(
                self._manager.perf_ledger_hks_to_invalidate[hotkey],
                timestamp_ms
            )
        else:
            self._manager.perf_ledger_hks_to_invalidate[hotkey] = timestamp_ms

    def add_elimination_row_rpc(self, elimination_row: dict) -> None:
        """
        Add an elimination row to the perf ledger eliminations via RPC.

        This is used by tests to simulate performance ledger eliminations.

        Args:
            elimination_row: Elimination dict with hotkey, reason, dd, etc.
        """
        self._manager.pl_elimination_rows.append(elimination_row)

    # ==================== Direct Access (for backward compatibility in tests) ====================

    @property
    def perf_ledger_hks_to_invalidate(self):
        """Direct access to invalidation dict (for tests)."""
        return self._manager.perf_ledger_hks_to_invalidate

    @property
    def pl_elimination_rows(self):
        """Direct access to elimination rows (for tests)."""
        return self._manager.pl_elimination_rows

    @property
    def hotkey_to_perf_bundle(self):
        """Direct access to hotkey to perf bundle dict (for tests)."""
        return self._manager.hotkey_to_perf_bundle

