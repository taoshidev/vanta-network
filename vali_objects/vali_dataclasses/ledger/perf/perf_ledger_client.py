from typing import List

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger


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

    def get_returns(self, hotkey: str) -> float | None:
        """
        Get returns for a specific hotkey's portfolio.

        Args:
            hotkey: Miner hotkey

        Returns:
            Returns as float (e.g., 0.08 for 8%), or None if no data exists
        """
        return self._server.get_returns_rpc(hotkey)

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

    def get_bypass_values_if_applicable(
        self,
        ledger: PerfLedger,
        trade_pair: str,
        tp_status: str,
        tp_return: float,
        spread_fee_pct: float,
        carry_fee_pct: float,
        active_positions: dict
    ) -> tuple:
        """
        Test-only method to get bypass values if applicable.

        Args:
            ledger: PerfLedger instance
            trade_pair: Trade pair identifier
            tp_status: TradePairReturnStatus value
            tp_return: Trade pair return value
            spread_fee_pct: Spread fee percentage
            carry_fee_pct: Carry fee percentage
            active_positions: Dict of active positions

        Returns:
            Tuple of (return, spread_fee, carry_fee)
        """
        return self._server.get_bypass_values_if_applicable_rpc(
            ledger, trade_pair, tp_status, tp_return, spread_fee_pct, carry_fee_pct, active_positions
        )

    def health_check(self) -> dict:
        """Check server health."""
        return self._server.health_check_rpc()

    def update(self, testing_one_hotkey=None, regenerate_all_ledgers=False, t_ms=None):
        return self._server.update_rpc(testing_one_hotkey=testing_one_hotkey, regenerate_all_ledgers=regenerate_all_ledgers, t_ms=t_ms)
