"""
Position Manager Server - RPC server for managing position data.

This server wraps PositionManager and exposes it via RPC.

Architecture:
- PositionManagerServer inherits from RPCServerBase for RPC infrastructure
- Creates PositionManager instance (self._manager) with all business logic
- All RPC methods delegate to self._manager
- Follows the PerfLedgerServer/Manager pattern

Usage:
    # Server (typically started by validator)
    server = PositionManagerServer(
        start_server=True,
        start_daemon=True  # Enable compaction daemon
    )

    # Client (can be created in any process)
    from vali_objects.utils.position_manager_client import PositionManagerClient
    client = PositionManagerClient()
    positions = client.get_positions_for_one_hotkey(hotkey)
"""
import time
import bittensor as bt
import traceback
from typing import List, Dict, Optional

from shared_objects.rpc.rpc_server_base import RPCServerBase
from time_util.time_util import MS_IN_24_HOURS, S_IN_24_HOURS, timeme
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class PositionManagerServer(RPCServerBase):
    """
    Server process that manages position data via RPC.

    Inherits from RPCServerBase for unified RPC server and daemon infrastructure.
    The daemon periodically compacts price sources from old closed positions.

    Architecture:
    - Creates PositionManager instance (self._manager) with all business logic
    - All RPC methods delegate to self._manager
    - Follows the PerfLedgerServer/Manager pattern
    """
    service_name = ValiConfig.RPC_POSITIONMANAGER_SERVICE_NAME
    service_port = ValiConfig.RPC_POSITIONMANAGER_PORT

    def __init__(
        self,
        running_unit_tests: bool = False,
        is_backtesting: bool = False,
        slack_notifier=None,
        load_from_disk: bool = None,
        split_positions_on_disk_load: bool = False,
        start_server: bool = True,
        start_daemon: bool = False,
        connection_mode = RPCConnectionMode.RPC
    ):
        """
        Initialize the PositionManagerServer.

        Args:
            running_unit_tests: Whether running in unit test mode
            is_backtesting: Whether running in backtesting mode
            slack_notifier: Optional SlackNotifier for alerts
            load_from_disk: Override disk loading behavior (None=auto, True=force load, False=skip)
            split_positions_on_disk_load: Whether to apply position splitting after loading from disk
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start compaction daemon
        """
        # Create the actual PositionManager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        from vali_objects.position_management.position_manager import PositionManager
        self._manager = PositionManager(
            running_unit_tests=running_unit_tests,
            is_backtesting=is_backtesting,
            load_from_disk=load_from_disk,
            split_positions_on_disk_load=split_positions_on_disk_load,
            connection_mode=connection_mode
        )

        bt.logging.success("PositionManager initialized")

        self._last_compact_time_s: float = 0.0  # Track last compact_price_sources run time

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        # daemon_interval_s: 1 hour (frequent carry fee charging; compact is separately rate-limited)
        # hang_timeout_s: Dynamically set to 2x interval to prevent false alarms during normal sleep
        daemon_interval_s = 3600 # 1 hour
        hang_timeout_s = daemon_interval_s * 2.0  # 2 hours (2x interval)

        super().__init__(
            service_name=ValiConfig.RPC_POSITIONMANAGER_SERVICE_NAME,
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connection_mode=connection_mode,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s
        )

        bt.logging.success("PositionManagerServer initialized")

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Daemon iteration that:
        1. Compacts price sources from old closed positions (guarded: at most every 12 hours)
        2. Charges carry fees on all open positions

        Runs every 1 hour. Compact price sources is rate-limited via
        ValiConfig.PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS to avoid expensive
        disk I/O on every iteration.
        Delegates to manager for direct memory access - no RPC overhead!
        """
        now = time.time()
        if now - self._last_compact_time_s >= ValiConfig.PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS:
            try:
                t0 = time.time()
                self._manager.compact_price_sources()
                bt.logging.info(f'Compacted price sources in {time.time() - t0:.2f} seconds')
                self._last_compact_time_s = now
            except Exception as e:
                bt.logging.error(f"Error in compaction daemon iteration: {traceback.format_exc()}")

        try:
            self._manager.settle_dividend_payments()
        except Exception as e:
            bt.logging.error(f"Error settling dividend payments: {traceback.format_exc()}")

        try:
            self._manager.refresh_position_fees()
        except Exception as e:
            bt.logging.error(f"Error in carry fee daemon iteration: {traceback.format_exc()}")

        # Align next daemon iteration to UTC hour boundary
        now = time.time()
        next_hour_s = (int(now) // 3600 + 1) * 3600
        self.daemon_interval_s = next_hour_s - now
        bt.logging.info(f"PositionManager daemon interval complete, next iteration in {self.daemon_interval_s} seconds")


    # ==================== RPC Methods (called by client via RPC) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return self._manager.health_check()

    def get_positions_for_one_hotkey_rpc(
        self,
        hotkey: str,
        only_open_positions=False,
        acceptable_position_end_ms=None,
        sort_positions=False,
        archived_positions=False
    ):
        """Get positions for a specific hotkey - delegates to manager."""
        return self._manager.get_positions_for_one_hotkey(
            hotkey, only_open_positions, acceptable_position_end_ms, sort_positions,
            archived_positions=archived_positions
        )

    def save_miner_position_rpc(self, position: Position, delete_open_position_if_exists: bool = True):
        """Save a position - delegates to manager."""
        self._manager.save_miner_position(position, delete_open_position_if_exists)

    def get_positions_for_hotkeys_rpc(
        self,
        hotkeys: List[str],
        only_open_positions=False,
        filter_eliminations: bool = False,
        acceptable_position_end_ms: int = None,
        sort_positions: bool = False,
        archived_positions: bool = False
    ) -> Dict[str, List[Position]]:
        """Get positions for multiple hotkeys - delegates to manager."""
        return self._manager.get_positions_for_hotkeys(
            hotkeys, only_open_positions, filter_eliminations, acceptable_position_end_ms, sort_positions,
            archived_positions=archived_positions
        )

    def clear_all_miner_positions_rpc(self):
        """Clear all positions from memory - delegates to manager."""
        self._manager.clear_all_miner_positions()

    def clear_all_miner_positions_and_disk_rpc(self, hotkey=None):
        """Clear all positions from memory AND disk - delegates to manager."""
        self._manager.clear_all_miner_positions_and_disk(hotkey=hotkey)

    def delete_position_rpc(self, hotkey: str, position_uuid: str):
        """Delete a specific position - delegates to manager."""
        return self._manager.delete_position(hotkey, position_uuid)

    def get_position_rpc(self, hotkey: str, position_uuid: str):
        """Get a specific position by UUID - delegates to manager."""
        return self._manager.get_position(hotkey, position_uuid)

    def get_dashboard_rpc(self, hotkey: str, positions_time_ms: int) -> dict | None:
        return self._manager.get_dashboard(hotkey, positions_time_ms)

    def get_open_position_for_trade_pair_rpc(self, hotkey: str, trade_pair_id: str) -> Optional[Position]:
        """Get open position for trade pair - delegates to manager."""
        return self._manager.get_open_position_for_trade_pair(hotkey, trade_pair_id)

    def get_all_hotkeys_rpc(self):
        """Get all hotkeys that have positions - delegates to manager."""
        return self._manager.get_all_hotkeys()

    def get_hotkey_to_archived_positions_rpc(self):
        """Get hotkey -> {uuid -> Position} dict for all archived positions."""
        return self._manager.get_hotkey_to_archived_positions()

    def get_extreme_position_order_processed_on_disk_ms_rpc(self):
        """
        Get the minimum and maximum processed_ms timestamps across all orders in all positions.
        Delegates to manager.

        Returns:
            tuple: (min_time, max_time) in milliseconds
        """
        return self._manager.get_extreme_position_order_processed_on_disk_ms()

    def calculate_net_portfolio_leverage_rpc(self, hotkey: str) -> float:
        """Calculate portfolio leverage - delegates to manager."""
        return self._manager.calculate_net_portfolio_leverage(hotkey)

    def compute_realtime_drawdown_rpc(self, hotkey: str) -> float:
        """Compute realtime drawdown - delegates to manager."""
        return self._manager.compute_realtime_drawdown(hotkey)

    def get_unrealized_pnl_rpc(self, hotkey: str) -> float:
        """Get total unrealized PnL across all open positions - delegates to manager."""
        return self._manager.get_unrealized_pnl(hotkey)

    def filtered_positions_for_scoring_rpc(
        self,
        hotkeys: List[str] = None,
        include_development_positions: bool = False
    ) -> tuple:
        """Filter positions for scoring - delegates to manager."""
        return self._manager.filtered_positions_for_scoring(hotkeys, include_development_positions)

    def close_open_orders_for_suspended_trade_pairs_rpc(self, live_price_fetcher=None) -> int:
        """Close positions for suspended trade pairs - delegates to manager."""
        return self._manager.close_open_orders_for_suspended_trade_pairs(live_price_fetcher)

    def close_all_positions_rpc(
        self,
        hotkey: str,
        close_time_ms: int,
        order_source: OrderSource,
        live_price_fetcher=None
    ) -> int:
        """
        RPC wrapper for close_all_positions.

        Args:
            hotkey: Hotkey whose positions should be closed
            close_time_ms: Timestamp for closing positions
            order_source: OrderSource enum value (as int)
            live_price_fetcher: Optional price fetcher

        Returns:
            int: Number of positions closed
        """
        return self._manager.close_all_positions(
            hotkey=hotkey,
            close_time_ms=close_time_ms,
            order_source=order_source,
            live_price_fetcher=live_price_fetcher
        )

    # ==================== Pre-run Setup RPC Methods ====================

    @timeme
    def pre_run_setup_rpc(self, perform_order_corrections: bool = True) -> None:
        """Run pre-run setup operations - delegates to manager."""
        self._manager.pre_run_setup(perform_order_corrections)

    # ==================== Position Splitting RPC Methods ====================

    def split_position_on_flat_rpc(self, position: Position, track_stats: bool = False) -> tuple[list[Position], dict]:
        """
        Split a position on FLAT orders or implicit flats - delegates to manager.

        Args:
            position: The position to split
            track_stats: Whether to track splitting statistics for this miner

        Returns:
            Tuple of (list of split positions, split_info dict)
        """
        return self._manager.split_position_on_flat(position, track_stats)

    def get_split_stats_rpc(self, hotkey: str) -> dict:
        """
        Get position splitting statistics for a miner - delegates to manager.

        Args:
            hotkey: The miner hotkey

        Returns:
            Dict with splitting statistics
        """
        return self._manager.get_split_stats(hotkey)

    def apply_stock_split_rpc(self, trade_pair_id: str, stock_split_ratio: float, execution_date: str):
        return self._manager.apply_stock_split(trade_pair_id, stock_split_ratio, execution_date)

    def process_dividend_ex_date(self, trade_pair_id: str, gross_dividend: float,
                                   payment_date_str: str, ex_date_str: str):
        return self._manager.process_dividend_ex_date(trade_pair_id, gross_dividend, payment_date_str, ex_date_str)

    # ==================== Bracket Order Attachment RPC Methods ====================

    def attach_bracket_order_to_position_rpc(self, miner_hotkey: str, trade_pair_id: str, order_dict: dict) -> bool:
        """
        Attach a bracket order to a position - delegates to manager.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_id: The trade pair ID
            order_dict: The order as a dictionary

        Returns:
            True if successfully attached, False if no open position found
        """
        return self._manager.attach_bracket_order_to_position(miner_hotkey, trade_pair_id, order_dict)

    def archive_positions_for_hotkey_rpc(
        self,
        hotkey: str,
        positions: list = None,
        archive_all: bool = False
    ) -> int:
        """Archive positions to archived_positions/ dir - delegates to manager."""
        return self._manager.archive_positions_for_hotkey(
            hotkey, positions=positions, archive_all=archive_all
        )

    def remove_bracket_order_from_position_rpc(self, miner_hotkey: str, trade_pair_id: str, order_uuid: str) -> bool:
        """
        Remove a bracket order from a position - delegates to manager.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_id: The trade pair ID
            order_uuid: The UUID of the order to remove

        Returns:
            True if found and removed, False otherwise
        """
        return self._manager.remove_bracket_order_from_position(miner_hotkey, trade_pair_id, order_uuid)
