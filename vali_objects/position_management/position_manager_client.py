# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
PositionManagerClient - Lightweight RPC client for position management.

This client can be created in ANY process to connect to the PositionManagerServer.
No server ownership, no pickle complexity.

Usage:
    # In any process that needs position data
    client = PositionManagerClient(port=50002)

    positions = client.get_positions_for_one_hotkey(hotkey)

For child processes:
    # Parent passes port number (not manager object!)
    Process(target=child_func, args=(position_manager_port,))

    # Child creates its own client
    def child_func(position_manager_port):
        client = PositionManagerClient(port=position_manager_port)
        client.get_positions_for_one_hotkey(hotkey)
"""
import json
import math
from typing import Dict, List, Optional

from shared_objects.rpc.rpc_client_base import RPCClientBase
from time_util.time_util import TimeUtil
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.vali_dataclasses.position import Position
from vali_objects.position_management.position_utils.position_filtering import PositionFiltering
from vali_objects.position_management.position_manager import PositionManager
from vali_objects.vali_config import TradePair, ValiConfig, RPCConnectionMode


class PositionManagerClient(RPCClientBase):
    """
    Lightweight RPC client for PositionManagerServer.

    Can be created in ANY process. No server ownership.
    No pickle complexity - just pass the port to child processes.
    """

    def __init__(
        self,
        port: int = None,
        connect_immediately: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False
    ):
        """
        Initialize position manager client.

        Args:
            port: Port number of the position server (default: ValiConfig.RPC_POSITIONMANAGER_PORT)
            connect_immediately: If True, connect in __init__. If False, call connect() later.
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - bypass RPC, use set_direct_server()
                - RPC (1): Normal RPC mode - connect via network
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_POSITIONMANAGER_SERVICE_NAME,
            port=port or ValiConfig.RPC_POSITIONMANAGER_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Query Methods ====================

    @staticmethod
    def positions_to_dashboard_dict(original_positions: list[Position], time_now_ms) -> dict:
        ans = {
            "positions": [],
            "thirty_day_returns": 1.0,
            "all_time_returns": 1.0,
            "n_positions": 0,
            "percentage_profitable": 0.0
        }
        acceptable_position_end_ms = TimeUtil.timestamp_to_millis(
            TimeUtil.generate_start_timestamp(
                ValiConfig.SET_WEIGHT_LOOKBACK_RANGE_DAYS
            ))
        positions_30_days = [
            position
            for position in original_positions
            if position.open_ms > acceptable_position_end_ms
        ]
        ps_30_days = PositionFiltering.filter_positions_for_duration(positions_30_days)
        return_per_position = PositionManagerClient.get_return_per_closed_position(ps_30_days)
        if len(return_per_position) > 0:
            curr_return = return_per_position[len(return_per_position) - 1]
            ans["thirty_day_returns"] = curr_return

        ps_all_time = PositionFiltering.filter_positions_for_duration(original_positions)
        return_per_position = PositionManagerClient.get_return_per_closed_position(ps_all_time)
        if len(return_per_position) > 0:
            curr_return = return_per_position[len(return_per_position) - 1]
            ans["all_time_returns"] = curr_return
            ans["n_positions"] = len(ps_all_time)
            ans["percentage_profitable"] = PositionManagerClient.get_percent_profitable_positions(ps_all_time)

        for p in original_positions:
            # Don't modify the position object in-place
            # Instead, create the dict representation and modify only the dict
            PositionManager.strip_old_price_sources(p, time_now_ms)

            position_dict = json.loads(str(p), cls=GeneralizedJSONDecoder)
            # Convert None to 0 for JSON serialization (avoids null in JSON)
            # This is safe because we're only modifying the dict, not the position object
            if position_dict.get('close_ms') is None:
                position_dict['close_ms'] = 0

            # Add unfilled_orders for API response (excluded from disk serialization)
            if p.unfilled_orders:
                position_dict['unfilled_orders'] = p.unfilled_orders

            ans["positions"].append(position_dict)
        return ans

    @staticmethod
    def get_percent_profitable_positions(positions: List[Position]) -> float:
        if len(positions) == 0:
            return 0.0

        profitable_positions = 0
        n_closed_positions = 0

        for position in positions:
            if position.is_open_position:
                continue

            n_closed_positions += 1
            if position.return_at_close > 1.0:
                profitable_positions += 1

        if n_closed_positions == 0:
            return 0.0

        return profitable_positions / n_closed_positions

    @staticmethod
    def get_return_per_closed_position(positions: List[Position]) -> List[float]:
        if len(positions) == 0:
            return []

        t0 = None
        closed_position_returns = []
        for position in positions:
            if position.is_open_position:
                continue
            elif t0 and position.close_ms < t0:
                raise ValueError("Positions must be sorted by close time for this calculation to work.")
            t0 = position.close_ms
            closed_position_returns.append(position.return_at_close)

        cumulative_return = 1
        per_position_return = []

        # calculate the return over time at each position close
        for value in closed_position_returns:
            cumulative_return *= value
            per_position_return.append(cumulative_return)
        return per_position_return

    def get_positions_for_one_hotkey(
        self,
        hotkey: str,
        only_open_positions: bool = False,
        acceptable_position_end_ms: int = None,
        sort_positions: bool = False
    ) -> List[Position]:
        """
        Get positions for a hotkey from the RPC server.

        Args:
            hotkey: The miner hotkey
            only_open_positions: If True, only return open positions
            acceptable_position_end_ms: Optional timestamp filter
            sort_positions: If True, sort positions by close_ms (closed first, then open)

        Returns:
            List of Position objects
        """
        return self._server.get_positions_for_one_hotkey_rpc(
            hotkey,
            only_open_positions,
            acceptable_position_end_ms,
            sort_positions
        )

    def get_positions_for_hotkeys(
        self,
        hotkeys: List[str],
        only_open_positions: bool = False,
        filter_eliminations: bool = False,
        acceptable_position_end_ms: int = None,
        sort_positions: bool = False
    ) -> Dict[str, List[Position]]:
        """
        Get positions for multiple hotkeys from the RPC server.

        Args:
            hotkeys: List of hotkeys to fetch positions for
            only_open_positions: If True, only return open positions
            filter_eliminations: If True, server will filter eliminations internally
            acceptable_position_end_ms: Optional timestamp filter
            sort_positions: If True, sort positions by close_ms (closed first, then open)

        Returns:
            Dict mapping hotkey to list of Position objects
        """
        return self._server.get_positions_for_hotkeys_rpc(
            hotkeys,
            only_open_positions=only_open_positions,
            filter_eliminations=filter_eliminations,
            acceptable_position_end_ms=acceptable_position_end_ms,
            sort_positions=sort_positions
        )

    def get_position(self, hotkey: str, position_uuid: str) -> Optional[Position]:
        """
        Get a specific position by hotkey and UUID.

        Args:
            hotkey: The miner hotkey
            position_uuid: The position UUID

        Returns:
            Position if found, None otherwise
        """
        return self._server.get_position_rpc(hotkey, position_uuid)

    def get_miner_position_by_uuid(self, hotkey: str, position_uuid: str) -> Optional[Position]:
        """
        Alias for get_position() for backward compatibility.
        """
        return self.get_position(hotkey, position_uuid)

    def get_open_position_for_trade_pair(
        self,
        hotkey: str,
        trade_pair_id: str
    ) -> Optional[Position]:
        """
        Get the open position for a specific miner and trade pair.

        Args:
            hotkey: The miner hotkey
            trade_pair_id: The trade pair ID

        Returns:
            Position if found, None otherwise
        """
        return self._server.get_open_position_for_trade_pair_rpc(hotkey, trade_pair_id)

    def get_all_hotkeys(self) -> List[str]:
        """Get all hotkeys that have at least one position."""
        return self._server.get_all_hotkeys_rpc()

    def get_extreme_position_order_processed_on_disk_ms(self) -> tuple:
        """
        Get the minimum and maximum processed_ms timestamps across all orders in all positions.

        Returns:
            tuple: (min_time, max_time) in milliseconds
        """
        return self._server.get_extreme_position_order_processed_on_disk_ms_rpc()

    def get_miner_hotkeys_with_at_least_one_position(self, include_development_positions=False) -> set:
        """Get all hotkeys that have at least one position (returns set for backward compatibility)."""
        hotkeys = set(self._server.get_all_hotkeys_rpc())

        # Filter out development hotkey unless explicitly requested
        if not include_development_positions and ValiConfig.DEVELOPMENT_HOTKEY in hotkeys:
            hotkeys = hotkeys - {ValiConfig.DEVELOPMENT_HOTKEY}

        return hotkeys

    def get_positions_for_all_miners(
        self,
        include_development_positions: bool = False,
        sort_positions: bool = False,
        filter_eliminations: bool = False
    ) -> Dict[str, List[Position]]:
        """
        Get positions for all miners from the RPC server.

        Args:
            include_development_positions: If True, include development hotkey positions
            sort_positions: If True, sort positions by close_ms

        Returns:
            Dict mapping hotkey to list of Position objects
        """
        all_hotkeys = self.get_all_hotkeys()

        # Filter out development hotkey unless explicitly requested
        if not include_development_positions:
            all_hotkeys = [hk for hk in all_hotkeys if hk != ValiConfig.DEVELOPMENT_HOTKEY]

        return self.get_positions_for_hotkeys(
            all_hotkeys,
            only_open_positions=False,
            filter_eliminations=filter_eliminations,
            sort_positions=sort_positions,
        )

    def get_number_of_miners_with_any_positions(self) -> int:
        """Get the number of miners that have at least one position."""
        return len(self.get_all_hotkeys())

    def calculate_net_portfolio_leverage(self, hotkey: str) -> float:
        """
        Calculate leverage across all open positions for a hotkey.
        Normalize each asset class with a multiplier.

        Args:
            hotkey: The miner hotkey

        Returns:
            Total portfolio leverage (sum of abs(leverage) * multiplier for each open position)
        """
        return self._server.calculate_net_portfolio_leverage_rpc(hotkey)

    def compute_realtime_drawdown(self, hotkey: str) -> float:
        """
        Compute the realtime drawdown from positions.
        Bypasses perf ledger, since perf ledgers are refreshed in 5 min intervals and may be out of date.
        Used to enable realtime withdrawals based on drawdown.

        Args:
            hotkey: The miner hotkey

        Returns:
            Drawdown ratio (1.0 = 0% drawdown, 0.9 = 10% drawdown)
        """
        return self._server.compute_realtime_drawdown_rpc(hotkey)

    # ==================== Mutation Methods ====================

    def save_miner_position(self, position: Position, delete_open_position_if_exists: bool = True) -> None:
        """
        Save a position to the server.

        Args:
            position: The position to save
            delete_open_position_if_exists: If True and position is closed, delete any existing
                open position for the same trade pair (liquidation scenario)
        """
        self._server.save_miner_position_rpc(position, delete_open_position_if_exists)

    def delete_position(self, hotkey: str, position_uuid: str) -> None:
        """
        Delete a position from the server.

        Args:
            hotkey: The miner hotkey
            position_uuid: The position UUID to delete
        """
        self._server.delete_position_rpc(hotkey, position_uuid)

    def clear_all_miner_positions(self) -> None:
        """Clear all positions from memory (use with caution!)."""
        self._server.clear_all_miner_positions_rpc()

    def clear_all_miner_positions_and_disk(self, hotkey=None) -> None:
        """Clear all positions from memory AND disk (use with caution!)."""
        self._server.clear_all_miner_positions_and_disk_rpc(hotkey=hotkey)

    def filtered_positions_for_scoring(
        self,
        hotkeys: List[str] = None,
        include_development_positions: bool = False
    ) -> tuple:
        """
        Filter the positions for a set of hotkeys for scoring purposes.
        Excludes development positions by default.

        Args:
            hotkeys: Optional list of hotkeys to filter. If None, uses all hotkeys with positions.
            include_development_positions: If True, include development hotkey positions.

        Returns:
            Tuple of (filtered_positions dict, hk_to_first_order_time dict)
        """
        return self._server.filtered_positions_for_scoring_rpc(
            hotkeys=hotkeys,
            include_development_positions=include_development_positions
        )

    def split_position_on_flat(self, position: Position, track_stats: bool = False) -> tuple[list[Position], dict]:
        """
        Split a position on FLAT orders or implicit flats.

        Args:
            position: The position to split
            track_stats: Whether to track splitting statistics for this miner

        Returns:
            Tuple of (list of split positions, split_info dict)
        """
        return self._server.split_position_on_flat_rpc(position, track_stats)

    def get_split_stats(self, hotkey: str) -> dict:
        """
        Get position splitting statistics for a miner.

        Args:
            hotkey: The miner hotkey

        Returns:
            Dict with splitting statistics
        """
        return self._server.get_split_stats_rpc(hotkey)

    def apply_stock_split(self, trade_pair_id: str, stock_split_ratio: float, execution_date: str):
        return self._server.apply_stock_split_rpc(trade_pair_id, stock_split_ratio, execution_date)

    @staticmethod
    def positions_are_the_same(position1: Position, position2: Position | dict) -> (bool, str):
        # Iterate through all the attributes of position1 and compare them to position2.
        # Get attributes programmatically.
        comparing_to_dict = isinstance(position2, dict)
        for attr in dir(position1):
            # Skip Pydantic internal attributes to avoid deprecation warnings
            if attr.startswith("_") or (attr in ('model_computed_fields', 'model_config', 'model_fields', 'model_fields_set', '__fields__', 'newest_order_age_ms')):
                continue

            attr_is_property = isinstance(getattr(type(position1), attr, None), property)
            if callable(getattr(position1, attr)) or (comparing_to_dict and attr_is_property):
                continue

            value1 = getattr(position1, attr)
            # Check if position2 is a dict and access the value accordingly.
            if comparing_to_dict:
                # Use .get() to avoid KeyError if the attribute is missing in the dictionary.
                value2 = position2.get(attr)
            else:
                value2 = getattr(position2, attr, None)

            # tolerant float comparison
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                value1 = float(value1)
                value2 = float(value2)
                if not math.isclose(value1, value2, rel_tol=1e-9, abs_tol=1e-9):
                    return False, f"{attr} is different. {value1} != {value2}"
            elif value1 != value2:
                return False, f"{attr} is different. {value1} != {value2}"
        return True, ""

    # ==================== Maintenance Methods ====================

    def close_open_orders_for_suspended_trade_pairs(self, live_price_fetcher=None) -> int:
        """
        Close all open positions for suspended trade pairs (SPX, DJI, NDX, VIX).

        Args:
            live_price_fetcher: Optional price fetcher to use. If None, uses server's internal client.
                               Pass a mock price fetcher for testing.

        Returns:
            Number of positions closed
        """
        return self._server.close_open_orders_for_suspended_trade_pairs_rpc(live_price_fetcher)

    def close_all_positions(
        self,
        hotkey: str,
        close_time_ms: int,
        order_source: "OrderSource",
        live_price_fetcher=None
    ) -> int:
        """
        Close all open positions for a specific hotkey.

        Args:
            hotkey: Hotkey whose positions should be closed
            close_time_ms: Timestamp for closing positions
            order_source: OrderSource enum value (e.g., SUBACCOUNT_PROMOTION)
            live_price_fetcher: Optional price fetcher for accurate closing prices

        Returns:
            int: Number of positions successfully closed
        """
        return self._server.close_all_positions_rpc(
            hotkey=hotkey,
            close_time_ms=close_time_ms,
            order_source=order_source,  # Pass as int for RPC
            live_price_fetcher=live_price_fetcher
        )

    # ==================== Pre-run Setup Methods ====================

    def pre_run_setup(self, perform_order_corrections: bool = True) -> List[str]:
        """
        Run pre-run setup operations on the server.
        This is called once at validator startup.

        Args:
            perform_order_corrections: Whether to run order corrections

        Returns:
            List of miner hotkeys that need their perf ledgers wiped
            (caller is responsible for updating perf ledgers)
        """
        return self._server.pre_run_setup_rpc(perform_order_corrections)

    # ==================== Bracket Order Attachment Methods ====================

    def attach_bracket_order_to_position(self, miner_hotkey: str, trade_pair_id: str, order_dict: dict) -> bool:
        """
        Attach a bracket order to a position.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_id: The trade pair ID
            order_dict: The order as a dictionary

        Returns:
            True if successfully attached, False if no open position found
        """
        return self._server.attach_bracket_order_to_position_rpc(miner_hotkey, trade_pair_id, order_dict)

    def remove_bracket_order_from_position(self, miner_hotkey: str, trade_pair_id: str, order_uuid: str) -> bool:
        """
        Remove a bracket order from a position.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_id: The trade pair ID
            order_uuid: The UUID of the order to remove

        Returns:
            True if found and removed, False otherwise
        """
        return self._server.remove_bracket_order_from_position_rpc(miner_hotkey, trade_pair_id, order_uuid)
