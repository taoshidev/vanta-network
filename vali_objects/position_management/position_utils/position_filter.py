"""
Position filtering utilities for filtering positions by date and asset type.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Optional
from vali_objects.vali_dataclasses.position import Position


@dataclass
class FilterStats:
    """Statistics from position filtering operations"""
    total_positions_before_filter: int = 0
    equities_positions_skipped: int = 0
    indices_positions_skipped: int = 0
    date_filtered_out: int = 0
    final_positions: int = 0

    def has_skipped_assets(self) -> bool:
        """Check if any equities or indices were skipped."""
        return self.equities_positions_skipped > 0 or self.indices_positions_skipped > 0


class PositionFilter:
    """Handles filtering of positions by date and asset type."""

    @staticmethod
    def filter_single_position(position: Position, cutoff_date_ms: int, live_price_fetcher) -> Tuple[Optional[Position], str]:
        """
        Filter a single position by date and asset type.

        Args:
            position: Position to filter
            cutoff_date_ms: Cutoff timestamp in milliseconds
            live_price_fetcher: Price fetcher for rebuilding position

        Returns:
            Tuple of (filtered_position, skip_reason). Position is None if skipped.
            skip_reason can be: "equities", "indices", "date_filtered", or "kept"
        """
        # Skip positions for equities and indices assets
        if position.trade_pair.is_equities:
            return None, "equities"
        elif position.trade_pair.is_indices:
            return None, "indices"

        # Filter orders to only include those before cutoff
        # Note: cutoff_date_ms should be the END of the day we want to include
        # So use < instead of <= to include all orders within the day
        filtered_orders = [
            order for order in position.orders
            if order.processed_ms < cutoff_date_ms  # Use < for end-of-day cutoff
        ]

        if not filtered_orders:
            return None, "date_filtered"

        if len(filtered_orders) == len(position.orders) and position.is_closed_position:
            return deepcopy(position), "kept"

        # Create a copy of the position with filtered orders
        filtered_position = Position(
            miner_hotkey=position.miner_hotkey,
            position_uuid=position.position_uuid,
            open_ms=position.open_ms,
            close_ms=position.close_ms if position.close_ms and position.close_ms <= cutoff_date_ms else None,
            trade_pair=position.trade_pair,
            orders=filtered_orders,
            position_type=position.position_type,
            is_closed_position=position.is_closed_position and position.close_ms and position.close_ms <= cutoff_date_ms,
        )
        filtered_position.rebuild_position_with_updated_orders(live_price_fetcher)
        return filtered_position, "kept"

    @staticmethod
    def filter_single_position_simple(position: Position, cutoff_date_ms: int) -> Optional[Position]:
        """
        Simplified version of filter_single_position that doesn't require live_price_fetcher.
        Does not call rebuild_position_with_updated_orders.

        Args:
            position: Position to filter
            cutoff_date_ms: Cutoff timestamp in milliseconds

        Returns:
            Filtered position or None if skipped
        """
        # Skip positions for equities and indices assets
        if position.trade_pair.is_equities or position.trade_pair.is_indices:
            return None

        # Filter orders to only include those before cutoff
        filtered_orders = [
            order for order in position.orders
            if order.processed_ms < cutoff_date_ms
        ]

        if not filtered_orders:
            return None

        if len(filtered_orders) == len(position.orders) and position.is_closed_position:
            return deepcopy(position)

        # Create a copy of the position with filtered orders
        filtered_position = Position(
            miner_hotkey=position.miner_hotkey,
            position_uuid=position.position_uuid,
            open_ms=position.open_ms,
            close_ms=position.close_ms if position.close_ms and position.close_ms <= cutoff_date_ms else None,
            trade_pair=position.trade_pair,
            orders=filtered_orders,
            position_type=position.position_type,
            is_closed_position=position.is_closed_position and position.close_ms and position.close_ms <= cutoff_date_ms,
            current_return=position.current_return,
            return_at_close=position.return_at_close,
            net_leverage=position.net_leverage,
            average_entry_price=position.average_entry_price,
            max_leverage_seen=position.max_leverage_seen
        )
        return filtered_position
