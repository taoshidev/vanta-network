"""
Position Splitter - Shared utility for splitting positions on FLAT orders.

This module contains the single source of truth for position splitting logic.
Both PositionManager (client) and PositionManagerServer (server) use this module
to avoid code duplication.
"""

import bittensor as bt
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position


class PositionSplitter:
    """
    Utility class for splitting positions based on FLAT orders or implicit flats.

    All methods are static since they operate on position data without maintaining state.
    """

    @staticmethod
    def find_split_points(position: Position) -> list[int]:
        """
        Find all valid split points in a position where splitting should occur.

        Returns a list of order indices where splits should happen.
        This is the single source of truth for split logic.

        A split occurs at an order index if:
        1. The order is an explicit FLAT, OR
        2. The cumulative leverage reaches zero (implicit flat), OR
        3. The cumulative leverage flips sign (implicit flat)

        AND the split would create two valid sub-positions:
        - First part: at least 2 orders, doesn't start with FLAT
        - Second part: at least 1 order, doesn't start with FLAT

        Args:
            position: The position to analyze for split points

        Returns:
            List of order indices where splits should happen
        """
        if len(position.orders) < 2:
            return []

        split_points = []
        cumulative_leverage = 0.0
        previous_sign = None

        for i, order in enumerate(position.orders):
            cumulative_leverage += order.leverage

            # Determine the sign of leverage (positive, negative, or zero)
            if abs(cumulative_leverage) < 1e-9:
                current_sign = 0
            elif cumulative_leverage > 0:
                current_sign = 1
            else:
                current_sign = -1

            # Check for leverage sign flip
            leverage_flipped = False
            if previous_sign is not None and previous_sign != 0 and current_sign != 0 and previous_sign != current_sign:
                leverage_flipped = True

            # Check for explicit FLAT or implicit flat (leverage reaches zero or flips sign)
            is_explicit_flat = order.order_type == OrderType.FLAT
            is_implicit_flat = (abs(cumulative_leverage) < 1e-9 or leverage_flipped) and not is_explicit_flat

            if is_explicit_flat or is_implicit_flat:
                # Don't split if this is the last order
                if i < len(position.orders) - 1:
                    # Check if the split would create valid sub-positions
                    orders_before = position.orders[:i+1]
                    orders_after = position.orders[i+1:]

                    # Check if first part is valid (2+ orders, doesn't start with FLAT)
                    first_valid = (len(orders_before) >= 2 and
                                 orders_before[0].order_type != OrderType.FLAT)

                    # Check if second part would be valid (at least 1 order, doesn't start with FLAT)
                    second_valid = (len(orders_after) >= 1 and
                                  orders_after[0].order_type != OrderType.FLAT)

                    if first_valid and second_valid:
                        split_points.append(i)
                        cumulative_leverage = 0.0  # Reset for next segment
                        previous_sign = 0
                        continue

            # Update previous sign for next iteration
            previous_sign = current_sign

        return split_points

    @staticmethod
    def position_needs_splitting(position: Position) -> bool:
        """
        Check if a position would actually be split by split_position_on_flat.

        Uses the same logic as split_position_on_flat but without creating new positions.

        Args:
            position: The position to check

        Returns:
            True if the position would be split, False otherwise
        """
        return len(PositionSplitter.find_split_points(position)) > 0

    @staticmethod
    def split_position_on_flat(position: Position, price_fetcher_client, track_stats: bool = False) -> tuple[list[Position], dict]:
        """
        Split a position into multiple positions separated by FLAT orders or implicit flats.

        Implicit flat is defined as:
        - Cumulative leverage reaches zero (abs(cumulative_leverage) < 1e-9), OR
        - Cumulative leverage flips sign (e.g., from positive to negative or vice versa)

        Uses find_split_points as the single source of truth for split logic.

        Ensures:
        - CLOSED positions have at least 2 orders
        - OPEN positions can have 1 order
        - No position starts with a FLAT order

        Args:
            position: The position to split
            price_fetcher_client: Price fetcher for rebuilding positions after splitting
            track_stats: If True, returns detailed statistics about split types

        Returns:
            tuple: (list of positions, split_info dict with 'implicit_flat_splits' and 'explicit_flat_splits')
        """
        try:
            split_points = PositionSplitter.find_split_points(position)

            if not split_points:
                return [position], {'implicit_flat_splits': 0, 'explicit_flat_splits': 0}

            # Track pre-split return if requested
            pre_split_return = position.return_at_close if track_stats else None

            # Count implicit vs explicit flats (always needed for statistics)
            implicit_flat_splits = 0
            explicit_flat_splits = 0

            cumulative_leverage = 0.0
            previous_sign = None

            for i, order in enumerate(position.orders):
                cumulative_leverage += order.leverage

                # Determine the sign of leverage (positive, negative, or zero)
                if abs(cumulative_leverage) < 1e-9:
                    current_sign = 0
                elif cumulative_leverage > 0:
                    current_sign = 1
                else:
                    current_sign = -1

                # Check for leverage sign flip
                leverage_flipped = False
                if previous_sign is not None and previous_sign != 0 and current_sign != 0 and previous_sign != current_sign:
                    leverage_flipped = True

                if i in split_points:
                    if order.order_type == OrderType.FLAT:
                        explicit_flat_splits += 1
                    elif abs(cumulative_leverage) < 1e-9 or leverage_flipped:
                        implicit_flat_splits += 1

                # Update previous sign for next iteration
                previous_sign = current_sign

            # Create order groups based on split points
            order_groups = []
            start_idx = 0

            for split_idx in split_points:
                # Add orders up to and including the split point
                order_group = position.orders[start_idx:split_idx + 1]
                order_groups.append(order_group)
                start_idx = split_idx + 1

            # Add remaining orders if any
            if start_idx < len(position.orders):
                order_groups.append(position.orders[start_idx:])

            # Update the original position with the first group
            position.orders = order_groups[0]
            position.rebuild_position_with_updated_orders(price_fetcher_client)

            positions = [position]

            # Create new positions for remaining groups
            for order_group in order_groups[1:]:
                new_position = Position(
                    miner_hotkey=position.miner_hotkey,
                    position_uuid=order_group[0].order_uuid,
                    open_ms=0,
                    trade_pair=position.trade_pair,
                    orders=order_group,
                    account_size=position.account_size
                )
                new_position.rebuild_position_with_updated_orders(price_fetcher_client)
                positions.append(new_position)

            split_info = {
                'implicit_flat_splits': implicit_flat_splits,
                'explicit_flat_splits': explicit_flat_splits,
                'pre_split_return': pre_split_return
            }

            return positions, split_info

        except Exception as e:
            bt.logging.error(f"Error during position splitting: {e}")
            bt.logging.error(f"Position details: UUID={position.position_uuid}, Orders={len(position.orders)}, Trade Pair={position.trade_pair}")
            # Return original position on error
            return [position], {'implicit_flat_splits': 0, 'explicit_flat_splits': 0}
