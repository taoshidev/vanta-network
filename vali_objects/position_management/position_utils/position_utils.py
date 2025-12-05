# developer: trdougherty

import numpy as np
import copy

from vali_objects.vali_dataclasses.position import Position, Order
from vali_objects.vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderType
import uuid
import logging

from time_util.time_util import TimeUtil


class PositionUtils:
    @staticmethod
    def cumulative_leverage_position(
            positions: list[Position],
            evaluation_time_ms: int = None
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
            evaluation_time_ms: int - the evaluation time in milliseconds
        """
        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        return PositionUtils.translate_current_leverage(
            positions,
            evaluation_time_ms
        )

    @staticmethod
    def build_pseudo_positions(
            positions: dict[str, list[Position]]
    ) -> dict[str, list[Position]]:
        """
        Args:
            positions: dict[str, list[Position]] - the positions
        """
        pseudo_positions = {}
        for miner_key, miner_positions in positions.items():
            pseudo_positions[miner_key] = PositionUtils.positional_equivalence(miner_positions)
        return pseudo_positions

    @staticmethod
    def positional_equivalence(
            cumulative_leverage_positions: list[Position],
            evaluation_window_ms: int = None
    ) -> list[Position]:
        if evaluation_window_ms is None:
            evaluation_window_ms = ValiConfig.POSITIONAL_EQUIVALENCE_WINDOW_MS

        pseudo_positions = []
        grouped_positions = {}

        for position in cumulative_leverage_positions:
            if position.trade_pair not in grouped_positions:
                grouped_positions[position.trade_pair] = []
            grouped_positions[position.trade_pair].append(position)

        for trade_pair, positions_for_pair in grouped_positions.items():
            # Assert that all positions in a group have the same miner_hotkey
            miner_hotkey = positions_for_pair[0].miner_hotkey
            for position in positions_for_pair:
                assert position.miner_hotkey == miner_hotkey, "Positions in the same group must have the same miner_hotkey"

            first_order_indices = [0] + [len(position.orders) for position in positions_for_pair[:-1]]
            first_order_indices = np.cumsum(first_order_indices)

            assert len(first_order_indices) == len(positions_for_pair)

            flattened_orders = PositionUtils.flatten_orders(positions_for_pair)

            for first_order_index in first_order_indices:
                window_orders = PositionUtils.collect_window_of_orders(
                    flattened_orders,
                    first_order_index,
                    evaluation_window_ms
                )

                if not window_orders:  # Handle empty window_orders
                    logging.warning("Encountered empty window_orders. Skipping pseudo-position creation.")
                    continue  # Or raise an exception if that's more appropriate

                open_ms = window_orders[0].processed_ms if window_orders else None
                close_ms = window_orders[-1].processed_ms if window_orders else None
                return_at_close = 1.0

                pseudo_positions.append(Position(
                    miner_hotkey=miner_hotkey,
                    position_uuid=str(uuid.uuid4()),
                    open_ms=open_ms,
                    trade_pair=trade_pair,
                    orders=window_orders,
                    close_ms=close_ms,
                    return_at_close=return_at_close
                ))

        return pseudo_positions

    @staticmethod
    def collect_window_of_orders(
            orders: list[Order],
            start_index: int,
            evaluation_window_ms: int
    ) -> list[Order]:
        """
        Args:
            orders: list[Order] - the orders
            start_index: int - the start index
            evaluation_window_ms: int - the end timestamp_ms
        """
        first_order = orders[start_index]
        first_order_time_ms = first_order.processed_ms
        last_order_time_ms = first_order_time_ms + evaluation_window_ms

        found_orders = []
        for i in range(start_index, len(orders)):
            order = orders[i]
            if order.processed_ms <= last_order_time_ms:
                found_orders.append(order)
            else:
                break

        return found_orders

    @staticmethod
    def translate_current_leverage(
            positions: list[Position],
            evaluation_time_ms: int = None
    ) -> list[Position]:
        """
        Adjusts the leverage of each position based on order types and adds a new order with the final leverage at the end.
        """
        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        positions_copy = copy.deepcopy(positions)
        for position in positions_copy:
            running_leverage = 0
            new_orders = []
            for order in position.orders:
                running_leverage += order.leverage

                if order.order_type == OrderType.FLAT:
                    running_leverage = 0  # Reset leverage if order type is FLAT

                order.leverage = running_leverage

            # Create and append a new order with the final running leverage
            new_order = copy.deepcopy(position.orders[-1])
            new_order.processed_ms = evaluation_time_ms
            new_order.leverage = running_leverage
            if new_order.order_type != OrderType.FLAT:
                new_orders.append(new_order)

            position.orders.extend(new_orders)  # Append all new orders after the loop

        return positions_copy

    @staticmethod
    def average_leverage(positions: list[Position]) -> float:
        """
        Computes the time-weighted average leverage of a list of positions.

        Args:
            positions: list[Position] - the list of positions

        Returns:
            float - the time-weighted average leverage
        """
        if not positions:
            return 0.0

        total_time = 0
        total_timeleverage = 0

        for position in positions:
            if len(position.orders) < 2:
                continue

            last_time = position.orders[0].processed_ms
            running_leverage = position.orders[0].leverage

            for i in range(1, len(position.orders)):
                current_time = position.orders[i].processed_ms
                time_delta = current_time - last_time
                total_time += time_delta
                total_timeleverage += time_delta * abs(running_leverage)
                last_time = current_time
                running_leverage += position.orders[i].leverage

        if total_time == 0:
            return 0.0

        return total_timeleverage / total_time

    @staticmethod
    def total_duration(
            positions: list[Position]
    ) -> int:
        """
        Args:
            positions: list[Position] - the list of positions
        """
        time_deltas = []

        for position in positions:
            if position.is_closed_position:
                time_deltas.append(position.close_ms - position.open_ms)

        return sum(time_deltas)

    @staticmethod
    def flatten_orders(
            positions: list[Position]
    ) -> list[Order]:
        """
        Args:
            positions: list[Position] - the positions
        """
        orders = []
        for position in positions:
            for order in position.orders:
                orders.append(order)
        return orders

    @staticmethod
    def flatten(
            positions: dict[str, list[Position]]
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
        """
        positions_list = []
        for miner_key, miner_positions in positions.items():
            for position in miner_positions:
                positions_list.append(position)

        return positions_list

    @staticmethod
    def running_leverage_computation(
            positions: list[Position]
    ) -> list[Position]:
        """
        Args:
            positions: list[Position] - the positions
        """
        positions_copy = copy.deepcopy(positions)
        for position in positions_copy:
            for order in position.orders:
                order.leverage = np.clip(order.leverage, 0, 1)

        return positions

    @staticmethod
    def to_state_list(
            positions: list[Position],
            current_time: int,
            constrain_lookback: bool = True
    ) -> tuple[list[str], list[str], list[dict]]:
        """
        Args:
            positions: list[Position] - the positions
            current_time: int - the current time
            constrain_lookback: bool - whether to constrain the lookback

        Return:
            return: list[dict] - the order list
        """
        order_list = []

        miners = set()
        trade_pairs = set()

        if constrain_lookback:
            start_time = current_time - ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        else:
            start_time = 0

        for position in positions:
            order_start = 0
            order_end = 0
            order_leverage = 0
            order_tradepair = None
            order_minerid = position.miner_hotkey

            if len(position.orders) == 0:
                continue

            for order_number, order in enumerate(position.orders):
                if order_number == 0:
                    order_start = order.processed_ms
                    order_leverage = order.leverage
                    order_tradepair = order.trade_pair.trade_pair_id
                    order_orderid = order.order_uuid
                    continue

                order_end = order.processed_ms

                if order_start >= start_time:
                    miners.add(order_minerid)
                    trade_pairs.add(order_tradepair)
                    order_list.append({
                        "miner_id": order_minerid,
                        "trade_pair": order_tradepair,
                        "leverage": order_leverage,
                        "start": order_start,
                        "end": order_end,
                        "order_id": order_orderid
                    })

                order_start = order_end
                order_leverage = order.leverage
                order_orderid = order.order_uuid

        return (
            sorted(list(miners)),
            sorted(list(trade_pairs)),
            order_list
        )
