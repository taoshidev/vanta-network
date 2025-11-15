"""
Order processing logic shared between validator.py and rest_server.py.

This module provides a single source of truth for processing orders,
ensuring consistent behavior whether orders come from miners via synapses
or from development/testing via REST API.
"""
import uuid
import bittensor as bt
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.vali_dataclasses.order import Order, OrderSource


class OrderProcessor:
    """
    Processes orders by routing them to the appropriate manager based on execution type.

    This class encapsulates the common logic for:
    - Parsing signals and trade pairs
    - Creating Order objects for LIMIT orders
    - Routing to limit_order_manager or market_order_manager
    """

    @staticmethod
    def parse_signal_data(signal: dict, miner_order_uuid: str = None) -> tuple:
        """
        Parse and validate common fields from a signal dict.

        Args:
            signal: Signal dictionary containing order details
            miner_order_uuid: Optional UUID (if not provided, will be generated)

        Returns:
            Tuple of (trade_pair, execution_type, order_uuid)

        Raises:
            SignalException: If required fields are missing or invalid
        """
        # Parse trade pair
        trade_pair = Order.parse_trade_pair_from_signal(signal)
        if trade_pair is None:
            raise SignalException(
                f"Invalid trade pair in signal. Raw signal: {signal}"
            )

        # Parse execution type (defaults to MARKET for backwards compatibility)
        try:
            execution_type = ExecutionType.from_string(signal.get("execution_type", "MARKET").upper())
        except ValueError as e:
            raise SignalException(f"Invalid execution_type: {str(e)}")

        # Generate UUID if not provided
        order_uuid = miner_order_uuid if miner_order_uuid else str(uuid.uuid4())

        return trade_pair, execution_type, order_uuid

    @staticmethod
    def process_limit_order(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                           miner_hotkey: str, limit_order_manager) -> Order:
        """
        Process a LIMIT order by creating an Order object and calling limit_order_manager.

        Args:
            signal: Signal dictionary with limit order details
            trade_pair: Parsed TradePair object
            order_uuid: Order UUID
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            limit_order_manager: Manager to process the limit order

        Returns:
            The created Order object

        Raises:
            SignalException: If required fields are missing or processing fails
        """
        # Extract signal data
        signal_leverage = signal.get("leverage")
        signal_order_type_str = signal.get("order_type")
        limit_price = signal.get("limit_price")

        # Validate required fields
        if not signal_leverage:
            raise SignalException("Missing required field: leverage")
        if not signal_order_type_str:
            raise SignalException("Missing required field: order_type")
        if not limit_price:
            raise SignalException("must set limit_price for limit order")

        # Parse order type
        try:
            signal_order_type = OrderType.from_string(signal_order_type_str)
        except ValueError as e:
            raise SignalException(f"Invalid order_type: {str(e)}")

        # Create order object
        order = Order(
            trade_pair=trade_pair,
            order_uuid=order_uuid,
            processed_ms=now_ms,
            price=0.0,
            order_type=signal_order_type,
            leverage=float(signal_leverage),
            execution_type=ExecutionType.LIMIT,
            limit_price=float(limit_price),
            src=OrderSource.ORDER_SRC_LIMIT_UNFILLED
        )

        # Process the limit order (may throw SignalException)
        limit_order_manager.process_limit_order(miner_hotkey, order)

        bt.logging.debug(f"Processed LIMIT order: {order.order_uuid} for {miner_hotkey}")
        return order

    @staticmethod
    def process_limit_cancel(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                            miner_hotkey: str, limit_order_manager) -> dict:
        """
        Process a LIMIT_CANCEL operation by calling limit_order_manager.

        Args:
            signal: Signal dictionary (order_uuid may be in here for specific cancel)
            trade_pair: Parsed TradePair object
            order_uuid: Order UUID to cancel (or None/empty for cancel all)
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            limit_order_manager: Manager to process the cancellation

        Returns:
            Result dictionary from limit_order_manager

        Raises:
            SignalException: If cancellation fails
        """
        # Call cancel limit order (may throw SignalException)
        result = limit_order_manager.cancel_limit_order(
            miner_hotkey,
            trade_pair.trade_pair_id,
            order_uuid,
            now_ms
        )

        bt.logging.debug(f"Cancelled LIMIT order(s) for {miner_hotkey}: {order_uuid or 'all'}")
        return result

    @staticmethod
    def process_market_order(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                            miner_hotkey: str, miner_repo_version: str,
                            market_order_manager, synapse=None) -> tuple:
        """
        Process a MARKET order by calling market_order_manager.

        Args:
            signal: Signal dictionary with market order details
            trade_pair: Parsed TradePair object
            order_uuid: Order UUID
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            miner_repo_version: Version of miner repo
            market_order_manager: Manager to process the market order
            synapse: Optional synapse object (for validator path)

        Returns:
            Tuple of (error_message, updated_position)

        Raises:
            SignalException: If processing fails
        """
        if synapse:
            # Validator path: use synapse-based method
            market_order_manager.process_market_order(
                synapse, order_uuid, miner_repo_version, trade_pair,
                now_ms, signal, miner_hotkey
            )
            # Error checking happens via synapse.error_message
            return None, None
        else:
            # REST API path: use direct method
            err_msg, updated_position = market_order_manager._process_market_order(
                order_uuid, miner_repo_version, trade_pair,
                now_ms, signal, miner_hotkey, price_sources=None
            )
            return err_msg, updated_position
