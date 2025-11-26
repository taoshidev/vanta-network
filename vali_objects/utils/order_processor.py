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
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")

        # Validate required fields
        if signal_leverage is None:
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

        limit_price = float(limit_price)

        if stop_loss is not None:
            stop_loss = float(stop_loss)
            if stop_loss <= 0:
                raise SignalException("stop_loss must be greater than 0")

            if signal_order_type == OrderType.LONG and stop_loss >= limit_price:
                raise SignalException(f"For LONG orders, stop_loss ({stop_loss}) must be less than limit_price ({limit_price})")
            elif signal_order_type == OrderType.SHORT and stop_loss <= limit_price:
                raise SignalException(f"For SHORT orders, stop_loss ({stop_loss}) must be greater than limit_price ({limit_price})")

        if take_profit is not None:
            take_profit = float(take_profit)
            if take_profit <= 0:
                raise SignalException("take_profit must be greater than 0")

            if signal_order_type == OrderType.LONG and take_profit <= limit_price:
                raise SignalException(f"For LONG orders, take_profit ({take_profit}) must be greater than limit_price ({limit_price})")
            elif signal_order_type == OrderType.SHORT and take_profit >= limit_price:
                raise SignalException(f"For SHORT orders, take_profit ({take_profit}) must be less than limit_price ({limit_price})")

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
            stop_loss=stop_loss,
            take_profit=take_profit,
            src=OrderSource.LIMIT_UNFILLED
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
    def process_bracket_order(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                             miner_hotkey: str, limit_order_manager) -> Order:
        """
        Process a BRACKET order by creating an Order object and calling limit_order_manager.

        Bracket orders set stop-loss and take-profit on existing positions.
        The limit_order_manager validates the position exists and forces the order type
        to match the position direction.

        Args:
            signal: Signal dictionary with bracket order details
            trade_pair: Parsed TradePair object
            order_uuid: Order UUID
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            limit_order_manager: Manager to process the bracket order

        Returns:
            The created Order object

        Raises:
            SignalException: If required fields are missing, no position exists, or processing fails
        """
        # Extract signal data
        leverage = signal.get("leverage", 0)  # If leverage is None, read leverage from position
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")

        # Validate that at least one of SL or TP is set
        if stop_loss is None and take_profit is None:
            raise SignalException("Bracket order must specify at least one of stop_loss or take_profit")

        # Parse and validate stop_loss
        if stop_loss is not None:
            stop_loss = float(stop_loss)
            if stop_loss <= 0:
                raise SignalException("stop_loss must be greater than 0")

        # Parse and validate take_profit
        if take_profit is not None:
            take_profit = float(take_profit)
            if take_profit <= 0:
                raise SignalException("take_profit must be greater than 0")

        # Create bracket order (order_type will be set by limit_order_manager)
        order = Order(
            trade_pair=trade_pair,
            order_uuid=order_uuid,
            processed_ms=now_ms,
            price=0.0,
            order_type=OrderType.LONG,  # Placeholder - will be overridden by manager
            leverage=float(leverage),
            execution_type=ExecutionType.BRACKET,
            limit_price=None,  # Not used for bracket orders
            stop_loss=stop_loss,
            take_profit=take_profit,
            src=OrderSource.BRACKET_UNFILLED
        )

        # Process the bracket order - manager validates position and sets correct order_type/leverage
        limit_order_manager.process_limit_order(miner_hotkey, order)

        bt.logging.debug(f"Processed BRACKET order: {order.order_uuid} for {miner_hotkey}")
        return order

    @staticmethod
    def process_market_order(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                            miner_hotkey: str, miner_repo_version: str,
                            market_order_manager, synapse=None):
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
            For validator path (synapse): Order object (or None if no order created)
            For REST API path: Tuple of (error_message, updated_position, created_order)

        Raises:
            SignalException: If processing fails
        """
        if synapse:
            # Validator path: use synapse-based method
            created_order = market_order_manager.process_market_order(
                synapse, order_uuid, miner_repo_version, trade_pair,
                now_ms, signal, miner_hotkey
            )
            # Error checking happens via synapse.error_message
            return created_order
        else:
            # REST API path: use direct method
            err_msg, updated_position, created_order = market_order_manager._process_market_order(
                order_uuid, miner_repo_version, trade_pair,
                now_ms, signal, miner_hotkey, price_sources=None
            )
            return err_msg, updated_position, created_order
