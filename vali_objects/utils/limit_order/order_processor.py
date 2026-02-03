"""
Order processing logic shared between validator.py and validator_rest_server.py.

This module provides a single source of truth for processing orders,
ensuring consistent behavior whether orders come from miners via synapses
or from development/testing via REST API.
"""
import uuid
import json
import bittensor as bt
from dataclasses import dataclass
from typing import Optional
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.bracket_order_exception import BracketOrderException
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position
from vali_objects.enums.order_source_enum import OrderSource


@dataclass(frozen=True)  # Immutable for thread safety
class OrderProcessingResult:
    """
    Standardized result from order processing.

    Attributes:
        execution_type: Type of execution (MARKET, LIMIT, BRACKET, LIMIT_CANCEL)
        success: Whether processing succeeded (always True if no exception raised)
        order: The created/processed Order object (None for LIMIT_CANCEL)
        result_dict: Result dictionary (used for LIMIT_CANCEL response)
        updated_position: Updated position (used for MARKET orders)
        should_track_uuid: Whether to add UUID to tracker (False for LIMIT_CANCEL)
    """
    execution_type: ExecutionType
    success: bool = True
    order: Optional[Order] = None
    result_dict: Optional[dict] = None
    updated_position: Optional[Position] = None
    should_track_uuid: bool = True

    @property
    def order_for_logging(self) -> Optional[Order]:
        """Get order object for logging (used by validator.py)."""
        return self.order

    def get_response_json(self) -> str:
        """
        Get JSON response string for synapse or REST API.

        Returns:
            JSON string representation of the result
        """
        if self.order:
            return self.order.__str__()
        elif self.result_dict:
            return json.dumps(self.result_dict)
        return ""


class OrderProcessor:
    """
    Processes orders by routing them to the appropriate manager based on execution type.

    This class encapsulates the common logic for:
    - Parsing signals and trade pairs
    - Creating Order objects for LIMIT orders
    - Routing to limit_order_manager or market_order_manager
    """

    @staticmethod
    def parse_size(signal: dict) -> tuple:
        """
        Parse and convert size fields (leverage, value, quantity) from signal.

        Args:
            signal: Signal dictionary containing size fields

        Returns:
            Tuple of (leverage, value, quantity) as floats or None

        Raises:
            SignalException: If conversion fails
        """
        leverage = signal.get("leverage")
        value = signal.get("value")
        quantity = signal.get("quantity")

        # Convert size fields to float if provided (needed for proper validation in Order model)
        try:
            leverage = float(leverage) if leverage is not None else None
            value = float(value) if value is not None else None
            quantity = float(quantity) if quantity is not None else None
        except (ValueError, TypeError) as e:
            raise SignalException(f"Invalid size field: {str(e)}")

        bt.logging.info(f"[ORDER_PROCESSOR] Parsed size fields - leverage: {leverage}, value: {value}, quantity: {quantity}")

        return leverage, value, quantity

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
        # Parse execution type (defaults to MARKET for backwards compatibility)
        try:
            execution_type = ExecutionType.from_string(signal.get("execution_type", "MARKET").upper())
        except ValueError as e:
            raise SignalException(f"Invalid execution_type: {str(e)}")

        # Parse trade pair (allow None for LIMIT_CANCEL operations)
        trade_pair = Order.parse_trade_pair_from_signal(signal)
        if trade_pair is None and execution_type not in [ExecutionType.LIMIT_CANCEL, ExecutionType.FLAT_ALL]:
            raise SignalException(
                f"Invalid trade pair in signal. Raw signal: {signal}"
            )

        # Generate UUID if not provided
        order_uuid = miner_order_uuid if miner_order_uuid else str(uuid.uuid4())

        return trade_pair, execution_type, order_uuid

    @staticmethod
    def process_limit_order(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                           miner_hotkey: str, limit_order_client, is_edit: bool = False) -> Order:
        """
        Process a LIMIT order by creating an Order object and calling limit_order_manager.

        Args:
            signal: Signal dictionary with limit order details
            trade_pair: Parsed TradePair object
            order_uuid: Order UUID
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            limit_order_client: Client to process the limit order
            is_edit: If True, this is an edit operation (replaces existing order)

        Returns:
            The created Order object

        Raises:
            SignalException: If required fields are missing or processing fails
        """
        # Parse size fields using common method
        leverage, value, quantity = OrderProcessor.parse_size(signal)
        if not leverage and not value and not quantity:
            raise SignalException("Order size must be set: leverage, value, or quantity")

        # Extract other signal data
        signal_order_type_str = signal.get("order_type")
        limit_price = signal.get("limit_price")
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        bracket_orders = signal.get("bracket_orders")

        # Validate required fields
        if not signal_order_type_str:
            raise SignalException("Missing required field: order_type")
        if not limit_price:
            raise SignalException("must set limit_price for limit order")

        # Parse order type
        try:
            signal_order_type = OrderType.from_string(signal_order_type_str)
        except ValueError as e:
            raise SignalException(f"Invalid order_type: {str(e)}")

        # Convert remaining numeric fields to float
        limit_price = float(limit_price)

        # Basic sanity validation only - business rule validation (SL/TP vs limit_price) is in LimitOrderManager
        if stop_loss is not None:
            stop_loss = float(stop_loss)
            if stop_loss <= 0:
                raise SignalException("stop_loss must be greater than 0")

        if take_profit is not None:
            take_profit = float(take_profit)
            if take_profit <= 0:
                raise SignalException("take_profit must be greater than 0")

        # Create order object
        order = Order(
            trade_pair=trade_pair,
            order_uuid=order_uuid,
            processed_ms=now_ms,
            price=0.0,
            order_type=signal_order_type,
            leverage=leverage,
            quantity=quantity,
            value=value,
            execution_type=ExecutionType.LIMIT,
            limit_price=float(limit_price),
            stop_loss=stop_loss,
            take_profit=take_profit,
            bracket_orders=bracket_orders,
            src=OrderSource.LIMIT_UNFILLED
        )

        # Process the limit order (may throw SignalException)
        limit_order_client.process_limit_order(miner_hotkey, order, is_edit=is_edit)

        bt.logging.info(f"[ORDER_PROCESSOR] Processed LIMIT order{'(EDIT)' if is_edit else ''}: {order.order_uuid} for {miner_hotkey}")
        return order

    @staticmethod
    def process_limit_cancel(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                            miner_hotkey: str, limit_order_client) -> dict:
        """
        Process a LIMIT_CANCEL operation by calling limit_order_client.

        Args:
            signal: Signal dictionary (order_uuid may be in here for specific cancel)
            trade_pair: Parsed TradePair object (can be None for cancel by UUID)
            order_uuid: Order UUID to cancel (or None/empty for cancel all)
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            limit_order_client: Client to process the cancellation

        Returns:
            Result dictionary from limit_order_client

        Raises:
            SignalException: If cancellation fails
        """

        # Call cancel limit order (may throw SignalException)
        result = limit_order_client.cancel_limit_order(
            miner_hotkey,
            None,  # TODO support cancel by trade pair in v2
            order_uuid,
            now_ms
        )

        bt.logging.debug(f"Cancelled LIMIT order(s) for {miner_hotkey}: {order_uuid or 'all'}")
        return result

    @staticmethod
    def process_bracket_order(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                             miner_hotkey: str, limit_order_client, is_edit: bool = False) -> Order:
        """
        Process a BRACKET order by creating an Order object and calling limit_order_manager.

        Bracket orders set stop-loss and take-profit on existing positions.
        The limit_order_manager validates the position exists and forces the order type
        to match the position direction.

        Supports two input formats:
        1. Top-level stop_loss/take_profit fields
        2. Single bracket_orders entry (when top-level fields are empty)

        Args:
            signal: Signal dictionary with bracket order details
            trade_pair: Parsed TradePair object
            order_uuid: Order UUID
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            limit_order_client: Client to process the bracket order
            is_edit: If True, this is an edit operation (replaces existing order)

        Returns:
            The created Order object

        Raises:
            SignalException: If required fields are missing, no position exists, or processing fails
        """
        # Extract signal data
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        bracket_orders = signal.get("bracket_orders")

        # If top-level SL/TP empty but bracket_orders provided, extract from first entry
        if stop_loss is None and take_profit is None and bracket_orders:
            if len(bracket_orders) != 1:
                raise SignalException("bracket_orders must contain exactly one entry when used for BRACKET orders")

            signal = {**signal, **bracket_orders[0]}
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")

        # Parse size fields using common method
        leverage, value, quantity = OrderProcessor.parse_size(signal)

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
            order_type=OrderType.FLAT,  # Placeholder - will be overridden by manager
            leverage=leverage,
            quantity=quantity,
            value=value,
            execution_type=ExecutionType.BRACKET,
            limit_price=None,  # Not used for bracket orders
            stop_loss=stop_loss,
            take_profit=take_profit,
            src=OrderSource.BRACKET_UNFILLED
        )

        # Process the bracket order - manager validates position and sets correct order_type/leverage
        limit_order_client.process_limit_order(miner_hotkey, order, is_edit=is_edit)

        bt.logging.info(f"[ORDER_PROCESSOR] Processed BRACKET order{'(EDIT)' if is_edit else ''}: {order.order_uuid} for {miner_hotkey}")
        return order

    @staticmethod
    def process_flat_all_order(order_uuid: str, now_ms: int, miner_hotkey: str, miner_repo_version: str, market_order_manager):
        return market_order_manager.process_flat_all_order(order_uuid, miner_repo_version, miner_hotkey, now_ms)

    @staticmethod
    def process_limit_edit(signal: dict, order_uuid: str, now_ms: int,
                          miner_hotkey: str, limit_order_client) -> Order:
        """
        Process a LIMIT_EDIT operation by validating the existing order and delegating
        to the appropriate handler based on execution type.

        Users must resubmit the entire signal (no partial updates).

        Supports bulk bracket order updates: If order_uuid doesn't exist but bracket_orders
        is provided, process each bracket order (update if order_uuid exists, create if not).

        Validation responsibilities:
        - OrderProcessor: Order exists, is unfilled, trade pair matches (fail fast)
        - Delegated handler: Field validation and business rules

        Args:
            signal: Signal dictionary with complete order details
            order_uuid: UUID of the order to edit
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            limit_order_client: Client to process the edit

        Returns:
            The updated Order object (or first order for bulk updates)

        Raises:
            SignalException: If order not found, not unfilled, or trade pair mismatch
        """
        # 1. LOOKUP existing order
        existing_order_dict = limit_order_client.get_limit_order_by_uuid(miner_hotkey, order_uuid)

        # 2. If order found, process single order edit
        if existing_order_dict:
            existing_order = Order.from_dict(existing_order_dict)

            # VALIDATE order is unfilled
            if existing_order.src not in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED]:
                raise SignalException(f"Cannot edit order {order_uuid}: order is not unfilled (status: {existing_order.src})")

            # VALIDATE trade pair matches
            trade_pair = Order.parse_trade_pair_from_signal(signal)
            if not trade_pair:
                raise SignalException("Invalid trade pair in signal for LIMIT_EDIT")
            if trade_pair != existing_order.trade_pair:
                raise SignalException(
                    f"Cannot edit order {order_uuid}: trade pair mismatch "
                    f"(order is for {existing_order.trade_pair.trade_pair_id}, not {trade_pair.trade_pair_id})"
                )

            # DELEGATE based on existing order's execution type
            if existing_order.execution_type == ExecutionType.BRACKET:
                return OrderProcessor.process_bracket_order(
                    signal, trade_pair, order_uuid, now_ms, miner_hotkey, limit_order_client, is_edit=True
                )
            else:
                return OrderProcessor.process_limit_order(
                    signal, trade_pair, order_uuid, now_ms, miner_hotkey, limit_order_client, is_edit=True
                )

        # 3. Order not found - check for bulk bracket update via bracket_orders field
        bracket_orders = signal.get("bracket_orders")
        if bracket_orders:
            trade_pair = Order.parse_trade_pair_from_signal(signal)
            if not trade_pair:
                raise SignalException("Invalid trade pair in signal for bulk bracket update")

            processed_orders = []
            for bracket in bracket_orders:
                bracket_uuid = bracket.get("order_uuid")

                # Check if this bracket order exists (for update)
                if bracket_uuid:
                    existing_bracket = limit_order_client.get_limit_order_by_uuid(miner_hotkey, bracket_uuid)
                    if not existing_bracket:
                        bt.logging.warning(f"Cannot edit order {bracket_uuid}: order not found, skipping")
                        continue

                    # If bracket_uuid exists but both SL and TP are None, cancel the order
                    if bracket.get("stop_loss") is None and bracket.get("take_profit") is None:
                        try:
                            OrderProcessor.process_limit_cancel(
                                None, None, bracket_uuid, now_ms,
                                miner_hotkey, limit_order_client
                            )
                            bt.logging.info(f"[ORDER_PROCESSOR] Cancelled bracket order {bracket_uuid} (no SL/TP provided)")
                        except SignalException as e:
                            bt.logging.warning(f"Failed to cancel bracket order {bracket_uuid}: {e}, skipping")
                        continue

                    is_edit = True
                else:
                    # No UUID provided - generate new one
                    is_edit = False
                    bracket_uuid = str(uuid.uuid4())

                # Build bracket signal
                bracket_signal = {
                    "trade_pair": signal.get("trade_pair"),
                    "stop_loss": bracket.get("stop_loss"),
                    "take_profit": bracket.get("take_profit"),
                    "leverage": bracket.get("leverage"),
                    "value": bracket.get("value"),
                    "quantity": bracket.get("quantity"),
                }

                order = OrderProcessor.process_bracket_order(
                    bracket_signal, trade_pair, bracket_uuid, now_ms,
                    miner_hotkey, limit_order_client, is_edit=is_edit
                )
                processed_orders.append(order)

            bt.logging.info(
                f"[ORDER_PROCESSOR] Processed bulk bracket update for {miner_hotkey}: "
                f"{len(processed_orders)} orders"
            )

            if processed_orders:
                return processed_orders[0]

            raise SignalException("No bracket edits to process")

        raise SignalException(f"Cannot edit order {order_uuid}: order not found")

    @staticmethod
    def process_market_order(signal: dict, trade_pair, order_uuid: str, now_ms: int,
                            miner_hotkey: str, miner_repo_version: str,
                            market_order_manager) -> tuple:
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

        Returns:
            Tuple of (error_message, updated_position, created_order):
                - error_message: Empty string if success, error string if failed
                - updated_position: Position object if successful, None otherwise
                - created_order: Order object if successful, None otherwise

        Raises:
            SignalException: If processing fails with validation error
        """
        # Use direct method for consistent interface across validator and REST API
        err_msg, updated_position, created_order = market_order_manager._process_market_order(
            order_uuid, miner_repo_version, trade_pair,
            now_ms, signal, miner_hotkey, price_sources=None
        )
        return err_msg, updated_position, created_order

    @staticmethod
    def process_order(
        signal: dict,
        miner_order_uuid: str,
        now_ms: int,
        miner_hotkey: str,
        miner_repo_version: str,
        limit_order_client,
        market_order_manager
    ) -> OrderProcessingResult:
        """
        Unified order processing dispatcher that routes to the appropriate handler.

        This method centralizes the execution type routing logic that was previously
        duplicated in validator.py (lines 607-661) and validator_rest_server.py (lines 1475-1549).

        Benefits:
        - Single source of truth for order processing logic
        - Consistent behavior across validator and REST API
        - Easier testing (can test without Flask/Axon)
        - Reduced code duplication (~113 lines eliminated)

        Args:
            signal: Signal dictionary containing order details
            miner_order_uuid: Order UUID (or None to auto-generate)
            now_ms: Current timestamp in milliseconds
            miner_hotkey: Miner's hotkey
            miner_repo_version: Version of miner repo (for MARKET orders)
            limit_order_client: Client for limit/bracket/cancel operations
            market_order_manager: Manager for market orders

        Returns:
            OrderProcessingResult with standardized response data

        Raises:
            SignalException: If processing fails with validation error
        """
        # Parse common fields (may raise SignalException)
        trade_pair, execution_type, order_uuid = OrderProcessor.parse_signal_data(
            signal, miner_order_uuid
        )

        # Route based on execution type
        if execution_type == ExecutionType.LIMIT:
            order = OrderProcessor.process_limit_order(
                signal, trade_pair, order_uuid, now_ms,
                miner_hotkey, limit_order_client
            )
            return OrderProcessingResult(
                execution_type=ExecutionType.LIMIT,
                order=order,
                should_track_uuid=True
            )

        elif execution_type == ExecutionType.BRACKET:
            order = OrderProcessor.process_bracket_order(
                signal, trade_pair, order_uuid, now_ms,
                miner_hotkey, limit_order_client
            )
            return OrderProcessingResult(
                execution_type=ExecutionType.BRACKET,
                order=order,
                should_track_uuid=True
            )

        elif execution_type == ExecutionType.LIMIT_CANCEL:
            result = OrderProcessor.process_limit_cancel(
                signal, trade_pair, order_uuid, now_ms,
                miner_hotkey, limit_order_client
            )
            return OrderProcessingResult(
                execution_type=ExecutionType.LIMIT_CANCEL,
                result_dict=result,
                should_track_uuid=False  # No UUID tracking for cancellations
            )

        elif execution_type == ExecutionType.LIMIT_EDIT:
            order = OrderProcessor.process_limit_edit(
                signal, order_uuid, now_ms,
                miner_hotkey, limit_order_client
            )
            return OrderProcessingResult(
                execution_type=ExecutionType.LIMIT_EDIT,
                order=order,
                should_track_uuid=False  # No UUID tracking for edits (keeps same UUID)
            )

        elif execution_type == ExecutionType.FLAT_ALL:
            result = OrderProcessor.process_flat_all_order(
                order_uuid, now_ms, miner_hotkey,
                miner_repo_version, market_order_manager
            )

            return OrderProcessingResult(
                execution_type=ExecutionType.FLAT_ALL,
                result_dict=result,
                should_track_uuid=True
            )

        else:  # ExecutionType.MARKET
            err_msg, updated_position, created_order = OrderProcessor.process_market_order(
                signal, trade_pair, order_uuid, now_ms,
                miner_hotkey, miner_repo_version,
                market_order_manager
            )

            # Raise exception on error (consistent with validator.py:654)
            if err_msg:
                raise SignalException(err_msg)

            # Create bracket order(s) if market order succeeded and position is open
            if created_order and created_order.bracket_orders:
                if updated_position and not updated_position.is_closed_position:
                    try:
                        limit_order_client.create_sltp_order(miner_hotkey, created_order)
                    except BracketOrderException as e:
                        raise SignalException(
                            f"Market order filled successfully, but bracket order creation failed: {e}"
                        )

            return OrderProcessingResult(
                execution_type=ExecutionType.MARKET,
                order=created_order,
                updated_position=updated_position,
                should_track_uuid=True
            )
