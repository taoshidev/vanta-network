# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Comprehensive unit tests for OrderProcessor.
Tests all production code paths to ensure high confidence in production releases.
"""
import unittest
from unittest.mock import Mock
import uuid

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.limit_order.order_processor import OrderProcessor
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource


class TestOrderProcessor(TestBase):
    """
    Comprehensive tests for OrderProcessor static methods.
    Tests cover all production code paths including validation, error handling, and edge cases.
    """

    # Test constants
    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_TRADE_PAIR = TradePair.BTCUSD
    DEFAULT_NOW_MS = 1700000000000

    # ============================================================================
    # Test: parse_signal_data
    # ============================================================================

    def test_parse_signal_data_valid_signal_with_all_fields(self):
        """Test parsing valid signal with all required fields"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "LIMIT",
        }
        miner_order_uuid = str(uuid.uuid4())

        trade_pair, execution_type, order_uuid = OrderProcessor.parse_signal_data(
            signal, miner_order_uuid
        )

        self.assertEqual(trade_pair, TradePair.BTCUSD)
        self.assertEqual(execution_type, ExecutionType.LIMIT)
        self.assertEqual(order_uuid, miner_order_uuid)

    def test_parse_signal_data_generates_uuid_when_not_provided(self):
        """Test UUID generation when miner_order_uuid not provided"""
        signal = {
            "trade_pair": {"trade_pair_id": "ETHUSD"},
            "execution_type": "MARKET",
        }

        trade_pair, execution_type, order_uuid = OrderProcessor.parse_signal_data(signal)

        self.assertEqual(trade_pair, TradePair.ETHUSD)
        self.assertEqual(execution_type, ExecutionType.MARKET)
        self.assertIsNotNone(order_uuid)
        self.assertIsInstance(order_uuid, str)
        # Verify it's a valid UUID format
        uuid.UUID(order_uuid)

    def test_parse_signal_data_defaults_to_market_execution(self):
        """Test execution_type defaults to MARKET when not specified"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
        }

        trade_pair, execution_type, order_uuid = OrderProcessor.parse_signal_data(signal)

        self.assertEqual(execution_type, ExecutionType.MARKET)

    def test_parse_signal_data_case_insensitive_execution_type(self):
        """Test execution_type parsing is case insensitive"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "limit",  # lowercase
        }

        trade_pair, execution_type, order_uuid = OrderProcessor.parse_signal_data(signal)

        self.assertEqual(execution_type, ExecutionType.LIMIT)

    def test_parse_signal_data_invalid_trade_pair(self):
        """Test error handling for invalid trade pair"""
        signal = {
            "trade_pair": "INVALID_PAIR",
        }

        with self.assertRaises(SignalException) as context:
            OrderProcessor.parse_signal_data(signal)

        self.assertIn("Invalid trade pair", str(context.exception))

    def test_parse_signal_data_missing_trade_pair(self):
        """Test error handling for missing trade pair"""
        signal = {}

        with self.assertRaises(SignalException) as context:
            OrderProcessor.parse_signal_data(signal)

        self.assertIn("Invalid trade pair", str(context.exception))

    def test_parse_signal_data_invalid_execution_type(self):
        """Test error handling for invalid execution_type"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "INVALID_TYPE",
        }

        with self.assertRaises(SignalException) as context:
            OrderProcessor.parse_signal_data(signal)

        self.assertIn("Invalid execution_type", str(context.exception))

    # ============================================================================
    # Test: process_limit_order - Valid Orders
    # ============================================================================

    def test_process_limit_order_valid_long_order(self):
        """Test processing valid LONG limit order"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, OrderType.LONG)
        self.assertEqual(order.leverage, 1.0)
        self.assertEqual(order.limit_price, 50000.0)
        self.assertEqual(order.execution_type, ExecutionType.LIMIT)
        self.assertEqual(order.src, OrderSource.LIMIT_UNFILLED)
        mock_limit_order_client.process_limit_order.assert_called_once()

    def test_process_limit_order_valid_short_order(self):
        """Test processing valid SHORT limit order"""
        signal = {
            "order_type": "SHORT",
            "leverage": 0.5,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertIsNotNone(order)
        self.assertEqual(order.order_type, OrderType.SHORT)
        # SHORT orders have negative leverage internally
        self.assertEqual(order.leverage, -0.5)

    def test_process_limit_order_with_stop_loss_long(self):
        """Test LONG limit order with valid stop loss"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "stop_loss": 49000.0,  # Below limit_price for LONG
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertEqual(order.stop_loss, 49000.0)

    def test_process_limit_order_with_stop_loss_short(self):
        """Test SHORT limit order with valid stop loss"""
        signal = {
            "order_type": "SHORT",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "stop_loss": 51000.0,  # Above limit_price for SHORT
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertEqual(order.stop_loss, 51000.0)

    def test_process_limit_order_with_take_profit_long(self):
        """Test LONG limit order with valid take profit"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "take_profit": 52000.0,  # Above limit_price for LONG
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertEqual(order.take_profit, 52000.0)

    def test_process_limit_order_with_take_profit_short(self):
        """Test SHORT limit order with valid take profit"""
        signal = {
            "order_type": "SHORT",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "take_profit": 48000.0,  # Below limit_price for SHORT
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertEqual(order.take_profit, 48000.0)

    def test_process_limit_order_with_both_sl_and_tp(self):
        """Test limit order with both stop loss and take profit"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertEqual(order.stop_loss, 49000.0)
        self.assertEqual(order.take_profit, 52000.0)

    def test_process_limit_order_without_sl_and_tp(self):
        """Test limit order without stop loss or take profit"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        self.assertIsNone(order.stop_loss)
        self.assertIsNone(order.take_profit)

    # ============================================================================
    # Test: process_limit_order - Missing Required Fields
    # ============================================================================

    def test_process_limit_order_missing_leverage(self):
        """Test error handling for missing leverage"""
        signal = {
            "order_type": "LONG",
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("leverage", str(context.exception))

    def test_process_limit_order_missing_order_type(self):
        """Test error handling for missing order_type"""
        signal = {
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("order_type", str(context.exception))

    def test_process_limit_order_missing_limit_price(self):
        """Test error handling for missing limit_price"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("limit_price", str(context.exception))

    # ============================================================================
    # Test: process_limit_order - Invalid Field Values
    # ============================================================================

    def test_process_limit_order_invalid_order_type(self):
        """Test error handling for invalid order_type"""
        signal = {
            "order_type": "INVALID",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("Invalid order_type", str(context.exception))

    def test_process_limit_order_invalid_stop_loss_zero(self):
        """Test error handling for zero stop_loss"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "stop_loss": 0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("stop_loss must be greater than 0", str(context.exception))

    def test_process_limit_order_invalid_stop_loss_negative(self):
        """Test error handling for negative stop_loss"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "stop_loss": -100.0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("stop_loss must be greater than 0", str(context.exception))

    def test_process_limit_order_invalid_take_profit_zero(self):
        """Test error handling for zero take_profit"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "take_profit": 0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("take_profit must be greater than 0", str(context.exception))

    def test_process_limit_order_invalid_take_profit_negative(self):
        """Test error handling for negative take_profit"""
        signal = {
            "order_type": "SHORT",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "take_profit": -100.0,
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("take_profit must be greater than 0", str(context.exception))

    # ============================================================================
    # Test: process_limit_order - Stop Loss/Take Profit Validation
    # ============================================================================

    def test_process_limit_order_long_stop_loss_above_limit_price(self):
        """Test error for LONG order with stop_loss >= limit_price"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "stop_loss": 50000.0,  # Equal to limit_price
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("stop_loss", str(context.exception))
        self.assertIn("less than limit_price", str(context.exception))

    def test_process_limit_order_short_stop_loss_below_limit_price(self):
        """Test error for SHORT order with stop_loss <= limit_price"""
        signal = {
            "order_type": "SHORT",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "stop_loss": 50000.0,  # Equal to limit_price
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("stop_loss", str(context.exception))
        self.assertIn("greater than limit_price", str(context.exception))

    def test_process_limit_order_long_take_profit_below_limit_price(self):
        """Test error for LONG order with take_profit <= limit_price"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "take_profit": 49000.0,  # Below limit_price
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("take_profit", str(context.exception))
        self.assertIn("greater than limit_price", str(context.exception))

    def test_process_limit_order_short_take_profit_above_limit_price(self):
        """Test error for SHORT order with take_profit >= limit_price"""
        signal = {
            "order_type": "SHORT",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "take_profit": 51000.0,  # Above limit_price
        }

        mock_limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("take_profit", str(context.exception))
        self.assertIn("less than limit_price", str(context.exception))

    # ============================================================================
    # Test: process_limit_order - Manager Integration
    # ============================================================================

    def test_process_limit_order_calls_manager_with_correct_order(self):
        """Test that process_limit_order calls manager with correct Order object"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.5,
            "limit_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()

        OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        # Verify manager was called with correct arguments
        call_args = mock_limit_order_client.process_limit_order.call_args
        self.assertEqual(call_args[0][0], self.DEFAULT_MINER_HOTKEY)

        order_arg = call_args[0][1]
        self.assertIsInstance(order_arg, Order)
        self.assertEqual(order_arg.order_type, OrderType.LONG)
        self.assertEqual(order_arg.leverage, 1.5)
        self.assertEqual(order_arg.limit_price, 50000.0)
        self.assertEqual(order_arg.stop_loss, 49000.0)
        self.assertEqual(order_arg.take_profit, 52000.0)

    def test_process_limit_order_client_raises_exception(self):
        """Test that exceptions from manager are propagated"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock(
            side_effect=SignalException("Manager error")
        )

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=mock_limit_order_client
            )

        self.assertIn("Manager error", str(context.exception))

    # ============================================================================
    # Test: process_limit_cancel
    # ============================================================================

    def test_process_limit_cancel_specific_order(self):
        """Test cancelling a specific limit order"""
        signal = {}
        order_uuid = "test_order_uuid"

        mock_limit_order_client = Mock()
        mock_limit_order_client.cancel_limit_order = Mock(
            return_value={"status": "cancelled"}
        )

        result = OrderProcessor.process_limit_cancel(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid=order_uuid,
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=mock_limit_order_client
        )

        mock_limit_order_client.cancel_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY,
            # self.DEFAULT_TRADE_PAIR.trade_pair_id, TODO support cancel by trade pair in v2
            None,
            order_uuid,
            self.DEFAULT_NOW_MS
        )
        self.assertEqual(result, {"status": "cancelled"})

    # TODO support cancel by trade pair in v2
    # def test_process_limit_cancel_all_orders(self):
    #     """Test cancelling all limit orders (empty uuid)"""
    #     signal = {}
    #     order_uuid = ""

    #     limit_order_client = Mock()
    #     limit_order_client.cancel_limit_order = Mock(
    #         return_value={"status": "all_cancelled", "count": 3}
    #     )

    #     result = OrderProcessor.process_limit_cancel(
    #         signal=signal,
    #         trade_pair=self.DEFAULT_TRADE_PAIR,
    #         order_uuid=order_uuid,
    #         now_ms=self.DEFAULT_NOW_MS,
    #         miner_hotkey=self.DEFAULT_MINER_HOTKEY,
    #         limit_order_client=limit_order_client
    #     )

    #     limit_order_client.cancel_limit_order.assert_called_once_with(
    #         self.DEFAULT_MINER_HOTKEY,
    #         self.DEFAULT_TRADE_PAIR.trade_pair_id,
    #         order_uuid,
    #         self.DEFAULT_NOW_MS
    #     )
    #     self.assertEqual(result["status"], "all_cancelled")
    #     self.assertEqual(result["count"], 3)

    # def test_process_limit_cancel_none_uuid(self):
    #     """Test cancelling with None uuid (cancel all)"""
    #     signal = {}
    #     order_uuid = None

    #     mock_limit_order_client = Mock()
    #     mock_limit_order_client.cancel_limit_order = Mock(
    #         return_value={"status": "all_cancelled"}
    #     )

    #     result = OrderProcessor.process_limit_cancel(
    #         signal=signal,
    #         trade_pair=self.DEFAULT_TRADE_PAIR,
    #         order_uuid=order_uuid,
    #         now_ms=self.DEFAULT_NOW_MS,
    #         miner_hotkey=self.DEFAULT_MINER_HOTKEY,
    #         limit_order_client=mock_limit_order_client
    #     )

    #     mock_limit_order_client.cancel_limit_order.assert_called_once_with(
    #         self.DEFAULT_MINER_HOTKEY,
    #         self.DEFAULT_TRADE_PAIR.trade_pair_id,
    #         order_uuid,
    #         self.DEFAULT_NOW_MS
    #     )

    def test_process_limit_cancel_manager_raises_exception(self):
        """Test that exceptions from cancel are propagated"""
        signal = {}
        order_uuid = "test_uuid"

        limit_order_client = Mock()
        limit_order_client.cancel_limit_order = Mock(
            side_effect=SignalException("Order not found")
        )

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_limit_cancel(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid=order_uuid,
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=limit_order_client
            )

        self.assertIn("Order not found", str(context.exception))

    # ============================================================================
    # Test: process_bracket_order - Valid Orders
    # ============================================================================

    def test_process_bracket_order_with_both_sl_and_tp(self):
        """Test bracket order with both stop loss and take profit"""
        signal = {
            "leverage": 1.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        self.assertIsNotNone(order)
        self.assertEqual(order.execution_type, ExecutionType.BRACKET)
        self.assertEqual(order.stop_loss, 49000.0)
        self.assertEqual(order.take_profit, 52000.0)
        self.assertEqual(order.leverage, 1.0)
        self.assertEqual(order.src, OrderSource.BRACKET_UNFILLED)
        self.assertIsNone(order.limit_price)
        limit_order_client.process_limit_order.assert_called_once()

    def test_process_bracket_order_with_only_stop_loss(self):
        """Test bracket order with only stop loss"""
        signal = {
            "leverage": 0.5,
            "stop_loss": 49000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        self.assertEqual(order.stop_loss, 49000.0)
        self.assertIsNone(order.take_profit)

    def test_process_bracket_order_with_only_take_profit(self):
        """Test bracket order with only take profit"""
        signal = {
            "leverage": 1.5,
            "take_profit": 52000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        self.assertIsNone(order.stop_loss)
        self.assertEqual(order.take_profit, 52000.0)

    def test_process_bracket_order_leverage_defaults_to_none(self):
        """Test bracket order with no leverage defaults to None"""
        signal = {
            "stop_loss": 49000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        # Leverage should be None when not provided (will be determined by manager)
        self.assertIsNone(order.leverage)

    # ============================================================================
    # Test: process_bracket_order - Validation Errors
    # ============================================================================

    def test_process_bracket_order_missing_both_sl_and_tp(self):
        """Test error for bracket order without stop loss or take profit"""
        signal = {
            "leverage": 1.0,
        }

        limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_bracket_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=limit_order_client
            )

        self.assertIn("must specify at least one", str(context.exception))

    def test_process_bracket_order_invalid_stop_loss_zero(self):
        """Test error for bracket order with zero stop_loss"""
        signal = {
            "leverage": 1.0,
            "stop_loss": 0,
        }

        limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_bracket_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=limit_order_client
            )

        self.assertIn("stop_loss must be greater than 0", str(context.exception))

    def test_process_bracket_order_invalid_stop_loss_negative(self):
        """Test error for bracket order with negative stop_loss"""
        signal = {
            "leverage": 1.0,
            "stop_loss": -100.0,
        }

        limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_bracket_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=limit_order_client
            )

        self.assertIn("stop_loss must be greater than 0", str(context.exception))

    def test_process_bracket_order_invalid_take_profit_zero(self):
        """Test error for bracket order with zero take_profit"""
        signal = {
            "leverage": 1.0,
            "take_profit": 0,
        }

        limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_bracket_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=limit_order_client
            )

        self.assertIn("take_profit must be greater than 0", str(context.exception))

    def test_process_bracket_order_invalid_take_profit_negative(self):
        """Test error for bracket order with negative take_profit"""
        signal = {
            "leverage": 1.0,
            "take_profit": -100.0,
        }

        limit_order_client = Mock()

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_bracket_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=limit_order_client
            )

        self.assertIn("take_profit must be greater than 0", str(context.exception))

    # ============================================================================
    # Test: process_bracket_order - Manager Integration
    # ============================================================================

    def test_process_bracket_order_calls_manager_with_correct_order(self):
        """Test that process_bracket_order calls manager with correct Order object"""
        signal = {
            "leverage": 1.5,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        # Verify manager was called with correct arguments
        call_args = limit_order_client.process_limit_order.call_args
        self.assertEqual(call_args[0][0], self.DEFAULT_MINER_HOTKEY)

        order_arg = call_args[0][1]
        self.assertIsInstance(order_arg, Order)
        self.assertEqual(order_arg.execution_type, ExecutionType.BRACKET)
        self.assertEqual(order_arg.leverage, 1.5)
        self.assertEqual(order_arg.stop_loss, 49000.0)
        self.assertEqual(order_arg.take_profit, 52000.0)
        self.assertIsNone(order_arg.limit_price)

    def test_process_bracket_order_manager_raises_exception(self):
        """Test that exceptions from manager are propagated"""
        signal = {
            "stop_loss": 49000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock(
            side_effect=SignalException("No position found")
        )

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_bracket_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                limit_order_client=limit_order_client
            )

        self.assertIn("No position found", str(context.exception))

    # ============================================================================
    # Test: process_market_order
    # ============================================================================

    def test_process_market_order_success(self):
        """Test processing successful market order"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
        }

        mock_market_order_manager = Mock()
        mock_position = Mock()
        mock_order = Mock()
        mock_market_order_manager._process_market_order = Mock(
            return_value=("", mock_position, mock_order)
        )

        err_msg, position, order = OrderProcessor.process_market_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            market_order_manager=mock_market_order_manager
        )

        self.assertEqual(err_msg, "")
        self.assertIsNotNone(position)
        self.assertIsNotNone(order)

        # Verify manager was called correctly
        mock_market_order_manager._process_market_order.assert_called_once_with(
            "test_uuid",
            "1.0.0",
            self.DEFAULT_TRADE_PAIR,
            self.DEFAULT_NOW_MS,
            signal,
            self.DEFAULT_MINER_HOTKEY,
            price_sources=None
        )

    def test_process_market_order_with_error(self):
        """Test processing market order that returns error"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
        }

        mock_market_order_manager = Mock()
        mock_market_order_manager._process_market_order = Mock(
            return_value=("Order too soon", None, None)
        )

        err_msg, position, order = OrderProcessor.process_market_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            market_order_manager=mock_market_order_manager
        )

        self.assertEqual(err_msg, "Order too soon")
        self.assertIsNone(position)
        self.assertIsNone(order)

    def test_process_market_order_manager_raises_exception(self):
        """Test that exceptions from manager are propagated"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
        }

        mock_market_order_manager = Mock()
        mock_market_order_manager._process_market_order = Mock(
            side_effect=SignalException("Invalid signal")
        )

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_market_order(
                signal=signal,
                trade_pair=self.DEFAULT_TRADE_PAIR,
                order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                miner_repo_version="1.0.0",
                market_order_manager=mock_market_order_manager
            )

        self.assertIn("Invalid signal", str(context.exception))

    # ============================================================================
    # Test: Edge Cases and Data Type Conversions
    # ============================================================================

    def test_process_limit_order_converts_string_numbers_to_float(self):
        """Test that string numbers are properly converted to float"""
        signal = {
            "order_type": "LONG",
            "leverage": "1.5",  # String
            "limit_price": "50000.0",  # String
            "stop_loss": "49000.0",  # String
            "take_profit": "52000.0",  # String
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        # Verify all values are floats
        self.assertIsInstance(order.leverage, float)
        self.assertIsInstance(order.limit_price, float)
        self.assertIsInstance(order.stop_loss, float)
        self.assertIsInstance(order.take_profit, float)

    def test_process_bracket_order_converts_string_numbers_to_float(self):
        """Test that string numbers are properly converted to float in bracket orders"""
        signal = {
            "leverage": "1.5",  # String
            "stop_loss": "49000.0",  # String
            "take_profit": "52000.0",  # String
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        order = OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        # Verify all values are floats
        self.assertIsInstance(order.leverage, float)
        self.assertIsInstance(order.stop_loss, float)
        self.assertIsInstance(order.take_profit, float)

    def test_parse_signal_data_multiple_trade_pairs(self):
        """Test parsing signals for different trade pairs"""
        test_pairs = [
            ("BTCUSD", TradePair.BTCUSD),
            ("ETHUSD", TradePair.ETHUSD),
            ("EURUSD", TradePair.EURUSD),
        ]

        for pair_str, expected_pair in test_pairs:
            signal = {"trade_pair": {"trade_pair_id": pair_str}}
            trade_pair, _, _ = OrderProcessor.parse_signal_data(signal)
            self.assertEqual(trade_pair, expected_pair)

    def test_process_limit_order_order_uuid_propagated(self):
        """Test that order_uuid is correctly set in the created order"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        test_uuid = "custom-uuid-12345"
        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid=test_uuid,
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        self.assertEqual(order.order_uuid, test_uuid)

    def test_process_bracket_order_order_uuid_propagated(self):
        """Test that order_uuid is correctly set in bracket orders"""
        signal = {
            "stop_loss": 49000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        test_uuid = "bracket-uuid-67890"
        order = OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid=test_uuid,
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        self.assertEqual(order.order_uuid, test_uuid)

    def test_process_limit_order_timestamp_propagated(self):
        """Test that processed_ms timestamp is correctly set"""
        signal = {
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        custom_timestamp = 1234567890000
        order = OrderProcessor.process_limit_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=custom_timestamp,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        self.assertEqual(order.processed_ms, custom_timestamp)

    def test_process_bracket_order_timestamp_propagated(self):
        """Test that processed_ms timestamp is correctly set in bracket orders"""
        signal = {
            "stop_loss": 49000.0,
        }

        limit_order_client = Mock()
        limit_order_client.process_limit_order = Mock()

        custom_timestamp = 1234567890000
        order = OrderProcessor.process_bracket_order(
            signal=signal,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_uuid="test_uuid",
            now_ms=custom_timestamp,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            limit_order_client=limit_order_client
        )

        self.assertEqual(order.processed_ms, custom_timestamp)

    # ============================================================================
    # Test: process_order (Unified Dispatcher)
    # ============================================================================

    def test_process_order_routes_to_limit_order(self):
        """Test that process_order correctly routes LIMIT execution type"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "LIMIT",
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()
        mock_market_order_manager = Mock()

        result = OrderProcessor.process_order(
            signal=signal,
            miner_order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            limit_order_client=mock_limit_order_client,
            market_order_manager=mock_market_order_manager
        )

        # Verify result
        self.assertEqual(result.execution_type, ExecutionType.LIMIT)
        self.assertIsNotNone(result.order)
        self.assertTrue(result.should_track_uuid)
        self.assertTrue(result.success)
        self.assertIsNone(result.result_dict)

        # Verify limit order client was called
        mock_limit_order_client.process_limit_order.assert_called_once()
        # Verify market order manager was NOT called
        mock_market_order_manager._process_market_order.assert_not_called()

    def test_process_order_routes_to_bracket_order(self):
        """Test that process_order correctly routes BRACKET execution type"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "BRACKET",
            "leverage": 1.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()
        mock_market_order_manager = Mock()

        result = OrderProcessor.process_order(
            signal=signal,
            miner_order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            limit_order_client=mock_limit_order_client,
            market_order_manager=mock_market_order_manager
        )

        # Verify result
        self.assertEqual(result.execution_type, ExecutionType.BRACKET)
        self.assertIsNotNone(result.order)
        self.assertTrue(result.should_track_uuid)
        self.assertTrue(result.success)

        # Verify limit order client was called
        mock_limit_order_client.process_limit_order.assert_called_once()

    def test_process_order_routes_to_limit_cancel(self):
        """Test that process_order correctly routes LIMIT_CANCEL execution type"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "LIMIT_CANCEL",
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.cancel_limit_order = Mock(
            return_value={"status": "cancelled", "count": 2}
        )
        mock_market_order_manager = Mock()

        result = OrderProcessor.process_order(
            signal=signal,
            miner_order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            limit_order_client=mock_limit_order_client,
            market_order_manager=mock_market_order_manager
        )

        # Verify result
        self.assertEqual(result.execution_type, ExecutionType.LIMIT_CANCEL)
        self.assertIsNone(result.order)
        self.assertFalse(result.should_track_uuid)  # LIMIT_CANCEL doesn't track UUID
        self.assertTrue(result.success)
        self.assertIsNotNone(result.result_dict)
        self.assertEqual(result.result_dict["status"], "cancelled")

        # Verify cancel was called
        mock_limit_order_client.cancel_limit_order.assert_called_once()

    def test_process_order_routes_to_market_order(self):
        """Test that process_order correctly routes MARKET execution type"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "MARKET",
            "order_type": "LONG",
            "leverage": 1.0,
        }

        mock_limit_order_client = Mock()
        mock_market_order_manager = Mock()
        mock_position = Mock()
        mock_order = Mock()
        mock_market_order_manager._process_market_order = Mock(
            return_value=("", mock_position, mock_order)
        )

        result = OrderProcessor.process_order(
            signal=signal,
            miner_order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            limit_order_client=mock_limit_order_client,
            market_order_manager=mock_market_order_manager
        )

        # Verify result
        self.assertEqual(result.execution_type, ExecutionType.MARKET)
        self.assertIsNotNone(result.order)
        self.assertIsNotNone(result.updated_position)
        self.assertTrue(result.should_track_uuid)
        self.assertTrue(result.success)

        # Verify market order manager was called
        mock_market_order_manager._process_market_order.assert_called_once()
        # Verify limit order client was NOT called
        mock_limit_order_client.process_limit_order.assert_not_called()

    def test_process_order_market_raises_exception_on_error(self):
        """Test that process_order raises SignalException for MARKET order errors"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "MARKET",
            "order_type": "LONG",
            "leverage": 1.0,
        }

        mock_limit_order_client = Mock()
        mock_market_order_manager = Mock()
        mock_market_order_manager._process_market_order = Mock(
            return_value=("Order too soon", None, None)
        )

        with self.assertRaises(SignalException) as context:
            OrderProcessor.process_order(
                signal=signal,
                miner_order_uuid="test_uuid",
                now_ms=self.DEFAULT_NOW_MS,
                miner_hotkey=self.DEFAULT_MINER_HOTKEY,
                miner_repo_version="1.0.0",
                limit_order_client=mock_limit_order_client,
                market_order_manager=mock_market_order_manager
            )

        self.assertIn("Order too soon", str(context.exception))

    def test_process_order_defaults_to_market_execution(self):
        """Test that process_order defaults to MARKET when execution_type not specified"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "order_type": "LONG",
            "leverage": 1.0,
        }

        mock_limit_order_client = Mock()
        mock_market_order_manager = Mock()
        mock_position = Mock()
        mock_order = Mock()
        mock_market_order_manager._process_market_order = Mock(
            return_value=("", mock_position, mock_order)
        )

        result = OrderProcessor.process_order(
            signal=signal,
            miner_order_uuid="test_uuid",
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            limit_order_client=mock_limit_order_client,
            market_order_manager=mock_market_order_manager
        )

        # Verify it routed to MARKET
        self.assertEqual(result.execution_type, ExecutionType.MARKET)
        mock_market_order_manager._process_market_order.assert_called_once()

    def test_process_order_generates_uuid_when_not_provided(self):
        """Test that process_order generates UUID when miner_order_uuid is None"""
        signal = {
            "trade_pair": {"trade_pair_id": "BTCUSD"},
            "execution_type": "LIMIT",
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
        }

        mock_limit_order_client = Mock()
        mock_limit_order_client.process_limit_order = Mock()
        mock_market_order_manager = Mock()

        result = OrderProcessor.process_order(
            signal=signal,
            miner_order_uuid=None,  # No UUID provided
            now_ms=self.DEFAULT_NOW_MS,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            miner_repo_version="1.0.0",
            limit_order_client=mock_limit_order_client,
            market_order_manager=mock_market_order_manager
        )

        # Verify UUID was generated
        self.assertIsNotNone(result.order.order_uuid)
        # Verify it's a valid UUID format
        uuid.UUID(result.order.order_uuid)

    # ============================================================================
    # Test: OrderProcessingResult
    # ============================================================================

    def test_order_processing_result_get_response_json_with_order(self):
        """Test get_response_json returns order JSON when order is present"""
        from vali_objects.utils.limit_order.order_processor import OrderProcessingResult

        mock_order = Mock()
        mock_order.__str__ = Mock(return_value='{"order": "data"}')

        result = OrderProcessingResult(
            execution_type=ExecutionType.LIMIT,
            order=mock_order
        )

        response_json = result.get_response_json()
        self.assertEqual(response_json, '{"order": "data"}')

    def test_order_processing_result_get_response_json_with_result_dict(self):
        """Test get_response_json returns JSON dict when result_dict is present"""
        from vali_objects.utils.limit_order.order_processor import OrderProcessingResult

        result_dict = {"status": "cancelled", "count": 3}

        result = OrderProcessingResult(
            execution_type=ExecutionType.LIMIT_CANCEL,
            result_dict=result_dict,
            should_track_uuid=False
        )

        response_json = result.get_response_json()
        import json
        parsed = json.loads(response_json)
        self.assertEqual(parsed["status"], "cancelled")
        self.assertEqual(parsed["count"], 3)

    def test_order_processing_result_get_response_json_empty(self):
        """Test get_response_json returns empty string when no data"""
        from vali_objects.utils.limit_order.order_processor import OrderProcessingResult

        result = OrderProcessingResult(
            execution_type=ExecutionType.LIMIT
        )

        response_json = result.get_response_json()
        self.assertEqual(response_json, "")

    def test_order_processing_result_order_for_logging(self):
        """Test order_for_logging property returns order"""
        from vali_objects.utils.limit_order.order_processor import OrderProcessingResult

        mock_order = Mock()

        result = OrderProcessingResult(
            execution_type=ExecutionType.LIMIT,
            order=mock_order
        )

        self.assertEqual(result.order_for_logging, mock_order)

    def test_order_processing_result_is_frozen(self):
        """Test that OrderProcessingResult is immutable (frozen dataclass)"""
        from vali_objects.utils.limit_order.order_processor import OrderProcessingResult

        result = OrderProcessingResult(
            execution_type=ExecutionType.LIMIT
        )

        # Attempting to modify a frozen dataclass should raise an error
        with self.assertRaises(Exception):  # FrozenInstanceError in Python 3.10+
            result.success = False


if __name__ == '__main__':
    unittest.main()
