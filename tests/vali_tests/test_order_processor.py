# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
"""
Unit tests for OrderProcessor, covering the current implementation in
vali_objects/utils/order_processor.py: validate(), process_vanta_signal()
dispatch, market/flat/hyperliquid execution, unfilled (LIMIT/BRACKET/
STOP_LIMIT) order creation, limit cancel, and limit edit (including the
bulk bracket-update path).
"""
import unittest
from unittest.mock import Mock, PropertyMock, patch
import uuid

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.order_processor import OrderProcessor, OrderProcessingResult
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order_signal import Signal


class OrderProcessorTestBase(TestBase):
    DEFAULT_MINER_HOTKEY = "test_miner"
    # BTCUSD is a blocked native pair (remapped to BTCUSDC); use BTCUSDC directly
    # as the default "already resolved" trade pair for most tests.
    DEFAULT_TRADE_PAIR = TradePair.BTCUSDC
    DEFAULT_NOW_MS = 1700000000000

    def setUp(self) -> None:
        super().setUp()
        self.limit_order_client = Mock()
        self.market_order_client = Mock()
        self.miner_account_client = Mock()
        self.processor = OrderProcessor(
            limit_order_client=self.limit_order_client,
            market_order_client=self.market_order_client,
            miner_account_client=self.miner_account_client,
        )

    def make_miner_account(self, bucket=MinerBucket.MAINCOMP, asset_class=MinerAssetClass.HL_ALL):
        account = Mock()
        account.miner_bucket = bucket
        account.asset_class = asset_class
        return account


class TestValidate(OrderProcessorTestBase):

    def test_native_crypto_remap(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, TradePair.BTCUSD, OrderType.LONG
        )
        self.assertTrue(ok)
        self.assertEqual(resolved, TradePair.BTCUSDC)

    def test_missing_trade_pair_for_trading_execution_type(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, None, OrderType.LONG
        )
        self.assertFalse(ok)
        self.assertIn("Invalid trade pair", msg)
        self.assertIsNone(resolved)

    def test_blocked_trade_pair(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, TradePair.XAUUSD, OrderType.LONG
        )
        self.assertFalse(ok)
        self.assertIn("no longer supported", msg)
        self.assertEqual(resolved, TradePair.XAUUSD)

    def test_flat_only_blocks_non_flat_orders(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        with patch.object(TradePair, 'is_flat_only', new_callable=PropertyMock) as mock_flat_only:
            mock_flat_only.return_value = True
            ok, msg, resolved = self.processor.validate(
                self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, self.DEFAULT_TRADE_PAIR, OrderType.LONG
            )
        self.assertFalse(ok)
        self.assertIn("being discontinued", msg)

    def test_flat_only_allows_flat_orders(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        with patch.object(TradePair, 'is_flat_only', new_callable=PropertyMock) as mock_flat_only:
            mock_flat_only.return_value = True
            ok, msg, resolved = self.processor.validate(
                self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, self.DEFAULT_TRADE_PAIR, OrderType.FLAT
            )
        self.assertTrue(ok)

    def test_uninitialized_miner_account(self):
        self.miner_account_client.get_account.return_value = None
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, self.DEFAULT_TRADE_PAIR, OrderType.LONG
        )
        self.assertFalse(ok)
        self.assertIn("not yet initialized", msg)

    def test_eliminated_miner(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account(bucket=MinerBucket.ELIMINATED)
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, self.DEFAULT_TRADE_PAIR, OrderType.LONG
        )
        self.assertFalse(ok)
        self.assertIn("eliminated", msg)
        self.assertIsNone(resolved)

    def test_entity_hotkey_cannot_place_orders(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account(bucket=MinerBucket.ENTITY)
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, self.DEFAULT_TRADE_PAIR, OrderType.LONG
        )
        self.assertFalse(ok)
        self.assertIn("cannot place orders directly", msg)
        self.assertIsNone(resolved)

    def test_missing_asset_class_blocks_limit_orders(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account(asset_class=None)
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.LIMIT, self.DEFAULT_TRADE_PAIR, OrderType.LONG
        )
        self.assertFalse(ok)
        self.assertIn("No asset class selected", msg)

    def test_asset_class_cannot_trade_pair(self):
        account = self.make_miner_account()
        account.asset_class = Mock()
        account.asset_class.can_trade.return_value = False
        self.miner_account_client.get_account.return_value = account
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.LIMIT, self.DEFAULT_TRADE_PAIR, OrderType.LONG
        )
        self.assertFalse(ok)
        self.assertIn("cannot submit orders for trade pair", msg)

    def test_asset_class_check_bypassed_for_market_orders(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account(asset_class=None)
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.MARKET, self.DEFAULT_TRADE_PAIR, OrderType.LONG
        )
        self.assertTrue(ok)

    def test_asset_class_check_bypassed_for_flat_order_type(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account(asset_class=None)
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.LIMIT, self.DEFAULT_TRADE_PAIR, OrderType.FLAT
        )
        self.assertTrue(ok)

    def test_trade_pair_check_bypassed_for_limit_cancel(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.LIMIT_CANCEL, None, None
        )
        self.assertTrue(ok)

    def test_trade_pair_check_bypassed_for_limit_edit(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.LIMIT_EDIT, None, None
        )
        self.assertTrue(ok)

    def test_trade_pair_check_bypassed_for_flat_all(self):
        self.miner_account_client.get_account.return_value = self.make_miner_account()
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.FLAT_ALL, None, None
        )
        self.assertTrue(ok)

    def test_success_full_path(self):
        account = self.make_miner_account()
        account.asset_class = Mock()
        account.asset_class.can_trade.return_value = True
        self.miner_account_client.get_account.return_value = account
        ok, msg, resolved = self.processor.validate(
            self.DEFAULT_MINER_HOTKEY, ExecutionType.LIMIT, self.DEFAULT_TRADE_PAIR, OrderType.LONG
        )
        self.assertTrue(ok)
        self.assertEqual(msg, "")
        self.assertEqual(resolved, self.DEFAULT_TRADE_PAIR)


class TestProcessVantaSignalDispatch(OrderProcessorTestBase):

    def test_routes_to_market_order(self):
        self.market_order_client.execute_order.return_value = (Mock(bracket_orders=None), Mock(is_closed_position=False))
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.MARKET,
                         order_type=OrderType.LONG, leverage=1.0)
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.MARKET)
        self.market_order_client.execute_order.assert_called_once()

    def test_routes_to_flat_all(self):
        signal = Signal(execution_type=ExecutionType.FLAT_ALL)
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "ALL", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.FLAT_ALL)
        self.market_order_client.close_positions.assert_called_once()

    def test_routes_to_limit_order(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT,
                         order_type=OrderType.LONG, leverage=1.0, limit_price=50000.0)
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.LIMIT)
        self.assertIsNotNone(result.order)
        self.assertTrue(result.should_track_uuid)
        self.limit_order_client.process_limit_order.assert_called_once()

    def test_routes_to_bracket_order(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.BRACKET,
                         stop_loss=49000.0)
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.BRACKET)
        self.assertIsNotNone(result.order)

    def test_routes_to_stop_limit_order(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.STOP_LIMIT,
                         order_type=OrderType.LONG, leverage=1.0, limit_price=50000.0,
                         stop_price=51000.0, stop_condition=StopCondition.GTE)
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.STOP_LIMIT)
        self.assertEqual(result.order.src, OrderSource.STOP_LIMIT_UNFILLED)

    def test_routes_to_limit_cancel(self):
        self.limit_order_client.cancel_limit_order.return_value = {"status": "cancelled"}
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_CANCEL)
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.LIMIT_CANCEL)
        self.assertIsNone(result.order)
        self.assertFalse(result.should_track_uuid)
        self.assertEqual(result.result_dict, {"status": "cancelled"})

    def test_routes_to_limit_edit(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = None
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         leverage=1.0, bracket_orders=[{"order_uuid": "b1", "stop_loss": 49000.0}])
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.LIMIT_EDIT)
        self.assertFalse(result.should_track_uuid)

    def test_generates_uuid_when_not_provided(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT,
                         order_type=OrderType.LONG, leverage=1.0, limit_price=50000.0)
        result = self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, None, self.DEFAULT_NOW_MS)
        self.assertIsNotNone(result.order.order_uuid)
        uuid.UUID(result.order.order_uuid)

    def test_invalid_execution_type_raises(self):
        signal = Mock()
        signal.execution_type = "NOT_A_REAL_TYPE"
        with self.assertRaises(SignalException):
            self.processor.process_vanta_signal(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)


class TestProcessMarketOrder(OrderProcessorTestBase):

    def _signal(self, **overrides):
        params = dict(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.MARKET,
                      order_type=OrderType.LONG, leverage=1.0)
        params.update(overrides)
        return Signal(**params)

    def test_success(self):
        created_order = Mock(bracket_orders=None)
        updated_position = Mock(is_closed_position=False)
        self.market_order_client.execute_order.return_value = (created_order, updated_position)

        result = self.processor.process_market_order(
            self.DEFAULT_MINER_HOTKEY, self._signal(), "uuid1", self.DEFAULT_NOW_MS
        )

        self.assertEqual(result.execution_type, ExecutionType.MARKET)
        self.assertEqual(result.order, created_order)
        self.assertEqual(result.updated_position, updated_position)
        self.assertTrue(result.should_track_uuid)

        call_kwargs = self.market_order_client.execute_order.call_args
        self.assertEqual(call_kwargs.args[0], self.DEFAULT_MINER_HOTKEY)
        self.assertEqual(call_kwargs.args[1], "uuid1")
        self.assertEqual(call_kwargs.args[2], self.DEFAULT_TRADE_PAIR)
        self.assertEqual(call_kwargs.kwargs["order_src"], OrderSource.ORGANIC)
        self.assertTrue(call_kwargs.kwargs["enforce_cooldown"])

    def test_none_result_returns_bare_result(self):
        self.market_order_client.execute_order.return_value = None
        result = self.processor.process_market_order(
            self.DEFAULT_MINER_HOTKEY, self._signal(), "uuid1", self.DEFAULT_NOW_MS
        )
        self.assertEqual(result, OrderProcessingResult(ExecutionType.MARKET))

    def test_closed_position_cancels_brackets(self):
        created_order = Mock(bracket_orders=None)
        updated_position = Mock(is_closed_position=True)
        self.market_order_client.execute_order.return_value = (created_order, updated_position)

        self.processor.process_market_order(
            self.DEFAULT_MINER_HOTKEY, self._signal(order_type=OrderType.FLAT, leverage=None), "uuid1", self.DEFAULT_NOW_MS
        )

        self.limit_order_client.cancel_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id, "ALL", self.DEFAULT_NOW_MS, ExecutionType.BRACKET
        )

    def test_open_position_with_bracket_orders_creates_sltp(self):
        created_order = Mock(bracket_orders=None)
        updated_position = Mock(is_closed_position=False)
        self.market_order_client.execute_order.return_value = (created_order, updated_position)

        signal = self._signal(stop_loss=49000.0)
        self.processor.process_market_order(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)

        self.assertEqual(created_order.bracket_orders, signal.bracket_orders)
        self.limit_order_client.create_sltp_order.assert_called_once_with(self.DEFAULT_MINER_HOTKEY, created_order)

    def test_closed_position_does_not_create_sltp(self):
        created_order = Mock(bracket_orders=None)
        updated_position = Mock(is_closed_position=True)
        self.market_order_client.execute_order.return_value = (created_order, updated_position)

        signal = self._signal(order_type=OrderType.FLAT, leverage=None, stop_loss=49000.0)
        self.processor.process_market_order(self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS)

        self.limit_order_client.create_sltp_order.assert_not_called()

    def test_no_bracket_orders_does_not_create_sltp(self):
        created_order = Mock(bracket_orders=None)
        updated_position = Mock(is_closed_position=False)
        self.market_order_client.execute_order.return_value = (created_order, updated_position)

        self.processor.process_market_order(self.DEFAULT_MINER_HOTKEY, self._signal(), "uuid1", self.DEFAULT_NOW_MS)

        self.limit_order_client.create_sltp_order.assert_not_called()

    def test_client_exception_propagates(self):
        self.market_order_client.execute_order.side_effect = SignalException("Order too soon")
        with self.assertRaises(SignalException):
            self.processor.process_market_order(self.DEFAULT_MINER_HOTKEY, self._signal(), "uuid1", self.DEFAULT_NOW_MS)
        self.limit_order_client.create_sltp_order.assert_not_called()


class TestProcessFlatAll(OrderProcessorTestBase):

    def test_close_all_when_uuid_is_all(self):
        result = self.processor.process_flat_all(self.DEFAULT_MINER_HOTKEY, "ALL", self.DEFAULT_NOW_MS)
        self.assertEqual(result.execution_type, ExecutionType.FLAT_ALL)
        self.market_order_client.close_positions.assert_called_once_with(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=None, close_all=True, now_ms=self.DEFAULT_NOW_MS
        )
        self.limit_order_client.cancel_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY, None, "ALL", self.DEFAULT_NOW_MS, ExecutionType.BRACKET
        )

    def test_close_specific_positions(self):
        self.processor.process_flat_all(self.DEFAULT_MINER_HOTKEY, "pos1,pos2", self.DEFAULT_NOW_MS)
        self.market_order_client.close_positions.assert_called_once_with(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=["pos1", "pos2"], close_all=False, now_ms=self.DEFAULT_NOW_MS
        )

    def test_empty_uuid_closes_none(self):
        self.processor.process_flat_all(self.DEFAULT_MINER_HOTKEY, "", self.DEFAULT_NOW_MS)
        self.market_order_client.close_positions.assert_called_once_with(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=[], close_all=False, now_ms=self.DEFAULT_NOW_MS
        )

    def test_case_insensitive_all(self):
        self.processor.process_flat_all(self.DEFAULT_MINER_HOTKEY, "all", self.DEFAULT_NOW_MS)
        self.market_order_client.close_positions.assert_called_once_with(
            hotkey=self.DEFAULT_MINER_HOTKEY, position_uuids=None, close_all=True, now_ms=self.DEFAULT_NOW_MS
        )


class TestProcessHyperliquidOrder(OrderProcessorTestBase):

    def test_calls_market_order_client_with_hl_defaults(self):
        from vali_objects.utils.limit_order.order_utils import OrderSize
        self.market_order_client.execute_order.return_value = (Mock(), Mock())

        self.processor.process_hyperliquid_order(
            self.DEFAULT_MINER_HOTKEY, "uuid1", self.DEFAULT_TRADE_PAIR, OrderType.LONG,
            OrderSize(leverage=1.0), fill_price=50000.0, is_taker=True, now_ms=self.DEFAULT_NOW_MS,
        )

        call = self.market_order_client.execute_order.call_args
        self.assertEqual(call.kwargs["order_src"], OrderSource.HYPERLIQUID)
        self.assertTrue(call.kwargs["is_hl"])
        self.assertTrue(call.kwargs["is_hl_taker"])
        self.assertFalse(call.kwargs["enforce_cooldown"])
        self.assertEqual(call.kwargs["slippage"], 0.0)
        self.assertEqual(call.kwargs["fill_price"], 50000.0)

    def test_missing_trade_pair_raises(self):
        from vali_objects.utils.limit_order.order_utils import OrderSize
        with self.assertRaises(SignalException):
            self.processor.process_hyperliquid_order(
                self.DEFAULT_MINER_HOTKEY, "uuid1", None, OrderType.LONG, OrderSize(leverage=1.0)
            )

    def test_multiple_sizes_raises(self):
        from vali_objects.utils.limit_order.order_utils import OrderSize
        with self.assertRaises(SignalException):
            self.processor.process_hyperliquid_order(
                self.DEFAULT_MINER_HOTKEY, "uuid1", self.DEFAULT_TRADE_PAIR, OrderType.LONG,
                OrderSize(leverage=1.0, value=100.0)
            )


class TestProcessUnfilledOrder(OrderProcessorTestBase):

    def test_limit_order_created(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT,
                         order_type=OrderType.LONG, leverage=1.0, limit_price=50000.0,
                         stop_loss=49000.0, take_profit=52000.0)

        order = self.processor.process_unfilled_order(
            self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.LIMIT
        )

        self.assertEqual(order.order_type, OrderType.LONG)
        self.assertEqual(order.leverage, 1.0)
        self.assertEqual(order.limit_price, 50000.0)
        self.assertEqual(order.src, OrderSource.LIMIT_UNFILLED)
        self.assertEqual(order.trade_pair, self.DEFAULT_TRADE_PAIR)
        self.limit_order_client.process_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY, order, is_edit=False
        )

    def test_limit_order_short_negates_leverage(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT,
                         order_type=OrderType.SHORT, leverage=0.5, limit_price=50000.0)

        order = self.processor.process_unfilled_order(
            self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.LIMIT
        )
        # Signal normalizes SHORT size fields to negative before OrderProcessor sees them.
        self.assertEqual(order.leverage, -0.5)

    def test_stop_limit_order_created(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.STOP_LIMIT,
                         order_type=OrderType.LONG, leverage=1.0, limit_price=50000.0,
                         stop_price=51000.0, stop_condition=StopCondition.GTE)

        order = self.processor.process_unfilled_order(
            self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.STOP_LIMIT
        )

        self.assertEqual(order.src, OrderSource.STOP_LIMIT_UNFILLED)
        self.assertEqual(order.stop_price, 51000.0)
        self.assertEqual(order.stop_condition, StopCondition.GTE)

    def test_missing_size_raises(self):
        signal = Mock()
        signal.leverage = None
        signal.value = None
        signal.quantity = None
        signal.order_type = OrderType.LONG
        signal.trade_pair = self.DEFAULT_TRADE_PAIR
        with self.assertRaises(SignalException):
            self.processor.process_unfilled_order(
                self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.LIMIT
            )

    def test_bracket_order_created_with_sl_and_tp(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.BRACKET,
                         leverage=1.0, stop_loss=49000.0, take_profit=52000.0)

        order = self.processor.process_unfilled_order(
            self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.BRACKET
        )

        self.assertEqual(order.execution_type, ExecutionType.BRACKET)
        self.assertEqual(order.order_type, OrderType.FLAT)
        self.assertEqual(order.stop_loss, 49000.0)
        self.assertEqual(order.take_profit, 52000.0)
        self.assertEqual(order.src, OrderSource.BRACKET_UNFILLED)
        # BRACKET orders never carry a limit_price, even if one was set upstream.
        self.assertIsNone(order.limit_price)

    def test_bracket_order_defaults_bracket_pct_to_full_when_no_size_given(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.BRACKET,
                         stop_loss=49000.0)

        order = self.processor.process_unfilled_order(
            self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.BRACKET
        )

        self.assertEqual(order.bracket_pct, 1.0)

    def test_bracket_order_keeps_explicit_size_instead_of_default_pct(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.BRACKET,
                         leverage=0.5, stop_loss=49000.0)

        order = self.processor.process_unfilled_order(
            self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.BRACKET
        )

        self.assertIsNone(order.bracket_pct)
        self.assertEqual(order.leverage, 0.5)

    def test_is_edit_propagated_to_client(self):
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT,
                         order_type=OrderType.LONG, leverage=1.0, limit_price=50000.0)

        self.processor.process_unfilled_order(
            self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.LIMIT, is_edit=True
        )

        self.limit_order_client.process_limit_order.assert_called_once()
        self.assertTrue(self.limit_order_client.process_limit_order.call_args.kwargs["is_edit"])

    def test_client_exception_propagates(self):
        self.limit_order_client.process_limit_order.side_effect = SignalException("Manager error")
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT,
                         order_type=OrderType.LONG, leverage=1.0, limit_price=50000.0)
        with self.assertRaises(SignalException):
            self.processor.process_unfilled_order(
                self.DEFAULT_MINER_HOTKEY, signal, "uuid1", self.DEFAULT_NOW_MS, ExecutionType.LIMIT
            )


class TestProcessLimitCancel(OrderProcessorTestBase):

    def test_delegates_to_client(self):
        self.limit_order_client.cancel_limit_order.return_value = {"status": "cancelled"}

        result = self.processor.process_limit_cancel(
            self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR, "order1", self.DEFAULT_NOW_MS, ExecutionType.LIMIT
        )

        self.limit_order_client.cancel_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR.trade_pair_id, "order1", self.DEFAULT_NOW_MS, ExecutionType.LIMIT
        )
        self.assertEqual(result, {"status": "cancelled"})

    def test_none_trade_pair_passes_none_id(self):
        self.processor.process_limit_cancel(self.DEFAULT_MINER_HOTKEY, None, "ALL", self.DEFAULT_NOW_MS)
        self.limit_order_client.cancel_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY, None, "ALL", self.DEFAULT_NOW_MS, None
        )

    def test_client_exception_propagates(self):
        self.limit_order_client.cancel_limit_order.side_effect = SignalException("Order not found")
        with self.assertRaises(SignalException):
            self.processor.process_limit_cancel(self.DEFAULT_MINER_HOTKEY, self.DEFAULT_TRADE_PAIR, "order1", self.DEFAULT_NOW_MS)


class TestProcessLimitEdit(OrderProcessorTestBase):

    def _existing_order_dict(self, order_uuid="order1", src=OrderSource.LIMIT_UNFILLED, execution_type=ExecutionType.LIMIT):
        return {
            "order_uuid": order_uuid,
            "trade_pair_id": self.DEFAULT_TRADE_PAIR.trade_pair_id,
            "execution_type": execution_type.value,
            "src": int(src),
            "order_type": "LONG",
            "leverage": 1.0,
            "limit_price": 50000.0,
            "processed_ms": self.DEFAULT_NOW_MS,
            "price": 0.0,
        }

    def test_edit_existing_unfilled_order(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = self._existing_order_dict()
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         order_type=OrderType.LONG, leverage=2.0, limit_price=51000.0)

        order = self.processor.process_limit_edit(self.DEFAULT_MINER_HOTKEY, signal, "order1", self.DEFAULT_NOW_MS)

        self.assertEqual(order.leverage, 2.0)
        self.assertEqual(order.limit_price, 51000.0)
        self.assertEqual(order.execution_type, ExecutionType.LIMIT)
        self.limit_order_client.process_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY, order, is_edit=True
        )

    def test_edit_rejects_already_filled_order(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = self._existing_order_dict(
            src=OrderSource.LIMIT_FILLED
        )
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         order_type=OrderType.LONG, leverage=2.0, limit_price=51000.0)

        with self.assertRaises(SignalException) as ctx:
            self.processor.process_limit_edit(self.DEFAULT_MINER_HOTKEY, signal, "order1", self.DEFAULT_NOW_MS)
        self.assertIn("not unfilled", str(ctx.exception))

    def test_edit_rejects_missing_trade_pair(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = self._existing_order_dict()
        signal = Signal(execution_type=ExecutionType.LIMIT_EDIT, order_type=OrderType.LONG, leverage=2.0)

        with self.assertRaises(SignalException) as ctx:
            self.processor.process_limit_edit(self.DEFAULT_MINER_HOTKEY, signal, "order1", self.DEFAULT_NOW_MS)
        self.assertIn("Invalid trade pair", str(ctx.exception))

    def test_edit_rejects_trade_pair_mismatch(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = self._existing_order_dict()
        signal = Signal(trade_pair=TradePair.ETHUSDC, execution_type=ExecutionType.LIMIT_EDIT,
                         order_type=OrderType.LONG, leverage=2.0, limit_price=51000.0)

        with self.assertRaises(SignalException) as ctx:
            self.processor.process_limit_edit(self.DEFAULT_MINER_HOTKEY, signal, "order1", self.DEFAULT_NOW_MS)
        self.assertIn("trade pair mismatch", str(ctx.exception))

    def test_edit_not_found_and_no_bracket_orders_raises(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = None
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         order_type=OrderType.FLAT)

        with self.assertRaises(SignalException) as ctx:
            self.processor.process_limit_edit(self.DEFAULT_MINER_HOTKEY, signal, "order1", self.DEFAULT_NOW_MS)
        self.assertIn("order not found", str(ctx.exception))

    def test_edit_defers_to_bulk_bracket_update_when_uuid_in_bracket_list(self):
        self.limit_order_client.get_limit_order_by_uuid.side_effect = [
            self._existing_order_dict(order_uuid="order1"),  # lookup for the envelope uuid itself
            None,  # lookup inside _apply_bulk_bracket_update for bracket "order1"
        ]
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         leverage=1.0, bracket_orders=[{"order_uuid": "order1", "stop_loss": 49000.0}])

        result = self.processor.process_limit_edit(self.DEFAULT_MINER_HOTKEY, signal, "order1", self.DEFAULT_NOW_MS)

        self.assertIsNone(result)
        self.limit_order_client.process_limit_order.assert_called_once()

    def test_bulk_bracket_update_processes_each_bracket(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = None
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         leverage=1.0,
                         bracket_orders=[
                             {"order_uuid": "b1", "stop_loss": 49000.0},
                             {"order_uuid": "b2", "take_profit": 52000.0},
                         ])

        result = self.processor.process_limit_edit(self.DEFAULT_MINER_HOTKEY, signal, "envelope-uuid", self.DEFAULT_NOW_MS)

        self.assertIsNone(result)
        self.assertEqual(self.limit_order_client.process_limit_order.call_count, 2)

    def test_bulk_bracket_update_no_edits_raises(self):
        with self.assertRaises(SignalException):
            self.processor._apply_bulk_bracket_update(
                self.DEFAULT_MINER_HOTKEY,
                Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                       order_type=OrderType.FLAT),
                self.DEFAULT_NOW_MS,
            )

    def test_bulk_bracket_update_cancels_when_cancel_intent(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = self._existing_order_dict(
            order_uuid="b1", execution_type=ExecutionType.BRACKET
        )
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         leverage=1.0, bracket_orders=[{"order_uuid": "b1"}])

        self.processor._apply_bulk_bracket_update(self.DEFAULT_MINER_HOTKEY, signal, self.DEFAULT_NOW_MS)

        self.limit_order_client.cancel_limit_order.assert_called_once_with(
            self.DEFAULT_MINER_HOTKEY, None, "b1", self.DEFAULT_NOW_MS, None
        )
        self.limit_order_client.process_limit_order.assert_not_called()

    def test_bulk_bracket_update_edits_existing_bracket(self):
        self.limit_order_client.get_limit_order_by_uuid.return_value = self._existing_order_dict(
            order_uuid="b1", execution_type=ExecutionType.BRACKET
        )
        signal = Signal(trade_pair=self.DEFAULT_TRADE_PAIR, execution_type=ExecutionType.LIMIT_EDIT,
                         leverage=1.0, bracket_orders=[{"order_uuid": "b1", "stop_loss": 48000.0}])

        self.processor._apply_bulk_bracket_update(self.DEFAULT_MINER_HOTKEY, signal, self.DEFAULT_NOW_MS)

        self.limit_order_client.process_limit_order.assert_called_once()
        self.assertTrue(self.limit_order_client.process_limit_order.call_args.kwargs["is_edit"])

    def test_is_bracket_cancel_intent(self):
        self.assertTrue(OrderProcessor._is_bracket_cancel_intent({"order_uuid": "b1"}))
        self.assertFalse(OrderProcessor._is_bracket_cancel_intent({"order_uuid": "b1", "stop_loss": 1.0}))
        self.assertFalse(OrderProcessor._is_bracket_cancel_intent({"order_uuid": "b1", "trailing_percent": 0.1}))

    def test_build_bracket_signal_with_trailing_percent(self):
        signal = OrderProcessor._build_bracket_signal(
            self.DEFAULT_TRADE_PAIR, {"order_uuid": "b1", "trailing_percent": 0.05, "leverage": 1.0}
        )
        self.assertEqual(signal.execution_type, ExecutionType.BRACKET)
        self.assertEqual(signal.trailing_stop, {"trailing_percent": 0.05})
        self.assertEqual(signal.leverage, 1.0)


class TestOrderProcessingResult(OrderProcessorTestBase):

    def test_get_response_json_with_order(self):
        mock_order = Mock()
        mock_order.__str__ = Mock(return_value='{"order": "data"}')
        result = OrderProcessingResult(execution_type=ExecutionType.LIMIT, order=mock_order)
        self.assertEqual(result.get_response_json(), '{"order": "data"}')

    def test_get_response_json_with_result_dict(self):
        result = OrderProcessingResult(
            execution_type=ExecutionType.LIMIT_CANCEL,
            result_dict={"status": "cancelled", "count": 3},
            should_track_uuid=False,
        )
        import json
        parsed = json.loads(result.get_response_json())
        self.assertEqual(parsed["status"], "cancelled")
        self.assertEqual(parsed["count"], 3)

    def test_get_response_json_empty(self):
        result = OrderProcessingResult(execution_type=ExecutionType.LIMIT)
        self.assertEqual(result.get_response_json(), "")

    def test_order_for_logging(self):
        mock_order = Mock()
        result = OrderProcessingResult(execution_type=ExecutionType.LIMIT, order=mock_order)
        self.assertEqual(result.order_for_logging, mock_order)

    def test_is_frozen(self):
        result = OrderProcessingResult(execution_type=ExecutionType.LIMIT)
        with self.assertRaises(Exception):
            result.success = False


if __name__ == '__main__':
    unittest.main()
