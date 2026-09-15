"""
Unit tests for what happens to a miner's resting orders once they enter PRO_CHALLENGE_TRANSITION.

PRO_CHALLENGE_TRANSITION is not an account switch: the miner keeps trading the standard account and
has to wind it down themselves before their pro account starts. They may no longer open a position
or add to one, but they still need their brackets and their resting exits to close out with. So the
orders that have to go are exactly the resting entries, and nothing else.

Covers:
  * LimitOrderManager.cancel_entry_orders: which resting orders the sweep takes and which it leaves.
  * The fill path: an entry order that triggers during the transition is cancelled with a reason
    rather than being reported as a failed fill, including an order that only became an entry order
    after its position closed; reducing orders and brackets still fill.
  * _convert_stop_limit_to_limit_order: a rejected conversion leaves the parent recorded as
    cancelled instead of vanishing from the trader's order history as STOP_LIMIT_FILLED.
  * ChallengePeriodManager.admin_set_bucket: the sweep runs on the way into the transition, and the
    blanket cancel is still what an account switch does.
"""
import contextlib
import unittest
from unittest.mock import MagicMock, patch

from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.limit_order.limit_order_manager import LimitOrderManager
from vali_objects.utils.market_order.market_order_manager import OrderExecution
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_dataclasses.price_source import PriceSource

NOW_MS = 1_748_000_000_000
HOTKEY = "test_miner"

_LIMIT_ORDER_CLIENT_PATHS = [
    "vali_objects.utils.market_order.market_order_client.MarketOrderClient",
    "vali_objects.price_fetcher.live_price_client.LivePriceFetcherClient",
    "vali_objects.position_management.position_manager_client.PositionManagerClient",
    "vali_objects.miner_account.miner_account_client.MinerAccountClient",
]

_CP_CLIENT_PATHS = [
    "vali_objects.challenge_period.challengeperiod_manager.PerfLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.PositionManagerClient",
    "vali_objects.challenge_period.challengeperiod_manager.LimitOrderClient",
    "vali_objects.challenge_period.challengeperiod_manager.EliminationClient",
    "vali_objects.challenge_period.challengeperiod_manager.PlagiarismClient",
    "vali_objects.challenge_period.challengeperiod_manager.MinerAccountClient",
    "vali_objects.challenge_period.challengeperiod_manager.CommonDataClient",
    "vali_objects.challenge_period.challengeperiod_manager.AssetSelectionClient",
    "vali_objects.challenge_period.challengeperiod_manager.DebtLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.EntityClient",
]


def _order(uuid, order_type, execution_type=ExecutionType.LIMIT, src=None, trade_pair=TradePair.BTCUSD):
    if src is None:
        src = {
            ExecutionType.LIMIT: OrderSource.LIMIT_UNFILLED,
            ExecutionType.BRACKET: OrderSource.BRACKET_UNFILLED,
            ExecutionType.STOP_LIMIT: OrderSource.STOP_LIMIT_UNFILLED,
        }[execution_type]
    return Order(
        trade_pair=trade_pair,
        order_uuid=uuid,
        processed_ms=NOW_MS - 60_000,
        price=0.0,
        order_type=order_type,
        execution_type=execution_type,
        limit_price=50_000,
        stop_price=50_000 if execution_type == ExecutionType.STOP_LIMIT else None,
        leverage=0.1,
        src=src,
    )


def _position(position_type, trade_pair=TradePair.BTCUSD):
    return Position(
        miner_hotkey=HOTKEY,
        position_uuid=f"pos_{trade_pair.trade_pair_id}",
        open_ms=NOW_MS - 120_000,
        trade_pair=trade_pair,
        position_type=position_type,
        account_size=100_000.0,
    )


class ProTransitionRestingOrdersTest(unittest.TestCase):
    """The limit order manager side: which orders the transition takes, and when."""

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for path in _LIMIT_ORDER_CLIENT_PATHS:
            stack.enter_context(patch(path))
        stack.enter_context(patch.object(LimitOrderManager, "_read_limit_orders_from_disk"))

        self.manager = LimitOrderManager(running_unit_tests=True, serve=False)
        # Disk is not what these tests are about; record the writes instead of making them.
        self.written = []
        stack.enter_context(
            patch.object(LimitOrderManager, "_write_to_disk",
                         side_effect=lambda hk, order: self.written.append((order.order_uuid, order.src)))
        )
        self._set_bucket(MinerBucket.PRO_CHALLENGE_TRANSITION)
        self._set_open_positions([])

    # ---- fixtures -------------------------------------------------------------------

    def _set_bucket(self, bucket):
        self.manager._miner_account_client.get_account.return_value = MagicMock(miner_bucket=bucket)

    def _set_open_positions(self, positions):
        by_trade_pair = {p.trade_pair.trade_pair_id: p for p in positions}
        self.manager._position_client.get_positions_for_one_hotkey.return_value = positions
        self.manager._position_client.get_open_position_for_trade_pair.side_effect = (
            lambda hotkey, trade_pair_id: by_trade_pair.get(trade_pair_id)
        )

    def _rest(self, *orders):
        for order in orders:
            self.manager._limit_orders.setdefault(order.trade_pair, {}).setdefault(HOTKEY, []).append(order)

    def _resting_uuids(self):
        return {o.order_uuid
                for hotkey_dict in self.manager._limit_orders.values()
                for o in hotkey_dict.get(HOTKEY, [])}

    # ---- cancel_entry_orders --------------------------------------------------------

    def test_sweep_takes_entries_and_leaves_everything_else(self):
        opener = _order("opener", OrderType.LONG)
        adder = _order("adder", OrderType.LONG, trade_pair=TradePair.ETHUSD)
        reducer = _order("reducer", OrderType.SHORT, trade_pair=TradePair.ETHUSD)
        bracket = _order("bracket", OrderType.LONG, ExecutionType.BRACKET, trade_pair=TradePair.ETHUSD)
        stop_entry = _order("stop_entry", OrderType.SHORT, ExecutionType.STOP_LIMIT, trade_pair=TradePair.SOLUSD)
        self._rest(opener, adder, reducer, bracket, stop_entry)
        self._set_open_positions([_position(OrderType.LONG, TradePair.ETHUSD)])

        result = self.manager.cancel_entry_orders(HOTKEY, NOW_MS, OrderSource.PRO_TRANSITION_CANCELLED)

        self.assertEqual(result["num_cancelled"], 3)
        # The brackets and the resting exit are what the miner winds the account down with.
        self.assertEqual(self._resting_uuids(), {"reducer", "bracket"})
        self.assertEqual(
            dict(self.written),
            {"opener": OrderSource.PRO_TRANSITION_CANCELLED,
             "adder": OrderSource.PRO_TRANSITION_CANCELLED,
             "stop_entry": OrderSource.PRO_TRANSITION_CANCELLED},
        )

    def test_sweep_derives_a_cancel_src_when_none_is_given(self):
        self._rest(_order("opener", OrderType.LONG),
                   _order("stop_entry", OrderType.LONG, ExecutionType.STOP_LIMIT, trade_pair=TradePair.ETHUSD))

        self.manager.cancel_entry_orders(HOTKEY, NOW_MS)

        self.assertEqual(
            dict(self.written),
            {"opener": OrderSource.LIMIT_CANCELLED, "stop_entry": OrderSource.STOP_LIMIT_CANCELLED},
        )

    def test_sweep_ignores_orders_that_are_no_longer_resting(self):
        self._rest(_order("already_filled", OrderType.LONG, src=OrderSource.LIMIT_FILLED),
                   _order("already_cancelled", OrderType.LONG, src=OrderSource.LIMIT_CANCELLED,
                          trade_pair=TradePair.ETHUSD))

        result = self.manager.cancel_entry_orders(HOTKEY, NOW_MS, OrderSource.PRO_TRANSITION_CANCELLED)

        self.assertEqual(result["num_cancelled"], 0)
        self.assertEqual(self.written, [])

    # ---- the fill path --------------------------------------------------------------

    def _fill(self, order):
        price_source = PriceSource(source="test", start_ms=NOW_MS, open=50_000, close=50_000,
                                   high=50_000, low=50_000, bid=49_999, ask=50_001)
        return self.manager._fill_limit_order_with_price_source(HOTKEY, order, price_source, 50_000)

    def _expect_fill(self, order, position):
        filled = _order(order.order_uuid, order.order_type)
        filled.price = 50_000
        self.manager._market_order_client.execute_order.return_value = OrderExecution(filled, position)

    def test_a_triggered_entry_order_is_cancelled_with_a_reason(self):
        order = _order("opener", OrderType.LONG)
        self._rest(order)

        error_msg = self._fill(order)

        self.manager._market_order_client.execute_order.assert_not_called()
        self.assertIn("transitioning to a Pro Account", error_msg)
        self.assertEqual(self.written, [("opener", OrderSource.PRO_TRANSITION_CANCELLED)])
        self.assertEqual(self._resting_uuids(), set())

    def test_an_order_that_became_an_entry_order_is_caught_at_fill_time(self):
        """A resting exit that outlives its position is an opener by the time it triggers, so the
        sweep at the start of the transition cannot have caught it."""
        order = _order("was_an_exit", OrderType.SHORT)
        self._rest(order)
        self._set_open_positions([])  # the LONG it was closing is already gone

        self._fill(order)

        self.manager._market_order_client.execute_order.assert_not_called()
        self.assertEqual(self.written, [("was_an_exit", OrderSource.PRO_TRANSITION_CANCELLED)])

    def test_a_reducing_order_still_fills_during_the_transition(self):
        order = _order("reducer", OrderType.SHORT)
        position = _position(OrderType.LONG)
        self._rest(order)
        self._set_open_positions([position])
        self._expect_fill(order, position)

        self.assertIsNone(self._fill(order))
        self.manager._market_order_client.execute_order.assert_called_once()

    def test_a_bracket_still_fills_during_the_transition(self):
        order = _order("bracket", OrderType.LONG, ExecutionType.BRACKET)
        position = _position(OrderType.LONG)
        self._rest(order)
        self._set_open_positions([position])
        self._expect_fill(order, position)

        self.assertIsNone(self._fill(order))
        self.manager._market_order_client.execute_order.assert_called_once()

    def test_an_entry_order_fills_normally_outside_the_transition(self):
        self._set_bucket(MinerBucket.SUBACCOUNT_FUNDED)
        order = _order("opener", OrderType.LONG)
        position = _position(OrderType.LONG)
        self._rest(order)
        self._expect_fill(order, position)

        self.assertIsNone(self._fill(order))
        self.manager._market_order_client.execute_order.assert_called_once()

    # ---- stop-limit conversion ------------------------------------------------------

    def test_a_rejected_stop_limit_conversion_records_the_parent_as_cancelled(self):
        parent = _order("stop_entry", OrderType.LONG, ExecutionType.STOP_LIMIT)
        self._rest(parent)

        with patch.object(self.manager, "process_limit_order",
                          side_effect=SignalException("transitioning to a Pro Account")):
            self.manager._convert_stop_limit_to_limit_order(HOTKEY, parent, NOW_MS)

        # Without the cancellation record the parent leaves no trace: _close_limit_order removes the
        # unfilled file and only persists orders whose src is a cancellation.
        self.assertEqual(self.written, [("stop_entry", OrderSource.STOP_LIMIT_CANCELLED)])
        self.assertEqual(parent.src, OrderSource.STOP_LIMIT_CANCELLED)

    def test_a_successful_stop_limit_conversion_still_closes_the_parent_as_filled(self):
        parent = _order("stop_entry", OrderType.LONG, ExecutionType.STOP_LIMIT)
        self._rest(parent)

        with patch.object(self.manager, "process_limit_order") as process:
            self.manager._convert_stop_limit_to_limit_order(HOTKEY, parent, NOW_MS)

        process.assert_called_once()
        self.assertEqual(parent.src, OrderSource.STOP_LIMIT_FILLED)
        self.assertEqual(self.written, [])  # filled orders are not persisted


class AdminSetBucketCancelsEntryOrdersTest(unittest.TestCase):
    """The challenge period side: the sweep is wired into the move into the transition."""

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for path in _CP_CLIENT_PATHS:
            stack.enter_context(patch(path))
        self.manager = ChallengePeriodManager(is_backtesting=True)
        self.manager._entity_client.apply_bucket_account_size.return_value = (True, "account size set")
        stack.enter_context(patch.object(self.manager, "_sync_buckets_to_accounts"))
        stack.enter_context(patch.object(self.manager, "_save_to_disk"))
        self.manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS)

    def test_entering_the_transition_sweeps_entry_orders_only(self):
        success, message = self.manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION, NOW_MS)

        self.assertTrue(success, message)
        self.manager._limit_order_client.cancel_entry_orders.assert_called_once_with(
            HOTKEY, NOW_MS, OrderSource.PRO_TRANSITION_CANCELLED
        )
        # The transition is not an account switch, so the brackets and the resting exits survive.
        self.manager._limit_order_client.cancel_limit_order.assert_not_called()
        self.manager._position_client.close_all_positions.assert_not_called()

    def test_an_account_switch_still_cancels_everything(self):
        self.manager.set_miner_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION, NOW_MS)

        success, message = self.manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, NOW_MS)

        self.assertTrue(success, message)
        self.manager._limit_order_client.cancel_limit_order.assert_called_once_with(HOTKEY, None, "ALL", NOW_MS)
        self.manager._limit_order_client.cancel_entry_orders.assert_not_called()

    def test_a_standard_bucket_move_touches_no_orders(self):
        success, message = self.manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_ALPHA, NOW_MS)

        self.assertTrue(success, message)
        self.manager._limit_order_client.cancel_entry_orders.assert_not_called()
        self.manager._limit_order_client.cancel_limit_order.assert_not_called()


if __name__ == "__main__":
    unittest.main()
