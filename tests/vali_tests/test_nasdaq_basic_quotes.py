"""
Nasdaq Basic quotes for equities.

Covers:
- Quote message conversion, untradable quotes dropped on receipt, 250ms tracker sampling, out-of-order handling
- Entitlement detection from subscribe status replies, reconnect spacing
- PriceSource.apply_nasdaq_fmv_rule: quotes replace FMV (within 8s for a single price, any quote in a window)
- Equities events, the limit/stop trigger window and get_quote through LivePriceFetcher
- Equities sessions (pre/regular/post/closed) and Nasdaq Basic trades kept for pre-market/after-hours
"""
import asyncio
import json
import time
import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

from data_generator.polygon_nasdaq_basic_data_service import PolygonNasdaqBasicDataService, NASDAQ_BASIC_SOURCE
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_objects.vali_dataclasses.price_source import NASDAQ_BASIC_TRADE_SOURCE
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.limit_order.order_trigger import evaluate_order_trigger
from vali_objects.price_fetcher.live_price_fetcher import LivePriceFetcher
from vali_objects.trade_pair import TradePairCategory
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.price_source import PriceSource


def quote_msg(t_ms, bid, ask, bid_size=100, ask_size=100, sym='AAPL'):
    return {'ev': 'Q', 'sym': sym, 'bx': 12, 'bp': bid, 'bs': bid_size, 'ax': 12, 'ap': ask, 'as': ask_size,
            't': t_ms, 'z': 3}


def fmv_source(price, t_ms):
    return PriceSource(source='Polygon_ws', timespan_ms=0, open=price, close=price, vwap=price, high=price,
                       low=price, start_ms=t_ms, websocket=True, lag_ms=0, bid=price, ask=price)


class TestNasdaqBasicService(unittest.TestCase):
    def setUp(self):
        self.svc = PolygonNasdaqBasicDataService(api_key='test', disable_ws=True, running_unit_tests=True)
        self.svc.set_test_entitlement(True)
        self.svc.set_test_market_open(True)
        self.now_ms = TimeUtil.now_in_millis()

    def feed(self, *msgs, recv_ms=None):
        self.svc._process_raw(json.dumps(list(msgs)), recv_ms or TimeUtil.now_in_millis())

    # ---------- ingestion ----------

    def test_quote_converted_to_price_source(self):
        self.feed(quote_msg(self.now_ms - 100, 99.99, 100.01))
        ps = self.svc.latest_websocket_events['AAPL']
        self.assertEqual(ps.source, NASDAQ_BASIC_SOURCE)
        self.assertEqual((ps.bid, ps.ask), (99.99, 100.01))
        self.assertEqual(ps.open, 100.0)
        self.assertEqual(ps.start_ms, self.now_ms - 100)
        self.assertTrue(ps.websocket)

    def test_untradable_quotes_dropped_on_receipt(self):
        self.feed(quote_msg(self.now_ms, 99.99, 100.01, bid_size=0),
                  quote_msg(self.now_ms, 99.99, 0.0),
                  quote_msg(self.now_ms, 100.02, 100.01),
                  quote_msg(self.now_ms, 100.0, 100.0),
                  quote_msg(self.now_ms, 99.70, 100.30))  # 60 bps > 50 bps cap
        self.assertNotIn('AAPL', self.svc.latest_websocket_events)
        self.assertEqual(dict(self.svc.dropped_counts), {'missing_side': 2, 'crossed_or_locked': 2, 'wide_spread': 1})

    def test_dropped_quote_leaves_previous_quote_current(self):
        # Same as the Polygon forex spread filter: the previous quote stays until it ages out
        self.feed(quote_msg(self.now_ms - 1000, 99.99, 100.01), quote_msg(self.now_ms, 100.02, 100.01))
        self.assertEqual(self.svc.latest_websocket_events['AAPL'].start_ms, self.now_ms - 1000)

    def test_set_test_quote_applies_receipt_filter(self):
        wide = PriceSource(source=NASDAQ_BASIC_SOURCE, open=100.0, start_ms=self.now_ms, websocket=True, bid=99.0, ask=101.0)
        self.svc.set_test_quote(TradePair.AAPL, wide)
        self.assertNotIn('AAPL', self.svc.latest_websocket_events)

    def test_unknown_and_non_equity_symbols_ignored(self):
        self.feed(quote_msg(self.now_ms, 1.0, 2.0, sym='NOTATICKER'), {'ev': 'T', 'sym': 'AAPL', 'p': 100, 't': self.now_ms})
        self.assertEqual(self.svc.latest_websocket_events, {})

    def test_tracker_keeps_last_quote_of_each_window(self):
        t0 = self.now_ms - 10_000
        sample = ValiConfig.NASDAQ_QUOTE_TRACKER_SAMPLE_MS
        self.feed(quote_msg(t0, 100.00, 100.10),
                  quote_msg(t0 + 100, 100.01, 100.10),
                  quote_msg(t0 + 200, 100.02, 100.10),   # last of window 1
                  quote_msg(t0 + sample + 50, 100.03, 100.10),   # opens window 2, flushes t0+200
                  quote_msg(t0 + sample + 100, 100.04, 100.10),  # last of window 2
                  quote_msg(t0 + 2 * sample + 60, 100.05, 100.10))  # opens window 3, flushes window 2
        stored = self.svc.trade_pair_to_recent_events['AAPL'].get_events_in_range(t0 - 1, self.now_ms)
        self.assertEqual([ps.start_ms for ps in stored], [t0 + 200, t0 + sample + 100])
        self.assertEqual(self.svc.latest_websocket_events['AAPL'].start_ms, t0 + 2 * sample + 60)

    def test_quote_older_than_latest_ignored(self):
        # Observed live: quotes can arrive 1-15ms behind the previous quote. Timestamp order wins, like other sources.
        self.feed(quote_msg(self.now_ms, 99.99, 100.01), quote_msg(self.now_ms - 5, 99.98, 100.02))
        latest = self.svc.latest_websocket_events['AAPL']
        self.assertEqual((latest.bid, latest.ask, latest.start_ms), (99.99, 100.01, self.now_ms))
        self.assertEqual(self.svc.n_quotes_out_of_order, 1)

    def test_replayed_quote_keeps_its_real_timestamp(self):
        # With no newer quote it is stored as-is; its age then fails the 8s check when pricing
        self.feed(quote_msg(self.now_ms - 60_000, 99.99, 100.01))
        self.assertEqual(self.svc.latest_websocket_events['AAPL'].start_ms, self.now_ms - 60_000)

    def test_get_closest_quote_checks_latest_and_tracker(self):
        t0 = self.now_ms - 10_000
        sample = ValiConfig.NASDAQ_QUOTE_TRACKER_SAMPLE_MS
        # t0 is flushed to the tracker when t0 + sample opens a new window; t0 + sample stays the latest
        self.feed(quote_msg(t0, 99.99, 100.01), quote_msg(t0 + sample, 99.98, 100.02))
        self.assertEqual(self.svc.get_closest_quote(TradePair.AAPL, self.now_ms).start_ms, t0 + sample)
        self.assertEqual(self.svc.get_closest_quote(TradePair.AAPL, t0 + 10).start_ms, t0)
        # Like other websocket sources, a quote after the requested time is used when it is closest
        self.assertEqual(self.svc.get_closest_quote(TradePair.AAPL, t0 - 1000).start_ms, t0)
        self.assertEqual(self.svc.get_closes_websocket([TradePair.AAPL], self.now_ms)[TradePair.AAPL].start_ms, t0 + sample)
        self.assertEqual(self.svc.get_closes_websocket([TradePair.MSFT], self.now_ms), {})

    def test_get_events_in_range_includes_latest_quote(self):
        t0 = self.now_ms - 10_000
        sample = ValiConfig.NASDAQ_QUOTE_TRACKER_SAMPLE_MS
        self.feed(quote_msg(t0, 99.99, 100.01), quote_msg(t0 + sample, 99.98, 100.02))
        in_range = self.svc.get_events_in_range(TradePair.AAPL, t0 - 1, self.now_ms)
        self.assertEqual([ps.start_ms for ps in in_range], [t0, t0 + sample])
        self.assertEqual([ps.start_ms for ps in self.svc.get_events_in_range(TradePair.AAPL, t0 + 1, self.now_ms)], [t0 + sample])
        self.assertEqual(self.svc.get_events_in_range(TradePair.MSFT, t0, self.now_ms), [])

    # ---------- entitlement ----------

    def test_subscribe_success_enables(self):
        self.svc.set_test_entitlement(None)
        self.assertEqual(self.svc._apply_status({'ev': 'status', 'status': 'success', 'message': 'subscribed to: Q.AAPL'}),
                         'subscribed')
        self.assertTrue(self.svc.is_enabled())

    def test_not_authorized_disables_and_schedules_retry(self):
        self.svc.tpc_to_last_event_time[TradePairCategory.EQUITIES] = time.time()
        before_s = time.time()
        self.assertEqual(self.svc._apply_status({'ev': 'status', 'status': 'error', 'message': 'not authorized'}), 'denied')
        self.assertFalse(self.svc.is_enabled())
        self.assertIs(self.svc.entitled, False)
        self.assertGreaterEqual(self.svc._entitlement_retry_at_s, before_s + ValiConfig.NASDAQ_ENTITLEMENT_RETRY_S)
        self.assertEqual(self.svc.tpc_to_last_event_time[TradePairCategory.EQUITIES], 0)

    def test_max_connections_backs_off_without_changing_entitlement(self):
        before_s = time.time()
        self.assertEqual(self.svc._apply_status({'ev': 'status', 'status': 'max_connections', 'message': 'Maximum'}),
                         'max_connections')
        self.assertTrue(self.svc.is_enabled())
        self.assertGreaterEqual(self.svc._next_connect_allowed_at_s, before_s + ValiConfig.NASDAQ_ENTITLEMENT_RETRY_S)

    def test_denied_status_closes_client_once(self):
        client = mock.MagicMock()
        client.close = mock.AsyncMock()
        self.svc.WEBSOCKET_OBJECTS[TradePairCategory.EQUITIES] = client
        denied = json.dumps([{'ev': 'status', 'status': 'error', 'message': 'not authorized'}] * 3)
        asyncio.run(self.svc.handle_msg(denied))
        asyncio.run(self.svc.handle_msg(denied))
        client.close.assert_awaited_once()
        self.assertFalse(self.svc.is_enabled())

    def test_quote_batches_are_queued_not_parsed_on_socket(self):
        raw = json.dumps([quote_msg(self.now_ms, 99.0, 101.0)])
        asyncio.run(self.svc.handle_msg(raw))
        self.assertEqual(self.svc._raw_queue.qsize(), 1)
        self.assertNotIn('AAPL', self.svc.latest_websocket_events)

    def test_create_client_respects_retry_and_reconnect_interval(self):
        tpc = TradePairCategory.EQUITIES
        self.svc.entitled = False
        self.svc._entitlement_retry_at_s = time.time() + 600
        self.svc._create_websocket_client(tpc)
        self.assertIsNone(self.svc.WEBSOCKET_OBJECTS[tpc])
        self.assertGreater(self.svc._client_unavailable_delay_s(tpc), 590)

        self.svc._entitlement_retry_at_s = 0
        self.svc._create_websocket_client(tpc)
        self.assertIsNotNone(self.svc.WEBSOCKET_OBJECTS[tpc])

        # A second connect inside the reconnect interval is deferred
        self.svc.WEBSOCKET_OBJECTS[tpc] = None
        self.svc._create_websocket_client(tpc)
        self.assertIsNone(self.svc.WEBSOCKET_OBJECTS[tpc])
        self.assertGreater(self.svc._client_unavailable_delay_s(tpc), ValiConfig.NASDAQ_MIN_RECONNECT_INTERVAL_S - 2)


class TestNasdaqQuoteSelection(unittest.TestCase):
    """sorted_valid_price_sources rule, and the equities events and get_quote through LivePriceFetcher."""

    def setUp(self):
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True, running_unit_tests=True)
        self.fetcher.set_test_market_open(True)
        self.now_ms = TimeUtil.now_in_millis()
        self.fetcher.set_test_price_source(TradePair.AAPL, fmv_source(100.0, self.now_ms - 50))

    def tearDown(self):
        self.fetcher.clear_test_price_sources()
        self.fetcher.clear_test_market_open()

    def nasdaq_quote(self, bid, ask, t_ms):
        mid = (bid + ask) / 2
        return PriceSource(source=NASDAQ_BASIC_SOURCE, timespan_ms=0, open=mid, close=mid, vwap=mid, high=mid,
                           low=mid, start_ms=t_ms, websocket=True, lag_ms=0, bid=bid, ask=ask)

    # ---------- sorted_valid_price_sources ----------

    def select(self, *events):
        return [ps.source for ps in self.fetcher.sorted_valid_price_sources(list(events), self.now_ms)]

    def test_recent_quote_replaces_fmv(self):
        self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, self.now_ms - 100), fmv_source(100.0, self.now_ms - 50)),
                         [NASDAQ_BASIC_SOURCE])

    def test_stale_quote_dropped_in_either_direction(self):
        stale_ms = ValiConfig.WEBSOCKET_PRICE_MAX_AGE_MS + 1000
        for t_ms in (self.now_ms - stale_ms, self.now_ms + stale_ms):
            self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, t_ms), fmv_source(100.0, self.now_ms - 50)),
                             ['Polygon_ws'])

    def test_quote_replaces_fmv_regardless_of_fmv_price(self):
        # No FMV band: FMV can lag quotes in a fast move, so it is not used to reject them
        self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, self.now_ms), fmv_source(101.0, self.now_ms)),
                         [NASDAQ_BASIC_SOURCE])

    def test_quote_without_fmv_kept(self):
        self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, self.now_ms)), [NASDAQ_BASIC_SOURCE])

    def test_lists_without_a_quote_unchanged(self):
        self.assertEqual(self.select(fmv_source(100.0, self.now_ms - 50), None), ['Polygon_ws'])

    # ---------- apply_nasdaq_fmv_rule over a window (no time_ms) ----------

    def test_window_keeps_all_quotes_and_drops_fmv(self):
        fmv = fmv_source(100.0, self.now_ms - 11_000)
        old_quote = self.nasdaq_quote(99.99, 100.01, self.now_ms - 20_000)  # >8s old is fine in a window
        new_quote = self.nasdaq_quote(100.99, 101.01, self.now_ms - 1000)
        other = PriceSource(source='Tiingo_ws', open=100.0, start_ms=self.now_ms, websocket=True)
        self.assertEqual(PriceSource.apply_nasdaq_fmv_rule([fmv, old_quote, new_quote, other]), [old_quote, new_quote, other])

    def test_events_without_quote_returned_unchanged(self):
        events = [fmv_source(100.0, self.now_ms)]
        self.assertIs(PriceSource.apply_nasdaq_fmv_rule(events, self.now_ms), events)

    # ---------- trigger window ----------

    def limit_buy(self, limit_price):
        return Order(trade_pair=TradePair.AAPL, order_uuid='aapl_limit', processed_ms=self.now_ms - 1000, price=0.0,
                     order_type=OrderType.LONG, leverage=0.1, execution_type=ExecutionType.LIMIT,
                     limit_price=limit_price, src=OrderSource.LIMIT_UNFILLED)

    def test_trigger_window_uses_quotes_instead_of_fmv(self):
        # setUp FMV is 100.0. A limit buy at 100.00 would trigger on FMV (bid = ask = 100.0).
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms - 100))
        window = self.fetcher.get_ws_price_sources_in_window(TradePair.AAPL, self.now_ms - 30_000, self.now_ms)
        self.assertEqual([ps.source for ps in window], [NASDAQ_BASIC_SOURCE])
        _, trigger_price, _ = evaluate_order_trigger('hk', self.limit_buy(100.00), None, window)
        self.assertIsNone(trigger_price)

        # Ask at the limit triggers, and the fill side comes from the quote
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.98, 100.00, self.now_ms - 50))
        window = self.fetcher.get_ws_price_sources_in_window(TradePair.AAPL, self.now_ms - 30_000, self.now_ms)
        trigger_ps, trigger_price, _ = evaluate_order_trigger('hk', self.limit_buy(100.00), None, window)
        self.assertEqual(trigger_price, 100.00)
        self.assertEqual(trigger_ps.source, NASDAQ_BASIC_SOURCE)
        self.assertEqual(trigger_ps.parse_appropriate_price(self.now_ms, False, OrderType.LONG, OrderType.LONG), 100.00)

    def test_trigger_window_without_nasdaq_basic_is_fmv(self):
        window = self.fetcher.get_ws_price_sources_in_window(TradePair.AAPL, self.now_ms - 30_000, self.now_ms)
        self.assertEqual([ps.source for ps in window], ['Polygon_ws'])
        _, trigger_price, _ = evaluate_order_trigger('hk', self.limit_buy(100.00), None, window)
        self.assertEqual(trigger_price, 100.00)

    # ---------- equities events through LivePriceFetcher ----------

    def test_disabled_by_default(self):
        self.assertFalse(self.fetcher.nasdaq_quotes_enabled())

    def test_fmv_only_without_nasdaq_basic(self):
        sources = self.fetcher.get_sorted_price_sources_for_trade_pair(TradePair.AAPL, self.now_ms)
        self.assertEqual([ps.source for ps in sources], ['Polygon_ws'])

    def test_valid_quote_replaces_fmv(self):
        # The quote is older than FMV, but FMV is no longer in the race
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms - 100))
        sources = self.fetcher.get_sorted_price_sources_for_trade_pair(TradePair.AAPL, self.now_ms)
        self.assertEqual([ps.source for ps in sources], [NASDAQ_BASIC_SOURCE])
        # Fill side follows the #941 bid/ask logic on the first source
        self.assertEqual(sources[0].parse_appropriate_price(self.now_ms, False, OrderType.LONG, OrderType.LONG), 100.01)
        self.assertEqual(sources[0].parse_appropriate_price(self.now_ms, False, OrderType.SHORT, OrderType.SHORT), 99.99)
        self.assertEqual(sources[0].parse_appropriate_price(self.now_ms, False, OrderType.FLAT, OrderType.LONG), 99.99)

    def test_stale_quote_leaves_fmv(self):
        stale_ms = self.now_ms - ValiConfig.WEBSOCKET_PRICE_MAX_AGE_MS - 1000
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, stale_ms))
        sources = self.fetcher.get_sorted_price_sources_for_trade_pair(TradePair.AAPL, self.now_ms)
        self.assertEqual([ps.source for ps in sources], ['Polygon_ws'])

    def test_closer_databento_quote_wins_over_nasdaq(self):
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms - 300))
        databento = PriceSource(source='Databento_ws', timespan_ms=1000, open=100.0, close=100.0, vwap=100.0,
                                high=100.0, low=100.0, start_ms=self.now_ms - 20, websocket=True, bid=99.98, ask=100.02)
        with mock.patch.object(self.fetcher, 'databento_data_service') as db_svc:
            db_svc.get_closes_websocket.return_value = {TradePair.AAPL: databento}
            sources = self.fetcher.get_sorted_price_sources_for_trade_pair(TradePair.AAPL, self.now_ms)
        self.assertEqual([ps.source for ps in sources], ['Databento_ws', NASDAQ_BASIC_SOURCE])

    def test_quote_after_lookup_time_used_when_closest(self):
        # Same as the other websocket sources: closest by absolute time difference
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms + 1000))
        sources = self.fetcher.get_sorted_price_sources_for_trade_pair(TradePair.AAPL, self.now_ms)
        self.assertEqual([ps.source for ps in sources], [NASDAQ_BASIC_SOURCE])

    def test_non_equities_unaffected(self):
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms - 100))
        eur = PriceSource(source='Polygon_ws', open=1.1, close=1.1, high=1.1, low=1.1, vwap=1.1,
                          start_ms=self.now_ms, websocket=True, bid=1.0999, ask=1.1001)
        self.fetcher.set_test_price_source(TradePair.EURUSD, eur)
        tp_to_sources = self.fetcher.get_tp_to_sorted_price_sources([TradePair.AAPL, TradePair.EURUSD], self.now_ms)
        self.assertEqual([ps.source for ps in tp_to_sources[TradePair.AAPL]], [NASDAQ_BASIC_SOURCE])
        self.assertEqual([ps.source for ps in tp_to_sources[TradePair.EURUSD]], ['Polygon_ws'])

    def test_clear_test_price_sources_disables_nasdaq(self):
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms - 100))
        self.fetcher.clear_test_price_sources()
        self.assertFalse(self.fetcher.nasdaq_quotes_enabled())

    # ---------- get_quote ----------

    def test_get_quote_uses_closest_nasdaq_quote(self):
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms - 5000))
        self.assertEqual(self.fetcher.get_quote(TradePair.AAPL, self.now_ms), (99.99, 100.01, self.now_ms - 5000))

    def test_get_quote_ignores_nasdaq_quote_older_than_max_age(self):
        stale_ms = self.now_ms - ValiConfig.WEBSOCKET_PRICE_MAX_AGE_MS - 1000
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, stale_ms))
        with mock.patch.object(self.fetcher.polygon_data_service, 'get_quote', return_value=(1.0, 2.0, 3)), \
                mock.patch.object(self.fetcher, 'databento_data_service', None):
            self.assertEqual(self.fetcher.get_quote(TradePair.AAPL, self.now_ms), (1.0, 2.0, 3))

    def test_get_quote_without_nasdaq_basic_uses_existing_path(self):
        with mock.patch.object(self.fetcher.polygon_data_service, 'get_quote', return_value=(1.0, 2.0, 3)) as poly, \
                mock.patch.object(self.fetcher, 'databento_data_service', None):
            self.assertEqual(self.fetcher.get_quote(TradePair.AAPL, self.now_ms), (1.0, 2.0, 3))
        poly.assert_called_once()


def et_ms(year, month, day, hour, minute=0):
    return int(datetime(year, month, day, hour, minute, tzinfo=ZoneInfo('America/New_York')).timestamp() * 1000)


def trade_msg(t_ms, price, size=100, conditions=(), sym='AAPL'):
    return {'ev': 'T', 'sym': sym, 'x': 4, 'p': price, 's': size, 'c': list(conditions), 't': t_ms, 'z': 3}


class TestEquitySession(unittest.TestCase):
    calendar = UnifiedMarketCalendar()

    def session(self, *et, trade_pair=TradePair.AAPL):
        return self.calendar.get_equity_session(trade_pair, et_ms(*et))

    def test_normal_day_boundaries(self):
        # 2026-10-01 is a Thursday (EDT)
        self.assertEqual(self.session(2026, 10, 1, 3, 59), 'closed')
        self.assertEqual(self.session(2026, 10, 1, 4, 0), 'pre')
        self.assertEqual(self.session(2026, 10, 1, 9, 29), 'pre')
        self.assertEqual(self.session(2026, 10, 1, 9, 30), 'regular')
        self.assertEqual(self.session(2026, 10, 1, 15, 59), 'regular')
        self.assertEqual(self.session(2026, 10, 1, 16, 0), 'post')
        self.assertEqual(self.session(2026, 10, 1, 19, 59), 'post')
        self.assertEqual(self.session(2026, 10, 1, 20, 0), 'closed')

    def test_winter_after_hours_past_midnight_utc(self):
        # 19:30 EST is 00:30 UTC the next day; the session follows the Eastern date
        self.assertEqual(self.session(2026, 12, 1, 19, 30), 'post')
        self.assertEqual(self.session(2026, 12, 1, 20, 0), 'closed')

    def test_early_close_has_no_after_hours(self):
        # Day after Thanksgiving: 13:00 close, and the calendar has no after-hours session
        self.assertEqual(self.session(2026, 11, 27, 8, 0), 'pre')
        self.assertEqual(self.session(2026, 11, 27, 12, 59), 'regular')
        self.assertEqual(self.session(2026, 11, 27, 13, 0), 'closed')
        self.assertEqual(self.session(2026, 11, 27, 16, 30), 'closed')

    def test_holiday_and_weekend_closed(self):
        self.assertEqual(self.session(2026, 11, 26, 6, 0), 'closed')  # Thanksgiving
        self.assertEqual(self.session(2026, 10, 3, 6, 0), 'closed')   # Saturday

    def test_bounds_span_the_session(self):
        session, start_ms, end_ms = self.calendar.get_equity_session_bounds(TradePair.AAPL, et_ms(2026, 10, 1, 17, 0))
        self.assertEqual((session, start_ms, end_ms), ('post', et_ms(2026, 10, 1, 16, 0), et_ms(2026, 10, 1, 20, 0)))

    def test_only_vanta_equities_have_sessions(self):
        for trade_pair in (TradePair.EURUSD, TradePair.AAPLUSDC, TradePair.BTCUSDC):
            self.assertIsNone(self.session(2026, 10, 1, 12, 0, trade_pair=trade_pair))

    def test_fetcher_session_override_cleared_with_market_open_override(self):
        fetcher = LivePriceFetcher(secrets=ValiUtils.get_secrets(running_unit_tests=True), disable_ws=True,
                                   running_unit_tests=True)
        fetcher.set_test_equity_session('post')
        self.assertEqual(fetcher.get_equity_session(TradePair.AAPL), 'post')
        self.assertIsNone(fetcher.get_equity_session(TradePair.EURUSD))
        fetcher.clear_test_market_open()
        self.assertEqual(fetcher.get_equity_session(TradePair.AAPL, et_ms(2026, 10, 1, 12, 0)), 'regular')


class TestNasdaqBasicTrades(unittest.TestCase):
    def setUp(self):
        self.svc = PolygonNasdaqBasicDataService(api_key='test', disable_ws=True, running_unit_tests=True)
        self.svc.set_test_equity_session('post')
        self.now_ms = TimeUtil.now_in_millis()

    def feed(self, *msgs):
        self.svc._process_raw(json.dumps(list(msgs)), TimeUtil.now_in_millis())

    def stored(self, trade_pair=TradePair.AAPL):
        return self.svc.get_trades_in_range(trade_pair, self.now_ms - 300_000, self.now_ms + 1000)

    def test_subscribes_to_quotes_and_trades(self):
        client = mock.MagicMock()
        self.svc.WEBSOCKET_OBJECTS[TradePairCategory.EQUITIES] = client
        self.svc._subscribe_websockets(TradePairCategory.EQUITIES)
        symbols = client.subscribe.call_args.args
        self.assertIn('Q.AAPL', symbols)
        self.assertIn('T.AAPL', symbols)
        self.assertEqual(len(symbols), 2 * self.svc.n_subscribed)

    def test_round_lot_trade_stored_in_post(self):
        self.feed(trade_msg(self.now_ms - 100, 330.25, size=200, conditions=(12, 14, 41)))
        trades = self.stored()
        self.assertEqual(len(trades), 1)
        ps = trades[0]
        self.assertEqual(ps.source, NASDAQ_BASIC_TRADE_SOURCE)
        self.assertEqual((ps.open, ps.close, ps.bid, ps.ask, ps.start_ms), (330.25, 330.25, 330.25, 330.25, self.now_ms - 100))

    def test_trades_stored_in_pre_but_not_regular_or_closed(self):
        self.svc.set_test_equity_session('pre')
        self.feed(trade_msg(self.now_ms - 300, 330.0))
        for session in ('regular', 'closed'):
            self.svc.set_test_equity_session(session)
            self.feed(trade_msg(self.now_ms - 200, 331.0))
        self.assertEqual([ps.open for ps in self.stored()], [330.0])

    def test_untradable_trades_dropped(self):
        self.feed(trade_msg(self.now_ms, 330.0, size=99),                 # odd lot
                  trade_msg(self.now_ms, 263.88, conditions=(32, 41)),    # sold out of sequence
                  trade_msg(self.now_ms, 277.64, conditions=(2, 12)),     # average price
                  trade_msg(self.now_ms, 330.5, conditions=(8,)),         # closing print
                  trade_msg(self.now_ms, 0.0))
        self.assertEqual(self.stored(), [])
        self.assertEqual(dict(self.svc.trade_dropped_counts), {'odd_lot': 1, 'excluded_condition': 3, 'bad_price': 1})

    def test_trades_in_same_millisecond_all_kept(self):
        self.feed(trade_msg(self.now_ms, 330.10), trade_msg(self.now_ms, 330.05), trade_msg(self.now_ms, 330.00))
        self.assertEqual(sorted(ps.open for ps in self.stored()), [330.00, 330.05, 330.10])

    def test_trades_in_range_and_pruning(self):
        old_ms = self.now_ms - ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS - 1000
        self.feed(trade_msg(old_ms, 329.0), trade_msg(self.now_ms - 2000, 330.0), trade_msg(self.now_ms - 1000, 330.5))
        self.assertEqual([ps.open for ps in self.stored()], [330.0, 330.5])
        self.assertEqual([ps.open for ps in self.svc.get_trades_in_range(TradePair.AAPL, self.now_ms - 1500, self.now_ms)],
                         [330.5])
        self.assertEqual(self.stored(TradePair.MSFT), [])

    def test_set_test_trade_applies_filter_and_clear_removes_trades(self):
        self.svc.set_test_trade(TradePair.AAPL, 330.0, 50, self.now_ms)
        self.svc.set_test_trade(TradePair.AAPL, 330.0, 100, self.now_ms)
        self.assertEqual(len(self.stored()), 1)
        self.svc.clear_test_quotes()
        self.assertEqual(self.stored(), [])

    def test_websocket_stays_up_in_pre_and_post(self):
        for session, expected in (('pre', True), ('regular', True), ('post', True), ('closed', False)):
            self.svc.set_test_equity_session(session)
            self.assertEqual(self.svc._websocket_session_active(TradePair.AAPL), expected)

    def test_session_lookup_cached_per_span(self):
        self.svc.clear_test_market_open()
        with mock.patch.object(self.svc.market_calendar, 'get_equity_session_bounds',
                               wraps=self.svc.market_calendar.get_equity_session_bounds) as bounds:
            self.assertEqual(self.svc._session_at(et_ms(2026, 10, 1, 17, 0)), 'post')
            self.assertEqual(self.svc._session_at(et_ms(2026, 10, 1, 19, 0)), 'post')
            self.assertEqual(self.svc._session_at(et_ms(2026, 10, 2, 5, 0)), 'pre')
        self.assertEqual(bounds.call_count, 2)

    def test_trigger_window_in_extended_hours_is_trades_only(self):
        fetcher = LivePriceFetcher(secrets=ValiUtils.get_secrets(running_unit_tests=True), disable_ws=True,
                                   running_unit_tests=True)
        now_ms = TimeUtil.now_in_millis()
        fetcher.set_test_price_source(TradePair.AAPL, fmv_source(100.0, now_ms - 50))
        fetcher.set_test_equity_session('post')
        # Without Nasdaq Basic nothing fills outside regular hours, even with FMV in the window
        self.assertEqual(fetcher.get_ws_price_sources_in_window(TradePair.AAPL, now_ms - 30_000, now_ms), [])
        trade = PriceSource(source=NASDAQ_BASIC_TRADE_SOURCE, open=99.5, start_ms=now_ms - 100, websocket=True,
                            bid=99.5, ask=99.5)
        fetcher.set_test_price_source(TradePair.AAPL, trade)
        window = fetcher.get_ws_price_sources_in_window(TradePair.AAPL, now_ms - 30_000, now_ms)
        self.assertEqual([(ps.source, ps.open) for ps in window], [(NASDAQ_BASIC_TRADE_SOURCE, 99.5)])
        # Regular hours: trades are not used
        fetcher.set_test_equity_session('regular')
        self.assertEqual([ps.source for ps in fetcher.get_ws_price_sources_in_window(TradePair.AAPL, now_ms - 30_000, now_ms)],
                         ['Polygon_ws'])

    def test_other_services_keep_market_hours_for_their_websocket(self):
        fetcher = LivePriceFetcher(secrets=ValiUtils.get_secrets(running_unit_tests=True), disable_ws=True,
                                   running_unit_tests=True)
        fetcher.set_test_market_open(False)
        fetcher.set_test_equity_session('post')
        self.assertFalse(fetcher.polygon_data_service._websocket_session_active(TradePair.AAPL))


if __name__ == '__main__':
    unittest.main()
