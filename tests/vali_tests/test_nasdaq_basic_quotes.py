"""
Nasdaq Basic quotes for equities.

Covers:
- Quote message conversion, untradable quotes dropped on receipt, 250ms tracker sampling, out-of-order handling
- Entitlement detection from subscribe status replies, reconnect spacing
- sorted_valid_price_sources: a recent quote that agrees with FMV replaces FMV, otherwise the quote is dropped
- Equities events and get_quote through LivePriceFetcher
"""
import asyncio
import json
import time
import unittest
from unittest import mock

from data_generator.polygon_nasdaq_basic_data_service import PolygonNasdaqBasicDataService, NASDAQ_BASIC_SOURCE
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.price_fetcher.live_price_fetcher import LivePriceFetcher
from vali_objects.trade_pair import TradePairCategory
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
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

    def test_slightly_late_quote_becomes_latest_with_clamped_time(self):
        # Observed live: quotes can arrive 1-15ms behind the previous quote's timestamp
        self.feed(quote_msg(self.now_ms, 99.99, 100.01), quote_msg(self.now_ms - 5, 99.98, 100.02))
        latest = self.svc.latest_websocket_events['AAPL']
        self.assertEqual((latest.bid, latest.ask), (99.98, 100.02))
        self.assertEqual(latest.start_ms, self.now_ms)
        self.assertEqual(self.svc.n_quotes_out_of_order, 1)

    def test_far_out_of_order_quote_dropped(self):
        late_ms = ValiConfig.NASDAQ_QUOTE_OUT_OF_ORDER_TOLERANCE_MS + 500
        self.feed(quote_msg(self.now_ms, 99.99, 100.01), quote_msg(self.now_ms - late_ms, 50.00, 50.01))
        latest = self.svc.latest_websocket_events['AAPL']
        self.assertEqual((latest.bid, latest.ask), (99.99, 100.01))
        self.assertEqual(self.svc.n_quotes_out_of_order, 1)

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

    def test_recent_quote_agreeing_with_fmv_replaces_fmv(self):
        self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, self.now_ms - 100), fmv_source(100.0, self.now_ms - 50)),
                         [NASDAQ_BASIC_SOURCE])

    def test_stale_quote_dropped_in_either_direction(self):
        stale_ms = ValiConfig.WEBSOCKET_PRICE_MAX_AGE_MS + 1000
        for t_ms in (self.now_ms - stale_ms, self.now_ms + stale_ms):
            self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, t_ms), fmv_source(100.0, self.now_ms - 50)),
                             ['Polygon_ws'])

    def test_quote_away_from_fmv_dropped(self):
        # ~50 bps from FMV, outside max(25 bps, 2 bps spread)
        self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, self.now_ms), fmv_source(100.5, self.now_ms)),
                         ['Polygon_ws'])

    def test_fmv_band_widens_to_spread(self):
        # 40 bps spread, mid 30 bps from FMV: inside max(25, 40)
        self.assertEqual(self.select(self.nasdaq_quote(100.10, 100.50, self.now_ms), fmv_source(100.0, self.now_ms)),
                         [NASDAQ_BASIC_SOURCE])

    def test_old_or_missing_fmv_not_used_for_band(self):
        old_fmv = fmv_source(105.0, self.now_ms - ValiConfig.NASDAQ_QUOTE_FMV_MAX_AGE_MS - 1000)
        self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, self.now_ms), old_fmv), [NASDAQ_BASIC_SOURCE])
        self.assertEqual(self.select(self.nasdaq_quote(99.99, 100.01, self.now_ms)), [NASDAQ_BASIC_SOURCE])

    def test_lists_without_a_quote_unchanged(self):
        self.assertEqual(self.select(fmv_source(100.0, self.now_ms - 50), None), ['Polygon_ws'])

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

    def test_quote_away_from_fmv_leaves_fmv(self):
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(100.99, 101.01, self.now_ms - 100))
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
        # Same as the Databento branch: closest quote with a bid and ask, no age limit
        self.fetcher.set_test_price_source(TradePair.AAPL, self.nasdaq_quote(99.99, 100.01, self.now_ms - 20_000))
        self.assertEqual(self.fetcher.get_quote(TradePair.AAPL, self.now_ms), (99.99, 100.01, self.now_ms - 20_000))

    def test_get_quote_without_nasdaq_basic_uses_existing_path(self):
        with mock.patch.object(self.fetcher.polygon_data_service, 'get_quote', return_value=(1.0, 2.0, 3)) as poly, \
                mock.patch.object(self.fetcher, 'databento_data_service', None):
            self.assertEqual(self.fetcher.get_quote(TradePair.AAPL, self.now_ms), (1.0, 2.0, 3))
        poly.assert_called_once()


if __name__ == '__main__':
    unittest.main()
