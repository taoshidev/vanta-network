"""
Nasdaq Basic quote ingestion and validation (phase 1, shadow mode).

Covers:
- RecentEventTracker.get_latest_event_at_or_before (no look-ahead)
- Quote message conversion, 250ms tracker sampling, out-of-order handling
- Entitlement detection from subscribe status replies, reconnect spacing
- Every validity rule in PolygonNasdaqBasicDataService.get_valid_quote
- LivePriceFetcher FMV lookup and shadow fill logging
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
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker


def quote_msg(t_ms, bid, ask, bid_size=100, ask_size=100, sym='AAPL'):
    return {'ev': 'Q', 'sym': sym, 'bx': 12, 'bp': bid, 'bs': bid_size, 'ax': 12, 'ap': ask, 'as': ask_size,
            't': t_ms, 'z': 3}


def fmv_source(price, t_ms):
    return PriceSource(source='Polygon_ws', timespan_ms=0, open=price, close=price, vwap=price, high=price,
                       low=price, start_ms=t_ms, websocket=True, lag_ms=0, bid=price, ask=price)


class TestRecentEventTrackerAtOrBefore(unittest.TestCase):
    def test_never_returns_a_later_event(self):
        now_ms = TimeUtil.now_in_millis()
        tracker = RecentEventTracker()
        early = fmv_source(100.0, now_ms - 2000)
        late = fmv_source(101.0, now_ms - 1000)
        tracker.add_event(early)
        tracker.add_event(late)

        self.assertIsNone(tracker.get_latest_event_at_or_before(now_ms - 2001))
        self.assertIs(tracker.get_latest_event_at_or_before(now_ms - 2000), early)
        self.assertIs(tracker.get_latest_event_at_or_before(now_ms - 1001), early)
        self.assertIs(tracker.get_latest_event_at_or_before(now_ms - 1000), late)
        self.assertIs(tracker.get_latest_event_at_or_before(now_ms), late)
        # get_closest_event would pick the later event here; at-or-before must not
        self.assertIs(tracker.get_closest_event(now_ms - 1400), late)
        self.assertIs(tracker.get_latest_event_at_or_before(now_ms - 1400), early)


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
        self.feed(quote_msg(self.now_ms - 100, 99.0, 101.0))
        ps = self.svc.latest_websocket_events['AAPL']
        self.assertEqual(ps.source, NASDAQ_BASIC_SOURCE)
        self.assertEqual((ps.bid, ps.ask), (99.0, 101.0))
        self.assertEqual(ps.open, 100.0)
        self.assertEqual(ps.start_ms, self.now_ms - 100)
        self.assertTrue(ps.websocket)

    def test_zero_size_side_is_zeroed(self):
        self.feed(quote_msg(self.now_ms, 99.0, 101.0, bid_size=0))
        ps = self.svc.latest_websocket_events['AAPL']
        self.assertEqual(ps.bid, 0.0)
        self.assertEqual(ps.ask, 101.0)

    def test_unknown_and_non_equity_symbols_ignored(self):
        self.feed(quote_msg(self.now_ms, 1.0, 2.0, sym='NOTATICKER'), {'ev': 'T', 'sym': 'AAPL', 'p': 100, 't': self.now_ms})
        self.assertEqual(self.svc.latest_websocket_events, {})

    def test_tracker_keeps_last_quote_of_each_window(self):
        t0 = self.now_ms - 10_000
        sample = ValiConfig.NASDAQ_QUOTE_TRACKER_SAMPLE_MS
        self.feed(quote_msg(t0, 99.0, 101.0),
                  quote_msg(t0 + 100, 99.1, 101.0),
                  quote_msg(t0 + 200, 99.2, 101.0),   # last of window 1
                  quote_msg(t0 + sample + 50, 99.3, 101.0),   # opens window 2, flushes t0+200
                  quote_msg(t0 + sample + 100, 99.4, 101.0),  # last of window 2
                  quote_msg(t0 + 2 * sample + 60, 99.5, 101.0))  # opens window 3, flushes window 2
        stored = self.svc.trade_pair_to_recent_events['AAPL'].get_events_in_range(t0 - 1, self.now_ms)
        self.assertEqual([ps.start_ms for ps in stored], [t0 + 200, t0 + sample + 100])
        self.assertEqual(self.svc.latest_websocket_events['AAPL'].start_ms, t0 + 2 * sample + 60)

    def test_slightly_late_quote_becomes_latest_with_clamped_time(self):
        # Observed live: quotes can arrive 1-15ms behind the previous quote's timestamp
        self.feed(quote_msg(self.now_ms, 99.0, 101.0), quote_msg(self.now_ms - 5, 99.5, 100.5))
        latest = self.svc.latest_websocket_events['AAPL']
        self.assertEqual((latest.bid, latest.ask), (99.5, 100.5))
        self.assertEqual(latest.start_ms, self.now_ms)
        self.assertEqual(self.svc.n_quotes_out_of_order, 1)

    def test_far_out_of_order_quote_dropped(self):
        late_ms = ValiConfig.NASDAQ_QUOTE_OUT_OF_ORDER_TOLERANCE_MS + 500
        self.feed(quote_msg(self.now_ms, 99.0, 101.0), quote_msg(self.now_ms - late_ms, 50.0, 51.0))
        latest = self.svc.latest_websocket_events['AAPL']
        self.assertEqual((latest.bid, latest.ask), (99.0, 101.0))
        self.assertEqual(self.svc.n_quotes_out_of_order, 1)

    def test_get_quote_at_or_before_uses_latest_then_tracker(self):
        t0 = self.now_ms - 10_000
        sample = ValiConfig.NASDAQ_QUOTE_TRACKER_SAMPLE_MS
        self.feed(quote_msg(t0, 99.0, 101.0), quote_msg(t0 + sample, 98.0, 100.0))
        self.assertEqual(self.svc.get_quote_at_or_before(TradePair.AAPL, self.now_ms).start_ms, t0 + sample)
        self.assertEqual(self.svc.get_quote_at_or_before(TradePair.AAPL, t0 + sample - 1).start_ms, t0)
        self.assertIsNone(self.svc.get_quote_at_or_before(TradePair.AAPL, t0 - 1))

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

    # ---------- validity rules ----------

    def assert_rejected(self, reason, time_ms=None, fmv_ps=None, trade_pair=TradePair.AAPL, max_age_ms=None):
        quote, got = self.svc.get_valid_quote(trade_pair, time_ms or TimeUtil.now_in_millis(), fmv_ps, max_age_ms)
        self.assertIsNone(quote)
        self.assertEqual(got, reason)

    def assert_valid(self, time_ms=None, fmv_ps=None, max_age_ms=None):
        quote, reason = self.svc.get_valid_quote(TradePair.AAPL, time_ms or TimeUtil.now_in_millis(), fmv_ps, max_age_ms)
        self.assertIsNone(reason)
        self.assertIsNotNone(quote)
        return quote

    def test_valid_quote_passes(self):
        self.feed(quote_msg(self.now_ms - 100, 99.99, 100.01))
        quote = self.assert_valid(fmv_ps=fmv_source(100.0, self.now_ms - 50))
        self.assertEqual((quote.bid, quote.ask), (99.99, 100.01))
        self.assertEqual(self.svc.n_valid, 1)

    def test_disabled(self):
        self.feed(quote_msg(self.now_ms, 99.99, 100.01))
        self.svc.set_test_entitlement(False)
        self.assert_rejected('disabled')

    def test_not_applicable_for_non_vanta_equities(self):
        self.assert_rejected('not_applicable', trade_pair=TradePair.EURUSD)
        self.assert_rejected('not_applicable', trade_pair=TradePair.AAPLUSDC)

    def test_market_closed(self):
        self.feed(quote_msg(self.now_ms, 99.99, 100.01))
        self.svc.set_test_market_open(False)
        self.assert_rejected('market_closed')

    def test_no_quote(self):
        self.assert_rejected('no_quote')

    def test_feed_unhealthy(self):
        self.feed(quote_msg(self.now_ms, 99.99, 100.01))
        self.svc.last_message_ms = TimeUtil.now_in_millis() - ValiConfig.NASDAQ_QUOTE_FEED_HEALTH_MS - 1000
        self.assert_rejected('feed_unhealthy')

    def test_stale(self):
        self.feed(quote_msg(self.now_ms - ValiConfig.NASDAQ_QUOTE_MAX_AGE_MS - 1000, 99.99, 100.01))
        self.assert_rejected('stale')

    def test_quote_unchanged_for_25s_is_still_valid(self):
        # Mid-caps can go 25s between quote changes; the quote stays valid until replaced
        self.feed(quote_msg(self.now_ms - 25_000, 99.99, 100.01))
        self.assert_valid()

    def test_max_age_override(self):
        self.feed(quote_msg(self.now_ms - 2000, 99.99, 100.01))
        self.assert_rejected('stale', max_age_ms=1000)

    def test_missing_side(self):
        self.feed(quote_msg(self.now_ms, 99.99, 100.01, ask_size=0))
        self.assert_rejected('missing_side')

    def test_bad_current_quote_does_not_fall_back_to_older_good_quote(self):
        self.feed(quote_msg(self.now_ms - 1000, 99.99, 100.01), quote_msg(self.now_ms, 100.02, 100.01))
        self.assert_rejected('crossed_or_locked')

    def test_locked(self):
        self.feed(quote_msg(self.now_ms, 100.0, 100.0))
        self.assert_rejected('crossed_or_locked')

    def test_wide_spread(self):
        # 60 bps spread > 50 bps cap
        self.feed(quote_msg(self.now_ms, 99.70, 100.30))
        self.assert_rejected('wide_spread')

    def test_fmv_deviation(self):
        self.feed(quote_msg(self.now_ms, 99.99, 100.01))
        self.assert_rejected('fmv_deviation', fmv_ps=fmv_source(100.5, self.now_ms))  # ~50 bps away

    def test_fmv_band_widens_to_spread(self):
        # 40 bps spread, mid 30 bps from FMV: inside max(25, 40)
        self.feed(quote_msg(self.now_ms, 100.10, 100.50))
        self.assert_valid(fmv_ps=fmv_source(100.0, self.now_ms))

    def test_old_fmv_not_used_for_band(self):
        self.feed(quote_msg(self.now_ms, 99.99, 100.01))
        self.assert_valid(fmv_ps=fmv_source(105.0, self.now_ms - ValiConfig.NASDAQ_QUOTE_FMV_MAX_AGE_MS - 1000))

    def test_rejections_counted_by_reason(self):
        self.assert_rejected('no_quote')
        self.assert_rejected('no_quote')
        self.assertEqual(self.svc.rejection_counts['no_quote'], 2)


class TestLivePriceFetcherNasdaqQuotes(unittest.TestCase):
    def setUp(self):
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True, running_unit_tests=True)
        self.svc = self.fetcher.polygon_nasdaq_basic_data_service
        self.svc.set_test_market_open(True)
        self.now_ms = TimeUtil.now_in_millis()

    def enable_with_quote(self, bid, ask):
        self.svc.set_test_entitlement(True)
        self.svc._process_raw(json.dumps([quote_msg(self.now_ms - 100, bid, ask)]), TimeUtil.now_in_millis())

    def test_disabled_by_default(self):
        self.assertFalse(self.fetcher.nasdaq_quotes_enabled())
        self.assertEqual(self.fetcher.get_valid_nasdaq_quote(TradePair.AAPL, self.now_ms), (None, 'disabled'))

    def test_fmv_band_uses_polygon_fmv_at_or_before(self):
        self.enable_with_quote(99.99, 100.01)
        self.fetcher.polygon_data_service.trade_pair_to_recent_events['AAPL'].add_event(fmv_source(100.0, self.now_ms - 200))
        quote, reason = self.fetcher.get_valid_nasdaq_quote(TradePair.AAPL, self.now_ms)
        self.assertIsNone(reason)
        # An FMV after the lookup time is ignored, so the band check still uses the earlier FMV
        self.fetcher.polygon_data_service.trade_pair_to_recent_events['AAPL'].add_event(fmv_source(110.0, self.now_ms + 500))
        self.assertIsNone(self.fetcher.get_valid_nasdaq_quote(TradePair.AAPL, self.now_ms)[1])
        self.fetcher.polygon_data_service.trade_pair_to_recent_events['AAPL'].add_event(fmv_source(101.0, self.now_ms - 50))
        self.assertEqual(self.fetcher.get_valid_nasdaq_quote(TradePair.AAPL, self.now_ms)[1], 'fmv_deviation')

    def test_shadow_log_reports_would_be_fill(self):
        self.enable_with_quote(99.9, 100.0)
        with mock.patch('vali_objects.price_fetcher.live_price_fetcher.logger') as log:
            self.fetcher.log_nasdaq_shadow_fill(TradePair.AAPL, self.now_ms, OrderType.LONG, OrderType.LONG,
                                                99.95, 'Polygon_ws', 'uuid-1')
            self.fetcher.log_nasdaq_shadow_fill(TradePair.AAPL, self.now_ms, OrderType.FLAT, OrderType.LONG,
                                                99.95, 'Polygon_ws', 'uuid-2')
        buy_line, sell_line = [c.args[0] for c in log.info.call_args_list]
        self.assertIn('nasdaq fill=100.0', buy_line)
        self.assertIn('diff=+5.00bps', buy_line)
        self.assertIn('nasdaq fill=99.9', sell_line)
        self.assertIn('diff=-5.00bps', sell_line)

    def test_shadow_log_reports_rejection_reason(self):
        self.enable_with_quote(99.0, 101.0)  # 200 bps spread
        with mock.patch('vali_objects.price_fetcher.live_price_fetcher.logger') as log:
            self.fetcher.log_nasdaq_shadow_fill(TradePair.AAPL, self.now_ms, OrderType.LONG, OrderType.LONG,
                                                100.0, 'Polygon_ws', 'uuid-3')
        self.assertIn('no valid quote (wide_spread)', log.info.call_args.args[0])

    def test_shadow_log_silent_when_disabled_or_not_equities(self):
        with mock.patch('vali_objects.price_fetcher.live_price_fetcher.logger') as log:
            self.fetcher.log_nasdaq_shadow_fill(TradePair.AAPL, self.now_ms, OrderType.LONG, OrderType.LONG,
                                                100.0, 'Polygon_ws', 'uuid-4')
            self.svc.set_test_entitlement(True)
            self.fetcher.log_nasdaq_shadow_fill(TradePair.EURUSD, self.now_ms, OrderType.LONG, OrderType.LONG,
                                                1.1, 'Polygon_ws', 'uuid-5')
        log.info.assert_not_called()


if __name__ == '__main__':
    unittest.main()
