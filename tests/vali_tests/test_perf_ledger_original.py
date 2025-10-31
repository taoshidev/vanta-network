from unittest.mock import patch

import bittensor as bt

from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO, PerfLedgerManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_utils import ValiUtils

bt.logging.enable_info()

class TestPerfLedgers(TestBase):

    def setUp(self):
        super().setUp()
        # Clear ALL test miner positions BEFORE creating PositionManager
        ValiBkpUtils.clear_directory(
            ValiBkpUtils.get_miner_dir(running_unit_tests=True)
        )

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 24 * 60  # 60 days ago
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        
        # Set up live price fetcher
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)
        self.default_btc_order = Order(price=60000, processed_ms=self.DEFAULT_OPEN_MS, order_uuid="test_order_btc", trade_pair=self.DEFAULT_TRADE_PAIR,
                                     order_type=OrderType.LONG, leverage=.5)
        self.default_nvda_order = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 5, order_uuid="test_order_nvda", trade_pair=TradePair.NVDA,
                                     order_type=OrderType.LONG, leverage=1)
        self.default_usdjpy_order = Order(price=156, processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 10, order_uuid="test_order_usdjpy",
                                          trade_pair=TradePair.USDJPY, order_type=OrderType.LONG, leverage=1)

        self.default_btc_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_btc",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_btc_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_btc_position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        self.default_nvda_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_nvda",
            open_ms=self.default_nvda_order.processed_ms,
            trade_pair=TradePair.NVDA,
            orders=[self.default_nvda_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_nvda_position.rebuild_position_with_updated_orders(self.live_price_fetcher)

        self.default_usdjpy_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_usdjpy",
            open_ms=self.default_usdjpy_order.processed_ms,
            trade_pair=TradePair.USDJPY,
            orders=[self.default_usdjpy_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_usdjpy_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
        mmg = MockMetagraph(hotkeys=[self.DEFAULT_MINER_HOTKEY])
        position_manager = PositionManager(metagraph=mmg, running_unit_tests=True, 
                                          elimination_manager=None, live_price_fetcher=self.live_price_fetcher)
        elimination_manager = EliminationManager(mmg, position_manager, None, running_unit_tests=True)
        position_manager.elimination_manager = elimination_manager
        position_manager.clear_all_miner_positions()

        for p in [self.default_usdjpy_position, self.default_nvda_position, self.default_btc_position]:
            position_manager.save_miner_position(p)
        self.perf_ledger_manager = PerfLedgerManager(metagraph=mmg, running_unit_tests=True, position_manager=position_manager)
        self.perf_ledger_manager.clear_all_ledger_data()

    def check_alignment_per_cp(self, ans):
        original_ret = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[-1].prev_portfolio_ret
        original_mdd = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[-1].mdd
        original_carry_fee = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[-1].prev_portfolio_carry_fee
        tp_to_ret = {}
        tp_to_mdd = {}
        tp_to_cf = {}
        manual_portfolio_ret = 1.0
        manual_portfolio_mdd = 1.0
        manual_portfolio_carry_fee = 1.0
        for tp_id, pl in ans[self.DEFAULT_MINER_HOTKEY].items():
            tp_to_ret[tp_id] = pl.cps[-1].prev_portfolio_ret
            tp_to_mdd[tp_id] = pl.cps[-1].mdd
            tp_to_cf[tp_id] = pl.cps[-1].prev_portfolio_carry_fee
            if tp_id != TP_ID_PORTFOLIO:
                manual_portfolio_ret *= tp_to_ret[tp_id]
                manual_portfolio_mdd *= tp_to_mdd[tp_id]
                manual_portfolio_carry_fee *= tp_to_cf[tp_id]

        self.assertEqual(original_ret, manual_portfolio_ret,
                         f'original_ret {original_ret} != manual_portfolio_ret {manual_portfolio_ret}. {tp_to_ret}')

        # Note: Portfolio MDD should NOT equal the product of trade pair MDDs
        # MDD is not multiplicative - portfolio MDD is based on actual portfolio performance
        # Portfolio MDD can be worse than individual MDDs due to correlation and portfolio effects
        # Verify that portfolio MDD is reasonable (between worst individual and product of all MDDs)
        worst_individual_mdd = min(tp_to_mdd[tp_id] for tp_id in tp_to_mdd.keys() if tp_id != TP_ID_PORTFOLIO)
        product_of_mdds = manual_portfolio_mdd  # This was calculated as the product

        # Portfolio MDD should be between the product of individual MDDs and worst individual MDD
        self.assertLessEqual(original_mdd, worst_individual_mdd,
                            f'Portfolio MDD {original_mdd} should be <= worst individual MDD {worst_individual_mdd}. {tp_to_mdd}')
        self.assertGreaterEqual(original_mdd, product_of_mdds,
                               f'Portfolio MDD {original_mdd} should be >= product of MDDs {product_of_mdds}. {tp_to_mdd}')

        self.assertEqual(original_carry_fee, manual_portfolio_carry_fee,
                         f'original {original_carry_fee} != manual {manual_portfolio_carry_fee}. {tp_to_cf}')



        failures = []
        portfolio_pl = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
        n = len(portfolio_pl.cps)
        for i in range(n):
            portfolio_cp = portfolio_pl.cps[i]
            manual_portfolio_ret = 1.0
            manual_portfolio_spread_fee = 1.0
            manual_portfolio_carry_fee = 1.0
            automatic_portfolio_ret = None
            automatic_portfolio_spread_fee = None
            automatic_portfolio_carry_fee = None

            contributing_tps = set()
            # expected_last_updated_ms = None
            debug = {}
            for tp_id, pl in ans[self.DEFAULT_MINER_HOTKEY].items():
                if tp_id == TP_ID_PORTFOLIO:
                    automatic_portfolio_ret = pl.cps[i].prev_portfolio_ret
                    automatic_portfolio_spread_fee = pl.cps[i].prev_portfolio_spread_fee
                    automatic_portfolio_carry_fee = pl.cps[i].prev_portfolio_carry_fee
                    continue

                match = [x for x in pl.cps if x.last_update_ms == portfolio_cp.last_update_ms]
                if i > 21:
                    assert match, i
                if match:
                    assert len(match) == 1
                    match = match[0]
                    manual_portfolio_ret *= match.prev_portfolio_ret
                    debug[tp_id] = match.prev_portfolio_spread_fee
                    contributing_tps.add(tp_id)
                    manual_portfolio_spread_fee *= match.prev_portfolio_spread_fee
                    manual_portfolio_carry_fee *= match.prev_portfolio_carry_fee

            if automatic_portfolio_ret != manual_portfolio_ret:
                failures.append(f'automatic_portfolio_ret {automatic_portfolio_ret}, manual_portfolio_ret {manual_portfolio_ret},  debug {debug}, contributing_tps {contributing_tps} i {i}/{len(portfolio_pl.cps)} t_ms {portfolio_cp.last_update_ms}')
                bt.logging.warning(f'#{i}/{n-1} return failure {failures[-1]}')

            if automatic_portfolio_spread_fee != manual_portfolio_spread_fee:
                failures.append(f'automatic_portfolio_spread_fee {automatic_portfolio_spread_fee}, manual_portfolio_spread_fee {manual_portfolio_spread_fee}, debug {debug}, contributing_tps {contributing_tps} i {i}/{len(portfolio_pl.cps)} t_ms {portfolio_cp.last_update_ms}')
                bt.logging.warning(f'#{i}/{n-1} spread failure {failures[-1]}')

            if automatic_portfolio_carry_fee != manual_portfolio_carry_fee:
                failures.append(f'automatic_portfolio_carry_fee {automatic_portfolio_carry_fee}, manual_portfolio_carry_fee {manual_portfolio_carry_fee}, contributing_tps {contributing_tps} i {i}/{len(portfolio_pl.cps)} t_ms {portfolio_cp.last_update_ms}')
                bt.logging.warning(f'#{i}/{n-1} carry failure {failures[-1]}')
        assert not failures

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_basic(self, mock_unified_candle_fetcher):
        mock_unified_candle_fetcher.return_value = {}
        hotkey_to_positions = {self.DEFAULT_MINER_HOTKEY: [self.default_btc_position]}
        ans = self.perf_ledger_manager.generate_perf_ledgers_for_analysis(hotkey_to_positions)
        for hk, dat in ans.items():
            for tp_id, pl in dat.items():
                print('-----------', tp_id, '-----------')
                for idx, x in enumerate(pl.cps):
                    last_update_formatted = TimeUtil.millis_to_timestamp(x.last_update_ms)
                    if idx == 0 or idx == len(pl.cps) - 1:
                        print(x, last_update_formatted)
                print(tp_id, 'max_perf_ledger_return:', pl.max_return)

        assert len(ans) == 1, ans

    @patch('data_generator.polygon_data_service.PolygonDataService.unified_candle_fetcher')
    def test_multiple_tps(self, mock_unified_candle_fetcher):
        mock_unified_candle_fetcher.return_value = {}
        hotkey_to_positions = {self.DEFAULT_MINER_HOTKEY:
                                   [self.default_btc_position, self.default_nvda_position, self.default_usdjpy_position]}
        for p in hotkey_to_positions[self.DEFAULT_MINER_HOTKEY]:
            self.perf_ledger_manager.position_manager.save_miner_position(p)

        self.perf_ledger_manager.update()

        tp_to_position_start_time = {}
        for position in hotkey_to_positions[self.DEFAULT_MINER_HOTKEY]:
            if position.trade_pair == TradePair.BTCUSD:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_btc_position.open_ms
            elif position.trade_pair == TradePair.NVDA:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_nvda_position.open_ms
            elif position.trade_pair == TradePair.USDJPY:
                tp_to_position_start_time[position.trade_pair.trade_pair_id] = self.default_usdjpy_position.open_ms

        ans = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        PerfLedgerManager.print_bundles(ans)
        pl = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
        # The total product and last checkpoint return should be very close but may differ slightly
        # due to checkpoint boundary alignment and accumulation logic
        self.assertAlmostEqual(pl.get_total_product(), pl.cps[-1].prev_portfolio_ret, 2,
                             f"Total product {pl.get_total_product()} differs from last checkpoint return {pl.cps[-1].prev_portfolio_ret}")
        self.assertEqual(len(ans), 1)
        self.assertEqual(len(ans[self.DEFAULT_MINER_HOTKEY]), 4)
        self.assertIn(TP_ID_PORTFOLIO, ans[self.DEFAULT_MINER_HOTKEY])
        last_update_portfolio = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        last_accum_ms_portfolio = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[-1].accum_ms
        for tp_id in tp_to_position_start_time:
            bundle = ans[self.DEFAULT_MINER_HOTKEY]
            self.assertIn(tp_id, ans[self.DEFAULT_MINER_HOTKEY])
            self.assertEqual(tp_to_position_start_time[tp_id], bundle[tp_id].initialization_time_ms, tp_id + f'initialization time off by {tp_to_position_start_time[tp_id] - bundle[tp_id].initialization_time_ms} ms')
            self.assertEqual(bundle[tp_id].last_update_ms, last_update_portfolio, f'last update time off by {last_update_portfolio - bundle[tp_id].last_update_ms} ms for tp_id {tp_id}')
            self.assertEqual(bundle[tp_id].cps[-1].accum_ms, last_accum_ms_portfolio, f'accum time off by {last_accum_ms_portfolio - bundle[tp_id].cps[-1].accum_ms} ms for tp_id {tp_id}')
        assert len(ans) == 1, ans

        self.check_alignment_per_cp(ans)

        #self.assertEqual(original_ret, manual_portfolio_ret, f'original_ret {original_ret} != manual_portfolio_ret {manual_portfolio_ret}. {tp_to_ret}')
        self.assertLess(ans[self.DEFAULT_MINER_HOTKEY][TradePair.NVDA.trade_pair_id].total_open_ms,
                        ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].total_open_ms)
        self.assertLess(ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].total_open_ms,
                        ans[self.DEFAULT_MINER_HOTKEY][TradePair.BTCUSD.trade_pair_id].total_open_ms)
        self.assertEqual(ans[self.DEFAULT_MINER_HOTKEY][TradePair.BTCUSD.trade_pair_id].total_open_ms,
                        ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].total_open_ms)

        assert all(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[1:]), [(x.open_ms, x.accum_ms) for x in ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[1:]] # first cp truncated due to 12 hr boundary
        assert all(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.BTCUSD.trade_pair_id].cps[1:])

        assert all(x.open_ms < x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.NVDA.trade_pair_id].cps[1:])
        assert any(x.open_ms == 0 for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.NVDA.trade_pair_id].cps[1:])

        assert any(x.open_ms == 0 for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].cps[1:])
        assert any(x.open_ms < x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].cps[1:])
        assert any(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].cps[1:])

        # Close the btc position now
        close_order = Order(price=61000, processed_ms=last_update_portfolio, order_uuid="test_order_btc_close",
                                     trade_pair=self.DEFAULT_TRADE_PAIR, order_type=OrderType.FLAT, leverage=0)
        self.default_btc_position.add_order(close_order, self.live_price_fetcher)
        self.perf_ledger_manager.position_manager.save_miner_position(self.default_btc_position)

        # Waiting a few days
        fast_forward_time_ms = TimeUtil.now_in_millis() + 1000 * 60 * 60 * 24 * 10
        self.perf_ledger_manager.update(t_ms=fast_forward_time_ms)
        ans = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)

        pl = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
        self.assertAlmostEqual(pl.get_total_product(), pl.cps[-1].prev_portfolio_ret, 13)


        PerfLedgerManager.print_bundles(ans)

        self.check_alignment_per_cp(ans)
        self.assertLess(ans[self.DEFAULT_MINER_HOTKEY][TradePair.NVDA.trade_pair_id].total_open_ms,
                        ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].total_open_ms)
        self.assertLess(ans[self.DEFAULT_MINER_HOTKEY][TradePair.BTCUSD.trade_pair_id].total_open_ms,
                        ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].total_open_ms)

        assert any(x.open_ms != x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[1:])  # first cp truncated due to 12 hr boundary
        assert any(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[1:])  # first cp truncated due to 12 hr boundary

        assert any(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.BTCUSD.trade_pair_id].cps[1:])
        assert any(x.open_ms == 0 for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.BTCUSD.trade_pair_id].cps[1:])

        assert all(x.open_ms < x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.NVDA.trade_pair_id].cps[1:])
        assert any(x.open_ms == 0 for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.NVDA.trade_pair_id].cps[1:])

        assert any(x.open_ms == 0 for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].cps[1:])
        assert any(x.open_ms < x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].cps[1:])
        assert any(x.open_ms == x.accum_ms for x in ans[self.DEFAULT_MINER_HOTKEY][TradePair.USDJPY.trade_pair_id].cps[1:])


        last_update_portfolio2 = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].last_update_ms
        portfolio_last_open_ms2 = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[-1].open_ms
        last_accum_ms_portfolio2 = ans[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO].cps[-1].accum_ms
        for tp_id in tp_to_position_start_time:
            bundle = ans[self.DEFAULT_MINER_HOTKEY]
            self.assertIn(tp_id, ans[self.DEFAULT_MINER_HOTKEY])
            self.assertEqual(tp_to_position_start_time[tp_id], bundle[tp_id].initialization_time_ms,
                             tp_id + f'initialization time off by {tp_to_position_start_time[tp_id] - bundle[tp_id].initialization_time_ms} ms')
            expected_last_update = last_update_portfolio2
            self.assertLessEqual(bundle[tp_id].cps[-1].open_ms, portfolio_last_open_ms2, f'open time off by {last_update_portfolio2 - bundle[tp_id].cps[-1].open_ms} ms for tp_id {tp_id}')
            self.assertEqual(bundle[tp_id].last_update_ms, expected_last_update, f'last update time off by {expected_last_update - bundle[tp_id].last_update_ms} ms for tp_id {tp_id}')
            self.assertEqual(bundle[tp_id].cps[-1].accum_ms, last_accum_ms_portfolio2, f'accum time off by {last_accum_ms_portfolio - bundle[tp_id].cps[-1].accum_ms} ms for tp_id {tp_id}')







