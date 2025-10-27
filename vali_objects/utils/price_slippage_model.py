import math
from collections import defaultdict
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig, ForexSubcategory
from vali_objects.vali_dataclasses.order import Order

SLIPPAGE_V2_TIME_MS = 1759431540000

class PriceSlippageModel:
    features = defaultdict(dict)
    parameters: dict = {}
    slippage_estimates: dict = {}
    live_price_fetcher: LivePriceFetcher = None
    holidays_nyse = None
    eastern_tz = ZoneInfo("America/New_York")
    is_backtesting = False
    fetch_slippage_data = False
    recalculate_slippage = False
    capital = ValiConfig.DEFAULT_CAPITAL
    last_refresh_time_ms = 0

    def __init__(self, live_price_fetcher=None, running_unit_tests=False, is_backtesting=False,
                 fetch_slippage_data=False, recalculate_slippage=False, capital=ValiConfig.DEFAULT_CAPITAL):
        if not PriceSlippageModel.parameters:
            PriceSlippageModel.holidays_nyse = holidays.financial_holidays('NYSE')
            PriceSlippageModel.parameters = self.read_slippage_model_parameters()

            if live_price_fetcher is None:
                secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
                live_price_fetcher = LivePriceFetcher(secrets, disable_ws=False)
            PriceSlippageModel.live_price_fetcher = live_price_fetcher

        PriceSlippageModel.is_backtesting = is_backtesting
        PriceSlippageModel.fetch_slippage_data = fetch_slippage_data
        PriceSlippageModel.recalculate_slippage = recalculate_slippage
        PriceSlippageModel.capital = capital

    @classmethod
    def calculate_slippage(cls, bid:float, ask:float, order:Order):
        """
        returns the percentage slippage of the current order.
        each asset class uses a unique model
        """
        if not PriceSlippageModel.slippage_estimates:
            PriceSlippageModel.slippage_estimates = cls.read_slippage_estimates()

        trade_pair = order.trade_pair
        if bid * ask == 0:
            if not trade_pair.is_crypto:  # For now, crypto does not have bid/ask data
                bt.logging.warning(f'Tried to calculate slippage with bid: {bid} and ask: {ask}. order: {order}. Returning 0')
            return 0  # Need valid bid and ask.
        if abs(order.value) <= 1000:
            return 0
        cls.refresh_features_daily(order.processed_ms, write_to_disk=not cls.is_backtesting)

        if trade_pair.is_equities:
            slippage_percentage = cls.calc_slippage_equities(bid, ask, order)
        elif trade_pair.is_forex:
            slippage_percentage = cls.calc_slippage_forex(bid, ask, order)
        elif trade_pair.is_crypto:
            slippage_percentage = cls.calc_slippage_crypto(order)
        else:
            raise ValueError(f"Invalid trade pair {trade_pair.trade_pair_id} to calculate slippage")
        return float(np.clip(slippage_percentage, 0.0, 0.03))

    @classmethod
    def calc_slippage_equities(cls, bid:float, ask:float, order:Order) -> float:
        """
        Fitted BB+ model (dec 2024)
        Slippage percentage = intercept + c1 * spread/price + c2 * annualized_volatility + c3 * order_volume/avg_daily_volume

        Use the direct BB+ model for pre jan 2025 orders
        slippage percentage = 0.433 * spread/mid_price + 0.335 * sqrt(annualized_volatility**2 / 3 / 250) * sqrt(volume / (0.3 * estimated daily volume))
        """
        order_date = TimeUtil.millis_to_short_date_str(order.processed_ms)
        annualized_volatility = cls.features[order_date]["vol"][order.trade_pair.trade_pair_id]
        avg_daily_volume = cls.features[order_date]["adv"][order.trade_pair.trade_pair_id]
        spread = ask - bid
        mid_price = (bid + ask) / 2

        size = abs(order.leverage) * ValiConfig.DEFAULT_CAPITAL
        volume_shares = size / mid_price

        if order.processed_ms > 1735718400000:  # Use fitted BB+ for orders after jan 1, 2024, 08:00:00 UTC
            size_str = cls.get_order_size_bucket(size)
            side = "buy" if order.leverage > 0 else "sell"
            model_config = cls.parameters["equity"][order.trade_pair.trade_pair_id][side][size_str]
            intercept, c1, c2, c3 = (model_config[key] for key in ["intercept", "spread/price", "annualized_vol", f"{side}_order_size/adv"])
            slippage_pct = intercept + (c1 * spread / mid_price) + (c2 * annualized_volatility) + (c3 * volume_shares / avg_daily_volume)
        else:
            # direct BB+ model for orders before
            term1 = 0.433 * spread / mid_price
            term2 = 0.335 * math.sqrt(annualized_volatility ** 2 / 3 / 250)
            term3 = math.sqrt(volume_shares / (0.3 * avg_daily_volume))
            slippage_pct = term1 + term2 * term3
        return slippage_pct

    @classmethod
    def calc_slippage_forex(cls, bid:float, ask:float, order:Order) -> float:
        """
        V2: 5 bps slippage daily from 5-6 pm EST, and 2 bps slippage otherwise

        V1: Using the direct BB+ model as a stand-in for forex
        slippage percentage = 0.433 * spread/mid_price + 0.335 * sqrt(annualized_volatility**2 / 3 / 250) * sqrt(volume / (0.3 * estimated daily volume))
        """
        if order.processed_ms > SLIPPAGE_V2_TIME_MS:
            order_datetime = TimeUtil.millis_to_datetime(order.processed_ms).astimezone(cls.eastern_tz)
            hour = order_datetime.hour
            if 17 <= hour < 18:    # Daily 5-6 pm EST. Higher slippage during market open/closing hours
                return 0.0005       # 5 bps
            else:
                return 0.0002       # 2 bps

        order_date = TimeUtil.millis_to_short_date_str(order.processed_ms)
        annualized_volatility = cls.features[order_date]["vol"][order.trade_pair.trade_pair_id]
        avg_daily_volume = cls.features[order_date]["adv"][order.trade_pair.trade_pair_id]
        spread = ask - bid
        mid_price = (bid + ask) / 2

        # bt.logging.info(f"bid: {bid}, ask: {ask}, adv: {avg_daily_volume}, vol: {annualized_volatility}")

        size = abs(order.leverage) * ValiConfig.DEFAULT_CAPITAL
        base, _ = order.trade_pair.trade_pair.split("/")
        base_to_usd_conversion = cls.live_price_fetcher.polygon_data_service.get_currency_conversion(base=base, quote="USD") if base != "USD" else 1  # TODO: fallback?
        # print(base_to_usd_conversion)
        volume_standard_lots = size / (100_000 * base_to_usd_conversion)  # Volume expressed in terms of standard lots (1 std lot = 100,000 base currency)

        term1 = 0.433 * spread / mid_price
        term2 = 0.335 * math.sqrt(annualized_volatility ** 2 / 3 / 250)
        term3 = math.sqrt(volume_standard_lots / (0.3 * avg_daily_volume))
        slippage_pct = term1 + term2 * term3
        # bt.logging.info(f"slippage_pct: {slippage_pct}")
        return slippage_pct

    @classmethod
    def calc_slippage_crypto(cls, order:Order) -> float:
        """
        V2: price slippage model

        V1: 0.2 bps for majors, 2 bps for alts
        """
        if order.processed_ms > SLIPPAGE_V2_TIME_MS:
            side = "long" if order.leverage > 0 else "short"
            slippage_size_buckets = cls.slippage_estimates["crypto"][order.trade_pair.trade_pair_id+"C"][side]
            last_slippage = 0
            for bucket, slippage in slippage_size_buckets.items():
                low, high = bucket[1:-1].split(",")
                last_slippage = slippage
                if int(low) <= abs(order.value) < int(high):
                    return last_slippage * 3     # conservative 3x multiplier on slippage
            return last_slippage * 3

        trade_pair = order.trade_pair
        if trade_pair in [TradePair.BTCUSD, TradePair.ETHUSD]:
            return 0.00002
        elif trade_pair in [TradePair.SOLUSD, TradePair.XRPUSD, TradePair.DOGEUSD, TradePair.ADAUSD]:
            return 0.0002
        else:
            raise ValueError(f"Unknown crypto slippage for trade pair {trade_pair.trade_pair_id}")


    @classmethod
    def refresh_features_daily(cls, time_ms:int=None, write_to_disk:bool=True):
        """
        Calculate and store model features (average daily volume and annualized volatility) for new days
        """
        if not time_ms:
            time_ms = TimeUtil.now_in_millis()
        current_date = TimeUtil.millis_to_short_date_str(time_ms)

        if current_date in cls.features:
            return

        if not cls.is_backtesting and time_ms - cls.last_refresh_time_ms < 1000:
            return

        bt.logging.info(
            f"Calculating avg daily volume and annualized volatility for new day UTC {current_date}")
        trade_pairs = [tp for tp in TradePair if tp.is_forex or tp.is_equities]
        tp_to_adv, tp_to_vol = cls.get_features(trade_pairs=trade_pairs, processed_ms=time_ms)
        if tp_to_adv and tp_to_vol:
            cls.features[current_date] = {
                "adv": tp_to_adv,
                "vol": tp_to_vol
            }

            if write_to_disk:
                cls.write_features_from_memory_to_disk()
            bt.logging.info(
                    f"Completed refreshing avg daily volume and annualized volatility for new day UTC {current_date}")
        else:
            bt.logging.info(f"Skipping feature update for {current_date} due to missing data. tp_to_adv: {bool(tp_to_adv)}, tp_to_vol: {bool(tp_to_vol)}")
        cls.last_refresh_time_ms = time_ms

    @classmethod
    def get_features(cls, trade_pairs: list[TradePair], processed_ms: int, adv_lookback_window: int = 10,
                     calc_vol_window: int = 30):
        """
        return dict of features (avg daily volume and annualized volatility) for each trade pair
        """
        tp_to_adv = defaultdict()
        tp_to_vol = defaultdict()
        for trade_pair in trade_pairs:
            try:
                bars_df = cls.get_bars_with_features(trade_pair, processed_ms, adv_lookback_window, calc_vol_window)
                row_selected = bars_df.iloc[-1]
                annualized_volatility = row_selected['annualized_vol']
                avg_daily_volume = row_selected[f'adv_last_{adv_lookback_window}_days']

                tp_to_vol[trade_pair.trade_pair_id] = annualized_volatility
                tp_to_adv[trade_pair.trade_pair_id] = avg_daily_volume
            except Exception as e:
                bt.logging.info(f"Unable to calculate slippage model features for trade pair {trade_pair.trade_pair_id} with exception {e}")
        return tp_to_adv, tp_to_vol

    @classmethod
    def get_bars_with_features(cls, trade_pair: TradePair, processed_ms: int, adv_lookback_window: int=10, calc_vol_window: int=30, trading_days_in_a_year: int=252) -> pd.DataFrame:
        """
        fetch data for average daily volume (estimated daily volume) and annualized volatility
        """
        order_date = TimeUtil.millis_to_short_date_str(processed_ms)
        days_ago = max(adv_lookback_window, calc_vol_window) + 4  # +1 for last day, +1 because daily_returns is NaN for 1st day, +2 for padding (unexpected holidays)
        start_date = cls.holidays_nyse.get_nth_working_day(order_date, -days_ago).strftime("%Y-%m-%d")

        price_info_raw = cls.live_price_fetcher.polygon_data_service.unified_candle_fetcher(trade_pair, start_date, order_date, timespan="day")
        aggs = []
        try:
            for a in price_info_raw:
                aggs.append(a)
        except Exception as e:
            print(f"Error fetching data from Polygon: {e}")

        bars_pd = pd.DataFrame(aggs)
        bars_pd['datetime'] = pd.to_datetime(bars_pd['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
        bars_pd[f'adv_last_{adv_lookback_window}_days'] = (bars_pd['volume'].rolling(window=adv_lookback_window + 1).sum() - bars_pd['volume']) / adv_lookback_window  # excluding the current day when calculating adv
        bars_pd['daily_returns'] = np.log(bars_pd["close"] / bars_pd["close"].shift(1))
        # excluding the current day when calculating vol,
        bars_pd[f'rolling_avg_daily_vol_{calc_vol_window}d'] = bars_pd['daily_returns'].rolling(window=calc_vol_window, closed='left').std()  # requires +2 days
        bars_pd["annualized_vol"] = bars_pd[f'rolling_avg_daily_vol_{calc_vol_window}d'] * np.sqrt(trading_days_in_a_year)
        return bars_pd

    @classmethod
    def write_features_from_memory_to_disk(cls):
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_slippage_model_features_file(),
            cls.features
        )

    @staticmethod
    def get_order_size_bucket(size: float) -> str:
        """
        bucket the order size float into a string for reading from model parameters file
        """
        all_order_value_ranges = [(1_000, 10_000), (10_000, 50_000), (50_000, 100_000), (100_000, 200_000), (200_000, 300_000)]
        order_value_labels = ["1k_10k", "10k_50k", "50k_100k", "100k_200k", "200k_300k"]

        for (low, high), label in zip(all_order_value_ranges, order_value_labels):
            if low < size <= high:
                return label

        # TODO: order size range
        if size > 300_000:
            raise ValueError("Order size must be less than $300K")
        else:
            raise ValueError("Order size must be at least $1K")

    @staticmethod
    def read_slippage_model_parameters() -> dict:
        equity_parameters = ValiUtils.get_vali_json_file_dict(ValiBkpUtils.get_slippage_model_parameters_file())
        return equity_parameters

    @staticmethod
    def read_slippage_estimates() -> dict:
        slippage_estimates = ValiUtils.get_vali_json_file_dict(ValiBkpUtils.get_slippage_estimates_file())
        return slippage_estimates

    def update_historical_slippage(self, positions_at_t_f):
        assert self.is_backtesting, "This method is only for backtesting"
        #mutates the original orders
        bt.logging.info("Starting slippage order pre-processing.")
        for hk, positions in positions_at_t_f.items():
            for position in positions:
                order_updated = False
                for o in position.orders:
                    #Check if orders need to have slippage calculations
                    if o.bid == 0 and o.ask == 0 and o.slippage == 0 and not (self.recalculate_slippage or self.fetch_slippage_data):
                        bt.logging.info(f"Not recalculating slippage, but order doesn't have these attributes set: {o}")
                        break

                    if not (self.recalculate_slippage or self.fetch_slippage_data):
                        break

                    bt.logging.info(f"updating order attributes {o}")
                    bid = o.bid
                    ask = o.ask

                    if self.fetch_slippage_data:

                        price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair=o.trade_pair, time_ms=o.processed_ms)
                        if not price_sources:
                            raise ValueError(
                                f"Ignoring order for [{hk}] due to no live prices being found for trade_pair [{o.trade_pair}]. Please try again.")
                        best_price_source = price_sources[0]
                        bid = best_price_source.bid
                        ask = best_price_source.ask

                    slippage = self.calculate_slippage(bid, ask, o)
                    o.bid = bid
                    o.ask = ask
                    o.slippage = slippage
                    bt.logging.info(f"updated order attributes {o}")
                    order_updated = True
                if order_updated:
                    position.rebuild_position_with_updated_orders(self.live_price_fetcher)


if __name__ == "__main__":
    psm = PriceSlippageModel()
    equities_order_buy = Order(price=100, processed_ms=TimeUtil.now_in_millis() - 1000 * 200,
                                    order_uuid="test_order",
                                    trade_pair=TradePair.NVDA,
                                    order_type=OrderType.LONG, leverage=1)
    slippage_buy = PriceSlippageModel.calculate_slippage(bid=99, ask=100, order=equities_order_buy)
    print(slippage_buy)
