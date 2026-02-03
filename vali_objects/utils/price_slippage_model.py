import math
import time
from setproctitle import setproctitle
from shared_objects.error_utils import ErrorUtils
import traceback
from collections import defaultdict
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order

SLIPPAGE_V2_TIME_MS = 1759431540000

class PriceSlippageModel:
    features = defaultdict(dict)
    parameters: dict = {}
    slippage_estimates: dict = {}
    live_price_fetcher = None  # LivePriceFetcherClient - created on first use
    holidays_nyse = None
    eastern_tz = ZoneInfo("America/New_York")
    is_backtesting = False
    fetch_slippage_data = False
    recalculate_slippage = False
    capital = ValiConfig.DEFAULT_CAPITAL
    last_refresh_time_ms = 0
    _running_unit_tests = False

    # Refresh coordination (no lock needed - dict writes are atomic)
    _refresh_in_progress = False
    _refresh_current_date = None

    def __init__(self, running_unit_tests=False, is_backtesting=False,
                 fetch_slippage_data=False, recalculate_slippage=False, capital=ValiConfig.DEFAULT_CAPITAL):
        PriceSlippageModel._running_unit_tests = running_unit_tests
        if not PriceSlippageModel.parameters:
            PriceSlippageModel.holidays_nyse = holidays.financial_holidays('NYSE')
            PriceSlippageModel.parameters = self.read_slippage_model_parameters()

            # Create own LivePriceFetcherClient (forward compatibility - no parameter passing)
            from vali_objects.price_fetcher import LivePriceFetcherClient
            PriceSlippageModel.live_price_fetcher = LivePriceFetcherClient(running_unit_tests=running_unit_tests)

        PriceSlippageModel.is_backtesting = is_backtesting
        PriceSlippageModel.fetch_slippage_data = fetch_slippage_data
        PriceSlippageModel.recalculate_slippage = recalculate_slippage
        PriceSlippageModel.capital = capital

    @classmethod
    def calculate_slippage(cls, bid:float, ask:float, order:Order, capital:float=None):
        """
        returns the percentage slippage of the current order.
        each asset class uses a unique model
        """
        if order.execution_type != ExecutionType.MARKET:
            return 0

        if not PriceSlippageModel.slippage_estimates:
            PriceSlippageModel.slippage_estimates = cls.read_slippage_estimates()

        trade_pair = order.trade_pair
        if bid * ask == 0:
            if not trade_pair.is_crypto:  # For now, crypto does not have slippage
                bt.logging.warning(f'Tried to calculate slippage with bid: {bid} and ask: {ask}. order: {order}. Returning 0')
                return 0  # Need valid bid and ask.
        if capital is None:
            capital = ValiConfig.MIN_CAPITAL
        size = abs(order.value)
        if size <= 1000:
            return 0  # assume 0 slippage when order size is under 1k
        if cls.is_backtesting:
            cls.refresh_features_daily(order.processed_ms, write_to_disk=False)

        if trade_pair.is_equities:
            slippage_percentage = cls.calc_slippage_equities(bid, ask, order)
        elif trade_pair.is_forex:
            slippage_percentage = cls.calc_slippage_forex(bid, ask, order)
        elif trade_pair.is_crypto:
            slippage_percentage = cls.calc_slippage_crypto(order, capital)
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

        # Check if features are available for this date
        if order_date not in cls.features:
            bt.logging.error(f"Features not found for date {order_date} in equities slippage calculation")
            return 0.0001  # Return minimal slippage as fallback

        # Check if volatility and volume data exist for this trade pair
        if (order.trade_pair.trade_pair_id not in cls.features[order_date].get("vol", {}) or
            order.trade_pair.trade_pair_id not in cls.features[order_date].get("adv", {})):
            bt.logging.error(f"Features not found for trade pair {order.trade_pair.trade_pair_id} on {order_date}")
            return 0.0001  # Return minimal slippage as fallback

        annualized_volatility = cls.features[order_date]["vol"][order.trade_pair.trade_pair_id]
        avg_daily_volume = cls.features[order_date]["adv"][order.trade_pair.trade_pair_id]
        spread = ask - bid
        mid_price = (bid + ask) / 2

        size = abs(order.value)
        volume_shares = size / mid_price

        # if order.processed_ms > 1735718400000:  # Use fitted BB+ for orders after jan 1, 2024, 08:00:00 UTC
        #     size_str = cls.get_order_size_bucket(size)
        #     side = "buy" if order.leverage > 0 else "sell"
        #     model_config = cls.parameters["equity"][order.trade_pair.trade_pair_id][side][size_str]
        #     intercept, c1, c2, c3 = (model_config[key] for key in ["intercept", "spread/price", "annualized_vol", f"{side}_order_size/adv"])
        #     slippage_pct = intercept + (c1 * spread / mid_price) + (c2 * annualized_volatility) + (c3 * volume_shares / avg_daily_volume)
        # else:

        # use BB+ model for orders for equities from 2026 Feb 1
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

        # Check if features are available for this date
        if order_date not in cls.features:
            bt.logging.error(f"Features not found for date {order_date} in forex slippage calculation")
            return 0.0002  # Return 2 bps slippage as fallback

        # Check if volatility and volume data exist for this trade pair
        if (order.trade_pair.trade_pair_id not in cls.features[order_date].get("vol", {}) or
            order.trade_pair.trade_pair_id not in cls.features[order_date].get("adv", {})):
            bt.logging.error(f"Features not found for trade pair {order.trade_pair.trade_pair_id} on {order_date}")
            return 0.0002  # Return 2 bps slippage as fallback

        annualized_volatility = cls.features[order_date]["vol"][order.trade_pair.trade_pair_id]
        avg_daily_volume = cls.features[order_date]["adv"][order.trade_pair.trade_pair_id]
        spread = ask - bid
        mid_price = (bid + ask) / 2

        # bt.logging.info(f"bid: {bid}, ask: {ask}, adv: {avg_daily_volume}, vol: {annualized_volatility}")

        size = abs(order.value)
        base, _ = order.trade_pair.trade_pair.split("/")
        if base != "USD":
            base_to_usd_conversion = cls.live_price_fetcher.get_currency_conversion(base=base, quote="USD")
            if base_to_usd_conversion is None or base_to_usd_conversion == 0:
                bt.logging.error(f"Invalid currency conversion for {base}/USD (returned {base_to_usd_conversion})")
                return 0.0002  # Return 2 bps slippage as fallback
        else:
            base_to_usd_conversion = 1
        # print(base_to_usd_conversion)
        volume_standard_lots = size / (100_000 * base_to_usd_conversion)  # Volume expressed in terms of standard lots (1 std lot = 100,000 base currency)

        term1 = 0.433 * spread / mid_price
        term2 = 0.335 * math.sqrt(annualized_volatility ** 2 / 3 / 250)
        term3 = math.sqrt(volume_standard_lots / (0.3 * avg_daily_volume))
        slippage_pct = term1 + term2 * term3
        # bt.logging.info(f"slippage_pct: {slippage_pct}")
        return slippage_pct

    @classmethod
    def calc_slippage_crypto(cls, order:Order, capital:float) -> float:
        """
        slippage values for crypto
        """
        if order.processed_ms > SLIPPAGE_V2_TIME_MS:
            side = "long" if order.leverage > 0 else "short"
            size = abs(order.value)

            # Check if slippage estimates are loaded
            if "crypto" not in cls.slippage_estimates:
                bt.logging.error(f"Crypto slippage estimates not loaded")
                return 0.0001  # Return minimal slippage as fallback

            trade_pair_key = order.trade_pair.trade_pair_id + "C"
            if trade_pair_key not in cls.slippage_estimates["crypto"]:
                bt.logging.error(f"Slippage estimates not found for crypto trade pair {trade_pair_key}")
                return 0.0001  # Return minimal slippage as fallback

            if side not in cls.slippage_estimates["crypto"][trade_pair_key]:
                bt.logging.error(f"Slippage estimates not found for side {side} on trade pair {trade_pair_key}")
                return 0.0001  # Return minimal slippage as fallback

            slippage_size_buckets = cls.slippage_estimates["crypto"][trade_pair_key][side]
            last_slippage = 0
            for bucket, slippage in slippage_size_buckets.items():
                low, high = bucket[1:-1].split(",")
                last_slippage = slippage
                if int(low) <= size < int(high):
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
    def refresh_features_daily(cls, time_ms: int = None, write_to_disk: bool = True, allow_blocking: bool = False):
        """
        Calculate and store model features (average daily volume and annualized volatility) for new days.

        No locks needed - dict writes are atomic. Uses flag to prevent duplicate expensive work.

        Args:
            time_ms: Timestamp in milliseconds (defaults to now)
            write_to_disk: Whether to persist features to disk
            allow_blocking: If False (default), uses fallback if refresh in progress.
                           If True, skips refresh if already in progress (daemon will retry).

        Returns:
            True if features are available for the date, False otherwise
        """
        if not time_ms:
            time_ms = TimeUtil.now_in_millis()
        current_date = TimeUtil.millis_to_short_date_str(time_ms)

        # Fast path: Features already cached
        if current_date in cls.features:
            return True

        # Throttle: Prevent rapid successive calls (except in backtesting)
        if not cls.is_backtesting and time_ms - cls.last_refresh_time_ms < 1000:
            return current_date in cls.features

        # Check if refresh is already in progress for this date
        if cls._refresh_in_progress and cls._refresh_current_date == current_date:
            if allow_blocking:
                # Background daemon path: skip this cycle, will retry in 10 minutes
                bt.logging.debug(f"Refresh already in progress for {current_date}, daemon will retry later")
                return current_date in cls.features
            else:
                # Order filling path: don't block, use fallback from previous day
                bt.logging.warning(
                    f"Refresh in progress for {current_date}, using fallback to avoid blocking order fill"
                )
                return cls._get_fallback_features(current_date)

        # CRITICAL: If allow_blocking=False and features missing, use fallback
        # This should NEVER happen with pre-population, but serves as safety net
        if not allow_blocking:
            bt.logging.error(
                f"[FEATURE_MISSING] Features not available for {current_date} and allow_blocking=False! "
                f"Pre-population failed. Using fallback. THIS SHOULD NOT HAPPEN - investigate daemon!"
            )
            return cls._get_fallback_features(current_date)

        # Delegate to shared helper
        return cls._calculate_and_store_features(current_date, time_ms, write_to_disk)

    @classmethod
    def pre_populate_next_day(cls, write_to_disk: bool = True) -> bool:
        """
        Pre-calculate slippage features for tomorrow using today's historical data.

        This ensures features are ready when the day rolls over at 00:00 UTC,
        eliminating the need for fallback data on first order of the day.

        Strategy:
        - Calculate features for tomorrow (current_date + 1 day)
        - Use historical data up to today (features are predictive, not reactive)
        - Store in cls.features with tomorrow's date as key

        Should be called daily at 22:00 UTC (after US markets close at 20:00 UTC).

        Returns:
            True if features were successfully pre-populated, False otherwise
        """
        now_ms = TimeUtil.now_in_millis()
        current_date = TimeUtil.millis_to_short_date_str(now_ms)

        # Calculate tomorrow's date
        tomorrow_ms = now_ms + (24 * 60 * 60 * 1000)  # Add 1 day
        tomorrow_date = TimeUtil.millis_to_short_date_str(tomorrow_ms)

        # Check if tomorrow's features already exist
        if tomorrow_date in cls.features:
            bt.logging.debug(
                f"[FEATURE_PREPOP] Features for {tomorrow_date} already exist, skipping"
            )
            return True

        # Check if another process is already pre-populating
        if cls._refresh_in_progress and cls._refresh_current_date == tomorrow_date:
            bt.logging.debug(
                f"[FEATURE_PREPOP] Pre-population already in progress for {tomorrow_date}"
            )
            return False

        bt.logging.info(
            f"[FEATURE_PREPOP] Pre-populating features for {tomorrow_date} "
            f"(current date: {current_date})"
        )

        # Delegate to shared helper
        # Use tomorrow_ms as the processed_ms - this tells get_features to calculate
        # features AS IF we're on tomorrow, using historical data up to today
        return cls._calculate_and_store_features(tomorrow_date, tomorrow_ms, write_to_disk)

    @classmethod
    def _calculate_and_store_features(cls, target_date: str, target_ms: int, write_to_disk: bool) -> bool:
        """
        Core feature calculation and storage logic used by both refresh and pre-population.

        Args:
            target_date: Date string for the features (e.g., "2025-01-15")
            target_ms: Timestamp to use for feature calculation
            write_to_disk: Whether to persist features to disk

        Returns:
            True if features were successfully calculated and stored, False otherwise
        """
        # Set flag to indicate refresh in progress (atomic)
        cls._refresh_in_progress = True
        cls._refresh_current_date = target_date

        try:
            bt.logging.info(
                f"Calculating avg daily volume and annualized volatility for {target_date}"
            )
            trade_pairs = [tp for tp in TradePair if tp.is_forex or tp.is_equities]
            tp_to_adv, tp_to_vol = cls.get_features(trade_pairs=trade_pairs, processed_ms=target_ms)

            if tp_to_adv and tp_to_vol:
                # Atomic dict write - no lock needed
                cls.features[target_date] = {
                    "adv": tp_to_adv,
                    "vol": tp_to_vol
                }

                if write_to_disk:
                    cls.write_features_from_memory_to_disk()

                bt.logging.success(
                    f"✅ Successfully calculated features for {target_date}. "
                    f"Features now available for {len(cls.features)} dates."
                )
                cls.last_refresh_time_ms = TimeUtil.now_in_millis()
                return True
            else:
                bt.logging.warning(
                    f"Failed to calculate features for {target_date}. "
                    f"tp_to_adv: {bool(tp_to_adv)}, tp_to_vol: {bool(tp_to_vol)}"
                )
                return False

        except Exception as e:
            bt.logging.error(f"Error calculating features for {target_date}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False

        finally:
            # Clear flag (atomic)
            cls._refresh_in_progress = False
            cls._refresh_current_date = None

    @classmethod
    def _get_fallback_features(cls, current_date: str) -> bool:
        """
        Attempt to use features from previous day as fallback when current day is refreshing.

        Returns:
            True if fallback features were found and copied, False otherwise
        """
        # Try to find most recent cached features
        if not cls.features:
            bt.logging.error(f"No cached features available for fallback on {current_date}")
            return False

        # Get most recent date from cache
        most_recent_date = max(cls.features.keys())

        bt.logging.warning(
            f"Using fallback features from {most_recent_date} for date {current_date} "
            f"(refresh in progress)"
        )

        # Create shallow copy of most recent features for current date
        # This is temporary - will be overwritten when background refresh completes
        cls.features[current_date] = cls.features[most_recent_date].copy()

        return True

    @classmethod
    def get_features(cls, trade_pairs: list[TradePair], processed_ms: int, adv_lookback_window: int = 10,
                     calc_vol_window: int = 30):
        """
        return dict of features (avg daily volume and annualized volatility) for each trade pair
        """
        tp_to_adv = {}
        tp_to_vol = {}
        for trade_pair in trade_pairs:
            try:
                bars_df = cls.get_bars_with_features(trade_pair, processed_ms, adv_lookback_window, calc_vol_window)
                if bars_df.empty:
                    bt.logging.warning(f"Empty DataFrame returned for trade pair {trade_pair.trade_pair_id}, skipping")
                    continue
                row_selected = bars_df.iloc[-1]
                annualized_volatility = row_selected['annualized_vol'] # recalculate slippage false
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

        price_info_raw = cls.live_price_fetcher.unified_candle_fetcher(trade_pair, start_date, order_date, timespan="day")
        aggs = []
        try:
            for a in price_info_raw:
                aggs.append(a)
        except Exception as e:
            print(f"Error fetching data from Polygon: {e}")

        bars_pd = pd.DataFrame(aggs)
        if bars_pd.empty:
            bt.logging.warning(f"No data returned for trade pair {trade_pair.trade_pair_id} from {start_date} to {order_date}")
            return bars_pd  # Return empty DataFrame
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

                        price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair=o.trade_pair, time_ms=o.processed_ms, live=False)
                        if not price_sources:
                            raise ValueError(
                                f"Ignoring order for [{hk}] due to no live prices being found for trade_pair [{o.trade_pair}]. Please try again.")
                        best_price_source = price_sources[0]
                        bid = best_price_source.bid
                        ask = best_price_source.ask

                    slippage = self.calculate_slippage(bid, ask, o, capital=self.capital)
                    o.bid = bid
                    o.ask = ask
                    o.slippage = slippage
                    bt.logging.info(f"updated order attributes {o}")
                    order_updated = True
                if order_updated:
                    position.rebuild_position_with_updated_orders(self.live_price_fetcher)

    class FeatureRefresher:
        """Daemon process that refreshes price slippage model features daily"""

        def __init__(self, price_slippage_model, slack_notifier=None):
            self.price_slippage_model = price_slippage_model
            self.slack_notifier = slack_notifier

        def run_update_loop(self):
            setproctitle("vali_SlippageRefresher")
            bt.logging.info("PriceSlippageFeatureRefresher daemon started")

            # Load persisted features from disk on startup
            try:
                bt.logging.info("Loading persisted slippage features from disk...")
                features_file = ValiBkpUtils.get_slippage_model_features_file()
                persisted_features = ValiUtils.get_vali_json_file_dict(features_file)

                if persisted_features:
                    # Convert persisted dict to defaultdict(dict) and load
                    PriceSlippageModel.features = defaultdict(dict, persisted_features)

                    # Log what was loaded
                    dates_loaded = sorted(persisted_features.keys())
                    most_recent = dates_loaded[-1] if dates_loaded else None

                    bt.logging.success(
                        f"✅ Loaded {len(dates_loaded)} days of slippage features from disk. "
                        f"Date range: {dates_loaded[0] if dates_loaded else 'N/A'} to {most_recent or 'N/A'}"
                    )

                    if most_recent:
                        # Show sample of most recent data
                        recent_data = persisted_features[most_recent]
                        trade_pairs_count = len(recent_data.get('adv', {}))
                        bt.logging.info(
                            f"Most recent date ({most_recent}) has features for {trade_pairs_count} trade pairs"
                        )
                else:
                    bt.logging.warning("⚠️  No persisted slippage features found on disk - starting fresh")
            except FileNotFoundError:
                bt.logging.warning("⚠️  Slippage features file not found - starting fresh")
            except Exception as e:
                bt.logging.error(f"❌ Error loading persisted slippage features: {e}")
                # Continue with empty features - will be populated by refresh

            # Pre-warm cache on startup to ensure features available for today
            try:
                bt.logging.info("Pre-warming slippage feature cache for current date...")
                success = self.price_slippage_model.refresh_features_daily(allow_blocking=True)

                if success:
                    current_date = TimeUtil.millis_to_short_date_str(TimeUtil.now_in_millis())
                    bt.logging.success(f"✅ Slippage feature cache pre-warmed successfully for {current_date}")
                else:
                    bt.logging.warning("⚠️  Pre-warming completed but features may not be available")
            except Exception as e:
                bt.logging.error(f"❌ Failed to pre-warm slippage feature cache: {e}")
                import traceback as tb
                bt.logging.error(tb.format_exc())
                # Continue anyway - will retry in main loop

            # Run indefinitely - process will terminate when main process exits (daemon=True)
            while True:
                try:
                    # 1. Refresh features for TODAY (if new day)
                    # The method has built-in date checking and will only update if it's a new day
                    self.price_slippage_model.refresh_features_daily(allow_blocking=True)

                    # 2. Pre-populate TOMORROW's features at 22:00 UTC (after US markets close)
                    # This ensures features are ready when day rolls over at 00:00 UTC
                    current_datetime = TimeUtil.millis_to_datetime(TimeUtil.now_in_millis())
                    current_hour_utc = current_datetime.hour

                    if current_hour_utc == 22:
                        # Pre-population window: 22:00-22:59 UTC
                        bt.logging.info("[FEATURE_PREPOP] Entering pre-population window (22:00 UTC)")
                        success = PriceSlippageModel.pre_populate_next_day(write_to_disk=True)

                        if success:
                            bt.logging.success(
                                "[FEATURE_PREPOP] Tomorrow's features pre-populated successfully"
                            )
                        else:
                            bt.logging.warning(
                                "[FEATURE_PREPOP] Failed to pre-populate tomorrow's features, "
                                "will retry in next cycle"
                            )

                            # Send Slack alert if pre-population fails
                            if self.slack_notifier:
                                self.slack_notifier.send_message(
                                    "⚠️ Slippage feature pre-population failed!\n"
                                    "Tomorrow's first orders may use fallback data.\n"
                                    "Check daemon logs for details.",
                                    level="warning"
                                )

                    # Sleep for 10 minutes between checks
                    time.sleep(10 * 60)

                except Exception as e:
                    error_traceback = traceback.format_exc()
                    bt.logging.error(f"Error in PriceSlippageFeatureRefresher: {e}")
                    bt.logging.error(error_traceback)

                    # Send Slack notification
                    if self.slack_notifier:
                        error_message = ErrorUtils.format_error_for_slack(
                            error=e,
                            traceback_str=error_traceback,
                            include_operation=True,
                            include_timestamp=True
                        )
                        self.slack_notifier.send_message(
                            f"❌ PriceSlippageFeatureRefresher Error!\n{error_message}",
                            level="error"
                        )

                    # Sleep before retrying
                    time.sleep(10 * 60)


if __name__ == "__main__":
    psm = PriceSlippageModel()
    equities_order_buy = Order(price=100, processed_ms=TimeUtil.now_in_millis() - 1000 * 200,
                                    order_uuid="test_order",
                                    trade_pair=TradePair.NVDA,
                                    order_type=OrderType.LONG, leverage=1)
    slippage_buy = PriceSlippageModel.calculate_slippage(bid=99, ask=100, order=equities_order_buy)
    print(slippage_buy)
