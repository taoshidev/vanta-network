from collections import defaultdict

import pandas as pd

from data_generator.polygon_data_service import PolygonDataService
from shared_objects.cache_controller import CacheController
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.live_price_server import LivePriceFetcherServer
from vali_objects.utils.mdd_checker_server import MDDCheckerServer
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.vali_config import TradePair, RPCConnectionMode
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.vali_dataclasses.price_source import PriceSource


class MockMDDChecker(MDDCheckerServer):
    def __init__(self, metagraph, position_manager, live_price_fetcher):
        super().__init__(running_unit_tests=True, slack_notifier=None, start_server=False, start_daemon=False)

    # Lets us bypass the wait period in MDDChecker
    def get_last_update_time_ms(self):
        return 0


class MockCacheController(CacheController):
    def __init__(self, metagraph):
        super().__init__(metagraph, running_unit_tests=True)


class MockPositionManager(PositionManager):
    def __init__(self, metagraph, perf_ledger_manager, live_price_fetcher=None):
        super().__init__(running_unit_tests=True)

    def _start_server_process(self, address, authkey, server_ready):
        """Mock implementation - tests don't start actual server process."""
        return None


class MockPerfLedgerManager(PerfLedgerManager):
    def __init__(self, metagraph):
        super().__init__(connection_mode=RPCConnectionMode.LOCAL)


class MockPlagiarismDetector(PlagiarismDetector):
    def __init__(self):
        # Use RPC mode so clients connect to orchestrator servers
        # (LOCAL mode would create disconnected clients expecting set_direct_server())
        super().__init__(connection_mode=RPCConnectionMode.RPC)
        # Override to get test-specific behaviors (fixed time, test directories)
        self.running_unit_tests = True

    # Lets us bypass the wait period in PlagiarismDetector
    def get_last_update_time_ms(self):
        return 0


class MockChallengePeriodManager(ChallengePeriodManager):
    def __init__(self, metagraph):
        super().__init__(metagraph, running_unit_tests=True)

class MockLivePriceFetcherServer(LivePriceFetcherServer):
    def __init__(self, secrets, disable_ws):
        super().__init__(
            secrets=secrets,
            disable_ws=disable_ws,
            connection_mode=RPCConnectionMode.LOCAL,
            start_server=False,
            start_daemon=False
        )
        self.polygon_data_service = MockPolygonDataService(api_key=secrets["polygon_apikey"], disable_ws=disable_ws)

    def get_close_at_date(self, trade_pair, timestamp_ms, order=None, verbose=True):
        return PriceSource(open=1, high=1, close=1, low=1, bid=1, ask=1)

    def get_sorted_price_sources_for_trade_pair(self, trade_pair, time_ms=None, live=True):
        return [PriceSource(open=1, high=1, close=1, low=1, bid=1, ask=1)]


class MockPolygonDataService(PolygonDataService):
    def __init__(self, api_key, disable_ws=True):
        super().__init__(api_key, disable_ws=disable_ws)
        self.trade_pair_to_recent_events_realtime = defaultdict()

    def get_last_quote(self, trade_pair: TradePair, processed_ms: int) -> (float, float):
        ask = 1.10
        bid = 1.08
        return ask, bid

    def get_currency_conversion(self, trade_pair: TradePair=None, base: str=None, quote: str=None) -> float:
        if (base and quote) and base == quote:
            return 1
        else:
            return 0.5  # 1 base = 0.5 quote

    # def get_candles_for_trade_pair_simple(self, trade_pair: TradePair, start_timestamp_ms: int, end_timestamp_ms: int, timespan: str="second"):
    #     pass

class MockPriceSlippageModel(PriceSlippageModel):
    def __init__(self, live_price_fetcher):
        super().__init__(live_price_fetcher)

    @classmethod
    def get_bars_with_features(cls, trade_pair: TradePair, processed_ms: int, adv_lookback_window: int=10, calc_vol_window: int=30, trading_days_in_a_year: int=252) -> pd.DataFrame:
        adv_lookback_window = 10  # 10-day average daily volume

        # Create a single-row DataFrame
        if trade_pair.is_forex:
            bars_df = pd.DataFrame({
                'annualized_vol': [0.5],  # Mock annualized volatility
                f'adv_last_{adv_lookback_window}_days': [100_000],  # Mock 10-day average daily volume
            })
        else:  # equities
            bars_df = pd.DataFrame({
                'annualized_vol': [0.5],  # Mock annualized volatility
                f'adv_last_{adv_lookback_window}_days': [100_000_000],  # Mock 10-day average daily volume
            })
        return bars_df


