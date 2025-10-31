from tests.shared_objects.mock_classes import (
    MockLivePriceFetcher,
    MockPriceSlippageModel,
)
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position

# from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order


class TestPriceSlippageModel(TestBase):
    def setUp(self):
        super().setUp()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.psm = MockPriceSlippageModel(live_price_fetcher=self.live_price_fetcher)
        self.psm.refresh_features_daily(write_to_disk=False)

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_ORDER_UUID = "test_order"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()  # 1718071209000
        self.default_bid = 99
        self.default_ask = 100
        self.DEFAULT_ACCOUNT_SIZE = 100_000


    def test_open_position_returns_with_slippage(self):
        """
        for an open position, the entry should include slippage.
        the unrealized pnl does not include slippage
        """
        self.open_order = Order(
            price=100,
            slippage=0.05,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid=self.DEFAULT_ORDER_UUID,
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=1,
        )
        self.open_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.EURUSD,
            orders=[],
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        self.open_position.add_order(self.open_order, self.live_price_fetcher)
        # print(self.open_position)
        assert self.open_position.initial_entry_price == 105  # 100 * (1 + 0.05) = 105
        assert self.open_position.average_entry_price == 105

        self.open_position.set_returns(110, self.live_price_fetcher)  # say the current price has grown from 100 -> 110
        # the current return only applies slippage to the entry price, for unrealized PnL
        assert self.open_position.current_return == 1.0476190476190477  # (110-105) / 105

    def test_closed_position_returns_with_slippage(self):
        """
        for a closed position, the entry and exits should both include slippage
        the realized pnl includes slippage
        """
        self.open_order = Order(
            price=100,
            slippage=0.01,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid=self.DEFAULT_ORDER_UUID,
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.LONG,
            leverage=1,
        )
        self.close_order = Order(
            price=110,
            slippage=0.01,
            processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid=self.DEFAULT_ORDER_UUID+"_close",
            trade_pair=TradePair.EURUSD,
            order_type=OrderType.FLAT,
            leverage=-1,
        )
        self.closed_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.EURUSD,
            orders=[],
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.closed_position.add_order(self.open_order, self.live_price_fetcher)
        self.closed_position.add_order(self.close_order, self.live_price_fetcher)

        assert self.closed_position.initial_entry_price == 101  # 100 * (1 + 0.01) = 101
        assert self.closed_position.average_entry_price == 101
        # the current return has a slippage on both the entry and exit prices when calculating a realized PnL
        assert self.closed_position.current_return == 1.0782178217821783  # (108.9-101) / 101

    # def test_equities_slippage(self):
    #     """
    #     test buy and sell order slippage, using slippage model
    #     """
    #     self.equities_order_buy = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
    #                                         order_uuid=self.DEFAULT_ORDER_UUID,
    #                                         trade_pair=TradePair.NVDA,
    #                                         order_type=OrderType.LONG, leverage=1)
    #     slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask, self.equities_order_buy, capital=100_000)
    #
    #     self.equities_order_sell = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
    #                                         order_uuid=self.DEFAULT_ORDER_UUID,
    #                                         trade_pair=TradePair.NVDA,
    #                                         order_type=OrderType.SHORT, leverage=-1)
    #     slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask, self.equities_order_sell, capital=100_000)
    #
    #     ## assert slippage is proportional to order size
    #     self.equities_order_buy_large = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
    #                                     order_uuid=self.DEFAULT_ORDER_UUID,
    #                                     trade_pair=TradePair.NVDA,
    #                                     order_type=OrderType.LONG, leverage=3)
    #     large_slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask, self.equities_order_buy_large, capital=100_000)
    #     assert large_slippage_buy > slippage_buy
    #
    #     self.equities_order_sell_small = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
    #                                      order_uuid=self.DEFAULT_ORDER_UUID,
    #                                      trade_pair=TradePair.NVDA,
    #                                      order_type=OrderType.SHORT, leverage=-0.1)
    #     small_slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
    #                                                           self.equities_order_sell_small, capital=100_000)
    #     assert small_slippage_sell < slippage_sell
    #
    #     ##

    def test_forex_slippage(self):
        """
        test buy and sell order slippage, using BB+ model
        """
        self.forex_order_buy = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                        order_uuid=self.DEFAULT_ORDER_UUID,
                                        trade_pair=TradePair.USDCAD,
                                        order_type=OrderType.LONG, leverage=1, value=1*self.DEFAULT_ACCOUNT_SIZE)
        slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                             self.forex_order_buy)

        self.forex_order_sell = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                         order_uuid=self.DEFAULT_ORDER_UUID,
                                         trade_pair=TradePair.USDCAD,
                                         order_type=OrderType.SHORT, leverage=-3, value=-3*self.DEFAULT_ACCOUNT_SIZE)
        slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                              self.forex_order_sell)

        ## assert slippage is proportional to order size
        self.forex_order_buy_large = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                              order_uuid=self.DEFAULT_ORDER_UUID,
                                              trade_pair=TradePair.USDCAD,
                                              order_type=OrderType.LONG, leverage=3, value=3*self.DEFAULT_ACCOUNT_SIZE)
        large_slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                   self.forex_order_buy_large)
        # forex slippage does not depend on size
        assert large_slippage_buy == slippage_buy

        self.forex_order_sell_small = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                               order_uuid=self.DEFAULT_ORDER_UUID,
                                               trade_pair=TradePair.USDCAD,
                                               order_type=OrderType.SHORT, leverage=-1, value=-1*self.DEFAULT_ACCOUNT_SIZE)
        small_slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                    self.forex_order_sell_small)
        assert small_slippage_sell == slippage_sell

    def test_crypto_slippage(self):
        """
        test buy and sell order slippage
        """
        self.crypto_order_buy = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                     order_uuid=self.DEFAULT_ORDER_UUID,
                                     trade_pair=TradePair.BTCUSD,
                                     order_type=OrderType.LONG, leverage=0.25, value=0.25*self.DEFAULT_ACCOUNT_SIZE)
        slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                             self.crypto_order_buy)

        self.crypto_order_sell = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                      order_uuid=self.DEFAULT_ORDER_UUID,
                                      trade_pair=TradePair.SOLUSD,
                                      order_type=OrderType.SHORT, leverage=-0.25, value=0.25*self.DEFAULT_ACCOUNT_SIZE)
        slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                              self.crypto_order_sell)

        ## assert slippage is proportional to order size
        self.crypto_order_buy_large = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                           order_uuid=self.DEFAULT_ORDER_UUID,
                                           trade_pair=TradePair.BTCUSD,
                                           order_type=OrderType.LONG, leverage=0.5, value=0.5*self.DEFAULT_ACCOUNT_SIZE)
        large_slippage_buy = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                   self.crypto_order_buy_large)
        ## crypto slippage depends on size
        assert large_slippage_buy > slippage_buy

        self.crypto_order_sell_small = Order(price=100, processed_ms=self.DEFAULT_OPEN_MS,
                                            order_uuid=self.DEFAULT_ORDER_UUID,
                                            trade_pair=TradePair.SOLUSD,
                                            order_type=OrderType.SHORT, leverage=-0.1, value=-0.1*self.DEFAULT_ACCOUNT_SIZE)
        small_slippage_sell = PriceSlippageModel.calculate_slippage(self.default_bid, self.default_ask,
                                                                    self.crypto_order_sell_small)
        assert small_slippage_sell < slippage_sell



