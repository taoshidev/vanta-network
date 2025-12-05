# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

from datetime import datetime
from typing import List, Tuple

import requests
from requests import Response
import time

from data_generator.financial_markets_generator.base_financial_markets_generator.base_financial_markets_generator import \
    BaseFinancialMarketsGenerator
from vali_objects.vali_config import ValiConfig


class BinanceData(BaseFinancialMarketsGenerator):
    def __init__(self):
        super().__init__()
        self._symbols = {
            "BTCUSD": "BTCUSDT"
        }
        self._tf = {
            5: "5m"
        }

    def get_data(self,
                 symbol='BTCUSDT',
                 interval=ValiConfig.STANDARD_TF,
                 start=None,
                 end=None,
                 limit=1000,
                 retries=0) -> Response:

        if type(interval) == int:  # noqa: E721
            binance_interval = self._tf[interval]
        else:
            raise Exception("no mapping for binance interval")

        if start is None:
            # minute, minutes, hours, days, weeks
            start = str(int(datetime.now().timestamp() * 1000) - 60000 * 60 * 24 * 7 * 2)
            # start = str(int(datetime.now().timestamp() * 1000) - 60000 * 1250)
        if end is None:
            end = str(int(datetime.now().timestamp() * 1000))

        if symbol != "BTCUSDT":
            symbol = self._symbols[symbol]

        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}' \
              f'&interval={binance_interval}&startTime={start}&endTime={end}&limit={limit}'

        response = requests.get(url)

        try:
            if response.status_code == 200:
                return response
            else:
                raise Exception("received error status code")
        except Exception:
            if retries < 5:
                time.sleep(retries)
                retries += 1
                # print("retrying getting historical binance data")
                self.get_data(symbol, interval, start, end, limit, retries)
            else:
                raise ConnectionError("max number of retries exceeded trying to get binance data")

    def get_data_and_structure_data_points(self, symbol: str, tf: int, data_structure: List[List], ts_range: Tuple[int, int]):
        bd = self.get_data(symbol=symbol, interval=tf, start=ts_range[0], end=ts_range[1]).json()
        if "msg" in bd:
            raise Exception("error occurred getting Binance data, please review", bd["msg"])
        else:
            # print("received binance historical data from : ", TimeUtil.millis_to_timestamp(ts_range[0]),
            #       TimeUtil.millis_to_timestamp(ts_range[1]))
            self.convert_output_to_data_points(data_structure,
                                               bd,
                                               [0, 4, 2, 3, 5]
                                               )
