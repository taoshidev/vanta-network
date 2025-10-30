import json
import logging
from copy import deepcopy
from typing import Optional, List
from pydantic import model_validator, BaseModel, Field

from time_util.time_util import TimeUtil, MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils import leverage_utils
import bittensor as bt
import re
import math

CRYPTO_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - 0.1095) / (365.0*3.0))  # 10.95% per year for 1x leverage. Each interval is 8 hrs
FOREX_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - .03) / 365.0)  # 3% per year for 1x leverage. Each interval is 24 hrs
INDICES_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - .0525) / 365.0)  # 5.25% per year for 1x leverage. Each interval is 24 hrs
FEE_V6_TIME_MS = 1720843707000  # V6 PR merged
SLIPPAGE_V1_TIME_MS = 1739937600000  # Slippage PR merged
ALWAYS_USE_SLIPPAGE = None  # set as either True or False to control whether slippage is always or never applied. if set as None, slippage will be applied based on SLIPPAGE_V1_TIME_MS time gate

class Position(BaseModel):
    """Represents a position in a trading system.

    As a miner, you need to send in signals to the validators, who will keep track
    of your closed and open positions based on your signals. Miners are judged based
    on a 30-day rolling window of return with time decay, so they must continuously perform.

    A signal contains the following information:
    - Trade Pair: The trade pair you want to trade (e.g., major indexes, forex, BTC, ETH).
    - Order Type: SHORT, LONG, or FLAT.
    - Leverage: The amount of leverage for the order type.

    On the validator's side, signals are converted into orders. The validator specifies
    the price at which they fulfilled a signal, which is then used for the order.
    Positions are composed of orders.

    Rules:
    - Please refer to README.md for the rules of the trading system.
    """

    miner_hotkey: str
    position_uuid: str
    open_ms: int
    trade_pair: TradePair
    orders: List[Order] = Field(default_factory=list)
    current_return: float = 1.0  # excludes fees
    close_ms: Optional[int] = None
    net_leverage: float = 0.0
    net_value: float = 0.0
    net_quantity: float = 0.0
    return_at_close: float = 1.0  # includes all fees
    average_entry_price: float = 0.0
    cumulative_entry_value: float = 0.0     # in base currency
    cumulative_entry_value_usd: float = 0.0 # in USD
    account_size: float = 0.0
    realized_pnl: float = 0.0       # in base currency
    realized_pnl_usd: float = 0.0   # in USD
    unrealized_pnl: float = 0.0     # in base currency
    unrealized_pnl_usd: float = 0.0 # in USD
    position_type: Optional[OrderType] = None
    is_closed_position: bool = False

    @model_validator(mode='before')
    def add_trade_pair_to_orders_and_self(cls, values):
        if isinstance(values['trade_pair'], TradePair):
            trade_pair_id = values['trade_pair'].trade_pair_id
        else:
            trade_pair_id = values['trade_pair'][0]

        trade_pair = TradePair.get_latest_trade_pair_from_trade_pair_id(trade_pair_id)
        orders = values.get('orders', [])

        # Add the position-level trade_pair to each order
        updated_orders = []
        for order in orders:
            if not isinstance(order, Order):
                order['trade_pair'] = trade_pair
            else:
                order = order.copy(update={'trade_pair': trade_pair})

            updated_orders.append(order)
        values['orders'] = updated_orders
        values['trade_pair'] = trade_pair
        return values

    @property
    def start_carry_fee_accrual_ms(self):
        return max(FEE_V6_TIME_MS, self.open_ms)

    def get_cumulative_leverage(self) -> float:
        current_leverage = 0.0
        cumulative_leverage = 0.0
        for order in self.orders:
            # Explicit flat
            if order.order_type == OrderType.FLAT:
                cumulative_leverage += abs(current_leverage)
                break

            prev_leverage = current_leverage
            current_leverage += order.leverage

            # Implicit FLAT
            if current_leverage == 0.0 or self._leverage_flipped(prev_leverage, current_leverage):
                cumulative_leverage += abs(prev_leverage)
                break
            else:
                cumulative_leverage += abs(current_leverage - prev_leverage)

        return cumulative_leverage


    def get_spread_fee(self, timestamp_ms: int) -> float:
        """
        transaction fee
        only applied to crypto
        """
        if not self.trade_pair.is_crypto:
            return 1.0
        ans = 1.0 - (self.get_cumulative_leverage() * .001)
        return ans

    def crypto_carry_fee(self, current_time_ms: int) -> (float, int):
        # print(f'accrual time {TimeUtil.millis_to_formatted_date_str(self.start_carry_fee_accrual_ms)} now {TimeUtil.millis_to_formatted_date_str(current_time_ms)}')
        # Fees every 8 hrs. 4 UTC, 12 UTC, 20 UTC
        n_intervals_elapsed, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_crypto(self.start_carry_fee_accrual_ms, current_time_ms)
        fee_product = 1.0
        start_ms = self.start_carry_fee_accrual_ms
        end_ms = start_ms + time_until_next_interval_ms
        for n in range(n_intervals_elapsed):
            if n != 0:
                start_ms = end_ms
                end_ms = start_ms + MS_IN_8_HOURS

            max_lev = self.max_leverage_seen_in_interval(start_ms, end_ms)
            fee_product *= CRYPTO_CARRY_FEE_PER_INTERVAL ** max_lev

        final_fee = fee_product
        #ct_formatted = TimeUtil.millis_to_formatted_date_str(current_time_ms)
        #start_formatted = TimeUtil.millis_to_formatted_date_str(self.open_ms)
        #print(f"start time {start_formatted}, end time {ct_formatted}, delta (days) {(current_time_ms - self.open_ms) / (1000 * 60 * 60 * 24)} final fee {final_fee}")
        return final_fee, current_time_ms + time_until_next_interval_ms

    def forex_indices_carry_fee(self, current_time_ms: int) -> (float, int):
        # Fees M-F where W gets triple fee.
        n_intervals_elapsed, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_forex_indices(self.start_carry_fee_accrual_ms, current_time_ms)
        fee_product = 1.0
        start_ms = self.start_carry_fee_accrual_ms
        end_ms = start_ms + time_until_next_interval_ms
        for n in range(n_intervals_elapsed):
            if n != 0:
                start_ms = end_ms
                end_ms = start_ms + MS_IN_8_HOURS
            # Monday == 0...Sunday == 6
            day_of_week_index = TimeUtil.get_day_of_week_from_timestamp(end_ms)
            assert day_of_week_index in range(7)
            if day_of_week_index in (5, 6):
                continue  # no fees on Saturday, Sunday
            else:
                fee = 1.0
                max_lev = self.max_leverage_seen_in_interval(start_ms, end_ms)
                if self.trade_pair.is_forex:
                    fee *= FOREX_CARRY_FEE_PER_INTERVAL ** max_lev
                elif self.trade_pair.is_indices or self.trade_pair.is_equities:
                    fee *= INDICES_CARRY_FEE_PER_INTERVAL ** max_lev
                else:
                    raise ValueError(f"Unexpected trade pair: {self.trade_pair.trade_pair_id}")
                if day_of_week_index == 2:
                    fee = fee ** 3  # triple fee on Wednesday

            fee_product *= fee

        next_update_time_ms = current_time_ms + time_until_next_interval_ms
        assert next_update_time_ms > current_time_ms, (next_update_time_ms, current_time_ms, fee_product, n_intervals_elapsed, time_until_next_interval_ms)
        return fee_product, next_update_time_ms

    def get_carry_fee(self, current_time_ms) -> (float, int):
        # Calculate the number of times a new day occurred (UTC). If a position is opened at 23:59:58 and this function is
        # called at 00:00:02, the carry fee will be calculated as if a day has passed. Another example: if a position is
        # opened at 23:59:58 and this function is called at 23:59:59, the carry fee will be calculated as 0 days have passed
        # Recalculate and update cache
        assert current_time_ms

        if self.is_closed_position and current_time_ms > self.close_ms:
            current_time_ms = self.close_ms

        if current_time_ms < self.start_carry_fee_accrual_ms:
            delta = MS_IN_8_HOURS if self.trade_pair.is_crypto else MS_IN_24_HOURS
            return 1.0, min(current_time_ms + delta, self.start_carry_fee_accrual_ms)

        if self.trade_pair.is_crypto:
            carry_fee, next_update_time_ms = self.crypto_carry_fee(current_time_ms)
        elif self.trade_pair.is_forex or self.trade_pair.is_indices or self.trade_pair.is_equities:
            carry_fee, next_update_time_ms = self.forex_indices_carry_fee(current_time_ms)
        else:
            raise Exception(f'Unexpected trade pair: {self.trade_pair.trade_pair_id}')

        return carry_fee, next_update_time_ms


    @property
    def initial_entry_price(self) -> float:
        if not self.orders or len(self.orders) == 0:
            return 0.0
        first_order = self.orders[0]
        if ALWAYS_USE_SLIPPAGE or (ALWAYS_USE_SLIPPAGE is None and first_order.processed_ms >= SLIPPAGE_V1_TIME_MS):
            return first_order.price * (1 + first_order.slippage) if first_order.quantity > 0 else first_order.price * (1 - first_order.slippage)
        else:
            return first_order.price

    def __hash__(self):
        # Include specified fields in the hash, assuming trade_pair is accessible and immutable
        return hash((self.miner_hotkey, self.position_uuid, self.open_ms, self.current_return,
                     self.net_quantity, self.net_value, self.initial_entry_price, self.trade_pair.trade_pair))

    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.miner_hotkey == other.miner_hotkey and
                self.position_uuid == other.position_uuid and
                self.open_ms == other.open_ms and
                self.current_return == other.current_return and
                self.net_quantity == other.net_quantity and
                self.net_value == other.net_value and
                self.initial_entry_price == other.initial_entry_price and
                self.trade_pair.trade_pair == other.trade_pair.trade_pair)

    def _handle_trade_pair_encoding(self, d):
        # Remove trade_pair from orders
        if 'orders' in d:
            for order in d['orders']:
                if 'trade_pair' in order:
                    del order['trade_pair']
        # Write the trade_pair in the legacy tuple format as to not break generate_request_outputs. This is temporary
        # code until generate_request_outputs is updated to have the new TradePair decoding logic. If BTC or ETH, put
        # the legacy fee value so that pydantic can validate the JSON with the original decoding logic
        if isinstance(d['trade_pair'], TradePair):
            tp = d['trade_pair']
            fee = .003 if tp.is_crypto else tp.fees
            d['trade_pair'] = [tp.trade_pair_id, tp.trade_pair, fee, tp.min_leverage, tp.max_leverage]
        else:
            d['trade_pair'] = d['trade_pair'][:5]
            if d['trade_pair'][0] in (TradePair.BTCUSD.trade_pair_id, TradePair.ETHUSD.trade_pair_id):
                d['trade_pair'][2] = 0.003
        return d

    def to_dict(self):
        d = deepcopy(self.dict())
        return self._handle_trade_pair_encoding(d)

    def compact_dict_no_orders(self):
        temp = self.to_dict()
        temp.pop('orders')
        return temp

    def to_websocket_dict(self, miner_repo_version=None):
        ans = {'position': self.to_dict()}
        if miner_repo_version is not None:
            ans['miner_repo_version'] = miner_repo_version
        return ans

    @property
    def is_open_position(self):
        return not self.is_closed_position

    def newest_order_age_ms(self, now_ms):
        if len(self.orders) > 0:
            return now_ms - self.orders[-1].processed_ms
        return -1

    def __str__(self):
        return self.to_json_string()

    def to_copyable_str(self):
        ans = self.dict()
        ans['trade_pair'] = f'TradePair.{self.trade_pair.trade_pair_id}'
        ans['position_type'] = f'OrderType.{self.position_type.name}'
        for o in ans['orders']:
            o['trade_pair'] = f'TradePair.{self.trade_pair.trade_pair_id}'
            o['order_type'] = f'OrderType.{o["order_type"].name}'

        s = str(ans)
        s = re.sub(r"'(TradePair\.[A-Z]+|OrderType\.[A-Z]+|FLAT|SHORT|LONG)'", r"\1", s)

        return s


    def to_json_string(self) -> str:
        # Using pydantic's json method with built-in validation
        json_str = self.json()
        # Unfortunately, we can't tell pydantic v1 to strip certain fields so we do that here
        json_loaded = json.loads(json_str)
        json_compressed = self._handle_trade_pair_encoding(json_loaded)
        return json.dumps(json_compressed)

    @classmethod
    def from_dict(cls, position_dict):
        # Assuming 'orders' and 'trade_pair' need to be parsed from dict representations
        # Adjust as necessary based on the actual structure and types of Order and TradePair
        if 'orders' in position_dict:
            position_dict['orders'] = [Order.parse_obj(order) for order in position_dict['orders']]
        if 'trade_pair' in position_dict and isinstance(position_dict['trade_pair'], dict):
            # This line assumes TradePair can be initialized directly from a dict or has a similar parsing method
            position_dict['trade_pair'] = TradePair.from_trade_pair_id(position_dict['trade_pair']['trade_pair_id'])

        # Convert is_closed_position to bool if necessary
        # (assuming this conversion logic is no longer needed if input is properly formatted for Pydantic)

        return cls(**position_dict)

    @staticmethod
    def _position_log(message):
        bt.logging.trace("Position Notification - " + message)

    def get_net_leverage(self):
        return self.net_leverage

    def rebuild_position_with_updated_orders(self, live_price_fetcher):
        self.current_return = 1.0
        self.close_ms = None
        self.return_at_close = 1.0
        self.net_quantity = 0.0
        self.net_value = 0.0
        self.average_entry_price = 0.0
        self.cumulative_entry_value = 0.0
        self.cumulative_entry_value_usd = 0.0
        self.realized_pnl = 0.0
        self.realized_pnl_usd = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_usd = 0.0
        self.position_type = None
        self.is_closed_position = False
        self.position_type = None

        self._update_position(live_price_fetcher)

    def log_position_status(self):
        bt.logging.debug(
            f"position details: "
            f"close_ms [{self.close_ms}] "
            f"initial entry price [{self.initial_entry_price}] "
            f"net quantity [{self.net_quantity}] "
            f"net value [{self.net_value}] "
            f"average entry price [{self.average_entry_price}] "
            f"return_at_close [{self.return_at_close}]"
        )
        order_info = [
            {
                "order type": order.order_type.value,
                "quantity": order.quantity,
                "price": order,
            }
            for order in self.orders
        ]
        bt.logging.debug(f"position order details: " f"close_ms [{order_info}] ")

    def add_order(self, order: Order, live_price_fetcher, net_portfolio_leverage: float=0.0) -> bool:
        """
        Add an order to a position, and adjust its size to stay within
        the trade pair max and portfolio max.
        """
        if self.is_closed_position:
            raise ValueError("Miner attempted to add order to a closed/liquidated position. Ignoring.")
        if order.trade_pair != self.trade_pair:
            raise ValueError(
                f"Order trade pair [{order.trade_pair}] does not match position trade pair [{self.trade_pair}]")

        if self._clamp_and_validate_leverage(order, abs(net_portfolio_leverage)):
            # This order's leverage got clamped to zero.
            # Skip it since we don't want to consider this a FLAT position and we don't want to allow bad actors
            # to send in a bunch of spam orders.
            max_portfolio_leverage = leverage_utils.get_portfolio_leverage_cap(order.processed_ms)
            if abs(net_portfolio_leverage) >= max_portfolio_leverage:
                raise ValueError(
                    f"Miner {self.miner_hotkey} attempted to exceed max adjusted portfolio leverage of {max_portfolio_leverage}. Ignoring order.")
            else:
                if order.leverage >= 0:
                    raise ValueError(
                        f"Miner {self.miner_hotkey} attempted to exceed max leverage {self.trade_pair.max_leverage} for trade pair {self.trade_pair.trade_pair_id}. Ignoring order.")
                else:
                    raise ValueError(
                        f"Miner {self.miner_hotkey} attempted to go below min leverage {self.trade_pair.min_leverage} for trade pair {self.trade_pair.trade_pair_id}. Ignoring order.")

        # Set order value and leverage based on clamped quantity.
        order.value = order.quantity * (order.price * self.trade_pair.lot_size)
        order.leverage = order.value / account_size
        order.slippage = PriceSlippageModel.calculate_slippage(order.bid, order.ask, order)
        self.orders.append(order)
        self._update_position(live_price_fetcher)

    def calculate_pnl(self, current_price, live_price_fetcher, t_ms=None, order=None):
        if self.initial_entry_price == 0 or self.average_entry_price is None:
            return 1

        if not t_ms:
            t_ms = TimeUtil.now_in_millis()

        # pnl with slippage
        order_realized_pnl = 0
        if ALWAYS_USE_SLIPPAGE or (ALWAYS_USE_SLIPPAGE is None and t_ms >= SLIPPAGE_V1_TIME_MS):
            if order:
                # update realized pnl for orders that reduce the size of a position
                if order.order_type != self.position_type or self.position_type == OrderType.FLAT:
                    exit_price = current_price * (1 + order.slippage) if order.quantity > 0 else current_price * (1 - order.slippage)
                    order_realized_pnl = -1 * (exit_price - self.average_entry_price) * (order.quantity * order.trade_pair.lot_size)
                    self.realized_pnl += order_realized_pnl
                self.unrealized_pnl = (current_price - self.average_entry_price) * (min(self.net_quantity, self.net_quantity + order.quantity, key=abs) * order.trade_pair.lot_size)
            else:
                self.unrealized_pnl = (current_price - self.average_entry_price) * (self.net_quantity * self.trade_pair.lot_size)

            gain = (self.realized_pnl + self.unrealized_pnl) / self.initial_entry_price
        else:
            gain = (
                (current_price - self.average_entry_price)
                * (self.net_quantity * self.trade_pair.lot_size)
                / self.initial_entry_price
            )

        # convert pnl to usd
        order_type = order.order_type if order else self.orders[-1].order_type
        quote_usd_conversion = live_price_fetcher.get_usd_conversion(self.trade_pair.quote, t_ms, order_type, self.position_type)
        print(f"get_usd_conversion for {self.trade_pair}: {quote_usd_conversion}")
        self.realized_pnl_usd += order_realized_pnl * quote_usd_conversion
        self.unrealized_pnl_usd = self.unrealized_pnl * quote_usd_conversion

        # Check if liquidated
        if gain <= -1.0:
            return 0
        net_return = 1 + gain
        return net_return

    def _leverage_flipped(self, prev_leverage, cur_leverage):
        return prev_leverage * cur_leverage < 0 or prev_leverage != 0 and cur_leverage == 0

    def max_leverage_seen_in_interval(self, start_ms: int, end_ms: int) -> float:
        #print(f"Seeking max leverage between {TimeUtil.millis_to_formatted_date_str(start_ms)} and {TimeUtil.millis_to_formatted_date_str(end_ms)}")
        #for x in self.orders:
        #    print(f"    Found order at time {TimeUtil.millis_to_formatted_date_str(x.processed_ms)}")
        """
        Returns the max leverage seen in the interval [start_ms, end_ms] (inclusive). If no orders are in the interval,
        raise an exception
        """
        # check valid bounds and throw ValueError if bad data
        if start_ms > end_ms:
            raise ValueError(f"start_ms [{start_ms}] is greater than end_ms [{end_ms}]")
        if end_ms < self.open_ms:
            raise ValueError(f"end_ms [{end_ms}] is less than open_ms [{self.open_ms}]")
        if end_ms < 0 or start_ms < 0:
            raise ValueError(f"start_ms [{start_ms}] or end_ms [{end_ms}] is less than 0")
        if len(self.orders) == 0:
            raise ValueError("No orders in position")
        if self.orders[0].processed_ms > end_ms:
            raise ValueError(f"First order processed_ms [{self.orders[0].processed_ms}] is greater than end_ms [{end_ms}]")
        if self.is_closed_position and start_ms > self.close_ms:
            raise ValueError(f"Position closed before interval start_ms [{start_ms}]")


        interval_data = {'start_ms': start_ms, 'end_ms': end_ms, 'max_leverage': -float('inf')}
        self.max_leverage_seen(interval_data=interval_data)

        if interval_data['max_leverage'] == -float('inf'):
            raise ValueError('Unable to find max leverage in interval')
        assert interval_data['max_leverage'] > 0, (interval_data['max_leverage'], self.orders)
        return interval_data['max_leverage']

    def max_leverage_seen(self, interval_data=None):
        max_leverage = 0
        current_leverage = 0
        stop_signaled = False
        for idx, order in enumerate(self.orders):
            if stop_signaled:
                break

            prev_leverage = current_leverage
            current_leverage += order.leverage
            # Explicit flat / implicit FLAT
            if order.order_type == OrderType.FLAT or self._leverage_flipped(prev_leverage, current_leverage):
                stop_signaled = True
                current_leverage = 0

            if abs(current_leverage) > max_leverage:
                max_leverage = abs(current_leverage)

            if interval_data:
                if order.processed_ms < interval_data['start_ms']:
                    pass
                elif order.processed_ms == interval_data['start_ms']:
                    interval_data['max_leverage'] = max(abs(current_leverage), interval_data['max_leverage'])
                elif order.processed_ms <= interval_data['end_ms']:
                    interval_data['max_leverage'] = max(abs(current_leverage), interval_data['max_leverage'], abs(prev_leverage))

                # An order passes the interval for the first time
                elif order.processed_ms > interval_data['end_ms']:
                    interval_data['max_leverage'] = max(abs(prev_leverage), interval_data['max_leverage'])
                    stop_signaled = True

        if interval_data:
            # The position's last order is way before the interval start. Use the last known position leverage
            if interval_data['max_leverage'] == -float('inf'):
                interval_data['max_leverage'] = abs(current_leverage)

        return max_leverage

    def _handle_liquidation(self, time_ms, live_price_fetcher):
        self._position_log("position liquidated. Trade pair: " + str(self.trade_pair.trade_pair_id))
        if self.is_closed_position:
            return
        else:
            self.orders.append(self.generate_fake_flat_order(self, time_ms, live_price_fetcher))
            self.close_out_position(time_ms)

    @staticmethod
    def generate_fake_flat_order(position, elimination_time_ms, live_price_fetcher, extra_price_source=None):
        fake_flat_order_time = elimination_time_ms
        price_source = live_price_fetcher.get_close_at_date(
            trade_pair=position.trade_pair,
            timestamp_ms=elimination_time_ms,
            verbose=False
        )

        if price_source:
            # Parse the appropriate price
            price = price_source.parse_appropriate_price(
                now_ms=elimination_time_ms,
                is_forex=position.trade_pair.is_forex,
                order_type=OrderType.FLAT,
                position_type=position.orders[0].order_type
            )
            src = OrderSource.PRICE_FILLED_ELIMINATION_FLAT
        else:
            bt.logging.warning(f'Unexpectedly unable to fetch price for trade pair {position.trade_pair.trade_pair_id}'
                               f' at time {TimeUtil.millis_to_formatted_date_str(elimination_time_ms)} during fake flat order'
                               f'creation. Setting price to 0. and src to OrderSource.ELIMINATION_FLAT')
            price = 0
            src = OrderSource.ELIMINATION_FLAT


        flat_order = Order(price=price,
                           processed_ms=fake_flat_order_time,
                           order_uuid=position.position_uuid[::-1],  # deterministic across validators. Won't mess with p2p sync
                           trade_pair=position.trade_pair,
                           order_type=OrderType.FLAT,
                           quantity=-position.net_quantity,
                           src=src,
                           price_sources=[x for x in (price_source, extra_price_source) if x is not None])
        return flat_order

    def calculate_return_with_fees(self, current_return_no_fees, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()
        # Note: Closed positions will have static returns. This method is only called for open positions.
        # V2 fee calculation. Crypto fee lowered from .003 to .002. Multiply fee by leverage for crypto pairs.
        # V3 calculation. All fees scaled by leverage. Updated forex and indices fees.
        # V4 calculation. Fees are now based on cumulative leverage
        # V5 Crypto fees cut in half
        # V6 introduce "carry fee"
        if timestamp_ms < 1713198680000:  # V4 PR merged
            fee = 1.0 - self.trade_pair.fees * self.max_leverage_seen()
        else:
            fee = self.get_carry_fee(timestamp_ms)[0] * self.get_spread_fee(timestamp_ms)
        return current_return_no_fees * fee

    def get_open_position_return_with_fees(self, realtime_price, live_price_fetcher, time_ms):
        current_return = self.calculate_pnl(realtime_price, live_price_fetcher)
        return self.calculate_return_with_fees(current_return, timestamp_ms=time_ms)

    def set_returns_with_updated_fees(self, total_fees, time_ms, live_price_fetcher):
        self.return_at_close = self.current_return * total_fees
        if self.current_return == 0:
            self._handle_liquidation(TimeUtil.now_in_millis() if time_ms is None else time_ms, live_price_fetcher)


    def set_returns(self, realtime_price, live_price_fetcher, time_ms=None, total_fees=None, order=None):
        # We used to multiple trade_pair.fees by net_leverage. Eventually we will
        # Update this calculation to approximate actual exchange fees.
        self.current_return = self.calculate_pnl(realtime_price, live_price_fetcher, t_ms=time_ms, order=order)
        if total_fees is None:
            self.return_at_close = self.calculate_return_with_fees(self.current_return,
                               timestamp_ms=TimeUtil.now_in_millis() if time_ms is None else time_ms)
        else:
            self.return_at_close = self.current_return * total_fees

        if self.current_return < 0:
            raise ValueError(f"current return must be positive {self.current_return}")

        if self.current_return == 0:
            self._handle_liquidation(TimeUtil.now_in_millis() if time_ms is None else time_ms, live_price_fetcher)

    def update_position_state_for_new_order(self, order, delta_quantity, live_price_fetcher):
        """
        Must be called after every order to maintain accurate internal state. The variable average_entry_price has
        a name that can be a little confusing. Although it claims to be the average price, it really isn't.
        For example, it can take a negative value. A more accurate name for this variable is the weighted average
        entry price.
        """
        realtime_price = order.price
        assert self.initial_entry_price > 0, self.initial_entry_price
        new_net_quantity = self.net_quantity + delta_quantity
        if order.src in (OrderSource.ELIMINATION_FLAT, OrderSource.DEPRECATION_FLAT):
            self.net_quantity = 0.0
            self.net_value = 0.0
            return  # Don't set returns since the price is zero'd out.
        self.set_returns(realtime_price, live_price_fetcher, time_ms=order.processed_ms, order=order)

        # Liquidated
        if self.current_return == 0:
            return
        self._position_log(f"closed position total w/o fees [{self.current_return}]. Trade pair: {self.trade_pair.trade_pair_id}")
        self._position_log(f"closed return with fees [{self.return_at_close}]. Trade pair: {self.trade_pair.trade_pair_id}")

        if self.position_type == OrderType.FLAT:
            self.net_quantity = 0.0
            self.net_value = 0.0
        else:
            if ALWAYS_USE_SLIPPAGE is False or (ALWAYS_USE_SLIPPAGE is None and order.processed_ms < SLIPPAGE_V1_TIME_MS):
                # no slippage
                self.average_entry_price = (
                    self.average_entry_price * self.net_quantity
                    + realtime_price * delta_quantity
                ) / new_net_quantity

                entry_value = realtime_price * (order.quantity * order.trade_pair.lot_size)
            elif self.position_type == order.order_type:
                # after SLIPPAGE_V1_TIME_MS, average entry price now reflects the average price
                # average entry price only changes when an order is in the same direction as the position. reducing a position does not affect average entry price.
                entry_price = order.price * (1 + order.slippage) if order.quantity > 0 else order.price * (1 - order.slippage)
                entry_value = entry_price * (order.quantity * order.trade_pair.lot_size)

                self.average_entry_price = (
                    self.average_entry_price * self.net_quantity
                    + entry_price * delta_quantity
                ) / new_net_quantity
            else:
                # order is reducing the size of a position, so there is no entry cost.
                entry_value = 0

            quote_usd_conversion = live_price_fetcher.get_usd_conversion(self.trade_pair.quote, order.processed_ms, order.order_type, self.position_type)
            print(f"get_usd_conversion2 for {self.trade_pair}: {quote_usd_conversion}")
            self.cumulative_entry_value += entry_value
            self.cumulative_entry_value_usd += entry_value * quote_usd_conversion
            self.net_quantity = new_net_quantity
            self.net_value = realtime_price * (self.net_quantity * self.trade_pair.lot_size)

    def initialize_position_from_first_order(self, order):
        self.open_ms = order.processed_ms
        if self.initial_entry_price <= 0:
            raise ValueError("Initial entry price must be > 0")
        # Initialize the position type. It will stay the same until the position is closed.
        if order.quantity > 0:
            self._position_log("setting new position type as LONG. Trade pair: " + str(self.trade_pair.trade_pair_id))
            self.position_type = OrderType.LONG
        elif order.quantity < 0:
            self._position_log("setting new position type as SHORT. Trade pair: " + str(self.trade_pair.trade_pair_id))
            self.position_type = OrderType.SHORT
        else:
            raise ValueError("quantity of 0 provided as initial order.")

    def close_out_position(self, close_ms):
        self.position_type = OrderType.FLAT
        self.is_closed_position = True
        self.close_ms = close_ms

    def reopen_position(self):
        self.position_type = self.orders[0].order_type
        self.is_closed_position = False
        self.close_ms = None

    def _clamp_and_validate_leverage(self, order: Order, net_portfolio_leverage: float) -> bool:
        """
        If an order's leverage would make the position's leverage higher than max_position_leverage,
        we clamp the order's leverage. If clamping causes the order's leverage to be below
        ValiConfig.ORDER_MIN_LEVERAGE, we raise an error.

        If an order's leverage would take the position leverage below min_position_leverage, we raise an error.

        Return true if the order should be ignored. Only happens when the order attempts to exceed max_position_leverage
        and is already at max_position_leverage.
        """
        should_ignore_order = False
        if order.order_type == OrderType.FLAT:
            order.quantity = -self.net_quantity
            return should_ignore_order

        is_first_order = len(self.orders) == 0
        proposed_leverage = self.net_leverage + order.leverage
        min_position_leverage, max_position_leverage = leverage_utils.get_position_leverage_bounds(self.trade_pair, order.processed_ms)

        current_adjusted_leverage = abs(self.net_leverage) * self.trade_pair.leverage_multiplier
        proposed_portfolio_leverage = (net_portfolio_leverage - current_adjusted_leverage +
                                       (abs(proposed_leverage) * self.trade_pair.leverage_multiplier))
        max_portfolio_leverage = leverage_utils.get_portfolio_leverage_cap(order.processed_ms)

        # we only need to worry about clamping if the sign of the position leverage remains the same i.e. position does not flip and close
        if is_first_order or self.net_leverage * proposed_leverage > 0:
            if (abs(proposed_leverage) > max_position_leverage
                    or proposed_portfolio_leverage > max_portfolio_leverage):
                if (is_first_order
                        or abs(proposed_leverage) >= abs(self.net_leverage)
                        or proposed_portfolio_leverage >= net_portfolio_leverage):

                    clamped_position_leverage = max_position_leverage - abs(self.net_leverage)
                    clamped_portfolio_leverage = (max_portfolio_leverage - net_portfolio_leverage) / self.trade_pair.leverage_multiplier
                    # take leverage up to the limit for position or portfolio, whichever is hit first
                    clamped_leverage = min(clamped_position_leverage, clamped_portfolio_leverage)
                    order.leverage = max(0.0, clamped_leverage)  # ensure leverage is always >= 0
                    # TODO: set order.quantity here

                    if order.order_type == OrderType.SHORT:
                        order.leverage *= -1
                        order.quantity *= -1
                    should_ignore_order = order.leverage == 0
                    if not should_ignore_order:
                        logging.warning(f"Miner {self.miner_hotkey} {self.trade_pair.trade_pair_id} order leverage clamped to {order.leverage}")
                else:
                    # portfolio leverage attempts to go under min
                    if abs(proposed_leverage) < min_position_leverage:
                        raise ValueError(f"Miner {self.miner_hotkey} attempted to set {self.trade_pair.trade_pair_id} position leverage below min_position_leverage {min_position_leverage} while exceeding max_portfolio_leverage {max_portfolio_leverage}")
                    else:
                        pass  #  We are getting the leverage closer to the new boundary (decrease) so allow it
            elif abs(proposed_leverage) < min_position_leverage:
                if is_first_order or abs(proposed_leverage) < abs(self.net_leverage):
                    raise ValueError(f'Miner {self.miner_hotkey} attempted to set {self.trade_pair.trade_pair_id} position leverage below min_position_leverage {min_position_leverage}')
                else:
                    pass  # We are trying to increase the leverage here so let it happen
        # attempting to flip position
        else:
            order.quantity = -self.net_quantity
            order.order_type = OrderType.FLAT

        if abs(order.leverage) < ValiConfig.ORDER_MIN_LEVERAGE and (should_ignore_order is False):
            raise ValueError(f'Clamped order leverage [{order.leverage}] is below ValiConfig.ORDER_MIN_LEVERAGE {ValiConfig.ORDER_MIN_LEVERAGE}')

        return should_ignore_order

    def _update_position(self, live_price_fetcher):
        self.net_quantity = 0.0
        self.net_value = 0.0
        self.cumulative_entry_value = 0.0
        self.cumulative_entry_value_usd = 0.0
        self.realized_pnl = 0.0
        self.realized_pnl_usd = 0.0
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_usd = 0.0
        bt.logging.trace(f"Updating position {self.trade_pair.trade_pair_id} with n orders: {len(self.orders)}")
        for order in self.orders:
            if self.position_type is None:
                self.initialize_position_from_first_order(order)

            # Check if the new order flattens the position, explicitly or implicitly
            if (
                (
                    self.position_type == OrderType.LONG
                    and self.net_quantity + order.quantity <= 0
                )
                or (
                    self.position_type == OrderType.SHORT
                    and self.net_quantity + order.quantity >= 0
                )
                or order.order_type == OrderType.FLAT
            ):
                #self._position_log(
                #    f"Flattening {self.position_type.value} position from order {order}"
                #)
                self.close_out_position(order.processed_ms)
                # Set the order quantity
                order.quantity = -self.net_quantity

            # Reflect the current order in the current position's return.
            adjusted_quantity = (
                0.0 if self.position_type == OrderType.FLAT else order.quantity
            )
            #bt.logging.info(
            #    f"Updating position state for new order {order} with adjusted leverage {adjusted_quantity}"
            #)
            self.update_position_state_for_new_order(order, adjusted_quantity, live_price_fetcher)

            # If the position is already closed, we don't need to process any more orders. break in case there are more orders.
            if self.position_type == OrderType.FLAT:
                break
