import json
import logging
from typing import Dict, Optional, List
from pydantic import model_validator, BaseModel, Field

from time_util.time_util import TimeUtil, MS_IN_1_HOUR, MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.vali_config import TradePair, TradePairCategory, TradePairLike, DynamicTradePair, ValiConfig
from vali_objects.vali_dataclasses.corporate_actions import DividendHistoryEntry
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils import leverage_utils
import bittensor as bt
import re
import math

# TODO update with ledger updates
CRYPTO_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - 0.1095) / (365.0*3.0))  # 10.95% per year for 1x leverage. Each interval is 8 hrs
FOREX_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - .03) / 365.0)  # 3% per year for 1x leverage. Each interval is 24 hrs
INDICES_CARRY_FEE_PER_INTERVAL = math.exp(math.log(1 - .0525) / 365.0)  # 5.25% per year for 1x leverage. Each interval is 24 hrs


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
    trade_pair: TradePairLike
    orders: List[Order] = Field(default_factory=list)
    current_return: float = 1.0             # Excludes fees
    close_ms: Optional[int] = None
    net_leverage: float = 0.0
    net_value: float = 0.0                  # USD
    net_quantity: float = 0.0               # Base currency lots
    return_at_close: float = 1.0            # Includes all fees
    average_entry_price: float = 0.0        # Quote currency
    cumulative_entry_value: float = 0.0     # USD
    account_size: float = 0.0               # USD
    realized_pnl: float = 0.0               # USD
    unrealized_pnl: float = 0.0             # USD
    position_type: Optional[OrderType] = None
    # TODO: Replace this with a property that checks if close_ms is None
    is_closed_position: bool = False
    fee_history: List[Dict] = Field(default_factory=list) # [{"fee_type": "carry", "amount": 123, "time_ms": 123}]
    is_hl: bool = False  # True for Hyperliquid entity miner positions
    last_stock_split_date: Optional[str] = None  # Only set for equities
    dividend_history: List[DividendHistoryEntry] = Field(default_factory=list)  # Audit log of dividend events
    unfilled_orders: list = Field(default=[], exclude=True)

    @model_validator(mode='before')
    def add_trade_pair_to_orders_and_self(cls, values):
        tp = values['trade_pair']
        if hasattr(tp, 'trade_pair_id'):
            trade_pair_id = tp.trade_pair_id
        else:
            trade_pair_id = tp[0]  # legacy list from disk

        trade_pair = TradePair.get_latest_trade_pair_from_trade_pair_id(trade_pair_id)
        orders = values.get('orders', [])

        # Add the position-level trade_pair to each order
        updated_orders = []
        for order in orders:
            if not isinstance(order, Order):
                order['trade_pair'] = trade_pair
            else:
                order = order.model_copy(update={'trade_pair': trade_pair})

            updated_orders.append(order)
        values['orders'] = updated_orders
        values['trade_pair'] = trade_pair
        return values

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

        # HL positions use per-order taker/maker rates
        if self.is_hl:
            fee = 1.0
            for order in self.orders:
                if order.is_hl_taker is True:
                    fee *= (1 - ValiConfig.HL_TAKER_FEE * abs(order.leverage))
                elif order.is_hl_taker is False:
                    fee *= (1 - ValiConfig.HL_MAKER_FEE * abs(order.leverage))
            return fee

        ans = 1.0 - (self.get_cumulative_leverage() * .001)
        return ans

    # TODO update with ledger update
    def crypto_carry_fee(self, current_time_ms: int) -> (float, int):
        # Fees every 8 hrs. 4 UTC, 12 UTC, 20 UTC
        n_intervals_elapsed, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_crypto(self.open_ms, current_time_ms)
        fee_product = 1.0
        start_ms = self.open_ms
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

    # TODO update with ledger update
    def forex_indices_carry_fee(self, current_time_ms: int) -> (float, int):
        # Fees M-F where W gets triple fee.
        n_intervals_elapsed, time_until_next_interval_ms = TimeUtil.n_intervals_elapsed_forex_indices(self.open_ms, current_time_ms)
        fee_product = 1.0
        start_ms = self.open_ms
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

    def hl_carry_fee(self, current_time_ms: int, funding_rates: dict) -> (float, int):
        """Carry fee for Hyperliquid positions using actual funding rates.

        funding_rates: dict mapping settlement_ms -> rate (e.g. {1722474000000: 0.001})
        Returns (fee_product, next_update_time_ms).
        """
        sign = 1.0 if self.position_type == OrderType.LONG else -1.0
        fee_product = 1.0
        for settlement_ms, rate in sorted(funding_rates.items()):
            if settlement_ms <= self.open_ms:
                continue
            if settlement_ms > current_time_ms:
                break
            lev = self.leverage_at_time(settlement_ms)
            fee_product *= (1 - sign * rate * lev)
        next_update_ms = current_time_ms + TimeUtil.ms_to_next_hour(current_time_ms)
        return fee_product, next_update_ms

    # TODO update with ledger update
    def get_carry_fee(self, current_time_ms, funding_rates=None) -> (float, int):
        # Calculate the number of times a new day occurred (UTC). If a position is opened at 23:59:58 and this function is
        # called at 00:00:02, the carry fee will be calculated as if a day has passed. Another example: if a position is
        # opened at 23:59:58 and this function is called at 23:59:59, the carry fee will be calculated as 0 days have passed
        # Recalculate and update cache
        assert current_time_ms

        if self.is_closed_position and current_time_ms > self.close_ms:
            current_time_ms = self.close_ms

        if current_time_ms < self.open_ms:
            delta = MS_IN_1_HOUR if (self.is_hl and funding_rates is not None) else (MS_IN_8_HOURS if self.trade_pair.is_crypto else MS_IN_24_HOURS)
            return 1.0, min(current_time_ms + delta, self.open_ms)

        # HL positions use actual funding rates when available
        if self.is_hl and funding_rates is not None:
            carry_fee, next_update_time_ms = self.hl_carry_fee(current_time_ms, funding_rates)
        elif self.trade_pair.is_crypto:
            carry_fee, next_update_time_ms = self.crypto_carry_fee(current_time_ms)
        elif self.trade_pair.is_forex or self.trade_pair.is_indices or self.trade_pair.is_equities:
            carry_fee, next_update_time_ms = self.forex_indices_carry_fee(current_time_ms)
        else:
            raise Exception(f'Unexpected trade pair: {self.trade_pair.trade_pair_id}')

        return carry_fee, next_update_time_ms

    def refresh_carry_fee_usd(self, current_time_ms: int, hl_funding_rates: Optional[dict] = None) -> float:
        if self.is_closed_position:
            current_time_ms = self.close_ms

        market_value = abs(self.net_value) + self.unrealized_pnl
        if market_value <= 0:
            return 0.0

        if self.is_hl:
            if not hl_funding_rates:
                return 0

            last_accrual_ms = self._last_fee_time_ms("hl_funding")
            sign = 1.0 if self.position_type == OrderType.LONG else -1.0
            total_fee = 0.0
            last_settlement_ms = last_accrual_ms
            for settlement_ms, rate in sorted(hl_funding_rates.items()):
                if settlement_ms <= last_accrual_ms:
                    continue
                if settlement_ms > current_time_ms:
                    break
                total_fee += market_value * rate * sign
                last_settlement_ms = settlement_ms
            if total_fee > 0:
                self.record_fee_event("hl_funding", total_fee, last_settlement_ms)

            return total_fee

        last_accrual_ms = self._last_fee_time_ms("carry")

        if self.trade_pair.is_crypto:
            interval_ms = MS_IN_8_HOURS
            intervals = (current_time_ms - last_accrual_ms) // interval_ms
            rate = ValiConfig.CARRY_FEE_RATE_PER_INTERVAL[TradePairCategory.CRYPTO]
        elif self.trade_pair.is_forex:
            interval_ms = MS_IN_24_HOURS
            intervals = (current_time_ms - last_accrual_ms) // interval_ms
            rate = ValiConfig.CARRY_FEE_RATE_PER_INTERVAL[TradePairCategory.FOREX]
        else:
            return 0.0

        if intervals <= 0:
            return 0.0

        carry_fee = market_value * rate * intervals
        record_time_ms = last_accrual_ms + intervals * interval_ms
        if carry_fee > 0:
            self.record_fee_event("carry", carry_fee, record_time_ms)

        return carry_fee

    def refresh_equities_fee_usd(self, current_time_ms: int) -> float:
        """
        Calculate and record equity-specific fees accruing at UTC midnight:
          - SHORT positions: stock borrow fee (3% annual / 365) on position market value.
          - LONG positions: margin interest (6.6% annual / 365) on borrowed (margin loan) amount.
        Returns total fee charged.
        """
        if self.is_closed_position or not self.trade_pair.is_equities:
            return 0.0

        most_recent_midnight_ms = (current_time_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS
        total_fee = 0.0

        if self.position_type == OrderType.SHORT:
            short_position_value = abs(self.net_value) + self.unrealized_pnl
            if short_position_value > 0:
                last_borrow_accrual_ms = self._last_fee_time_ms("borrow")
                intervals = (most_recent_midnight_ms - last_borrow_accrual_ms) // MS_IN_24_HOURS
                if intervals > 0:
                    borrow_fee = short_position_value * ValiConfig.DAILY_STOCK_BORROW_RATE * intervals
                    if borrow_fee > 0:
                        self.record_fee_event("borrow", borrow_fee, most_recent_midnight_ms)
                        total_fee += borrow_fee

        elif self.position_type == OrderType.LONG:
            borrowed = self.margin_loan
            if borrowed > 0:
                last_interest_accrual_ms = self._last_fee_time_ms("interest")
                intervals = (most_recent_midnight_ms - last_interest_accrual_ms) // MS_IN_24_HOURS
                if intervals > 0:
                    interest_fee = borrowed * ValiConfig.DAILY_INTEREST_RATE * intervals
                    if interest_fee > 0:
                        self.record_fee_event("interest", interest_fee, most_recent_midnight_ms)
                        total_fee += interest_fee

        return total_fee

    def _last_fee_time_ms(self, fee_type: str) -> int:
        for fee_event in reversed(self.fee_history):
            if fee_event["fee_type"] == fee_type:
                return fee_event["time_ms"]
        return self.open_ms

    def record_fee_event(self, fee_type: str, amount: float, time_ms: int):
        if amount <= 0:
            return

        self.fee_history.append({
            "fee_type": fee_type,
            "amount": amount,
            "time_ms": time_ms
        })
        self.fee_history.sort(key=lambda fee: fee["time_ms"])


    @property
    def total_fees(self) -> float:
        return sum(fee["amount"] for fee in self.fee_history)

    @property
    def initial_entry_price(self) -> float:
        if not self.orders or len(self.orders) == 0:
            return 0.0
        first_order = self.orders[0]
        return first_order.price * (1 + first_order.slippage) if first_order.leverage > 0 else first_order.price * (1 - first_order.slippage)

    @property
    def margin_loan(self) -> float:
        """Total margin loan for this position (sum of all orders' margin loans)"""
        if not self.orders:
            return 0.0
        return sum(order.margin_loan for order in self.orders)

    def __hash__(self):
        # Include specified fields in the hash, assuming trade_pair is accessible and immutable
        return hash((self.miner_hotkey, self.position_uuid, self.open_ms, self.current_return,
                     self.net_leverage, self.net_quantity, self.net_value, self.initial_entry_price, self.trade_pair.trade_pair))

    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.miner_hotkey == other.miner_hotkey and
                self.position_uuid == other.position_uuid and
                self.open_ms == other.open_ms and
                self.current_return == other.current_return and
                self.net_leverage == other.net_leverage and
                self.net_quantity == other.net_quantity and
                self.net_value == other.net_value and
                self.initial_entry_price == other.initial_entry_price and
                self.trade_pair.trade_pair == other.trade_pair.trade_pair)

    def _handle_trade_pair_encoding(self, d):
        # Remove trade_pair from orders
        orders = d.get("orders", None)
        if orders:
            for order in orders:
                order.pop('trade_pair', None)

        tp = d['trade_pair']
        if isinstance(tp, list):
            d['trade_pair'] = tp[:5]
        else:
            # Pydantic v2 serializes TradePairLike as a dict in Union contexts;
            # reconstruct the 5-element list from the live object instead
            tp_obj = self.trade_pair
            if isinstance(tp_obj, TradePair):
                d['trade_pair'] = tp_obj.value[:5]
            else:
                # DynamicTradePair (HL-only pair not in the static enum)
                d['trade_pair'] = [
                    tp_obj.trade_pair_id,
                    tp_obj.trade_pair,
                    tp_obj.fees,
                    tp_obj.min_leverage,
                    tp_obj.max_leverage,
                ]
        return d

    def to_dict(self):
        d = self.model_dump(mode="json")
        return self._handle_trade_pair_encoding(d)

    def to_dashboard(self, positions_time_ms: int, filled_orders, unfilled_orders) -> dict:
        results = {
            "tp": self.trade_pair.trade_pair,
            "t": self.position_type.name,
            "o": self.open_ms,
            "r": self.current_return,
            "ap": self.average_entry_price,
            "rp": self.realized_pnl,
            "up": self.unrealized_pnl,
        }

        if self.net_leverage:
            results["nl"] = self.net_leverage

        if self.is_closed_position:
            results["c"] = self.close_ms
            results["rc"] = self.return_at_close

        if filled_orders:
            results["fo"] = filled_orders

        if unfilled_orders:
            results["uo"] = unfilled_orders

        dashboard_fee_history = {}
        for fee_event in self.fee_history:
            fee_time_ms = fee_event["time_ms"]
            if fee_time_ms > positions_time_ms:
                dashboard_fee_history[str(fee_time_ms)] = {
                    "t": fee_event["fee_type"],
                    "a": fee_event["amount"]
                }

        if dashboard_fee_history:
            results["fh"] = dashboard_fee_history

        return results

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

    def add_unfilled_order(self, order_dict: dict) -> None:
        """Add or update an unfilled bracket order dict on this position."""
        order_uuid = order_dict.get('order_uuid')
        if order_uuid:
            self.unfilled_orders = [o for o in self.unfilled_orders if o.order_uuid != order_uuid]
            self.unfilled_orders.append(Order.from_dict(order_dict))

    def remove_unfilled_order(self, order_uuid: str) -> bool:
        """Remove an unfilled order by UUID. Returns True if found."""
        for i, order in enumerate(self.unfilled_orders):
            if order.order_uuid == order_uuid:
                self.unfilled_orders.pop(i)
                return True
        return False

    def clear_unfilled_orders(self) -> None:
        """Clear all unfilled orders."""
        self.unfilled_orders = []

    def newest_order_age_ms(self, now_ms):
        if len(self.orders) > 0:
            return now_ms - self.orders[-1].processed_ms
        return -1

    def __str__(self):
        return json.dumps(self.to_dict())

    def to_copyable_str(self):
        ans = self.model_dump()
        ans['trade_pair'] = f'TradePair.{self.trade_pair.trade_pair_id}'
        ans['position_type'] = f'OrderType.{self.position_type.name}'
        for o in ans['orders']:
            o['trade_pair'] = f'TradePair.{self.trade_pair.trade_pair_id}'
            o['order_type'] = f'OrderType.{o["order_type"].name}'

        s = str(ans)
        s = re.sub(r"'(TradePair\.[A-Z]+|OrderType\.[A-Z]+|FLAT|SHORT|LONG)'", r"\1", s)

        return s

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

    def rebuild_position_with_updated_orders(self, price_fetcher_client):
        self.current_return = 1.0
        self.close_ms = None
        self.return_at_close = 1.0
        self.net_leverage = 0.0
        self.net_quantity = 0.0
        self.net_value = 0.0
        self.average_entry_price = 0.0
        self.cumulative_entry_value = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.position_type = None
        self.is_closed_position = False
        self.position_type = None

        self._update_position(price_fetcher_client)

    def log_position_status(self):
        bt.logging.debug(
            f"position details: "
            f"close_ms [{self.close_ms}] "
            f"initial entry price [{self.initial_entry_price}] "
            f"net leverage [{self.net_leverage}] "
            f"net quantity [{self.net_quantity}] "
            f"net value [{self.net_value}] "
            f"average entry price [{self.average_entry_price}] "
            f"return_at_close [{self.return_at_close}]"
        )
        order_info = [
            {
                "order type": order.order_type.value,
                "leverage": order.leverage,
                "quantity": order.quantity,
                "price": order,
            }
            for order in self.orders
        ]
        bt.logging.debug(f"position order details: " f"close_ms [{order_info}] ")

    def add_order(self, order: Order, live_price_fetcher=None, transaction_fee: float = 0):
        """
        Add an order to a position, and adjust its size to stay within
        the trade pair max and portfolio max.

        Args:
            order: The order to add
            live_price_fetcher: Price fetcher for position updates
            net_portfolio_leverage: Deprecated, no longer used
            skip_validation: If True, skip order size validation
            balance: Miner's balance for USD-based validation. If None, uses account_size.
            max_position_leverage: Max leverage for trade pair. If None, uses trade pair max.
        """
        if self.is_closed_position:
            raise ValueError("Miner attempted to add order to a closed/liquidated position. Ignoring.")
        if order.trade_pair != self.trade_pair:
            raise ValueError(
                f"Order trade pair [{order.trade_pair}] does not match position trade pair [{self.trade_pair}]")

        self.validate_order_size(order)
        self.orders.append(order)

        if transaction_fee:
            self.record_fee_event("transaction", transaction_fee, order.processed_ms)

        self._update_position(live_price_fetcher)

    def calculate_pnl(self, current_price, live_price_fetcher, t_ms=None, order=None):
        if self.initial_entry_price == 0 or self.average_entry_price is None:
            return 1

        if not t_ms:
            t_ms = TimeUtil.now_in_millis()

        # pnl with slippage
        if order:
            # update realized pnl for orders that reduce the size of a position
            if order.order_type != self.position_type or self.position_type == OrderType.FLAT:
                exit_price = current_price * (1 + order.slippage) if order.leverage > 0 else current_price * (1 - order.slippage)
                order_realized_pnl_quote = -1 * (exit_price - self.average_entry_price) * (order.quantity * order.trade_pair.lot_size)
                order.realized_pnl = order_realized_pnl_quote * order.quote_usd_rate
                self.realized_pnl += order.realized_pnl

            unrealized_quantity = min(self.net_quantity, self.net_quantity + order.quantity, key=abs)
            unrealized_pnl_quote = (current_price - self.average_entry_price) * (unrealized_quantity * order.trade_pair.lot_size)
            self.unrealized_pnl = unrealized_pnl_quote * order.quote_usd_rate
        else:
            unrealized_pnl_quote = (current_price - self.average_entry_price) * (self.net_quantity * self.trade_pair.lot_size)
            quote_usd_conversion = self.orders[-1].quote_usd_rate # live_price_fetcher.get_usd_conversion(self.trade_pair.quote, t_ms, self.orders[-1].order_type, self.position_type)  # TODO: calculate conversion rate at current time instead of last order time
            self.unrealized_pnl = unrealized_pnl_quote * quote_usd_conversion

        if self.cumulative_entry_value == 0:
            gain = 0
        else:
            gain = (self.realized_pnl + self.unrealized_pnl) / self.account_size

        # Check if liquidated
        if gain <= -1.0:
            return 0
        net_return = 1 + gain
        return net_return

    def leverage_at_time(self, target_ms: int) -> float:
        """Return the absolute net leverage at a specific timestamp.

        Walks orders up to target_ms and returns the cumulative absolute leverage,
        handling FLAT orders and leverage flips the same way as max_leverage_seen.
        """
        current_leverage = 0.0
        for order in self.orders:
            if order.processed_ms > target_ms:
                break
            prev_leverage = current_leverage
            current_leverage += order.leverage
            if order.order_type == OrderType.FLAT or self._leverage_flipped(prev_leverage, current_leverage):
                current_leverage = 0.0
        return abs(current_leverage)

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
        assert interval_data['max_leverage'] >= 0, (interval_data, self.orders, str(self))
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

    def _handle_liquidation(self, time_ms, price_fetcher_client):
        self._position_log("position liquidated. Trade pair: " + str(self.trade_pair.trade_pair_id))
        if self.is_closed_position:
            return
        else:
            self.orders.append(self.generate_fake_flat_order(self, time_ms, price_fetcher_client))
            self.close_out_position(time_ms)

    @staticmethod
    def generate_fake_flat_order(position, elimination_time_ms, price_fetcher_client, extra_price_source=None, src=None):
        fake_flat_order_time = elimination_time_ms
        price_source = price_fetcher_client.get_close_at_date(
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
                position=position
            )
            # Use provided src or default to PRICE_FILLED_ELIMINATION_FLAT
            if src is None:
                src = OrderSource.PRICE_FILLED_ELIMINATION_FLAT
        else:
            bt.logging.warning(f'Unexpectedly unable to fetch price for trade pair {position.trade_pair.trade_pair_id}'
                               f' at time {TimeUtil.millis_to_formatted_date_str(elimination_time_ms)} during fake flat order'
                               f'creation. Setting price to 0. and src to OrderSource.ELIMINATION_FLAT')
            price = 0
            # Use provided src or default to ELIMINATION_FLAT
            if src is None:
                src = OrderSource.ELIMINATION_FLAT


        flat_order = Order(price=price,
                           processed_ms=fake_flat_order_time,
                           order_uuid=position.position_uuid[::-1],  # deterministic across validators. Won't mess with p2p sync
                           trade_pair=position.trade_pair,
                           order_type=OrderType.FLAT,
                           leverage=-position.net_leverage,
                           value=-position.net_value,
                           quantity=-position.net_quantity,
                           src=src,
                           price_sources=[x for x in (price_source, extra_price_source) if x is not None])
        flat_order.quote_usd_rate = price_fetcher_client.get_quote_usd_conversion(flat_order, position)
        flat_order.usd_base_rate = price_fetcher_client.get_usd_base_conversion(position.trade_pair, fake_flat_order_time, price, OrderType.FLAT, position)
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

    def set_returns(self, realtime_price, price_fetcher_client=None, time_ms=None, total_fees=None, order=None):
        # We used to multiple trade_pair.fees by net_leverage. Eventually we will
        # Update this calculation to approximate actual exchange fees.
        self.current_return = self.calculate_pnl(realtime_price, price_fetcher_client, t_ms=time_ms, order=order)
        if total_fees is None:
            self.return_at_close = self.calculate_return_with_fees(self.current_return,
                               timestamp_ms=TimeUtil.now_in_millis() if time_ms is None else time_ms)
        else:
            self.return_at_close = self.current_return * total_fees

        if self.current_return < 0:
            raise ValueError(f"current return must be positive {self.current_return}")

        if self.current_return == 0:
            self._handle_liquidation(TimeUtil.now_in_millis() if time_ms is None else time_ms, price_fetcher_client)

    def update_position_state_for_new_order(self, order, delta_quantity, delta_leverage, price_fetcher_client=None):
        """
        Must be called after every order to maintain accurate internal state. The variable average_entry_price has
        a name that can be a little confusing. Although it claims to be the average price, it really isn't.
        For example, it can take a negative value. A more accurate name for this variable is the weighted average
        entry price.
        """
        realtime_price = order.price
        assert self.initial_entry_price > 0, self.initial_entry_price
        new_net_quantity = self.net_quantity + delta_quantity
        new_net_leverage = self.net_leverage + delta_leverage
        if order.src in (OrderSource.ELIMINATION_FLAT, OrderSource.DEPRECATION_FLAT) and (order.price==0 or order.usd_base_rate==0 or order.quote_usd_rate==0):
            self.net_leverage = 0.0
            self.net_quantity = 0.0
            self.net_value = 0.0
            return  # Don't set returns since the price is zero'd out.
        self.set_returns(realtime_price, price_fetcher_client, time_ms=order.processed_ms, order=order)

        # Liquidated
        if self.current_return == 0:
            return
        self._position_log(f"closed position total w/o fees [{self.current_return}]. Trade pair: {self.trade_pair.trade_pair_id}")
        self._position_log(f"closed return with fees [{self.return_at_close}]. Trade pair: {self.trade_pair.trade_pair_id}")

        if self.position_type == OrderType.FLAT:
            self.net_leverage = 0.0
            self.net_quantity = 0.0
            self.net_value = 0.0
        else:
            if self.position_type == order.order_type:
                # average entry price only changes when an order is in the same direction as the position. reducing a position does not affect average entry price.
                entry_price = order.price * (1 + order.slippage) if order.leverage > 0 else order.price * (1 - order.slippage)
                self.average_entry_price = (
                    self.average_entry_price * self.net_quantity
                    + entry_price * delta_quantity
                ) / new_net_quantity
                entry_value = order.value
            else:
                # order is reducing the size of a position, so there is no entry cost.
                entry_value = 0

            self.cumulative_entry_value += entry_value
            self.net_quantity = new_net_quantity
            self.net_value = (realtime_price * order.quote_usd_rate) * (self.net_quantity * self.trade_pair.lot_size)
            self.net_leverage = new_net_leverage    # self.net_value / self.account_size

    def initialize_position_from_first_order(self, order):
        self.open_ms = order.processed_ms
        if self.initial_entry_price <= 0:
            raise ValueError("Initial entry price must be > 0")
        # Initialize the position type. It will stay the same until the position is closed.
        if order.leverage > 0:
            self._position_log("setting new position type as LONG. Trade pair: " + str(self.trade_pair.trade_pair_id))
            self.position_type = OrderType.LONG
        elif order.leverage < 0:
            self._position_log("setting new position type as SHORT. Trade pair: " + str(self.trade_pair.trade_pair_id))
            self.position_type = OrderType.SHORT
        else:
            bt.logging.error(
                f"Position {self.position_uuid} has zero leverage initial order for "
                f"{self.trade_pair.trade_pair_id}. Closing with 0 realized PnL."
            )
            self.position_type = order.order_type if order.order_type != OrderType.FLAT else OrderType.LONG
            self.close_out_position(order.processed_ms)

    def close_out_position(self, close_ms):
        self.position_type = OrderType.FLAT
        self.is_closed_position = True
        self.close_ms = close_ms

    def reopen_position(self):
        self.position_type = self.orders[0].order_type
        self.is_closed_position = False
        self.close_ms = None

    def validate_order_size(self, order: Order, max_position_value: Optional[float] = None) -> bool:
        """
        returns True if clamped due to max position value
        """
        if order.order_type == OrderType.FLAT:
            return False

        # Validate order min leverage
        min_order_lev, max_order_lev = leverage_utils.get_order_leverage_bounds()
        if abs(order.leverage) > max_order_lev:
            raise ValueError(
                f"{self.trade_pair.trade_pair_id}: order leverage {abs(order.leverage):.5f} exceeds maximum {max_order_lev}")
        is_opening_or_increasing = self.position_type is None or order.order_type == self.position_type
        if is_opening_or_increasing and abs(order.leverage) < min_order_lev:
            raise ValueError(
                f"{self.trade_pair.trade_pair_id}: order leverage {abs(order.leverage):.5f} below minimum {min_order_lev}")

        proposed_leverage = self.net_leverage + (order.leverage or 0)
        proposed_quantity = self.net_quantity + (order.quantity or 0)
        proposed_value = self.net_value + self.unrealized_pnl + (order.value or 0)

        bt.logging.info(f"[POSITION VALIDATION] unrealized pnl: {self.unrealized_pnl}")
        bt.logging.info(f"[POSITION VALIDATION] proposed quantity: {proposed_quantity}, proposed_value: {proposed_value}")

        # Flatten order
        flatten = False
        if self.position_type == OrderType.LONG:
            flatten = proposed_quantity <= 0 or proposed_value <= 0
        elif self.position_type == OrderType.SHORT:
            flatten = proposed_quantity >= 0 or proposed_value >= 0

        if flatten:
            order.order_type = OrderType.FLAT
            order.leverage = -self.net_leverage
            order.quantity = -self.net_quantity
            order.value = -self.net_value
            return False

        # If order increases position size, validate max position size
        clamped = False
        if order.order_type == self.position_type and max_position_value is not None:
            if abs(self.net_value + self.unrealized_pnl) >= max_position_value:
                raise ValueError(f"Position at max ${abs(self.net_value):.2f} (limit: ${max_position_value:.2f})")

            max_order_value = max_position_value - abs(self.net_value)
            if abs(order.value) > max_order_value:
                sign = 1 if self.position_type == OrderType.LONG else -1
                order.value = sign * max_order_value
                order.quantity = (order.value * order.usd_base_rate) / order.trade_pair.lot_size
                proposed_quantity = self.net_quantity + order.quantity
                clamped = True

        # Validate against min position size
        if self.trade_pair.is_forex:
            proposed_lots = abs(proposed_quantity)
            if proposed_lots > 0 and proposed_lots < ValiConfig.FOREX_MIN_POSITION_SIZE_LOTS_NANO:
                raise ValueError(
                    f"{self.trade_pair.trade_pair_id}: position size {proposed_lots:.4f} lots is below minimum {ValiConfig.FOREX_MIN_POSITION_SIZE_LOTS_NANO} lots")
        elif self.trade_pair.is_crypto:
            if abs(proposed_value) > 0 and abs(proposed_value) < ValiConfig.CRYPTO_MIN_POSITION_SIZE_USD:
                raise ValueError(
                    f"{self.trade_pair.trade_pair_id}: position size ${abs(proposed_value):.2f} is below minimum ${ValiConfig.CRYPTO_MIN_POSITION_SIZE_USD:.2f}")
        elif self.trade_pair.is_equities:
            proposed_shares = abs(proposed_quantity)
            if proposed_shares > 0 and proposed_shares < ValiConfig.EQUITIES_MIN_POSITION_SIZE_SHARES:
                raise ValueError(
                    f"{self.trade_pair.trade_pair_id}: position size {proposed_shares:.4f} shares is below minimum {ValiConfig.EQUITIES_MIN_POSITION_SIZE_SHARES} shares")
        else:  # for other asset classes
            min_position_leverage, _ = leverage_utils.get_position_leverage_bounds(self.trade_pair)
            if abs(proposed_leverage) < min_position_leverage:
                raise ValueError(
                    f"{self.trade_pair.trade_pair_id}: position leverage {abs(proposed_leverage):.4f}x is below minimum {min_position_leverage}x")

        return clamped

    def apply_stock_split(self, stock_split_ratio: float, execution_date: str) -> bool:
        """
        Apply stock split to position. Returns True if applied, False if already applied.
        Only applicable to equities positions.
        """
        if not self.trade_pair.is_equities:
            return False

        if self.last_stock_split_date == execution_date:
            bt.logging.info(f"Stock split for {execution_date} already applied to position {self.position_uuid}")
            return False

        for order in self.orders:
            order.quantity *= stock_split_ratio
            order.price /= stock_split_ratio

        self.last_stock_split_date = execution_date
        self._update_position()
        return True

    def apply_dividend(self, gross_dividend: float, ex_date_str: str, payment_date_str: str, time_ms: int) -> Optional[float]:
        """
        Apply dividend at ex-date.
        - SHORT positions dividends are deducted on ex-date
        - LONG positions are entitled to dividends for shares held before the ex date.

        Returns -amount for shorts (immediate debit), None for longs (pending credit recorded), or None if inapplicable.
        """
        if self.is_closed_position or not self.trade_pair.is_equities:
            return None

        # Position must have been opened before the ex-dividend date to be eligible
        if TimeUtil.millis_to_short_date_str(self.open_ms) >= ex_date_str:
            return None

        # only one entry per ex_date per position
        if any(e.ex_date == ex_date_str for e in self.dividend_history):
            return None

        shares = self.net_quantity  # positive = long, negative = short
        if shares == 0:
            return None

        amount = abs(self.net_quantity) * gross_dividend
        if shares > 0:  # LONG: record pending credit to be released on payment_date
            self.dividend_history.append(DividendHistoryEntry(
                type="long_credit",
                gross_dividend=gross_dividend,
                quantity=shares,
                amount=amount,
                ex_date=ex_date_str,
                payment_date=payment_date_str,
                time_ms=time_ms,
                applied=False,
            ))
            return 0.0
        else:  # SHORT: debit immediately
            self.dividend_history.append(DividendHistoryEntry(
                type="short_debit",
                gross_dividend=gross_dividend,
                quantity=abs(shares),
                amount=amount,
                ex_date=ex_date_str,
                payment_date=ex_date_str,
                time_ms=time_ms,
                applied=True,
            ))
            self.record_fee_event("dividend_liability", amount, time_ms)
            return -amount

    def settle_pending_dividends(self, current_date_str: str) -> float:
        """Mark long_credit entries with matching payment_date as applied. Returns total USD credit."""
        total = 0.0
        for entry in self.dividend_history:
            if (entry.type == "long_credit"
                    and entry.payment_date <= current_date_str
                    and not entry.applied):
                entry.applied = True
                total += entry.amount
        return total

    def _update_position(self, price_fetcher_client=None):
        self.net_leverage = 0.0
        self.net_quantity = 0.0
        self.net_value = 0.0
        self.cumulative_entry_value = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        bt.logging.trace(f"Updating position {self.trade_pair.trade_pair_id} with n orders: {len(self.orders)}")
        for order in self.orders:
            # set value and quantity if not set
            if (order.value is None or order.quantity is None) and order.leverage is not None:
                order.value = order.leverage * self.account_size
                if order.price == 0:
                    order.quantity = 0
                else:
                    order.quantity = (order.value * order.usd_base_rate) / order.trade_pair.lot_size

            if self.position_type is None:
                self.initialize_position_from_first_order(order)

            # Check if the new order flattens the position, explicitly or implicitly
            if self.position_type == OrderType.LONG and self.net_quantity + order.quantity <= 0 or \
               self.position_type == OrderType.SHORT and self.net_quantity + order.quantity >= 0 or \
               order.order_type == OrderType.FLAT:
                #self._position_log(
                #    f"Flattening {self.position_type.value} position from order {order}"
                #)
                self.close_out_position(order.processed_ms)
                # Set the order quantity
                order.leverage = -self.net_leverage
                order.quantity = -self.net_quantity
                order.value = -self.net_value

            # Reflect the current order in the current position's return.
            adjusted_quantity = (
                0.0 if self.position_type == OrderType.FLAT else order.quantity
            )
            adjusted_leverage = (
                0.0 if self.position_type == OrderType.FLAT else order.leverage
            )
            #bt.logging.info(
            #    f"Updating position state for new order {order} with adjusted leverage {adjusted_quantity}"
            #)
            self.update_position_state_for_new_order(order, adjusted_quantity, adjusted_leverage, price_fetcher_client)


            # If the position is already closed, we don't need to process any more orders. break in case there are more orders.
            if self.position_type == OrderType.FLAT:
                break
