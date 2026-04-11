import json
import logging
from copy import deepcopy
from typing import Dict, Optional, List
from pydantic import model_validator, BaseModel, Field

from time_util.time_util import TimeUtil, MS_IN_8_HOURS, MS_IN_24_HOURS
from vali_objects.vali_config import TradePair, TradePairCategory, TradePairLike, DynamicTradePair, ValiConfig
from vali_objects.vali_dataclasses.corporate_actions import DividendHistoryEntry
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils import leverage_utils
import bittensor as bt
import math



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
    account_size: float = 0.0               # USD (deprecated, retained for backward compatibility)
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
        if 'orders' in d:
            for order in d['orders']:
                if 'trade_pair' in order:
                    del order['trade_pair']
        # Write the trade_pair in the legacy tuple format as to not break generate_request_outputs. This is temporary
        # code until generate_request_outputs is updated to have the new TradePair decoding logic. If BTC or ETH, put
        # the legacy fee value so that pydantic can validate the JSON with the original decoding logic
        tp_val = d['trade_pair']
        if isinstance(tp_val, TradePair):
            fee = .003 if tp_val.is_crypto else tp_val.fees
            d['trade_pair'] = [tp_val.trade_pair_id, tp_val.trade_pair, fee, tp_val.min_leverage, tp_val.max_leverage]
        elif isinstance(tp_val, DynamicTradePair):
            # Defensive: shouldn't reach here after model_dump(), but handle correctly if it does.
            d['trade_pair'] = [tp_val.trade_pair_id, tp_val.trade_pair, tp_val.fees, tp_val.min_leverage, tp_val.max_leverage]
        elif isinstance(tp_val, dict) and 'hl_coin' in tp_val:
            # Pydantic v2 model_dump() converts Python @dataclass fields to plain dicts.
            d['trade_pair'] = [tp_val['trade_pair_id'], tp_val['trade_pair'], tp_val.get('fees', 0.001), tp_val.get('min_leverage', 0.01), tp_val['max_leverage']]
        else:
            d['trade_pair'] = tp_val[:5]
            if d['trade_pair'][0] in (TradePair.BTCUSD.trade_pair_id, TradePair.ETHUSD.trade_pair_id):
                d['trade_pair'][2] = 0.003
        return d

    def to_dict(self):
        d = deepcopy(self.model_dump())
        return self._handle_trade_pair_encoding(d)

    def to_dashboard(self, positions_time_ms: int, filled_orders, unfilled_orders) -> dict:
        results = {
            "tp": self.trade_pair.trade_pair,
            "t": self.position_type.name,
            "o": self.open_ms,
            "r": self.current_return,
            "ap": self.average_entry_price,
            "rp": self.realized_pnl,
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
            self.unfilled_orders = [o for o in self.unfilled_orders if o.get('order_uuid') != order_uuid]
            self.unfilled_orders.append(order_dict)

    def remove_unfilled_order(self, order_uuid: str) -> bool:
        """Remove an unfilled order by UUID. Returns True if found."""
        for i, order_dict in enumerate(self.unfilled_orders):
            if order_dict.get('order_uuid') == order_uuid:
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
        return self.to_json_string()

    def to_json_string(self) -> str:
        # Using pydantic's model_dump_json method with built-in validation
        json_str = self.model_dump_json()
        # Unfortunately, we can't tell pydantic v2 to strip certain fields so we do that here
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

        self._update_position()

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
            transaction_fee: Optional transaction fee in USD to record.
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

        self._update_position()

    def _leverage_flipped(self, prev_leverage, cur_leverage):
        return prev_leverage * cur_leverage < 0 or prev_leverage != 0 and cur_leverage == 0

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

    def set_returns(self, realtime_price, quote_usd_conversion=None):
        if self.initial_entry_price == 0 or self.average_entry_price is None:
            self.current_return = 1
            return

        if quote_usd_conversion is None:
            quote_usd_conversion = self.orders[-1].quote_usd_rate

        unrealized_pnl_quote = (realtime_price - self.average_entry_price) * (self.net_quantity * self.trade_pair.lot_size)
        self.unrealized_pnl = unrealized_pnl_quote * quote_usd_conversion

        self.current_return = 1 + (self.realized_pnl + self.unrealized_pnl) / self.cumulative_entry_value
        self.return_at_close = 1 + (self.realized_pnl + self.unrealized_pnl - self.total_fees) / self.cumulative_entry_value

    def update_position_state_for_new_order(self, order, delta_quantity, delta_leverage):
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

        # Update realized PnL for orders that reduce or close a position
        if self.initial_entry_price != 0 and self.average_entry_price is not None:
            if order.order_type != self.position_type or self.position_type == OrderType.FLAT:
                exit_price = realtime_price * (1 + order.slippage) if order.leverage > 0 else realtime_price * (1 - order.slippage)
                order_realized_pnl_quote = -1 * (exit_price - self.average_entry_price) * (order.quantity * order.trade_pair.lot_size)
                self.realized_pnl += order_realized_pnl_quote * order.quote_usd_rate

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
                self.cumulative_entry_value += order.value
            self.net_quantity = new_net_quantity
            self.net_value = (realtime_price * order.quote_usd_rate) * (self.net_quantity * self.trade_pair.lot_size)
            self.net_leverage = new_net_leverage

        self.set_returns(realtime_price, quote_usd_conversion=order.quote_usd_rate)

        # Liquidated
        if self.current_return == 0:
            return
        self._position_log(f"closed position total w/o fees [{self.current_return}]. Trade pair: {self.trade_pair.trade_pair_id}")
        self._position_log(f"closed return with fees [{self.return_at_close}]. Trade pair: {self.trade_pair.trade_pair_id}")

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
            if proposed_lots > 0 and proposed_lots < ValiConfig.FOREX_MIN_POSITION_SIZE_LOTS:
                raise ValueError(
                    f"{self.trade_pair.trade_pair_id}: position size {proposed_lots:.4f} lots is below minimum {ValiConfig.FOREX_MIN_POSITION_SIZE_LOTS} lots")
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

    def _update_position(self, price_fetcher_client=None):
        self.net_leverage = 0.0
        self.net_quantity = 0.0
        self.net_value = 0.0
        self.cumulative_entry_value = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        bt.logging.trace(f"Updating position {self.trade_pair.trade_pair_id} with n orders: {len(self.orders)}")
        for order in self.orders:
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
            self.update_position_state_for_new_order(order, adjusted_quantity, adjusted_leverage)


            # If the position is already closed, we don't need to process any more orders. break in case there are more orders.
            if self.position_type == OrderType.FLAT:
                break

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
