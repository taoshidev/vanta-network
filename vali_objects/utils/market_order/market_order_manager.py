"""

Modularize the logic that was originally in validator.py. No IPC communication here.
"""
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.miner_account.miner_account_manager import MinerAccount
from vali_objects.trade_pair import TradePairSource
from vali_objects.vali_dataclasses.price_source import PriceSource
from vanta_api.websocket_notifier import WebSocketNotifierClient
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
import bittensor as bt

from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.trade_pair import TradePair
from vali_objects.utils.leverage_utils import get_max_order_size
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.utils.limit_order.order_utils import OrderSize, convert_order_sizes
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from shared_objects.locks.position_lock_client import PositionLockClient
from shared_objects.log import logger


class MarketOrderManager():

    def __init__(self, serve:bool, running_unit_tests=False, connection_mode=RPCConnectionMode.RPC):
        self.serve = serve
        self.running_unit_tests = running_unit_tests

        ws_connection_mode = RPCConnectionMode.LOCAL if running_unit_tests else connection_mode

        self.websocket_notifier = WebSocketNotifierClient(connection_mode=ws_connection_mode, connect_immediately=False)

        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._live_price_client = LivePriceFetcherClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._entity_collateral_client = EntityCollateralClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests, connect_immediately=False)
        self._position_client = PositionManagerClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._position_lock_client = PositionLockClient(running_unit_tests=running_unit_tests)

        self.last_order_time_cache = {}

    def execute_order(
        self,
        hotkey: str,
        order_uuid: str,
        trade_pair: TradePair,
        execution_type: ExecutionType,
        order_type: OrderType,
        order_size: OrderSize,
        *,
        fill_price: float | None = None,
        price_sources: list[PriceSource] | None = None,
        slippage: float | None = None,
        order_src: OrderSource = OrderSource.ORGANIC,
        is_hl: bool = False,
        is_hl_taker: bool | None = None,
        now_ms: int | None = None,
        enforce_cooldown: bool = True,
    ) -> tuple[Order, Position] | None:
        _start = TimeUtil.now_in_millis()
        now_ms = now_ms or _start
        if enforce_cooldown:
            err = self.enforce_order_cooldown(trade_pair.trade_pair_id, now_ms, hotkey)
            if err:
                raise SignalException(err)

        if not self._live_price_client.is_market_open(trade_pair):
            raise SignalException(f"The market for {trade_pair.trade_pair_id} is currently closed.")

        logger.info(
            f"[ORDER_EXECUTION] {hotkey} {order_uuid} {trade_pair.trade_pair_id} "
            f"{order_type} {execution_type} size={order_size} src={order_src}"
        )

        with self._position_lock_client.get_lock(hotkey, trade_pair.trade_pair_id):

            miner_account = self._miner_account_client.get_or_create(hotkey)
            position = self._position_client.get_open_position_for_trade_pair(hotkey, trade_pair.trade_pair_id)
            position_type = position.position_type if position else order_type

            if fill_price is None or not price_sources:
                price_sources = self._live_price_client.get_sorted_price_sources_for_trade_pair(trade_pair, now_ms)
                if not price_sources:
                    raise SignalException(f"Order Rejected: no live prices being found for {trade_pair.trade_pair_id}. Please try again.")

                if fill_price is None:
                    fill_price = price_sources[0].parse_appropriate_price(now_ms, trade_pair.is_forex, order_type, position_type)

            usd_base_rate = self._live_price_client.get_usd_base_conversion(trade_pair, now_ms, fill_price, order_type, position_type)
            quote_usd_rate = self._live_price_client.get_quote_usd_conversion(trade_pair, now_ms, fill_price, order_type, position_type)

            logger.info(
                f"[ORDER_EXECUTION] {hotkey} {order_uuid} pricing: fill={fill_price} usd_base={usd_base_rate:.6g} "
                f"quote_usd={quote_usd_rate:.6g} source={price_sources[0].source} "
                f"bid/ask={price_sources[0].bid}/{price_sources[0].ask}"
            )

            if position is not None and len(position.orders) >= ValiConfig.MAX_ORDERS_PER_POSITION and order_type != OrderType.FLAT:
                logger.info(
                    f"[ORDER_EXECUTION] {hotkey} hit {ValiConfig.MAX_ORDERS_PER_POSITION} order limit. "
                    f"Auto-closing {trade_pair.trade_pair_id} with {len(position.orders)} orders."
                )
                flat_uuid = position.position_uuid[::-1]
                self._apply_order(
                    position, miner_account,
                    ExecutionType.MARKET, OrderType.FLAT, OrderSize(quantity=-position.net_quantity),
                    flat_uuid, now_ms - 1, price_sources,
                    OrderSource.MAX_ORDERS_PER_POSITION_CLOSE,
                    fill_price, usd_base_rate, quote_usd_rate,
                    slippage=0,
                )
                position = None

            if not position:
                if order_type == OrderType.FLAT:
                    return None
                position = Position(
                    miner_hotkey=hotkey,
                    position_uuid=order_uuid,
                    open_ms=now_ms,
                    trade_pair=trade_pair,
                    position_type=order_type,
                    account_size=miner_account.account_size,
                    is_hl=is_hl,
                )

            order = self._apply_order(
                position, miner_account,
                execution_type, order_type, order_size,
                order_uuid, now_ms, price_sources, order_src,
                fill_price, usd_base_rate, quote_usd_rate,
                slippage, is_hl_taker
            )
            logger.info(f"[ORDER_EXECUTION] {hotkey} {order_uuid} completed in {TimeUtil.now_in_millis() - _start}ms")
            return order, position

    def _apply_order(
        self,
        position: Position,
        miner_account: MinerAccount,
        execution_type: ExecutionType,
        order_type: OrderType,
        order_size: OrderSize,
        order_uuid: str,
        now_ms: int,
        price_sources: list[PriceSource],
        order_src: OrderSource,
        fill_price: float,
        usd_base_rate: float,
        quote_usd_rate: float,
        slippage: float | None = None,
        is_hl_taker: bool | None = None,
    ) -> Order:
        """Build and execute an order. Caller must hold the position lock."""
        trade_pair = position.trade_pair
        hotkey = position.miner_hotkey
        balance = miner_account.balance
        best_price_source = price_sources[0]

        quantity, leverage, value, bracket_pct = order_size
        if bracket_pct:
            quantity = -position.net_quantity * bracket_pct
        if order_type == OrderType.FLAT or quantity == -position.net_quantity:
            order_type = OrderType.FLAT
            sizing = OrderSize(quantity=-position.net_quantity)
        else:
            sizing = OrderSize(quantity=quantity, leverage=leverage, value=value)

        use_nano_increment = miner_account.account_size <= ValiConfig.FOREX_SMALL_ACCOUNT_THRESHOLD
        quantity, leverage, value = convert_order_sizes(
            sizing, usd_base_rate, trade_pair, balance,
            round_qty=(order_type != OrderType.FLAT),
            use_nano_increment=use_nano_increment,
        )

        logger.info(
            f"[ORDER_EXECUTION] {hotkey} {order_uuid} sizing: qty={quantity:.6g} val=${value:.4f} (from {order_size})"
        )

        position.set_returns(fill_price, quote_usd_conversion=quote_usd_rate)
        if self._is_effective_close(position, order_type, quantity, value):
            order_type = OrderType.FLAT
            quantity, leverage, value = -position.net_quantity, -position.net_leverage, -position.net_value

        is_buy = order_type == position.position_type
        if is_buy:
            max_order_value, binding_cap = get_max_order_size(miner_account, position)
            if max_order_value <= 0:
                raise SignalException(f"{hotkey} {order_uuid} No buying power remaining for {trade_pair.trade_pair_id} (capped by {binding_cap})")
            sign = -1 if order_type == OrderType.SHORT else 1
            clamped_value = sign * min(abs(value), max_order_value)
            if abs(clamped_value) < abs(value):
                logger.info(
                    f"[ORDER_EXECUTION] {hotkey} {order_uuid} order value clamped from ${value:.4f} to ${clamped_value:.4f} by {binding_cap}"
                )
            value = clamped_value
            quantity, leverage, value = convert_order_sizes(
                OrderSize(value=value), usd_base_rate, trade_pair, balance,
                round_qty=True,
                use_nano_increment=use_nano_increment,
            )

        if abs(value) < 1e-9 or abs(quantity) < 1e-9:
            raise SignalException("Error processing order: 0 order size after clamping")

        order = Order(
            trade_pair=trade_pair,
            order_type=order_type,
            quantity=quantity,
            value=value,
            leverage=leverage,
            price=fill_price,
            processed_ms=now_ms,
            order_uuid=order_uuid,
            price_sources=price_sources,
            bid=best_price_source.bid or 0,
            ask=best_price_source.ask or 0,
            src=order_src,
            execution_type=execution_type,
            usd_base_rate=usd_base_rate,
            quote_usd_rate=quote_usd_rate,
            is_hl_taker=is_hl_taker,
        )

        if slippage is None:
            slippage = self._live_price_client.calculate_slippage(order.bid, order.ask, order)
        order.slippage = slippage

        logger.info(f"[ORDER_EXECUTION] {hotkey} {order_uuid} slippage={order.slippage:.6g}")

        if not order.value or not order.quantity:
            raise SignalException(f"Order value and quantity must be set before applying order. value={order.value}, quantity={order.quantity}")

        if is_buy:
            allowed, reason = self._entity_collateral_client.try_gate_position_open(hotkey, order.value)
            if not allowed:
                logger.error(f"{hotkey} {order_uuid} Entity cross-margin check failed for subaccount [{hotkey}]: {reason}")
                raise SignalException(f"Account cross-margin validation failed. Please contact an administrator.")  # better msg for the user?

        if is_buy and trade_pair.is_equities and trade_pair.src == TradePairSource.VANTA and miner_account.asset_class == MinerAssetClass.EQUITIES:
            cash_available = balance - (miner_account.capital_used - miner_account.total_borrowed_amount)
            if abs(order.value) > cash_available:
                order.margin_loan = abs(order.value) * 0.5

        realized_pnl, transaction_fee, loan_repaid = position.add_order(order)

        if is_buy:
            self._miner_account_client.process_order_buy(
                hotkey, abs(order.value), order.margin_loan, transaction_fee, trade_pair.trade_pair_category
            )
        else:
            entry_value = abs(order.quantity) * trade_pair.lot_size * position.average_entry_price * order.quote_usd_rate
            self._miner_account_client.process_order_sell(
                hotkey, entry_value, realized_pnl, loan_repaid, transaction_fee, trade_pair.trade_pair_category
            )
            self._entity_collateral_client.try_slash_on_position_close(hotkey, realized_pnl)

        self._position_client.save_miner_position(position)

        self.last_order_time_cache[(hotkey, trade_pair.trade_pair_id)] = now_ms

        if self.serve:
            self.websocket_notifier.broadcast_position_update(position)

        return order

    def close_positions(self, hotkey: str, position_uuids: list[str] | None = None, close_all: bool = False, now_ms: int | None = None):
        logger.info(f"Processing close_positions for miner [{hotkey}] (close_all={close_all})")

        open_positions = self._position_client.get_positions_for_hotkeys([hotkey], only_open_positions=True).get(hotkey)

        if not open_positions:
            logger.warning(f"No open positions found for miner [{hotkey}]")
            return

        if close_all:
            positions_to_close = open_positions
        else:
            target_uuids = set(position_uuids or [])
            positions_to_close = [p for p in open_positions if p.position_uuid in target_uuids]
            if not positions_to_close:
                logger.info(f"No matching positions found for UUIDs {position_uuids} for miner [{hotkey}]")
                return

        total_positions = len(positions_to_close)
        cnt_closed = 0

        logger.info(f"Found {total_positions} positions to close for miner [{hotkey}]")

        for i, position in enumerate(positions_to_close):
            position_close_uuid = f"{position.position_uuid}-flat-all"
            try:
                self.execute_order(
                    hotkey=hotkey,
                    order_uuid=position_close_uuid,
                    trade_pair=position.trade_pair,
                    execution_type=ExecutionType.MARKET,
                    order_type=OrderType.FLAT,
                    order_size=OrderSize(),
                    now_ms=now_ms,
                    enforce_cooldown=False,
                )
                cnt_closed += 1

            except Exception as e:
                logger.error(
                    f"[FLAT_ALL] Failed to close position {i+1}/{total_positions} "
                    f"[{position.trade_pair.trade_pair_id}] for miner [{hotkey}]: {str(e)}"
                )
                continue

        logger.info(
            f"[FLAT_ALL] Completed for miner [{hotkey}]: "
            f"{cnt_closed}/{total_positions} positions closed"
        )

    @staticmethod
    def _is_effective_close(position: Position, order_type: OrderType, order_quantity: float, order_value: float) -> bool:
        if order_type == OrderType.FLAT:
            return True
        if position.position_type is None or position.position_type == order_type:
            return False
        proposed_quantity = position.net_quantity + order_quantity
        position_sign = 1 if position.position_type == OrderType.LONG else -1
        proposed_value = position.net_value + position_sign * position.unrealized_pnl + order_value
        if position.position_type == OrderType.LONG:
            return proposed_quantity <= 1e-6 or proposed_value <= 1.0
        if position.position_type == OrderType.SHORT:
            return proposed_quantity >= -1e-6 or proposed_value >= -1.0
        return False

    def enforce_order_cooldown(self, trade_pair_id, now_ms, miner_hotkey) -> str | None:
        """
        Enforce cooldown between orders for the same trade pair using an efficient cache.
        This method must be called within the position lock to prevent race conditions.
        """
        cache_key = (miner_hotkey, trade_pair_id)
        current_order_time_ms = now_ms

        # Get the last order time from cache
        cached_last_order_time = self.last_order_time_cache.get(cache_key, 0)
        msg = None
        if cached_last_order_time > 0:
            time_since_last_order_ms = current_order_time_ms - cached_last_order_time

            if time_since_last_order_ms < ValiConfig.ORDER_COOLDOWN_MS:
                previous_order_time = TimeUtil.millis_to_formatted_date_str(cached_last_order_time)
                current_time = TimeUtil.millis_to_formatted_date_str(current_order_time_ms)
                time_to_wait_in_s = (ValiConfig.ORDER_COOLDOWN_MS - time_since_last_order_ms) / 1000
                msg = (
                    f"Order for trade pair [{trade_pair_id}] was placed too soon after the previous order. "
                    f"Last order was placed at [{previous_order_time}] and current order was placed at [{current_time}]. "
                    f"Please wait {time_to_wait_in_s:.1f} seconds before placing another order."
                )

        return msg

    def clear_order_cooldown_cache(self):
        """Clear the order cooldown cache. Used for testing."""
        if not self.running_unit_tests:
            raise Exception('clear_order_cooldown_cache can only be called in unit test mode')
        self.last_order_time_cache.clear()
        logger.debug("Cleared market order cooldown cache")

