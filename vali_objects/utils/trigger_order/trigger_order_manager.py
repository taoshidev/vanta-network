"""
Trigger order rules: placement checks, cancellation, and the daemon tick that fires orders.

State lives in TriggerOrderStore and evaluation in trigger_engine; this class only decides.
Orders whose trigger is already crossed at placement are rejected, never filled as market orders.
"""
from shared_objects.log import logger
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.limit_order.order_utils import OrderSize
from vali_objects.utils.trigger_order import trigger_engine
from vali_objects.utils.trigger_order.trigger_order_store import TriggerOrderStore
from vali_objects.vali_config import RPCConnectionMode, TradePair, ValiConfig
from vali_objects.vali_dataclasses.price_source import NASDAQ_BASIC_TRADE_SOURCE
from vali_objects.vali_dataclasses.trigger_order import (
    FixedTrigger,
    TrailingTrigger,
    TriggerAction,
    TriggerOrder,
)

PRO_TRANSITION_REJECTION = (
    "Your account is transitioning to a Pro Account. You cannot open new positions or increase "
    "existing ones - close your open positions to begin trading your Pro Account."
)

FILL_SOURCE = {
    ExecutionType.LIMIT: OrderSource.LIMIT_FILLED,
    ExecutionType.BRACKET: OrderSource.BRACKET_FILLED,
    ExecutionType.STOP_LIMIT: OrderSource.STOP_LIMIT_FILLED,
}

CANCEL_REASON = {
    ExecutionType.LIMIT: OrderSource.LIMIT_CANCELLED,
    ExecutionType.BRACKET: OrderSource.BRACKET_CANCELLED,
    ExecutionType.STOP_LIMIT: OrderSource.STOP_LIMIT_CANCELLED,
}

POSITION_COPY_INTERVAL_MS = 60 * 1000  # how often a moving trailing stop is re-copied to its position


class TriggerOrderManager:
    def __init__(self, store: TriggerOrderStore | None = None, running_unit_tests=False,
                 connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
        from vali_objects.utils.market_order.market_order_client import MarketOrderClient

        self._market_order_client = MarketOrderClient(running_unit_tests=running_unit_tests, connection_mode=connection_mode)
        self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests, connection_mode=connection_mode)
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT, connect_immediately=False, connection_mode=connection_mode
        )
        self._miner_account_client = MinerAccountClient(running_unit_tests=running_unit_tests, connection_mode=connection_mode)

        self.store = store or TriggerOrderStore(running_unit_tests, position_client=self._position_client)
        self._last_fill_ms: dict[tuple[str, str], int] = {}  # (hotkey, trade_pair_id) -> last fill
        self._last_copy_ms: dict[str, int] = {}              # order_uuid -> last position copy
        self._needs_position_copy = True                     # re-copy bound orders on the first tick

    # ==================== Placement ====================

    def place(self, hotkey: str, order: TriggerOrder) -> dict:
        """Validate and open an order. Placing a uuid that is already open is a no-op (a retry)."""
        if order.miner_hotkey != hotkey:
            raise SignalException(f"Trigger order [{order.order_uuid}] belongs to another hotkey")

        with self.store.lock(hotkey, order.trade_pair):
            # Before validation, so a retry isn't rejected for a price move or its own slot
            if self.store.get(hotkey, order.order_uuid):
                return {"status": "success", "order_uuid": order.order_uuid}

            open_orders = len(self.store.select(hotkey=hotkey))
            if open_orders >= ValiConfig.MAX_UNFILLED_LIMIT_ORDERS:
                raise SignalException(
                    f"Too many open trigger orders [{open_orders}] >= [{ValiConfig.MAX_UNFILLED_LIMIT_ORDERS}]"
                )

            self._validate(order, self._open_position(hotkey, order.trade_pair))
            self.store.add(order)

        logger.info(f"[TRIGGER] Placed {order.execution_type.name} [{order.order_uuid}] {order.trade_pair.trade_pair_id} for [{hotkey}]")
        return {"status": "success", "order_uuid": order.order_uuid}

    def modify(self, hotkey: str, order: TriggerOrder) -> dict:
        """Replace an open order with the same uuid. Runs the same checks as place."""
        with self.store.lock(hotkey, order.trade_pair):
            existing = self.store.get(hotkey, order.order_uuid)
            if existing is None:
                raise SignalException(f"Cannot modify [{order.order_uuid}]: no open trigger order")
            if existing.trade_pair != order.trade_pair:
                raise SignalException(f"Cannot modify [{order.order_uuid}]: trade pair does not match")

            self._validate(order, self._open_position(hotkey, order.trade_pair))
            self.store.replace(order)

        logger.info(f"[TRIGGER] Modified [{order.order_uuid}] for [{hotkey}]")
        return {"status": "success", "order_uuid": order.order_uuid}

    def _validate(self, order: TriggerOrder, position) -> None:
        if self._is_entry(order, position) and self._is_transitioning_to_pro(order.miner_hotkey):
            raise SignalException(PRO_TRANSITION_REJECTION)

        if order.bind_to_position:
            if position is None:
                raise SignalException(f"Trigger order [{order.order_uuid}] requires an open position")
            action = order.market_action
            if action and action.reduce_only and action.order_type == position.position_type:
                raise SignalException(
                    f"Trigger order [{order.order_uuid}] is reduce_only but would add to the {position.position_type.name} position"
                )
            order.position_uuid = position.position_uuid

        self._reject_if_crossed(order)
        parent_price = order.trigger.price if isinstance(order.trigger, FixedTrigger) else None
        self._validate_children(order, parent_price)

    def _reject_if_crossed(self, order: TriggerOrder) -> None:
        """Seed an unset trailing trigger, then reject if the trigger is already crossed."""
        trade_pair = order.trade_pair
        price_sources = None
        if self._is_trading(trade_pair, order.processed_ms):
            price_sources = self._live_price_client.get_sorted_price_sources_for_trade_pair(trade_pair, order.processed_ms)

        trigger = order.trigger
        if isinstance(trigger, TrailingTrigger) and trigger.best_price is None:
            if not price_sources:
                raise SignalException(f"No price available to start trailing stop [{order.order_uuid}]")
            trigger.update_best_price(trigger_engine.observed_price(price_sources[0], trigger.price_field))

        if trigger.price <= 0:
            raise SignalException(f"Trigger order [{order.order_uuid}] trigger price {trigger.price} must be positive")

        if price_sources and trigger_engine.evaluate(trigger, price_sources[:1]):
            raise SignalException(
                f"{order.execution_type.name} order for {trade_pair.trade_pair_id} rejected: trigger price "
                f"{trigger.price} is already crossed. Submit a MARKET order to fill now, or adjust the trigger."
            )

    def _validate_children(self, action: TriggerAction, parent_price: float | None) -> None:
        """reduce_only children (SL/TP) must not already be crossed at the parent's trigger price."""
        for child in action.child_actions:
            if (
                parent_price is not None
                and child.market_action and child.market_action.reduce_only
                and isinstance(child.trigger, FixedTrigger)
                and trigger_engine.would_fire(child.trigger, parent_price)
            ):
                raise SignalException(
                    f"Child [{child.order_uuid}] trigger {child.trigger.price} would fire at its parent's price {parent_price}"
                )
            self._validate_children(child, child.trigger.price if isinstance(child.trigger, FixedTrigger) else None)

    # ==================== Cancellation and reads ====================

    def cancel(self, hotkey: str, now_ms: int, *, order_uuid: str | None = None, trade_pair: TradePair | None = None,
               position_uuid: str | None = None, oco_group: str | None = None, entry_only: bool = False,
               reason: OrderSource | None = None) -> list[str]:
        """
        Cancel open orders matching every given filter; no filters cancels all of the hotkey's
        orders. entry_only keeps orders that only reduce. Returns the cancelled uuids.
        """
        if order_uuid:
            orders = [o for o in [self.store.get(hotkey, order_uuid)] if o]
        else:
            orders = self.store.select(hotkey=hotkey, trade_pair=trade_pair)
        if position_uuid:
            orders = [o for o in orders if o.position_uuid == position_uuid]
        if oco_group:
            orders = [o for o in orders if o.oco_group == oco_group]
        if entry_only and orders:
            positions = {
                p.trade_pair.trade_pair_id: p
                for p in self._position_client.get_positions_for_one_hotkey(hotkey, only_open_positions=True)
            }
            orders = [o for o in orders if self._is_entry(o, positions.get(o.trade_pair.trade_pair_id))]

        cancelled = []
        for order in orders:
            if self.store.remove(hotkey, order.order_uuid, now_ms, reason or CANCEL_REASON[order.execution_type]):
                self._last_copy_ms.pop(order.order_uuid, None)
                cancelled.append(order.order_uuid)
        return cancelled

    def get(self, hotkey: str, order_uuid: str) -> TriggerOrder | None:
        return self.store.get(hotkey, order_uuid)

    def select(self, hotkey: str | None = None, trade_pair: TradePair | None = None) -> list[TriggerOrder]:
        return self.store.select(hotkey=hotkey, trade_pair=trade_pair)

    def reopen(self, hotkey: str, reason: OrderSource = OrderSource.ELIMINATION_CANCELLED) -> int:
        return self.store.reopen(hotkey, reason)

    def health_check(self) -> dict:
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "open_orders": len(self.store.select()),
            "num_trade_pairs": len(self.store.trade_pairs()),
        }

    # ==================== Daemon ====================

    def tick(self, now_ms: int | None = None) -> dict:
        """Evaluate every open order. At most one fill per hotkey per trade pair per fill interval."""
        now_ms = now_ms or TimeUtil.now_in_millis()
        checked = fired = 0

        if self._needs_position_copy:
            self.store.copy_all_to_positions()
            self._needs_position_copy = False

        for trade_pair in self.store.trade_pairs():
            if trade_pair.is_blocked:
                continue
            if not self._is_trading(trade_pair, now_ms):
                continue
            start_ms = now_ms - ValiConfig.LIMIT_ORDER_PRICE_BUFFER_MS
            price_sources = self._live_price_client.get_ws_price_sources_in_window(trade_pair, start_ms, now_ms)
            if not price_sources:
                continue

            for hotkey in self.store.hotkeys(trade_pair):
                c, f = self._tick_hotkey(hotkey, trade_pair, price_sources, now_ms)
                checked += c
                fired += f

        if fired:
            logger.info(f"[TRIGGER] Tick checked={checked} fired={fired}")
        return {"checked": checked, "fired": fired, "timestamp_ms": now_ms}

    def _tick_hotkey(self, hotkey, trade_pair, price_sources, now_ms) -> tuple[int, int]:
        key = (hotkey, trade_pair.trade_pair_id)
        last_fill_ms = self._last_fill_ms.get(key, 0)
        fill_allowed = now_ms - last_fill_ms >= ValiConfig.LIMIT_ORDER_FILL_INTERVAL_MS
        checked = 0

        with self.store.lock(hotkey, trade_pair):
            position = self._open_position(hotkey, trade_pair)
            for order in self.store.select(hotkey=hotkey, trade_pair=trade_pair):
                checked += 1
                if order.position_uuid and not self._still_bound(order, position):
                    self.cancel(hotkey, now_ms, order_uuid=order.order_uuid)
                    continue

                # Trailing stops track the window even when no fill is allowed this tick
                if trigger_engine.update_trailing(order.trigger, price_sources, order.processed_ms):
                    copy = now_ms - self._last_copy_ms.get(order.order_uuid, 0) >= POSITION_COPY_INTERVAL_MS
                    self.store.replace(order, copy_to_position=copy)
                    if copy:
                        self._last_copy_ms[order.order_uuid] = now_ms

                if not fill_allowed:
                    continue

                fire = trigger_engine.evaluate(order.trigger, price_sources, max(order.processed_ms, last_fill_ms))
                if fire and self._fire(order, fire, position):
                    return checked, 1
        return checked, 0

    def _fire(self, order: TriggerOrder, fire: trigger_engine.Fire, position) -> bool:
        """Fill the order (if it has a fill), then close it and open its children."""
        fire_ms = fire.price_source.start_ms
        hotkey = order.miner_hotkey
        position_uuid = order.position_uuid
        fill_price = fire.trigger_price

        action = order.market_action
        if action:
            if self._is_entry(order, position) and self._is_transitioning_to_pro(hotkey):
                self.cancel(hotkey, fire_ms, order_uuid=order.order_uuid, reason=OrderSource.PRO_TRANSITION_CANCELLED)
                return False
            try:
                filled_order, updated_position = self._market_order_client.execute_order(
                    hotkey, order.order_uuid, order.trade_pair, order.execution_type,
                    action.order_type, _signed_size(action.order_type, action.size),
                    # A trade (pre-market/after-hours) at or through the limit fills at the limit price
                    fill_price=fire.trigger_price if fire.price_source.source == NASDAQ_BASIC_TRADE_SOURCE else None,
                    trigger_price=fire.trigger_price,
                    price_sources=[fire.price_source],
                    order_src=FILL_SOURCE[order.execution_type],
                    now_ms=fire_ms,
                    slippage=0,
                    is_hl_taker=action.is_taker,
                    enforce_cooldown=False,
                )
            except Exception as e:
                logger.error(f"[TRIGGER] Fill failed for [{order.order_uuid}], cancelling: {e}")
                self.cancel(hotkey, fire_ms, order_uuid=order.order_uuid)
                return False

            fill_price = filled_order.price
            open_after_fill = updated_position and not updated_position.is_closed_position
            position_uuid = updated_position.position_uuid if open_after_fill else None
            self._last_fill_ms[(hotkey, order.trade_pair.trade_pair_id)] = fire_ms

        children = order.create_child_orders(fire_ms, fill_price, position_uuid)
        dropped = [c.order_uuid for c in children if c.bind_to_position and c.position_uuid is None]
        if dropped:
            logger.info(f"[TRIGGER] [{order.order_uuid}] left no open position, dropping bound children {dropped}")
        self.store.fire(order, [c for c in children if c.order_uuid not in dropped], fire_ms)
        self._last_copy_ms.pop(order.order_uuid, None)

        logger.info(
            f"[TRIGGER] Fired {order.execution_type.name} [{order.order_uuid}] {order.trade_pair.trade_pair_id} "
            f"for [{hotkey}] at trigger {fire.trigger_price}"
        )
        return True

    # ==================== Helpers ====================

    def _open_position(self, hotkey: str, trade_pair: TradePair):
        return self._position_client.get_open_position_for_trade_pair(hotkey, trade_pair.trade_pair_id)

    @staticmethod
    def _still_bound(order: TriggerOrder, position) -> bool:
        """The bound position is still open, is the same position, and hasn't flipped."""
        if position is None or position.position_uuid != order.position_uuid:
            return False
        action = order.market_action
        return not (action and action.reduce_only and action.order_type == position.position_type)

    def _is_trading(self, trade_pair: TradePair, time_ms: int) -> bool:
        """Market open, or an equity in pre-market/after-hours (priced from Nasdaq Basic trades)."""
        return (
            self._live_price_client.is_market_open(trade_pair, time_ms)
            or self._live_price_client.get_equity_session(trade_pair, time_ms) in ('pre', 'post')
        )

    @classmethod
    def _is_entry(cls, action: TriggerAction, position) -> bool:
        """True when the action (or any child it places) would open a position or add to one."""
        fill = action.market_action
        if fill:
            if fill.reduce_only or fill.order_type == OrderType.FLAT:
                return False
            return position is None or fill.order_type == position.position_type
        return any(cls._is_entry(child, position) for child in action.child_actions)

    def _is_transitioning_to_pro(self, hotkey: str) -> bool:
        account = self._miner_account_client.get_account(hotkey)
        return account is not None and account.miner_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION


def _signed_size(order_type: OrderType, size: OrderSize) -> OrderSize:
    """execute_order expects LONG sizes positive and SHORT negative; bracket_pct is unsigned."""
    if order_type == OrderType.FLAT:
        return size
    sign = 1 if order_type == OrderType.LONG else -1
    return OrderSize(
        quantity=sign * abs(size.quantity) if size.quantity is not None else None,
        leverage=sign * abs(size.leverage) if size.leverage is not None else None,
        value=sign * abs(size.value) if size.value is not None else None,
        bracket_pct=size.bracket_pct,
    )
