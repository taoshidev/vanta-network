import uuid
import json
from dataclasses import dataclass
from typing import Optional

import bittensor as bt

from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.trade_pair import NATIVE_CRYPTO_TO_HL_TRADE_PAIR, TradePair
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
from vali_objects.utils.market_order.market_order_client import MarketOrderClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.utils.limit_order.order_utils import OrderSize

def _validate_required(value, field_type: type):
    if not value:
        raise SignalException(f"{field_type.__name__} could not be validated")
    return value


def _validate_single_size(quantity: float | None, value: float | None, leverage: float | None):
    specified = sum(x is not None for x in (quantity, value, leverage))
    if specified != 1:
        raise SignalException(
            "Exactly one of quantity, value, or leverage must be provided."
        )

@dataclass(frozen=True)
class OrderProcessingResult:
    """
    Standardized result from order processing.

    Attributes:
        execution_type: Type of execution (MARKET, LIMIT, BRACKET, LIMIT_CANCEL)
        success: Whether processing succeeded (always True if no exception raised)
        order: The created/processed Order object (None for LIMIT_CANCEL)
        result_dict: Result dictionary (used for LIMIT_CANCEL response)
        updated_position: Updated position (used for MARKET orders)
        should_track_uuid: Whether to add UUID to tracker (False for LIMIT_CANCEL)
    """
    execution_type: ExecutionType
    success: bool = True
    order: Optional[Order] = None
    result_dict: Optional[dict] = None
    updated_position: Optional[Position] = None
    should_track_uuid: bool = True

    @property
    def order_for_logging(self) -> Optional[Order]:
        return self.order

    def get_response_json(self) -> str:
        if self.order:
            return self.order.__str__()
        elif self.result_dict:
            return json.dumps(self.result_dict)
        return ""


class OrderProcessor:
    def __init__(
        self,
        limit_order_client: LimitOrderClient,
        market_order_client: MarketOrderClient,
        miner_account_client: MinerAccountClient,
    ):
        self.limit_order_client = limit_order_client
        self.market_order_client = market_order_client
        self.miner_account_client = miner_account_client

    def validate(
        self,
        hotkey: str,
        execution_type: ExecutionType,
        trade_pair: TradePair | None,
        order_type: OrderType | None,
    ) -> tuple[bool, str, Optional[TradePair]]:
        """
        Returns (ok, error_msg, resolved_trade_pair). Covers elimination,
        native crypto remapping, trade pair validity, asset class, and market hours.
        """
        # Native crypto remapping
        if trade_pair is not None and trade_pair in NATIVE_CRYPTO_TO_HL_TRADE_PAIR:
            trade_pair = NATIVE_CRYPTO_TO_HL_TRADE_PAIR[trade_pair]
            bt.logging.info(f"Remapped native trade pair {trade_pair.trade_pair_id}")

        # Trade pair checks
        if execution_type not in (ExecutionType.LIMIT_CANCEL, ExecutionType.LIMIT_EDIT, ExecutionType.FLAT_ALL):
            if trade_pair is None:
                return False, "Invalid trade pair in signal.", None
            if trade_pair.is_blocked:
                return False, f"Trade pair [{trade_pair.trade_pair_id}] is no longer supported.", trade_pair

        # Flat-only check
        if trade_pair is not None and trade_pair.is_flat_only and order_type != OrderType.FLAT:
            return False, f"Trade pair [{trade_pair.trade_pair_id}] is being discontinued. Please close your position.", trade_pair

        miner_account = self.miner_account_client.get_account(hotkey)
        if not miner_account:
            msg = f"Miner account for hotkey [{hotkey}] is not yet initialized. Please try again shortly."
            bt.logging.warning(msg)
            return False, msg, trade_pair

        # Elimination check
        if miner_account.miner_bucket == MinerBucket.ELIMINATED:
            msg = (
                f"This miner hotkey {hotkey} has been eliminated and cannot participate in this subnet. "
                f"Try again after re-registering."
            )
            bt.logging.warning(msg)
            return False, msg, None

        # Asset class check (FLAT market orders bypass this)
        if trade_pair is not None and order_type != OrderType.FLAT and execution_type not in (ExecutionType.MARKET, ExecutionType.FLAT_ALL):
            if not miner_account.asset_class:
                msg = (
                    f"No asset class selected for hotkey [{hotkey}]. "
                    f"Select an asset class using the Vanta CLI before placing orders: "
                    f"https://github.com/taoshidev/vanta-cli"
                )
                return False, msg, trade_pair
            if not miner_account.asset_class.can_trade(trade_pair):
                msg = f"Selected asset class [{miner_account.asset_class}] cannot submit orders for trade pair [{trade_pair.trade_pair}]."
                return False, msg, trade_pair

        return True, "", trade_pair

    def process_vanta_signal(
        self,
        hotkey: str,
        signal: Signal,
        order_uuid: str,
        now_ms: int,
    ) -> OrderProcessingResult:
        order_uuid = order_uuid or str(uuid.uuid4())
        execution_type = signal.execution_type

        if execution_type == ExecutionType.MARKET:
            return self.process_market_order(hotkey, signal, order_uuid, now_ms)

        elif execution_type == ExecutionType.FLAT_ALL:
            return self.process_flat_all(hotkey, order_uuid, now_ms)

        elif execution_type in (ExecutionType.LIMIT, ExecutionType.BRACKET, ExecutionType.STOP_LIMIT):
            order = self.process_unfilled_order(hotkey, signal, order_uuid, now_ms, execution_type)
            return OrderProcessingResult(execution_type=execution_type, order=order, should_track_uuid=True)

        elif execution_type == ExecutionType.LIMIT_CANCEL:
            result = self.process_limit_cancel(hotkey, signal.trade_pair, order_uuid, now_ms)
            return OrderProcessingResult(execution_type=ExecutionType.LIMIT_CANCEL, result_dict=result, should_track_uuid=False)

        elif execution_type == ExecutionType.LIMIT_EDIT:
            order = self.process_limit_edit(hotkey, signal, order_uuid, now_ms)
            return OrderProcessingResult(execution_type=ExecutionType.LIMIT_EDIT, order=order, should_track_uuid=False)

        else:
            raise SignalException(f"{hotkey} {order_uuid} invalid execution type")

    def process_market_order(
        self,
        hotkey: str,
        signal: Signal,
        order_uuid: str,
        now_ms: int,
    ) -> OrderProcessingResult:
        order_size = OrderSize(leverage=signal.leverage, value=signal.value, quantity=signal.quantity)
        trade_pair = _validate_required(signal.trade_pair, TradePair)
        order_type = _validate_required(signal.order_type, OrderType)
        result = self.market_order_client.execute_order(
            hotkey,
            order_uuid,
            trade_pair,
            ExecutionType.MARKET,
            order_type,
            order_size,
            order_src=OrderSource.ORGANIC,
            now_ms=now_ms,
            enforce_cooldown=True,
        )
        if result is None:
            return OrderProcessingResult(ExecutionType.MARKET)

        created_order, updated_position = result
        if updated_position and updated_position.is_closed_position:
            self.process_limit_cancel(hotkey, trade_pair, "ALL", now_ms, ExecutionType.BRACKET)

        if created_order and signal.bracket_orders:
            created_order.bracket_orders = signal.bracket_orders
            self.limit_order_client.create_sltp_order(hotkey, created_order)

        return OrderProcessingResult(
            execution_type=ExecutionType.MARKET,
            order=created_order,
            updated_position=updated_position,
            should_track_uuid=True,
        )

    def process_flat_all(
        self,
        hotkey: str,
        order_uuid: str,
        now_ms: int,
    ) -> OrderProcessingResult:
        close_all = bool(order_uuid and order_uuid.strip().upper() == "ALL")
        position_uuids = None if close_all else [u.strip() for u in order_uuid.split(',')] if order_uuid else []
        self.market_order_client.close_positions(
            hotkey=hotkey,
            position_uuids=position_uuids,
            close_all=close_all,
            now_ms=now_ms,
        )
        self.process_limit_cancel(hotkey, None, "ALL", now_ms, ExecutionType.BRACKET)
        return OrderProcessingResult(execution_type=ExecutionType.FLAT_ALL, should_track_uuid=True)


    def process_hyperliquid_order(
        self,
        hotkey: str,
        order_uuid: str,
        trade_pair: TradePair,
        order_type: OrderType,
        order_size: OrderSize,
        *,
        fill_price: Optional[float] = None,
        price_sources: Optional[list[PriceSource]] = None,
        is_taker: Optional[bool] = None,
        now_ms: Optional[int] = None,
    ):
        """Direct market execution with HL defaults: no cooldown, HYPERLIQUID source, is_hl=True."""
        _validate_required(trade_pair, TradePair)
        _validate_required(order_type, OrderType)
        _validate_single_size(order_size.quantity, order_size.value, order_size.leverage)
        return self.market_order_client.execute_order(
            hotkey,
            order_uuid,
            trade_pair,
            ExecutionType.MARKET,
            order_type,
            order_size,
            fill_price=fill_price,
            price_sources=price_sources,
            slippage=0.0,
            order_src=OrderSource.HYPERLIQUID,
            is_hl=True,
            is_hl_taker=is_taker,
            now_ms=now_ms,
            enforce_cooldown=False,
        )

    _UNFILLED_SRC_BY_EXEC = {
        ExecutionType.LIMIT: OrderSource.LIMIT_UNFILLED,
        ExecutionType.STOP_LIMIT: OrderSource.STOP_LIMIT_UNFILLED,
        ExecutionType.BRACKET: OrderSource.BRACKET_UNFILLED,
    }

    def process_unfilled_order(
        self,
        hotkey: str,
        signal: Signal,
        order_uuid: str,
        now_ms: int,
        execution_type: ExecutionType,
        is_edit: bool = False,
    ) -> Order:
        if execution_type == ExecutionType.BRACKET:
            # Default bracket_pct=1.0 when no explicit size is provided.
            if (signal.leverage is None and signal.value is None
                    and signal.quantity is None and signal.bracket_pct is None):
                signal = signal.model_copy(update={"bracket_pct": 1.0})
            order_type = OrderType.FLAT
        else:
            _validate_single_size(signal.quantity, signal.value, signal.leverage)
            order_type = _validate_required(signal.order_type, OrderType)

        trade_pair = _validate_required(signal.trade_pair, TradePair)
        order_src = _validate_required(self._UNFILLED_SRC_BY_EXEC.get(execution_type), OrderSource)
        order = Order(
            trade_pair=trade_pair,
            order_uuid=order_uuid,
            processed_ms=now_ms,
            price=0.0,
            order_type=order_type,
            leverage=signal.leverage,
            value=signal.value,
            quantity=signal.quantity,
            execution_type=execution_type,
            limit_price=signal.limit_price,
            stop_price=signal.stop_price,
            stop_condition=signal.stop_condition,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=signal.trailing_stop,
            bracket_orders=signal.bracket_orders,
            bracket_pct=signal.bracket_pct,
            src=order_src,
        )
        if execution_type == ExecutionType.BRACKET:
            # BRACKET orders never carry a limit_price.
            order.limit_price = None

        self.limit_order_client.process_limit_order(hotkey, order, is_edit=is_edit)
        bt.logging.info(
            f"[ORDER_PROCESSOR] Processed {execution_type.name} order{'(EDIT)' if is_edit else ''}: "
            f"{order.order_uuid} for {hotkey}"
        )
        return order

    def process_limit_cancel(
        self,
        hotkey: str,
        trade_pair: Optional[TradePair],
        order_uuid: str,
        now_ms: int,
        execution_type: Optional[ExecutionType] = None,
    ) -> dict:
        result = self.limit_order_client.cancel_limit_order(
            hotkey,
            trade_pair.trade_pair_id if trade_pair else None,
            order_uuid,
            now_ms,
            execution_type,
        )
        bt.logging.debug(f"Cancelled LIMIT order(s) for {hotkey}: {order_uuid or 'all'}")
        return result

    def process_limit_edit(
        self,
        hotkey: str,
        signal: Signal,
        order_uuid: str,
        now_ms: int,
    ) -> Optional[Order]:
        existing_order_dict = self.limit_order_client.get_limit_order_by_uuid(hotkey, order_uuid)

        # If the envelope uuid is also listed in bracket_orders, defer to the
        # bulk path so a single request can't double-process the same order.
        if existing_order_dict:
            bracket_orders_list = signal.bracket_orders or []
            if any(b.get("order_uuid") == order_uuid for b in bracket_orders_list):
                existing_order_dict = None

        if existing_order_dict:
            existing_order = Order.from_dict(existing_order_dict)

            if existing_order.src not in [OrderSource.LIMIT_UNFILLED, OrderSource.BRACKET_UNFILLED, OrderSource.STOP_LIMIT_UNFILLED]:
                raise SignalException(f"Cannot edit order {order_uuid}: order is not unfilled (status: {existing_order.src})")
            if signal.trade_pair is None:
                raise SignalException("Invalid trade pair in signal for LIMIT_EDIT")
            if signal.trade_pair != existing_order.trade_pair:
                raise SignalException(f"Cannot edit order {order_uuid}: trade pair mismatch")

            # Execution type cannot be mutated via edit; preserve the existing one.
            edit_signal = signal.model_copy(update={"execution_type": existing_order.execution_type})
            return self.process_unfilled_order(
                hotkey, edit_signal, order_uuid, now_ms, existing_order.execution_type, is_edit=True
            )

        if signal.bracket_orders:
            return self._apply_bulk_bracket_update(hotkey, signal, now_ms)

        raise SignalException(f"Cannot edit order {order_uuid}: order not found")

    def _apply_bulk_bracket_update(
        self, hotkey: str, signal: Signal, now_ms: int
    ) -> Optional[Order]:
        trade_pair = _validate_required(signal.trade_pair, TradePair)
        processed_orders: list[Order] = []
        cancelled_count = 0

        if not signal.bracket_orders:
            raise SignalException("No bracket edits to process")

        for bracket in signal.bracket_orders:
            bracket_uuid = bracket.get("order_uuid")
            if not bracket_uuid:
                bracket_uuid = str(uuid.uuid4())
                # raise SignalException("bulk bracket order updates must require bracket uuid") TODO enforce

            existing_bracket = self.limit_order_client.get_limit_order_by_uuid(hotkey, bracket_uuid)

            if existing_bracket and self._is_bracket_cancel_intent(bracket):
                self.process_limit_cancel(hotkey, None, bracket_uuid, now_ms)
                cancelled_count += 1
                continue

            bracket_signal = self._build_bracket_signal(trade_pair, bracket)
            order = self.process_unfilled_order(
                hotkey, bracket_signal, bracket_uuid, now_ms,
                ExecutionType.BRACKET, is_edit=bool(existing_bracket),
            )
            processed_orders.append(order)

        bt.logging.info(f"[ORDER_PROCESSOR] Processed bulk bracket update for {hotkey}: {len(processed_orders)} orders")
        return None

    @staticmethod
    def _is_bracket_cancel_intent(bracket: dict) -> bool:
        # An existing bracket with no SL/TP/trailing in the update payload
        # is interpreted as a cancel request.
        return (
            bracket.get("stop_loss") is None
            and bracket.get("take_profit") is None
            and bracket.get("trailing_percent") is None
            and bracket.get("trailing_value") is None
        )

    @staticmethod
    def _build_bracket_signal(trade_pair: TradePair, bracket: dict) -> Signal:
        trailing_stop = None
        if bracket.get("trailing_percent") is not None:
            trailing_stop = {"trailing_percent": bracket["trailing_percent"]}
        elif bracket.get("trailing_value") is not None:
            trailing_stop = {"trailing_value": bracket["trailing_value"]}

        return Signal(
            execution_type=ExecutionType.BRACKET,
            trade_pair=trade_pair,
            stop_loss=bracket.get("stop_loss"),
            take_profit=bracket.get("take_profit"),
            trailing_stop=trailing_stop,
            leverage=bracket.get("leverage"),
            value=bracket.get("value"),
            quantity=bracket.get("quantity"),
            bracket_pct=bracket.get("bracket_pct"),
        )

