from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from vali_objects.trade_pair import TradePair
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.utils.limit_order.order_utils import OrderSize


class MarketOrderClient(RPCClientBase):

    def __init__(self, port: int = None, connect_immediately: bool = False, running_unit_tests=False,
                 connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_MARKET_ORDER_SERVICE_NAME,
            port=port or ValiConfig.RPC_MARKET_ORDER_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode,
        )

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
    ):
        return self._server.execute_order_rpc(
            hotkey=hotkey,
            order_uuid=order_uuid,
            trade_pair=trade_pair,
            execution_type=execution_type,
            order_type=order_type,
            order_size=order_size,
            fill_price=fill_price,
            price_sources=price_sources,
            slippage=slippage,
            order_src=order_src,
            is_hl=is_hl,
            is_hl_taker=is_hl_taker,
            now_ms=now_ms,
            enforce_cooldown=enforce_cooldown,
        )

    def close_positions(self, hotkey: str, position_uuids: list[str] | None = None, close_all: bool = False, now_ms: int | None = None):
        return self._server.close_positions_rpc(
            hotkey=hotkey,
            position_uuids=position_uuids,
            close_all=close_all,
            now_ms=now_ms,
        )

    def clear_order_cooldown_cache(self):
        return self._server.clear_order_cooldown_cache_rpc()
