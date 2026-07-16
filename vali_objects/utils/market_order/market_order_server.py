from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class MarketOrderServer(RPCServerBase):

    service_name = ValiConfig.RPC_MARKET_ORDER_SERVICE_NAME
    service_port = ValiConfig.RPC_MARKET_ORDER_PORT

    def __init__(
        self,
        running_unit_tests=False,
        start_server=True,
        start_daemon=False,
        serve=True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        **kwargs,
    ):
        from vali_objects.utils.market_order.market_order_manager import MarketOrderManager
        self._manager = MarketOrderManager(
            serve=serve,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode,
        )

        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_MARKET_ORDER_SERVICE_NAME,
            port=ValiConfig.RPC_MARKET_ORDER_PORT,
            connection_mode=connection_mode,
            start_server=start_server,
            start_daemon=False,
            daemon_interval_s=60,
            hang_timeout_s=120.0,
        )

    def run_daemon_iteration(self):
        return None

    # ==================== RPC Methods ====================

    def execute_order_rpc(
        self,
        hotkey,
        order_uuid,
        trade_pair,
        execution_type,
        order_type,
        order_size,
        fill_price=None,
        price_sources=None,
        slippage=None,
        order_src=None,
        is_hl=False,
        is_hl_taker=None,
        now_ms=None,
        enforce_cooldown=True,
    ):
        from vali_objects.enums.order_source_enum import OrderSource
        if order_src is None:
            order_src = OrderSource.ORGANIC
        return self._manager.execute_order(
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

    def close_positions_rpc(self, hotkey, position_uuids=None, close_all=False, now_ms=None):
        return self._manager.close_positions(
            hotkey=hotkey,
            position_uuids=position_uuids,
            close_all=close_all,
            now_ms=now_ms,
        )

    def clear_order_cooldown_cache_rpc(self):
        return self._manager.clear_order_cooldown_cache()
