"""
MarketOrderManagerServer - RPC server wrapping MarketOrderManager.

Exposes _process_market_order and process_flat_all_order via RPC so that
OrderProcessor can call them from any process without receiving a full
MarketOrderManager instance.
"""
import bittensor as bt

from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class MarketOrderManagerServer(RPCServerBase):
    """
    RPC server that owns a MarketOrderManager and exposes its order-filling
    methods to clients in other processes.

    No daemon loop is needed: the SlippageRefresher thread inside
    MarketOrderManager runs automatically on construction.
    """

    service_name = ValiConfig.RPC_MARKETORDERMANAGER_SERVICE_NAME
    service_port = ValiConfig.RPC_MARKETORDERMANAGER_PORT

    def __init__(
        self,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = False,
        running_unit_tests: bool = False,
        serve: bool = True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        is_backtesting: bool = False,
    ):
        from vali_objects.utils.limit_order.market_order_manager import MarketOrderManager

        # Create the manager BEFORE calling RPCServerBase.__init__ so that RPC
        # calls are never dispatched to a partially-initialised object.
        self._manager = MarketOrderManager(
            serve=serve,
            slack_notifier=slack_notifier,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode,
        )

        super().__init__(
            service_name=ValiConfig.RPC_MARKETORDERMANAGER_SERVICE_NAME,
            port=ValiConfig.RPC_MARKETORDERMANAGER_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # no periodic daemon; SlippageRefresher runs as its own thread
            connection_mode=connection_mode,
        )

    # ------------------------------------------------------------------ #
    #  RPCServerBase abstract method                                       #
    # ------------------------------------------------------------------ #

    def run_daemon_iteration(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  RPC methods                                                         #
    # ------------------------------------------------------------------ #

    def process_market_order_rpc(
        self,
        miner_order_uuid,
        miner_repo_version,
        trade_pair,
        now_ms,
        signal,
        miner_hotkey,
        price_sources=None,
        enforce_market_cooldown=True,
    ):
        """
        Delegate to MarketOrderManager._process_market_order.

        Returns:
            (err_msg, existing_position, created_order)  – pickled by RPC layer.
        """
        return self._manager._process_market_order(
            miner_order_uuid,
            miner_repo_version,
            trade_pair,
            now_ms,
            signal,
            miner_hotkey,
            price_sources=price_sources,
            enforce_market_cooldown=enforce_market_cooldown,
        )

    def process_flat_all_order_rpc(self, order_uuid, miner_repo_version, miner_hotkey, now_ms):
        """Delegate to MarketOrderManager.process_flat_all_order."""
        return self._manager.process_flat_all_order(order_uuid, miner_repo_version, miner_hotkey, now_ms)

    def clear_order_cooldown_cache_rpc(self) -> None:
        """Clear order cooldown cache (test isolation)."""
        self._manager.clear_order_cooldown_cache()

    def get_health_check_details(self) -> dict:
        return {"manager_initialised": self._manager is not None}
