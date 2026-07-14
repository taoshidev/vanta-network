"""
MarketOrderManagerClient - RPC client for MarketOrderManagerServer.

Provides the same interface that callers previously used on a local
MarketOrderManager instance, but routes calls over RPC.
"""
import bittensor as bt

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class MarketOrderManagerClient(RPCClientBase):
    """
    Lightweight RPC client for MarketOrderManagerServer.

    Exposes _process_market_order and process_flat_all_order with the same
    signature as MarketOrderManager so callers are duck-type compatible.
    """

    def __init__(
        self,
        port: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False,
        connect_immediately: bool = False,
    ):
        super().__init__(
            service_name=ValiConfig.RPC_MARKETORDERMANAGER_SERVICE_NAME,
            port=port or ValiConfig.RPC_MARKETORDERMANAGER_PORT,
            connection_mode=connection_mode,
            connect_immediately=connect_immediately,
        )
        self.running_unit_tests = running_unit_tests

    # ------------------------------------------------------------------ #
    #  Public interface (mirrors MarketOrderManager)                       #
    # ------------------------------------------------------------------ #

    def _process_market_order(
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
        """Route to MarketOrderManagerServer.process_market_order_rpc."""
        return self._server.process_market_order_rpc(
            miner_order_uuid,
            miner_repo_version,
            trade_pair,
            now_ms,
            signal,
            miner_hotkey,
            price_sources,
            enforce_market_cooldown,
        )

    def process_flat_all_order(self, order_uuid, miner_repo_version, miner_hotkey, now_ms):
        """Route to MarketOrderManagerServer.process_flat_all_order_rpc."""
        return self._server.process_flat_all_order_rpc(order_uuid, miner_repo_version, miner_hotkey, now_ms)

    def clear_order_cooldown_cache(self) -> None:
        """Clear order cooldown cache on the server (test isolation)."""
        self._server.clear_order_cooldown_cache_rpc()
