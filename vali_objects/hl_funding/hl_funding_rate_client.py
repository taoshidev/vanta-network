"""
HLFundingRateClient - Lightweight RPC client for HL funding rate data.
"""
from typing import Dict, List, Optional

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class HLFundingRateClient(RPCClientBase):
    """RPC client for HLFundingRateServer."""

    def __init__(
        self,
        port: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        connect_immediately: bool = True,
    ):
        super().__init__(
            service_name=ValiConfig.RPC_HL_FUNDING_SERVICE_NAME,
            port=port or ValiConfig.RPC_HL_FUNDING_PORT,
            connection_mode=connection_mode,
            connect_immediately=connect_immediately,
        )

    def get_rates_for_position(self, coin: str, open_ms: int, current_ms: int) -> Dict[int, float]:
        """Get funding rates between open_ms and current_ms for a coin."""
        return self._server.get_rates_for_position_rpc(coin, open_ms, current_ms)

    def get_rate_at_time(self, coin: str, time_ms: int) -> Optional[float]:
        """Get the funding rate at (or just before) a specific time."""
        return self._server.get_rate_at_time_rpc(coin, time_ms)

    def fetch_and_store_rates(self, coins: List[str], start_ms: int, end_ms: int):
        """Fetch and store rates from HL API."""
        return self._server.fetch_and_store_rates_rpc(coins, start_ms, end_ms)
