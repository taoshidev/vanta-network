"""
HLFundingRateServer - RPC server for HL funding rate management.

Follows the EntityServer pattern. Runs a daemon that periodically fetches
recent funding rates from the HL API.
"""
import bittensor as bt
from typing import Dict, List, Optional

from time_util.time_util import TimeUtil
from vali_objects.hl_funding.hl_funding_rate_manager import HLFundingRateManager
from vali_objects.vali_config import ValiConfig, RPCConnectionMode, HL_DYNAMIC_REGISTRY, load_hl_dynamic_registry
from shared_objects.rpc.rpc_server_base import RPCServerBase


class HLFundingRateServer(RPCServerBase):
    """
    RPC server for HL funding rate data.
    Exposes HLFundingRateManager methods via RPC.
    """
    service_name = ValiConfig.RPC_HL_FUNDING_SERVICE_NAME
    service_port = ValiConfig.RPC_HL_FUNDING_PORT

    def __init__(
        self,
        *,
        slack_notifier=None,
        start_server=True,
        start_daemon=False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
    ):
        self.running_unit_tests = running_unit_tests

        # Create manager FIRST before RPCServerBase.__init__
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        self._manager = HLFundingRateManager(running_unit_tests=running_unit_tests)

        daemon_interval_s = ValiConfig.HL_FUNDING_DAEMON_INTERVAL_S
        hang_timeout_s = daemon_interval_s * 2.0

        super().__init__(
            service_name=self.service_name,
            port=self.service_port,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            connection_mode=connection_mode,
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s,
        )

        # Backfill on startup
        if not running_unit_tests:
            self._backfill()

    def _get_coins(self) -> list:
        """Return all known HL coins, refreshing registry from disk first."""
        load_hl_dynamic_registry()
        coins = [dtp.hl_coin for dtp in list(HL_DYNAMIC_REGISTRY.values())]
        if not coins:
            # Fallback: static 6 coins before any tracker run has written the registry.
            return list(ValiConfig.TRADE_PAIR_ID_TO_HL_COIN.values())
        return coins

    def _backfill(self):
        """Backfill the last N hours of funding rates on startup."""
        now_ms = TimeUtil.now_in_millis()
        start_ms = now_ms - ValiConfig.HL_FUNDING_BACKFILL_HOURS * 3600 * 1000
        coins = self._get_coins()
        if not coins:
            bt.logging.warning("[HL_FUNDING_SERVER] No coins available, skipping backfill.")
            return
        bt.logging.info(f"[HL_FUNDING_SERVER] Backfilling {ValiConfig.HL_FUNDING_BACKFILL_HOURS}h of rates for {len(coins)} coins")
        self._manager.fetch_and_store_rates(coins, start_ms, now_ms)

    def run_daemon_iteration(self):
        """Fetch last 2 hours of rates for all HL coins."""
        now_ms = TimeUtil.now_in_millis()
        start_ms = now_ms - 2 * 3600 * 1000  # 2 hours back
        coins = self._get_coins()
        if not coins:
            return
        self._manager.fetch_and_store_rates(coins, start_ms, now_ms)

    # === RPC methods ===

    def get_rates_for_position_rpc(self, coin: str, open_ms: int, current_ms: int) -> Dict[int, float]:
        return self._manager.get_rates_for_position(coin, open_ms, current_ms)

    def get_rate_at_time_rpc(self, coin: str, time_ms: int) -> Optional[float]:
        return self._manager.get_rate_at_time(coin, time_ms)

    def fetch_and_store_rates_rpc(self, coins: List[str], start_ms: int, end_ms: int):
        return self._manager.fetch_and_store_rates(coins, start_ms, end_ms)
