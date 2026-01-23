# developer: Taoshi
# Copyright (c) 2024 Taoshi Inc
"""
MinerAccountServer - RPC server for miner account management.

This server runs in its own process and exposes miner account management via RPC.
Clients connect using MinerAccountClient.

Usage:
    # Validator spawns the server via ServerOrchestrator
    from shared_objects.rpc.server_orchestrator import ServerOrchestrator
    orchestrator = ServerOrchestrator.get_instance()
    orchestrator.start_all_servers(mode=ServerMode.VALIDATOR, context=context)

    # Other processes connect via MinerAccountClient
    from vali_objects.miner_account.miner_account_client import MinerAccountClient
    client = MinerAccountClient()
"""
import bittensor as bt
from typing import Optional, Dict, List, Any
from vali_objects.vali_config import ValiConfig, RPCConnectionMode, TradePairCategory
from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.miner_account.miner_account_manager import MinerAccountManager, MinerAccount


class MinerAccountServer(RPCServerBase):
    """
    RPC Server for miner account management.

    Inherits from RPCServerBase for RPC server lifecycle management.
    """
    service_name = ValiConfig.RPC_MINERACCOUNT_SERVICE_NAME
    service_port = ValiConfig.RPC_MINERACCOUNT_PORT

    def __init__(
        self,
        running_unit_tests=False,
        start_server=True,
        start_daemon=True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        collateral_balance_getter=None
    ):
        """
        Initialize MinerAccountServer.

        Args:
            running_unit_tests: Whether running in test mode
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            connection_mode: RPC or LOCAL mode
            collateral_balance_getter: Callable to get collateral balance for a hotkey
        """
        # Create the manager FIRST, before RPCServerBase.__init__
        self._manager = MinerAccountManager(
            running_unit_tests=running_unit_tests,
            collateral_balance_getter=collateral_balance_getter,
            connection_mode=connection_mode
        )

        # Store is_mothership status (set by contract manager later)
        self._is_mothership = False

        # Daemon configuration
        daemon_interval_s = 3600
        hang_timeout_s = daemon_interval_s * 2

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_MINERACCOUNT_SERVICE_NAME,
            port=ValiConfig.RPC_MINERACCOUNT_PORT,
            connection_mode=connection_mode,
            slack_notifier=None,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s,
        )

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Daemon loop that runs every hour.
        Calls apply_daily_interest() which handles per-account 24-hour interval checks.
        """
        try:
            # Apply interest to accounts that need it (24-hour check is handled in the method)
            result = self._manager.apply_daily_interest()

            if result > 0:
                bt.logging.success(
                    f"Interest application completed: {result} accounts processed"
                )
            else:
                bt.logging.info("No interest application needed (no accounts ready for interest)")

        except Exception as e:
            bt.logging.error(f"Error in interest calculation daemon: {e}")

    # ==================== Setup Methods ====================

    def set_collateral_balance_getter(self, getter):
        """Set the collateral balance getter."""
        self._manager.set_collateral_balance_getter(getter)

    def set_is_mothership(self, is_mothership: bool):
        """Set whether this validator is the mothership."""
        self._is_mothership = is_mothership

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "account_count": len(self._manager.accounts),
        }

    # ==================== Account Size Methods ====================

    def set_miner_account_size(
        self,
        hotkey: str,
        collateral_balance_theta: float,
        timestamp_ms: Optional[int] = None
    ) -> Optional[dict]:
        """Set the account size for a miner. Returns CollateralRecord as dict if successful."""
        collateral_record = self._manager.set_miner_account_size(hotkey, collateral_balance_theta, timestamp_ms)
        if collateral_record is None:
            return None
        return vars(collateral_record)

    def get_miner_account_size(
        self,
        hotkey: str,
        timestamp_ms: Optional[int] = None,
        most_recent: bool = False,
        use_account_floor: bool = False
    ) -> Optional[float]:
        """Get the account size for a miner at a given timestamp."""
        return self._manager.get_miner_account_size(
            hotkey, timestamp_ms, most_recent, use_account_floor=use_account_floor
        )

    def get_all_miner_account_sizes(self, timestamp_ms: Optional[int] = None) -> Dict[str, float]:
        """Return a dict of all miner account sizes at a timestamp_ms."""
        return self._manager.get_all_miner_account_sizes(timestamp_ms=timestamp_ms)

    def accounts_dict(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Convert miner account sizes to checkpoint format for backup/sync."""
        return self._manager.accounts_dict(most_recent_only)

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Sync miner account sizes data from external source (backup/sync)."""
        self._manager.sync_miner_account_sizes_data(account_sizes_data)

    def re_init_account_sizes(self) -> None:
        """Reload account sizes from disk."""
        self._manager.re_init_account_sizes()

    def receive_collateral_record_update(self, collateral_record_data: dict) -> bool:
        """Process an incoming CollateralRecord synapse."""
        return self._manager.receive_collateral_record_update(collateral_record_data, self._is_mothership)

    # ==================== MinerAccount Cache Methods ====================

    def get_or_create(self, hotkey: str) -> dict:
        """Get existing account or create from CollateralRecord. Returns dict representation."""
        account = self._manager.get_or_create(hotkey)
        return account.to_dict()

    def get_account(self, hotkey: str) -> Optional[dict]:
        """Get account if it exists, without creating. Returns dict representation."""
        account = self._manager.get_account(hotkey)
        if account is None:
            return None
        return account.to_dict()

    def get_all_hotkeys(self) -> list:
        """Get all hotkeys with accounts."""
        return self._manager.get_all_hotkeys()

    def get_cash_balance(self, hotkey: str) -> Optional[float]:
        """Get cash balance for a miner."""
        account = self._manager.get_account(hotkey)
        if account is None:
            return None
        return account.cash_balance

    def set_cash_balance(self, hotkey: str, cash_balance: float) -> bool:
        """Set cash balance for a miner."""
        account = self._manager.get_account(hotkey)
        if account is None:
            return False
        account.cash_balance = cash_balance
        return True

    def health_check(self) -> dict:
        """Health check for monitoring."""
        return self._manager.health_check()

    # ==================== Margin/Cash Processing Methods ====================

    def process_order_buy(self, hotkey: str, order_value_usd: float,
                          trade_pair_category: TradePairCategory) -> float:
        """Process buy order cash/margin. Returns borrowed amount."""
        return self._manager.process_order_buy(hotkey, order_value_usd, trade_pair_category)

    def process_order_sell(self, hotkey: str, sale_proceeds_usd: float,
                           position_margin_loan: float, trade_pair_category: TradePairCategory) -> float:
        """Process sell/close order."""
        return self._manager.process_order_sell(hotkey, sale_proceeds_usd, position_margin_loan, trade_pair_category)

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        return self._manager.get_total_borrowed_amount(hotkey)

    def can_withdraw_collateral(self, hotkey: str, amount_theta: float) -> bool:
        """Check if miner can withdraw the specified amount of collateral."""
        return self._manager.can_withdraw_collateral(hotkey, amount_theta)

    def recalculate_cash_balance_for_asset_selection(
        self, hotkey: str, asset_selection: TradePairCategory
    ) -> bool:
        """
        Recalculate cash balance when a miner selects an asset class.

        Args:
            hotkey: Miner's hotkey
            asset_selection: The TradePairCategory the miner selected

        Returns:
            True if cash balance was updated, False otherwise
        """
        return self._manager.recalculate_cash_balance_for_asset_selection(hotkey, asset_selection)
