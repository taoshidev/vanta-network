# developer: Taoshi
# Copyright (c) 2024 Taoshi Inc
"""
MinerAccountServer - RPC server for miner account management.

This server runs in its own process and exposes miner account management via RPC.
Clients connect using MinerAccountClient.

Usage:
    # Validator spawns the server at startup
    from vali_objects.miner_account.miner_account_server import start_miner_account_server
    process = Process(target=start_miner_account_server, args=(...))
    process.start()

    # Other processes connect via MinerAccountClient
    from vali_objects.miner_account.miner_account_client import MinerAccountClient
    client = MinerAccountClient()
"""
import bittensor as bt
from typing import Optional, Dict, List, Any
import time
from setproctitle import setproctitle
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
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        collateral_balance_getter=None
    ):
        """
        Initialize MinerAccountServer.

        Args:
            running_unit_tests: Whether running in test mode
            start_server: Whether to start RPC server immediately
            connection_mode: RPC or LOCAL mode
            collateral_balance_getter: Callable to get collateral balance for a hotkey
        """
        # Create the manager FIRST, before RPCServerBase.__init__
        self._manager = MinerAccountManager(
            running_unit_tests=running_unit_tests,
            collateral_balance_getter=collateral_balance_getter
        )

        # Store is_mothership status (set by contract manager later)
        self._is_mothership = False

        # Initialize RPCServerBase
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_MINERACCOUNT_SERVICE_NAME,
            port=ValiConfig.RPC_MINERACCOUNT_PORT,
            connection_mode=connection_mode,
            slack_notifier=None,
            start_server=start_server,
            start_daemon=False,  # MinerAccount server doesn't need a daemon loop
        )

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """MinerAccount server doesn't need a daemon loop."""
        pass

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

    def miner_account_sizes_dict(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Convert miner account sizes to checkpoint format for backup/sync."""
        return self._manager.miner_account_sizes_dict(most_recent_only)

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
        return {
            'miner_hotkey': account.miner_hotkey,
            'account_size': account.account_size,
            'cash_balance': account.cash_balance,
            'total_borrowed_amount': account.total_borrowed_amount,
        }

    def get_account(self, hotkey: str) -> Optional[dict]:
        """Get account if it exists, without creating. Returns dict representation."""
        account = self._manager.get_account(hotkey)
        if account is None:
            return None
        return {
            'miner_hotkey': account.miner_hotkey,
            'account_size': account.account_size,
            'cash_balance': account.cash_balance,
            'total_borrowed_amount': account.total_borrowed_amount,
        }

    def get_all_hotkeys(self) -> list:
        """Get all hotkeys with accounts."""
        return self._manager.get_all_hotkeys()

    def update_account_size(self, hotkey: str, new_size: float) -> bool:
        """Update account size directly (triggers cash_balance adjustment)."""
        if hotkey not in self._manager.accounts:
            return False
        self._manager.accounts[hotkey].update_account_size(new_size)
        return True

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
                           borrowed_for_position: float, trade_pair_category: TradePairCategory) -> dict:
        """Process sell/close order."""
        return self._manager.process_order_sell(hotkey, sale_proceeds_usd, borrowed_for_position, trade_pair_category)

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        return self._manager.get_total_borrowed_amount(hotkey)


# ==================== Server Entry Point ====================

def start_miner_account_server(
    running_unit_tests=False,
    server_ready=None,
):
    """
    Entry point for server process.

    Args:
        running_unit_tests: Whether running in test mode
        server_ready: Event to signal when server is ready
    """
    from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
    setproctitle("vali_MinerAccountServerProcess")

    server_instance = MinerAccountServer(
        running_unit_tests=running_unit_tests,
        start_server=True,
    )

    bt.logging.success(f"MinerAccountServer ready on port {ValiConfig.RPC_MINERACCOUNT_PORT}")

    if server_ready:
        server_ready.set()

    # Block until shutdown
    while not ShutdownCoordinator.is_shutdown():
        time.sleep(1)

    server_instance.shutdown()
    bt.logging.info("MinerAccountServer process exiting")
