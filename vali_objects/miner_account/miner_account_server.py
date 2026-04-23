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

import template.protocol
from vali_objects.vali_config import ValiConfig, RPCConnectionMode, TradePairCategory
from vali_objects.enums.miner_bucket_enum import MinerBucket
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
        config=None,
        running_unit_tests=False,
        is_backtesting=False,
        slack_notifier=None,
        start_server=True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        collateral_balance_getter=None
    ):
        """
        Initialize MinerAccountServer.

        Args:
            config: Bittensor config (for ValidatorBroadcastBase)
            running_unit_tests: Whether running in test mode
            is_backtesting: Whether backtesting
            slack_notifier: Slack notifier for health check alerts
            start_server: Whether to start RPC server immediately
            connection_mode: RPC or LOCAL mode
            collateral_balance_getter: Callable to get collateral balance for a hotkey
        """
        # Create mock config if running tests and config not provided
        if running_unit_tests:
            from shared_objects.rpc.test_mock_factory import TestMockFactory
            config = TestMockFactory.create_mock_config_if_needed(config, netuid=116, network="test")

        # Derive is_testnet from config
        is_testnet = config.subtensor.network == "test" if config else False

        # Create the manager FIRST, before RPCServerBase.__init__
        self._manager = MinerAccountManager(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode,
            config=config,
            is_testnet=is_testnet
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
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # Daemon started later via orchestrator
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s,
        )

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
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
        timestamp_ms: Optional[int] = None,
        account_size: float = None,
        bucket: Optional[MinerBucket] = None,
    ) -> Optional[dict]:
        """Set the account size for a miner. Returns CollateralRecord as dict if successful."""
        collateral_record = self._manager.set_miner_account_size(hotkey, collateral_balance_theta, timestamp_ms, account_size)
        if collateral_record is None:
            return None
        return vars(collateral_record)

    def delete_miner_account_size(self, hotkey: str) -> bool:
        """Delete the account size for a miner. Returns True if successful."""
        return self._manager.delete_miner_account_size(hotkey)

    def reset_account_fields(self, hotkey: str) -> bool:
        """Reset account fields (PnL, capital used, borrowed amount, interest) for a miner."""
        return self._manager.reset_account_fields(hotkey)

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

    def receive_collateral_record_rpc(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        """
        Receive collateral record update synapse (for axon attachment).

        This method is called when a CollateralRecord broadcast is received from another validator.

        Args:
            synapse: CollateralRecord synapse from the sending validator

        Returns:
            Updated synapse with successfully_processed and error_message fields
        """
        try:
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(f"Received collateral record update from validator hotkey [{sender_hotkey}].")

            # Extract collateral record data from synapse
            collateral_record_data = synapse.collateral_record

            # Process the update through the manager
            success = self._manager.receive_collateral_record_update(collateral_record_data, sender_hotkey)

            if success:
                synapse.successfully_processed = True
                synapse.error_message = ""
                bt.logging.info(f"Successfully processed CollateralRecord synapse from {sender_hotkey}")
            else:
                synapse.successfully_processed = False
                synapse.error_message = "Failed to process collateral record update"
                bt.logging.warning(f"Failed to process CollateralRecord synapse from {sender_hotkey}")

        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Error processing collateral record: {str(e)}"
            bt.logging.error(f"Exception in receive_collateral_record: {e}")

        return synapse

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

    def get_accounts(self, hotkeys: list) -> Dict[str, dict]:
        """Get accounts for multiple hotkeys. Returns dict of hotkey -> account dict."""
        accounts = self._manager.get_accounts(hotkeys)
        return {hk: account.to_dict() for hk, account in accounts.items()}

    def get_dashboard_rpc(self, hotkey: str) -> dict | None:
        return self._manager.get_dashboard(hotkey)

    def update_max_returns(self, hotkey_to_return: dict) -> None:
        """Batch update HWM for multiple hotkeys. Saves to disk once."""
        self._manager.update_max_returns(hotkey_to_return)

    def set_miner_bucket(self, hotkey: str, bucket_value: Optional[str]) -> None:
        """Set the miner bucket on an account. Converts string to MinerBucket enum."""
        bucket = MinerBucket(bucket_value) if bucket_value else None
        self._manager.set_miner_bucket(hotkey, bucket)

    def get_hl_address(self, hotkey: str) -> Optional[str]:
        """Return the HL address for an account, or None if not an HS subaccount."""
        return self._manager.get_hl_address(hotkey)

    def set_hl_address(self, hotkey: str, hl_address: Optional[str]) -> None:
        """Set the HL address on an account."""
        self._manager.set_hl_address(hotkey, hl_address)

    def get_all_hotkeys(self) -> list:
        """Get all hotkeys with accounts."""
        return self._manager.get_all_hotkeys()

    def get_buying_power(self, hotkey: str) -> Optional[float]:
        """Get buying power for a miner."""
        account = self._manager.get_account(hotkey)
        if account is None:
            return None
        return account.buying_power

    def get_balance(self, hotkey: str) -> Optional[float]:
        """Get balance for a miner."""
        account = self._manager.get_account(hotkey)
        if account is None:
            return None
        return account.balance

    def health_check(self) -> dict:
        """Health check for monitoring."""
        return self._manager.health_check()

    # ==================== Margin/Cash Processing Methods ====================

    def process_order_buy(self, hotkey: str, order_value_usd: float, borrowed_amount: float, fee_usd: float) -> None:
        """Process buy order cash/margin."""
        self._manager.process_order_buy(hotkey, order_value_usd, borrowed_amount, fee_usd)

    def process_order_sell(self, hotkey: str, entry_value_usd: float, realized_pnl: float, loan_repaid: float, fee_usd: float) -> None:
        """Process sell/close order."""
        self._manager.process_order_sell(hotkey, entry_value_usd, realized_pnl, loan_repaid, fee_usd)

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        return self._manager.get_total_borrowed_amount(hotkey)

    def can_withdraw_collateral(self, hotkey: str, amount_theta: float) -> bool:
        """Check if miner can withdraw the specified amount of collateral."""
        return self._manager.can_withdraw_collateral(hotkey, amount_theta)

    def rebuild_account_state_from_positions(
        self,
        hotkey: str,
        positions: list,
        miner_bucket: Optional[str] = None,
        max_return: float = 1.0,
    ) -> None:
        """Rebuild a miner's account state from a list of Position dicts."""
        from vali_objects.vali_dataclasses.position import Position
        position_objects = [Position(**p) if isinstance(p, dict) else p for p in positions]
        bucket = MinerBucket(miner_bucket) if miner_bucket else None
        self._manager.rebuild_account_state_from_positions(hotkey, position_objects, bucket, max_return)

    def update_asset_selection(
        self, hotkey: str, asset_selection: TradePairCategory
    ) -> bool:
        """
        Returns:
            True if cash balance was updated, False otherwise
        """
        return self._manager.update_asset_selection(hotkey, asset_selection)

    def process_fees(self, hotkey_to_fee: dict) -> None:
        """Batch update total_fees_paid for multiple hotkeys. Saves to disk once."""
        self._manager.process_fees(hotkey_to_fee)

    def process_dividend_income(self, hotkey_to_credit: dict) -> None:
        """Batch update total_dividend_income for multiple hotkeys. Saves to disk once."""
        self._manager.process_dividend_income(hotkey_to_credit)
