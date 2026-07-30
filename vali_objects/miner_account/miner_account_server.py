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
from datetime import datetime, timezone, timedelta

from typing import Optional, Dict, List, Any

import template.protocol
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.miner_account.miner_account_manager import MinerAccountManager, MinerAccount
from shared_objects.log import logger


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
        self._first_snapshot_iteration = True

        # Daemon configuration: align first run to the top of the next UTC hour
        daemon_interval_s = 3600   # 1 hour
        hang_timeout_s = 3600 * 2  # 2 hours
        # daemon_stagger_s = MinerAccountServer._seconds_until_next_utc_hour()

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
            # daemon_stagger_s=daemon_stagger_s,
        )

    # ==================== RPCServerBase Abstract Methods ====================

    @staticmethod
    def _seconds_until_next_utc_hour() -> float:
        now = datetime.now(tz=timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return (next_hour - now).total_seconds()

    def run_daemon_iteration(self) -> str | None:
        if self._first_snapshot_iteration:
            self._first_snapshot_iteration = False
            self.daemon_interval_s = MinerAccountServer._seconds_until_next_utc_hour()
            bt.logging.info(f"Next MinerAccount daemon in {self.daemon_interval_s:.0f}s")
            return None
        now = datetime.now(tz=timezone.utc)
        update_daily_open = (now.hour == 0)
        count = self._manager.take_account_snapshot(update_daily_open=update_daily_open)
        self.daemon_interval_s = MinerAccountServer._seconds_until_next_utc_hour()
        return f"MinerAccount daemon iteration complete. Snapshots taken: {count}."

    # ==================== Setup Methods ====================

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

    def reset_account(self, hotkey: str, miner_bucket: MinerBucket | None = None) -> bool:
        """Reset account fields (PnL, capital used, borrowed amount, interest) for a miner."""
        return self._manager.reset_account(hotkey, miner_bucket)

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
            logger.info(f"Received collateral record update from validator hotkey [{sender_hotkey}].")

            # Extract collateral record data from synapse
            collateral_record_data = synapse.collateral_record

            # Process the update through the manager
            success = self._manager.receive_collateral_record_update(collateral_record_data, sender_hotkey)

            if success:
                synapse.successfully_processed = True
                synapse.error_message = ""
                logger.info(f"Successfully processed CollateralRecord synapse from {sender_hotkey}")
            else:
                synapse.successfully_processed = False
                synapse.error_message = "Failed to process collateral record update"
                logger.warning(f"Failed to process CollateralRecord synapse from {sender_hotkey}")

        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Error processing collateral record: {str(e)}"
            logger.error(f"Exception in receive_collateral_record: {e}")

        return synapse

    # ==================== MinerAccount Cache Methods ====================

    def get_or_create(self, hotkey: str) -> MinerAccount:
        """Get existing account or create from CollateralRecord."""
        return self._manager.get_or_create(hotkey)

    def get_account(self, hotkey: str) -> Optional[MinerAccount]:
        """Get account if it exists, without creating. Returns None if not found."""
        return self._manager.get_account(hotkey)

    def get_accounts(self, hotkeys: list) -> Dict[str, MinerAccount]:
        """Get accounts for multiple hotkeys. Returns dict of hotkey -> MinerAccount."""
        return self._manager.get_accounts(hotkeys)

    def get_daily_open_snapshot(self, hotkey: str) -> Optional[dict]:
        """Return the most recent daily open snapshot for a miner, or None if not yet recorded."""
        account = self._manager.get_account(hotkey)
        if account is None or account.daily_open_snapshot is None:
            return None
        return account.daily_open_snapshot.to_dict()

    def get_dashboard_rpc(self, hotkey: str) -> dict | None:
        return self._manager.get_dashboard(hotkey)

    def update_unrealized_pnl(self, hotkey_to_unrealized_pnl: dict) -> None:
        """Batch update unrealized PNL for multiple hotkeys."""
        self._manager.update_unrealized_pnl(hotkey_to_unrealized_pnl)

    def set_miner_bucket(self, hotkey: str, bucket_value: Optional[str]) -> None:
        """Set the miner bucket on an account. Converts string to MinerBucket enum."""
        bucket = MinerBucket(bucket_value) if bucket_value else None
        self._manager.set_miner_bucket(hotkey, bucket)

    def set_miner_buckets(self, hotkey_to_bucket: Dict[str, MinerBucket]) -> None:
        """Bulk set miner buckets across multiple accounts."""
        self._manager.set_miner_buckets(hotkey_to_bucket)

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

    def process_order_buy(self, hotkey: str, order_value_usd: float, borrowed_amount: float, fee_usd: float, trade_pair_category: Optional[TradePairCategory] = None) -> None:
        """Process buy order cash/margin."""
        self._manager.process_order_buy(hotkey, order_value_usd, borrowed_amount, fee_usd, trade_pair_category)

    def process_order_sell(self, hotkey: str, entry_value_usd: float, realized_pnl: float, loan_repaid: float, fee_usd: float, trade_pair_category: Optional[TradePairCategory] = None, unrealized_pnl_released: float = 0.0) -> None:
        """Process sell/close order."""
        self._manager.process_order_sell(hotkey, entry_value_usd, realized_pnl, loan_repaid, fee_usd, trade_pair_category, unrealized_pnl_released)

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        return self._manager.get_total_borrowed_amount(hotkey)

    def rebuild_account_state_from_positions(self, hotkey: str, positions: list) -> None:
        """Rebuild a miner's account state from a list of Position dicts."""
        from vali_objects.vali_dataclasses.position import Position
        position_objects = [Position(**p) if isinstance(p, dict) else p for p in positions]
        self._manager.rebuild_account_state_from_positions(hotkey, position_objects)

    def update_asset_selection(
        self, hotkey: str, asset_selection: MinerAssetClass
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
