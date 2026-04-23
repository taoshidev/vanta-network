# developer: Taoshi
# Copyright (c) 2024 Taoshi Inc
"""
MinerAccountClient - RPC client for MinerAccountServer.

Lightweight client that connects to MinerAccountServer via RPC.
Can be created in ANY process. No server ownership.

Usage:
    from vali_objects.miner_account.miner_account_client import MinerAccountClient

    # In RPC mode (normal usage)
    client = MinerAccountClient()
    account_size = client.get_miner_account_size(hotkey)

    # In LOCAL mode (for testing)
    client = MinerAccountClient(connection_mode=RPCConnectionMode.LOCAL)
    client.set_direct_server(server_instance)
"""
from typing import Optional, Dict, List, Any

import template.protocol
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_server import MinerAccountServer
from vali_objects.vali_config import RPCConnectionMode, ValiConfig, TradePairCategory


class MinerAccountClient(RPCClientBase):
    """
    Lightweight RPC client for MinerAccountServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_MINERACCOUNT_PORT.

    In test mode (LOCAL connection_mode), use set_direct_server() to provide
    a direct MinerAccountServer instance instead of RPC connection.
    """

    @property
    def _server(self) -> MinerAccountServer:
        """Typed override of base class _server property."""
        return super()._server

    def __init__(
        self,
        port: Optional[int] = None,
        connect_immediately: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False
    ):
        """
        Initialize MinerAccountClient.

        Args:
            port: Port number of the server (default: ValiConfig.RPC_MINERACCOUNT_PORT)
            connect_immediately: If True, connect in __init__. If False, connect lazily.
            connection_mode: RPC or LOCAL mode
            running_unit_tests: If True, running in test mode
        """
        self.running_unit_tests = running_unit_tests

        super().__init__(
            service_name=ValiConfig.RPC_MINERACCOUNT_SERVICE_NAME,
            port=port or ValiConfig.RPC_MINERACCOUNT_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connection_mode=connection_mode,
            connect_immediately=connect_immediately
        )

    # ==================== Account Size Methods ====================

    def set_miner_account_size(
        self,
        hotkey: str,
        collateral_balance_theta: float,
        timestamp_ms: Optional[int] = None,
        account_size: float = None,
        bucket: Optional[MinerBucket] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Set the account size for a miner.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            collateral_balance_theta: Collateral balance in theta tokens
            timestamp_ms: Timestamp for the record (defaults to now)
            account_size: Optional USD account size. If not provided, calculated from collateral balance

        Returns:
            CollateralRecord as dict if successful, None otherwise.
            Dict contains: account_size, account_size_theta, update_time_ms, valid_date_timestamp
        """
        if bucket:
            self._server.set_miner_bucket(hotkey, bucket.value)
        return self._server.set_miner_account_size(hotkey, collateral_balance_theta, timestamp_ms, account_size)

    def delete_miner_account_size(self, hotkey: str) -> bool:
        """
        Delete the account size for a miner.

        This allows rollback when subaccount creation fails.

        Args:
            hotkey: Miner's hotkey (SS58 address)

        Returns:
            bool: True if successful, False otherwise
        """
        return self._server.delete_miner_account_size(hotkey)

    def reset_account_fields(self, hotkey: str) -> bool:
        """
        Reset account fields for a miner.

        Resets: total_realized_pnl, capital_used, total_borrowed_amount,
        and total_fees_paid to zero.

        Args:
            hotkey: Miner's hotkey (SS58 address)

        Returns:
            bool: True if successful, False if account doesn't exist
        """
        return self._server.reset_account_fields(hotkey)

    def get_miner_account_size(
        self,
        hotkey: str,
        timestamp_ms: Optional[int] = None,
        most_recent: bool = False,
        use_account_floor: bool = False
    ) -> Optional[float]:
        """
        Get the account size for a miner at a given timestamp.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp to query for (defaults to now)
            most_recent: If True, return most recent record regardless of timestamp
            use_account_floor: If True, return MIN_CAPITAL instead of None when no records exist

        Returns:
            Account size in USD, or None if no applicable records
        """
        return self._server.get_miner_account_size(
            hotkey, timestamp_ms, most_recent, use_account_floor
        )

    def get_all_miner_account_sizes(self, timestamp_ms: Optional[int] = None) -> Dict[str, float]:
        """Return a dict of all miner account sizes at a timestamp_ms."""
        return self._server.get_all_miner_account_sizes(timestamp_ms)

    def accounts_dict(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Convert miner account sizes to checkpoint format for backup/sync."""
        return self._server.accounts_dict(most_recent_only)

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Sync miner account sizes data from external source (backup/sync)."""
        self._server.sync_miner_account_sizes_data(account_sizes_data)

    def re_init_account_sizes(self) -> None:
        """Reload account sizes from disk."""
        self._server.re_init_account_sizes()

    def receive_collateral_record(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        """
        Receive collateral record update synapse (for axon attachment).

        This method is called directly by the validator's axon when a CollateralRecord
        broadcast is received from another validator.

        Args:
            synapse: CollateralRecord synapse from the sending validator

        Returns:
            Updated synapse with successfully_processed and error_message fields
        """
        return self._server.receive_collateral_record_rpc(synapse)

    # ==================== MinerAccount Cache Methods ====================

    def get_or_create(self, hotkey: str) -> dict:
        """
        Get existing account or create from CollateralRecord.

        Returns dict with:
            - miner_hotkey: str
            - account_size: float
            - total_realized_pnl: float
            - capital_used: float
            - balance: float
            - buying_power: float
            - total_borrowed_amount: float
        """
        return self._server.get_or_create(hotkey)

    def get_account(self, hotkey: str) -> Optional[dict]:
        """
        Get account if it exists, without creating.

        Returns dict with:
            - miner_hotkey: str
            - account_size: float
            - total_realized_pnl: float
            - capital_used: float
            - balance: float
            - buying_power: float
            - total_borrowed_amount: float
        Or None if account doesn't exist.
        """
        return self._server.get_account(hotkey)

    def get_accounts(self, hotkeys: list) -> Dict[str, dict]:
        """
        Get accounts for multiple hotkeys in a single RPC call.

        Args:
            hotkeys: List of miner hotkeys to look up

        Returns:
            Dict of hotkey -> account dict for existing accounts.
            Hotkeys without accounts are omitted from the result.
        """
        return self._server.get_accounts(hotkeys)

    def get_dashboard(self, hotkey: str) -> dict | None:
        return self._server.get_dashboard_rpc(hotkey)

    def update_max_returns(self, hotkey_to_return: Dict[str, float]) -> None:
        """Batch update HWM for multiple hotkeys. Saves to disk once."""
        self._server.update_max_returns(hotkey_to_return)

    def set_miner_bucket(self, hotkey: str, bucket: Optional[MinerBucket]) -> None:
        """Set the miner bucket on an account. Converts MinerBucket to string for RPC."""
        bucket_value = bucket.value if bucket else None
        self._server.set_miner_bucket(hotkey, bucket_value)

    def get_hl_address(self, hotkey: str) -> Optional[str]:
        """Return the HL address for an account, or None if not an HS subaccount."""
        return self._server.get_hl_address(hotkey)

    def set_hl_address(self, hotkey: str, hl_address: Optional[str]) -> None:
        """Set the HL address on an account."""
        self._server.set_hl_address(hotkey, hl_address)

    def get_all_hotkeys(self) -> list:
        """Get all hotkeys with accounts."""
        return self._server.get_all_hotkeys()

    # ==================== Buying Power / balance Methods ====================

    def get_buying_power(self, hotkey: str) -> Optional[float]:
        return self._server.get_buying_power(hotkey)

    def get_balance(self, hotkey: str) -> Optional[float]:
        return self._server.get_balance(hotkey)

    def health_check(self) -> dict:
        return self._server.health_check()

    # ==================== Margin/Cash Processing Methods ====================

    def process_order_buy(self, hotkey: str, order_value_usd: float, borrowed_amount: float, fee_usd: float = 0) -> None:
        """
        Process buy order cash/margin.

        Args:
            hotkey: Miner's hotkey
            order_value_usd: Order value in USD
            borrowed_amount: Amount borrowed (calculated by caller, equities only)
            fee_usd: Transaction fee in USD

        Raises: SignalException if insufficient funds for margin
        """
        self._server.process_order_buy(hotkey, order_value_usd, borrowed_amount, fee_usd)

    def process_order_sell(self, hotkey: str, entry_value_usd: float, realized_pnl: float, loan_repaid: float, fee_usd: float = 0) -> None:
        """
        Process sell/close order. Free capital_used, compound realized PNL to balance.

        Args:
            hotkey: Miner's hotkey
            entry_value_usd: Original entry value of the position being closed
            realized_pnl: Realized PNL from this sale
            loan_repaid: Amount of loan repaid (calculated by caller, equities only)
            fee_usd: Transaction fee in USD
        """
        self._server.process_order_sell(hotkey, entry_value_usd, realized_pnl, loan_repaid, fee_usd)

    def get_total_borrowed_amount(self, hotkey: str) -> float:
        """Get total borrowed amount for a miner."""
        return self._server.get_total_borrowed_amount(hotkey)

    def can_withdraw_collateral(self, hotkey: str, amount_theta: float) -> bool:
        """
        Check if miner can withdraw the specified amount of collateral.

        Args:
            hotkey: Miner's hotkey
            amount_theta: Requested withdrawal amount in theta

        Returns:
            True if withdrawal is allowed, False otherwise
        """
        return self._server.can_withdraw_collateral(hotkey, amount_theta)

    def rebuild_account_state_from_positions(
        self,
        hotkey: str,
        positions: list,
        miner_bucket: Optional[MinerBucket] = None,
        max_return: float = 1.0,
    ) -> None:
        """
        Rebuild a miner's account state from a list of positions.

        Resets capital_used, total_realized_pnl, total_fees_paid, and total_borrowed_amount,
        then recomputes them from the provided positions.

        Args:
            hotkey: Miner's hotkey
            positions: List of Position objects or dicts for this miner
            miner_bucket: Miner bucket to restore after reset
            max_return: Max return (high water mark) to restore after reset
        """
        bucket_value = miner_bucket.value if miner_bucket else None
        self._server.rebuild_account_state_from_positions(hotkey, positions, bucket_value, max_return)

    def update_asset_selection(
        self, hotkey: str, asset_selection: TradePairCategory
    ) -> bool:
        """
        Returns:
            True if cash balance was updated, False otherwise
        """
        return self._server.update_asset_selection(hotkey, asset_selection)

    def process_fees(self, hotkey_to_fee: Dict[str, float]) -> None:
        """Batch update total_fees_paid for multiple hotkeys. Saves to disk once."""
        self._server.process_fees(hotkey_to_fee)

    def process_dividend_income(self, hotkey_to_credit: Dict[str, float]) -> None:
        """Batch update total_dividend_income for multiple hotkeys. Saves to disk once."""
        self._server.process_dividend_income(hotkey_to_credit)

