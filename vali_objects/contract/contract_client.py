from typing import Optional, Dict, Any

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.contract.contract_server import ContractServer
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class ContractClient(RPCClientBase):
    """
    Lightweight RPC client for ContractServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_CONTRACTMANAGER_PORT.

    In test mode (running_unit_tests=True), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct ContractServer instance.
    """

    def __init__(self, port: int = None, running_unit_tests: bool = False,
                 connect_immediately: bool = False, connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
        """
        Initialize contract client.

        Args:
            port: Port number of the contract server (default: ValiConfig.RPC_CONTRACTMANAGER_PORT)
            running_unit_tests: If True, don't connect via RPC (use set_direct_server() instead)
            connect_immediately: If True, connect in __init__. If False, call connect() later.
        """
        self.running_unit_tests = running_unit_tests
        self._direct_server = None

        super().__init__(
            service_name=ValiConfig.RPC_CONTRACTMANAGER_SERVICE_NAME,
            port=port or ValiConfig.RPC_CONTRACTMANAGER_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connection_mode=connection_mode,
            connect_immediately=connect_immediately
        )

    # ==================== Slashing Methods ====================

    # fail-fast (retry=False): these mutate irreversible on-chain state (slash/burn/withdraw/deposit,
    # vault-signed with a fresh nonce = NOT idempotent). _invoke_rpc's self-heal RE-EXECUTES a call on
    # a transient error, so auto-retrying a lost ACK here would double-slash/double-withdraw real
    # funds. retry=False surfaces the failure to the caller instead of silently re-applying it.
    def slash_miner_collateral_proportion(self, miner_hotkey: str, slash_proportion: float=None) -> bool:
        """Slash miner's collateral by a proportion."""
        return self._invoke_rpc("slash_miner_collateral_proportion_rpc",
                                args=(miner_hotkey, slash_proportion), retry=False)

    def slash_miner_collateral(self, miner_hotkey: str, slash_amount: float = None) -> bool:
        """Slash miner's collateral by a raw theta amount."""
        return self._invoke_rpc("slash_miner_collateral_rpc",
                                args=(miner_hotkey, slash_amount), retry=False)

    # ==================== Collateral Balance Methods ====================

    def get_miner_collateral_balance(self, miner_address: str, max_retries: int = 4) -> Optional[float]:
        """Get a miner's current collateral balance in theta tokens."""
        return self._server.get_miner_collateral_balance_rpc(miner_address, max_retries)

    def get_total_collateral(self) -> int:
        """Get total collateral in the contract in theta."""
        return self._server.get_total_collateral_rpc()

    def get_slashed_collateral(self) -> int:
        """Get total slashed collateral in theta."""
        return self._server.get_slashed_collateral_rpc()

    # ==================== Deposit/Withdrawal Methods ====================

    def process_deposit_request(self, extrinsic_hex: str) -> Dict[str, Any]:
        """Process a collateral deposit request."""
        return self._server.process_deposit_request_rpc(extrinsic_hex)

    def process_withdrawal_request(
        self,
        amount: float,
        miner_coldkey: str,
        miner_hotkey: str
    ) -> Dict[str, Any]:
        """Process a collateral withdrawal request."""
        # fail-fast: irreversible on-chain slash + withdraw (see slashing note) — never auto-retry.
        return self._invoke_rpc("process_withdrawal_request_rpc",
                                args=(amount, miner_coldkey, miner_hotkey), retry=False)

    def query_withdrawal_request(self, amount: float, miner_hotkey: str) -> Dict[str, Any]:
        """Query withdrawal request (preview only - no execution)."""
        return self._server.query_withdrawal_request_rpc(amount, miner_hotkey)

    def force_deposit(self, amount: float, miner_hotkey: str) -> None:
        """Update contract deposit without a stake transfer."""
        # fail-fast: irreversible on-chain deposit (see slashing note) — never auto-retry.
        return self._invoke_rpc("force_deposit_rpc", args=(amount, miner_hotkey), retry=False)

    def refresh_miner_account_size(self, hotkey: str) -> bool:
        """Refresh a miner's cached account size from their current on-chain collateral balance."""
        return self._server.refresh_miner_account_size_rpc(hotkey)

    # ==================== Verification Methods ====================

    def verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """Verify that a coldkey owns a specific hotkey using subtensor."""
        return self._server.verify_coldkey_owns_hotkey_rpc(coldkey_ss58, hotkey_ss58)

    # ==================== Test Data Injection Methods ====================

    def set_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """Inject test collateral balance (TEST ONLY - requires running_unit_tests=True)."""
        return self._server.set_test_collateral_balance_rpc(miner_hotkey, balance_rao)

    def queue_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """Queue test collateral balance (TEST ONLY - requires running_unit_tests=True)."""
        return self._server.queue_test_collateral_balance_rpc(miner_hotkey, balance_rao)

    def clear_test_collateral_balances(self) -> None:
        """Clear all test collateral balances (TEST ONLY)."""
        return self._server.clear_test_collateral_balances_rpc()

    # ==================== Static Methods ====================

    @staticmethod
    def min_collateral_penalty(collateral: float) -> float:
        """Penalize miners who do not reach the min collateral."""
        return ContractServer.min_collateral_penalty(collateral)
