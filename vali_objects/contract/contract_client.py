from typing import Optional, Dict, Any

import template.protocol
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

    def slash_miner_collateral_proportion(self, miner_hotkey: str, slash_proportion: float=None) -> bool:
        """Slash miner's collateral by a proportion."""
        return self._server.slash_miner_collateral_proportion_rpc(miner_hotkey, slash_proportion)

    def slash_miner_collateral(self, miner_hotkey: str, slash_amount: float = None) -> bool:
        """Slash miner's collateral by a raw theta amount."""
        return self._server.slash_miner_collateral_rpc(miner_hotkey, slash_amount)

    def compute_slash_amount(self, miner_hotkey: str, drawdown: float = None) -> float:
        """Compute the slash amount based on drawdown."""
        return self._server.compute_slash_amount_rpc(miner_hotkey, drawdown)

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
        return self._server.process_withdrawal_request_rpc(amount, miner_coldkey, miner_hotkey)

    def query_withdrawal_request(self, amount: float, miner_hotkey: str) -> Dict[str, Any]:
        """Query withdrawal request (preview only - no execution)."""
        return self._server.query_withdrawal_request_rpc(amount, miner_hotkey)

    # ==================== CollateralRecord Methods ====================

    def receive_collateral_record(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        """Receive collateral record update synapse (for axon attachment)."""
        return self._server.receive_collateral_record_rpc(synapse)

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
