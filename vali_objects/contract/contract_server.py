# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ContractServer - RPC server for contract/collateral management.

This server runs in its own process and exposes contract management via RPC.
Clients connect using ContractClient.

Account size operations are delegated to MinerAccountClient.

Usage:
    # Validator spawns the server at startup
    from vali_objects.utils.contract_server import start_contract_server
    process = Process(target=start_contract_server, args=(...))
    process.start()

    # Other processes connect via ContractClient
    from vali_objects.utils.contract_server import ContractClient
    client = ContractClient()  # Uses ValiConfig.RPC_CONTRACTMANAGER_PORT
"""
import bittensor as bt
from typing import Dict, Any, Optional
import time
from setproctitle import setproctitle
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc.rpc_server_base import RPCServerBase
import template.protocol
from vali_objects.miner_account.miner_account_client import MinerAccountClient


# ==================== Server Implementation ====================

class ContractServer(RPCServerBase):
    """
    RPC Server for contract/collateral management.

    Inherits from RPCServerBase for RPC server lifecycle management.

    Account size operations are delegated to MinerAccountClient.

    All public methods ending in _rpc are exposed via RPC to ContractClient.
    """
    service_name = ValiConfig.RPC_CONTRACTMANAGER_SERVICE_NAME
    service_port = ValiConfig.RPC_CONTRACTMANAGER_PORT

    def __init__(
        self,
        config=None,
        running_unit_tests=False,
        is_backtesting=False,
        slack_notifier=None,
        start_server=True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize ContractServer.

        Creates ValidatorContractManager instance (all business logic lives there).

        Args:
            config: Bittensor config
            running_unit_tests: Whether running in test mode
            is_backtesting: Whether backtesting
            slack_notifier: Slack notifier for health check alerts
            start_server: Whether to start RPC server immediately
            connection_mode: RPC or LOCAL mode
        """
        # Create mock config if running tests and config not provided
        if running_unit_tests:
            from shared_objects.rpc.test_mock_factory import TestMockFactory
            config = TestMockFactory.create_mock_config_if_needed(config, netuid=116, network="test")

        # Create the manager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        from vali_objects.contract.validator_contract_manager import ValidatorContractManager
        self._manager = ValidatorContractManager(
            config=config,
            running_unit_tests=running_unit_tests,
            is_backtesting=is_backtesting,
            connection_mode=connection_mode
        )

        # MinerAccountClient for receive_collateral_record synapse handling
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode)

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_CONTRACTMANAGER_SERVICE_NAME,
            port=ValiConfig.RPC_CONTRACTMANAGER_PORT,
            connection_mode=connection_mode,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # Contract server doesn't need a daemon loop
        )

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """Contract server doesn't need a daemon loop."""
        pass


    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return self._manager.health_check()

    # ==================== CollateralRecord RPC Methods ====================

    def receive_collateral_record_rpc(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        """Receive collateral record update, and update miner account sizes."""
        try:
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(f"Received collateral record update from validator hotkey [{sender_hotkey}].")
            success = self._miner_account_client.receive_collateral_record_update(synapse.collateral_record)

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

    # ==================== Collateral RPC Methods (from ValidatorContractManager) ====================

    def process_deposit_request_rpc(self, extrinsic_hex: str) -> Dict[str, Any]:
        """Process a collateral deposit request using raw data."""
        return self._manager.process_deposit_request(extrinsic_hex)

    def process_withdrawal_request_rpc(self, amount: float, miner_coldkey: str, miner_hotkey: str) -> Dict[str, Any]:
        """Process a collateral withdrawal request."""
        return self._manager.process_withdrawal_request(amount, miner_coldkey, miner_hotkey)

    def query_withdrawal_request_rpc(self, amount: float, miner_hotkey: str) -> Dict[str, Any]:
        """Query withdrawal request (preview only - no execution)."""
        return self._manager.query_withdrawal_request(amount, miner_hotkey)

    def slash_miner_collateral_proportion_rpc(self, miner_hotkey: str, slash_proportion: float=None) -> bool:
        """Slash miner's collateral by a proportion."""
        return self._manager.slash_miner_collateral_proportion(miner_hotkey, slash_proportion)

    def slash_miner_collateral_rpc(self, miner_hotkey: str, slash_amount: float = None) -> bool:
        """Slash miner's collateral by a raw theta amount."""
        return self._manager.slash_miner_collateral(miner_hotkey, slash_amount)

    def compute_slash_amount_rpc(self, miner_hotkey: str, drawdown: float = None) -> float:
        """Compute the slash amount based on drawdown."""
        return self._manager.compute_slash_amount(miner_hotkey, drawdown)

    def get_miner_collateral_balance_rpc(self, miner_address: str, max_retries: int = 4) -> Optional[float]:
        """Get a miner's current collateral balance in theta tokens."""
        return self._manager.get_miner_collateral_balance(miner_address, max_retries)

    def get_total_collateral_rpc(self) -> int:
        """Get total collateral in the contract in theta."""
        return self._manager.get_total_collateral()

    def get_slashed_collateral_rpc(self) -> int:
        """Get total slashed collateral in theta."""
        return self._manager.get_slashed_collateral()

    def verify_coldkey_owns_hotkey_rpc(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """Verify that a coldkey owns a specific hotkey using subtensor."""
        return self._manager.verify_coldkey_owns_hotkey(coldkey_ss58, hotkey_ss58)

    def set_test_collateral_balance_rpc(self, miner_hotkey: str, balance_rao: int) -> None:
        """Inject test collateral balance (TEST ONLY - requires running_unit_tests=True)."""
        return self._manager.set_test_collateral_balance(miner_hotkey, balance_rao)

    def queue_test_collateral_balance_rpc(self, miner_hotkey: str, balance_rao: int) -> None:
        """Queue test collateral balance (TEST ONLY - requires running_unit_tests=True)."""
        return self._manager.queue_test_collateral_balance(miner_hotkey, balance_rao)

    def clear_test_collateral_balances_rpc(self) -> None:
        """Clear all test collateral balances (TEST ONLY)."""
        return self._manager.clear_test_collateral_balances()

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def get_miner_collateral_balance(self, miner_address: str, max_retries: int = 4) -> Optional[float]:
        return self._manager.get_miner_collateral_balance(miner_address, max_retries)

    def process_deposit_request(self, extrinsic_hex: str) -> Dict[str, Any]:
        return self._manager.process_deposit_request(extrinsic_hex)

    def query_withdrawal_request(self, amount: float, miner_hotkey: str) -> Dict[str, Any]:
        return self._manager.query_withdrawal_request(amount, miner_hotkey)

    def process_withdrawal_request(self, amount: float, miner_coldkey: str, miner_hotkey: str) -> Dict[str, Any]:
        return self._manager.process_withdrawal_request(amount, miner_coldkey, miner_hotkey)

    def slash_miner_collateral(self, miner_hotkey: str, slash_amount: float = None) -> bool:
        return self._manager.slash_miner_collateral(miner_hotkey, slash_amount)

    def slash_miner_collateral_proportion(self, miner_hotkey: str, slash_proportion: float) -> bool:
        return self._manager.slash_miner_collateral_proportion(miner_hotkey, slash_proportion)

    def compute_slash_amount(self, miner_hotkey: str, drawdown: float = None) -> float:
        return self._manager.compute_slash_amount(miner_hotkey, drawdown)

    def get_total_collateral(self) -> int:
        return self._manager.get_total_collateral()

    def get_slashed_collateral(self) -> int:
        return self._manager.get_slashed_collateral()

    def receive_collateral_record(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        return self.receive_collateral_record_rpc(synapse)

    def verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        return self._manager.verify_coldkey_owns_hotkey(coldkey_ss58, hotkey_ss58)

    def set_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """Inject test collateral balance (forward-compatible alias)."""
        return self._manager.set_test_collateral_balance(miner_hotkey, balance_rao)

    def queue_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """Queue test collateral balance (forward-compatible alias)."""
        return self._manager.queue_test_collateral_balance(miner_hotkey, balance_rao)

    def clear_test_collateral_balances(self) -> None:
        """Clear all test collateral balances (forward-compatible alias)."""
        return self._manager.clear_test_collateral_balances()

    @staticmethod
    def min_collateral_penalty(collateral: float) -> float:
        """Penalize miners who do not reach the min collateral."""
        from vali_objects.contract.validator_contract_manager import ValidatorContractManager
        return ValidatorContractManager.min_collateral_penalty(collateral)


# ==================== Server Entry Point ====================

def start_contract_server(
    config,
    running_unit_tests,
    is_backtesting,
    slack_notifier,
    server_ready=None,
):
    """
    Entry point for server process.

    The server creates its own PositionManagerClient internally (forward compatibility pattern).
    For tests, use ContractServer directly with set_direct_position_server().

    Args:
        config: Bittensor config
        running_unit_tests: Whether running in test mode
        is_backtesting: Whether backtesting
        slack_notifier: Slack notifier
        server_ready: Event to signal when server is ready
    """
    from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
    setproctitle("vali_ContractServerProcess")

    server_instance = ContractServer(
        config=config,
        running_unit_tests=running_unit_tests,
        is_backtesting=is_backtesting,
        slack_notifier=slack_notifier,
        start_server=True,
    )

    bt.logging.success(f"ContractServer ready on port {ValiConfig.RPC_CONTRACTMANAGER_PORT}")

    if server_ready:
        server_ready.set()

    # Block until shutdown
    while not ShutdownCoordinator.is_shutdown():
        time.sleep(1)

    server_instance.shutdown()
    bt.logging.info("ContractServer process exiting")
