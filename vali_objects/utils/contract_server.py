# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ContractServer - RPC server for contract/collateral management.

This server runs in its own process and exposes contract management via RPC.
Clients connect using ContractClient.

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
from typing import Dict, Any, Optional, List
import time
from setproctitle import setproctitle
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc_server_base import RPCServerBase
from shared_objects.rpc_client_base import RPCClientBase
import template.protocol


# ==================== Server Implementation ====================

class ContractServer(RPCServerBase):
    """
    RPC Server for contract/collateral management.

    Inherits from RPCServerBase for RPC server lifecycle management.

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
        # Create the manager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        from vali_objects.utils.validator_contract_manager import ValidatorContractManager
        self._manager = ValidatorContractManager(
            config=config,
            running_unit_tests=running_unit_tests,
            is_backtesting=is_backtesting,
            connection_mode=connection_mode
        )

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

    # ==================== Properties ====================

    @property
    def vault_wallet(self):
        """Get vault wallet from manager."""
        return self._manager.vault_wallet

    @vault_wallet.setter
    def vault_wallet(self, value):
        """Set vault wallet on manager."""
        self._manager.vault_wallet = value


    # ==================== Setup Methods ====================

    def load_contract_owner(self):
        """Load EVM contract owner secrets and vault wallet."""
        self._manager.load_contract_owner()

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return self._manager.health_check()

    def miner_account_sizes_dict_rpc(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Convert miner account sizes to checkpoint format for backup/sync."""
        return self._manager.miner_account_sizes_dict(most_recent_only)

    def sync_miner_account_sizes_data_rpc(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]):
        """Sync miner account sizes data from external source (backup/sync)."""
        return self._manager.sync_miner_account_sizes_data(account_sizes_data)

    def re_init_account_sizes_rpc(self):
        """Reload account sizes from disk (useful for tests)."""
        return self._manager.re_init_account_sizes()

    def process_deposit_request_rpc(self, extrinsic_hex: str) -> Dict[str, Any]:
        """Process a collateral deposit request using raw data."""
        return self._manager.process_deposit_request(extrinsic_hex)

    def process_withdrawal_request_rpc(self, amount: float, miner_coldkey: str, miner_hotkey: str) -> Dict[str, Any]:
        """Process a collateral withdrawal request."""
        return self._manager.process_withdrawal_request(amount, miner_coldkey, miner_hotkey)

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

    def set_miner_account_size_rpc(self, hotkey: str, timestamp_ms: int = None) -> bool:
        """Set the account size for a miner."""
        return self._manager.set_miner_account_size(hotkey, timestamp_ms)

    def get_miner_account_size_rpc(self, hotkey: str, timestamp_ms: int = None, most_recent: bool = False,
                                   records_dict: dict = None, use_account_floor: bool = False) -> Optional[float]:
        """Get the account size for a miner at a given timestamp."""
        return self._manager.get_miner_account_size(hotkey, timestamp_ms, most_recent, records_dict, use_account_floor)

    def get_all_miner_account_sizes_rpc(self, miner_account_sizes: dict = None, timestamp_ms: int = None) -> Dict[str, float]:
        """Return a dict of all miner account sizes at a timestamp_ms."""
        return self._manager.get_all_miner_account_sizes(miner_account_sizes, timestamp_ms)

    def receive_collateral_record_rpc(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        """Receive collateral record update, and update miner account sizes."""
        try:
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(f"Received collateral record update from validator hotkey [{sender_hotkey}].")
            success = self.receive_collateral_record_update_rpc(synapse.collateral_record)

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

    def receive_collateral_record_update_rpc(self, collateral_record_data: dict) -> bool:
        """Process an incoming CollateralRecord synapse and update miner_account_sizes."""
        return self._manager.receive_collateral_record_update(collateral_record_data)

    def verify_coldkey_owns_hotkey_rpc(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """Verify that a coldkey owns a specific hotkey using subtensor."""
        return self._manager.verify_coldkey_owns_hotkey(coldkey_ss58, hotkey_ss58)

    def set_test_collateral_balance_rpc(self, miner_hotkey: str, balance_rao: int) -> None:
        """Inject test collateral balance (TEST ONLY - requires running_unit_tests=True)."""
        return self._manager.set_test_collateral_balance(miner_hotkey, balance_rao)

    def clear_test_collateral_balances_rpc(self) -> None:
        """Clear all test collateral balances (TEST ONLY)."""
        return self._manager.clear_test_collateral_balances()

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def get_miner_collateral_balance(self, miner_address: str, max_retries: int = 4) -> Optional[float]:
        return self._manager.get_miner_collateral_balance(miner_address, max_retries)

    def get_miner_account_size(self, hotkey: str, timestamp_ms: int = None, most_recent: bool = False,
                               records_dict: dict = None, use_account_floor: bool = False) -> Optional[float]:
        return self._manager.get_miner_account_size(hotkey, timestamp_ms, most_recent, records_dict, use_account_floor)

    def set_miner_account_size(self, hotkey: str, timestamp_ms: int = None) -> bool:
        return self._manager.set_miner_account_size(hotkey, timestamp_ms)

    def get_all_miner_account_sizes(self, miner_account_sizes: dict = None, timestamp_ms: int = None) -> Dict[str, float]:
        return self._manager.get_all_miner_account_sizes(miner_account_sizes, timestamp_ms)

    def miner_account_sizes_dict(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        return self._manager.miner_account_sizes_dict(most_recent_only)

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]):
        return self._manager.sync_miner_account_sizes_data(account_sizes_data)

    def re_init_account_sizes(self):
        return self._manager.re_init_account_sizes()

    def process_deposit_request(self, extrinsic_hex: str) -> Dict[str, Any]:
        return self._manager.process_deposit_request(extrinsic_hex)

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

    def receive_collateral_record_update(self, collateral_record_data: dict) -> bool:
        return self._manager.receive_collateral_record_update(collateral_record_data)

    def verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        return self._manager.verify_coldkey_owns_hotkey(coldkey_ss58, hotkey_ss58)

    def set_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """Inject test collateral balance (forward-compatible alias)."""
        return self._manager.set_test_collateral_balance(miner_hotkey, balance_rao)

    def clear_test_collateral_balances(self) -> None:
        """Clear all test collateral balances (forward-compatible alias)."""
        return self._manager.clear_test_collateral_balances()

    @staticmethod
    def min_collateral_penalty(collateral: float) -> float:
        """Penalize miners who do not reach the min collateral."""
        from vali_objects.utils.validator_contract_manager import ValidatorContractManager
        return ValidatorContractManager.min_collateral_penalty(collateral)


# ==================== Client Implementation ====================

class ContractClient(RPCClientBase):
    """
    Lightweight RPC client for ContractServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_CONTRACTMANAGER_PORT.

    In test mode (running_unit_tests=True), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct ContractServer instance.
    """

    def __init__(self, port: int = None, running_unit_tests: bool = False,
                 connect_immediately: bool = True, connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
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

    # ==================== Account Size Methods ====================

    def get_miner_account_size(
        self,
        hotkey: str,
        timestamp_ms: int = None,
        most_recent: bool = False,
        records_dict: dict = None,
        use_account_floor: bool = False
    ) -> Optional[float]:
        """Get the account size for a miner at a given timestamp."""
        return self._server.get_miner_account_size_rpc(
            hotkey, timestamp_ms, most_recent, records_dict, use_account_floor
        )

    def set_miner_account_size(self, hotkey: str, timestamp_ms: int = None) -> bool:
        """Set the account size for a miner."""
        return self._server.set_miner_account_size_rpc(hotkey, timestamp_ms)

    def get_all_miner_account_sizes(
        self,
        miner_account_sizes: dict = None,
        timestamp_ms: int = None
    ) -> Dict[str, float]:
        """Get all miner account sizes at a timestamp."""
        return self._server.get_all_miner_account_sizes_rpc(miner_account_sizes, timestamp_ms)

    def miner_account_sizes_dict(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Get miner account sizes dict for backup/sync."""
        return self._server.miner_account_sizes_dict_rpc(most_recent_only)

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Sync miner account sizes data from external source."""
        return self._server.sync_miner_account_sizes_data_rpc(account_sizes_data)

    def re_init_account_sizes(self) -> None:
        """Reload account sizes from disk (useful for tests)."""
        return self._server.re_init_account_sizes_rpc()

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

    # ==================== CollateralRecord Methods ====================

    def receive_collateral_record(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        """Receive collateral record update synapse (for axon attachment)."""
        return self._server.receive_collateral_record_rpc(synapse)

    def receive_collateral_record_update(self, collateral_record_data: dict) -> bool:
        """Process an incoming CollateralRecord and update miner_account_sizes."""
        return self._server.receive_collateral_record_update_rpc(collateral_record_data)

    def verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """Verify that a coldkey owns a specific hotkey using subtensor."""
        return self._server.verify_coldkey_owns_hotkey_rpc(coldkey_ss58, hotkey_ss58)

    # ==================== Test Data Injection Methods ====================

    def set_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """Inject test collateral balance (TEST ONLY - requires running_unit_tests=True)."""
        return self._server.set_test_collateral_balance_rpc(miner_hotkey, balance_rao)

    def clear_test_collateral_balances(self) -> None:
        """Clear all test collateral balances (TEST ONLY)."""
        return self._server.clear_test_collateral_balances_rpc()

    # ==================== Setup Methods ====================

    def load_contract_owner(self):
        """Load EVM contract owner secrets and vault wallet."""
        self._server.load_contract_owner()

    # ==================== Static Methods ====================

    @staticmethod
    def min_collateral_penalty(collateral: float) -> float:
        """Penalize miners who do not reach the min collateral."""
        return ContractServer.min_collateral_penalty(collateral)


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
    from shared_objects.shutdown_coordinator import ShutdownCoordinator
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
