# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ValidatorContractManager - Business logic for contract/collateral management.

This manager handles all collateral operations including:
- Deposit/withdrawal processing
- Account size tracking
- Slashing calculations
- Collateral record broadcasting

The manager contains NO RPC infrastructure - that lives in ContractServer.
This is pure business logic that can be tested independently.
"""
import threading
from datetime import timezone, datetime, timedelta
import bittensor as bt
from collateral_sdk import CollateralManager, Network
from typing import Dict, Any, Optional, List
import time
from time_util.time_util import TimeUtil
from vali_objects.validator_broadcast_base import ValidatorBroadcastBase
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import template.protocol
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient


# ==================== Data Classes ====================


class CollateralRecord:
    def __init__(self, account_size, account_size_theta, update_time_ms):
        self.account_size = account_size
        self.account_size_theta = account_size_theta
        self.update_time_ms = update_time_ms
        self.valid_date_timestamp = CollateralRecord.valid_from_ms(update_time_ms)

    @staticmethod
    def valid_from_ms(update_time_ms) -> int:
        """Returns timestamp of start of next day (00:00:00 UTC) when this record is valid"""
        dt = datetime.fromtimestamp(update_time_ms / 1000, tz=timezone.utc)
        start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        # Record is valid from the start of the next day
        start_of_next_day = start_of_day + timedelta(days=1)
        return int(start_of_next_day.timestamp() * 1000)

    @property
    def valid_date_str(self) -> str:
        """Returns YYYY-MM-DD format for easy reading"""
        return TimeUtil.millis_to_short_date_str(self.valid_date_timestamp)

    def __repr__(self):
        """String representation"""
        return str(vars(self))



# ==================== Constants ====================

TARGET_MS = 1762308000000
NOV_1_MS = 1761951599000


# ==================== Manager Implementation ====================

class ValidatorContractManager(ValidatorBroadcastBase):
    """
    Business logic for contract/collateral management.

    This manager contains ALL business logic for:
    - Deposit/withdrawal processing
    - Account size tracking and disk persistence
    - Slashing calculations based on drawdown
    - Collateral record broadcasting to validators

    NO RPC infrastructure here - pure business logic only.
    ContractServer wraps this manager and exposes methods via RPC.

    Inherits from ValidatorBroadcastBase for shared broadcast functionality.
    """

    def __init__(
        self,
        config=None,
        running_unit_tests=False,
        is_backtesting=False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize ValidatorContractManager.

        Creates own RPC clients internally (forward compatibility pattern):
        - PositionManagerClient
        - PerfLedgerClient
        - MetagraphClient

        Args:
            config: Bittensor config
            running_unit_tests: Whether running in test mode
            is_backtesting: Whether backtesting
            connection_mode: RPC or LOCAL mode
        """
        self.running_unit_tests = running_unit_tests
        self.config = config
        self.is_backtesting = is_backtesting
        self.connection_mode = connection_mode
        self.secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)

        # Create RPC clients (forward compatibility - no parameter passing)
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connection_mode=connection_mode
        )
        self._perf_ledger_client = PerfLedgerClient(connection_mode=connection_mode)

        # Store network type for dynamic max_theta property (before initializing base class)
        self.is_testnet = config.subtensor.network == "test"

        # Initialize ValidatorBroadcastBase with broadcast configuration (derives is_mothership internally)
        ValidatorBroadcastBase.__init__(
            self,
            running_unit_tests=running_unit_tests,
            is_testnet=self.is_testnet,
            config=config,
            connection_mode=connection_mode
        )

        # Locking strategy - EAGER initialization (not lazy!)
        # RLock allows same thread to acquire lock multiple times (needed for nested calls)
        self._account_sizes_lock = threading.RLock()
        # Lock for disk I/O serialization to prevent concurrent file writes
        self._disk_lock = threading.Lock()
        # Lock for test collateral balances dict (prevents concurrent modifications in tests)
        self._test_balances_lock = threading.Lock()

        # Initialize collateral manager based on network type
        if self.is_testnet:
            bt.logging.info("Using testnet collateral manager")
            self.collateral_manager = CollateralManager(Network.TESTNET)
        else:
            bt.logging.info("Using mainnet collateral manager")
            self.collateral_manager = CollateralManager(Network.MAINNET)

        # GCP secret manager
        self._gcp_secret_manager_client = None

        # Initialize miner account sizes file location
        self.MINER_ACCOUNT_SIZES_FILE = ValiBkpUtils.get_miner_account_sizes_file_location(
            running_unit_tests=running_unit_tests
        )

        # Use normal Python dict (no IPC overhead)
        self.miner_account_sizes: Dict[str, List[CollateralRecord]] = {}
        self._load_miner_account_sizes_from_disk()

        # Test collateral balance registry (only used when running_unit_tests=True)
        # Allows tests to inject specific collateral balances instead of making blockchain calls
        # Key: miner_hotkey -> Value: balance in rao (int)
        self._test_collateral_balances: Dict[str, int] = {}

        # Test collateral balance queue (only used when running_unit_tests=True)
        # Allows tests to inject a sequence of balances for the same miner
        # Key: miner_hotkey -> Value: list of balances (FIFO queue)
        # This is needed for race condition tests that simulate multiple concurrent balance changes
        self._test_collateral_balance_queues: Dict[str, List[int]] = {}

        self.setup()

    # ==================== Properties ====================

    @property
    def max_theta(self) -> float:
        """Get the current maximum collateral balance limit in theta tokens."""
        if self.is_testnet:
            return ValiConfig.MAX_COLLATERAL_BALANCE_TESTNET
        else:
            return ValiConfig.MAX_COLLATERAL_BALANCE_THETA

    @property
    def min_theta(self) -> float:
        """Get the current minimum collateral balance limit in theta tokens."""
        if self.is_testnet:
            return ValiConfig.MIN_COLLATERAL_BALANCE_TESTNET
        else:
            return ValiConfig.MIN_COLLATERAL_BALANCE_THETA


    # ==================== Setup Methods ====================

    def setup(self):
        """
        reinstate wrongfully eliminated miner deposits
        update all miner account sizes when COST_PER_THETA changes
        """
        if not self.is_mothership:
            return

        now_ms = TimeUtil.now_in_millis()
        if now_ms > TARGET_MS:
            return

        # miners_to_reinstate = {}
        # for miner, amount in miners_to_reinstate.items():
        #     self.force_deposit(amount, miner)

        # Update CPT
        # update_thread = threading.Thread(target=self.refresh_miner_account_sizes, daemon=True)
        # update_thread.start()
        # bt.logging.info("COST_PER_THETA migration started in background thread")

    def refresh_miner_account_sizes(self):
        """
        refresh miner account sizes for new CPT
        """
        update_count = 0

        # Acquire lock and copy keys to avoid iterator invalidation
        with self._account_sizes_lock:
            hotkeys = list(self.miner_account_sizes.keys())

        # Process each miner (set_miner_account_size will acquire lock for each)
        for hotkey in hotkeys:
            try:
                prev_acct_size = self.get_miner_account_size(hotkey)
                bt.logging.info(f"Current account size for {hotkey}: ${prev_acct_size:,.2f}")
                self.set_miner_account_size(hotkey, NOV_1_MS)
                update_count += 1
                time.sleep(0.5)
            except Exception as e:
                bt.logging.error(f"Failed to update account size for {hotkey}: {e}")
        bt.logging.info(f"COST_PER_THETA update completed for {update_count} miners")

    def _load_miner_account_sizes_from_disk(self):
        """Load miner account sizes from disk during initialization - protected by locks"""
        with self._disk_lock:
            try:
                disk_data = ValiUtils.get_vali_json_file_dict(self.MINER_ACCOUNT_SIZES_FILE)
                parsed_data = self._parse_miner_account_sizes_dict(disk_data)

                # Acquire account_sizes_lock to update the dict
                with self._account_sizes_lock:
                    self.miner_account_sizes.clear()
                    self.miner_account_sizes.update(parsed_data)

                bt.logging.info(f"Loaded {len(self.miner_account_sizes)} miner account size records from disk")
            except Exception as e:
                bt.logging.warning(f"Failed to load miner account sizes from disk: {e}")

    def re_init_account_sizes(self):
        """Public method to reload account sizes from disk (useful for tests)"""
        self._load_miner_account_sizes_from_disk()

    def _save_miner_account_sizes_to_disk(self):
        """Save miner account sizes to disk - protected by _disk_lock to prevent concurrent writes"""
        with self._disk_lock:
            try:
                data_dict = self.miner_account_sizes_dict()
                ValiBkpUtils.write_file(self.MINER_ACCOUNT_SIZES_FILE, data_dict)
            except Exception as e:
                bt.logging.error(f"Failed to save miner account sizes to disk: {e}")

    def miner_account_sizes_dict(self, most_recent_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Convert miner account sizes to checkpoint format for backup/sync

        Args:
            most_recent_only: If True, only return the most recent record for each miner

        Returns:
            Dictionary with hotkeys as keys and list of record dicts as values
        """
        with self._account_sizes_lock:
            json_dict = {}
            for hotkey, records in self.miner_account_sizes.items():
                if most_recent_only and records:
                    # Only include the most recent (last) record
                    json_dict[hotkey] = [vars(records[-1])]
                else:
                    json_dict[hotkey] = [vars(record) for record in records]
            return json_dict

    @staticmethod
    def _parse_miner_account_sizes_dict(data_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[
        str, List[CollateralRecord]]:
        """Parse miner account sizes from disk format back to CollateralRecord objects"""
        parsed_dict = {}
        for hotkey, records_data in data_dict.items():
            try:
                parsed_records = []
                for record_data in records_data:
                    if isinstance(record_data, dict) and all(
                            key in record_data for key in ["account_size", "update_time_ms"]):
                        record = CollateralRecord(record_data["account_size"], record_data["account_size_theta"],
                                                  record_data["update_time_ms"])
                        parsed_records.append(record)

                if parsed_records:  # Only add if we have valid records
                    parsed_dict[hotkey] = parsed_records
            except Exception as e:
                bt.logging.warning(f"Failed to parse account size records for {hotkey}: {e}")

        return parsed_dict

    def health_check(self) -> dict:
        """Health check for monitoring."""
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "num_account_records": len(self.miner_account_sizes)
        }

    def sync_miner_account_sizes_data(self, account_sizes_data: Dict[str, List[Dict[str, Any]]]):
        """
        Sync miner account sizes data from external source (backup/sync).
        If empty dict is passed, clears all account sizes (useful for tests).
        """
        try:
            with self._account_sizes_lock:
                if not account_sizes_data:
                    assert self.running_unit_tests, "Empty account sizes data can only be used in test mode"
                    # Empty dict = clear all data (useful for test cleanup)
                    bt.logging.info("Clearing all miner account sizes")
                    self.miner_account_sizes.clear()
                    self._save_miner_account_sizes_to_disk()
                    return

                synced_data = self._parse_miner_account_sizes_dict(account_sizes_data)
                self.miner_account_sizes.clear()
                self.miner_account_sizes.update(synced_data)
                self._save_miner_account_sizes_to_disk()
                bt.logging.info(f"Synced {len(self.miner_account_sizes)} miner account size records")
        except Exception as e:
            bt.logging.error(f"Failed to sync miner account sizes data: {e}")

    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Get secret with fallback to local secrets

        Args:
            secret_name (str): name of secret
        """
        secret = self._get_gcp_secret(secret_name)
        if secret is not None:
            return secret

        secret = self.secrets.get(secret_name)
        if secret is not None:
            bt.logging.info(f"{secret_name} retrieved from local secrets file")
        return secret

    def _get_gcp_secret(self, secret_name: str) -> Optional[str]:
        """
        Get vault password from Google Cloud Secret Manager.

        Args:
            secret_name (str): name of secret

        Returns:
            str: Vault password or None if not found
        """
        try:
            if self._gcp_secret_manager_client is None:
                # noinspection PyPackageRequirements
                from google.cloud import secretmanager

                self._gcp_secret_manager_client = secretmanager.SecretManagerServiceClient()

            secret_path = self._gcp_secret_manager_client.secret_version_path(
                self.secrets.get('gcp_project_name'), self.secrets.get(secret_name), "latest"
            )
            response = self._gcp_secret_manager_client.access_secret_version(name=secret_path)
            secret = response.payload.data.decode()

            if secret:
                bt.logging.info(f"{secret_name} retrieved from Google Cloud Secret Manager")
                return secret
            else:
                bt.logging.debug(f"{secret_name} not found in Google Cloud Secret Manager")
                return None
        except Exception as e:
            bt.logging.debug(f"Failed to retrieve {secret_name} from Google Cloud: {e}")

    def to_theta(self, rao_amount: int) -> float:
        """
        Convert rao_theta amount to theta tokens.

        Args:
            rao_amount (int): Amount in RAO units

        Returns:
            float: Amount in theta tokens
        """
        theta_amount = rao_amount / 10 ** 9  # Convert rao_theta to theta
        return theta_amount

    def process_deposit_request(self, extrinsic_hex: str) -> Dict[str, Any]:
        """
        Process a collateral deposit request using raw data.

        Args:
            extrinsic_hex (str): Hex-encoded extrinsic data
            amount (float): Amount in theta tokens
            miner_address (str): Miner's SS58 address

        Returns:
            Dict[str, Any]: Result of deposit operation
        """
        try:
            bt.logging.info("Received deposit request")
            # Decode and validate the extrinsic
            try:
                encoded_extrinsic = bytes.fromhex(extrinsic_hex)
                extrinsic = self.collateral_manager.decode_extrinsic(encoded_extrinsic)
                bt.logging.info("Extrinsic decoded successfully")
            except Exception as e:
                error_msg = f"Invalid extrinsic data: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }

            # Execute the deposit through the collateral manager
            try:
                miner_hotkey = next(
                    arg["value"] for arg in extrinsic.value["call"]["call_args"] if arg["name"] == "hotkey")
                deposit_amount = next(
                    arg["value"] for arg in extrinsic.value["call"]["call_args"] if arg["name"] == "alpha_amount")
                deposit_amount_theta = self.to_theta(deposit_amount)

                # Check collateral balance limit before processing
                try:
                    current_balance_theta = self.to_theta(self.collateral_manager.balance_of(miner_hotkey))

                    if current_balance_theta + deposit_amount_theta > self.max_theta:
                        error_msg = (f"Deposit would exceed maximum balance limit. "
                                     f"Current: {current_balance_theta:.2f} Theta, "
                                     f"Deposit: {deposit_amount_theta:.2f} Theta, "
                                     f"Limit: {self.max_theta} Theta")
                        bt.logging.warning(error_msg)
                        return {
                            "successfully_processed": False,
                            "error_message": error_msg
                        }

                except Exception as e:
                    bt.logging.error(f"Failed to check balance limit: {e}")
                    return {
                        "successfully_processed": False,
                        "error_message": e
                    }

                # # All positions must be closed before a miner can deposit or withdraw
                # if len(self.position_manager.get_positions_for_one_hotkey(miner_hotkey, only_open_positions=True)) > 0:
                #     return {
                #         "successfully_processed": False,
                #         "error_message": "Miner has open positions, please close all positions before depositing or withdrawing collateral"
                #     }

                bt.logging.info(f"Processing deposit for: {deposit_amount_theta} Theta to miner: {miner_hotkey}")
                owner_address = self.get_secret("collateral_owner_address")
                owner_private_key = self.get_secret("collateral_owner_private_key")
                vault_password = self.get_secret("gcp_vali_pw_name")
                try:
                    deposited_balance = self.collateral_manager.deposit(
                        extrinsic=extrinsic,
                        source_hotkey=miner_hotkey,
                        vault_stake=self.wallet.hotkey.ss58_address,
                        vault_wallet=self.wallet,
                        owner_address=owner_address,
                        owner_private_key=owner_private_key,
                        wallet_password=vault_password
                    )
                finally:
                    del owner_address
                    del owner_private_key
                    del vault_password

                msg = f"Deposit successful: {self.to_theta(deposited_balance.rao)} Theta deposited to miner: {miner_hotkey}"
                bt.logging.info(msg)
                self.set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
                return {
                    "successfully_processed": True,
                    "error_message": ""
                }

            except Exception as e:
                error_msg = f"Deposit execution failed: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }

        except Exception as e:
            error_msg = f"Deposit processing error: {str(e)}"
            bt.logging.error(error_msg)
            return {
                "successfully_processed": False,
                "error_message": error_msg
            }

    def force_deposit(self, amount: float, miner_hotkey: str):
        """
        Update contract deposit without a stake transfer.
        Used to reinstate miners wrongfully slashed.

        Args:
            amount (float): Amount in theta tokens
            miner_hotkey (str): Miner's SS58 hotkey address
        """
        try:
            bt.logging.info(f"Processing force deposit to {miner_hotkey} for {amount} Theta")
            owner_address = self.get_secret("collateral_owner_address")
            owner_private_key = self.get_secret("collateral_owner_private_key")
            try:
                self.collateral_manager.force_deposit(
                    address=miner_hotkey,
                    amount=int(amount * 10 ** 9),  # convert theta to rao_theta
                    owner_address=owner_address,
                    owner_private_key=owner_private_key
                )
            finally:
                del owner_address
                del owner_private_key
            bt.logging.info(f"Force deposit successful: {amount} Theta deposited for {miner_hotkey}")
        except Exception as e:
            bt.logging.error(f"Force deposit execution failed: {str(e)}")

    def query_withdrawal_request(self, amount: float, miner_hotkey: str) -> Dict[str, Any]:
        """
        Query for slashed amount when a withdrawal request is received.

        Args:
            amount (float): Amount to withdraw in theta tokens
            miner_hotkey (str): Miner's SS58 hotkey

        Returns:
            Dict[str, Any]: Result of withdrawal operation
        """
        try:
            bt.logging.info("Received withdrawal query")
            # Check current collateral balance (uses test balance injection in test mode)
            try:
                theta_current_balance = self.get_miner_collateral_balance(miner_hotkey)
                if theta_current_balance is None:
                    error_msg = f"Failed to retrieve collateral balance for {miner_hotkey}"
                    bt.logging.error(error_msg)
                    return {
                        "successfully_processed": False,
                        "error_message": error_msg
                    }
                if amount > theta_current_balance:
                    error_msg = f"Insufficient collateral balance. Available: {theta_current_balance}, Requested: {amount}"
                    bt.logging.error(error_msg)
                    return {
                        "successfully_processed": False,
                        "error_message": error_msg
                    }
            except Exception as e:
                error_msg = f"Failed to check collateral balance: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }

            # Determine amount slashed and remaining amount eligible for withdrawal
            drawdown = self._position_client.compute_realtime_drawdown(miner_hotkey)

            # penalty free withdrawals down to MAX_COLLATERAL_BALANCE_THETA
            penalty_free_amount = max(0.0, theta_current_balance - self.max_theta)
            penalty_amount = max(0.0, amount - penalty_free_amount)
            withdrawal_proportion = penalty_amount / theta_current_balance if theta_current_balance > 0 else 0

            slashed_amount = self.compute_slash_amount(miner_hotkey, drawdown) * withdrawal_proportion
            withdrawal_amount = amount - slashed_amount
            new_balance = theta_current_balance - amount

            return {
                "successfully_processed": True,
                "error_message": "",
                "drawdown": drawdown,
                "slashed_amount": slashed_amount,
                "withdrawal_amount": withdrawal_amount,
                "new_balance": new_balance
            }
        except Exception as e:
            error_msg = f"Withdrawal query error: {str(e)}"
            bt.logging.error(error_msg)
            return {
                "successfully_processed": False,
                "error_message": error_msg
            }

    def process_withdrawal_request(self, amount: float, miner_coldkey: str, miner_hotkey: str) -> Dict[str, Any]:
        """
        Process a collateral withdrawal request, and slash proportionally.

        Args:
            amount (float): Amount to withdraw in theta tokens
            miner_coldkey (str): Miner's SS58 wallet coldkey address to return collateral to
            miner_hotkey (str): Miner's SS58 hotkey

        Returns:
            Dict[str, Any]: Result of withdrawal operation
        """
        try:
            bt.logging.info("Received withdrawal request")
            try:
                current_balance = self.collateral_manager.balance_of(miner_hotkey)
                theta_current_balance = self.to_theta(current_balance)
                if amount > theta_current_balance:
                    error_msg = f"Insufficient collateral balance. Available: {theta_current_balance}, Requested: {amount}"
                    bt.logging.error(error_msg)
                    return {
                        "successfully_processed": False,
                        "error_message": error_msg
                    }
            except Exception as e:
                error_msg = f"Failed to check collateral balance: {str(e)}"
                bt.logging.error(error_msg)
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }

            # Determine amount slashed and remaining amount eligible for withdrawal
            drawdown = self._position_client.compute_realtime_drawdown(miner_hotkey)

            # penalty free withdrawals down to MAX_COLLATERAL_BALANCE_THETA
            penalty_free_amount = max(0.0, theta_current_balance - self.max_theta)
            penalty_amount = max(0.0, amount - penalty_free_amount)
            withdrawal_proportion = penalty_amount / theta_current_balance if theta_current_balance > 0 else 0

            slashed_amount = self.compute_slash_amount(miner_hotkey, drawdown) * withdrawal_proportion
            withdrawal_amount = amount - slashed_amount

            bt.logging.info(
                f"Processing withdrawal request from {miner_hotkey} for {amount} Theta. Current drawdown: {(1 - drawdown) * 100}%. {slashed_amount} Theta will be slashed. {withdrawal_amount} Theta will be withdrawn.")
            self.slash_miner_collateral(miner_hotkey, slashed_amount)

            owner_address = self.get_secret("collateral_owner_address")
            owner_private_key = self.get_secret("collateral_owner_private_key")
            vault_password = self.get_secret("gcp_vali_pw_name")
            try:
                withdrawn_balance = self.collateral_manager.withdraw(
                    amount=int(withdrawal_amount * 10 ** 9),  # convert theta to rao_theta
                    source_coldkey=miner_coldkey,
                    source_hotkey=miner_hotkey,
                    vault_stake=self.vault_wallet.hotkey.ss58_address,
                    vault_wallet=self.vault_wallet,
                    owner_address=owner_address,
                    owner_private_key=owner_private_key,
                    wallet_password=vault_password
                )
            finally:
                del owner_address
                del owner_private_key
                del vault_password
            returned_theta = self.to_theta(withdrawn_balance.rao)
            msg = f"Withdrawal successful: {returned_theta} Theta withdrawn for {miner_hotkey}, returned to {miner_coldkey}"
            bt.logging.info(msg)
            self.set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
            return {
                "successfully_processed": True,
                "error_message": "",
                "returned_amount": returned_theta,
                "returned_to": miner_coldkey
            }

        except Exception as e:
            error_msg = f"Withdrawal processing execution failed: {str(e)}"
            bt.logging.error(error_msg)
            return {
                "successfully_processed": False,
                "error_message": error_msg,
                "returned_amount": 0.0,
                "returned_to": ""
            }

    def compute_slash_amount(self, miner_hotkey: str, drawdown: float = None) -> float:
        """
        Compute the amount of collateral balance to slash, depending on current drawdown.

        The amount slashed is proportional to the drawdown, scaled to the total collateral balance.
        For ex:
        10% drawdown (elimination) -> Slash 100%
        5% drawdown -> Slash 50%
        3% drawdown -> Slash 30%

        Args:
            miner_hotkey: miner hotkey to slash from

        Returns:
            float: amount to slash
        """
        try:
            if drawdown is None:
                # Get current drawdown percentage
                drawdown = self._position_client.compute_realtime_drawdown(miner_hotkey)

            # Get current balance
            current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
            if current_balance_theta is None or current_balance_theta <= 0:
                bt.logging.warning(f"No collateral balance for {miner_hotkey}")
                return 0.0

            # Calculate slash amount (based on drawdown percentage)
            drawdown_proportion = 1 - ((drawdown - ValiConfig.MAX_TOTAL_DRAWDOWN) / (
                    1 - ValiConfig.MAX_TOTAL_DRAWDOWN))  # scales x% drawdown to 100% of collateral
            slash_proportion = min(1.0,
                                   drawdown_proportion * ValiConfig.DRAWDOWN_SLASH_PROPORTION)  # cap slashed proportion at 100%
            slash_amount = current_balance_theta * slash_proportion

            bt.logging.info(f"Computed slashing for {miner_hotkey}: "
                            f"Drawdown: {drawdown}, "
                            f"Slash: {slash_proportion} = {slash_amount} Theta")

            return slash_amount

        except Exception as e:
            bt.logging.error(f"Failed to compute slash amount for {miner_hotkey}: {e}")
            return 0.0

    def slash_miner_collateral_proportion(self, miner_hotkey: str, slash_proportion: float = None) -> bool:
        """
        Slash miner's collateral by a proportion
        """
        if not self.is_mothership:
            return False
        current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
        if current_balance_theta is None or current_balance_theta <= 0:
            bt.logging.info(f"No slashing available for {miner_hotkey}, balance is {current_balance_theta}")
            return False

        if slash_proportion is None:
            # slash based on current drawdown
            slash_amount = self.compute_slash_amount(miner_hotkey)
        else:
            slash_amount = current_balance_theta * slash_proportion
        return self.slash_miner_collateral(miner_hotkey, slash_amount)

    def slash_miner_collateral(self, miner_hotkey: str, slash_amount: float = None) -> bool:
        """
        Slash miner's collateral by a raw theta amount

        Args:
            miner_hotkey: miner hotkey to slash from
        """
        if not self.is_mothership:
            return False
        current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
        if current_balance_theta is None or current_balance_theta <= 0:
            bt.logging.info(f"No slashing available for {miner_hotkey}, balance is {current_balance_theta}")
            return False

        if slash_amount is None:
            slash_amount = self.compute_slash_amount(miner_hotkey)

        # Ensure we don't slash more than the current balance
        slash_amount = min(slash_amount, current_balance_theta)
        # Limit slashing to max theta
        slash_amount = min(slash_amount, self.max_theta)
        if slash_amount <= 0:
            bt.logging.info(f"No slashing required for {miner_hotkey} (calculated amount: {slash_amount})")
            return True

        # Call collateral SDK slash method
        try:
            bt.logging.info(f"Processing slash of {slash_amount} Theta from {miner_hotkey}")
            owner_address = self.get_secret("collateral_owner_address")
            owner_private_key = self.get_secret("collateral_owner_private_key")
            try:
                self.collateral_manager.slash(
                    address=miner_hotkey,
                    amount=int(slash_amount * 10 ** 9),
                    owner_address=owner_address,
                    owner_private_key=owner_private_key,
                )
            finally:
                del owner_address
                del owner_private_key
            bt.logging.info(f"Successfully slashed {slash_amount} Theta from {miner_hotkey}")
            return True

        except Exception as e:
            bt.logging.error(f"Failed to execute slashing for {miner_hotkey}: {e}")
            return False

    def get_miner_collateral_balance(self, miner_address: str, max_retries: int = 4) -> Optional[float]:
        """
        Get a miner's current collateral balance in theta tokens.

        Args:
            miner_address (str): Miner's SS58 address
            max_retries (int): Maximum number of retry attempts

        Returns:
            Optional[float]: Balance in theta tokens, or None if error
        """
        # Return test data in unit test mode (data injection pattern from polygon_data_service.py)
        test_balance_rao = self._get_test_collateral_balance(miner_address)
        if test_balance_rao is not None:
            return self.to_theta(test_balance_rao)

        for attempt in range(max_retries):
            try:
                rao_balance = self.collateral_manager.balance_of(miner_address)
                return self.to_theta(rao_balance)
            except Exception as e:
                # Check if this is a rate limiting error (429)
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s
                    bt.logging.warning(
                        f"Rate limited getting balance for {miner_address}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    bt.logging.error(f"Failed to get collateral balance for {miner_address}: {e}")
                    return None
        return None

    def get_total_collateral(self) -> int:
        """Get total collateral in the contract in theta."""
        try:
            return self.collateral_manager.get_total_collateral()
        except Exception as e:
            bt.logging.error(f"Failed to get total collateral: {e}")
            return 0

    def get_slashed_collateral(self) -> int:
        """Get total slashed collateral in theta."""
        try:
            return self.collateral_manager.get_slashed_collateral()
        except Exception as e:
            bt.logging.error(f"Failed to get slashed collateral: {e}")
            return 0

    def set_miner_account_size(self, hotkey: str, timestamp_ms: int = None) -> bool:
        """
        Set the account size for a miner. Saves to memory and disk.
        Records are kept in chronological order.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp for the record (defaults to now)
        """
        # Get collateral balance outside lock (external RPC call)
        collateral_balance = self.get_miner_collateral_balance(hotkey)
        if collateral_balance is None:
            bt.logging.warning(f"Could not retrieve collateral balance for {hotkey}")
            return False

        # CRITICAL SECTION: Acquire lock for timestamp + record creation + append + save
        # Timestamp MUST be generated inside lock to ensure chronological ordering
        with self._account_sizes_lock:
            # Generate timestamp inside lock if not provided
            # This ensures records are added in strictly chronological order
            if timestamp_ms is None:
                timestamp_ms = TimeUtil.now_in_millis()

            account_size = min(ValiConfig.MAX_COLLATERAL_BALANCE_THETA, collateral_balance) * ValiConfig.COST_PER_THETA
            collateral_record = CollateralRecord(account_size, collateral_balance, timestamp_ms)
            # Skip if the new record matches the last existing record
            if hotkey in self.miner_account_sizes and self.miner_account_sizes[hotkey]:
                last_record = self.miner_account_sizes[hotkey][-1]
                if (last_record.account_size == collateral_record.account_size and
                        last_record.account_size_theta == collateral_record.account_size_theta):
                    bt.logging.info(f"Skipping save for {hotkey} - new record matches last record")
                    return True

            if hotkey not in self.miner_account_sizes:
                self.miner_account_sizes[hotkey] = []

            # Add the new record, IPC dict requires reassignment of entire k, v pair
            self.miner_account_sizes[hotkey] = self.miner_account_sizes[hotkey] + [collateral_record]

            # Save to disk (still inside account_sizes_lock, but _save will acquire _disk_lock)
            self._save_miner_account_sizes_to_disk()

        # Broadcast OUTSIDE lock to avoid holding lock during network I/O
        if self.is_mothership:
            self._broadcast_collateral_record_update_to_validators(hotkey, collateral_record)

        if hasattr(account_size, '_mock_name'):  # It's a mock
            bt.logging.info(
                f"Updated account size for {hotkey}: ${account_size} (valid from {collateral_record.valid_date_str})")
        else:
            bt.logging.info(
                f"Updated account size for {hotkey}: ${account_size:,.2f} (valid from {collateral_record.valid_date_str})")
        return True

    def get_miner_account_size(self, hotkey: str, timestamp_ms: int = None, most_recent: bool = False,
                               records_dict: dict = None, use_account_floor: bool = False) -> float | None:
        """
        Get the account size for a miner at a given timestamp. Iterate list in reverse chronological order, and return
        the first record whose valid_date_timestamp <= start_of_day_ms

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp to query for (defaults to now)
            most_recent: If True, return most recent record regardless of timestamp
            records_dict: Optional dict to use instead of self.miner_account_sizes (for cached lookups)
            use_account_floor: If True, return MIN_CAPITAL instead of None when no records exist

        Returns:
            Account size in USD, or None if no applicable records (or MIN_CAPITAL if use_account_floor=True)
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        # Use provided records_dict or default to self.miner_account_sizes
        # If using external dict, assume caller handles locking
        # If using self.miner_account_sizes, acquire lock
        if records_dict is not None:
            source_records = records_dict
            lock_needed = False
        else:
            source_records = self.miner_account_sizes
            lock_needed = True

        def _get_account_size_locked():
            """Inner function with the actual logic"""
            if hotkey not in source_records or not source_records[hotkey]:
                # Use account floor if requested (for miners without collateral records)
                return ValiConfig.MIN_CAPITAL if use_account_floor else None

            # Get start of the requested day
            start_of_day_ms = int(
                datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .timestamp() * 1000
            )

            # Return most recent record
            if most_recent:
                most_recent_record = source_records[hotkey][-1]
                return most_recent_record.account_size

            # Iterate in reversed order, and return the first record that is valid for or before the requested day
            for record in reversed(source_records[hotkey]):
                if record.valid_date_timestamp <= start_of_day_ms:
                    return record.account_size

            # No applicable records found - use account floor if requested
            return ValiConfig.MIN_CAPITAL if use_account_floor else None

        # Execute with or without lock depending on source
        if lock_needed:
            with self._account_sizes_lock:
                return _get_account_size_locked()
        else:
            return _get_account_size_locked()

    def get_all_miner_account_sizes(self, miner_account_sizes: dict[str, List[CollateralRecord]] = None,
                                    timestamp_ms: int = None) -> dict[str, float]:
        """
        Return a dict of all miner account sizes at a timestamp_ms
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()

        # If external dict provided, use it directly (caller handles locking)
        if miner_account_sizes is not None:
            all_miner_account_sizes = {}
            for hotkey in miner_account_sizes.keys():
                all_miner_account_sizes[hotkey] = self.get_miner_account_size(
                    hotkey, timestamp_ms=timestamp_ms, records_dict=miner_account_sizes
                )
            return all_miner_account_sizes

        # Using self.miner_account_sizes - must prevent race conditions
        # Copy the ENTIRE dict (not just keys) while holding lock to prevent iterator invalidation
        # This prevents sync_miner_account_sizes_data() from clearing the dict while we're reading it
        with self._account_sizes_lock:
            # Deep copy: create new dict with shallow copies of record lists
            # We don't need deep copy of CollateralRecord objects (they're immutable)
            miner_account_sizes_snapshot = {
                hotkey: list(records)  # Shallow copy of list
                for hotkey, records in self.miner_account_sizes.items()
            }

        # Now work with the snapshot (no lock needed - we own this copy)
        all_miner_account_sizes = {}
        for hotkey in miner_account_sizes_snapshot.keys():
            all_miner_account_sizes[hotkey] = self.get_miner_account_size(
                hotkey, timestamp_ms=timestamp_ms, records_dict=miner_account_sizes_snapshot
            )
        return all_miner_account_sizes

    @staticmethod
    def min_collateral_penalty(collateral: float) -> float:
        """
        Penalize miners who do not reach the min collateral
        """
        if collateral >= ValiConfig.MIN_COLLATERAL_VALUE:
            return 1
        return 0.01

    def _broadcast_collateral_record_update_to_validators(self, hotkey: str, collateral_record: CollateralRecord):
        """
        Broadcast CollateralRecord synapse to other validators using shared broadcast base.
        """
        def create_collateral_synapse():
            """Factory function to create the CollateralRecord synapse."""
            collateral_record_data = {
                "hotkey": hotkey,
                "account_size": collateral_record.account_size,
                "account_size_theta": collateral_record.account_size_theta,
                "update_time_ms": collateral_record.update_time_ms
            }
            return template.protocol.CollateralRecord(
                collateral_record=collateral_record_data
            )

        # Use shared broadcast method from base class
        self._broadcast_to_validators(
            synapse_factory=create_collateral_synapse,
            broadcast_name="CollateralRecord",
            context={"hotkey": hotkey}
        )

    def receive_collateral_record_update(self, collateral_record_data: dict, sender_hotkey: str = None) -> bool:
        """
        Process an incoming CollateralRecord synapse and update miner_account_sizes.

        Args:
            collateral_record_data: Dictionary containing hotkey, account_size, update_time_ms, valid_date_timestamp
            sender_hotkey: The hotkey of the validator that sent this broadcast

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # SECURITY: Verify sender using shared base class method
            if not self.verify_broadcast_sender(sender_hotkey, "CollateralRecord"):
                return False

            with self._account_sizes_lock:
                # Extract data from the synapse
                hotkey = collateral_record_data.get("hotkey")
                account_size = collateral_record_data.get("account_size")
                account_size_theta = collateral_record_data.get("account_size_theta")
                update_time_ms = collateral_record_data.get("update_time_ms")
                bt.logging.info(f"Processing collateral record update for miner {hotkey}")

                if not all([hotkey, account_size is not None, update_time_ms]):
                    bt.logging.warning(f"Invalid collateral record data received: {collateral_record_data}")
                    return False

                # Create a CollateralRecord object
                collateral_record = CollateralRecord(account_size, account_size_theta, update_time_ms)

                # Update miner account sizes
                if hotkey not in self.miner_account_sizes:
                    self.miner_account_sizes[hotkey] = []

                # Check if we already have this record (avoid duplicates)
                if self.get_miner_account_size(hotkey, most_recent=True) == account_size:
                    bt.logging.debug(f"Most recent collateral record for {hotkey} already exists")
                    return True

                # Add the new record, IPC dict requires reassignment of entire k, v pair
                self.miner_account_sizes[hotkey] = self.miner_account_sizes[hotkey] + [collateral_record]

                # Save to disk
                self._save_miner_account_sizes_to_disk()

                bt.logging.info(
                    f"Updated miner account size for {hotkey}: ${account_size} (valid from {collateral_record.valid_date_str})")
                return True

        except Exception as e:
            bt.logging.error(f"Error processing collateral record update: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False

    def verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """
        Verify that a coldkey owns a specific hotkey using subtensor.

        Args:
            coldkey_ss58: The coldkey SS58 address
            hotkey_ss58: The hotkey SS58 address to verify ownership of

        Returns:
            bool: True if coldkey owns the hotkey, False otherwise
        """
        try:
            subtensor_api = self.collateral_manager.subtensor_api
            coldkey_owner = subtensor_api.queries.query_subtensor("Owner", None, [hotkey_ss58])
            return coldkey_owner == coldkey_ss58
        except Exception as e:
            bt.logging.error(f"Error verifying coldkey-hotkey ownership: {e}")
            return False

    # ==================== Test Data Injection Methods ====================

    def set_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """
        Test-only method to inject collateral balances for specific miners.
        Only works when running_unit_tests=True for safety.

        This follows the same pattern as polygon_data_service.py's set_test_price_source().
        Allows tests to inject mock collateral balances without making blockchain calls.

        Args:
            miner_hotkey: Miner's hotkey (SS58 address)
            balance_rao: Collateral balance in rao units (int)
        """
        if not self.running_unit_tests:
            raise RuntimeError("set_test_collateral_balance can only be used in unit test mode")

        # Acquire lock to prevent concurrent modifications (race condition fix)
        with self._test_balances_lock:
            self._test_collateral_balances[miner_hotkey] = balance_rao

    def queue_test_collateral_balance(self, miner_hotkey: str, balance_rao: int) -> None:
        """
        Test-only method to queue a collateral balance for a miner.
        Multiple balances can be queued and will be consumed in FIFO order.
        Only works when running_unit_tests=True for safety.

        This is useful for race condition tests where multiple concurrent operations
        need different balances for the same miner.

        Args:
            miner_hotkey: Miner's hotkey (SS58 address)
            balance_rao: Collateral balance in rao units (int) to add to queue
        """
        if not self.running_unit_tests:
            raise RuntimeError("queue_test_collateral_balance can only be used in unit test mode")

        # Acquire lock to prevent concurrent modifications (race condition fix)
        with self._test_balances_lock:
            if miner_hotkey not in self._test_collateral_balance_queues:
                self._test_collateral_balance_queues[miner_hotkey] = []
            self._test_collateral_balance_queues[miner_hotkey].append(balance_rao)

    def clear_test_collateral_balances(self) -> None:
        """Clear all test collateral balances and queues (for test isolation)."""
        if not self.running_unit_tests:
            return

        # Acquire lock to prevent concurrent access (race condition fix)
        with self._test_balances_lock:
            self._test_collateral_balances.clear()
            self._test_collateral_balance_queues.clear()

    def _get_test_collateral_balance(self, miner_hotkey: str) -> Optional[int]:
        """
        Helper method to get test collateral balance for a miner.
        Returns None if not in unit test mode or if no test balance registered.

        Checks the queue first (for race condition tests), then falls back to direct balance.

        Args:
            miner_hotkey: Miner's hotkey (SS58 address)

        Returns:
            Balance in rao (int) if in test mode and registered, None otherwise
        """
        if not self.running_unit_tests:
            return None

        # Acquire lock to prevent concurrent access (race condition fix)
        with self._test_balances_lock:
            # Check if there's a queued balance (for race condition tests)
            if miner_hotkey in self._test_collateral_balance_queues:
                queue = self._test_collateral_balance_queues[miner_hotkey]
                if queue:
                    # Pop from front of queue (FIFO)
                    return queue.pop(0)

            # Fall back to direct balance lookup
            return self._test_collateral_balances.get(miner_hotkey)