# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ValidatorContractManager - Business logic for contract/collateral management.

This manager handles all collateral operations including:
- Deposit/withdrawal processing
- Slashing calculations
- Collateral record broadcasting

The manager contains NO RPC infrastructure - that lives in ContractServer.
This is pure business logic that can be tested independently.
"""
import threading
import bittensor as bt
from collateral_sdk import CollateralManager, Network
from typing import Dict, Any, Optional, List
import time
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.validator_broadcast_base import ValidatorBroadcastBase
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
import template.protocol
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from entity_management.entity_client import EntityClient
from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient


# ==================== Constants ====================

TARGET_MS = 1782338400000


# ==================== Manager Implementation ====================

class ValidatorContractManager(ValidatorBroadcastBase):
    """
    Business logic for contract/collateral management.

    This manager contains ALL business logic for:
    - Deposit/withdrawal processing
    - Slashing calculations based on drawdown
    - Collateral record broadcasting to validators

    Account size tracking is delegated to MinerAccountManager via MinerAccountClient.

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
        - MinerAccountClient

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

        # MinerAccountClient for account size operations
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode)

        # EntityClient for checking entity subaccounts during withdrawals
        self._entity_client = EntityClient(connection_mode=connection_mode, connect_immediately=False)

        # EntityCollateralClient for required collateral checks during withdrawals
        self._entity_collateral_client = EntityCollateralClient(connection_mode=connection_mode, connect_immediately=False)

        self._challenge_period_client = ChallengePeriodClient(connection_mode=connection_mode)
        self._elimination_client = EliminationClient(connection_mode=connection_mode)

        # Lock for test collateral balances dict (prevents concurrent modifications in tests)
        self._test_balances_lock = threading.Lock()
        # Lock for coldkey-hotkey ownership cache (prevents concurrent modifications)
        self._coldkey_hotkey_cache_lock = threading.Lock()

        # Initialize collateral manager based on network type
        if self.is_testnet:
            bt.logging.info("Using testnet collateral manager")
            self.collateral_manager = CollateralManager(Network.TESTNET)
        else:
            bt.logging.info("Using mainnet collateral manager")
            self.collateral_manager = CollateralManager(Network.MAINNET)

        # GCP secret manager
        self._gcp_secret_manager_client = None

        # Test collateral balance registry (only used when running_unit_tests=True)
        # Allows tests to inject specific collateral balances instead of making blockchain calls
        # Key: miner_hotkey -> Value: balance in rao (int)
        self._test_collateral_balances: Dict[str, int] = {}

        # Test collateral balance queue (only used when running_unit_tests=True)
        # Allows tests to inject a sequence of balances for the same miner
        # Key: miner_hotkey -> Value: list of balances (FIFO queue)
        # This is needed for race condition tests that simulate multiple concurrent balance changes
        self._test_collateral_balance_queues: Dict[str, list] = {}

        # Coldkey-hotkey ownership cache
        # Key: (coldkey_ss58, hotkey_ss58) -> Value: is_owner (bool)
        # Cached permanently in memory (cleared only on restart)
        self._coldkey_hotkey_cache: Dict[tuple[str, str], bool] = {}

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

        miners_to_reinstate = {}
        for miner, amount in miners_to_reinstate.items():
            self.force_deposit(amount, miner)

        update_thread = threading.Thread(target=self.refresh_miner_account_sizes, daemon=True)
        update_thread.start()
        bt.logging.info("Miner account size refresh started in background thread")

    def refresh_miner_account_sizes(self):
        """
        refresh miner account sizes
        """
        # Let the orchestrator finish bringing up the miner_account RPC server before the first call.
        time.sleep(10)
        hotkeys = []
        update_count = 0
        for hotkey in hotkeys:
            try:
                prev_acct_size = self._miner_account_client.get_miner_account_size(hotkey)
                bt.logging.info(f"Current account size for {hotkey}: ${prev_acct_size:,.2f}")
                self._set_miner_account_size(hotkey)
                update_count += 1
                time.sleep(0.5)
            except Exception as e:
                bt.logging.error(f"Failed to update account size for {hotkey}: {e}")
        bt.logging.info(f"Account size refresh completed for {update_count} miners")

    def health_check(self) -> dict:
        """Health check for monitoring."""
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
        }

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

                # # Check collateral balance limit before processing
                # try:
                #     current_balance_theta = self.to_theta(self.collateral_manager.balance_of(miner_hotkey))
                #
                #     if current_balance_theta + deposit_amount_theta > self.max_theta:
                #         error_msg = (f"Deposit would exceed maximum balance limit. "
                #                      f"Current: {current_balance_theta:.2f} Theta, "
                #                      f"Deposit: {deposit_amount_theta:.2f} Theta, "
                #                      f"Limit: {self.max_theta} Theta")
                #         bt.logging.warning(error_msg)
                #         return {
                #             "successfully_processed": False,
                #             "error_message": error_msg
                #         }
                #
                # except Exception as e:
                #     bt.logging.error(f"Failed to check balance limit: {e}")
                #     return {
                #         "successfully_processed": False,
                #         "error_message": e
                #     }

                # # All positions must be closed before a miner can deposit or withdraw
                # if len(self.position_manager.get_positions_for_one_hotkey(miner_hotkey, only_open_positions=True)) > 0:
                #     return {
                #         "successfully_processed": False,
                #         "error_message": "Miner has open positions, please close all positions before depositing or withdrawing collateral"
                #     }

                bt.logging.info(f"Processing deposit for: {deposit_amount_theta} Theta to miner: {miner_hotkey}")
                owner_address = ValiUtils.get_secret("collateral_owner_address")
                owner_private_key = ValiUtils.get_secret("collateral_owner_private_key")
                vault_password = ValiUtils.get_secret("gcp_vali_pw_name")
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

                deposited_theta = self.to_theta(deposited_balance.rao)
                msg = f"Deposit successful: {deposited_theta} Theta deposited to miner: {miner_hotkey}"
                bt.logging.info(msg)
                self._set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
                self._entity_collateral_client.offset_collateral_cache(miner_hotkey, deposited_theta)
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
            owner_address = ValiUtils.get_secret("collateral_owner_address")
            owner_private_key = ValiUtils.get_secret("collateral_owner_private_key")
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
            raise

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
            bt.logging.info("Received withdrawal query {miner_hotkey} withdraw amount: {amount:.4f}")

            # Check collateral balance
            theta_current_balance = self.get_miner_collateral_balance(miner_hotkey)
            if theta_current_balance is None:
                error_msg = f"Failed to retrieve collateral balance for {miner_hotkey}"
                bt.logging.error(error_msg)
                return {"successfully_processed": False, "error_message": error_msg}

            if amount > theta_current_balance:
                error_msg = f"Insufficient collateral balance. Available: {theta_current_balance}, Requested: {amount}"
                bt.logging.error(error_msg)
                return {"successfully_processed": False, "error_message": error_msg}

            required_min_theta = self._entity_collateral_client.compute_entity_required_collateral(miner_hotkey)
            if theta_current_balance - amount < required_min_theta:
                error_msg = f"Insufficient collateral: {theta_current_balance - amount:.2f} theta after withdrawal < {required_min_theta:.2f} theta required"
                bt.logging.error(f"{error_msg}")
                return {"successfully_processed": False, "error_message": error_msg}

            positions = self._position_client.get_positions_for_one_hotkey(miner_hotkey)
            open_positions = [pos for pos in positions if pos.is_open_position]
            if open_positions:
                error_msg = (
                    f"Cannot withdraw collateral with open positions, please close all positions before withdrawing collateral. "
                    f"Open positions: {[pos.trade_pair.trade_pair_id for pos in open_positions]}"
                )
                bt.logging.error(error_msg)
                return {"successfully_processed": False, "error_message": error_msg}

            slashed_amount = 0
            drawdown = 0
            elimination = self._elimination_client.get_elimination(miner_hotkey)
            if not elimination or not elimination["collateral_slashed"]:
                perf_ledger = self._perf_ledger_client.get_perf_ledger_for_hotkey(miner_hotkey)
                self._miner_account_client.rebuild_account_state_from_positions(miner_hotkey, positions)
                account = self._miner_account_client.get_account(miner_hotkey)

                max_return = 1.0
                if perf_ledger:
                    ledger = perf_ledger[miner_hotkey]
                    max_return = max(max(cp.equity_ret for cp in ledger.cps), 1.0) if ledger.cps else 1.0
                elif account:
                    max_return = account.get("max_return", 1.0)

                if account:
                    current_return = account["balance"] / account["account_size"]
                    drawdown = 1.0 - current_return / max_return

                # penalty free withdrawals down to MAX_COLLATERAL_BALANCE_THETA
                # Prioritize withdrawing penalty free theta
                penalty_free_amount = max(0.0, theta_current_balance - self.max_theta)
                penalty_amount = max(0.0, amount - penalty_free_amount)

                # max drawdown closer to eod drawdown
                bucket = self._challenge_period_client.get_miner_bucket(miner_hotkey)
                if bucket is None and elimination:
                    bucket_at_elimination = elimination["bucket_at_elimination"]
                    if bucket_at_elimination:
                        bucket = MinerBucket(bucket_at_elimination)

                drawdown_threshold = bucket.eod_drawdown_threshold() if bucket and bucket.is_active else (1 - ValiConfig.MAX_TOTAL_DRAWDOWN)

                # Don't slash penalty_free theta on withdrawal
                slashed_amount = penalty_amount * min(max(drawdown, 0) / drawdown_threshold, 1.0)

            withdrawal_amount = amount - slashed_amount
            new_balance = theta_current_balance - amount

            result = {
                "successfully_processed": True,
                "error_message": "",
                "drawdown": drawdown,
                "slashed_amount": slashed_amount,
                "withdrawal_amount": withdrawal_amount,
                "new_balance": new_balance
            }
            bt.logging.info(f"{miner_hotkey} Query withdrawal request results: {result}")
            return result

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

            query_result = self.query_withdrawal_request(amount, miner_hotkey)
            if not query_result["successfully_processed"]:
                return query_result
            withdrawal_amount = query_result["withdrawal_amount"]
            slashed_amount = query_result["slashed_amount"]
            drawdown = query_result["drawdown"]

            bt.logging.info(
                f"Processing withdrawal request from {miner_hotkey} for {amount} Theta. Current drawdown: {drawdown*100}%. {slashed_amount} Theta will be slashed. {withdrawal_amount} Theta will be withdrawn.")
            if slashed_amount > 0:
                self.slash_miner_collateral(miner_hotkey, slashed_amount)

            owner_address = ValiUtils.get_secret("collateral_owner_address")
            owner_private_key = ValiUtils.get_secret("collateral_owner_private_key")
            vault_password = ValiUtils.get_secret("gcp_vali_pw_name")
            try:
                withdrawn_balance = self.collateral_manager.withdraw(
                    amount=int(withdrawal_amount * 10 ** 9),  # convert theta to rao_theta
                    source_coldkey=miner_coldkey,
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
            returned_theta = self.to_theta(withdrawn_balance.rao)
            msg = f"Withdrawal successful: {returned_theta} Theta withdrawn for {miner_hotkey}, returned to {miner_coldkey}"
            bt.logging.info(msg)
            self._set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
            self._entity_collateral_client.offset_collateral_cache(miner_hotkey, -returned_theta)
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

    def slash_miner_collateral_proportion(self, miner_hotkey: str, slash_proportion: float) -> bool:
        """
        Slash miner's collateral by a proportion (1.0 for 100%)
        """
        if not self.is_mothership:
            return False

        if not (0.0 <= slash_proportion <= 1.0):
            bt.logging.error(f"Invalid collateral slash proportion: {slash_proportion}")
            return False

        current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
        if current_balance_theta is None or current_balance_theta <= 0:
            bt.logging.info(f"No slashing available for {miner_hotkey}, balance is {current_balance_theta}")
            return False

        slash_amount = min(current_balance_theta, self.max_theta) * slash_proportion
        return self.slash_miner_collateral(miner_hotkey, slash_amount)

    def slash_miner_collateral(self, miner_hotkey: str, slash_amount: float) -> bool:
        """
        Slash miner's collateral by a raw theta amount

        Args:
            miner_hotkey: miner hotkey to slash from
        """
        if not self.is_mothership:
            return False

        if slash_amount is None or slash_amount < 0:
            bt.logging.error(f"Invalid collateral slash amount: {slash_amount}")
            return False

        current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
        if current_balance_theta is None or current_balance_theta <= 0:
            bt.logging.info(f"No slashing available for {miner_hotkey}, balance is {current_balance_theta}")
            return False

        # Ensure we don't slash more than the current balance or max theta
        slash_amount = min(slash_amount, current_balance_theta, self.max_theta)
        if slash_amount <= 0:
            bt.logging.info(f"No slashing required for {miner_hotkey} (calculated amount: {slash_amount})")
            return True

        # Call collateral SDK slash method
        bt.logging.info(f"Processing slash of {slash_amount} Theta from {miner_hotkey}")
        slash_amount_rao = int(slash_amount * 10 ** 9)
        owner_address = ValiUtils.get_secret("collateral_owner_address")
        owner_private_key = ValiUtils.get_secret("collateral_owner_private_key")
        vault_password = ValiUtils.get_secret("gcp_vali_pw_name")
        try:
            try:
                self.collateral_manager.slash(
                    address=miner_hotkey,
                    amount=slash_amount_rao,
                    owner_address=owner_address,
                    owner_private_key=owner_private_key,
                )
            except Exception as e:
                bt.logging.error(f"Failed to execute slashing for {miner_hotkey}: {e}")
                return False

            bt.logging.info(f"Successfully slashed {slash_amount} Theta from {miner_hotkey}")

            try:
                self.collateral_manager.burn(
                    amount=slash_amount_rao,
                    vault_stake=self.wallet.hotkey.ss58_address,
                    vault_wallet=self.wallet,
                    wallet_password=vault_password,
                )
                bt.logging.info(f"Successfully burned {slash_amount} Theta for {miner_hotkey} slash")
            except Exception as e:
                bt.logging.error(
                    f"Slash succeeded but burn failed for {miner_hotkey} - "
                    f"{slash_amount} Theta funds remain in slashedCollateral pool: {e}"
                )

            return True
        finally:
            del owner_address
            del owner_private_key
            del vault_password

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

    def _set_miner_account_size(self, hotkey: str, timestamp_ms: int = None, account_size: float = None) -> bool:
        """
        Set the account size for a miner by fetching collateral balance and updating via MinerAccountClient.

        Args:
            hotkey: Miner's hotkey (SS58 address)
            timestamp_ms: Timestamp for the record (defaults to now)
            account_size: Optional explicit account size in USD. If not provided, calculated from collateral balance.

        Returns:
            bool: True if successful, False otherwise
        """
        if account_size is None:
            # Get collateral balance outside lock (external RPC call)
            collateral_balance = self.get_miner_collateral_balance(hotkey)
            if collateral_balance is None:
                bt.logging.warning(f"Could not retrieve collateral balance for {hotkey}")
                return False
        else:
            # Subaccount miner
            cpt = ValiConfig.ENTITY_COST_PER_THETA_LOW if account_size <= ValiConfig.ENTITY_COST_PER_THETA_LOW_THRESHOLD else ValiConfig.ENTITY_COST_PER_THETA
            collateral_balance = account_size / cpt

        if account_size is None:
            account_size = min(ValiConfig.MAX_COLLATERAL_BALANCE_THETA, collateral_balance) * ValiConfig.COST_PER_THETA

        # Update account size via MinerAccountClient - returns CollateralRecord dict if successful
        collateral_record_dict = self._miner_account_client.set_miner_account_size(hotkey, collateral_balance, timestamp_ms, account_size)

        # Broadcast to other validators if mothership
        if collateral_record_dict and self.is_mothership:
            self._broadcast_collateral_record_update_to_validators(hotkey, collateral_record_dict)

        return collateral_record_dict is not None

    @staticmethod
    def min_collateral_penalty(collateral: float) -> float:
        """
        Penalize miners who do not reach the min collateral
        """
        if collateral >= ValiConfig.MIN_COLLATERAL_VALUE:
            return 1
        return 0.01

    def _broadcast_collateral_record_update_to_validators(self, hotkey: str, collateral_record_dict: dict):
        """
        Broadcast CollateralRecord synapse to other validators using shared broadcast base.

        Args:
            hotkey: Miner's hotkey
            collateral_record_dict: CollateralRecord as dict with keys:
                account_size, account_size_theta, update_time_ms, valid_date_timestamp
        """
        def create_collateral_synapse():
            """Factory function to create the CollateralRecord synapse."""
            collateral_record_data = {
                "hotkey": hotkey,
                "account_size": collateral_record_dict["account_size"],
                "account_size_theta": collateral_record_dict["account_size_theta"],
                "update_time_ms": collateral_record_dict["update_time_ms"]
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

    def verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """
        Verify that a coldkey owns a specific hotkey.
        Uses metagraph first (fast), falls back to subtensor query (cached).
        Results are cached in memory for the lifetime of the validator process.

        Args:
            coldkey_ss58: The coldkey SS58 address
            hotkey_ss58: The hotkey SS58 address to verify ownership of

        Returns:
            bool: True if coldkey owns the hotkey, False otherwise
        """
        cache_key = (coldkey_ss58, hotkey_ss58)

        # 1. Check in-memory cache first
        with self._coldkey_hotkey_cache_lock:
            if cache_key in self._coldkey_hotkey_cache:
                bt.logging.info(f"Using cached ownership result for {hotkey_ss58}")
                return self._coldkey_hotkey_cache[cache_key]

        # 2. Try metagraph (fast - already in memory, no blockchain query)
        try:
            neurons = self._metagraph_client.get_neurons()
            for neuron in neurons:
                if neuron.hotkey == hotkey_ss58 and neuron.coldkey == coldkey_ss58:
                    # Cache the result
                    with self._coldkey_hotkey_cache_lock:
                        self._coldkey_hotkey_cache[cache_key] = True
                    bt.logging.info(f"Verified ownership via metagraph for {coldkey_ss58} and {hotkey_ss58}")
                    return True
        except Exception as e:
            bt.logging.warning(f"Failed to check metagraph for {hotkey_ss58}: {e}")

        # 3. Fallback to subtensor for non-registered hotkeys
        try:
            bt.logging.info(f"Hotkey {hotkey_ss58} not in metagraph, querying subtensor")
            subtensor_api = self.collateral_manager.subtensor_api
            coldkey_owner = subtensor_api.queries.query_subtensor("Owner", None, [hotkey_ss58])
            is_owner = coldkey_owner == coldkey_ss58

            # Cache the result
            with self._coldkey_hotkey_cache_lock:
                self._coldkey_hotkey_cache[cache_key] = is_owner

            bt.logging.info(f"Verified ownership via subtensor for {coldkey_ss58} and {hotkey_ss58}")
            return is_owner
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
