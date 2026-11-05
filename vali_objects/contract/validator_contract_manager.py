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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collateral_sdk import CollateralManager, Network
from typing import Dict, Any, Optional
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
from vali_objects.miner_account.account_size_lock import AccountSizeLock
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.miner_account.standard_miner import (StandardMinerAccount, OpeningRejected, open_account,
                                                          opening_bucket_thresholds, closing_slash_proportion,
                                                          elimination_slash_proportion)
from vali_objects.enums.elimination_reason_enum import EliminationReason
from shared_objects.rpc.rpc_client_base import RPCCallTimeoutError
from shared_objects.subtensor_ops.metagraph_utils import is_validator_stake_weight
from entity_management.entity_client import EntityClient
from entity_management.entity_utils import is_synthetic_hotkey
from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient
from shared_objects.log import logger


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
        # One account opening at a time per hotkey
        self._account_locks: Dict[str, threading.Lock] = {}
        self._account_locks_guard = threading.Lock()

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
        # A standard account trades all markets: the opening sets the asset selection
        self._asset_selection_client = AssetSelectionClient(connection_mode=connection_mode)

        # Lock for test collateral balances dict (prevents concurrent modifications in tests)
        self._test_balances_lock = threading.Lock()
        # Lock for coldkey-hotkey ownership cache (prevents concurrent modifications)
        self._coldkey_hotkey_cache_lock = threading.Lock()

        # Ownership chain-lookup bounds (seconds); see verify_coldkey_owns_hotkey.
        self.OWNERSHIP_QUERY_TIMEOUT_S = 10.0
        self.OWNERSHIP_FAILURE_COOLDOWN_S = 30.0

        # Initialize collateral manager based on network type
        if self.is_testnet:
            logger.info("Using testnet collateral manager")
            self.collateral_manager = CollateralManager(Network.TESTNET)
        else:
            logger.info("Using mainnet collateral manager")
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

        # Bounded on-chain ownership lookups. query_subtensor has no timeout of
        # its own, so a slow/hung chain endpoint would pin the calling
        # (REST-facing) thread indefinitely; the query runs on this small
        # executor and the caller waits at most OWNERSHIP_QUERY_TIMEOUT_S.
        # After a failure, retries for the same key are short-circuited for
        # OWNERSHIP_FAILURE_COOLDOWN_S so a down chain can't stack up threads.
        self._ownership_query_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="owner-query")
        self._ownership_failure_ts: Dict[tuple[str, str], float] = {}

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
        logger.info("Miner account size refresh started in background thread")

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
                logger.info(f"Current account size for {hotkey}: ${prev_acct_size:,.2f}")
                self._set_miner_account_size(hotkey)
                update_count += 1
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to update account size for {hotkey}: {e}")
        logger.info(f"Account size refresh completed for {update_count} miners")

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
            logger.info("Received deposit request")
            # Decode and validate the extrinsic
            try:
                encoded_extrinsic = bytes.fromhex(extrinsic_hex)
                extrinsic = self.collateral_manager.decode_extrinsic(encoded_extrinsic)
                logger.info("Extrinsic decoded successfully")
            except Exception as e:
                error_msg = f"Invalid extrinsic data: {str(e)}"
                logger.error(error_msg)
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
                #         logger.warning(error_msg)
                #         return {
                #             "successfully_processed": False,
                #             "error_message": error_msg
                #         }
                #
                # except Exception as e:
                #     logger.error(f"Failed to check balance limit: {e}")
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

                logger.info(f"Processing deposit for: {deposit_amount_theta} Theta to miner: {miner_hotkey}")
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
                logger.info(msg)
                self._set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
                self._entity_collateral_client.offset_collateral_cache(miner_hotkey, deposited_theta)
                return {
                    "successfully_processed": True,
                    "error_message": ""
                }

            except Exception as e:
                error_msg = f"Deposit execution failed: {str(e)}"
                logger.error(error_msg)
                if "miner_hotkey" in locals():
                    try:
                        self._set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
                    except Exception as refresh_error:
                        logger.error(f"Failed to refresh account size for {miner_hotkey} after deposit failure: {refresh_error}")
                return {
                    "successfully_processed": False,
                    "error_message": error_msg
                }

        except Exception as e:
            error_msg = f"Deposit processing error: {str(e)}"
            logger.error(error_msg)
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
            logger.info(f"Processing force deposit to {miner_hotkey} for {amount} Theta")
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
            logger.info(f"Force deposit successful: {amount} Theta deposited for {miner_hotkey}")
        except Exception as e:
            logger.error(f"Force deposit execution failed: {str(e)}")
            raise

    def refresh_miner_account_size(self, hotkey: str) -> bool:
        """
        Refresh a miner's cached account size from their current on-chain collateral balance.
        Read-only -- makes no on-chain transaction.

        Args:
            hotkey (str): Miner's SS58 hotkey

        Returns:
            bool: True if successful, False otherwise
        """
        return self._set_miner_account_size(hotkey, TimeUtil.now_in_millis())

    def open_standard_miner_account(self, hotkey: str, leverage_tier: int) -> tuple[bool, str, Optional[dict]]:
        """Open the account a hotkey registered after the standard miner cutoff block trades on, sized from its
        collateral at today's price (standard_miner.open_account), and broadcast the record; only the mothership does.
        Returns (opened, message for the miner, the record as a dict); an opened account is returned as it is, completed
        and broadcast again only when a step had been left missing."""
        if not self.running_unit_tests and not self.is_mothership:
            return False, "Account openings are only processed by the mothership validator", None
        with self._account_lock(hotkey):
            try:
                record, already_opened = self._standard_miner_opening(hotkey, leverage_tier)
            except OpeningRejected as e:
                logger.info(f"Account opening for {hotkey} rejected: {e}")
                return False, str(e), None
            repeat = already_opened and self._challenge_period_client.has_miner(hotkey)
            self._trade_all_markets(hotkey)
            record = self._miner_account_client.open_standard_account(hotkey, record)
            if not repeat:
                # Sent as soon as the record exists: a failure in the state step costs the miner a retry, not the
                # peers their copy of the record
                self._broadcast_standard_miner_opening(hotkey, record)
            self._add_opening_state(hotkey, record)
        if repeat:
            return True, "This hotkey already opened its account", record.to_dict()
        message = (f"Opened a ${record.account_size:,.2f} account at tier {record.opening_leverage_tier}, locking "
                   f"{record.locked_theta:.4f} theta at ${record.opening_price_usd:,.4f} per theta")
        logger.info(f"{hotkey}: {message}")
        return True, message, record.to_dict()

    def _account_lock(self, hotkey: str) -> threading.Lock:
        """One account's opening and withdrawals run one at a time."""
        with self._account_locks_guard:
            return self._account_locks.setdefault(hotkey, threading.Lock())

    def _trade_all_markets(self, hotkey: str) -> None:
        """A standard account trades all markets, as a standard subaccount does, whatever the hotkey selected before."""
        if self._asset_selection_client.get_asset_selection(hotkey) == MinerAssetClass.ALL_MARKETS:
            return
        result = self._asset_selection_client.process_asset_selection_request(
            MinerAssetClass.ALL_MARKETS.value, hotkey, overwrite=True)
        if not result.get("successfully_processed"):
            raise RuntimeError(f"Could not move {hotkey} onto all markets: {result.get('error_message')}")

    def _standard_miner_opening(self, hotkey: str, leverage_tier: int) -> tuple[StandardMinerAccount, bool]:
        """The record an opening earns and whether it existed, or OpeningRejected: only a post-cutoff hotkey with collateral
        that is not eliminated, a validator, an entity or trading on the existing rules opens (an open account stays)."""
        after_cutoff = self._metagraph_client.is_registered_after_cutoff(hotkey)
        if after_cutoff is None:
            raise OpeningRejected("This hotkey is not in the last metagraph sync; try again shortly")
        if after_cutoff is False:
            raise OpeningRejected("Only a hotkey registered after the cutoff block opens an account this way; this "
                                  "one trades under the existing rules")
        if self._elimination_client.is_hotkey_eliminated(hotkey):
            raise OpeningRejected("An eliminated hotkey cannot open a trading account")
        account = self._miner_account_client.get_account(hotkey)
        existing = account.standard_account if account else None
        if existing is not None:
            if existing.is_closed:
                raise OpeningRejected("This account is closed and cannot reopen")
            return existing, True
        hotkeys, stake_weights = self._metagraph_client.get_hotkeys(), self._metagraph_client.get_stake_weight()
        index = hotkeys.index(hotkey) if hotkey in hotkeys else -1
        if 0 <= index < len(stake_weights) and is_validator_stake_weight(stake_weights[index]):
            raise OpeningRejected("A validator hotkey cannot open a trading account")
        if self._entity_client.get_entity_data(hotkey) is not None:
            raise OpeningRejected("An entity hotkey cannot open a trading account")
        if self._challenge_period_client.has_miner(hotkey):
            raise OpeningRejected("This hotkey already trades under the existing rules and cannot open an account")
        collateral_theta = self.get_miner_collateral_balance(hotkey)
        if collateral_theta is None:
            raise OpeningRejected("The collateral balance could not be read; try again shortly")
        from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
        try:
            price_usd = DebtBasedScoring._convert_alpha_to_usd(1.0, self._metagraph_client)
        except Exception as e:
            logger.warning(f"Theta price unavailable for the opening of {hotkey}: {e}")
            price_usd = 0.0
        record = open_account(collateral_theta, price_usd, leverage_tier, MinerAssetClass.ALL_MARKETS.value,
                              TimeUtil.now_in_millis())
        return record, False

    def apply_standard_miner_opening(self, hotkey: str, record: StandardMinerAccount) -> StandardMinerAccount:
        """Put an opened account in place: the record, the opening tier and the bucket on the account in one step,
        the SUBACCOUNT_CHALLENGE state unless one exists or the account is closed, and all markets."""
        record = self._miner_account_client.open_standard_account(hotkey, record)
        self._add_opening_state(hotkey, record)
        try:
            self._trade_all_markets(hotkey)
        except Exception as e:
            # The mothership's own selection broadcast, or the nightly sync, still brings it
            logger.error(f"Could not move {hotkey} onto all markets on its opening: {e}")
        return record

    def _add_opening_state(self, hotkey: str, record: StandardMinerAccount) -> None:
        if record.is_closed or self._challenge_period_client.has_miner(hotkey):
            return
        self._challenge_period_client.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_CHALLENGE, record.opened_at_ms,
                                                       **opening_bucket_thresholds())

    def _broadcast_standard_miner_opening(self, hotkey: str, record: StandardMinerAccount) -> None:
        def create_synapse():
            return template.protocol.StandardMinerOpening(opening={"hotkey": hotkey, "record": record.to_dict()})

        self._broadcast_to_validators(synapse_factory=create_synapse, broadcast_name="StandardMinerOpening",
                                      context={"hotkey": hotkey})

    def receive_standard_miner_opening_update(self, opening_data: dict, sender_hotkey: str = None) -> bool:
        """Apply an account opening the mothership broadcast, as it is: nothing is recomputed, a record already held
        stays and no second challenge period state is made (apply_standard_miner_opening)."""
        def apply(data: dict):
            hotkey, record_data = data.get("hotkey"), data.get("record")
            if not hotkey or not isinstance(record_data, dict):
                raise ValueError(f"Invalid account opening data: {data}")
            self.apply_standard_miner_opening(hotkey, StandardMinerAccount.from_dict(record_data))

        return self._handle_incoming_broadcast(synapse_data=opening_data, sender_hotkey=sender_hotkey,
                                               broadcast_name="StandardMinerOpening", update_callback=apply)

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
            logger.info(f"Received withdrawal query {miner_hotkey} withdraw amount: {amount:.4f}")

            # Check collateral balance
            theta_current_balance = self.get_miner_collateral_balance(miner_hotkey)
            if theta_current_balance is None:
                error_msg = f"Failed to retrieve collateral balance for {miner_hotkey}"
                logger.error(error_msg)
                return {"successfully_processed": False, "error_message": error_msg}

            if amount > theta_current_balance:
                error_msg = f"Insufficient collateral balance. Available: {theta_current_balance}, Requested: {amount}"
                logger.error(error_msg)
                return {"successfully_processed": False, "error_message": error_msg}

            record = self._miner_account_client.get_standard_account(miner_hotkey)
            if record is not None:
                # The elimination row is the word on a standard miner's account: none means it is open; one whose
                # slash is still owed takes it now; one slashed leaves the rest to the path below, which slashes nothing
                elimination = self._elimination_client.get_elimination(miner_hotkey)
                result = None
                if elimination is None:
                    result = self._query_locked_withdrawal(amount, miner_hotkey, record.locked_theta, record.is_closed,
                                                           theta_current_balance)
                elif not elimination.get("collateral_slashed", True):
                    result = self._query_owed_slash_withdrawal(miner_hotkey, record, theta_current_balance, elimination)
                if result is not None:
                    logger.info(f"{miner_hotkey} Query withdrawal request results: {result}")
                    return result
            elif not is_synthetic_hotkey(miner_hotkey):
                # A miner on the existing rules with a fixed size withdraws as a standard miner does while it is open;
                # once eliminated, the path below applies as for every other miner
                size_lock = getattr(self._miner_account_client.get_account(miner_hotkey), "size_lock", None)
                if isinstance(size_lock, AccountSizeLock) and self._elimination_client.get_elimination(miner_hotkey) is None:
                    result = self._query_locked_withdrawal(amount, miner_hotkey, size_lock.locked_theta,
                                                           size_lock.is_closed, theta_current_balance)
                    logger.info(f"{miner_hotkey} Query withdrawal request results: {result}")
                    return result

            required_min_theta = self._entity_collateral_client.compute_entity_required_collateral(miner_hotkey)
            if theta_current_balance - amount < required_min_theta:
                error_msg = f"Insufficient collateral: {theta_current_balance - amount:.2f} theta after withdrawal < {required_min_theta:.2f} theta required"
                logger.error(f"{error_msg}")
                return {"successfully_processed": False, "error_message": error_msg}

            positions = self._position_client.get_positions_for_one_hotkey(miner_hotkey)
            open_positions = [pos for pos in positions if pos.is_open_position]
            if open_positions:
                error_msg = (
                    f"Cannot withdraw collateral with open positions, please close all positions before withdrawing collateral. "
                    f"Open positions: {[pos.trade_pair.trade_pair_id for pos in open_positions]}"
                )
                logger.error(error_msg)
                return {"successfully_processed": False, "error_message": error_msg}

            slashed_amount = 0
            drawdown = 0
            elimination = self._elimination_client.get_elimination(miner_hotkey)
            if not elimination or not elimination["collateral_slashed"]:
                perf_ledger = self._perf_ledger_client.get_perf_ledger_for_hotkey(miner_hotkey)
                account = self._miner_account_client.get_account(miner_hotkey)

                max_return = 1.0
                if perf_ledger:
                    ledger = perf_ledger[miner_hotkey]
                    max_return = max(max(cp.equity_ret for cp in ledger.cps), 1.0) if ledger.cps else 1.0
                elif account:
                    max_return = account.max_return

                if account:
                    current_return = account.balance / account.account_size
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
            logger.info(f"{miner_hotkey} Query withdrawal request results: {result}")
            return result

        except Exception as e:
            error_msg = f"Withdrawal query error: {str(e)}"
            logger.error(error_msg)
            return {
                "successfully_processed": False,
                "error_message": error_msg
            }

    def _open_positions_error(self, hotkey: str, what: str) -> Optional[str]:
        positions = self._position_client.get_positions_for_one_hotkey(hotkey)
        open_positions = [pos for pos in positions if pos.is_open_position]
        if not open_positions:
            return None
        error_msg = f"{what}; close all positions first. Open positions: {[pos.trade_pair.trade_pair_id for pos in open_positions]}"
        logger.error(error_msg)
        return error_msg

    def _query_locked_withdrawal(self, amount: float, hotkey: str, locked_theta: float, is_closed: bool,
                                 balance: float) -> Dict[str, Any]:
        """Collateral above the locked theta leaves without a slash, whatever the positions. A withdrawal into the
        locked theta closes the account: every position closed, then the whole balance less the closing slash."""
        result = {"successfully_processed": True, "error_message": "", "locked_theta": locked_theta}
        if not is_closed and balance - amount >= locked_theta - 1e-9:
            return {**result, "closes_account": False, "drawdown": 0.0, "slashed_amount": 0.0,
                    "withdrawal_amount": amount, "new_balance": balance - amount}

        error_msg = self._open_positions_error(
            hotkey, f"Withdrawing {amount:.4f} theta goes into the {locked_theta:.4f} theta locked by the account "
                    f"and closes it")
        if error_msg:
            return {"successfully_processed": False, "error_message": error_msg}

        # An account closed without its elimination row has the close recorded again, never slashed again: its slash
        # was taken by the close that was cut short, or is owed through the row
        drawdown = {} if is_closed else dict(self._challenge_period_client.get_drawdown_stats(hotkey) or {})
        if drawdown:
            # The stats are a refresh pass old; with every position closed the balance is the equity right now
            account = self._miner_account_client.get_account(hotkey)
            if account is not None and account.get_account_size():
                current = account.balance / account.get_account_size()
                day_open = drawdown.get("daily_open_balance") or drawdown.get("daily_open_equity")
                if day_open:
                    drawdown["intraday_drawdown_pct"] = (1.0 - current / day_open) * 100.0
                if drawdown.get("eod_hwm"):
                    drawdown["trailing_drawdown_pct"] = (1.0 - current / drawdown["eod_hwm"]) * 100.0
        proportion = closing_slash_proportion(drawdown)
        intraday_pct = max(0.0, drawdown.get("intraday_drawdown_pct") or 0.0)
        trailing_pct = max(0.0, drawdown.get("trailing_drawdown_pct") or 0.0)
        slashed_amount = min(locked_theta * proportion, balance)
        return {**result, "closes_account": True, "drawdown": max(intraday_pct, trailing_pct) / 100.0,
                "intraday_drawdown_pct": intraday_pct, "trailing_drawdown_pct": trailing_pct,
                "slash_proportion": proportion, "slashed_amount": slashed_amount,
                "withdrawal_amount": balance - slashed_amount, "new_balance": 0.0}

    def _query_owed_slash_withdrawal(self, hotkey: str, record: StandardMinerAccount, balance: float,
                                     elimination: Dict[str, Any]) -> Dict[str, Any]:
        """An eliminated account whose slash failed at the time: the share its reason takes of the locked theta, on the
        drawdown the row recorded, comes out of this withdrawal and the rest of the balance with it."""
        error_msg = self._open_positions_error(hotkey, "The account is eliminated")
        if error_msg:
            return {"successfully_processed": False, "error_message": error_msg}
        limits = self._challenge_period_client.get_drawdown_stats(hotkey) or {}
        drawdown = {"intraday_drawdown_pct": elimination.get("intraday_dd"),
                    "trailing_drawdown_pct": elimination.get("eod_dd"),
                    "intraday_drawdown_threshold": limits.get("intraday_drawdown_threshold"),
                    "eod_drawdown_threshold": limits.get("eod_drawdown_threshold")}
        proportion = elimination_slash_proportion(
            EliminationReason(elimination["reason"]), elimination.get("elimination_drawdown_pct"), drawdown)
        slashed_amount = min(record.locked_theta * proportion, balance)
        return {"successfully_processed": True, "error_message": "", "locked_theta": record.locked_theta,
                "closes_account": False, "owed_slash": True,
                "drawdown": (elimination.get("elimination_drawdown_pct") or 0.0) / 100.0,
                "slash_proportion": proportion, "slashed_amount": slashed_amount,
                "withdrawal_amount": balance - slashed_amount, "new_balance": 0.0}

    def _close_account(self, hotkey: str, query_result: Dict[str, Any]) -> tuple[Optional[str], float]:
        """Close the account through its elimination (slash, record, account bucket, row) and return (error, the
        balance left to pay out), which is nothing unless the row says closed and slashed and the chain shows it."""
        try:
            self._elimination_client.append_elimination_row(
                hotkey=hotkey,
                reason=EliminationReason.ACCOUNT_CLOSED,
                elimination_drawdown_pct=query_result["drawdown"] * 100.0,
                intraday_drawdown_pct=query_result.get("intraday_drawdown_pct"),
                eod_drawdown_pct=query_result.get("trailing_drawdown_pct"),
                elimination_time_ms=TimeUtil.now_in_millis(),
                slash_theta=query_result["slashed_amount"],
            )
        except RPCCallTimeoutError as e:
            logger.error(f"The closing of the account of {hotkey} timed out; outcome unknown: {e}")
            return "The closing of the account is still being processed; check the account and withdraw again", 0.0
        row = self._elimination_client.get_elimination(hotkey)
        if row is None:
            return "The closing of the account was not recorded; the account stays open, withdraw again", 0.0
        if row["reason"] != EliminationReason.ACCOUNT_CLOSED.value:
            return (f"The account was eliminated ({row['reason']}) before this withdrawal closed it; "
                    f"withdraw again"), 0.0
        if not row.get("collateral_slashed", True):
            return "The closing slash is still owed; withdraw again", 0.0
        balance = self.get_miner_collateral_balance(hotkey)
        if balance is None:
            return "The account is closed but its balance could not be read; withdraw again", 0.0
        if balance > query_result["withdrawal_amount"] + 1e-9:
            return "The account is closed and its slash is still settling; withdraw again", 0.0
        return None, balance

    def _roll_back_slash(self, slashed_amount: float, miner_hotkey: str, failure: Exception) -> None:
        """A slash that ran for real is restored when the step after it failed, so a retry is not processed against an
        already-reduced balance. Non-mothership slashes are dry runs, so there is nothing to roll back."""
        if slashed_amount <= 0 or not self.is_mothership:
            return
        try:
            self.force_deposit(slashed_amount, miner_hotkey)
            logger.info(f"Rolled back {slashed_amount} Theta slash for {miner_hotkey} after a failure: {failure}")
        except Exception as rollback_error:
            logger.error(
                f"CRITICAL: failed to roll back {slashed_amount} Theta slash for {miner_hotkey} after a failure "
                f"({failure}); manual reinstatement required: {rollback_error}")

    def process_withdrawal_request(self, amount: float, miner_coldkey: str, miner_hotkey: str,
                                   close_account: bool = False) -> Dict[str, Any]:
        """
        Process a collateral withdrawal request, and slash proportionally.

        Args:
            amount (float): Amount to withdraw in theta tokens
            miner_coldkey (str): Miner's SS58 wallet coldkey address to return collateral to
            miner_hotkey (str): Miner's SS58 hotkey
            close_account (bool): The miner's consent to a withdrawal that closes an account with locked collateral

        Returns:
            Dict[str, Any]: Result of withdrawal operation
        """
        try:
            logger.info("Received withdrawal request")
            with self._account_lock(miner_hotkey):
                query_result = self.query_withdrawal_request(amount, miner_hotkey)
                if not query_result["successfully_processed"]:
                    return query_result
                return self._withdraw(miner_coldkey, miner_hotkey, query_result, close_account)
        except Exception as e:
            error_msg = f"Withdrawal processing execution failed: {str(e)}"
            logger.error(error_msg)
            try:
                self._set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
            except Exception as refresh_error:
                logger.error(f"Failed to refresh account size for {miner_hotkey} after withdrawal failure: {refresh_error}")
            return {
                "successfully_processed": False,
                "error_message": error_msg,
                "returned_amount": 0.0,
                "returned_to": ""
            }

    def _withdraw(self, miner_coldkey: str, miner_hotkey: str, query_result: Dict[str, Any],
                  close_account: bool = False) -> Dict[str, Any]:
        """Carry out a withdrawal the query allowed: the slash, or the elimination that closes the account, then the
        transfer. Caller holds the account's lock."""
        withdrawal_amount = query_result["withdrawal_amount"]
        slashed_amount = query_result["slashed_amount"]
        drawdown = query_result["drawdown"]
        closes_account = query_result.get("closes_account", False)

        logger.info(
            f"Processing withdrawal request from {miner_hotkey}. Current drawdown: {drawdown*100}%. {slashed_amount} Theta will be slashed. {withdrawal_amount} Theta will be withdrawn.")
        if closes_account:
            if not close_account:
                error_msg = (f"Withdrawing into the {query_result['locked_theta']:.4f} Theta locked by the account closes "
                             f"it for good, paying out {withdrawal_amount:.4f} Theta after a slash of "
                             f"{slashed_amount:.4f} Theta; send close_account true to confirm")
                logger.warning(f"{error_msg}: {miner_hotkey}")
                return {**query_result, "successfully_processed": False, "error_message": error_msg,
                        "returned_amount": 0.0, "returned_to": ""}
            if not self.running_unit_tests and not self.is_mothership:
                error_msg = "A withdrawal that closes the account is only processed by the mothership validator"
                logger.error(f"{error_msg}: {miner_hotkey}")
                return {"successfully_processed": False, "error_message": error_msg, "returned_amount": 0.0, "returned_to": ""}
            error_msg, withdrawal_amount = self._close_account(miner_hotkey, query_result)
            if error_msg:
                logger.error(f"{error_msg}: {miner_hotkey}")
                return {"successfully_processed": False, "error_message": error_msg, "returned_amount": 0.0, "returned_to": ""}
        slashed = slashed_amount > 0 and not closes_account and self.slash_miner_collateral(miner_hotkey, slashed_amount)
        if query_result.get("owed_slash") and slashed_amount > 0 and not slashed:
            error_msg = f"The slash of {slashed_amount} Theta owed by the account failed; nothing was withdrawn"
            logger.error(f"{error_msg}: {miner_hotkey}")
            return {"successfully_processed": False, "error_message": error_msg, "returned_amount": 0.0, "returned_to": ""}

        returned_theta = 0.0
        if withdrawal_amount > 0:
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
            except Exception as withdraw_error:
                # A closed account keeps its slash: what is left stays withdrawable as a closed account
                if slashed:
                    self._roll_back_slash(slashed_amount, miner_hotkey, withdraw_error)
                raise
            finally:
                del owner_address
                del owner_private_key
                del vault_password
            returned_theta = self.to_theta(withdrawn_balance.rao)
        if query_result.get("owed_slash") and slashed:
            try:
                if not self._elimination_client.set_collateral_slashed(miner_hotkey, True):
                    raise RuntimeError("no elimination row")
            except Exception as e:
                logger.error(f"The slash owed by {miner_hotkey} was taken but its elimination row could not be updated: {e}")
        msg = f"Withdrawal successful: {returned_theta} Theta withdrawn for {miner_hotkey}, returned to {miner_coldkey}"
        logger.info(msg)
        self._set_miner_account_size(miner_hotkey, TimeUtil.now_in_millis())
        self._entity_collateral_client.offset_collateral_cache(miner_hotkey, -returned_theta)
        return {
            "successfully_processed": True,
            "error_message": "",
            "returned_amount": returned_theta,
            "returned_to": miner_coldkey,
            "account_closed": closes_account,
        }

    def slash_miner_collateral_proportion(self, miner_hotkey: str, slash_proportion: float) -> bool:
        """
        Slash miner's collateral by a proportion (1.0 for 100%)
        """
        if not (0.0 <= slash_proportion <= 1.0):
            logger.error(f"Invalid collateral slash proportion: {slash_proportion}")
            return False

        current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
        if current_balance_theta is None or current_balance_theta <= 0:
            logger.info(f"No slashing available for {miner_hotkey}, balance is {current_balance_theta}")
            return False

        slash_amount = min(current_balance_theta, self.max_theta) * slash_proportion
        return self.slash_miner_collateral(miner_hotkey, slash_amount)

    def slash_miner_collateral(self, miner_hotkey: str, slash_amount: float) -> bool:
        """
        Slash miner's collateral by a raw theta amount

        Args:
            miner_hotkey: miner hotkey to slash from

        Non-mothership validators only log the slash and burn they would perform and return True,
        so callers proceed as they do on the mothership.
        """
        if slash_amount is None or slash_amount < 0:
            logger.error(f"Invalid collateral slash amount: {slash_amount}")
            return False

        current_balance_theta = self.get_miner_collateral_balance(miner_hotkey)
        if current_balance_theta is None or current_balance_theta <= 0:
            logger.info(f"No slashing available for {miner_hotkey}, balance is {current_balance_theta}")
            return False

        # Ensure we don't slash more than the current balance or max theta
        slash_amount = min(slash_amount, current_balance_theta, self.max_theta)
        if slash_amount <= 0:
            logger.info(f"No slashing required for {miner_hotkey} (calculated amount: {slash_amount})")
            return True

        if not self.is_mothership:
            logger.info(
                f"[DRY RUN] Non-mothership: would slash {slash_amount} Theta and burn "
                f"{slash_amount * ValiConfig.SLASH_BURN_PROPORTION} Theta from {miner_hotkey}"
            )
            return True

        # Call collateral SDK slash method
        logger.info(f"Processing slash of {slash_amount} Theta from {miner_hotkey}")
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
                logger.error(f"Failed to execute slashing for {miner_hotkey}: {e}")
                return False

            logger.info(f"Successfully slashed {slash_amount} Theta from {miner_hotkey}")

            # Only burn a portion of the slash; the remainder stays in the vault stake
            burn_amount = slash_amount * ValiConfig.SLASH_BURN_PROPORTION
            burn_amount_rao = int(slash_amount_rao * ValiConfig.SLASH_BURN_PROPORTION)
            try:
                self.collateral_manager.burn(
                    amount=burn_amount_rao,
                    vault_stake=self.wallet.hotkey.ss58_address,
                    vault_wallet=self.wallet,
                    wallet_password=vault_password,
                )
                logger.info(f"Successfully burned {burn_amount} of {slash_amount} slashed Theta for {miner_hotkey}")
            except Exception as e:
                logger.error(
                    f"Slash succeeded but burn failed for {miner_hotkey} - "
                    f"{burn_amount} Theta to burn remains in slashedCollateral pool: {e}"
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
                    logger.warning(
                        f"Rate limited getting balance for {miner_address}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to get collateral balance for {miner_address}: {e}")
                    return None
        return None

    def get_total_collateral(self) -> int:
        """Get total collateral in the contract in theta."""
        try:
            return self.collateral_manager.get_total_collateral()
        except Exception as e:
            logger.error(f"Failed to get total collateral: {e}")
            return 0

    def get_slashed_collateral(self) -> int:
        """Get total slashed collateral in theta."""
        try:
            return self.collateral_manager.get_slashed_collateral()
        except Exception as e:
            logger.error(f"Failed to get slashed collateral: {e}")
            return 0

    def _set_miner_account_size(self, hotkey: str, timestamp_ms: int | None = None, account_size: float | None = None) -> bool:
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
                logger.warning(f"Could not retrieve collateral balance for {hotkey}")
                return False
        else:
            # Subaccount miner
            cpt = ValiConfig.std_reg_cpt(account_size)
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
                logger.info(f"Using cached ownership result for {hotkey_ss58}")
                return self._coldkey_hotkey_cache[cache_key]

        # 2. Try metagraph (fast - already in memory, no blockchain query)
        try:
            neurons = self._metagraph_client.get_neurons()
            for neuron in neurons:
                if neuron.hotkey == hotkey_ss58 and neuron.coldkey == coldkey_ss58:
                    # Cache the result
                    with self._coldkey_hotkey_cache_lock:
                        self._coldkey_hotkey_cache[cache_key] = True
                    logger.info(f"Verified ownership via metagraph for {coldkey_ss58} and {hotkey_ss58}")
                    return True
        except Exception as e:
            logger.warning(f"Failed to check metagraph for {hotkey_ss58}: {e}")

        # 3. Fallback to subtensor for non-registered hotkeys — bounded (see __init__).
        with self._coldkey_hotkey_cache_lock:
            failed_at = self._ownership_failure_ts.get(cache_key, 0.0)
            in_cooldown = time.time() - failed_at < self.OWNERSHIP_FAILURE_COOLDOWN_S
        if in_cooldown:
            logger.warning(
                f"Ownership query for {hotkey_ss58} in failure cooldown; refusing without chain query")
            return False

        try:
            logger.info(f"Hotkey {hotkey_ss58} not in metagraph, querying subtensor")
            future = self._ownership_query_executor.submit(self._query_owner_on_chain, hotkey_ss58)
            coldkey_owner = future.result(timeout=self.OWNERSHIP_QUERY_TIMEOUT_S)
            is_owner = coldkey_owner == coldkey_ss58

            # Cache the result (definitive answers only — both True and False)
            with self._coldkey_hotkey_cache_lock:
                self._coldkey_hotkey_cache[cache_key] = is_owner
                self._ownership_failure_ts.pop(cache_key, None)

            logger.info(f"Verified ownership via subtensor for {coldkey_ss58} and {hotkey_ss58}")
            return is_owner
        except FuturesTimeoutError:
            # cancel() only prevents QUEUED work from starting; a query that is
            # already running keeps its worker parked until the chain call
            # returns or the socket dies. The timeout is client-side only —
            # it frees THIS caller, it does not abort the chain request.
            future.cancel()
            logger.error(
                f"Ownership query for {hotkey_ss58} exceeded {self.OWNERSHIP_QUERY_TIMEOUT_S}s; "
                f"failing closed with {self.OWNERSHIP_FAILURE_COOLDOWN_S}s cooldown")
            with self._coldkey_hotkey_cache_lock:
                self._ownership_failure_ts[cache_key] = time.time()
            return False
        except Exception as e:
            logger.error(f"Error verifying coldkey-hotkey ownership: {e}")
            with self._coldkey_hotkey_cache_lock:
                self._ownership_failure_ts[cache_key] = time.time()
            return False

    def _query_owner_on_chain(self, hotkey_ss58: str):
        """Blocking chain query; runs on the bounded ownership executor."""
        subtensor_api = self.collateral_manager.subtensor_api
        return subtensor_api.queries.query_subtensor("Owner", None, [hotkey_ss58])

    def shutdown_ownership_executor(self):
        """Release the bounded ownership executor (operator/teardown hook).

        In production this manager lives inside the ContractServer process,
        which the orchestrator terminates with the process — a parked worker's
        chain socket dies with it, so leaked-thread warnings at clean interpreter
        exit are not a practical concern. This hook exists for explicit
        teardown paths and tests.
        """
        self._ownership_query_executor.shutdown(wait=False, cancel_futures=True)

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
