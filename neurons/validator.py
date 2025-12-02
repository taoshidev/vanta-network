# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import json
import os
import sys
import threading
import signal

from vanta_api.api_manager import APIManager


import template
import traceback
import time
import bittensor as bt

from typing import Tuple
from setproctitle import setproctitle
from neurons.validator_base import ValidatorBase
from shared_objects.metagraph_server import MetagraphServer, MetagraphClient
from enum import Enum
from runnable.core_outputs_server import CoreOutputsServer
from runnable.miner_statistics_server import MinerStatisticsServer
from template.protocol import SendSignal
from vali_objects.utils.asset_selection_manager import ASSET_CLASS_SELECTION_TIME_MS
from vali_objects.utils.asset_selection_client import AssetSelectionClient
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.utils.auto_sync import PositionSyncer
from vali_objects.utils.p2p_syncer import P2PSyncer
from vali_objects.utils.elimination_client import EliminationClient
from vali_objects.utils.limit_order_server import LimitOrderServer, LimitOrderClient
from vali_objects.utils.market_order_manager import MarketOrderManager
from shared_objects.rate_limiter import RateLimiter
from vali_objects.utils.plagiarism_server import PlagiarismServer
from vali_objects.utils.position_lock_server import PositionLockServer
from vali_objects.uuid_tracker import UUIDTracker
from time_util.time_util import TimeUtil, timeme
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.metagraph_updater import MetagraphUpdater
from shared_objects.error_utils import ErrorUtils
from shared_objects.slack_notifier import SlackNotifier
from vali_objects.utils.elimination_server import EliminationServer
from vali_objects.utils.live_price_server import LivePriceFetcherClient, LivePriceFetcherServer
from vali_objects.utils.weight_calculator_server import WeightCalculatorServer
from vali_objects.utils.mdd_checker_server import MDDCheckerServer
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.debt_ledger_server import DebtLedgerServer
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger_server import PerfLedgerServer, PerfLedgerClient
from vali_objects.vali_dataclasses.debt_ledger_server import DebtLedgerClient
from vali_objects.utils.position_manager_client import PositionManagerClient
from vali_objects.utils.position_manager_server import PositionManagerServer
from vali_objects.utils.challengeperiod_server import ChallengePeriodServer
from vali_objects.utils.challengeperiod_client import ChallengePeriodClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.plagiarism_detector_server import PlagiarismDetectorServer
from vali_objects.utils.contract_server import ContractServer
from vali_objects.utils.asset_selection_server import AssetSelectionServer
from vali_objects.utils.order_processor import OrderProcessor
from shared_objects.common_data_server import CommonDataServer, CommonDataClient
from shared_objects.shutdown_coordinator import ShutdownCoordinator

def is_shutdown() -> bool:
    """Check if shutdown is in progress via ShutdownCoordinator."""
    return ShutdownCoordinator.is_shutdown()

# Enum class that represents the method associated with Synapse
class SynapseMethod(Enum):
    POSITION_INSPECTOR = "GetPositions"
    SIGNAL = "SendSignal"
    CHECKPOINT = "SendCheckpoint"

def signal_handler(signum, frame):
    # Check if already shutting down
    if is_shutdown():
        return

    if signum in (signal.SIGINT, signal.SIGTERM):
        signal_message = "Handling SIGINT" if signum == signal.SIGINT else "Handling SIGTERM"
        print(f"{signal_message} - Initiating graceful shutdown")

        # Signal shutdown via ShutdownCoordinator (propagates to all servers)
        ShutdownCoordinator.signal_shutdown(
            "SIGINT received" if signum == signal.SIGINT else "SIGTERM received"
        )
        print("Shutdown signal propagated to all servers via ShutdownCoordinator")

        # Set a 2-second alarm
        signal.alarm(2)

def alarm_handler(signum, frame):
    print("Graceful shutdown failed, force killing the process")
    sys.exit(1)  # Exit immediately

# Set up signal handling
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGALRM, alarm_handler)


class Validator(ValidatorBase):
    def __init__(self):
        setproctitle(f"vali_{self.__class__.__name__}")
        # Try to read the file meta/meta.json and print it out
        try:
            with open("meta/meta.json", "r") as f:
                bt.logging.info(f"Found meta.json file {f.read()}")
        except Exception as e:
            bt.logging.error(f"Error reading meta/meta.json: {e}")

        ValiBkpUtils.clear_tmp_dir()
        self.uuid_tracker = UUIDTracker()
        # Lock to stop new signals from being processed while a validator is restoring
        self.signal_sync_lock = threading.Lock()
        self.signal_sync_condition = threading.Condition(self.signal_sync_lock)
        self.n_orders_being_processed = [0]  # Allow this to be updated across threads by placing it in a list (mutable)

        self.config = self.get_config()
        # Use the getattr function to safely get the autosync attribute with a default of False if not found.
        self.auto_sync = getattr(self.config, 'autosync', False) and 'ms' not in ValiUtils.get_secrets()
        self.is_mainnet = self.config.netuid == 8
        # Ensure the directory for logging exists, else create one.
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

        self.secrets = ValiUtils.get_secrets()
        if self.secrets is None:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )

        # Initialize Bittensor wallet objects FIRST (needed for SlackNotifier)
        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} with autosync set to: {self.auto_sync} "
            f"on network: {self.config.subtensor.chain_endpoint} with config:"
        )

        # This logs the active configuration to the specified logging directory for review.
        bt.logging.info(self.config)

        # Initialize Bittensor miner objects
        # These classes are vital to interact and function within the Bittensor network.
        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        bt.logging.info("Initializing validator wallet...")
        wallet_start_time = time.time()
        self.wallet = bt.wallet(config=self.config)
        wallet_elapsed_s = time.time() - wallet_start_time
        bt.logging.success(f"Validator wallet initialized in {wallet_elapsed_s:.2f}s")

        # Initialize Slack notifier for error reporting
        # Created before LivePriceFetcher so it can be passed for crash notifications
        self.slack_notifier = SlackNotifier(
            hotkey=self.wallet.hotkey.ss58_address,
            webhook_url=getattr(self.config, 'slack_webhook_url', None),
            error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
            is_miner=False  # This is a validator
        )

        # Initialize ShutdownCoordinator singleton for graceful shutdown coordination
        # Uses shared memory for cross-process communication (no RPC needed)
        # This must be initialized before any RPC servers are created
        # Reset flag on attach to clear any stale shutdown state from crashed/killed processes
        ShutdownCoordinator.initialize(reset_on_attach=True)
        bt.logging.success("[INIT] ShutdownCoordinator initialized (shared memory)")

        # Spawn CommonDataServer in separate process
        self.common_data_server_handle = CommonDataServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False
        )

        # Create global CommonDataClient for auto-sync coordination
        _common_data_client = CommonDataClient(connect_immediately=True)

        # Spawn LivePriceFetcherServer in separate process
        # Server inherits from RPCServerBase for unified lifecycle management
        # Daemon monitors health; shutdown handled automatically via ShutdownCoordinator
        self.live_price_fetcher_handle = LivePriceFetcherServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=True,  # Enable daemon for health monitoring
            secrets=self.secrets,
            disable_ws=False
        )

        # Create lightweight LivePriceFetcher client (connects to server via RPC)
        self.price_fetcher_client = LivePriceFetcherClient(running_unit_tests=False)

        bt.logging.info(f"Wallet: {self.wallet}")

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
        # Uses RPC-based MetagraphServer with server-side hotkeys_set cache for O(1) has_hotkey() lookups
        # MetagraphServer inherits from RPCServerBase and exposes data via RPC for consumers
        self.metagraph_server_handle = MetagraphServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False  # No daemon needed for metagraph server
        )

        # Create MetagraphClient for RPC communication with MetagraphServer
        self.metagraph_client = MetagraphClient()

        # Create MetagraphUpdater with simple parameters (no PTNManager)
        # This will run in a thread in the main process
        # MetagraphUpdater now exposes RPC server for weight setting (validators only)
        # MetagraphUpdater creates its own LivePriceFetcherClient internally (forward compatibility)
        self.metagraph_updater = MetagraphUpdater(
            self.config, self.wallet.hotkey.ss58_address,
            False,
            slack_notifier=self.slack_notifier
        )
        self.subtensor = self.metagraph_updater.subtensor
        bt.logging.info(f"Subtensor: {self.subtensor}")


        # Start the metagraph updater and wait for initial population. CRITICAL: This must complete before EliminationManager starts.
        self.metagraph_updater_thread = self.metagraph_updater.start_and_wait_for_initial_update(
            max_wait_time=60,
            slack_notifier=self.slack_notifier
        )

        # Spawn ContractServer in separate process for collateral operations (uses RPC - no IPC overhead)
        # ContractServer creates its own MetagraphClient internally (forward compatibility)
        # Consumers create their own ContractClient instances (forward compatibility pattern)
        self.contract_server_handle = ContractServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False,  # No daemon needed for contract server
            config=self.config
        )


        self.position_manager_server_handle = PositionManagerServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False   # Defer daemon start, will use client
        )

        self.elimination_server_handle = EliminationServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False,   # Defer daemon start, will use client
            serve=self.config.serve
        )

        # Spawn AssetSelectionServer in separate process
        # AssetSelectionServer creates its own MetagraphClient internally (forward compatibility)
        # Consumers create their own AssetSelectionClient instances
        self.asset_selection_server_handle = AssetSelectionServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False,  # No daemon needed
            config=self.config
        )

        # Spawn PerfLedgerServer in separate process
        # PerfLedgerServer manages perf ledgers and exposes them via RPC
        self.perf_ledger_server_handle = PerfLedgerServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False  # Defer daemon start, will use client
        )

        self.position_syncer = PositionSyncer(
            signal_sync_lock=self.signal_sync_lock,
            signal_sync_condition=self.signal_sync_condition,
            n_orders_being_processed=self.n_orders_being_processed,
            auto_sync_enabled=self.auto_sync
        )

        # P2P syncer for checkpoint handling (deprecated but still needed for axon attachment)
        self.p2p_syncer = P2PSyncer(
            wallet=self.wallet,
            is_testnet=not self.is_mainnet,
            signal_sync_lock=self.signal_sync_lock,
            signal_sync_condition=self.signal_sync_condition,
            n_orders_being_processed=self.n_orders_being_processed,
            running_unit_tests=False
        )

        # Create PositionManagerClient for runtime client operations (forward compatibility)
        # This is the primary way to interact with PositionManagerServer
        self.position_manager_client = PositionManagerClient()
        bt.logging.info("[INIT] PositionManagerClient created (for runtime operations)")

        self.position_lock_server_handle = PositionLockServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False  # No daemon needed for lock service
        )

        self.plagiarism_server_handle = PlagiarismServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False  # No daemon needed currently (refresh on-demand)
        )

        # MarketOrderManager creates its own ContractClient internally (forward compatibility)
        self.market_order_manager = MarketOrderManager(self.config.serve, slack_notifier=self.slack_notifier)

        #force_validator_to_restore_from_checkpoint(self.wallet.hotkey.ss58_address, self.metagraph, self.config, self.secrets)

        # Spawn ChallengePeriodServer in separate process (daemon start deferred)
        self.challengeperiod_handle = ChallengePeriodServer.spawn_process(slack_notifier=self.slack_notifier, start_daemon=False)

        # Create client for RPC communication
        self.challengeperiod_client = ChallengePeriodClient()
        # EliminationClient with local cache for fast lookups (no RPC per order!)
        # Cache refreshes every 5 seconds in background thread
        self.elimination_client = EliminationClient(local_cache_refresh_period_ms=5000)
        # Start daemons via clients (multi-process architecture)
        self.position_manager_client.start_daemon()
        self.elimination_client.start_daemon()
        self.challengeperiod_client.start_daemon()

        self.limit_order_server_handle = LimitOrderServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=True
        )

        # Run pre-run setup on the server (order corrections, perf ledger wiping, etc.)
        # PositionManagerServer handles perf ledger wiping internally via PerfLedgerClient
        self.position_manager_client.pre_run_setup(perform_order_corrections=True)

        self.uuid_tracker.add_initial_uuids(self.position_manager_client.get_positions_for_all_miners())

        # Spawn DebtLedgerServer in separate process
        # DebtLedgerManager creates its own clients internally (forward compatibility):
        # - PerfLedgerClient, AssetSelectionClient, ContractClient
        self.debt_ledger_server_handle = DebtLedgerServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=False,  # Defer daemon start, will use client
            slack_webhook_url=self.config.slack_error_webhook_url,
            validator_hotkey=self.wallet.hotkey.ss58_address
        )


        # Create LimitOrderClient for validator's own use
        # Consumers (EliminationServer, PositionSyncer, etc.) create their own LimitOrderClient instances
        self.limit_order_client = LimitOrderClient()
        # AssetSelectionClient with local cache for fast lookups (no RPC per order!)
        # Cache refreshes every 5 seconds in background thread (replaces _asset_selections_cache)
        self.asset_selection_client = AssetSelectionClient(local_cache_refresh_period_ms=5000)


        self.checkpoint_lock = threading.Lock()
        self.encoded_checkpoint = ""
        self.last_checkpoint_time = 0

        bt.logging.info(f"Metagraph n_entries: {len(self.metagraph_client.get_hotkeys())}")
        if not self.metagraph_client.has_hotkey(self.wallet.hotkey.ss58_address):
            bt.logging.error(
                f"\nYour validator hotkey: {self.wallet.hotkey.ss58_address} (wallet: {self.wallet.name}, hotkey: {self.wallet.hotkey_str}) "
                f"is not registered to chain connection: {self.metagraph_updater.get_subtensor()} \n"
                f"Run btcli register and try again. "
            )
            exit()

        # Build and link vali functions to the axon.
        # The axon handles request processing, allowing validators to send this process requests.
        # ValidatorBase creates its own clients internally (forward compatibility):
        # - AssetSelectionClient, ContractClient
        super().__init__(wallet=self.wallet, slack_notifier=self.slack_notifier, config=self.config,
                         metagraph=self.metagraph_client, p2p_syncer=self.p2p_syncer,
                         asset_selection_client=self.asset_selection_client, subtensor=self.subtensor)

        self.order_rate_limiter = RateLimiter()
        self.position_inspector_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 4)
        self.checkpoint_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 60 * 6)


        self.plagiarism_detector_server_handle = PlagiarismDetectorServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=True
        )

        self.mdd_checker_server_handle = MDDCheckerServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=True    # Start daemon immediately (all dependencies ready)
        )

        # Step 5 & 6: RPC servers (EliminationServer and ChallengePeriodManagerServer)
        # Already started above with deferred initialization pattern
        # Both servers use RPCServerBase for unified RPC server and daemon management
        bt.logging.info("Step 5-6: EliminationServer and ChallengePeriodManagerServer RPC servers and daemons already started")

        self.core_outputs_server_handle = CoreOutputsServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=True
        )
        self.statistics_outputs_server_handle = MinerStatisticsServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=True
        )

        # Create clients for daemon management (multi-process architecture)
        self.perf_ledger_client = PerfLedgerClient()
        self.debt_ledger_client = DebtLedgerClient()

        # Start daemons via clients
        self.perf_ledger_client.start_daemon()
        self.debt_ledger_client.start_daemon()

        self.weight_calculator_server_handle = WeightCalculatorServer.spawn_process(
            slack_notifier=self.slack_notifier,
            start_daemon=True,
            config=self.config,
            hotkey=self.wallet.hotkey.ss58_address,
            is_mainnet=self.is_mainnet
        )

        # Start API services (if enabled)
        if self.config.serve:
            # Create API Manager with configuration options
            self.api_manager = APIManager(
                slack_webhook_url=self.config.slack_webhook_url,
                validator_hotkey=self.wallet.hotkey.ss58_address,
                api_host=self.config.api_host,
                api_rest_port=self.config.api_rest_port,
                api_ws_port=self.config.api_ws_port
            )

            # Start the API Manager in a separate thread
            self.api_thread = threading.Thread(target=self.api_manager.run, daemon=True)
            self.api_thread.start()
            # Verify thread started
            time.sleep(0.1)
            if not self.api_thread.is_alive():
                raise RuntimeError("API thread failed to start")
            bt.logging.info(
                f"API services thread started - REST: {self.config.api_host}:{self.config.api_rest_port}, "
                f"WebSocket: {self.config.api_host}:{self.config.api_ws_port}")
        else:
            self.api_thread = None
            bt.logging.info("API services not enabled - skipping")

        bt.logging.info("[INIT] All initialization steps completed successfully!")

        # Send success notification to Slack
        if self.slack_notifier:
            self.slack_notifier.send_message(
                f"✅ Validator Initialization Complete!\n"
                f"All initialization steps completed successfully\n"
                f"Hotkey: {self.wallet.hotkey.ss58_address}\n"
                f"API services: {'Enabled' if self.config.serve else 'Disabled'}",
                level="info"
            )

        # Validators on mainnet net to be syned for the first time or after interruption need to resync their
        # positions. Assert there are existing orders that occurred > 24hrs in the past. Assert that the newest order
        # was placed within 24 hours.
        if self.is_mainnet:
            n_positions_on_disk = self.position_manager_client.get_number_of_miners_with_any_positions()
            # Get extreme timestamps from all positions using client
            oldest_disk_ms, youngest_disk_ms = float("inf"), 0
            all_positions = self.position_manager_client.get_positions_for_all_miners()
            for hotkey, positions in all_positions.items():
                for p in positions:
                    for o in p.orders:
                        oldest_disk_ms = min(oldest_disk_ms, o.processed_ms)
                        youngest_disk_ms = max(youngest_disk_ms, o.processed_ms)
            if oldest_disk_ms == float("inf"):
                oldest_disk_ms = 0  # No positions found
            if (n_positions_on_disk > 0):
                bt.logging.info(f"Found {n_positions_on_disk} positions on disk."
                                f" Found oldest_disk_ms: {TimeUtil.millis_to_datetime(oldest_disk_ms)},"
                                f" youngest_disk_ms: {TimeUtil.millis_to_datetime(youngest_disk_ms)}")
            one_day_ago = TimeUtil.timestamp_to_millis(TimeUtil.generate_start_timestamp(days=1))
            if (n_positions_on_disk == 0 or youngest_disk_ms < one_day_ago):
                msg = ("Validator data needs to be synced with mainnet validators. "
                       "Restoring validator with 24 hour lagged file. More info here: "
                       "https://github.com/taoshidev/proprietary-trading-network/"
                       "blob/main/docs/regenerating_validator_state.md")
                bt.logging.warning(msg)
                self.position_syncer.sync_positions(
                    False, candidate_data=self.position_syncer.read_validator_checkpoint_from_gcloud_zip())


    def check_shutdown(self):
        if not is_shutdown():
            return
        # Handle shutdown gracefully
        bt.logging.warning("Performing graceful exit...")

        # Send shutdown notification to Slack
        if self.slack_notifier:
            self.slack_notifier.send_message(
                f"🛑 Validator shutting down gracefully\n"
                f"Hotkey: {self.wallet.hotkey.ss58_address}",
                level="warning"
            )
        bt.logging.warning("Stopping axon...")
        self.axon.stop()
        bt.logging.warning("Stopping metagraph update...")
        self.metagraph_updater_thread.join()
        # All RPC servers shut down automatically via ShutdownCoordinator:
        if self.api_thread:
            bt.logging.warning("Stopping API manager...")
            self.api_thread.join()
        signal.alarm(0)
        print("Graceful shutdown completed")
        sys.exit(0)

    def main(self):
        # Keep the vali alive. This loop maintains the vali's operations until intentionally stopped.
        bt.logging.info("Starting main loop")

        # Send startup notification to Slack
        if self.slack_notifier:
            vm_info = f"VM: {self.slack_notifier.vm_hostname} ({self.slack_notifier.vm_ip})" if self.slack_notifier.vm_hostname else ""
            self.slack_notifier.send_message(
                f"🚀 Validator started successfully!\n"
                f"Hotkey: {self.wallet.hotkey.ss58_address}\n"
                f"Network: {self.config.subtensor.network}\n"
                f"Netuid: {self.config.netuid}\n"
                f"AutoSync: {self.auto_sync}\n"
                f"{vm_info}",
                level="info"
            )
        while not is_shutdown():
            try:
                self.position_syncer.sync_positions_with_cooldown(self.auto_sync)
                # All managers now run in their own daemon processes

            # In case of unforeseen errors, the validator will log the error and send notification to Slack
            except Exception as e:
                error_traceback = traceback.format_exc()
                bt.logging.error(error_traceback)

                error_message = ErrorUtils.format_error_for_slack(
                    error=e,
                    traceback_str=error_traceback,
                    include_operation=True,
                    include_timestamp=True
                )

                self.slack_notifier.send_message(
                    f"❌ Validator main loop error!\n"
                    f"{error_message}\n",
                    level="error"
                )

            time.sleep(10)

        self.check_shutdown()

    def should_fail_early(self, synapse: template.protocol.SendSignal | template.protocol.GetPositions | template.protocol.ValidatorCheckpoint, method:SynapseMethod,
                          signal:dict=None, now_ms=None) -> bool:
        if is_shutdown():
            synapse.successfully_processed = False
            synapse.error_message = "Validator is restarting due to update. Please try again later."
            bt.logging.trace(synapse.error_message)
            return True

        sender_hotkey = synapse.dendrite.hotkey
        # Don't allow miners to send too many signals in a short period of time
        if method == SynapseMethod.POSITION_INSPECTOR:
            allowed, wait_time = self.position_inspector_rate_limiter.is_allowed(sender_hotkey)
        elif method == SynapseMethod.SIGNAL:
            allowed, wait_time = self.order_rate_limiter.is_allowed(sender_hotkey)
        elif method == SynapseMethod.CHECKPOINT:
            allowed, wait_time = self.checkpoint_rate_limiter.is_allowed(sender_hotkey)
        else:
            msg = "Received synapse does not match one of expected methods for: receive_signal, get_positions, get_dash_data, or receive_checkpoint"
            bt.logging.trace(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if not allowed:
            msg = (f"Rate limited. Please wait {wait_time} seconds before sending another signal. "
                   f"{method.value}")
            bt.logging.trace(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if method == SynapseMethod.CHECKPOINT:
            return False
        elif method == SynapseMethod.POSITION_INSPECTOR:
            # Check version 0 (old version that was opt-in)
            if synapse.version == 0:
                synapse.successfully_processed = False
                synapse.error_message = "Please use the latest miner script that makes PI opt-in with the flag --run-position-inspector"
                #bt.logging.info((sender_hotkey, synapse.error_message))
                return True

        # don't process eliminated miners
        # Fast local lookup from EliminationClient cache (no RPC call!) - saves 66.81ms per order
        elim_check_start = time.perf_counter()
        elimination_info = self.elimination_client.get_elimination_local_cache(synapse.dendrite.hotkey)
        elim_check_ms = (time.perf_counter() - elim_check_start) * 1000
        bt.logging.info(f"[FAIL_EARLY_DEBUG] get_elimination_local_cache took {elim_check_ms:.2f}ms")

        if elimination_info:
            msg = f"This miner hotkey {synapse.dendrite.hotkey} has been eliminated and cannot participate in this subnet. Try again after re-registering. elimination_info {elimination_info}"
            bt.logging.debug(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        # don't process re-registered miners
        # Fast local lookup from EliminationClient cache (no RPC call!) - saves 11.26ms per order
        rereg_check_start = time.perf_counter()
        rereg_info = self.elimination_client.get_departed_hotkey_info_local_cache(synapse.dendrite.hotkey)
        rereg_check_ms = (time.perf_counter() - rereg_check_start) * 1000
        bt.logging.info(f"[FAIL_EARLY_DEBUG] get_departed_hotkey_info_local_cache took {rereg_check_ms:.2f}ms")

        if rereg_info:
            # Use cached departure info (already fetched in thread-safe read above)
            detected_ms = rereg_info.get("detected_ms", 0)
            dereg_date = TimeUtil.millis_to_formatted_date_str(detected_ms) if detected_ms else "unknown"

            msg = (f"This miner hotkey {synapse.dendrite.hotkey} was previously de-registered and is not allowed to re-register. "
                   f"De-registered on: {dereg_date} UTC. "
                   f"Re-registration is not permitted on this subnet.")
            bt.logging.warning(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        order_uuid = synapse.miner_order_uuid
        tp = Order.parse_trade_pair_from_signal(signal)
        if order_uuid and self.uuid_tracker.exists(order_uuid):
            # Parse execution type to check if this is a cancel operation
            execution_type = ExecutionType.from_string(signal.get("execution_type", "MARKET").upper()) if signal else ExecutionType.MARKET
            # Allow duplicate UUIDs for LIMIT_CANCEL (reusing UUID to identify order to cancel)
            if execution_type != ExecutionType.LIMIT_CANCEL:
                msg = (f"Order with uuid [{order_uuid}] has already been processed. "
                       f"Please try again with a new order.")
                bt.logging.error(msg)
                synapse.error_message = msg

        elif tp.is_blocked:
            msg = (f"Trade pair [{tp.trade_pair_id}] is no longer supported. "
                   f"Please try again with a different trade pair.")
            synapse.error_message = msg

        elif signal and tp and not synapse.error_message:
            # Fast local validation using background-refreshed cache (no RPC call, no refresh penalty!)
            asset_validate_start = time.perf_counter()
            # Check timestamp and validate locally using cached data
            if now_ms >= ASSET_CLASS_SELECTION_TIME_MS:
                # Fast local lookup from AssetSelectionClient cache
                selected_asset = self.asset_selection_client.get_selection_local_cache(synapse.dendrite.hotkey)
                is_valid_asset = selected_asset == tp.trade_pair_category if selected_asset is not None else False
            else:
                is_valid_asset = True  # Pre-cutoff, all assets allowed
                selected_asset = "unknown (pre-cutoff)"

            asset_validate_ms = (time.perf_counter() - asset_validate_start) * 1000
            bt.logging.info(f"[FAIL_EARLY_DEBUG] validate_order_asset_class_local_cache took {asset_validate_ms:.2f}ms")

            if not is_valid_asset:
                msg = (
                    f"miner [{synapse.dendrite.hotkey}] cannot trade asset class [{tp.trade_pair_category.value}]. "
                    f"Selected asset class: [{selected_asset or 'unknown'}]. Only trade pairs from your selected asset class are allowed. "
                    f"See https://docs.taoshi.io/ptn/ptncli#miner-operations for more information."
                )
                synapse.error_message = msg
            else:
                is_market_open = self.price_fetcher_client.is_market_open(tp, now_ms)
                execution_type = ExecutionType.from_string(signal.get("execution_type", "MARKET").upper())
                if execution_type == ExecutionType.MARKET and not is_market_open:
                    msg = (f"Market for trade pair [{tp.trade_pair_id}] is likely closed or this validator is"
                           f" having issues fetching live price. Please try again later.")
                    synapse.error_message = msg
                else:
                    unsupported_check_start = time.perf_counter()
                    unsupported_pairs = self.price_fetcher_client.get_unsupported_trade_pairs()
                    unsupported_check_ms = (time.perf_counter() - unsupported_check_start) * 1000
                    bt.logging.info(f"[FAIL_EARLY_DEBUG] get_unsupported_trade_pairs took {unsupported_check_ms:.2f}ms")

                    if tp in unsupported_pairs:
                        msg = (f"Trade pair [{tp.trade_pair_id}] has been temporarily halted. "
                               f"Please try again with a different trade pair.")
                        synapse.error_message = msg

        synapse.successfully_processed = not bool(synapse.error_message)
        if synapse.error_message:
            bt.logging.error(synapse.error_message)

        return bool(synapse.error_message)

    @timeme
    def blacklist_fn(self, synapse, metagraph) -> Tuple[bool, str]:
        """
        Override blacklist_fn to use metagraph_updater's cached hotkeys.

        Performance impact:
        - metagraph.has_hotkey() RPC call: ~5-10ms → <0.01ms (set lookup)

        Cache is atomically refreshed by metagraph_updater during metagraph updates.
        """
        # Fast local set lookup via metagraph_updater (no RPC call!)
        miner_hotkey = synapse.dendrite.hotkey
        is_registered = self.metagraph_updater.is_hotkey_registered_cached(miner_hotkey)

        if not is_registered:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {miner_hotkey}"
            )
            return True, miner_hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {miner_hotkey}"
        )
        return False, miner_hotkey


    # This is the core validator function to receive a signal
    def receive_signal(self, synapse: template.protocol.SendSignal,
                       ) -> template.protocol.SendSignal:
        # pull miner hotkey to reference in various activities
        now_ms = TimeUtil.now_in_millis()
        order = None
        miner_hotkey = synapse.dendrite.hotkey
        synapse.validator_hotkey = self.wallet.hotkey.ss58_address
        miner_repo_version = synapse.repo_version
        signal = synapse.signal
        bt.logging.info( f"received signal [{signal}] from miner_hotkey [{miner_hotkey}] using repo version [{miner_repo_version}].")

        # TIMING: Check should_fail_early timing
        fail_early_start = TimeUtil.now_in_millis()
        if self.should_fail_early(synapse, SynapseMethod.SIGNAL, signal=signal, now_ms=now_ms):
            fail_early_ms = TimeUtil.now_in_millis() - fail_early_start
            bt.logging.info(f"[TIMING] should_fail_early took {fail_early_ms}ms (rejected)")
            return synapse
        fail_early_ms = TimeUtil.now_in_millis() - fail_early_start
        bt.logging.info(f"[TIMING] should_fail_early took {fail_early_ms}ms")

        with self.signal_sync_lock:
            self.n_orders_being_processed[0] += 1

        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""
        try:
            # TIMING: Parse operations
            parse_start = TimeUtil.now_in_millis()
            miner_order_uuid = SendSignal.parse_miner_uuid(synapse)

            # Use OrderProcessor to parse common fields
            trade_pair, execution_type, _ = OrderProcessor.parse_signal_data(signal, miner_order_uuid)

            parse_ms = TimeUtil.now_in_millis() - parse_start
            bt.logging.info(f"[TIMING] Parse operations took {parse_ms}ms")

            if execution_type == ExecutionType.LIMIT:
                # Use OrderProcessor to handle LIMIT order
                order = OrderProcessor.process_limit_order(
                    signal, trade_pair, miner_order_uuid, now_ms,
                    miner_hotkey, self.limit_order_client
                )

                # Set synapse response (validator's responsibility)
                synapse.order_json = order.__str__()

                # UUID tracking happens HERE in validator process (limit_order_manager is separate process)
                self.uuid_tracker.add(miner_order_uuid)

            elif execution_type == ExecutionType.BRACKET:
                # Use OrderProcessor to handle BRACKET order
                order = OrderProcessor.process_bracket_order(
                    signal, trade_pair, miner_order_uuid, now_ms,
                    miner_hotkey, self.limit_order_client
                )

                # Set synapse response (validator's responsibility)
                synapse.order_json = order.__str__()

                # UUID tracking happens HERE in validator process (limit_order_manager is separate process)
                self.uuid_tracker.add(miner_order_uuid)

            elif execution_type == ExecutionType.LIMIT_CANCEL:
                # Use OrderProcessor to handle LIMIT_CANCEL
                result = OrderProcessor.process_limit_cancel(
                    signal, trade_pair, miner_order_uuid, now_ms,
                    miner_hotkey, self.limit_order_client
                )

                # Set synapse response (validator's responsibility)
                synapse.order_json = json.dumps(result)
                # No UUID tracking for cancel operations

            else:
                # Use OrderProcessor to handle MARKET order (consistent interface)
                err_msg, updated_position, created_order = OrderProcessor.process_market_order(
                    signal, trade_pair, miner_order_uuid, now_ms,
                    miner_hotkey, miner_repo_version,
                    self.market_order_manager
                )

                # Check for errors and raise SignalException if processing failed
                if err_msg:
                    raise SignalException(err_msg)

                # Set synapse response from created order
                if created_order:
                    synapse.order_json = created_order.__str__()

                # UUID tracking happens HERE in validator process
                self.uuid_tracker.add(miner_order_uuid)

        except SignalException as e:
            exception_time = TimeUtil.now_in_millis()
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())
            bt.logging.info(f"[TIMING] SignalException caught at {exception_time - now_ms}ms from start")
        except Exception as e:
            exception_time = TimeUtil.now_in_millis()
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())
            bt.logging.info(f"[TIMING] General Exception caught at {exception_time - now_ms}ms from start")
        finally:
            # CRITICAL: Always execute final processing and decrement counter, even on unhandled exceptions
            # This prevents deadlock if SystemExit, KeyboardInterrupt, or other BaseException occurs

            # TIMING: Final processing
            final_processing_start = TimeUtil.now_in_millis()
            if error_message == "":
                synapse.successfully_processed = True
            else:
                bt.logging.error(error_message)
                synapse.successfully_processed = False

            synapse.error_message = error_message
            final_processing_ms = TimeUtil.now_in_millis() - final_processing_start
            bt.logging.info(f"[TIMING] Final synapse setup took {final_processing_ms}ms")

            processing_time_s_3_decimals = round((TimeUtil.now_in_millis() - now_ms) / 1000.0, 3)
            bt.logging.success(f"Sending ack back to miner [{miner_hotkey}]. Synapse Message: {synapse.error_message}. "
                               f"Process time {processing_time_s_3_decimals} seconds. order {order}")

            # CRITICAL: Decrement counter in finally block to prevent deadlock
            # This ensures the counter is always decremented, even if an unhandled exception occurs
            with self.signal_sync_lock:
                self.n_orders_being_processed[0] -= 1
                if self.n_orders_being_processed[0] == 0:
                    self.signal_sync_condition.notify_all()

        return synapse

    def get_positions(self, synapse: template.protocol.GetPositions,
                      ) -> template.protocol.GetPositions:
        if self.should_fail_early(synapse, SynapseMethod.POSITION_INSPECTOR):
            return synapse
        t0 = time.time()
        miner_hotkey = synapse.dendrite.hotkey
        error_message = ""
        n_positions_sent = 0
        hotkey = None
        try:
            hotkey = synapse.dendrite.hotkey
            # Return the last n positions using PositionManagerClient
            positions = self.position_manager_client.get_positions_for_one_hotkey(hotkey, only_open_positions=True)
            synapse.positions = [position.to_dict() for position in positions]
            n_positions_sent = len(synapse.positions)
        except Exception as e:
            error_message = f"Error in GetPositions for [{miner_hotkey}] with error [{e}]. Perhaps the position was being written to disk at the same time."
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False
        synapse.error_message = error_message
        msg = f"Sending {n_positions_sent} positions back to miner: {hotkey} in {round(time.time() - t0, 3)} seconds."
        if synapse.error_message:
            msg += f" Error: {synapse.error_message}"
        bt.logging.info(msg)
        return synapse


# This is the main function, which runs the miner.
if __name__ == "__main__":
    validator = Validator()
    validator.main()
