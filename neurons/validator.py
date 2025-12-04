# The MIT License (MIT)
# Copyright (c) 2024 Yuma Rao
# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
import json
import os
import sys
import threading
import signal

from vali_objects.enums.misc import SynapseMethod
from vanta_api.api_manager import APIManager
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ValidatorContext


import template
import traceback
import time
import bittensor as bt

from typing import Tuple
from setproctitle import setproctitle
from neurons.validator_base import ValidatorBase
from template.protocol import SendSignal
from vali_objects.utils.asset_selection.asset_selection_manager import ASSET_CLASS_SELECTION_TIME_MS
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.data_sync.auto_sync import PositionSyncer
from vali_objects.data_sync.order_sync_state import OrderSyncState
from vali_objects.utils.limit_order.market_order_manager import MarketOrderManager
from shared_objects.rate_limiter import RateLimiter
from vali_objects.uuid_tracker import UUIDTracker
from time_util.time_util import TimeUtil, timeme
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.subtensor_ops.subtensor_ops import MetagraphUpdater
from shared_objects.error_utils import ErrorUtils
from shared_objects.slack_notifier import SlackNotifier
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.limit_order.order_processor import OrderProcessor
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator

def is_shutdown() -> bool:
    """Check if shutdown is in progress via ShutdownCoordinator."""
    return ShutdownCoordinator.is_shutdown()

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

        # Thread-safe state for coordinating order processing vs. position sync
        # Tracks in-flight orders and signals when sync is waiting
        self.order_sync = OrderSyncState()

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

        bt.logging.info(f"Wallet: {self.wallet}")

        # ============================================================================
        # SERVER ORCHESTRATOR - Centralized server lifecycle management
        # ============================================================================
        # Create validator context with all dependencies
        context = ValidatorContext(
            slack_notifier=self.slack_notifier,
            config=self.config,
            wallet=self.wallet,
            secrets=self.secrets,
            is_mainnet=self.is_mainnet
        )

        # Start all servers (but defer daemon/pre-run setup until after MetagraphUpdater)
        orchestrator = ServerOrchestrator.get_instance()
        orchestrator.start_validator_servers(context, start_daemons=False, run_pre_setup=False)
        bt.logging.success("[INIT] All servers started via ServerOrchestrator (daemons deferred)")

        # Get clients from orchestrator (cached, fast)
        self.metagraph_client = orchestrator.get_client('metagraph')
        self.price_fetcher_client = orchestrator.get_client('live_price_fetcher')
        self.position_manager_client = orchestrator.get_client('position_manager')
        self.elimination_client = orchestrator.get_client('elimination')
        self.challengeperiod_client = orchestrator.get_client('challenge_period')
        self.limit_order_client = orchestrator.get_client('limit_order')
        self.asset_selection_client = orchestrator.get_client('asset_selection')
        self.perf_ledger_client = orchestrator.get_client('perf_ledger')
        self.debt_ledger_client = orchestrator.get_client('debt_ledger')
        self.entity_client = orchestrator.get_client('entity')

        # Create MetagraphUpdater with simple parameters
        # This will run in a thread in the main process
        # MetagraphUpdater now exposes RPC server for weight setting (validators only)
        self.metagraph_updater = MetagraphUpdater(
            self.config, self.wallet.hotkey.ss58_address,
            False,
            slack_notifier=self.slack_notifier
        )
        self.subtensor = self.metagraph_updater.subtensor
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Start the metagraph updater and wait for initial population.
        # CRITICAL: This must complete before daemons start since they depend on metagraph data.
        # Weight setting also needs metagraph_updater to be running to receive weight set RPCs.
        self.metagraph_updater_thread = self.metagraph_updater.start_and_wait_for_initial_update(
            max_wait_time=60,
            slack_notifier=self.slack_notifier
        )
        bt.logging.success("[INIT] MetagraphUpdater started and populated")
        orchestrator.call_pre_run_setup(perform_order_corrections=True)

        # Now start server daemons and run pre-run setup (safe now that metagraph is populated)
        # Cache warmup happens automatically inside start_server_daemons() to eliminate race conditions
        orchestrator.start_server_daemons([
            'perf_ledger',
            'challenge_period',
            'elimination',
            'position_manager',
            'debt_ledger',
            'limit_order',
            'plagiarism_detector',
            'mdd_checker',
            'core_outputs',
            'miner_statistics',
            'weight_calculator'
        ])
        bt.logging.success("[INIT] Server daemons started, caches warmed, and pre-run setup completed")
        # ============================================================================

        # Create PositionSyncer (not a server, runs in main process)
        self.position_syncer = PositionSyncer(
            order_sync=self.order_sync,
            auto_sync_enabled=self.auto_sync
        )

        # MarketOrderManager creates its own ContractClient internally (forward compatibility)
        self.market_order_manager = MarketOrderManager(self.config.serve, slack_notifier=self.slack_notifier)

        # Initialize UUID tracker with existing positions
        self.uuid_tracker.add_initial_uuids(self.position_manager_client.get_positions_for_all_miners())

        # Verify hotkey is registered
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
                         metagraph=self.metagraph_client,
                         asset_selection_client=self.asset_selection_client, subtensor=self.subtensor)

        # Rate limiters for incoming requests
        self.order_rate_limiter = RateLimiter()
        self.position_inspector_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 4)

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
                bt.logging.info(f"Found {n_positions_on_disk} hotkeys with positions on disk."
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

    def should_fail_early(self, synapse: template.protocol.SendSignal | template.protocol.GetPositions, method: SynapseMethod,
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
        else:
            msg = "Received synapse does not match one of expected methods for: receive_signal or get_positions"
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

        if method == SynapseMethod.POSITION_INSPECTOR:
            # Check version 0 (old version that was opt-in)
            if synapse.version == 0:
                synapse.successfully_processed = False
                synapse.error_message = "Please use the latest miner script that makes PI opt-in with the flag --run-position-inspector"
                #bt.logging.info((sender_hotkey, synapse.error_message))
                return True
            else:
                return False

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

        # Entity hotkey validation: Don't allow orders from entity hotkeys (non-synthetic)
        # Only synthetic hotkeys (subaccounts) can place orders
        entity_check_start = time.perf_counter()
        if self.entity_client.is_synthetic_hotkey(sender_hotkey):
            # This is a synthetic hotkey - verify it's active
            found, status, _ = self.entity_client.get_subaccount_status(sender_hotkey)
            if not found or status != 'active':
                msg = (f"Synthetic hotkey {sender_hotkey} is not active or not found. "
                       f"Please ensure your subaccount is properly registered.")
                bt.logging.warning(msg)
                synapse.successfully_processed = False
                synapse.error_message = msg
                return True
        else:
            # Not a synthetic hotkey - check if it's an entity hotkey
            entity_data = self.entity_client.get_entity_data(sender_hotkey)
            if entity_data:
                msg = (f"Entity hotkey {sender_hotkey} cannot place orders directly. "
                       f"Please use a subaccount (synthetic hotkey) to place orders.")
                bt.logging.warning(msg)
                synapse.successfully_processed = False
                synapse.error_message = msg
                return True
        entity_check_ms = (time.perf_counter() - entity_check_start) * 1000
        bt.logging.info(f"[FAIL_EARLY_DEBUG] entity_hotkey_validation took {entity_check_ms:.2f}ms")

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

        # Early rejection if sync is waiting (fast local check, ~0.01ms)
        if self.order_sync.is_sync_waiting():
            synapse.successfully_processed = False
            synapse.error_message = "Validator is syncing positions. Please try again shortly."
            bt.logging.debug(f"Rejected order from {miner_hotkey} - sync waiting")
            return synapse

        # Track order processing with context manager (auto-increments/decrements counter)
        with self.order_sync.begin_order():
            # error message to send back to miners in case of a problem so they can fix and resend
            error_message = ""
            try:
                # TIMING: Parse operations
                parse_start = TimeUtil.now_in_millis()
                miner_order_uuid = SendSignal.parse_miner_uuid(synapse)
                parse_ms = TimeUtil.now_in_millis() - parse_start
                bt.logging.info(f"[TIMING] Parse operations took {parse_ms}ms")

                # Use unified OrderProcessor dispatcher (replaces lines 602-661)
                result = OrderProcessor.process_order(
                    signal=signal,
                    miner_order_uuid=miner_order_uuid,
                    now_ms=now_ms,
                    miner_hotkey=miner_hotkey,
                    miner_repo_version=miner_repo_version,
                    limit_order_client=self.limit_order_client,
                    market_order_manager=self.market_order_manager
                )

                # Set synapse response (centralized - single line instead of 4)
                synapse.order_json = result.get_response_json()

                # Track UUID if needed (centralized - single line instead of 3)
                if result.should_track_uuid:
                    self.uuid_tracker.add(miner_order_uuid)

                # For logging (used in line 691)
                order = result.order_for_logging

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

                processing_time_ms = TimeUtil.now_in_millis() - now_ms
                bt.logging.success(f"Sending ack back to miner [{miner_hotkey}]. Synapse Message: {synapse.error_message}. "
                                   f"Process time {processing_time_ms}ms. order {order}")
                # Context manager auto-decrements counter and notifies waiters on exit

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
