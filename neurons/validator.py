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
from vali_objects.enums.order_type_enum import OrderType
from vanta_api.validator_api_manager import ValidatorAPIManager
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, NeuronContext
from entity_management.entity_utils import is_synthetic_hotkey


import template
import traceback
import time
import bittensor as bt

from typing import Tuple
from setproctitle import setproctitle
from neurons.validator_base import ValidatorBase
from template.protocol import SendSignal
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.data_sync.auto_sync import PositionSyncer
from vali_objects.data_sync.order_sync_state import OrderSyncState
from vali_objects.data_sync.order_sync_client import OrderSyncClient
from vali_objects.order_uuid_dedup_client import OrderUuidDedupClient
from shared_objects.rate_limiter import RateLimiter
from vali_objects.uuid_tracker import UUIDTracker
from time_util.time_util import TimeUtil, timeme
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.subtensor_ops.subtensor_ops import SubtensorOpsManager
from shared_objects.error_utils import ErrorUtils
from shared_objects.slack_notifier import SlackNotifier
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.order_processor import OrderProcessor
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from runnable.run_migrations import main as run_migrations

# Clients connect by service-name+port authkey regardless of which process spawned the server, so
# these work whether the target servers are in-process or in another tier's process.
from shared_objects.rpc.metagraph_client import MetagraphClient
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from entity_management.entity_client import EntityClient
from vali_objects.utils.entity_collateral.entity_collateral_client import EntityCollateralClient
from vali_objects.utils.market_order.market_order_client import MarketOrderClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient

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
        # Note: Use print() instead of bt.logging before bt.logging is configured
        try:
            with open("meta/meta.json", "r") as f:
                meta_content = f.read()
                print(f"Found meta.json file: {meta_content}")
        except Exception as e:
            print(f"Error reading meta/meta.json: {e}")

        # Read config before migrations: the role flags gate what runs below.
        self.config = self.get_config()
        self.orders_app = getattr(self.config, 'orders_app', False)
        self.split_state = getattr(self.config, 'split_state', False)
        self.is_mainnet = self.config.netuid == 8

        # Migrations + tmp clear mutate on-disk state, which the client-only orders app does not own.
        if not self.orders_app:
            print("Checking for pending migrations...")
            if not run_migrations():
                print("ERROR: Migration failed. Starting validator without executing migrations")
            else:
                print("Migrations completed successfully.")
            ValiBkpUtils.clear_tmp_dir()

        # OrderUuidDedupClient is server-authoritative (survives a restart, holds across instances);
        # UUIDTracker is process-local. The handler claims a uuid BEFORE applying (check_and_add) and
        # releases on failure — this serializes concurrent same-uuid orders (32-worker executor),
        # which the old add-after-success left able to both apply.
        self.uuid_tracker = OrderUuidDedupClient() if self.orders_app else UUIDTracker()

        # When split, the order side (vanta-orders) and sync side (core) coordinate through
        # CommonDataServer; the monolith uses the in-memory equivalent.
        self.order_sync = OrderSyncClient() if (self.orders_app or self.split_state) else OrderSyncState()
        # ValiConfig.HL_USE_TESTNET = not self.is_mainnet
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

        # Initialize Bittensor miner objects
        # These classes are vital to interact and function within the Bittensor network.
        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        bt.logging.info("Initializing validator wallet...")
        wallet_start_time = time.time()
        self.wallet = bt.Wallet(config=self.config)
        wallet_elapsed_s = time.time() - wallet_start_time
        bt.logging.success(f"Validator wallet initialized in {wallet_elapsed_s:.2f}s")

        # Determine if this validator is the mothership using centralized utility
        self.is_mothership = ValiUtils.is_mothership_wallet(self.wallet, not self.is_mainnet)
        bt.logging.info(f"Is mothership validator: {self.is_mothership}")

        # Auto-sync disabled for mothership (it's the source of truth)
        self.auto_sync = getattr(self.config, 'autosync', False) and not self.is_mothership

        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} with autosync set to: {self.auto_sync} "
            f"on network: {self.config.subtensor.chain_endpoint} with config:"
        )

        # This logs the active configuration to the specified logging directory for review.
        bt.logging.info(self.config)

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
        context = NeuronContext(
            slack_notifier=self.slack_notifier,
            config=self.config,
            wallet=self.wallet,
            secrets=self.secrets,
            is_mainnet=self.is_mainnet
        )

        # orders-app is a pure client (starts no servers); core-split starts only its own tier;
        # the monolith starts everything.
        orchestrator = ServerOrchestrator.get_instance()
        if self.orders_app:
            bt.logging.info("[INIT] --orders-app: starting NO servers (pure client of vanta-state/core)")
        elif self.split_state:
            bt.logging.info("[INIT] --split-state: core hosts its tier only; state servers run in vanta-state")
            orchestrator.start_core_servers(context)
        else:
            orchestrator.start_validator_servers(context)
        bt.logging.success("[INIT] Server startup phase complete")

        self.metagraph_client = MetagraphClient()
        self.price_fetcher_client = LivePriceFetcherClient()
        self.position_manager_client = PositionManagerClient()
        # local cache: per-order elimination / departed-hotkey lookups without an RPC each.
        self.elimination_client = EliminationClient(local_cache_refresh_period_ms=5000)
        self.challengeperiod_client = ChallengePeriodClient()
        self.limit_order_client = LimitOrderClient()
        # local cache: per-order asset-class lookup without an RPC each.
        self.asset_selection_client = AssetSelectionClient(local_cache_refresh_period_ms=5000)
        self.perf_ledger_client = PerfLedgerClient()
        self.debt_ledger_client = DebtLedgerClient()
        self.entity_client = EntityClient()
        self.entity_collateral_client = EntityCollateralClient()
        self.market_order_client = MarketOrderClient()
        self.miner_account_client = MinerAccountClient()

        if self.orders_app:
            # subtensor_ops lives in core; build our own subtensor for the one-shot axon.serve, and
            # leave subtensor_ops_manager None so blacklist_fn uses the metagraph-fed hotkey cache.
            self.subtensor = bt.subtensor(config=self.config)
            self.subtensor_ops_manager = None
            self._init_blacklist_hotkey_cache()
        else:
            subtensor_ops_server = orchestrator.get_server('subtensor_ops')
            self.subtensor = subtensor_ops_server.get_subtensor()
            self.subtensor_ops_manager = subtensor_ops_server.manager  # For blacklist_fn

        if self.orders_app:
            bt.logging.info("[INIT] --orders-app: no server daemons / pre_run_setup (client of state tiers)")
        elif self.split_state:
            # Core starts only its own tier's daemons; vanta-state starts its own (incl. pre_run_setup).
            orchestrator.start_server_daemons([
                'perf_ledger',
                'challenge_period',
                'elimination',
                'debt_ledger',
                'mdd_checker',
                'core_outputs',
                'miner_statistics',
                'weight_calculator',
                'entity',
            ])
            bt.logging.success("[INIT] Core-tier daemons started (state daemons run in vanta-state)")
        else:
            # Single-process: core hosts everything (today's behavior).
            orchestrator.call_pre_run_setup(perform_order_corrections=True)
            orchestrator.start_server_daemons([
                'perf_ledger',
                'miner_account',
                'challenge_period',
                'elimination',
                'position_manager',
                'debt_ledger',
                'limit_order',
                'mdd_checker',
                'core_outputs',
                'miner_statistics',
                'weight_calculator',
                'entity',
                'entity_collateral'
            ])
            bt.logging.success("[INIT] All daemons started, caches warmed")
        # ============================================================================

        # Position sync runs in core/monolith, not the orders app.
        if self.orders_app:
            self.position_syncer = None
        else:
            self.position_syncer = PositionSyncer(
                order_sync=self.order_sync,
                auto_sync_enabled=self.auto_sync,
                is_mothership=self.is_mothership
            )

        self.order_processor = OrderProcessor(
            limit_order_client=self.limit_order_client,
            market_order_client=self.market_order_client,
            miner_account_client=self.miner_account_client,
        )

        # Initialize UUID tracker with existing positions
        self.uuid_tracker.add_initial_uuids(self.position_manager_client.get_positions_for_all_miners())

        # HL tracker is co-resident with the axon (monolith or orders app), never in a core-split
        # process, so a core restart doesn't drop HL fill ingestion.
        if getattr(self.config, 'serve_axon', True) or self.orders_app:
            from entity_management.hyperliquid_tracker import HyperliquidTracker
            from vanta_api.websocket_notifier import WebSocketNotifierClient
            hl_ws_notifier = WebSocketNotifierClient(connect_immediately=False)
            self.hl_tracker = HyperliquidTracker(
                entity_client=self.entity_client,
                price_fetcher_client=self.price_fetcher_client,
                order_processor=self.order_processor,
                ws_notifier_client=hl_ws_notifier,
            )
            self.hl_tracker.start()
        else:
            self.hl_tracker = None
            bt.logging.info("[INIT] --no-axon: HL tracker not started here (runs in vanta-orders app)")

        # Verify hotkey is registered (via metagraph_client — available in all roles).
        bt.logging.info(f"Metagraph n_entries: {len(self.metagraph_client.get_hotkeys())}")
        if not self.metagraph_client.has_hotkey(self.wallet.hotkey.ss58_address):
            bt.logging.error(
                f"\nYour validator hotkey: {self.wallet.hotkey.ss58_address} (wallet: {self.wallet.name}, hotkey: {self.wallet.hotkey_str}) "
                f"is not registered on netuid {self.config.netuid}. Run btcli register and try again."
            )
            exit()

        # orders-app already set self.subtensor above; otherwise take subtensor_ops_manager's.
        if self.subtensor_ops_manager is not None:
            self.subtensor = self.subtensor_ops_manager.get_subtensor()

        # Build and link vali functions to the axon.
        # The axon handles request processing, allowing validators to send this process requests.
        # ValidatorBase creates its own clients internally (forward compatibility):
        # - AssetSelectionClient, ContractClient
        super().__init__(wallet=self.wallet, slack_notifier=self.slack_notifier, config=self.config,
                         metagraph_client=self.metagraph_client,
                         asset_selection_client=self.asset_selection_client, subtensor=self.subtensor)

        # Rate limiters for incoming requests
        self.order_rate_limiter = RateLimiter()
        self.position_inspector_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 4)

        # spawn_api gates only in-core spawning; config.serve stays on because it also gates the
        # position-update broadcasts (market_order_manager.py) that feed the extracted WS server.
        # Under run.sh's split (--no-spawn-api), REST/WS run as their own PM2 apps — spawning here
        # too would double-bind 48888/8765/50014/50022. Defaults to spawning (backward-safe).
        if self.config.serve and getattr(self.config, 'spawn_api', True) and not self.orders_app:
            # Create API Manager with configuration options
            self.api_manager = ValidatorAPIManager(
                slack_webhook_url=getattr(self.config, 'slack_webhook_url', None),
                validator_hotkey=self.wallet.hotkey.ss58_address,
                api_host=getattr(self.config, 'api_host', '0.0.0.0'),
                api_rest_port=getattr(self.config, 'api_rest_port', 48888),
                api_ws_port=getattr(self.config, 'api_ws_port', 8765),
                is_mainnet=self.is_mainnet
            )

            # Start the API Manager in a separate thread. Handle seperately from other RPCServers as Flask was giving issues.
            self.api_thread = threading.Thread(target=self.api_manager.run, daemon=True)
            self.api_thread.start()
            # Verify thread started
            time.sleep(0.1)
            if not self.api_thread.is_alive():
                raise RuntimeError("API thread failed to start")
            bt.logging.info(
                f"API services thread started - REST: {getattr(self.config, 'api_host', '0.0.0.0')}:{getattr(self.config, 'api_rest_port', 48888)}, "
                f"WebSocket: {getattr(self.config, 'api_host', '0.0.0.0')}:{getattr(self.config, 'api_ws_port', 8765)}")
        else:
            self.api_thread = None
            if self.config.serve:
                bt.logging.info("API services not spawned by core (--no-spawn-api): "
                                "REST/WS run as separate PM2 apps (vanta-rest / vanta-ws)")
            else:
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
        # orders app doesn't sync (position_syncer is None).
        if self.is_mainnet and not self.orders_app:
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
        if self.axon is not None:
            bt.logging.warning("Stopping axon...")
            self.axon.stop()
        # SubtensorOpsServer and all RPC servers shut down automatically via ShutdownCoordinator:
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
                if self.orders_app:
                    # No sync loop; the axon and HL tracker run in their own threads.
                    pass
                else:
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

    def should_reject_synapse(self, sender_hotkey, synapse: template.protocol.SendSignal | template.protocol.GetPositions, method: SynapseMethod) -> bool:
        if is_shutdown():
            synapse.successfully_processed = False
            synapse.error_message = "Validator is restarting due to update. Please try again later."
            return True

        if method == SynapseMethod.POSITION_INSPECTOR:
            allowed, wait_time = self.position_inspector_rate_limiter.is_allowed(sender_hotkey)
        elif method == SynapseMethod.SIGNAL:
            allowed, wait_time = self.order_rate_limiter.is_allowed(sender_hotkey)
        else:
            msg = "Received synapse does not match one of expected methods for: receive_signal or get_positions"
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if not allowed:
            msg = f"Rate limited. Please wait {wait_time} seconds before sending another signal. {method.value}"
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        if method == SynapseMethod.POSITION_INSPECTOR:
            if synapse.version == 0:
                synapse.successfully_processed = False
                synapse.error_message = "Please use the latest miner script that makes PI opt-in with the flag --run-position-inspector"
                return True
            return False

        return False

    def _init_blacklist_hotkey_cache(self, refresh_period_s: float = 12.0) -> None:
        """
        Registered-hotkey set for blacklist_fn, refreshed in the background from the metagraph server.
        The orders app has no in-process subtensor_ops_manager to consult, and a per-request RPC would
        be too slow; a brief core absence just leaves the set stale rather than blocking reception.
        """
        self._blacklist_hotkeys = set()
        self._blacklist_cache_lock = threading.Lock()

        def _refresh_loop():
            while not is_shutdown():
                try:
                    hks = set(self.metagraph_client.get_hotkeys())
                    with self._blacklist_cache_lock:
                        self._blacklist_hotkeys = hks
                except Exception as e:
                    bt.logging.warning(f"[orders-app] blacklist hotkey cache refresh failed: {e}")
                time.sleep(refresh_period_s)

        # Prime once (best-effort) so the axon isn't cold when it starts serving.
        try:
            self._blacklist_hotkeys = set(self.metagraph_client.get_hotkeys())
        except Exception as e:
            bt.logging.warning(f"[orders-app] initial blacklist hotkey cache prime failed: {e}")
        threading.Thread(target=_refresh_loop, daemon=True, name="blacklist-hotkey-cache").start()

    @timeme
    def blacklist_fn(self, synapse, metagraph) -> Tuple[bool, str]:
        """Blacklist unregistered hotkeys via a fast local set lookup — subtensor_ops_manager's
        in-process cache in the monolith/core, or the metagraph-fed cache in the orders app."""
        miner_hotkey = synapse.dendrite.hotkey
        if self.subtensor_ops_manager is not None:
            is_registered = self.subtensor_ops_manager.is_hotkey_registered_cached(miner_hotkey)
        else:
            with self._blacklist_cache_lock:
                is_registered = miner_hotkey in self._blacklist_hotkeys

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
    def _receive_signal_sync(self, synapse: template.protocol.SendSignal,
                       ) -> template.protocol.SendSignal:
        # pull miner hotkey to reference in various activities
        now_ms = TimeUtil.now_in_millis()
        order = None
        miner_hotkey = synapse.dendrite.hotkey
        subaccount_id = synapse.subaccount_id
        synapse.validator_hotkey = self.wallet.hotkey.ss58_address
        miner_repo_version = synapse.repo_version
        signal_dict = synapse.signal
        order_uuid = SendSignal.parse_miner_uuid(synapse)

        if not miner_hotkey:
            synapse.successfully_processed = False
            synapse.error_message = "Missing miner hotkey"
            synapse.should_retry = False
            return synapse

        # For entity miners: construct synthetic hotkey if subaccount_id provided
        if subaccount_id is not None:
            synthetic_hotkey = f"{miner_hotkey}_{subaccount_id}"
            miner_hotkey = synthetic_hotkey  # Use synthetic hotkey for all downstream ops

        bt.logging.info(f"received signal [{order_uuid}] [{signal_dict}] from miner_hotkey [{miner_hotkey}] using repo version [{miner_repo_version}].")

        if self.should_reject_synapse(miner_hotkey, synapse, SynapseMethod.SIGNAL):
            bt.logging.info(f"Order rejected for {miner_hotkey}: {synapse.error_message}")
            return synapse

        # Advisory fast-path reject; the authoritative gate is begin_order() below.
        if self.order_sync.is_sync_waiting():
            synapse.successfully_processed = False
            synapse.should_retry = True
            synapse.error_message = "Validator is syncing positions. Please try again shortly."
            bt.logging.info(f"Rejected order from {miner_hotkey}: {synapse.error_message}")
            return synapse

        try:
            signal = Signal.model_validate(signal_dict)
        except Exception as e:
            synapse.successfully_processed = False
            synapse.should_retry = False
            synapse.error_message = f"Invalid signal payload: {e}"
            return synapse

        # exists() is an advisory local pre-filter; check_and_add below is the authoritative claim
        # (it also catches a lost-ack retry across a restart, when the local cache is cold).
        should_dedup = (
            bool(order_uuid)
            and signal.execution_type not in (ExecutionType.LIMIT_CANCEL, ExecutionType.LIMIT_EDIT, ExecutionType.FLAT_ALL)
        )
        if should_dedup and self.uuid_tracker.exists(order_uuid):
            synapse.successfully_processed = False
            synapse.should_retry = False
            synapse.error_message = f"Order with uuid [{order_uuid}] has already been processed. Please try again with a new order."
            bt.logging.error(synapse.error_message)
            return synapse

        ok, error_msg, resolved_tp = self.order_processor.validate(
            hotkey=miner_hotkey,
            execution_type=signal.execution_type,
            trade_pair=signal.trade_pair,
            order_type=signal.order_type,
        )
        if not ok:
            synapse.successfully_processed = False
            synapse.should_retry = False
            synapse.error_message = error_msg
            return synapse

        if resolved_tp is not None and resolved_tp != signal.trade_pair:
            signal = signal.model_copy(update={"trade_pair": resolved_tp})

        # begin_order registers the in-flight order (sync waits for it). It also refuses (rejected)
        # if a sync started since the advisory check above, so an order can't apply mid-rewrite;
        # the in-memory OrderSyncState never rejects (rejected=False) and relies on that check.
        with self.order_sync.begin_order() as _admission:
            if getattr(_admission, 'rejected', False):
                synapse.successfully_processed = False
                synapse.should_retry = True
                synapse.error_message = "Validator is syncing positions. Please try again shortly."
                bt.logging.info(f"Rejected order from {miner_hotkey}: sync in progress")
                return synapse

            # error message to send back to miners in case of a problem so they can fix and resend
            error_message = ""
            order_exc = None
            # Claim the uuid BEFORE applying so a duplicate can't double-apply; released below if the
            # apply fails (so a transient-error retry can re-claim) or the result is non-trackable.
            if should_dedup and not self.uuid_tracker.check_and_add(order_uuid):
                synapse.successfully_processed = False
                synapse.should_retry = False
                synapse.error_message = f"Order with uuid [{order_uuid}] has already been processed. Please try again with a new order."
                bt.logging.error(synapse.error_message)
                return synapse
            claimed = should_dedup
            try:
                result = self.order_processor.process_vanta_signal(
                    hotkey=miner_hotkey,
                    signal=signal,
                    order_uuid=order_uuid,
                    now_ms=now_ms,
                )

                # Set synapse response (centralized - single line instead of 4)
                synapse.order_json = result.get_response_json()

                # The claim was taken before we knew the result; drop it if this order isn't tracked.
                if claimed and not result.should_track_uuid:
                    self.uuid_tracker.release(order_uuid)
                    claimed = False

                # For logging (used in the ack below)
                order = result.order_for_logging

            except Exception as e:
                error_message = str(e)
                order_exc = e
                # Apply failed — release the claim so the placer's retry (transient infra errors) can
                # re-claim and succeed rather than being permanently rejected as a duplicate.
                if claimed:
                    self.uuid_tracker.release(order_uuid)
                    claimed = False
                bt.logging.error(f"Error processing signal {miner_hotkey} {order_uuid} {e}")

            finally:
                synapse.error_message = error_message
                if error_message == "":
                    synapse.successfully_processed = True
                else:
                    bt.logging.error(error_message)
                    synapse.successfully_processed = False
                    # Transient infra failures (an RPC state server — incl. MarketOrderServer
                    # :50027 — bouncing during a deploy) must be RETRIED by the placer, not
                    # treated as a permanent rejection that loses the order. Business-logic
                    # rejections (SignalException etc.) stay should_retry=False.
                    synapse.should_retry = ErrorUtils.is_transient_rpc_error(order_exc)

                # TODO Review overlap with serving in market order manager
                if is_synthetic_hotkey(miner_hotkey):
                    self.entity_client.broadcast_subaccount_dashboard(miner_hotkey)

                processing_time_ms = TimeUtil.now_in_millis() - now_ms
                bt.logging.success(f"Sending ack back to miner [{miner_hotkey}]. Synapse Message: {synapse.error_message}. "
                                   f"Process time {processing_time_ms}ms. order {order}")

                # Context manager auto-decrements counter and notifies waiters on exit

        return synapse

    def _get_positions(self, synapse: template.protocol.GetPositions,
                      ) -> template.protocol.GetPositions:
        miner_hotkey = synapse.dendrite.hotkey
        if self.should_reject_synapse(miner_hotkey, synapse, SynapseMethod.POSITION_INSPECTOR):
            return synapse
        t0 = time.time()
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
