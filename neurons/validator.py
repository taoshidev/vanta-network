# The MIT License (MIT)
# Copyright (c) 2024 Yuma Rao
# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc
import concurrent.futures
import os
import sys
import threading
import signal

from vali_objects.enums.misc import SynapseMethod
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, NeuronContext
from entity_management.entity_utils import is_synthetic_hotkey


import traceback
import time
import uuid
import bittensor as bt

from typing import Tuple
from setproctitle import setproctitle
from vali_objects.utils.vali_utils import ValiUtils
import vanta_protocol
from vanta_protocol import (
    GetPositionsRequest,
    GetPositionsResponse,
    SendSignalRequest,
    SendSignalResponse,
)
from shared_objects.http_auth import AuthVerifier
from shared_objects.http_server import HttpServer, build_app, register_handler
from shared_objects.locks.subtensor_lock import get_subtensor_lock
from shared_objects.net_utils import get_external_ip
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.data_sync.auto_sync import PositionSyncer
from vali_objects.data_sync.order_sync_state import OrderSyncState
from vali_objects.data_sync.order_sync_client import OrderSyncClient
from vali_objects.order_uuid_dedup_client import OrderUuidDedupClient
from shared_objects.rate_limiter import RateLimiter
from vali_objects.uuid_tracker import UUIDTracker
from time_util.time_util import TimeUtil, timeme
from shared_objects.error_utils import ErrorUtils
from shared_objects.slack_notifier import SlackNotifier
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order_signal import Signal
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.order_processor import OrderProcessor
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from runnable.run_migrations import main as run_migrations
from shared_objects.log import logger

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
from vali_objects.contract.contract_client import ContractClient

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

# Set up signal handling. These cover only early boot: ServerOrchestrator.get_instance() (in
# Validator.__init__) installs its own SIGINT/SIGTERM handler, which replaces these and runs
# shutdown_all_servers() directly (bounded: grace + kill, inside core's 10s PM2 kill_timeout).
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGALRM, alarm_handler)


class Validator:

    def __init__(self):
        setproctitle(f"vali_{self.__class__.__name__}")
        # Use print() here since self.config (and its logging level) isn't parsed yet
        try:
            with open("meta/meta.json", "r") as f:
                meta_content = f.read()
                print(f"Found meta.json file: {meta_content}")
        except Exception as e:
            print(f"Error reading meta/meta.json: {e}")

        # Read config before migrations: the role flags gate what runs below.
        self.config = ValiUtils.get_config()
        self.orders_app = getattr(self.config, 'orders_app', False)
        self.split_state = getattr(self.config, 'split_state', False)
        self.is_mainnet = self.config.netuid == 8

        # Consensus-affecting testnet knobs (vali_objects/vali_config.py) are inherited
        # through the environment, so a testnet deploy file copied onto a mainnet host
        # would silently diverge this validator's weights. Refuse to start instead.
        if self.is_mainnet:
            _config_overrides = [
                name for name in (
                    "PRO_CHALLENGE_MINIMUM_DAYS",
                    "PRO_CHALLENGE_CALMAR_THRESHOLD",
                    "PRO_CHALLENGE_DAILY_CONSISTENCY_THRESHOLD",
                    "PRO_CHALLENGE_RETURNS_THRESHOLD_DEFAULT",
                    "PTN_ALLOW_CONFIG_OVERRIDES",
                )
                if os.environ.get(name, "").strip()
            ]
            if _config_overrides:
                raise ValueError(
                    f"Refusing to start on netuid 8 with consensus config overrides in the "
                    f"environment: {_config_overrides}. Unset them before running on mainnet."
                )

        # Migrations + tmp clear mutate on-disk state, which the client-only orders app does not
        # own. Under --split-state, MIGRATIONS belong to vanta-state (run_state_server.py): it
        # starts first and its servers load the migrated files — core migrating afterwards would
        # rewrite files underneath the live state tier, whose next save would silently undo the
        # migration (while migrations_completed.txt marks it done).
        if not self.orders_app and not self.split_state:
            print("Checking for pending migrations...")
            if not run_migrations():
                print("ERROR: Migration failed. Starting validator without executing migrations")
            else:
                print("Migrations completed successfully.")
        if not self.orders_app:
            ValiBkpUtils.clear_tmp_dir()

        # OrderUuidDedupClient is server-authoritative (survives a restart, holds across instances);
        # UUIDTracker is process-local. The handler claims a uuid BEFORE applying (check_and_add) and
        # releases on failure — this serializes concurrent same-uuid orders (32-worker executor),
        # which the old add-after-success left able to both apply.
        self.uuid_tracker = OrderUuidDedupClient() if self.orders_app else UUIDTracker()

        # When split, the order side (vanta-orders) and sync side (core) coordinate through
        # CommonDataServer; the monolith uses the in-memory equivalent.
        self.order_sync = OrderSyncClient() if (self.orders_app or self.split_state) else OrderSyncState()

        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

        self.secrets = ValiUtils.get_secrets()
        if self.secrets is None:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )

        # Wallet is initialized first (needed for SlackNotifier below).
        logger.info("Setting up bittensor objects.")
        logger.info("Initializing validator wallet...")
        wallet_start_time = time.time()
        self.wallet = bt.Wallet(name=self.config.wallet.name, hotkey=self.config.wallet.hotkey, path=self.config.wallet.path)
        wallet_elapsed_s = time.time() - wallet_start_time
        logger.info(f"Validator wallet initialized in {wallet_elapsed_s:.2f}s")

        self.is_mothership = ValiUtils.is_mothership_wallet(self.wallet, not self.is_mainnet)
        logger.info(f"Is mothership validator: {self.is_mothership}")

        # Auto-sync disabled for mothership
        self.auto_sync = getattr(self.config, 'autosync', False) and not self.is_mothership

        logger.info(
            f"Running validator for subnet: {self.config.netuid} with autosync set to: {self.auto_sync} "
            f"on network: {self.config.subtensor.chain_endpoint} with config:"
        )

        logger.info(self.config)

        # Created early so it can be passed for crash notifications.
        self.slack_notifier = SlackNotifier(
            hotkey=self.wallet.hotkey.ss58_address,
            webhook_url=getattr(self.config, 'slack_webhook_url', None),
            error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
            is_miner=False  # This is a validator
        )

        # Must run before any RPC servers are created. Fresh segment (unlink any stale one,
        # never reset it in place): a stale segment is still mapped by orphans of a killed
        # previous run, and a reset would revive them.
        ShutdownCoordinator.initialize_fresh()
        logger.info("[INIT] ShutdownCoordinator initialized (shared memory)")

        logger.info(f"Wallet: {self.wallet}")

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
            logger.info("[INIT] --orders-app: starting NO servers (pure client of vanta-state/core)")
        elif self.split_state:
            logger.info("[INIT] --split-state: core hosts its tier only; state servers run in vanta-state")
            orchestrator.start_core_servers(context)
            # Core's clients, daemons and (on mainnet) startup checks call the vanta-state servers.
            # Wait for them instead of crash-looping on connection-refused while vanta-state loads.
            orchestrator.wait_for_state_tier()
        else:
            orchestrator.start_validator_servers(context)
        logger.info("[INIT] Server startup phase complete")

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
            self.subtensor = bt.Subtensor(
                network=self.config.subtensor.chain_endpoint or self.config.subtensor.network
            )
            self.subtensor_ops_manager = None
            self._init_blacklist_hotkey_cache()
        else:
            subtensor_ops_server = orchestrator.get_server('subtensor_ops')
            self.subtensor = subtensor_ops_server.get_subtensor()
            self.subtensor_ops_manager = subtensor_ops_server.manager  # For blacklist_fn

        if self.orders_app:
            logger.info("[INIT] --orders-app: no server daemons / pre_run_setup (client of state tiers)")
        elif self.split_state:
            # Core starts only its own tier's daemons; vanta-state starts its own (incl. pre_run_setup).
            time.sleep(1)  # Stagger for servers to finish initailizing
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
            logger.info("[INIT] Core-tier daemons started (state daemons run in vanta-state)")
        else:
            # Single-process: core hosts everything (today's behavior).
            # Pre-run setup disabled on main (#879) - re-enable if needed
            # orchestrator.call_pre_run_setup(perform_order_corrections=True)
            time.sleep(1)  # Stagger for servers to finish initailizing
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
            logger.info("[INIT] All daemons started, caches warmed")

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
            logger.info("[INIT] --no-axon: HL tracker not started here (runs in vanta-orders app)")

        # Verify hotkey is registered (via metagraph_client — available in all roles).
        #
        # The metagraph server binds its RPC port at spawn but serves EMPTY data until core's
        # subtensor daemon pushes the chain metagraph (minutes on mainnet: perf-ledger load runs
        # first). The monolith/core paths block on wait_for_initial_update before reaching here;
        # the orders app has no such gate, so it must WAIT for population rather than read the
        # empty metagraph as "not registered" and exit() — that crash-looped vanta-orders into
        # PM2 'errored' on every full-split deploy, silently dropping all order reception.
        _mg_wait_deadline = time.time() + 600  # generous: core's boot is minutes on mainnet
        while True:
            hotkeys = self.metagraph_client.get_hotkeys()
            if hotkeys:
                break
            if time.time() > _mg_wait_deadline:
                # Still empty after the deadline — core is not populating. Crash loudly (PM2
                # restarts us and the wait restarts with a fresh window); do NOT claim the
                # hotkey is unregistered, we never saw the real metagraph.
                raise RuntimeError(
                    "Metagraph still empty after 600s — core has not populated it; cannot "
                    "verify hotkey registration. Will retry on restart."
                )
            logger.info("[INIT] Metagraph empty (core still booting) — waiting to verify hotkey registration...")
            time.sleep(5)
        logger.info(f"Metagraph n_entries: {len(hotkeys)}")
        if not self.metagraph_client.has_hotkey(self.wallet.hotkey.ss58_address):
            logger.error(
                f"\nYour validator hotkey: {self.wallet.hotkey.ss58_address} (wallet: {self.wallet.name}, hotkey: {self.wallet.hotkey_str}) "
                f"is not registered on netuid {self.config.netuid}. Run btcli register and try again."
            )
            exit()

        # orders-app already set self.subtensor above; otherwise take subtensor_ops_manager's.
        if self.subtensor_ops_manager is not None:
            self.subtensor = self.subtensor_ops_manager.get_subtensor()

        # Rate limiters for incoming requests. Must be set before wire_http_server() below: that
        # call starts the signed-request HTTP server and wires should_reject_synapse's handlers
        # (_receive_signal_sync et al.) live immediately, so a request can land before this
        # constructor returns.
        self.order_rate_limiter = RateLimiter()
        self.position_inspector_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 4)

        # Own ContractClient for receiving account openings (StandardMinerOpening route).
        self._contract_client = ContractClient(running_unit_tests=False)

        # Dedicated thread pool for concurrent synchronous signed-request handlers.
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)

        # Build and link vali functions to the signed-request HTTP server (replaces bt.Axon).
        self.http_server = None
        # --no-axon (serve_axon=False): this process does not serve the signed-request HTTP server —
        # the vanta-orders app does. Core-split runs with serve_axon=False; the default monolith and
        # vanta-orders run with serve_axon=True. The orders app ALWAYS serves it (that is its whole
        # job), even if --no-axon leaked into its args. self.http_server stays None otherwise so
        # guards can check it. (Flag name kept as --no-axon/serve_axon for operator continuity even
        # though this no longer wires up a bt.Axon — see shared_objects/http_server/.)
        if getattr(self.config, 'serve_axon', True) or self.orders_app:
            self.wire_http_server()
        else:
            logger.info("[INIT] --no-axon: signed-request HTTP server not served here (runs in vanta-orders app)")

        my_subnet_uid = self.metagraph_client.get_hotkeys().index(self.wallet.hotkey.ss58_address)
        logger.info(f"Running validator on uid: {my_subnet_uid}")

        logger.info("[INIT] All initialization steps completed successfully!")

        # Send success notification to Slack
        if self.slack_notifier:
            self.slack_notifier.send_message(
                f"✅ Validator Initialization Complete!\n"
                f"All initialization steps completed successfully\n"
                f"Hotkey: {self.wallet.hotkey.ss58_address}\n"
                f"API services: {'Enabled' if self.config.serve else 'Disabled'}",
                level="info"
            )

        # Validators on mainnet syncing for the first time or after interruption need to resync
        # their positions if the newest order on disk isn't within the last 24 hours.
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
                logger.info(f"Found {n_positions_on_disk} hotkeys with positions on disk."
                                f" Found oldest_disk_ms: {TimeUtil.millis_to_datetime(oldest_disk_ms)},"
                                f" youngest_disk_ms: {TimeUtil.millis_to_datetime(youngest_disk_ms)}")
            one_day_ago = TimeUtil.timestamp_to_millis(TimeUtil.generate_start_timestamp(days=1))
            if (n_positions_on_disk == 0 or youngest_disk_ms < one_day_ago):
                msg = ("Validator data needs to be synced with mainnet validators. "
                       "Restoring validator with 24 hour lagged file. More info here: "
                       "https://github.com/taoshidev/proprietary-trading-network/"
                       "blob/main/docs/regenerating_validator_state.md")
                logger.warning(msg)
                self.position_syncer.sync_positions(
                    False, candidate_data=self.position_syncer.read_validator_checkpoint_from_gcloud_zip())


    def wire_http_server(self):
        """Builds and starts the signed-request FastAPI server, replacing bt.Axon.

        Authentication (signature verification, dual-scheme: new btauth/1 + legacy
        bittensor v10 bridge) runs as middleware (shared_objects/http_server/app.py);
        authorization (registered-hotkey check, this subnet's blacklist_fn-equivalent)
        runs per-route via shared_objects/http_server/routes.py's register_handler.
        """
        http_port = self.config.http_port
        logger.info(f"setting http port [{http_port}]")

        verifier = AuthVerifier(
            self_hotkey_ss58=self.wallet.hotkey.ss58_address,
            initial_hotkeys=self.metagraph_client.get_hotkeys(),
        )
        app = build_app(verifier, vanta_protocol.LEGACY_HASH_FIELDS)

        # Each business_fn already has the (request, caller_hotkey) -> response shape
        # register_handler expects — no internal model, no request/response conversion.
        route_business_fns = {
            "SendSignal": self._receive_signal_sync,
            "GetPositions": self._get_positions,
            "CollateralRecord": self.miner_account_client.receive_collateral_record,
            "AssetSelection": self.asset_selection_client.receive_asset_selection,
            "SubaccountRegistration": self.entity_client.receive_subaccount_registration,
            "EntityEndpointUpdate": self.entity_client.receive_entity_endpoint_update,
            "StandardMinerOpening": self._contract_client.receive_standard_miner_opening,
            "StandardMinerRecord": self.miner_account_client.receive_standard_miner_record,
            "AccountSizeLocks": self.miner_account_client.receive_size_locks,
        }
        for message_name, business_fn in route_business_fns.items():
            request_model, response_model = vanta_protocol.MESSAGE_TYPES[message_name]
            register_handler(
                app, message_name, request_model, response_model,
                business_fn, self._thread_pool, self.is_caller_authorized,
            )

        self.http_server = HttpServer(app, host=self.config.http_host, port=http_port)
        self.http_server.start()
        logger.info(f"Started signed-request HTTP server on {self.config.http_host}:{http_port}")

        self.auth_verifier = verifier
        self._start_trusted_hotkey_refresh_loop()

        # Publish our IP:port on chain so other nodes can discover this server via the metagraph's
        # per-neuron axon field (replaces axon.serve()). Use subtensor lock to prevent WebSocket
        # concurrency errors with the metagraph_updater thread.
        public_ip = get_external_ip()
        logger.info(f"Publishing ServeAxon on network {self.config.subtensor.network} "
                    f"netuid {self.config.netuid}: {public_ip}:{http_port}")
        with get_subtensor_lock():
            result = self.subtensor.execute(
                bt.ServeAxon(netuid=self.config.netuid, ip=public_ip, port=http_port),
                self.wallet,
            )
        if not result.success:
            logger.error(f"ServeAxon failed: {result.error}")

    def _start_trusted_hotkey_refresh_loop(self, interval_s: float = 30.0):
        """bittensor.http_auth.configure_verifier's trusted_sender_schemes mapping is
        process-local, so each process that runs an HttpServer (core monolith or the
        vanta-orders split) must periodically re-sync it against the metagraph itself —
        a newly-registered hotkey otherwise can't be verified until this refreshes."""

        def _loop():
            while True:
                time.sleep(interval_s)
                try:
                    self.auth_verifier.refresh_trusted_hotkeys(self.metagraph_client.get_hotkeys())
                except Exception as e:
                    logger.warning(f"[http_server] failed to refresh trusted hotkeys: {e}")

        threading.Thread(target=_loop, daemon=True, name="auth-trusted-hotkey-refresh").start()

    def check_shutdown(self):
        if not is_shutdown():
            return
        # Handle shutdown gracefully
        logger.warning("Performing graceful exit...")

        # Send shutdown notification to Slack
        if self.slack_notifier:
            self.slack_notifier.send_message(
                f"🛑 Validator shutting down gracefully\n"
                f"Hotkey: {self.wallet.hotkey.ss58_address}",
                level="warning"
            )
        # bt.Axon is gone (replaced by the signed-request HTTP server in shared_objects/http_server/,
        # already-done prior migration) - self.axon was never set, this was dead/broken code left
        # over from the old axon lifecycle.
        # SubtensorOpsServer and all RPC servers shut down automatically via ShutdownCoordinator.
        # REST/WS run as separate PM2 apps (vanta-rest / vanta-ws) with their own shutdown handling.
        signal.alarm(0)
        print("Graceful shutdown completed")
        sys.exit(0)

    def main(self):
        logger.info("Starting main loop")

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

            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(error_traceback)

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

    def should_reject_synapse(self, sender_hotkey, response: SendSignalResponse | GetPositionsResponse, method: SynapseMethod, version: int = 0) -> bool:
        """Mutates `response` (the outgoing *Response being built by the caller) in place and
        reports whether the request should be rejected. `version` is only meaningful for
        SynapseMethod.POSITION_INSPECTOR, and is the caller's GetPositionsRequest.version
        (read off the request by the caller, since GetPositionsRequest/SendSignalRequest are
        no longer the same mutable object as the response)."""
        if is_shutdown():
            response.successfully_processed = False
            response.error_message = "Validator is restarting due to update. Please try again later."
            return True

        if method == SynapseMethod.POSITION_INSPECTOR:
            allowed, wait_time = self.position_inspector_rate_limiter.is_allowed(sender_hotkey)
        elif method == SynapseMethod.SIGNAL:
            allowed, wait_time = self.order_rate_limiter.is_allowed(sender_hotkey)
        else:
            msg = "Received synapse does not match one of expected methods for: receive_signal or get_positions"
            response.successfully_processed = False
            response.error_message = msg
            return True

        if not allowed:
            msg = f"Rate limited. Please wait {wait_time} seconds before sending another signal. {method.value}"
            response.successfully_processed = False
            response.error_message = msg
            return True

        if method == SynapseMethod.POSITION_INSPECTOR:
            if version == 0:
                response.successfully_processed = False
                response.error_message = "Please use the latest miner script that makes PI opt-in with the flag --run-position-inspector"
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
                    logger.warning(f"[orders-app] blacklist hotkey cache refresh failed: {e}")
                time.sleep(refresh_period_s)

        # Prime once (best-effort) so the axon isn't cold when it starts serving.
        try:
            self._blacklist_hotkeys = set(self.metagraph_client.get_hotkeys())
        except Exception as e:
            logger.warning(f"[orders-app] initial blacklist hotkey cache prime failed: {e}")
        threading.Thread(target=_refresh_loop, daemon=True, name="blacklist-hotkey-cache").start()

    @timeme
    def is_caller_authorized(self, caller_hotkey: str) -> None:
        """Registered-hotkey authorization check used by every signed-request route
        (replaces bt.Axon's blacklist_fn): a fast local set lookup — subtensor_ops_manager's
        in-process cache in the monolith/core, or the metagraph-fed cache in the orders app —
        instead of a per-request RPC call to MetagraphClient.has_hotkey(). Every signed-request
        route (shared_objects/http_server) pays this check once, so avoiding an RPC round-trip
        here matters under concurrent order load. Raises shared_objects.http_auth.Forbidden if
        unregistered."""
        if self.subtensor_ops_manager is not None:
            is_registered = self.subtensor_ops_manager.is_hotkey_registered_cached(caller_hotkey)
        else:
            with self._blacklist_cache_lock:
                is_registered = caller_hotkey in self._blacklist_hotkeys

        if not is_registered:
            logger.debug(f"Rejecting unregistered hotkey {caller_hotkey}")
            from shared_objects.http_auth import Forbidden
            raise Forbidden(f"hotkey {caller_hotkey} is not registered on this subnet's metagraph")

        logger.debug(f"Not rejecting registered hotkey {caller_hotkey}")


    def _receive_signal_sync(self, request: SendSignalRequest, caller_hotkey: str) -> SendSignalResponse:
        now_ms = TimeUtil.now_in_millis()
        order = None
        miner_hotkey = caller_hotkey
        subaccount_id = request.subaccount_id
        miner_repo_version = request.repo_version
        signal_dict = request.signal
        response = SendSignalResponse(validator_hotkey=self.wallet.hotkey.ss58_address)

        # Inlined from the old template.protocol.SendSignal.parse_miner_uuid (that class is gone).
        order_uuid = request.miner_order_uuid
        assert isinstance(order_uuid, str), f"excepted string miner uuid but got {order_uuid}"
        if not order_uuid:
            logger.warning(f'miner_order_uuid is empty for miner_hotkey [{miner_hotkey}] miner_repo_version '
                               f'[{miner_repo_version}]. Generating a new one.')
            order_uuid = str(uuid.uuid4())

        if not miner_hotkey:
            response.successfully_processed = False
            response.error_message = "Missing miner hotkey"
            response.should_retry = False
            return response

        # Entity miners: downstream ops use the synthetic hotkey when a subaccount is given.
        if subaccount_id is not None:
            miner_hotkey = f"{miner_hotkey}_{subaccount_id}"

        logger.info(f"received signal [{order_uuid}] [{signal_dict}] from miner_hotkey [{miner_hotkey}] using repo version [{miner_repo_version}].")

        if self.should_reject_synapse(miner_hotkey, response, SynapseMethod.SIGNAL):
            logger.info(f"Order rejected for {miner_hotkey}: {response.error_message}")
            return response

        # Advisory fast-path reject; the authoritative gate is begin_order() below.
        if self.order_sync.is_sync_waiting():
            response.successfully_processed = False
            response.should_retry = True
            response.error_message = "Validator is syncing positions. Please try again shortly."
            logger.info(f"Rejected order from {miner_hotkey}: {response.error_message}")
            return response

        try:
            signal = Signal.model_validate(signal_dict)
        except Exception as e:
            response.successfully_processed = False
            response.should_retry = False
            response.error_message = f"Invalid signal payload: {e}"
            return response

        # exists() is an advisory local pre-filter; check_and_add below is the authoritative claim
        # (it also catches a lost-ack retry across a restart, when the local cache is cold).
        should_dedup = (
            bool(order_uuid)
            and signal.execution_type not in (ExecutionType.LIMIT_CANCEL, ExecutionType.LIMIT_EDIT, ExecutionType.FLAT_ALL)
        )
        if should_dedup and self.uuid_tracker.exists(order_uuid):
            response.successfully_processed = False
            response.should_retry = False
            response.error_message = f"Order with uuid [{order_uuid}] has already been processed. Please try again with a new order."
            logger.error(response.error_message)
            return response

        ok, error_msg, resolved_tp = self.order_processor.validate(
            hotkey=miner_hotkey,
            execution_type=signal.execution_type,
            trade_pair=signal.trade_pair,
            order_type=signal.order_type,
        )
        if not ok:
            response.successfully_processed = False
            response.should_retry = False
            response.error_message = error_msg
            return response

        if resolved_tp is not None and resolved_tp != signal.trade_pair:
            signal = signal.model_copy(update={"trade_pair": resolved_tp})

        # begin_order registers the in-flight order (sync waits for it). It also refuses (rejected)
        # if a sync started since the advisory check above, so an order can't apply mid-rewrite;
        # the in-memory OrderSyncState never rejects (rejected=False) and relies on that check.
        with self.order_sync.begin_order() as _admission:
            if getattr(_admission, 'rejected', False):
                response.successfully_processed = False
                response.should_retry = True
                response.error_message = "Validator is syncing positions. Please try again shortly."
                logger.info(f"Rejected order from {miner_hotkey}: sync in progress")
                return response

            # error message to send back to miners in case of a problem so they can fix and resend
            error_message = ""
            order_exc = None
            # Claim the uuid BEFORE applying so a duplicate can't double-apply; released below if the
            # apply fails (so a transient-error retry can re-claim) or the result is non-trackable.
            if should_dedup and not self.uuid_tracker.check_and_add(order_uuid):
                response.successfully_processed = False
                response.should_retry = False
                response.error_message = f"Order with uuid [{order_uuid}] has already been processed. Please try again with a new order."
                logger.error(response.error_message)
                return response
            claimed = should_dedup
            try:
                result = self.order_processor.process_vanta_signal(
                    hotkey=miner_hotkey,
                    signal=signal,
                    order_uuid=order_uuid,
                    now_ms=now_ms,
                )

                # Set response's order_json (centralized - single line instead of 4)
                response.order_json = result.get_response_json()

                # The claim was taken before we knew the result; drop it if this order isn't tracked,
                # else promote it to permanent (a provisional claim expires after
                # ORDER_UUID_CLAIM_TTL_MS so a hard-killed apply can't strand it — see
                # CommonDataServer.confirm_order_uuid_rpc).
                if claimed and not result.should_track_uuid:
                    self.uuid_tracker.release(order_uuid)
                    claimed = False
                elif claimed:
                    self.uuid_tracker.confirm(order_uuid)

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
                logger.error(f"Error processing signal {miner_hotkey} {order_uuid} {e}")

            finally:
                response.error_message = error_message
                if error_message == "":
                    response.successfully_processed = True
                else:
                    logger.error(error_message)
                    response.successfully_processed = False
                    # Transient infra failures (an RPC state server — incl. MarketOrderServer
                    # :50027 — bouncing during a deploy) must be RETRIED by the placer, not
                    # treated as a permanent rejection that loses the order. Business-logic
                    # rejections (SignalException etc.) stay should_retry=False.
                    response.should_retry = ErrorUtils.is_transient_rpc_error(order_exc)

                # TODO Review overlap with serving in market order manager
                # Best-effort: the entity server lives in CORE — during a core restart this call
                # raising inside `finally` would REPLACE the fully-crafted success response with an
                # axon error, and the still-claimed uuid would turn the miner's retry into a hard
                # duplicate rejection for an order that is live on their account.
                if is_synthetic_hotkey(miner_hotkey):
                    try:
                        self.entity_client.broadcast_subaccount_dashboard(miner_hotkey)
                    except Exception as broadcast_err:
                        logger.warning(
                            f"Entity dashboard broadcast failed for {miner_hotkey} (ack unaffected): {broadcast_err}"
                        )

                processing_time_ms = TimeUtil.now_in_millis() - now_ms
                logger.info(f"Sending ack back to miner [{miner_hotkey}]. Synapse Message: {response.error_message}. "
                                   f"Process time {processing_time_ms}ms. order {order}")

                # Context manager auto-decrements counter and notifies waiters on exit

        return response

    def _get_positions(self, request: GetPositionsRequest, caller_hotkey: str) -> GetPositionsResponse:
        miner_hotkey = caller_hotkey
        response = GetPositionsResponse()
        if self.should_reject_synapse(miner_hotkey, response, SynapseMethod.POSITION_INSPECTOR, version=request.version):
            return response
        t0 = time.time()
        error_message = ""
        n_positions_sent = 0
        hotkey = miner_hotkey
        try:
            # Return the last n positions using PositionManagerClient
            positions = self.position_manager_client.get_positions_for_one_hotkey(hotkey, only_open_positions=True)
            response.positions = [position.to_dict() for position in positions]
            n_positions_sent = len(response.positions)
        except Exception as e:
            error_message = f"Error in GetPositions for [{miner_hotkey}] with error [{e}]. Perhaps the position was being written to disk at the same time."
            logger.error(traceback.format_exc())

        if error_message == "":
            response.successfully_processed = True
        else:
            logger.error(error_message)
            response.successfully_processed = False
        response.error_message = error_message
        msg = f"Sending {n_positions_sent} positions back to miner: {hotkey} in {round(time.time() - t0, 3)} seconds."
        if response.error_message:
            msg += f" Error: {response.error_message}"
        logger.info(msg)
        return response

if __name__ == "__main__":
    validator = Validator()
    validator.main()
