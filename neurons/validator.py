# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import json
import os
import sys
import threading
import signal
from typing import Tuple

from setproctitle import setproctitle

from vanta_api.api_manager import APIManager
from neurons.validator_base import ValidatorBase
from shared_objects.metagraph_manager import MetagraphManager
from multiprocessing import Manager, Process
from enum import Enum

import template
import traceback
import time
import bittensor as bt

from runnable.generate_request_core import RequestCoreManager
from runnable.generate_request_minerstatistics import MinerStatisticsManager
from runnable.generate_request_outputs import RequestOutputGenerator
from template.protocol import SendSignal
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.utils.auto_sync import PositionSyncer
from vali_objects.utils.limit_order_manager import LimitOrderManagerClient
from vali_objects.utils.market_order_manager import MarketOrderManager
from vali_objects.utils.p2p_syncer import P2PSyncer
from shared_objects.rate_limiter import RateLimiter
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.uuid_tracker import UUIDTracker
from time_util.time_util import TimeUtil, timeme
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.metagraph_updater import MetagraphUpdater
from shared_objects.error_utils import ErrorUtils
from miner_objects.slack_notifier import SlackNotifier
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcherClient
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.debt_ledger import DebtLedgerManager
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig

from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.asset_selection_manager import AssetSelectionManager
from vali_objects.utils.order_processor import OrderProcessor

# Global flag used to indicate shutdown
shutdown_dict = {}

# Enum class that represents the method associated with Synapse
class SynapseMethod(Enum):
    POSITION_INSPECTOR = "GetPositions"
    SIGNAL = "SendSignal"
    CHECKPOINT = "SendCheckpoint"

def signal_handler(signum, frame):
    global shutdown_dict

    if shutdown_dict:
        return  # Ignore if already in shutdown

    if signum in (signal.SIGINT, signal.SIGTERM):
        signal_message = "Handling SIGINT" if signum == signal.SIGINT else "Handling SIGTERM"
        print(f"{signal_message} - Initiating graceful shutdown")

        shutdown_dict[True] = True
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

        # 1. Initialize Managers for shared state
        # PERFORMANCE OPTIMIZATION: Use separate managers to reduce IPC contention
        # Each manager runs in its own process with its own server thread
        self.ipc_manager = Manager()  # General-purpose manager for queues, values, etc.
        bt.logging.info(f"[IPC] Created IPC manager: general (PID: {self.ipc_manager._process.pid})")

        self.shared_queue_websockets = self.ipc_manager.Queue()

        # Create shared sync_in_progress flag for cross-process synchronization
        # When True, daemon processes should pause to allow position sync to complete
        self.sync_in_progress = self.ipc_manager.Value('b', False)

        # Sync epoch counter: incremented each time auto sync runs
        # Managers capture this at START of iteration and check before saving
        # If epoch changed during iteration, data is stale and save is aborted
        self.sync_epoch = self.ipc_manager.Value('i', 0)

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
        self.wallet = bt.wallet(config=self.config)

        # Initialize Slack notifier for error reporting
        # Created before LivePriceFetcher so it can be passed for crash notifications
        self.slack_notifier = SlackNotifier(
            hotkey=self.wallet.hotkey.ss58_address,
            webhook_url=getattr(self.config, 'slack_webhook_url', None),
            error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
            is_miner=False  # This is a validator
        )

        # Create LivePriceFetcher client with Slack notifier for crash notifications
        self.live_price_fetcher = LivePriceFetcherClient(
            secrets=self.secrets,
            disable_ws=False,
            slack_notifier=self.slack_notifier
        )

        self.price_slippage_model = PriceSlippageModel(live_price_fetcher=self.live_price_fetcher)

        # Track last error notification time to prevent spam
        self.last_error_notification_time = 0
        self.error_notification_cooldown = 300  # 5 minutes between error notifications

        bt.logging.info(f"Wallet: {self.wallet}")

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
        # Uses RPC-based MetagraphManager with server-side hotkeys_set cache for O(1) has_hotkey() lookups
        self.metagraph = MetagraphManager(running_unit_tests=False, slack_notifier=self.slack_notifier)

        # Create single weight request queue (validator only)
        weight_request_queue = self.ipc_manager.Queue()

        # Create MetagraphUpdater with simple parameters (no PTNManager)
        # This will run in a thread in the main process
        self.metagraph_updater = MetagraphUpdater(
            self.config, self.metagraph, self.wallet.hotkey.ss58_address,
            False, position_manager=None,
            shutdown_dict=shutdown_dict,
            slack_notifier=self.slack_notifier,
            weight_request_queue=weight_request_queue,
            live_price_fetcher=self.live_price_fetcher
        )
        self.subtensor = self.metagraph_updater.subtensor
        bt.logging.info(f"Subtensor: {self.subtensor}")


        # Start the metagraph updater and wait for initial population
        self.metagraph_updater_thread = self.metagraph_updater.start_and_wait_for_initial_update(
            max_wait_time=60,
            slack_notifier=self.slack_notifier
        )

        # Initialize ValidatorContractManager for collateral operations
        self.contract_manager = ValidatorContractManager(config=self.config, position_manager=None, ipc_manager=self.ipc_manager, metagraph=self.metagraph)


        # Create EliminationManager with RPC address injection (NO circular dependency)
        # Pass ChallengePeriodManager's RPC server address for client connection
        self.elimination_manager = EliminationManager(
            self.metagraph,
            None,  # position_manager set after self.pm creation
            challengeperiod_rpc_address=("localhost", ValiConfig.RPC_CHALLENGEPERIOD_PORT),
            shutdown_dict=shutdown_dict,
            use_ipc=True,
            shared_queue_websockets=self.shared_queue_websockets,
            contract_manager=self.contract_manager,
            sync_in_progress=self.sync_in_progress,
            slack_notifier=self.slack_notifier,
            sync_epoch=self.sync_epoch,
            start_server=False  # Defer server start until dependencies are ready
        )

        self.asset_selection_manager = AssetSelectionManager(config=self.config, metagraph=self.metagraph)

        # Asset selection local cache for fast validation (saves 81ms per order!)
        # Refreshed by background daemon - NO per-order overhead
        # Dict reference assignment is atomic in Python - no lock needed
        self._asset_selections_cache = {}  # {hotkey: TradePairCategory}

        # Start background daemon to refresh asset selections cache
        self._start_asset_cache_refresh_daemon()

        self.position_syncer = PositionSyncer(shutdown_dict=shutdown_dict, signal_sync_lock=self.signal_sync_lock,
                                              signal_sync_condition=self.signal_sync_condition,
                                              n_orders_being_processed=self.n_orders_being_processed,
                                              ipc_manager=self.ipc_manager,
                                              position_manager=None,
                                              auto_sync_enabled=self.auto_sync,
                                              contract_manager=self.contract_manager,
                                              sync_in_progress=self.sync_in_progress,
                                              sync_epoch=self.sync_epoch,
                                              # live_price_fetcher=self.live_price_fetcher,
                                              asset_selection_manager=self.asset_selection_manager)  # Set after self.pm creation

        self.p2p_syncer = P2PSyncer(wallet=self.wallet, metagraph=self.metagraph, is_testnet=not self.is_mainnet,
                                    shutdown_dict=shutdown_dict, signal_sync_lock=self.signal_sync_lock,
                                    signal_sync_condition=self.signal_sync_condition,
                                    n_orders_being_processed=self.n_orders_being_processed,
                                    ipc_manager=self.ipc_manager,
                                    position_manager=None)  # Set after self.pm creation


        self.perf_ledger_manager = PerfLedgerManager(self.metagraph, ipc_manager=self.ipc_manager,
                                                     shutdown_dict=shutdown_dict,
                                                     perf_ledger_hks_to_invalidate=self.position_syncer.perf_ledger_hks_to_invalidate,
                                                     position_manager=None,
                                                     contract_manager=self.contract_manager)  # Set after self.pm creation)


        self.position_manager = PositionManager(metagraph=self.metagraph,
                                                perform_order_corrections=True,
                                                live_price_fetcher=self.live_price_fetcher,
                                                perf_ledger_manager=self.perf_ledger_manager,
                                                elimination_manager=self.elimination_manager,
                                                challengeperiod_manager=None,
                                                secrets=self.secrets,
                                                shared_queue_websockets=self.shared_queue_websockets,
                                                closed_position_daemon=True)

        # Use RPC mode for production (centralized lock server, better performance)
        # Falls back to local threading locks for tests/single process
        self.position_locks = PositionLocks(
            hotkey_to_positions=self.position_manager.get_positions_for_all_miners(),
            mode='rpc'  # Options: 'local' (tests), 'ipc' (old), 'rpc' (recommended)
        )

        # Set position_locks on elimination_manager now that it exists
        self.elimination_manager.position_locks = self.position_locks

        self.plagiarism_manager = PlagiarismManager(slack_notifier=self.slack_notifier,
                                                    ipc_manager=self.ipc_manager)

        self.market_order_manager = MarketOrderManager(self.live_price_fetcher, self.position_locks, self.price_slippage_model,
               self.config, self.position_manager, self.shared_queue_websockets, self.contract_manager)

        # Attach the position manager to the other objects that need it (BEFORE creating ChallengePeriodManager)
        for idx, obj in enumerate([self.perf_ledger_manager, self.position_manager, self.position_syncer,
                                   self.p2p_syncer, self.elimination_manager, self.metagraph_updater,
                                   self.contract_manager]):
            obj.position_manager = self.position_manager

        self.position_manager.perf_ledger_manager = self.perf_ledger_manager

        #force_validator_to_restore_from_checkpoint(self.wallet.hotkey.ss58_address, self.metagraph, self.config, self.secrets)

        # RPC ADDRESS INJECTION PATTERN - No circular dependencies!
        #
        # Architecture:
        # - EliminationManager has challengeperiod_rpc_address=("localhost", 50003)
        # - ChallengePeriodManager has elimination_rpc_address=("localhost", 50004)
        # - Each server creates its own RPC client to communicate with peer
        # - No object references passed between managers
        # - Servers can start in any order (no initialization dependency)
        #
        # Benefits over old approach:
        # 1. NO circular dependencies (just simple address tuples)
        # 2. NO pickle issues (tuples pickle normally)
        # 3. Order independent initialization (servers start independently)
        # 4. Clean separation of concerns (RPC protocol defines interface)
        # 5. Easy to test (mock RPC clients in tests)
        #
        # Initialization sequence:
        # 1. Both managers already created with RPC addresses (line 221 for elimination_manager)
        # 2. Create ChallengePeriodManager with elimination_rpc_address
        # 3. Wire up position_manager dependency
        # 4. Initialize ChallengePeriodManager server (creates elim_client internally)
        # 5. Initialize EliminationManager server (creates cp_client internally)
        # 6. Start both daemons

        # Step 1: Create ChallengePeriodManager with RPC address injection (NO circular dependency)
        # Pass EliminationManager's RPC server address for client connection
        self.challengeperiod_manager = ChallengePeriodManager(
            self.metagraph,
            perf_ledger_manager=self.perf_ledger_manager,
            position_manager=self.position_manager,
            contract_manager=self.contract_manager,
            plagiarism_manager=self.plagiarism_manager,
            asset_selection_manager=self.asset_selection_manager,
            elimination_rpc_address=("localhost", ValiConfig.RPC_ELIMINATION_PORT),
            sync_in_progress=self.sync_in_progress,
            slack_notifier=self.slack_notifier,
            sync_epoch=self.sync_epoch,
            start_server=False,  # Defer server initialization
            start_daemon=False   # Defer daemon start
        )
        bt.logging.info("[INIT] ChallengePeriodManager created with elimination_rpc_address (no circular dependency)")

        # Step 2: Wire up position_manager dependency only (NO circular dependency with elimination_manager!)
        self.position_manager.challengeperiod_manager = self.challengeperiod_manager
        bt.logging.info("[INIT] position_manager.challengeperiod_manager wired up")

        # Step 3: Initialize ChallengePeriodManager server
        # Server will create RPC client to EliminationManager automatically
        self.challengeperiod_manager.initialize_server()
        bt.logging.success("[INIT] ChallengePeriodManager server initialized (has elim_client for RPC calls)")

        # Step 4: Initialize EliminationManager server
        # Server will create RPC client to ChallengePeriodManager automatically
        self.elimination_manager.initialize_server()
        bt.logging.success("[INIT] EliminationManager server initialized (has cp_client for RPC calls)")

        # Step 5: Start both daemons
        self.elimination_manager.start_daemon()
        bt.logging.success("[INIT] EliminationManager daemon started")

        self.challengeperiod_manager._start_daemon_process()
        bt.logging.success("[INIT] ChallengePeriodManager daemon started")

        # Create LimitOrderManagerClient AFTER elimination_manager server is initialized
        # This ensures elimination_manager._server_proxy is set before pickling
        self.limit_order_manager = LimitOrderManagerClient(
            position_manager=self.position_manager,
            live_price_fetcher=self.live_price_fetcher,
            market_order_manager=self.market_order_manager,
            shutdown_dict=shutdown_dict,
            running_unit_tests=False
        )

        # Set limit_order_manager on elimination_manager now that it exists
        self.elimination_manager.limit_order_manager = self.limit_order_manager

        # Set limit_order_manager on position_syncer
        self.position_syncer.limit_order_manager = self.limit_order_manager

        self.position_manager.pre_run_setup()
        self.uuid_tracker.add_initial_uuids(self.position_manager.get_positions_for_all_miners())

        self.debt_ledger_manager = DebtLedgerManager(self.perf_ledger_manager, self.position_manager, self.contract_manager,
                                     self.asset_selection_manager, challengeperiod_manager=self.challengeperiod_manager,
                                     slack_webhook_url=self.config.slack_error_webhook_url, start_server=True,
                                     ipc_manager=self.ipc_manager, validator_hotkey=self.wallet.hotkey.ss58_address)


        self.checkpoint_lock = threading.Lock()
        self.encoded_checkpoint = ""
        self.last_checkpoint_time = 0

        bt.logging.info(f"Metagraph n_entries: {len(self.metagraph.get_hotkeys())}")
        if not self.metagraph.has_hotkey(self.wallet.hotkey.ss58_address):
            bt.logging.error(
                f"\nYour validator hotkey: {self.wallet.hotkey.ss58_address} (wallet: {self.wallet.name}, hotkey: {self.wallet.hotkey_str}) "
                f"is not registered to chain connection: {self.metagraph_updater.get_subtensor()} \n"
                f"Run btcli register and try again. "
            )
            exit()

        # Build and link vali functions to the axon.
        # The axon handles request processing, allowing validators to send this process requests.
        super().__init__(wallet=self.wallet, slack_notifier=self.slack_notifier, config=self.config,
                         metagraph=self.metagraph, p2p_syncer=self.p2p_syncer, contract_manager=self.contract_manager,
                         asset_selection_manager=self.asset_selection_manager)

        self.order_rate_limiter = RateLimiter()
        self.position_inspector_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 4)
        self.checkpoint_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 60 * 6)


        # Step 1: Initialize PlagiarismDetector
        def step1():
            self.plagiarism_detector = PlagiarismDetector(self.metagraph, shutdown_dict=shutdown_dict,
                                                          position_manager=self.position_manager)
            return self.plagiarism_detector
        self.run_init_step_with_monitoring(1, "Initializing PlagiarismDetector", step1)

        # Step 2: Start plagiarism detector process
        def step2():
            self.plagiarism_thread = Process(target=self.plagiarism_detector.run_update_loop, daemon=True)
            self.plagiarism_thread.start()
            # Verify process started
            time.sleep(0.1)  # Give process a moment to start
            if not self.plagiarism_thread.is_alive():
                raise RuntimeError("Plagiarism detector process failed to start")
            bt.logging.info(f"Process started with PID: {self.plagiarism_thread.pid}")
            return self.plagiarism_thread
        self.run_init_step_with_monitoring(2, "Starting plagiarism detector process", step2)

        # Step 3: Initialize MDDChecker with auto-start daemon
        # MDDChecker runs in its own daemon process and updates position prices in local memory
        # No IPC required - all updates happen on local Python dicts within the MDDChecker process
        def step3():
            self.mdd_checker = MDDChecker(self.metagraph, self.position_manager, live_price_fetcher=self.live_price_fetcher,
                                          shutdown_dict=shutdown_dict, position_locks=self.position_locks,
                                          sync_in_progress=self.sync_in_progress, slack_notifier=self.slack_notifier,
                                          sync_epoch=self.sync_epoch, start_daemon=True)
            return self.mdd_checker
        self.run_init_step_with_monitoring(3, "Initializing MDDChecker with daemon", step3)

        # Step 5 & 6: RPC managers (EliminationManager and ChallengePeriodManager)
        # are automatically started in their __init__ methods via _initialize_service()
        # Both EliminationManager and ChallengePeriodManager daemon processes started via start_daemon=True
        bt.logging.info("Step 5-6: EliminationManager and ChallengePeriodManager RPC servers and daemons already started")

        # Step 7: Initialize SubtensorWeightSetter
        def step7():
            # Pass shared metagraph which contains substrate reserves refreshed by MetagraphUpdater
            # Pass debt_ledger_manager for encapsulated access to debt ledger data
            self.weight_setter = SubtensorWeightSetter(
                self.metagraph,
                position_manager=self.position_manager,
                use_slack_notifier=True,
                shutdown_dict=shutdown_dict,
                weight_request_queue=weight_request_queue,  # Same queue as MetagraphUpdater
                config=self.config,
                hotkey=self.wallet.hotkey.ss58_address,
                contract_manager=self.contract_manager,
                debt_ledger_manager=self.debt_ledger_manager,
                is_mainnet=self.is_mainnet
            )
            return self.weight_setter
        self.run_init_step_with_monitoring(7, "Initializing SubtensorWeightSetter", step7)

        # Step 8: Initialize RequestCoreManager and MinerStatisticsManager
        def step8():
            self.request_core_manager = RequestCoreManager(self.position_manager, self.weight_setter, self.plagiarism_detector,
                                                          contract_manager=self.contract_manager, limit_order_manager=self.limit_order_manager, ipc_manager=self.ipc_manager,
                                                          asset_selection_manager=self.asset_selection_manager)
            self.miner_statistics_manager = MinerStatisticsManager(self.position_manager, self.weight_setter,
                                                                   self.plagiarism_detector, contract_manager=self.contract_manager,
                                                                   ipc_manager=self.ipc_manager)
            return (self.request_core_manager, self.miner_statistics_manager)
        self.run_init_step_with_monitoring(8, "Initializing RequestCoreManager and MinerStatisticsManager", step8)

        # Step 9: Start perf ledger updater process
        def step9():
            self.perf_ledger_updater_thread = Process(target=self.perf_ledger_manager.run_update_loop, daemon=True)
            self.perf_ledger_updater_thread.start()
            # Verify process started
            time.sleep(0.1)  # Give process a moment to start
            if not self.perf_ledger_updater_thread.is_alive():
                raise RuntimeError("Perf ledger updater process failed to start")
            bt.logging.info(f"Process started with PID: {self.perf_ledger_updater_thread.pid}")
            return self.perf_ledger_updater_thread
        self.run_init_step_with_monitoring(9, "Starting perf ledger updater process", step9)

        # Step 10: Start weight setter process
        def step10():
            self.weight_setter_process = Process(target=self.weight_setter.run_update_loop, daemon=True)
            self.weight_setter_process.start()
            # Verify process started
            time.sleep(0.1)  # Give process a moment to start
            if not self.weight_setter_process.is_alive():
                raise RuntimeError("Weight setter process failed to start")
            bt.logging.info(f"Process started with PID: {self.weight_setter_process.pid}")
            return self.weight_setter_process
        self.run_init_step_with_monitoring(10, "Starting weight setter process", step10)

        # Step 11: Start weight processing thread
        def step11():
            if self.metagraph_updater.weight_request_queue:
                self.weight_processing_thread = threading.Thread(target=self.metagraph_updater.run_weight_processing_loop, daemon=True)
                self.weight_processing_thread.start()
                # Verify thread started
                time.sleep(0.1)
                if not self.weight_processing_thread.is_alive():
                    raise RuntimeError("Weight processing thread failed to start")
                return self.weight_processing_thread
            else:
                bt.logging.info("No weight request queue - skipping")
                return None
        self.run_init_step_with_monitoring(11, "Starting weight processing thread", step11)

        # Step 12: Start request output generator (if enabled)
        def step12():
            if self.config.start_generate:
                self.rog = RequestOutputGenerator(rcm=self.request_core_manager, msm=self.miner_statistics_manager)
                self.rog_thread = threading.Thread(target=self.rog.start_generation, daemon=True)
                self.rog_thread.start()
                # Verify thread started
                time.sleep(0.1)
                if not self.rog_thread.is_alive():
                    raise RuntimeError("Request output generator thread failed to start")
                return self.rog_thread
            else:
                self.rog_thread = None
                bt.logging.info("Request output generator not enabled - skipping")
                return None
        self.run_init_step_with_monitoring(12, "Starting request output generator (if enabled)", step12)

        # Step 13: Start API services (if enabled)
        def step13():
            if self.config.serve:
                # Create API Manager with configuration options
                self.api_manager = APIManager(
                    shared_queue=self.shared_queue_websockets,
                    ws_host=self.config.api_host,
                    ws_port=self.config.api_ws_port,
                    rest_host=self.config.api_host,
                    rest_port=self.config.api_rest_port,
                    position_manager=self.position_manager,
                    contract_manager=self.contract_manager,
                    miner_statistics_manager=self.miner_statistics_manager,
                    request_core_manager=self.request_core_manager,
                    asset_selection_manager=self.asset_selection_manager,
                    slack_webhook_url=self.config.slack_webhook_url,
                    debt_ledger_manager=self.debt_ledger_manager,
                    validator_hotkey=self.wallet.hotkey.ss58_address,
                    limit_order_manager=self.limit_order_manager
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
                return self.api_thread
            else:
                self.api_thread = None
                bt.logging.info("API services not enabled - skipping")
                return None
        self.run_init_step_with_monitoring(13, "Starting API services (if enabled)", step13)

        # Step 14: LivePriceFetcher RPC health checker
        # Health checking is automatically started in LivePriceFetcherClient.__init__ via RPCServiceBase
        # with enable_health_check=True, health_check_interval_s=60, enable_auto_restart=True
        bt.logging.info("Step 14: LivePriceFetcher RPC health checker already started")

        # Step 15: Start price slippage feature refresher process
        def step15():
            self.slippage_refresher = PriceSlippageModel.FeatureRefresher(
                price_slippage_model=self.price_slippage_model,
                slack_notifier=self.slack_notifier
            )
            self.slippage_refresher_process = Process(target=self.slippage_refresher.run_update_loop, daemon=True)
            self.slippage_refresher_process.start()
            # Verify process started
            time.sleep(0.1)
            if not self.slippage_refresher_process.is_alive():
                raise RuntimeError("Slippage refresher process failed to start")
            bt.logging.info(f"Slippage refresher process started with PID: {self.slippage_refresher_process.pid}")
            return self.slippage_refresher_process
        self.run_init_step_with_monitoring(15, "Starting price slippage feature refresher process", step15)

        # Signal watchdog that initialization is complete
        self.init_watchdog['current_step'] = 16
        bt.logging.info("[INIT] All 15 initialization steps completed successfully!")

        # Send success notification to Slack
        if self.slack_notifier:
            self.slack_notifier.send_message(
                f"✅ Validator Initialization Complete!\n"
                f"All 15 initialization steps completed successfully\n"
                f"Hotkey: {self.wallet.hotkey.ss58_address}\n"
                f"API services: {'Enabled' if self.config.serve else 'Disabled'}",
                level="info"
            )

        # Validators on mainnet net to be syned for the first time or after interruption need to resync their
        # positions. Assert there are existing orders that occurred > 24hrs in the past. Assert that the newest order
        # was placed within 24 hours.
        if self.is_mainnet:
            n_positions_on_disk = self.position_manager.get_number_of_miners_with_any_positions()
            oldest_disk_ms, youngest_disk_ms = (
                self.position_manager.get_extreme_position_order_processed_on_disk_ms())
            if (n_positions_on_disk > 0):
                bt.logging.info(f"Found {n_positions_on_disk} positions on disk."
                                f" Found oldest_disk_ms: {TimeUtil.millis_to_datetime(oldest_disk_ms)},"
                                f" oldest_disk_ms: {TimeUtil.millis_to_datetime(youngest_disk_ms)}")
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
        global shutdown_dict
        if not shutdown_dict:
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
        bt.logging.warning("Stopping live price fetcher...")
        self.live_price_fetcher.stop_all_threads()
        bt.logging.warning("Stopping perf ledger...")
        self.perf_ledger_updater_thread.join()
        bt.logging.warning("Stopping weight setter...")
        self.weight_setter_process.join()
        if hasattr(self, 'weight_processing_thread'):
            bt.logging.warning("Stopping weight processing thread...")
            self.weight_processing_thread.join()
        bt.logging.warning("Stopping plagiarism detector...")
        self.plagiarism_thread.join()
        # MDDChecker daemon shuts down automatically via shutdown_dict
        # ChallengePeriodManager daemon shuts down automatically via shutdown_dict
        # EliminationManager and ChallengePeriodManager RPC servers shutdown automatically via shutdown_dict
        # LivePriceFetcher RPC health checker shuts down automatically via RPCServiceBase
        bt.logging.warning("Stopping slippage refresher...")
        self.slippage_refresher_process.join()
        if self.rog_thread:
            bt.logging.warning("Stopping request output generator...")
            self.rog_thread.join()
        if self.api_thread:
            bt.logging.warning("Stopping API manager...")
            self.api_thread.join()
        signal.alarm(0)
        print("Graceful shutdown completed")
        sys.exit(0)

    def main(self):
        global shutdown_dict
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
        while not shutdown_dict:
            try:
                self.position_syncer.sync_positions_with_cooldown(self.auto_sync)
                # All managers now run in their own daemon processes

            # In case of unforeseen errors, the validator will log the error and send notification to Slack
            except Exception as e:
                error_traceback = traceback.format_exc()
                bt.logging.error(error_traceback)

                # Send error notification to Slack with rate limiting
                current_time_seconds = time.time()
                if self.slack_notifier and (current_time_seconds - self.last_error_notification_time) > self.error_notification_cooldown:
                    self.last_error_notification_time = current_time_seconds

                    # Use shared error formatting utility
                    error_message = ErrorUtils.format_error_for_slack(
                        error=e,
                        traceback_str=error_traceback,
                        include_operation=True,
                        include_timestamp=True
                    )

                    self.slack_notifier.send_message(
                        f"❌ Validator main loop error!\n"
                        f"{error_message}\n"
                        f"Note: Further errors suppressed for {self.error_notification_cooldown/60:.0f} minutes",
                        level="error"
                    )

                time.sleep(10)

        self.check_shutdown()

    def _asset_cache_refresh_loop(self):
        """
        Background daemon that refreshes asset selections cache every 5 seconds.

        This ensures the cache is always fresh without blocking order processing.
        Runs in a separate thread so NO order ever pays the refresh penalty.
        """
        setproctitle("vali_AssetCacheRefresher")
        bt.logging.info("Asset selections cache refresh daemon started (5-second interval)")

        while not shutdown_dict:
            try:
                sync_start = time.perf_counter()
                # Fetch all selections via RPC (happens in background thread)
                new_cache = self.asset_selection_manager.asset_selections
                sync_ms = (time.perf_counter() - sync_start) * 1000

                # Atomic cache update (dict reference assignment is atomic in Python)
                self._asset_selections_cache = new_cache

                bt.logging.debug(
                    f"[ASSET_CACHE] Refreshed {len(new_cache)} asset selections in {sync_ms:.2f}ms"
                )
                time.sleep(5)  # Refresh every 5 seconds

            except Exception as e:
                bt.logging.error(f"Error in asset cache refresh daemon: {e}")
                time.sleep(5)

        bt.logging.info("Asset selections cache refresh daemon shutting down")

    def _start_asset_cache_refresh_daemon(self):
        """Start the background asset cache refresh thread."""
        refresh_thread = threading.Thread(target=self._asset_cache_refresh_loop, daemon=True)
        refresh_thread.start()
        bt.logging.info("Started asset selections cache refresh daemon")

    def should_fail_early(self, synapse: template.protocol.SendSignal | template.protocol.GetPositions | template.protocol.ValidatorCheckpoint, method:SynapseMethod,
                          signal:dict=None, now_ms=None) -> bool:
        global shutdown_dict
        if shutdown_dict:
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
        # Single cache call: check elimination AND get info (no RPC call!) - saves 66.81ms per order
        elim_check_start = time.perf_counter()
        elimination_info = self.elimination_manager.get_elimination_cached(synapse.dendrite.hotkey)
        elim_check_ms = (time.perf_counter() - elim_check_start) * 1000
        bt.logging.info(f"[FAIL_EARLY_DEBUG] get_elimination_cached took {elim_check_ms:.2f}ms")

        if elimination_info:
            msg = f"This miner hotkey {synapse.dendrite.hotkey} has been eliminated and cannot participate in this subnet. Try again after re-registering. elimination_info {elimination_info}"
            bt.logging.debug(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        # don't process re-registered miners
        # Fast local lookup from elimination_manager cache (no RPC call!) - saves 11.26ms per order
        rereg_check_start = time.perf_counter()
        rereg_info = self.elimination_manager.get_departed_hotkey_info_cached(synapse.dendrite.hotkey)
        rereg_check_ms = (time.perf_counter() - rereg_check_start) * 1000
        bt.logging.info(f"[FAIL_EARLY_DEBUG] is_hotkey_re_registered took {rereg_check_ms:.2f}ms")

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

        elif signal and tp:
            # Fast local validation using background-refreshed cache (no RPC call, no refresh penalty!)
            asset_validate_start = time.perf_counter()

            # Check timestamp and validate locally using cached data
            from vali_objects.utils.asset_selection_manager import ASSET_CLASS_SELECTION_TIME_MS
            if now_ms >= ASSET_CLASS_SELECTION_TIME_MS:
                # Atomic cache read (dict reference is atomic in Python)
                selected_asset = self._asset_selections_cache.get(synapse.dendrite.hotkey, None)
                is_valid_asset = selected_asset == tp.trade_pair_category if selected_asset is not None else False
            else:
                is_valid_asset = True  # Pre-cutoff, all assets allowed

            asset_validate_ms = (time.perf_counter() - asset_validate_start) * 1000
            bt.logging.info(f"[FAIL_EARLY_DEBUG] validate_order_asset_class took {asset_validate_ms:.2f}ms")

            if not is_valid_asset:
                # Use cached value for error message (already fetched above with lock)
                pass  # selected_asset already set above

                msg = (
                    f"miner [{synapse.dendrite.hotkey}] cannot trade asset class [{tp.trade_pair_category.value}]. "
                    f"Selected asset class: [{selected_asset}]. Only trade pairs from your selected asset class are allowed. "
                    f"See https://docs.taoshi.io/ptn/ptncli#miner-operations for more information."
                )
                synapse.error_message = msg
            else:
                is_market_open = self.live_price_fetcher.is_market_open(tp, now_ms)
                execution_type = ExecutionType.from_string(signal.get("execution_type", "MARKET").upper())
                if execution_type == ExecutionType.MARKET and not is_market_open:
                    msg = (f"Market for trade pair [{tp.trade_pair_id}] is likely closed or this validator is"
                           f" having issues fetching live price. Please try again later.")
                    synapse.error_message = msg
                else:
                    unsupported_check_start = time.perf_counter()
                    unsupported_pairs = self.live_price_fetcher.get_unsupported_trade_pairs()
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
                    miner_hotkey, self.limit_order_manager
                )

                # Set synapse response (validator's responsibility)
                synapse.order_json = order.__str__()

                # UUID tracking happens HERE in validator process (limit_order_manager is separate process)
                self.uuid_tracker.add(miner_order_uuid)

            elif execution_type == ExecutionType.LIMIT_CANCEL:
                # Use OrderProcessor to handle LIMIT_CANCEL
                result = OrderProcessor.process_limit_cancel(
                    signal, trade_pair, miner_order_uuid, now_ms,
                    miner_hotkey, self.limit_order_manager
                )

                # Set synapse response (validator's responsibility)
                synapse.order_json = json.dumps(result)
                # No UUID tracking for cancel operations

            else:
                # Use OrderProcessor to handle MARKET order
                order = OrderProcessor.process_market_order(
                    signal, trade_pair, miner_order_uuid, now_ms,
                    miner_hotkey, miner_repo_version,
                    self.market_order_manager, synapse=synapse
                )
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
            # Return the last n positions
            positions = self.position_manager.get_positions_for_one_hotkey(hotkey, only_open_positions=True)
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
