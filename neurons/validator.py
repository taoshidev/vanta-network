# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import os
import sys
import threading
import signal
import uuid

from setproctitle import setproctitle

from vanta_api.api_manager import APIManager
from shared_objects.sn8_multiprocessing import get_ipc_metagraph
from multiprocessing import Manager, Process
from typing import Tuple
from enum import Enum

import template
import argparse
import traceback
import time
import bittensor as bt
import json
import gzip
import base64

from runnable.generate_request_core import RequestCoreManager
from runnable.generate_request_minerstatistics import MinerStatisticsManager
from runnable.generate_request_outputs import RequestOutputGenerator
from vali_objects.utils.auto_sync import PositionSyncer
from vali_objects.utils.p2p_syncer import P2PSyncer
from shared_objects.rate_limiter import RateLimiter
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.timestamp_manager import TimestampManager
from vali_objects.uuid_tracker import UUIDTracker
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair
from vali_objects.exceptions.signal_exception import SignalException
from shared_objects.metagraph_updater import MetagraphUpdater
from shared_objects.error_utils import ErrorUtils
from miner_objects.slack_notifier import SlackNotifier
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcherClient
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.mdd_checker import MDDChecker
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.vali_dataclasses.debt_ledger import DebtLedgerManager
from vali_objects.vali_dataclasses.emissions_ledger import EmissionsLedgerManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.vali_dataclasses.order import Order, OrderSource
from vali_objects.position import Position
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig

from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.asset_selection_manager import AssetSelectionManager

# Global flag used to indicate shutdown
shutdown_dict = {}

# Enum class that represents the method associated with Synapse
class SynapseMethod(Enum):
    POSITION_INSPECTOR = "GetPositions"
    DASHBOARD = "GetDashData"
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

class Validator:
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

        # 1. Initialize Manager for shared state
        self.ipc_manager = Manager()
        self.shared_queue_websockets = self.ipc_manager.Queue()

        # Create LivePriceFetcher client
        self.live_price_fetcher = LivePriceFetcherClient(
            secrets=self.secrets,
            disable_ws=False,
        )

        self.price_slippage_model = PriceSlippageModel(live_price_fetcher=self.live_price_fetcher)
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
        self.slack_notifier = SlackNotifier(
            hotkey=self.wallet.hotkey.ss58_address,
            webhook_url=getattr(self.config, 'slack_webhook_url', None),
            error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
            is_miner=False  # This is a validator
        )

        # Track last error notification time to prevent spam
        self.last_error_notification_time = 0
        self.error_notification_cooldown = 300  # 5 minutes between error notifications

        bt.logging.info(f"Wallet: {self.wallet}")

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
        self.metagraph = get_ipc_metagraph(self.ipc_manager)

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


        self.elimination_manager = EliminationManager(self.metagraph, None,  # Set after self.pm creation
                                                      None, shutdown_dict=shutdown_dict,
                                                      ipc_manager=self.ipc_manager,
                                                      shared_queue_websockets=self.shared_queue_websockets,
                                                      contract_manager=self.contract_manager)

        self.asset_selection_manager = AssetSelectionManager(config=self.config, metagraph=self.metagraph, ipc_manager=self.ipc_manager)

        self.position_syncer = PositionSyncer(shutdown_dict=shutdown_dict, signal_sync_lock=self.signal_sync_lock,
                                              signal_sync_condition=self.signal_sync_condition,
                                              n_orders_being_processed=self.n_orders_being_processed,
                                              ipc_manager=self.ipc_manager,
                                              position_manager=None,     # Set after self.pm creation
                                              auto_sync_enabled=self.auto_sync,
                                              contract_manager=self.contract_manager,
                                              live_price_fetcher=self.live_price_fetcher,
                                              asset_selection_manager=self.asset_selection_manager)

        self.p2p_syncer = P2PSyncer(wallet=self.wallet, metagraph=self.metagraph, is_testnet=not self.is_mainnet,
                                    shutdown_dict=shutdown_dict, signal_sync_lock=self.signal_sync_lock,
                                    signal_sync_condition=self.signal_sync_condition,
                                    n_orders_being_processed=self.n_orders_being_processed,
                                    ipc_manager=self.ipc_manager,
                                    position_manager=None)  # Set after self.pm creation


        self.perf_ledger_manager = PerfLedgerManager(self.metagraph, ipc_manager=self.ipc_manager,
                                                     shutdown_dict=shutdown_dict,
                                                     perf_ledger_hks_to_invalidate=self.position_syncer.perf_ledger_hks_to_invalidate,
                                                     position_manager=None)  # Set after self.pm creation)


        self.position_manager = PositionManager(metagraph=self.metagraph,
                                                perform_order_corrections=True,
                                                ipc_manager=self.ipc_manager,
                                                live_price_fetcher=self.live_price_fetcher,
                                                perf_ledger_manager=self.perf_ledger_manager,
                                                elimination_manager=self.elimination_manager,
                                                challengeperiod_manager=None,
                                                secrets=self.secrets,
                                                shared_queue_websockets=self.shared_queue_websockets,
                                                closed_position_daemon=True)

        self.position_locks = PositionLocks(hotkey_to_positions=self.position_manager.get_positions_for_all_miners())

        self.plagiarism_manager = PlagiarismManager(slack_notifier=self.slack_notifier,
                                                    ipc_manager=self.ipc_manager)
        self.challengeperiod_manager = ChallengePeriodManager(self.metagraph,
                                                              perf_ledger_manager=self.perf_ledger_manager,
                                                              position_manager=self.position_manager,
                                                              ipc_manager=self.ipc_manager,
                                                              contract_manager=self.contract_manager,
                                                              plagiarism_manager=self.plagiarism_manager)

        # Attach the position manager to the other objects that need it
        for idx, obj in enumerate([self.perf_ledger_manager, self.position_manager, self.position_syncer,
                                   self.p2p_syncer, self.elimination_manager, self.metagraph_updater,
                                   self.contract_manager]):
            obj.position_manager = self.position_manager

        self.position_manager.challengeperiod_manager = self.challengeperiod_manager

        #force_validator_to_restore_from_checkpoint(self.wallet.hotkey.ss58_address, self.metagraph, self.config, self.secrets)

        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        self.position_manager.perf_ledger_manager = self.perf_ledger_manager

        self.position_manager.pre_run_setup()
        self.uuid_tracker.add_initial_uuids(self.position_manager.get_positions_for_all_miners())

        self.debt_ledger_manager = DebtLedgerManager(self.perf_ledger_manager, self.position_manager, self.contract_manager,
                                     self.asset_selection_manager, challengeperiod_manager=self.challengeperiod_manager,
                                     slack_webhook_url=self.config.slack_error_webhook_url, start_daemon=True,
                                     ipc_manager=self.ipc_manager, validator_hotkey=self.wallet.hotkey.ss58_address)


        self.checkpoint_lock = threading.Lock()
        self.encoded_checkpoint = ""
        self.last_checkpoint_time = 0
        self.timestamp_manager = TimestampManager(metagraph=self.metagraph,
                                                  hotkey=self.wallet.hotkey.ss58_address)

        bt.logging.info(f"Metagraph n_entries: {len(self.metagraph.hotkeys)}")
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} is not registered to chain "
                f"connection: {self.metagraph_updater.get_subtensor()} \nRun btcli register and try again. "
            )
            exit()

        # Build and link vali functions to the axon.
        # The axon handles request processing, allowing validators to send this process requests.
        bt.logging.info(f"setting port [{self.config.axon.port}]")
        bt.logging.info(f"setting external port [{self.config.axon.external_port}]")
        self.axon = bt.axon(
            wallet=self.wallet, port=self.config.axon.port, external_port=self.config.axon.external_port
        )
        bt.logging.info(f"Axon {self.axon}")

        # Attach determines which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to axon.")

        self.order_rate_limiter = RateLimiter()
        self.position_inspector_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 4)
        self.dash_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60)
        self.checkpoint_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60 * 60 * 6)
        # Cache to track last order time for each (miner_hotkey, trade_pair) combination
        self.last_order_time_cache = {}  # Key: (miner_hotkey, trade_pair_id), Value: last_order_time_ms

        def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def rs_priority_fn(synapse: template.protocol.SendSignal) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def gp_blacklist_fn(synapse: template.protocol.GetPositions) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def gp_priority_fn(synapse: template.protocol.GetPositions) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def gd_blacklist_fn(synapse: template.protocol.GetDashData) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def gd_priority_fn(synapse: template.protocol.GetDashData) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def rc_blacklist_fn(synapse: template.protocol.ValidatorCheckpoint) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def rc_priority_fn(synapse: template.protocol.ValidatorCheckpoint) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def cr_blacklist_fn(synapse: template.protocol.CollateralRecord) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def cr_priority_fn(synapse: template.protocol.CollateralRecord) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        def as_blacklist_fn(synapse: template.protocol.AssetSelection) -> Tuple[bool, str]:
            return Validator.blacklist_fn(synapse, self.metagraph)

        def as_priority_fn(synapse: template.protocol.AssetSelection) -> float:
            return Validator.priority_fn(synapse, self.metagraph)

        self.axon.attach(
            forward_fn=self.receive_signal,
            blacklist_fn=rs_blacklist_fn,
            priority_fn=rs_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.get_positions,
            blacklist_fn=gp_blacklist_fn,
            priority_fn=gp_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.get_dash_data,
            blacklist_fn=gd_blacklist_fn,
            priority_fn=gd_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.receive_checkpoint,
            blacklist_fn=rc_blacklist_fn,
            priority_fn=rc_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.receive_collateral_record,
            blacklist_fn=cr_blacklist_fn,
            priority_fn=cr_priority_fn,
        )
        self.axon.attach(
            forward_fn=self.receive_asset_selection,
            blacklist_fn=as_blacklist_fn,
            priority_fn=as_priority_fn,
        )

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving attached axons on network:"
            f" {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Starts the miner's axon, making it active on the network.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

        # Each hotkey gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

        # Eliminations are read in validator, elimination_manager, mdd_checker, weight setter.
        # Eliminations are written in elimination_manager, mdd_checker
        # Since the mainloop is run synchronously, we just need to lock eliminations when writing to them and when
        # reading outside of the mainloop (validator).

        # Watchdog thread to detect hung initialization steps
        init_watchdog = {'current_step': 0, 'start_time': time.time(), 'step_desc': 'Starting', 'alerted': False}

        def initialization_watchdog():
            """Background thread that monitors for hung initialization steps"""
            HANG_TIMEOUT = 60  # Alert after 60 seconds on a single step
            while init_watchdog['current_step'] <= 10:
                time.sleep(5)  # Check every 5 seconds
                if init_watchdog['current_step'] > 10:
                    break  # Initialization complete

                elapsed = time.time() - init_watchdog['start_time']
                if elapsed > HANG_TIMEOUT and not init_watchdog['alerted']:
                    init_watchdog['alerted'] = True
                    hang_msg = (
                        f"⚠️ Validator Initialization Hang Detected!\n"
                        f"Step {init_watchdog['current_step']}/10 has been running for {elapsed:.1f}s\n"
                        f"Step: {init_watchdog['step_desc']}\n"
                        f"Hotkey: {self.wallet.hotkey.ss58_address}\n"
                        f"Timeout threshold: {HANG_TIMEOUT}s\n"
                        f"The validator may be stuck and require manual restart."
                    )
                    bt.logging.error(hang_msg)
                    if self.slack_notifier:
                        self.slack_notifier.send_message(hang_msg, level="error")

        # Start watchdog thread
        watchdog_thread = threading.Thread(target=initialization_watchdog, daemon=True)
        watchdog_thread.start()

        # Helper function to run initialization steps with timeout and error handling
        def run_init_step_with_monitoring(step_num, step_desc, step_func, timeout_seconds=30):
            """Execute an initialization step with timeout monitoring and error handling"""
            # Update watchdog state
            init_watchdog['current_step'] = step_num
            init_watchdog['step_desc'] = step_desc
            init_watchdog['start_time'] = time.time()
            init_watchdog['alerted'] = False

            bt.logging.info(f"[INIT] Step {step_num}/10: {step_desc}...")
            start_time = time.time()
            try:
                result = step_func()
                elapsed = time.time() - start_time
                bt.logging.info(f"[INIT] Step {step_num}/10 complete: {step_desc} (took {elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = f"[INIT] Step {step_num}/10 FAILED: {step_desc} after {elapsed:.2f}s - {str(e)}"
                bt.logging.error(error_msg)
                bt.logging.error(traceback.format_exc())

                # Send Slack alert
                if self.slack_notifier:
                    self.slack_notifier.send_message(
                        f"🚨 Validator Initialization Failed!\n"
                        f"Step: {step_num}/10 - {step_desc}\n"
                        f"Error: {str(e)}\n"
                        f"Hotkey: {self.wallet.hotkey.ss58_address}\n"
                        f"Time elapsed: {elapsed:.2f}s\n"
                        f"The validator may be hung or unable to start properly.",
                        level="error"
                    )
                raise

        # Step 1: Initialize PlagiarismDetector
        def step1():
            self.plagiarism_detector = PlagiarismDetector(self.metagraph, shutdown_dict=shutdown_dict,
                                                          position_manager=self.position_manager)
            return self.plagiarism_detector
        run_init_step_with_monitoring(1, "Initializing PlagiarismDetector", step1)

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
        run_init_step_with_monitoring(2, "Starting plagiarism detector process", step2)

        # Step 3: Initialize MDDChecker
        def step3():
            self.mdd_checker = MDDChecker(self.metagraph, self.position_manager, live_price_fetcher=self.live_price_fetcher,
                                          shutdown_dict=shutdown_dict)
            return self.mdd_checker
        run_init_step_with_monitoring(3, "Initializing MDDChecker", step3)

        # Step 4: Initialize SubtensorWeightSetter
        def step4():
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
        run_init_step_with_monitoring(4, "Initializing SubtensorWeightSetter", step4)

        # Step 5: Initialize RequestCoreManager and MinerStatisticsManager
        def step5():
            self.request_core_manager = RequestCoreManager(self.position_manager, self.weight_setter, self.plagiarism_detector,
                                                          self.contract_manager, ipc_manager=self.ipc_manager,
                                                          asset_selection_manager=self.asset_selection_manager)
            self.miner_statistics_manager = MinerStatisticsManager(self.position_manager, self.weight_setter,
                                                                   self.plagiarism_detector, contract_manager=self.contract_manager,
                                                                   ipc_manager=self.ipc_manager)
            return (self.request_core_manager, self.miner_statistics_manager)
        run_init_step_with_monitoring(5, "Initializing RequestCoreManager and MinerStatisticsManager", step5)

        # Step 6: Start perf ledger updater process
        def step6():
            self.perf_ledger_updater_thread = Process(target=self.perf_ledger_manager.run_update_loop, daemon=True)
            self.perf_ledger_updater_thread.start()
            # Verify process started
            time.sleep(0.1)  # Give process a moment to start
            if not self.perf_ledger_updater_thread.is_alive():
                raise RuntimeError("Perf ledger updater process failed to start")
            bt.logging.info(f"Process started with PID: {self.perf_ledger_updater_thread.pid}")
            return self.perf_ledger_updater_thread
        run_init_step_with_monitoring(6, "Starting perf ledger updater process", step6)

        # Step 7: Start weight setter process
        def step7():
            self.weight_setter_process = Process(target=self.weight_setter.run_update_loop, daemon=True)
            self.weight_setter_process.start()
            # Verify process started
            time.sleep(0.1)  # Give process a moment to start
            if not self.weight_setter_process.is_alive():
                raise RuntimeError("Weight setter process failed to start")
            bt.logging.info(f"Process started with PID: {self.weight_setter_process.pid}")
            return self.weight_setter_process
        run_init_step_with_monitoring(7, "Starting weight setter process", step7)

        # Step 8: Start weight processing thread
        def step8():
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
        run_init_step_with_monitoring(8, "Starting weight processing thread", step8)

        # Step 9: Start request output generator (if enabled)
        def step9():
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
        run_init_step_with_monitoring(9, "Starting request output generator (if enabled)", step9)

        # Step 10: Start API services (if enabled)
        def step10():
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
                    validator_hotkey=self.wallet.hotkey.ss58_address
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
        run_init_step_with_monitoring(10, "Starting API services (if enabled)", step10)

        # Signal watchdog that initialization is complete
        init_watchdog['current_step'] = 11
        bt.logging.info("[INIT] All 10 initialization steps completed successfully!")

        # Send success notification to Slack
        if self.slack_notifier:
            self.slack_notifier.send_message(
                f"✅ Validator Initialization Complete!\n"
                f"All 10 initialization steps completed successfully\n"
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

    @staticmethod
    def blacklist_fn(synapse, metagraph) -> Tuple[bool, str]:
        miner_hotkey = synapse.dendrite.hotkey
        # Ignore requests from unrecognized entities.
        if miner_hotkey not in metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    @staticmethod
    def priority_fn(synapse, metagraph) -> float:
        # simply just prioritize based on uid as it's not significant
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.uids[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    # subtensor is now a simple instance variable (no property needed)
    # It's created in __init__ and used directly throughout validator

    def get_config(self):
        # Step 2: Set up the configuration parser
        # This function initializes the necessary command-line arguments.
        # Using command-line arguments allows users to customize various miner settings.
        parser = argparse.ArgumentParser()
        # Set autosync to store true if flagged, otherwise defaults to False.
        parser.add_argument("--autosync", action='store_true',
                            help="Automatically sync order data with a validator trusted by Taoshi.")
        # Set run_generate to store true if flagged, otherwise defaults to False.
        parser.add_argument("--start-generate", action='store_true', dest='start_generate',
                            help="Run the request output generator.")

        # API Server related arguments
        parser.add_argument("--serve", action='store_true',
                            help="Start the API server for REST and WebSocket endpoints")
        parser.add_argument("--api-host", type=str, default="127.0.0.1",
                            help="Host address for the API server")
        parser.add_argument("--api-rest-port", type=int, default=48888,
                            help="Port for the REST API server")
        parser.add_argument("--api-ws-port", type=int, default=8765,
                            help="Port for the WebSocket server")

        # (developer): Adds your custom arguments to the parser.
        # Adds override arguments for network and netuid.
        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
        

        # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.wallet.add_args(parser)

        # Add Slack webhook arguments
        parser.add_argument(
            "--slack-webhook-url",
            type=str,
            default=None,
            help="Slack webhook URL for general notifications (optional)"
        )
        parser.add_argument(
            "--slack-error-webhook-url",
            type=str,
            default=None,
            help="Slack webhook URL for error notifications (optional, defaults to general webhook if not provided)"
        )
        # Adds axon specific arguments i.e. --axon.port ...
        bt.axon.add_args(parser)
        # Activating the parser to read any command-line inputs.
        # To print help message, run python3 template/miner.py --help
        config = bt.config(parser)
        bt.logging.enable_info()
        if config.logging.debug:
            bt.logging.enable_debug()
        if config.logging.trace:
            bt.logging.enable_trace()

        # Step 3: Set up logging directory
        # Logging captures events for diagnosis or understanding miner's behavior.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "validator",
            )
        )
        return config

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
                current_time = TimeUtil.now_in_millis()
                self.live_price_fetcher.health_check(current_time)
                self.price_slippage_model.refresh_features_daily()
                self.position_syncer.sync_positions_with_cooldown(self.auto_sync)
                self.mdd_checker.mdd_check(self.position_locks)
                self.challengeperiod_manager.refresh(current_time=current_time)
                self.elimination_manager.process_eliminations(self.position_locks)
                #self.position_locks.cleanup_locks(self.metagraph.hotkeys)
                # Weight setting now runs in its own process
                #self.p2p_syncer.sync_positions_with_cooldown()

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

    def parse_trade_pair_from_signal(self, signal) -> TradePair | None:
        if not signal or not isinstance(signal, dict):
            return None
        if 'trade_pair' not in signal:
            return None
        temp = signal["trade_pair"]
        if 'trade_pair_id' not in temp:
            return None
        string_trade_pair = signal["trade_pair"]["trade_pair_id"]
        trade_pair = TradePair.from_trade_pair_id(string_trade_pair)
        return trade_pair

    def _get_or_create_open_position_from_new_order(self, trade_pair: TradePair, order_type: OrderType, order_time_ms: int,
                                        miner_hotkey: str, miner_order_uuid: str, now_ms:int, price_sources, miner_repo_version, account_size):

        # gather open positions and see which trade pairs have an open position
        positions = self.position_manager.get_positions_for_one_hotkey(miner_hotkey, only_open_positions=True)
        trade_pair_to_open_position = {position.trade_pair: position for position in positions}

        existing_open_pos = trade_pair_to_open_position.get(trade_pair)
        if existing_open_pos:
            # If the position has too many orders, we need to close it out to make room.
            if len(existing_open_pos.orders) >= ValiConfig.MAX_ORDERS_PER_POSITION and order_type != OrderType.FLAT:
                bt.logging.info(
                    f"Miner [{miner_hotkey}] hit {ValiConfig.MAX_ORDERS_PER_POSITION} order limit. "
                    f"Automatically closing position for {trade_pair.trade_pair_id} "
                    f"with {len(existing_open_pos.orders)} orders to make room for new position."
                )
                force_close_order_time = now_ms - 1 # 2 orders for the same trade pair cannot have the same timestamp
                force_close_order_uuid = existing_open_pos.position_uuid[::-1] # uuid will stay the same across validators
                self._add_order_to_existing_position(existing_open_pos, trade_pair, OrderType.FLAT,
                                                     0.0, 0.0, 0.0, force_close_order_time, miner_hotkey,
                                                     price_sources, force_close_order_uuid, miner_repo_version,
                                                     OrderSource.MAX_ORDERS_PER_POSITION_CLOSE)
                time.sleep(0.1)  # Put 100ms between two consecutive websocket writes for the same trade pair and hotkey. We need the new order to be seen after the FLAT.
            else:
                # If the position is closed, raise an exception. This can happen if the miner is eliminated in the main
                # loop thread.
                if trade_pair_to_open_position[trade_pair].is_closed_position:
                    raise SignalException(
                        f"miner [{miner_hotkey}] sent signal for "
                        f"closed position [{trade_pair}]")
                bt.logging.debug("adding to existing position")
                # Return existing open position (nominal path)
                return trade_pair_to_open_position[trade_pair]


        # if the order is FLAT ignore (noop)
        if order_type == OrderType.FLAT:
            open_position = None
        else:
            # if a position doesn't exist, then make a new one
            open_position = Position(
                miner_hotkey=miner_hotkey,
                position_uuid=miner_order_uuid if miner_order_uuid else str(uuid.uuid4()),
                open_ms=order_time_ms,
                trade_pair=trade_pair,
                account_size=account_size
            )
        return open_position

    def should_fail_early(self, synapse: template.protocol.SendSignal | template.protocol.GetPositions | template.protocol.GetDashData | template.protocol.ValidatorCheckpoint, method:SynapseMethod,
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
        elif method == SynapseMethod.DASHBOARD:
            allowed, wait_time = self.dash_rate_limiter.is_allowed(sender_hotkey)
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

        if method == SynapseMethod.CHECKPOINT or method == SynapseMethod.DASHBOARD:
            return False
        elif method == SynapseMethod.POSITION_INSPECTOR:
            # Check version 0 (old version that was opt-in)
            if synapse.version == 0:
                synapse.successfully_processed = False
                synapse.error_message = "Please use the latest miner script that makes PI opt-in with the flag --run-position-inspector"
                #bt.logging.info((sender_hotkey, synapse.error_message))
                return True

        # don't process eliminated miners
        elimination_info = self.elimination_manager.hotkey_in_eliminations(synapse.dendrite.hotkey)
        if elimination_info:
            msg = f"This miner hotkey {synapse.dendrite.hotkey} has been eliminated and cannot participate in this subnet. Try again after re-registering. elimination_info {elimination_info}"
            bt.logging.debug(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        # don't process re-registered miners
        if self.elimination_manager.is_hotkey_re_registered(synapse.dendrite.hotkey):
            # Get deregistration timestamp and convert to human-readable date
            departed_info = self.elimination_manager.departed_hotkeys.get(synapse.dendrite.hotkey, {})
            detected_ms = departed_info.get("detected_ms", 0)
            dereg_date = TimeUtil.millis_to_formatted_date_str(detected_ms) if detected_ms else "unknown"

            msg = (f"This miner hotkey {synapse.dendrite.hotkey} was previously de-registered and is not allowed to re-register. "
                   f"De-registered on: {dereg_date} UTC. "
                   f"Re-registration is not permitted on this subnet.")
            bt.logging.warning(msg)
            synapse.successfully_processed = False
            synapse.error_message = msg
            return True

        order_uuid = synapse.miner_order_uuid
        tp = self.parse_trade_pair_from_signal(signal)
        if order_uuid and self.uuid_tracker.exists(order_uuid):
            msg = (f"Order with uuid [{order_uuid}] has already been processed. "
                   f"Please try again with a new order.")
            bt.logging.error(msg)
            synapse.error_message = msg

        elif signal and tp:
            # Validate asset class selection
            if not self.asset_selection_manager.validate_order_asset_class(synapse.dendrite.hotkey, tp.trade_pair_category, now_ms):
                msg = (
                    f"miner [{synapse.dendrite.hotkey}] cannot trade asset class [{tp.trade_pair_category.value}]. "
                    f"Selected asset class: [{self.asset_selection_manager.asset_selections.get(synapse.dendrite.hotkey, None)}]. Only trade pairs from your selected asset class are allowed. "
                    f"See https://docs.taoshi.io/vanta/vanta-cli#miner-operations for more information."
                )
                synapse.error_message = msg

            elif not self.live_price_fetcher.is_market_open(tp):
                msg = (f"Market for trade pair [{tp.trade_pair_id}] is likely closed or this validator is"
                       f" having issues fetching live price. Please try again later.")
                synapse.error_message = msg

            elif tp in self.live_price_fetcher.get_unsupported_trade_pairs():
                msg = (f"Trade pair [{tp.trade_pair_id}] has been temporarily halted. "
                       f"Please try again with a different trade pair.")
                synapse.error_message = msg

            elif tp.is_blocked:
                msg = (f"Trade pair [{tp.trade_pair_id}] is no longer supported. "
                       f"Please try again with a different trade pair.")
                synapse.error_message = msg

        synapse.successfully_processed = not bool(synapse.error_message)
        if synapse.error_message:
            bt.logging.error(synapse.error_message)

        return bool(synapse.error_message)

    def enforce_order_cooldown(self, trade_pair_id, now_ms, miner_hotkey):
        """
        Enforce cooldown between orders for the same trade pair using an efficient cache.
        This method must be called within the position lock to prevent race conditions.
        """
        cache_key = (miner_hotkey, trade_pair_id)
        current_order_time_ms = now_ms

        # Get the last order time from cache
        cached_last_order_time = self.last_order_time_cache.get(cache_key, 0)
        msg = None
        if cached_last_order_time > 0:
            time_since_last_order_ms = current_order_time_ms - cached_last_order_time

            if time_since_last_order_ms < ValiConfig.ORDER_COOLDOWN_MS:
                previous_order_time = TimeUtil.millis_to_formatted_date_str(cached_last_order_time)
                current_time = TimeUtil.millis_to_formatted_date_str(current_order_time_ms)
                time_to_wait_in_s = (ValiConfig.ORDER_COOLDOWN_MS - time_since_last_order_ms) / 1000
                msg = (
                    f"Order for trade pair [{trade_pair_id}] was placed too soon after the previous order. "
                    f"Last order was placed at [{previous_order_time}] and current order was placed at [{current_time}]. "
                    f"Please wait {time_to_wait_in_s:.1f} seconds before placing another order."
                )

        return msg

    def parse_miner_uuid(self, synapse: template.protocol.SendSignal):
        temp = synapse.miner_order_uuid
        assert isinstance(temp, str), f"excepted string miner uuid but got {temp}"
        if not temp:
            bt.logging.warning(f'miner_order_uuid is empty for miner_hotkey [{synapse.dendrite.hotkey}] miner_repo_version '
                               f'[{synapse.repo_version}]. Generating a new one.')
            temp = str(uuid.uuid4())
        return temp

    def _add_order_to_existing_position(self, existing_position, trade_pair, signal_order_type: OrderType,
                                        quantity: float, leverage: float, value: float, order_time_ms: int, miner_hotkey: str,
                                        price_sources, miner_order_uuid: str, miner_repo_version: str, src:OrderSource,
                                        usd_base_price=None) -> Order:
        # Must be locked by caller
        best_price_source = price_sources[0]
        price = best_price_source.parse_appropriate_price(order_time_ms, trade_pair.is_forex, signal_order_type, existing_position)

        if existing_position.account_size <= 0:
            bt.logging.warning(
                f"Invalid account_size {existing_position.account_size} for position {existing_position.position_uuid}. "
                f"Using MIN_CAPITAL as fallback."
            )
            existing_position.account_size = ValiConfig.MIN_CAPITAL
        order = Order(
            trade_pair=trade_pair,
            order_type=signal_order_type,
            quantity=quantity,
            value=value,
            leverage=leverage,
            price=price,
            processed_ms=order_time_ms,
            order_uuid=miner_order_uuid,
            price_sources=price_sources,
            bid=best_price_source.bid,
            ask=best_price_source.ask,
            src=src
        )
        if usd_base_price is None:
            usd_base_price = self.live_price_fetcher.get_usd_base_conversion(trade_pair, order_time_ms, price, signal_order_type, existing_position)
        order.usd_base_rate = usd_base_price
        order.quote_usd_rate = self.live_price_fetcher.get_quote_usd_conversion(order, existing_position)
        net_portfolio_leverage = self.position_manager.calculate_net_portfolio_leverage(miner_hotkey)
        order.slippage = PriceSlippageModel.calculate_slippage(order.bid, order.ask, order)
        existing_position.add_order(order, self.live_price_fetcher, net_portfolio_leverage)
        self.position_manager.save_miner_position(existing_position)
        # Update cooldown cache after successful order processing
        self.last_order_time_cache[(miner_hotkey, trade_pair.trade_pair_id)] = order_time_ms
        self.uuid_tracker.add(miner_order_uuid)

        if self.config.serve:
            # Add the position to the queue for broadcasting
            self.shared_queue_websockets.put(existing_position.to_websocket_dict(miner_repo_version=miner_repo_version))
        return order

    def _get_account_size(self, miner_hotkey, now_ms):
        account_size = self.contract_manager.get_miner_account_size(hotkey=miner_hotkey, timestamp_ms=now_ms)
        if account_size is None:
            account_size = ValiConfig.MIN_CAPITAL
        else:
            account_size = max(account_size, ValiConfig.MIN_CAPITAL)
        return account_size

    @staticmethod
    def parse_order_size(signal, usd_base_conversion, trade_pair, portfolio_value):
        """
        parses an order signal and calculates leverage, value, and quantity
        """
        leverage = signal.get("leverage")
        value = signal.get("value")
        quantity = signal.get("quantity")

        fields_set = [x is not None for x in (leverage, value, quantity)]
        if sum(fields_set) != 1:
            raise ValueError("Exactly one of 'leverage', 'value', or 'quantity' must be set")

        if quantity is not None:
            value = quantity * trade_pair.lot_size / usd_base_conversion
            leverage = value / portfolio_value
        if leverage is not None:
            value = leverage * portfolio_value
            quantity = (value * usd_base_conversion) / trade_pair.lot_size
        elif value is not None:
            leverage = value / portfolio_value
            quantity = (value * usd_base_conversion) / trade_pair.lot_size

        return quantity, leverage, value

    # This is the core validator function to receive a signal
    def receive_signal(self, synapse: template.protocol.SendSignal,
                       ) -> template.protocol.SendSignal:
        # pull miner hotkey to reference in various activities
        now_ms = TimeUtil.now_in_millis()
        order = None
        miner_hotkey = synapse.dendrite.hotkey
        miner_repo_version = synapse.repo_version
        synapse.validator_hotkey = self.wallet.hotkey.ss58_address
        signal = synapse.signal
        bt.logging.info( f"received signal [{signal}] from miner_hotkey [{miner_hotkey}] using repo version [{miner_repo_version}].")
        if self.should_fail_early(synapse, SynapseMethod.SIGNAL, signal=signal, now_ms=now_ms):
            return synapse

        with self.signal_sync_lock:
            self.n_orders_being_processed[0] += 1

        # error message to send back to miners in case of a problem so they can fix and resend
        error_message = ""
        try:
            miner_order_uuid = self.parse_miner_uuid(synapse)
            trade_pair = self.parse_trade_pair_from_signal(signal)
            if trade_pair is None:
                bt.logging.error(f"[{trade_pair}] not in TradePair enum.")
                raise SignalException(
                    f"miner [{miner_hotkey}] incorrectly sent trade pair. Raw signal: {signal}"
                )

            price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, now_ms)
            if not price_sources:
                raise SignalException(
                    f"Ignoring order for [{miner_hotkey}] due to no live prices being found for trade_pair [{trade_pair}]. Please try again.")

            signal_order_type = OrderType.from_string(signal["order_type"])

            # Multiple threads can run receive_signal at once. Don't allow two threads to trample each other.
            with self.position_locks.get_lock(miner_hotkey, trade_pair.trade_pair_id):
                # Check cooldown inside the lock to prevent race conditions
                err_msg = self.enforce_order_cooldown(trade_pair.trade_pair_id, now_ms, miner_hotkey)
                if err_msg:
                    bt.logging.error(err_msg)
                    synapse.successfully_processed = False
                    synapse.error_message = err_msg
                    return synapse

                # Get relevant account size
                account_size = self._get_account_size(miner_hotkey, now_ms)
                existing_position = self._get_or_create_open_position_from_new_order(trade_pair, signal_order_type,
                    now_ms, miner_hotkey, miner_order_uuid, now_ms, price_sources, miner_repo_version, account_size)
                if existing_position:
                    best_price_source = price_sources[0]
                    price = best_price_source.parse_appropriate_price(now_ms, trade_pair.is_forex, signal_order_type, existing_position)
                    usd_base_price = self.live_price_fetcher.get_usd_base_conversion(trade_pair, now_ms, price, signal_order_type, existing_position)
                    quantity, leverage, value = self.parse_order_size(signal, usd_base_price, trade_pair, existing_position.account_size)

                    order = self._add_order_to_existing_position(existing_position, trade_pair, signal_order_type,
                                                        quantity, leverage, value, now_ms, miner_hotkey,
                                                        price_sources, miner_order_uuid, miner_repo_version,
                                                        OrderSource.ORGANIC, usd_base_price)
                    synapse.order_json = existing_position.orders[-1].__str__()
                else:
                    # Happens if a FLAT is sent when no position exists
                    pass
                # Update the last received order time
                self.timestamp_manager.update_timestamp(now_ms)

        except SignalException as e:
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())
        except Exception as e:
            error_message = f"Error processing order for [{miner_hotkey}] with error [{e}]"
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False

        synapse.error_message = error_message
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

    def get_dash_data(self, synapse: template.protocol.GetDashData,
                      ) -> template.protocol.GetDashData:
        if self.should_fail_early(synapse, SynapseMethod.DASHBOARD):
            return synapse

        now_ms = TimeUtil.now_in_millis()
        miner_hotkey = synapse.dendrite.hotkey
        error_message = ""
        try:
            timestamp = self.timestamp_manager.get_last_order_timestamp()

            stats_all = json.loads(ValiBkpUtils.get_file(ValiBkpUtils.get_miner_stats_dir()))
            new_data = []
            for payload in stats_all['data']:
                if payload['hotkey'] == miner_hotkey:
                    new_data = [payload]
                    break
            stats_all['data'] = new_data
            positions = self.request_core_manager.generate_request_core(get_dash_data_hotkey=miner_hotkey)
            dash_data = {"timestamp": timestamp, "statistics": stats_all, **positions}

            if not stats_all["data"]:
                error_message = f"Validator {self.wallet.hotkey.ss58_address} has no stats for miner {miner_hotkey}"
            elif not positions:
                error_message = f"Validator {self.wallet.hotkey.ss58_address} has no positions for miner {miner_hotkey}"

            synapse.data = dash_data
            bt.logging.info("Sending data back to miner: " + miner_hotkey)
        except Exception as e:
            error_message = f"Error in GetData for [{miner_hotkey}] with error [{e}]."
            bt.logging.error(traceback.format_exc())

        if error_message == "":
            synapse.successfully_processed = True
        else:
            bt.logging.error(error_message)
            synapse.successfully_processed = False
        synapse.error_message = error_message
        processing_time_s_3_decimals = round((TimeUtil.now_in_millis() - now_ms) / 1000.0, 3)
        bt.logging.info(
            f"Sending dash data back to miner [{miner_hotkey}]. Synapse Message: {synapse.error_message}. "
            f"Process time {processing_time_s_3_decimals} seconds.")
        return synapse

    def receive_checkpoint(self, synapse: template.protocol.ValidatorCheckpoint) -> template.protocol.ValidatorCheckpoint:
        """
        receive checkpoint request, and ensure that only requests received from valid validators are processed.
        """
        sender_hotkey = synapse.dendrite.hotkey

        # validator responds to poke from validator and attaches their checkpoint
        if sender_hotkey in [axon.hotkey for axon in self.p2p_syncer.get_validators()]:
            synapse.validator_receive_hotkey = self.wallet.hotkey.ss58_address

            bt.logging.info(f"Received checkpoint request poke from validator hotkey [{sender_hotkey}].")
            if self.should_fail_early(synapse, SynapseMethod.CHECKPOINT):
                return synapse

            error_message = ""
            try:
                with self.checkpoint_lock:
                    # reset checkpoint after 10 minutes
                    if TimeUtil.now_in_millis() - self.last_checkpoint_time > 1000 * 60 * 10:
                        self.encoded_checkpoint = ""
                    # save checkpoint so we only generate it once for all requests
                    if not self.encoded_checkpoint:
                        # get our current checkpoint
                        self.last_checkpoint_time = TimeUtil.now_in_millis()
                        checkpoint_dict = self.request_core_manager.generate_request_core()

                        # compress json and encode as base64 to keep as a string
                        checkpoint_str = json.dumps(checkpoint_dict, cls=CustomEncoder)
                        compressed = gzip.compress(checkpoint_str.encode("utf-8"))
                        self.encoded_checkpoint = base64.b64encode(compressed).decode("utf-8")

                    # only send a checkpoint if we are an up-to-date validator
                    timestamp = self.timestamp_manager.get_last_order_timestamp()
                    if TimeUtil.now_in_millis() - timestamp < 1000 * 60 * 60 * 10:  # validators with no orders processed in 10 hrs are considered stale
                        synapse.checkpoint = self.encoded_checkpoint
                    else:
                        error_message = f"Validator is stale, no orders received in 10 hrs, last order timestamp {timestamp}, {round((TimeUtil.now_in_millis() - timestamp)/(1000 * 60 * 60))} hrs ago"
            except Exception as e:
                error_message = f"Error processing checkpoint request poke from [{sender_hotkey}] with error [{e}]"
                bt.logging.error(traceback.format_exc())

            if error_message == "":
                synapse.successfully_processed = True
            else:
                bt.logging.error(error_message)
                synapse.successfully_processed = False
            synapse.error_message = error_message
            bt.logging.success(f"Sending checkpoint back to validator [{sender_hotkey}]")
        else:
            bt.logging.info(f"Received a checkpoint poke from non validator [{sender_hotkey}]")
            synapse.error_message = "Rejecting checkpoint poke from non validator"
            synapse.successfully_processed = False
        return synapse

    def receive_collateral_record(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        """
        receive collateral record update, and update miner account sizes
        """
        try:
            # Process the collateral record through the contract manager
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(f"Received collateral record update from validator hotkey [{sender_hotkey}].")
            success = self.contract_manager.receive_collateral_record_update(synapse.collateral_record)
            
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

    def receive_asset_selection(self, synapse: template.protocol.AssetSelection) -> template.protocol.AssetSelection:
        """
        receive miner's asset selection
        """
        try:
            # Process the collateral record through the contract manager
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(f"Received miner asset selection from validator hotkey [{sender_hotkey}].")
            success = self.asset_selection_manager.receive_asset_selection_update(synapse.asset_selection)

            if success:
                synapse.successfully_processed = True
                synapse.error_message = ""
                bt.logging.info(f"Successfully processed AssetSelection synapse from {sender_hotkey}")
            else:
                synapse.successfully_processed = False
                synapse.error_message = "Failed to process miner's asset selection"
                bt.logging.warning(f"Failed to process AssetSelection synapse from {sender_hotkey}")

        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Error processing asset selection: {str(e)}"
            bt.logging.error(f"Exception in receive_asset_selection: {e}")

        return synapse

# This is the main function, which runs the miner.
if __name__ == "__main__":
    validator = Validator()
    validator.main()
