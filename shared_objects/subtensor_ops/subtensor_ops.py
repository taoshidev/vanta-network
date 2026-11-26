# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

import time
import traceback
import threading
import asyncio

from dataclasses import dataclass
from setproctitle import setproctitle

import vanta_protocol
from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.subtensor_ops.metagraph_utils import is_anomalous_hotkey_loss, is_validator_stake_weight
from shared_objects.locks.subtensor_lock import get_subtensor_lock
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from time_util.time_util import TimeUtil

import bittensor as bt
import logging
from shared_objects.log import logger
from miner_config import MinerConfig


# Simple picklable data structures - used both for unit-test mocks AND as the real
# production neuron/axon shape sent over RPC (see _convert_neuron below). bittensor v11's
# bt.MetagraphNeuron.axon is just an "ip:port" string or None, and has no validator_trust
# field (only `trust`, present regardless of permit) - these wrappers restore the shape
# every existing consumer (validator_broadcast_base.py, p2p_syncer.py, position_inspector.py,
# prop_net_order_placer.py, metagraph_utils.py) already expects: axon_info.ip/.port/.hotkey,
# neuron.validator_trust.
@dataclass
class SimpleAxonInfo:
    """Simple picklable axon info."""
    ip: str
    port: int
    hotkey: str = None


@dataclass
class SimpleNeuron:
    """Simple picklable neuron."""
    uid: int
    hotkey: str
    incentive: float
    validator_trust: float
    axon_info: SimpleAxonInfo
    coldkey: str = None
    stake: float = 0.0  # TAO-denominated chain stake weight (bittensor's metagraph.total_stake)


def _balance_to_tao_float(balance) -> float:
    """bt.Balance is no longer float()-able directly (verified interactively in bittensor 11.3.2).
    `.tao` is the safe conversion for a TAO/root-denominated Balance (verified: Balance.from_tao(5).tao == 5.0);
    total_stake is a root-denominated aggregate (alpha_stake converted + tao_stake), so `.tao` is expected to work.
    Falls back to raw rao/1e9 (unit-agnostic) if `.tao` ever raises on an unexpectedly alpha-denominated Balance."""
    if balance is None:
        return 0.0
    try:
        return float(balance.tao)
    except Exception:
        try:
            return float(balance.rao) / 1e9
        except Exception:
            return 0.0


def _convert_neuron(n) -> "SimpleNeuron":
    """Convert a real bt.MetagraphNeuron (v11) into the picklable SimpleNeuron/SimpleAxonInfo shape
    every consumer in this repo already expects (axon_info.ip/.port/.hotkey, neuron.validator_trust)."""
    axon_str = getattr(n, 'axon', None)
    if axon_str:
        ip, _, port_str = axon_str.rpartition(':')
        try:
            port = int(port_str)
        except (TypeError, ValueError):
            port = 0
        axon_info = SimpleAxonInfo(ip=ip or MinerConfig.AXON_NO_IP, port=port, hotkey=n.hotkey)
    else:
        # No served axon - represent the same way the old code represented one (AXON_NO_IP sentinel)
        axon_info = SimpleAxonInfo(ip=MinerConfig.AXON_NO_IP, port=0, hotkey=n.hotkey)

    # validator_trust no longer exists as its own chain field in v11 (only `trust`); this subnet
    # doesn't rely on it for logic anyway (CLAUDE.md: Null consensus, one UID holds permit/trust) -
    # it's log-only here, so trust-when-permitted (0 otherwise) reproduces the old semantics closely enough.
    validator_trust = float(n.trust) if getattr(n, 'validator_permit', False) else 0.0

    return SimpleNeuron(
        uid=n.uid,
        hotkey=n.hotkey,
        incentive=float(getattr(n, 'incentive', 0.0) or 0.0),
        validator_trust=validator_trust,
        axon_info=axon_info,
        coldkey=getattr(n, 'coldkey', None),
        stake=_balance_to_tao_float(getattr(n, 'total_stake', None)),
    )


# ==================== Client for WeightSetter RPC ====================


class WeightFailureTracker:
    """Track weight setting failures and manage alerting logic"""
    
    def __init__(self):
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.last_alert_time = 0
        self.failure_patterns = {}  # Track unknown error patterns
        self.had_critical_failure = False
        
    def classify_failure(self, err_msg, error_code=None):
        """Classify failure based on production patterns.

        error_code is bt.ExtrinsicResult.error.code when the v11 SDK classified the chain-level
        failure (an bt.ErrorCode enum member); primary classification checks that first, falling
        back to substring-matching err_msg for anything the SDK didn't classify, or that doesn't
        cleanly map to an ErrorCode (old-style exception-message patterns, connection errors, etc.)
        """
        # Primary: classify by the SDK's own error code when available.
        if error_code is not None:
            code_name = getattr(error_code, "name", str(error_code))
            if code_name in ("TOO_EARLY", "RATE_LIMITED"):
                return "benign"
            if code_name in ("INTERNAL", "POLICY_VIOLATION", "NOT_AUTHORIZED"):
                return "critical"

        error_lower = err_msg.lower()

        # BENIGN - Don't alert (expected behavior)
        if any(phrase in error_lower for phrase in [
            "no attempt made. perhaps it is too soon to commit weights",
            "too soon to commit weights",
            "too soon to commit",
            "empty extrinsicresult",
            "empty response from set_weights",
            "too_early",
            "rate_limited",
        ]):
            return "benign"

        # CRITICAL - Alert immediately (known problematic patterns)
        elif any(phrase in error_lower for phrase in [
            "maximum recursion depth exceeded",
            "invalid transaction",
            "subtensor returned: invalid transaction"
        ]):
            return "critical"

        # UNKNOWN - Alert after pattern emerges
        else:
            return "unknown"
    
    def should_alert(self, failure_type, consecutive_count):
        """Determine if we should send an alert"""
        # Get current time once for consistency
        current_time = time.time()
        time_since_success = current_time - self.last_success_time
        time_since_last_alert = current_time - self.last_alert_time
        
        # Alert if we haven't had a successful weight setting in 2 hours
        # This is an absolute timeout that bypasses all other checks
        if time_since_success > 7200:  # 2 hours
            return True
        
        # Rate limiting check - but exempt critical errors and 1+ hour timeouts
        if failure_type != "critical" and time_since_success <= 3600:
            if time_since_last_alert < 600:
                return False
        
        # Always alert for known critical errors (no rate limiting)
        if failure_type == "critical":
            return True
        
        # Alert if we haven't had a successful weight setting in 1 hour
        # This check happens before benign check to catch prolonged benign failures
        if time_since_success > 3600:
            return True
        
        # Never alert for benign "too soon" errors (unless prolonged, caught above)
        if failure_type == "benign":
            return False
        
        # For unknown errors, alert after 2 consecutive failures
        if failure_type == "unknown" and consecutive_count >= 2:
            return True
        
        return False
    
    def track_failure(self, err_msg, failure_type):
        """Track a failure"""
        self.consecutive_failures += 1
        
        # Track if this was a critical failure
        if failure_type == "critical":
            self.had_critical_failure = True
        
        # Track unknown error patterns
        if failure_type == "unknown":
            pattern_key = err_msg[:50] if len(err_msg) > 50 else err_msg
            self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
    
    def track_success(self):
        """Track a successful weight setting"""
        # Check if we should send recovery alert
        should_send_recovery = self.consecutive_failures > 0 and self.had_critical_failure
        
        # Reset tracking
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.had_critical_failure = False
        
        return should_send_recovery


class SubtensorOpsManager(CacheController):
    """
    Run locally to interface with the Subtensor object without RPC overhead.

    Handles all subtensor operations including:
    - Metagraph updates and caching
    - Weight setting via RPC
    - Validator broadcasting via RPC
    """
    def __init__(self, config, hotkey, is_miner, position_manager=None,
                 slack_notifier=None, running_unit_tests=False):
        super().__init__()
        self.is_miner = is_miner
        self.is_validator = not is_miner
        self.config = config
        self.running_unit_tests = running_unit_tests
        self.is_testnet = config.netuid in (116, 171)

        self._metagraph_client = None

        # Initialize failure tracking BEFORE subtensor creation (needed if creation fails)
        self.consecutive_failures = 0

        # Create subtensor (mock if running unit tests)
        if running_unit_tests:
            self.subtensor = self._create_mock_subtensor()
        else:
            try:
                self.subtensor = self._build_subtensor()
            except (ConnectionRefusedError, ConnectionError, OSError) as e:
                logger.error(f"Failed to create initial subtensor connection: {e}")
                logger.warning("Will retry during first metagraph update loop iteration")
                # Set to None - update loop will recreate it (using consecutive_failures > 0 logic)
                self.subtensor = None
                # Increment consecutive_failures so update loop tries to recreate immediately
                self.consecutive_failures = 1

        # Create own LivePriceFetcherClient for validators (forward compatibility - no parameter passing)
        # Only validators need this for TAO/USD price queries
        if self.is_validator:
            from vali_objects.price_fetcher import LivePriceFetcherClient
            self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests)
            from vali_objects.miner_account.miner_account_client import MinerAccountClient
            self._miner_account_client = MinerAccountClient(running_unit_tests=running_unit_tests)
        else:
            self._live_price_client = None
            self._miner_account_client = None
        # Round-robin was already degenerate (a single-entry ['finney'] list, i.e. a same-endpoint
        # reconnect) now removed entirely per the v11 migration - the exponential-backoff retry loop
        # in run_update_loop() and the build-new-then-cleanup-old reconnect pattern in
        # update_metagraph() are the real healing mechanism and are kept as-is.

        # Initialize likely validators and miners with empty dictionaries. This maps hotkey to timestamp.
        self.likely_validators = {}
        self.likely_miners = {}
        self.hotkey = hotkey
        self.interval_wait_time_ms = ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MINER_MS if self.is_miner else \
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_VALIDATOR_MS
        self.position_manager = position_manager
        self.slack_notifier = slack_notifier  # Add slack notifier for error reporting

        # Weight setting for validators only (RPC-based, no queue)
        self.last_weight_set = 0
        self.weight_failure_tracker = WeightFailureTracker() if not is_miner else None
        self.rpc_server = None
        self.rpc_thread = None

        # Exponential backoff parameters
        self.min_backoff = 120
        self.max_backoff = 43200  # 12 hours maximum (12 * 60 * 60)
        self.backoff_factor = 2  # Double the wait time on each retry
        self.current_backoff = self.min_backoff

        # Hotkeys cache for fast lookups (refreshed atomically during metagraph updates)
        # No lock needed - set assignment is atomic in Python
        self._hotkeys_cache = set()

        # Start RPC server (allows SubtensorWeightCalculator to call set_weights_rpc)
        # Skip RPC server in unit tests to avoid port conflicts
        if self.is_validator and not running_unit_tests:
            self._start_weight_setter_rpc_server()

        # Log mode
        mode = "miner" if is_miner else "validator"
        logger.info(f"SubtensorOpsManager initialized in {mode} mode, weight setting via RPC")

    def _create_mock_subtensor(self):
        """Create a mock subtensor for unit testing."""
        from unittest.mock import Mock

        mock_subtensor = Mock()

        # Mock metagraph() method to return empty metagraph
        def mock_metagraph_func(netuid):
            mock_metagraph = Mock()
            mock_metagraph.hotkeys = []
            mock_metagraph.uids = []
            mock_metagraph.neurons = []
            mock_metagraph.block_at_registration = []
            mock_metagraph.emission = []
            mock_metagraph.axons = []
            # Must be a real list: update_metagraph iterates it when logging
            # permitted validators (a bare Mock attribute is not iterable).
            mock_metagraph.validator_permit = []
            mock_metagraph.total_stake = []

            # Mock pool data
            mock_metagraph.pool = Mock()
            mock_metagraph.pool.tao_in = 1000.0
            mock_metagraph.pool.alpha_in = 5000.0

            return mock_metagraph

        mock_subtensor.metagraph = Mock(side_effect=mock_metagraph_func)

        # Mock execute() method (bt.Subtensor.execute(intent, wallet) - replaces v10's set_weights(); used by
        # both ServeAxon (validator_base.py) and SetWeights (_set_weights_with_retry below))
        mock_result = Mock()
        mock_result.success = True
        mock_result.error = None
        mock_result.message = None
        mock_subtensor.execute = Mock(return_value=mock_result)

        # mock_subtensor.close() needs no explicit setup - Mock() auto-creates it on access
        # (v11's _cleanup_subtensor_connection calls .close() directly, no .substrate.close()).

        return mock_subtensor

    def _create_mock_wallet(self):
        """Create a mock wallet for unit testing."""
        from unittest.mock import Mock

        mock_wallet = Mock()
        mock_wallet.hotkey = Mock()
        mock_wallet.hotkey.ss58_address = self.hotkey
        return mock_wallet

    def set_mock_metagraph_data(self, hotkeys, neurons=None):
        """
        Set mock metagraph data for unit testing.

        Args:
            hotkeys: List of hotkeys to populate mock metagraph with
            neurons: Optional list of neuron objects (if None, will create basic picklable neurons)
        """
        if not self.running_unit_tests:
            raise RuntimeError("set_mock_metagraph_data() can only be used in test mode")

        from unittest.mock import Mock

        # Create neurons if not provided (using module-level dataclasses)
        if neurons is None:
            neurons = []
            for i, hk in enumerate(hotkeys):
                axon_info = SimpleAxonInfo(ip="192.168.1.1", port=8091, hotkey=hk)
                neuron = SimpleNeuron(
                    uid=i,
                    hotkey=hk,
                    incentive=0.1,
                    validator_trust=0.1 if i == 0 else 0.0,  # First one is validator
                    axon_info=axon_info,
                    coldkey=hk,
                    stake=2 * ValiConfig.STAKE_MIN if i == 0 else 0.0,
                )
                neurons.append(neuron)

        # Update the mock metagraph function to return this data
        def mock_metagraph_func(netuid):
            mock_metagraph = Mock()
            mock_metagraph.hotkeys = hotkeys
            mock_metagraph.uids = list(range(len(hotkeys)))
            mock_metagraph.neurons = neurons
            mock_metagraph.block_at_registration = [1000] * len(hotkeys)
            mock_metagraph.emission = [1.0] * len(hotkeys)
            mock_metagraph.axons = [n.axon_info for n in neurons]
            # Must be a real list: update_metagraph iterates it when logging
            # permitted validators (a bare Mock attribute is not iterable).
            mock_metagraph.validator_permit = [
                n.validator_trust > 0 for n in neurons
            ]
            # Stake weight mirrors the permit above
            mock_metagraph.total_stake = [
                2 * ValiConfig.STAKE_MIN if n.validator_trust > 0 else 0.0 for n in neurons
            ]

            # Mock pool data
            mock_metagraph.pool = Mock()
            mock_metagraph.pool.tao_in = 1000.0
            mock_metagraph.pool.alpha_in = 5000.0

            return mock_metagraph

        self.subtensor.metagraph = Mock(side_effect=mock_metagraph_func)

    def _start_weight_setter_rpc_server(self):
        """Start RPC server for weight setting requests (validators only).
        Must run locally because the subtensor instance is on the main thread. """
        from multiprocessing.managers import BaseManager

        # Define RPC manager
        class WeightSetterRPC(BaseManager):
            pass

        # Register this instance to handle RPC calls
        WeightSetterRPC.register(
            'WeightSetterServer',
            callable=lambda: self
        )

        # Start RPC server in a thread
        address = ("localhost", ValiConfig.RPC_WEIGHT_SETTER_PORT)
        authkey = ValiConfig.get_rpc_authkey(
            ValiConfig.RPC_WEIGHT_SETTER_SERVICE_NAME,
            ValiConfig.RPC_WEIGHT_SETTER_PORT
        )

        manager = WeightSetterRPC(address=address, authkey=authkey)
        self.rpc_server = manager.get_server()

        # Run server in daemon thread
        self.rpc_thread = threading.Thread(
            target=self.rpc_server.serve_forever,
            daemon=True,
            name="WeightSetterRPC"
        )
        self.rpc_thread.start()

    # ==================== RPC Methods (exposed to clients) ====================

    def health_check_rpc(self) -> dict:
        """
        RPC method for health checks (called from SubtensorOpsClient.health_check()).

        SubtensorOpsManager doesn't inherit from RPCServerBase — it runs its own
        ad-hoc WeightSetterRPC BaseManager (see _start_weight_setter_rpc_server)
        exposing this instance directly, so it needs its own health_check_rpc
        rather than getting one for free.
        """
        return {
            "status": "ok",
            "service": ValiConfig.RPC_WEIGHT_SETTER_SERVICE_NAME,
            "timestamp_ms": TimeUtil.now_in_millis(),
            "consecutive_failures": self.consecutive_failures,
        }

    def broadcast_to_validators_rpc(self, request, validator_axons_list, message_name=None):
        """
        RPC method to broadcast a request to validators (called from ValidatorBroadcastBase).

        This method runs the broadcast using the SubtensorOpsManager's wallet and the
        signed-request HTTP client, allowing processes without direct subtensor/wallet
        access to send broadcasts.

        Args:
            request: One of vanta_protocol's *Request models — SendSignalRequest/
                GetPositionsRequest keep their own type, while
                every other broadcast (CollateralRecord, AssetSelection, SubaccountRegistration,
                EntityEndpointUpdate, StandardMinerOpening, StandardMinerRecord,
                AccountSizeLocks) shares the generic BroadcastRequest shape — already
                validated as picklable.
            validator_axons_list: List of axon_info objects to broadcast to
            message_name: The MESSAGE_TYPES/HTTP-path key for this broadcast (e.g.
                "CollateralRecord"). Required now that several message types share
                BroadcastRequest, so the class name alone no longer identifies one.

        Returns:
            dict: {
                "success": bool,
                "success_count": int,
                "total_count": int,
                "errors": list of error messages
            }
        """
        try:
            if self.running_unit_tests:
                logger.debug("[BROADCAST RPC] Running unit tests, skipping broadcast")
                return {"success": True, "success_count": 0, "total_count": 0, "errors": []}

            if not validator_axons_list:
                logger.debug("[BROADCAST RPC] No validators to broadcast to")
                return {"success": True, "success_count": 0, "total_count": 0, "errors": []}

            if not request or not hasattr(request, '__class__'):
                raise ValueError("Invalid request object")

            if not message_name:
                raise ValueError("message_name is required to broadcast a BroadcastRequest")

            # Create wallet from config
            wallet = bt.Wallet(name=self.config.wallet.name, hotkey=self.config.wallet.hotkey, path=self.config.wallet.path)

            target_hotkeys = [a.hotkey for a in validator_axons_list]
            logger.info(f"[BROADCAST RPC] Broadcasting {message_name} to {len(validator_axons_list)} validators: {target_hotkeys}")

            _, response_cls = vanta_protocol.MESSAGE_TYPES[message_name]

            async def do_broadcast():
                from shared_objects.http_client import HttpAuthClient
                async with HttpAuthClient(wallet) as client:
                    responses = await client.aquery(validator_axons_list, message_name, request, response_cls)

                    success_count = 0
                    errors = []
                    successful_hotkeys = []

                    for target, response in zip(validator_axons_list, responses):
                        if response.successfully_processed:
                            success_count += 1
                            successful_hotkeys.append(target.hotkey)
                        elif response.error_message:
                            errors.append(f"{target.hotkey}: {response.error_message}")

                    return success_count, errors, successful_hotkeys

            success_count, errors, successful_hotkeys = asyncio.run(do_broadcast())

            logger.info(
                f"[BROADCAST RPC] Broadcast completed: {success_count}/{len(validator_axons_list)} validators updated. "
                f"Successful: {successful_hotkeys}"
            )

            return {
                "success": True,
                "success_count": success_count,
                "total_count": len(validator_axons_list),
                "errors": errors
            }

        except Exception as e:
            error_msg = f"Error in broadcast_to_validators_rpc: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "success_count": 0, "total_count": 0, "errors": [str(e)]}

    def set_weights_rpc(self, uids, weights, version_key):
        """
        RPC method to set weights synchronously (called from SubtensorWeightCalculator).

        Args:
            uids: List of UIDs to set weights for
            weights: List of weights corresponding to UIDs
            version_key: Subnet version key

        Returns:
            dict: {"success": bool, "error": str}
        """
        try:
            # Use our own config for netuid
            netuid = self.config.netuid

            # Create wallet from our own config (mock if running unit tests)
            if self.running_unit_tests:
                wallet = self._create_mock_wallet()
            else:
                wallet = bt.Wallet(name=self.config.wallet.name, hotkey=self.config.wallet.hotkey, path=self.config.wallet.path)

            logger.info(f"[RPC] Processing weight setting request for {len(uids)} UIDs")

            # Set weights with retry logic
            success, error_msg, error_code = self._set_weights_with_retry(
                netuid=netuid,
                wallet=wallet,
                uids=uids,
                weights=weights,
                version_key=version_key
            )

            if success:
                self.last_weight_set = time.time()
                logger.info("[RPC] Weight setting completed successfully")

                # Track success and check for recovery alerts
                if self.weight_failure_tracker:
                    should_send_recovery = self.weight_failure_tracker.track_success()
                    if should_send_recovery and self.slack_notifier:
                        self._send_recovery_alert(wallet)

                return {"success": True, "error": None}
            else:
                logger.warning(f"[RPC] Weight setting failed: {error_msg}")

                # Track failure and send alerts
                if self.weight_failure_tracker:
                    failure_type = self.weight_failure_tracker.classify_failure(error_msg, error_code)
                    self.weight_failure_tracker.track_failure(error_msg, failure_type)

                    if self.weight_failure_tracker.should_alert(failure_type, self.weight_failure_tracker.consecutive_failures):
                        self._send_weight_failure_alert(error_msg, failure_type, wallet)
                        self.weight_failure_tracker.last_alert_time = time.time()

                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Error in set_weights_rpc: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"success": False, "error": error_msg}

    def _current_timestamp(self):
        return time.time()

    def _is_expired(self, timestamp):
        return (self._current_timestamp() - timestamp) > 86400  # 24 hours in seconds
    
    def _build_subtensor(self) -> bt.Subtensor:
        """bt.Subtensor.__init__ in v11 takes no `config=` kwarg (verified: network, policy,
        fallback_endpoints, archive_endpoints, retry_forever, substrate only) and no `chain_endpoint=`
        kwarg either - any ws://*/wss:// URL is accepted directly as `network` itself. So pass the
        configured chain_endpoint as the network argument when set, else the configured network name."""
        network = self.config.subtensor.chain_endpoint or self.config.subtensor.network
        return bt.Subtensor(network=network)

    def _cleanup_subtensor_connection(self):
        """Safely close the subtensor connection to prevent file descriptor leaks.
        bt.Subtensor.close() is the real method in v11 (no more .substrate.close() fallback needed)."""
        if hasattr(self, 'subtensor') and self.subtensor:
            try:
                logger.debug("Closing subtensor connection")
                self.subtensor.close()
            except Exception as e:
                logger.warning(f"Error during subtensor cleanup: {e}")
    
    def get_subtensor(self):
        """
        Get the current subtensor instance.
        This should be used instead of directly accessing self.subtensor
        to ensure you always have the current instance after round-robin switches.
        """
        return self.subtensor
    
    def start_and_wait_for_initial_update(self, max_wait_time=60, slack_notifier=None):
        """
        Start the metagraph updater thread and wait for initial population.
        
        This method provides a clean way to:
        1. Start the background metagraph update loop
        2. Wait for the metagraph to be initially populated
        3. Proceed with confidence that metagraph data is available
        
        Args:
            max_wait_time (int): Maximum time to wait for initial population (seconds)
            slack_notifier: Optional slack notifier for error reporting
            
        Returns:
            threading.Thread: The started metagraph updater thread
            
        Raises:
            SystemExit: If metagraph fails to populate within max_wait_time
        """
        # Start the metagraph updater loop in its own thread
        updater_thread = threading.Thread(target=self.run_update_loop, daemon=True)
        updater_thread.start()
        
        # Wait for initial metagraph population before proceeding
        logger.info("Waiting for initial metagraph population...")
        start_time = time.time()
        while not self._metagraph_client.get_hotkeys() and (time.time() - start_time) < max_wait_time:
            time.sleep(1)

        if not self._metagraph_client.get_hotkeys():
            error_msg = f"Failed to populate metagraph within {max_wait_time} seconds"
            logger.error(error_msg)
            if slack_notifier:
                slack_notifier.send_message(f"❌ {error_msg}", level="error")
            exit()

        logger.info(f"Metagraph populated with {len(self._metagraph_client.get_hotkeys())} hotkeys")
        return updater_thread

    def estimate_number_of_validators(self):
        # Filter out expired validators
        self.likely_validators = {k: v for k, v in self.likely_validators.items() if not self._is_expired(v)}
        hotkeys_with_v_trust = set() if self.is_miner else {self.hotkey}
        for neuron in self._metagraph_client.get_neurons():
            if neuron.validator_trust > 0:
                hotkeys_with_v_trust.add(neuron.hotkey)
        return len(hotkeys_with_v_trust.union(set(self.likely_validators.keys())))

    def run_update_loop(self):
        mode_name = "miner" if self.is_miner else "validator"
        setproctitle(f"metagraph_updater_{mode_name}_{self.hotkey}")
        logger.setLevel(logging.INFO)

        while not ShutdownCoordinator.is_shutdown():
            try:
                self.update_metagraph()
                # Reset backoff on successful update
                if self.consecutive_failures > 0:
                    logger.info(
                        f"Metagraph update successful after {self.consecutive_failures} failures. Resetting backoff.")
                    if self.slack_notifier:
                        self.slack_notifier.send_message(
                            f"✅ Metagraph update recovered after {self.consecutive_failures} consecutive failures.",
                            level="info"
                        )
                self.consecutive_failures = 0
                self.current_backoff = self.min_backoff
                
                time.sleep(1)  # Normal operation delay
            except Exception as e:
                self.consecutive_failures += 1
                # Calculate next backoff time
                self.current_backoff = min(self.current_backoff * self.backoff_factor, self.max_backoff)

                # Log error with backoff information
                error_msg = (f"Error during metagraph update (attempt #{self.consecutive_failures}): {e}. "
                             f"Next retry in {self.current_backoff} seconds.\n")
                logger.error(error_msg)
                logger.error(traceback.format_exc())

                if self.slack_notifier:
                    # Get compact traceback using shared utility
                    compact_trace = ErrorUtils.get_compact_stacktrace(e)
                    
                    hours = self.current_backoff / 3600
                    node_type = "miner" if self.is_miner else "validator"
                    self.slack_notifier.send_message(
                        f"❌ Metagraph update failing repeatedly!\n"
                        f"Consecutive failures: {self.consecutive_failures}\n"
                        f"Error: {str(e)}\n"
                        f"Trace: {compact_trace}\n"
                        f"Next retry in: {hours:.2f} hours\n"
                        f"Please check the {node_type} logs!",
                        level="error"
                    )

                # Wait with exponential backoff
                time.sleep(self.current_backoff)

    def _set_weights_with_retry(self, netuid, wallet, uids, weights, version_key):
        """Set weights using the existing subtensor connection.

        Returns: (success, error_msg, error_code) where error_code is the bt.ErrorCode the SDK
        classified the failure as (or None, when it couldn't classify it / on a raised exception).
        """
        # Check if subtensor is available before attempting weight setting
        if self.subtensor is None:
            error_msg = "Subtensor connection not available (initialization or reconnection in progress)"
            logger.error(error_msg)
            return False, error_msg, None

        max_retries = 1

        for attempt in range(max_retries):
            try:
                with get_subtensor_lock():
                    result = self.subtensor.execute(
                        bt.SetWeights(netuid=netuid, uids=uids, weights=weights, version_key=version_key),
                        wallet,
                    )

                success = result.success
                error_code = result.error.code if result.error is not None else None
                if success:
                    error_msg = None
                elif result.error is not None:
                    error_msg = f"{result.error.code}: {result.error.remediation}"
                elif result.message:
                    error_msg = result.message
                else:
                    error_msg = "empty ExtrinsicResult with success=False"
                logger.info(f"Weight setting attempt {attempt + 1}: success={success}, error={error_msg}")
                return success, error_msg, error_code

            except Exception as e:
                # Genuine connection-level failures (timeouts, dropped sockets) can still raise;
                # chain-level failures come back as success=False above, not an exception.
                logger.warning(f"Weight setting failed (attempt {attempt + 1}): {e}")

        return False, "All retry attempts failed", None

    def _send_weight_failure_alert(self, err_msg, failure_type, wallet):
        """Send contextual Slack alert for weight setting failure"""
        if not self.slack_notifier:
            return
        
        # Get context information
        hotkey = "unknown"
        if wallet:
            if hasattr(wallet, 'hotkey'):
                if hasattr(wallet.hotkey, 'ss58_address'):
                    hotkey = wallet.hotkey.ss58_address
                else:
                    logger.warning("Wallet hotkey missing ss58_address attribute")
            else:
                logger.warning("Wallet missing hotkey attribute")
        else:
            logger.warning("Wallet parameter is None in weight failure alert")
        
        netuid = "unknown"
        network = "unknown"
        if self.config:
            if hasattr(self.config, 'netuid'):
                netuid = self.config.netuid
            else:
                logger.warning("Config missing netuid attribute")
                
            if hasattr(self.config, 'subtensor'):
                if hasattr(self.config.subtensor, 'network'):
                    network = self.config.subtensor.network
                else:
                    logger.warning("Config subtensor missing network attribute")
            else:
                logger.warning("Config missing subtensor attribute")
        else:
            logger.warning("Config is None - cannot determine network/netuid for alert")
            
        consecutive = self.weight_failure_tracker.consecutive_failures
        
        # Build alert message based on failure type
        if "maximum recursion depth exceeded" in err_msg.lower():
            message = f"🚨 CRITICAL: Weight setting recursion error\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"Error: {err_msg}\n" \
                     f"This indicates a serious code issue that needs immediate attention."
        
        elif "invalid transaction" in err_msg.lower():
            message = f"🚨 CRITICAL: Subtensor rejected weight transaction\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"Error: {err_msg}\n" \
                     f"This may indicate wallet/balance issues or network problems."
        
        elif failure_type == "unknown":
            message = f"❓ NEW PATTERN: Unknown weight setting failure\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"Consecutive failures: {consecutive}\n" \
                     f"Error: {err_msg}\n" \
                     f"This is a new error pattern that needs investigation."
        
        else:
            # Prolonged failure alert
            time_since_success = time.time() - self.weight_failure_tracker.last_success_time
            hours_since_success = time_since_success / 3600
            
            if hours_since_success >= 2:
                urgency = "🚨 URGENT"
                time_msg = f"No successful weight setting in {hours_since_success:.1f} hours"
            else:
                urgency = "⚠️ WARNING"
                time_msg = f"No successful weight setting in {hours_since_success:.1f} hours"
            
            message = f"{urgency}: Weight setting issues detected\n" \
                     f"Network: {network}\n" \
                     f"Hotkey: {hotkey}\n" \
                     f"{time_msg}\n" \
                     f"Last error: {err_msg}"
        
        self.slack_notifier.send_message(message, level="error")
    
    def _send_recovery_alert(self, wallet):
        """Send recovery alert after critical failures"""
        if not self.slack_notifier:
            return
        
        hotkey = "unknown"
        if wallet:
            if hasattr(wallet, 'hotkey'):
                if hasattr(wallet.hotkey, 'ss58_address'):
                    hotkey = wallet.hotkey.ss58_address
                else:
                    logger.warning("Wallet hotkey missing ss58_address attribute in recovery alert")
            else:
                logger.warning("Wallet missing hotkey attribute in recovery alert")
        else:
            logger.warning("Wallet parameter is None in recovery alert")
            
        network = "unknown"
        if self.config:
            if hasattr(self.config, 'subtensor'):
                if hasattr(self.config.subtensor, 'network'):
                    network = self.config.subtensor.network
                else:
                    logger.warning("Config subtensor missing network attribute in recovery alert")
            else:
                logger.warning("Config missing subtensor attribute in recovery alert")
        else:
            logger.warning("Config is None - cannot determine network for recovery alert")
        
        message = f"✅ Weight setting recovered after failures\n" \
                 f"Network: {network}\n" \
                 f"Hotkey: {hotkey}"
        
        self.slack_notifier.send_message(message, level="info")


    def sync_lists(self, shared_list, updated_list, brute_force=False):
        if brute_force:
            prev_memory_location = id(shared_list)
            shared_list[:] = updated_list  # Update the proxy list in place without changing the reference
            assert prev_memory_location == id(shared_list), f"Memory location changed after brute force update from {prev_memory_location} to {id(shared_list)}"
            return

        # Convert to sets for fast comparison
        current_set = set(shared_list)
        updated_set = set(updated_list)

        # Find items to remove (in current but not in updated)
        items_to_remove = current_set - updated_set
        # Find items to add (in updated but not in current)
        items_to_add = updated_set - current_set

        # Remove items no longer present
        for item in items_to_remove:
            shared_list.remove(item)

        # Add new items
        for item in items_to_add:
            shared_list.append(item)

    def get_metagraph(self):
        """
        Returns the metagraph object.
        """
        return self._metagraph_client

    def is_hotkey_registered_cached(self, hotkey: str) -> bool:
        """
        Fast local check if hotkey is registered (no RPC call!).

        Uses local cache that is atomically refreshed during metagraph updates.
        Much faster than calling metagraph.has_hotkey() which does RPC.

        Args:
            hotkey: The hotkey to check

        Returns:
            True if hotkey is registered in metagraph, False otherwise
        """
        return hotkey in self._hotkeys_cache

    def _get_tao_per_alpha_price(self, metagraph_clone) -> float:
        """
        Get the TAO-per-ALPHA spot price for ALPHA->USD conversion.

        JUDGMENT CALL (v11 migration): bt.Metagraph no longer exposes raw pool reserves -
        `metagraph.pool.tao_in`/`.alpha_in` is gone, replaced by `metagraph.price` (the ratio
        tao_in/alpha_in directly). Grepping every downstream consumer of the old
        tao_reserve_rao/alpha_reserve_rao pair (DebtBasedScoring._convert_alpha_to_usd is the only
        one) showed they only ever compute `tao_reserve_rao / alpha_reserve_rao` - the RATIO, never
        the absolute magnitudes - so this now returns that ratio directly instead of two RAO figures.
        The RPC plumbing (metagraph_client/server, DebtBasedScoring) was updated to carry a single
        `tao_per_alpha_price` field end-to-end rather than synthesizing fake reserve magnitudes.

        Test mode keeps reading metagraph_clone.pool.tao_in/.alpha_in since the existing mocks
        (_create_mock_subtensor/set_mock_metagraph_data) still set that shape.

        Fails fast - exceptions propagate to the slack alert mechanism.
        """
        if self.running_unit_tests:
            if not hasattr(metagraph_clone, 'pool') or not metagraph_clone.pool:
                raise ValueError("metagraph.pool not available - cannot get reserve data")
            tao_in = metagraph_clone.pool.tao_in
            alpha_in = metagraph_clone.pool.alpha_in
            if not alpha_in:
                raise ValueError("Alpha reserve is zero - cannot calculate conversion rate")
            price = float(tao_in) / float(alpha_in)
        else:
            price = getattr(metagraph_clone, 'price', None)
            if price is None:
                raise ValueError("metagraph.price not available - cannot get tao/alpha conversion rate")
            price = float(price)
            if price <= 0:
                raise ValueError(f"Invalid tao/alpha price from metagraph: {price}")

        logger.info(f"Got tao/alpha price from metagraph: {price:.6f} TAO/ALPHA")
        return price

    def _get_tao_usd_rate(self):
        """
        Get current TAO/USD price using _live_price_client.
        Uses current timestamp to get latest available price.

        Non-blocking: If price fetch fails, logs error and returns None.
        Better to use a slightly stale TAO/USD price than block metagraph updates.

        Returns:
            float: TAO/USD rate, or None if unavailable
        """
        try:
            if not self._live_price_client:
                logger.warning(
                    "_live_price_client not available - cannot query TAO/USD price. "
                    "Using existing price from metagraph (may be stale)."
                )
                return None

            # Get current timestamp for price query
            current_time_ms = TimeUtil.now_in_millis()

            # Query TAO/USD price at current time
            price_source = self._live_price_client.get_close_at_date(
                TradePair.TAOUSD,
                current_time_ms
            )

            if not price_source or not hasattr(price_source, 'close') or price_source.close is None:
                logger.warning(
                    f"TAO/USD price unavailable at timestamp {current_time_ms}. "
                    f"Using existing price from metagraph (may be stale). "
                    f"price_source={price_source}"
                )
                return None

            tao_to_usd_rate = float(price_source.close)

            # Validate price is reasonable
            if tao_to_usd_rate <= 0:
                logger.warning(
                    f"Invalid TAO/USD price: ${tao_to_usd_rate}. "
                    f"Using existing price from metagraph (may be stale)."
                )
                return None

            logger.info(
                f"Got TAO/USD price: ${tao_to_usd_rate:.2f}/TAO "
                f"(timestamp: {current_time_ms})"
            )
            return tao_to_usd_rate

        except Exception as e:
            logger.error(
                f"Error fetching TAO/USD price: {e}. "
                f"Using existing price from metagraph (may be stale)."
            )
            logger.error(traceback.format_exc())
            return None

    def _fetch_metagraph(self):
        """bt.Subtensor has no top-level .metagraph() method in v11 (confirmed via inspection) -
        per MIGRATION-GUIDE.md ("The metagraph"), fetch via subtensor.subnets.metagraph(netuid)
        instead, the sync-wrapped equivalent of `await client.subnets.metagraph(netuid=...)`. Test
        mode keeps calling the mockable self.subtensor.metagraph(netuid) seam so existing mock
        setups (_create_mock_subtensor/set_mock_metagraph_data) keep working unchanged."""
        if self.running_unit_tests:
            return self.subtensor.metagraph(self.config.netuid)
        return self.subtensor.subnets.metagraph(self.config.netuid)

    def update_metagraph(self):
        if not self.refresh_allowed(self.interval_wait_time_ms):
            return

        if self.consecutive_failures > 0:
            # Try to create new subtensor BEFORE cleaning up old one
            # This ensures we never leave self.subtensor in a broken state that breaks other components
            try:
                if self.running_unit_tests:
                    new_subtensor = self._create_mock_subtensor()
                else:
                    new_subtensor = self._build_subtensor()

                # Only cleanup old connection after new one successfully created (prevents file descriptor leak).
                # Under the subtensor lock: weight-setter RPC threads hold it
                # while an extrinsic is in flight on the OLD connection — the
                # close+swap must not yank the websocket out from under them.
                with get_subtensor_lock():
                    self._cleanup_subtensor_connection()
                    self.subtensor = new_subtensor
                logger.info("Successfully recreated subtensor connection after previous failures")

            except (ConnectionRefusedError, ConnectionError, OSError) as e:
                # Connection errors during subtensor creation - keep old subtensor and re-raise
                logger.error(f"Failed to recreate subtensor connection (connection error): {e}")
                # Don't cleanup old connection - let it stay alive for other components (weight setting, etc.)
                # Re-raise so outer exception handler applies exponential backoff
                raise
            except Exception as e:
                # Other unexpected errors - still keep old subtensor but log differently
                logger.error(f"Failed to recreate subtensor connection (unexpected error): {e}")
                # Don't cleanup old connection
                raise

        # Check if subtensor is available before attempting metagraph sync
        if self.subtensor is None:
            raise RuntimeError("Subtensor connection not available - cannot sync metagraph")

        hotkeys_before = set(self._metagraph_client.get_hotkeys())

        # Synchronize with weight setting operations to prevent WebSocket concurrency errors
        with get_subtensor_lock():
            metagraph_clone = self._fetch_metagraph()

        assert hasattr(metagraph_clone, 'hotkeys'), "Metagraph clone does not have hotkeys attribute"
        logger.info("Updating metagraph...")
        # bt.metagraph.fetch() (v11) / subtensor.metagraph() (test mock) already syncs the metagraph.
        hotkeys_after = set(metagraph_clone.hotkeys)
        lost_hotkeys = hotkeys_before - hotkeys_after
        gained_hotkeys = hotkeys_after - hotkeys_before
        if lost_hotkeys:
            logger.info(f"metagraph has lost hotkeys: {lost_hotkeys}")
        if gained_hotkeys:
            logger.info(f"metagraph has gained hotkeys: {gained_hotkeys}")
        if not lost_hotkeys and not gained_hotkeys:
            logger.info(f"metagraph hotkeys remain the same. n = {len(hotkeys_after)}")

        # Use shared anomaly detection logic
        is_anomalous, percent_lost = is_anomalous_hotkey_loss(lost_hotkeys, len(hotkeys_before))
        # failsafe condition to reject new metagraph
        if is_anomalous:
            error_msg = (f"Too many hotkeys lost in metagraph update: {len(lost_hotkeys)} hotkeys lost, "
                         f"{percent_lost:.2f}% of total hotkeys. Rejecting new metagraph. ALERT A TEAM MEMBER ASAP...")
            logger.error(error_msg)
            if self.slack_notifier:
                self.slack_notifier.send_message(
                    f"🚨 CRITICAL: {error_msg}",
                    level="error"
                )
            return  # Actually block the metagraph update

        # v11's bt.Metagraph only carries per-neuron fields (no top-level .uids/.validator_permit/
        # .total_stake/.block_at_registration/.emission/.axons) - derive everything from .neurons,
        # converting each real bt.MetagraphNeuron into the SimpleNeuron/SimpleAxonInfo shape every
        # consumer already expects. Test mode keeps the mocks' old top-level-list shape untouched
        # (neurons there are already SimpleNeuron, built by _create_mock_subtensor/set_mock_metagraph_data).
        if self.running_unit_tests:
            converted_neurons = list(metagraph_clone.neurons)
            uids = list(metagraph_clone.uids)
            block_at_registration = list(metagraph_clone.block_at_registration)
            emission = list(metagraph_clone.emission)
        else:
            converted_neurons = [_convert_neuron(n) for n in metagraph_clone.neurons]
            uids = [n.uid for n in metagraph_clone.neurons]
            block_at_registration = [n.block_at_registration for n in metagraph_clone.neurons]
            emission = [_balance_to_tao_float(n.emission) for n in metagraph_clone.neurons]

        # Gather validator-specific data (tao/alpha price and TAO/USD price) if needed
        tao_per_alpha_price = None
        tao_to_usd_rate = None

        if self.is_validator:  # Only validators need price data for weight calculation
            tao_per_alpha_price = self._get_tao_per_alpha_price(metagraph_clone)
            tao_to_usd_rate = self._get_tao_usd_rate()

        # Log validator hotkeys (those with validator_permit=True) - log-only per CLAUDE.md (Null
        # consensus means only the weight setter holds the permit on this subnet)
        if self.running_unit_tests:
            validator_hotkeys = [
                metagraph_clone.hotkeys[i]
                for i, permit in enumerate(getattr(metagraph_clone, 'validator_permit', []))
                if permit
            ]
        else:
            validator_hotkeys = [orig.hotkey for orig in metagraph_clone.neurons if orig.validator_permit]
        logger.info(f"Validators with permit ({len(validator_hotkeys)}): {validator_hotkeys}")

        # The chain's per-UID stake weight (bittensor's metagraph.total_stake) identifies validators without the permit;
        # a failed subnet-state query leaves it empty, so the previous weights are kept rather than no validator at all
        if self.running_unit_tests:
            stake_weight = [float(s) for s in metagraph_clone.total_stake]
        else:
            stake_weight = [n.stake for n in converted_neurons]
        if len(stake_weight) != len(metagraph_clone.hotkeys):
            logger.warning(f"Stake weight has {len(stake_weight)} entries for {len(metagraph_clone.hotkeys)} hotkeys; "
                           f"keeping the previous stake weights until the next sync")
            stake_weight = None
        else:
            stake_validators = [hk for hk, s in zip(metagraph_clone.hotkeys, stake_weight)
                                if is_validator_stake_weight(s)]
            logger.info(f"Validators by stake weight ({len(stake_validators)}): {stake_validators}")

        # The metagraph server derives who registered after the standard miner cutoff block; the account
        # server gets a copy at the end, since its order path never reads core
        cutoff_block = ValiConfig.standard_miner_cutoff_block(self.is_testnet) if self.is_validator else None

        # Single atomic RPC call to update all metagraph fields
        # Much faster than multiple calls - all fields updated together under one lock
        if self.is_miner:
            axons = [n.axon_info for n in converted_neurons] if not self.running_unit_tests \
                else list(metagraph_clone.axons)
        else:
            axons = None
        self._metagraph_client.update_metagraph(
            neurons=converted_neurons,
            uids=uids,
            hotkeys=list(metagraph_clone.hotkeys),  # Server will update cached set
            block_at_registration=block_at_registration,
            axons=axons,
            emission=emission,
            tao_per_alpha_price=tao_per_alpha_price,
            tao_to_usd_rate=tao_to_usd_rate,
            stake_weight=stake_weight,
            standard_miner_cutoff_block=cutoff_block
        )

        # Update local hotkeys cache atomically (no lock needed - set assignment is atomic)
        self._hotkeys_cache = set(metagraph_clone.hotkeys)

        # self.log_metagraph_state()
        self.set_last_update_time()

        # Reset failure accounting only after a FULLY successful sync — the
        # early returns above (refresh rate-limit, anomalous metagraph) must
        # not clear a pending reconnect. This is the counterpart of the
        # increment in SubtensorOpsServer.run_daemon_iteration; a non-zero
        # counter is what arms the connection-recreation block at the top of
        # this method on the next call.
        if self.consecutive_failures > 0:
            logger.info(
                f"Metagraph update recovered after {self.consecutive_failures} "
                f"consecutive failures; resetting failure count."
            )
        self.consecutive_failures = 0

        # Last, so a slow account server never delays the sync itself
        if self.is_validator:
            self._push_registered_after_cutoff(cutoff_block)

    def _push_registered_after_cutoff(self, cutoff_block):
        """Copy the metagraph server's answers on who registered after the cutoff block to the account server."""
        try:
            self._miner_account_client.set_registered_after_cutoff(
                cutoff_block, self._metagraph_client.get_registered_after_cutoff())
        except Exception as e:
            logger.warning(f"Could not push who registered after the standard miner cutoff to the account "
                           f"server, which keeps its previous copy: {e}")


# len([x for x in self.metagraph.axons if '0.0.0.0' not in x.ip]), len([x for x in self.metagraph.neurons if '0.0.0.0' not in x.axon_info.for ip])
if __name__ == "__main__":
    from neurons.miner import Miner

    config = Miner.get_config()  # Must run this via commandline to populate correctly

    # Create SubtensorOpsManager (no position_inspector needed!)
    mgu = SubtensorOpsManager(config, config.wallet.hotkey, is_miner=True)

    while True:
        mgu.update_metagraph()
        time.sleep(60)
