# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time
import traceback
import threading
from dataclasses import dataclass
from setproctitle import setproctitle

from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.metagraph.metagraph_utils import is_anomalous_hotkey_loss
from shared_objects.locks.subtensor_lock import get_subtensor_lock
from shared_objects.rpc.rpc_client_base import RPCClientBase
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from time_util.time_util import TimeUtil

import bittensor as bt


# Simple picklable data structures for unit testing (must be module-level to be picklable)
@dataclass
class SimpleAxonInfo:
    """Simple picklable axon info for testing."""
    ip: str
    port: int


@dataclass
class SimpleNeuron:
    """Simple picklable neuron for testing."""
    uid: int
    hotkey: str
    incentive: float
    validator_trust: float
    axon_info: SimpleAxonInfo


# ==================== Client for WeightSetter RPC ====================

class MetagraphUpdaterClient(RPCClientBase):
    """
    RPC client for calling set_weights_rpc on MetagraphUpdater.

    Used by WeightCalculatorServer to send weight setting requests
    to MetagraphUpdater running in a separate process.

    Usage:
        client = MetagraphUpdaterClient()
        result = client.set_weights_rpc(uids=[1,2,3], weights=[0.3,0.3,0.4], version_key=200)
    """

    def __init__(self, running_unit_tests=False, connect_immediately=True):
        super().__init__(
            service_name=ValiConfig.RPC_WEIGHT_SETTER_SERVICE_NAME,
            port=ValiConfig.RPC_WEIGHT_SETTER_PORT,
            connect_immediately=connect_immediately and not running_unit_tests
        )
        self.running_unit_tests = running_unit_tests

    def set_weights_rpc(self, uids: list, weights: list, version_key: int) -> dict:
        """
        Send weight setting request to MetagraphUpdater.

        Args:
            uids: List of UIDs to set weights for
            weights: List of weights corresponding to UIDs
            version_key: Subnet version key

        Returns:
            dict: {"success": bool, "error": str or None}
        """
        return self.call("set_weights_rpc", uids, weights, version_key)


class WeightFailureTracker:
    """Track weight setting failures and manage alerting logic"""
    
    def __init__(self):
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.last_alert_time = 0
        self.failure_patterns = {}  # Track unknown error patterns
        self.had_critical_failure = False
        
    def classify_failure(self, err_msg):
        """Classify failure based on production patterns"""
        error_lower = err_msg.lower()
        
        # BENIGN - Don't alert (expected behavior)
        if any(phrase in error_lower for phrase in [
            "no attempt made. perhaps it is too soon to commit weights",
            "too soon to commit weights",
            "too soon to commit"
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


class MetagraphUpdater(CacheController):
    """
    Run locally to interface with the Subtensor object without RPC overhead
    """
    def __init__(self, config, hotkey, is_miner, position_inspector=None, position_manager=None,
                 slack_notifier=None, running_unit_tests=False):
        super().__init__()
        self.is_miner = is_miner
        self.is_validator = not is_miner
        self.config = config
        self.running_unit_tests = running_unit_tests

        # Initialize failure tracking BEFORE subtensor creation (needed if creation fails)
        self.consecutive_failures = 0

        # Create subtensor (mock if running unit tests)
        if running_unit_tests:
            self.subtensor = self._create_mock_subtensor()
        else:
            try:
                self.subtensor = bt.subtensor(config=self.config)
            except (ConnectionRefusedError, ConnectionError, OSError) as e:
                bt.logging.error(f"Failed to create initial subtensor connection: {e}")
                bt.logging.warning("Will retry during first metagraph update loop iteration")
                # Set to None - update loop will recreate it (using consecutive_failures > 0 logic)
                self.subtensor = None
                # Increment consecutive_failures so update loop tries to recreate immediately
                self.consecutive_failures = 1

        # Create own LivePriceFetcherClient for validators (forward compatibility - no parameter passing)
        # Only validators need this for TAO/USD price queries
        if self.is_validator:
            from vali_objects.price_fetcher import LivePriceFetcherClient
            self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests)
        else:
            assert position_inspector is not None, "Position inspector must be provided for miners"
            self._live_price_client = None
        # Parse out the arg for subtensor.network. If it is "finney" or "subvortex", we will roundrobin on metagraph failure
        self.round_robin_networks = ['finney', 'subvortex']
        self.round_robin_enabled = False
        self.current_round_robin_index = 0
        if self.config.subtensor.network in self.round_robin_networks:
            bt.logging.info(f"Using round-robin metagraph for network {self.config.subtensor.network}. ")
            self.round_robin_enabled = True
            self.current_round_robin_index = self.round_robin_networks.index(self.config.subtensor.network)

        # Initialize likely validators and miners with empty dictionaries. This maps hotkey to timestamp.
        self.likely_validators = {}
        self.likely_miners = {}
        self.hotkey = hotkey
        self.is_miner = is_miner
        self.interval_wait_time_ms = ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MINER_MS if self.is_miner else \
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_VALIDATOR_MS
        self.position_inspector = position_inspector
        self.position_manager = position_manager
        self.slack_notifier = slack_notifier  # Add slack notifier for error reporting

        # Weight setting for validators only (RPC-based, no queue)
        self.last_weight_set = 0
        self.weight_failure_tracker = WeightFailureTracker() if not is_miner else None
        self.rpc_server = None
        self.rpc_thread = None

        # Exponential backoff parameters
        self.min_backoff = 10 if self.round_robin_enabled else 120
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
        bt.logging.info(f"MetagraphUpdater initialized in {mode} mode, weight setting via RPC")

    @property
    def live_price_fetcher(self):
        """Get live price fetcher client (validators only)."""
        return self._live_price_client

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

            # Mock pool data
            mock_metagraph.pool = Mock()
            mock_metagraph.pool.tao_in = 1000.0
            mock_metagraph.pool.alpha_in = 5000.0

            return mock_metagraph

        mock_subtensor.metagraph = Mock(side_effect=mock_metagraph_func)

        # Mock set_weights method (for validators)
        mock_subtensor.set_weights = Mock(return_value=(True, None))

        # Mock substrate connection for cleanup
        mock_subtensor.substrate = Mock()
        mock_subtensor.substrate.close = Mock()

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
                axon_info = SimpleAxonInfo(ip="192.168.1.1", port=8091)
                neuron = SimpleNeuron(
                    uid=i,
                    hotkey=hk,
                    incentive=0.1,
                    validator_trust=0.1 if i == 0 else 0.0,  # First one is validator
                    axon_info=axon_info
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

            # Mock pool data
            mock_metagraph.pool = Mock()
            mock_metagraph.pool.tao_in = 1000.0
            mock_metagraph.pool.alpha_in = 5000.0

            return mock_metagraph

        self.subtensor.metagraph = Mock(side_effect=mock_metagraph_func)

    def _start_weight_setter_rpc_server(self):
        """Start RPC server for weight setting requests (validators only)."""
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

    # ==================== RPC Methods (exposed to SubtensorWeightCalculator) ====================

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
                wallet = bt.wallet(config=self.config)

            bt.logging.info(f"[RPC] Processing weight setting request for {len(uids)} UIDs")

            # Set weights with retry logic
            success, error_msg = self._set_weights_with_retry(
                netuid=netuid,
                wallet=wallet,
                uids=uids,
                weights=weights,
                version_key=version_key
            )

            if success:
                self.last_weight_set = time.time()
                bt.logging.success("[RPC] Weight setting completed successfully")

                # Track success and check for recovery alerts
                if self.weight_failure_tracker:
                    should_send_recovery = self.weight_failure_tracker.track_success()
                    if should_send_recovery and self.slack_notifier:
                        self._send_recovery_alert(wallet)

                return {"success": True, "error": None}
            else:
                bt.logging.warning(f"[RPC] Weight setting failed: {error_msg}")

                # Track failure and send alerts
                if self.weight_failure_tracker:
                    failure_type = self.weight_failure_tracker.classify_failure(error_msg)
                    self.weight_failure_tracker.track_failure(error_msg, failure_type)

                    if self.weight_failure_tracker.should_alert(failure_type, self.weight_failure_tracker.consecutive_failures):
                        self._send_weight_failure_alert(error_msg, failure_type, wallet)
                        self.weight_failure_tracker.last_alert_time = time.time()

                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Error in set_weights_rpc: {e}"
            bt.logging.error(error_msg)
            bt.logging.error(traceback.format_exc())
            return {"success": False, "error": error_msg}

    def _current_timestamp(self):
        return time.time()

    def _is_expired(self, timestamp):
        return (self._current_timestamp() - timestamp) > 86400  # 24 hours in seconds
    
    def _cleanup_subtensor_connection(self):
        """Safely close substrate connection to prevent file descriptor leaks"""
        if hasattr(self, 'subtensor') and self.subtensor:
            try:
                if hasattr(self.subtensor, 'substrate') and self.subtensor.substrate:
                    bt.logging.debug("Cleaning up substrate connection")
                    self.subtensor.substrate.close()
            except Exception as e:
                bt.logging.warning(f"Error during substrate cleanup: {e}")
    
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
        bt.logging.info("Waiting for initial metagraph population...")
        start_time = time.time()
        while not self._metagraph_client.get_hotkeys() and (time.time() - start_time) < max_wait_time:
            time.sleep(1)

        if not self._metagraph_client.get_hotkeys():
            error_msg = f"Failed to populate metagraph within {max_wait_time} seconds"
            bt.logging.error(error_msg)
            if slack_notifier:
                slack_notifier.send_message(f"❌ {error_msg}", level="error")
            exit()

        bt.logging.info(f"Metagraph populated with {len(self._metagraph_client.get_hotkeys())} hotkeys")
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
        bt.logging.enable_info()

        while not ShutdownCoordinator.is_shutdown():
            try:
                self.update_metagraph()
                # Reset backoff on successful update
                if self.consecutive_failures > 0:
                    rr_network = self.round_robin_networks[self.current_round_robin_index] if self.round_robin_enabled else "N/A"
                    bt.logging.info(
                        f"Metagraph update successful after {self.consecutive_failures} failures. Resetting backoff. "
                        f"round_robin_enabled: {self.round_robin_enabled}. rr_network: {rr_network}")
                    if self.slack_notifier:
                        self.slack_notifier.send_message(
                            f"✅ Metagraph update recovered after {self.consecutive_failures} consecutive failures."
                            f" round_robin_enabled: {self.round_robin_enabled}, rr_network: {rr_network}",
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
                rr_network = self.round_robin_networks[self.current_round_robin_index] if self.round_robin_enabled else "N/A"
                error_msg = (f"Error during metagraph update (attempt #{self.consecutive_failures}): {e}. "
                             f"Next retry in {self.current_backoff} seconds. round_robin_enabled: {self.round_robin_enabled}"
                             f" rr_network {rr_network}\n")
                bt.logging.error(error_msg)
                bt.logging.error(traceback.format_exc())

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
        """Set weights with round-robin retry using existing subtensor"""
        # Check if subtensor is available before attempting weight setting
        if self.subtensor is None:
            error_msg = "Subtensor connection not available (initialization or reconnection in progress)"
            bt.logging.error(error_msg)
            return False, error_msg

        max_retries = len(self.round_robin_networks) if self.round_robin_enabled else 1

        for attempt in range(max_retries):
            try:
                with get_subtensor_lock():
                    success, error_msg = self.subtensor.set_weights(
                        netuid=netuid,
                        wallet=wallet,
                        uids=uids,
                        weights=weights,
                        version_key=version_key
                    )

                bt.logging.info(f"Weight setting attempt {attempt + 1}: success={success}, error={error_msg}")
                return success, error_msg

            except Exception as e:
                bt.logging.warning(f"Weight setting failed (attempt {attempt + 1}): {e}")
                # Let the metagraph updater handle round-robin switching to avoid potential race conditions and rate limit issues
                #if self.round_robin_enabled and attempt < max_retries - 1:
                #    bt.logging.info("Switching to next network for weight setting retry")
                #    self._switch_to_next_network()
                #else:
                #    return False, str(e)

        return False, "All retry attempts failed"
    
    def _switch_to_next_network(self, cleanup_connection=True, create_new_subtensor=True):
        """Switch to the next network in round-robin
        
        Args:
            cleanup_connection (bool): Whether to cleanup existing subtensor connection
            create_new_subtensor (bool): Whether to create new subtensor instance
        """
        if not self.round_robin_enabled:
            return
            
        # Clean up existing connection if requested
        if cleanup_connection:
            self._cleanup_subtensor_connection()
        
        # Switch to next network
        self.current_round_robin_index = (self.current_round_robin_index + 1) % len(self.round_robin_networks)
        next_network = self.round_robin_networks[self.current_round_robin_index]
        
        bt.logging.info(f"Switching to next network: {next_network}")
        
        # Update config
        self.config.subtensor.network = next_network
        self.config.subtensor.chain_endpoint = f"wss://entrypoint-{next_network}.opentensor.ai:443"
        
        # For dict-style access (used in update_metagraph)
        if hasattr(self.config, '__getitem__'):
            self.config['subtensor']['network'] = next_network
        
        # Create new subtensor connection if requested
        if create_new_subtensor:
            if self.running_unit_tests:
                self.subtensor = self._create_mock_subtensor()
            else:
                self.subtensor = bt.subtensor(config=self.config)
    
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
                    bt.logging.warning("Wallet hotkey missing ss58_address attribute")
            else:
                bt.logging.warning("Wallet missing hotkey attribute")
        else:
            bt.logging.warning("Wallet parameter is None in weight failure alert")
        
        netuid = "unknown"
        network = "unknown"
        if self.config:
            if hasattr(self.config, 'netuid'):
                netuid = self.config.netuid
            else:
                bt.logging.warning("Config missing netuid attribute")
                
            if hasattr(self.config, 'subtensor'):
                if hasattr(self.config.subtensor, 'network'):
                    network = self.config.subtensor.network
                else:
                    bt.logging.warning("Config subtensor missing network attribute")
            else:
                bt.logging.warning("Config missing subtensor attribute")
        else:
            bt.logging.warning("Config is None - cannot determine network/netuid for alert")
            
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
                    bt.logging.warning("Wallet hotkey missing ss58_address attribute in recovery alert")
            else:
                bt.logging.warning("Wallet missing hotkey attribute in recovery alert")
        else:
            bt.logging.warning("Wallet parameter is None in recovery alert")
            
        network = "unknown"
        if self.config:
            if hasattr(self.config, 'subtensor'):
                if hasattr(self.config.subtensor, 'network'):
                    network = self.config.subtensor.network
                else:
                    bt.logging.warning("Config subtensor missing network attribute in recovery alert")
            else:
                bt.logging.warning("Config missing subtensor attribute in recovery alert")
        else:
            bt.logging.warning("Config is None - cannot determine network for recovery alert")
        
        message = f"✅ Weight setting recovered after failures\n" \
                 f"Network: {network}\n" \
                 f"Hotkey: {hotkey}"
        
        self.slack_notifier.send_message(message, level="info")

    def estimate_number_of_miners(self):
        # Filter out expired miners
        self.likely_miners = {k: v for k, v in self.likely_miners.items() if not self._is_expired(v)}
        hotkeys_with_incentive = {self.hotkey} if self.is_miner else set()
        for neuron in self._metagraph_client.get_neurons():
            if neuron.incentive > 0:
                hotkeys_with_incentive.add(neuron.hotkey)

        return len(hotkeys_with_incentive.union(set(self.likely_miners.keys())))

    def update_likely_validators(self, hotkeys):
        current_time = self._current_timestamp()
        for h in hotkeys:
            self.likely_validators[h] = current_time

    def update_likely_miners(self, hotkeys):
        current_time = self._current_timestamp()
        for h in hotkeys:
            self.likely_miners[h] = current_time

    def log_metagraph_state(self):
        n_validators = self.estimate_number_of_validators()
        n_miners = self.estimate_number_of_miners()
        if self.is_miner:
            n_miners = max(1, n_miners)
        else:
            n_validators = max(1, n_validators)

        bt.logging.info(
            f"metagraph state (approximation): {n_validators} active validators, {n_miners} active miners, hotkeys: "
            f"{len(self._metagraph_client.get_hotkeys())}")

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

    def _get_substrate_reserves(self, metagraph_clone):
        """
        Get TAO and ALPHA reserve balances from metagraph.pool.
        Uses built-in metagraph.pool data (verified to be identical to direct substrate queries).
        Fails fast - exceptions propagate to slack alert mechanism.

        Args:
            metagraph_clone: Freshly synced metagraph with pool data

        Returns:
            tuple: (tao_reserve_rao, alpha_reserve_rao)
        """
        # Extract reserve data from metagraph.pool
        if not hasattr(metagraph_clone, 'pool') or not metagraph_clone.pool:
            raise ValueError("metagraph.pool not available - cannot get reserve data")

        # Get reserves from pool (in tokens, need to convert to RAO)
        tao_reserve_tokens = metagraph_clone.pool.tao_in
        alpha_reserve_tokens = metagraph_clone.pool.alpha_in

        # Convert to RAO (1 token = 1e9 RAO)
        tao_reserve_rao = float(tao_reserve_tokens * 1e9)
        alpha_reserve_rao = float(alpha_reserve_tokens * 1e9)

        # Validate reserves
        if alpha_reserve_rao == 0:
            raise ValueError("Alpha reserve is zero - cannot calculate conversion rate")

        bt.logging.info(
            f"Got reserves from metagraph.pool: TAO={tao_reserve_rao / 1e9:.2f} TAO "
            f"({tao_reserve_rao:.0f} RAO), ALPHA={alpha_reserve_rao / 1e9:.2f} ALPHA "
            f"({alpha_reserve_rao:.0f} RAO)"
        )

        return tao_reserve_rao, alpha_reserve_rao

    def refresh_substrate_reserves(self, metagraph_clone):
        """
        Refresh TAO and ALPHA reserve balances from metagraph.pool and store in shared metagraph.
        DEPRECATED: Use _get_substrate_reserves() and update_metagraph() for atomic updates.

        Args:
            metagraph_clone: Freshly synced metagraph with pool data
        """
        tao_reserve_rao, alpha_reserve_rao = self._get_substrate_reserves(metagraph_clone)
        self._metagraph_client.set_tao_reserve_rao(tao_reserve_rao)
        self._metagraph_client.set_alpha_reserve_rao(alpha_reserve_rao)

    def _get_tao_usd_rate(self):
        """
        Get current TAO/USD price using live_price_fetcher.
        Uses current timestamp to get latest available price.

        Non-blocking: If price fetch fails, logs error and returns None.
        Better to use a slightly stale TAO/USD price than block metagraph updates.

        Returns:
            float: TAO/USD rate, or None if unavailable
        """
        try:
            if not self.live_price_fetcher:
                bt.logging.warning(
                    "live_price_fetcher not available - cannot query TAO/USD price. "
                    "Using existing price from metagraph (may be stale)."
                )
                return None

            # Get current timestamp for price query
            current_time_ms = TimeUtil.now_in_millis()

            # Query TAO/USD price at current time
            price_source = self.live_price_fetcher.get_close_at_date(
                TradePair.TAOUSD,
                current_time_ms
            )

            if not price_source or not hasattr(price_source, 'close') or price_source.close is None:
                bt.logging.warning(
                    f"TAO/USD price unavailable at timestamp {current_time_ms}. "
                    f"Using existing price from metagraph (may be stale). "
                    f"price_source={price_source}"
                )
                return None

            tao_to_usd_rate = float(price_source.close)

            # Validate price is reasonable
            if tao_to_usd_rate <= 0:
                bt.logging.warning(
                    f"Invalid TAO/USD price: ${tao_to_usd_rate}. "
                    f"Using existing price from metagraph (may be stale)."
                )
                return None

            bt.logging.info(
                f"Got TAO/USD price: ${tao_to_usd_rate:.2f}/TAO "
                f"(timestamp: {current_time_ms})"
            )
            return tao_to_usd_rate

        except Exception as e:
            bt.logging.error(
                f"Error fetching TAO/USD price: {e}. "
                f"Using existing price from metagraph (may be stale)."
            )
            bt.logging.error(traceback.format_exc())
            return None

    def refresh_tao_usd_price(self):
        """
        Refresh TAO/USD price using live_price_fetcher and store in shared metagraph.
        DEPRECATED: Use _get_tao_usd_rate() and update_metagraph() for atomic updates.

        Returns:
            bool: True if price was successfully updated, False otherwise
        """
        tao_to_usd_rate = self._get_tao_usd_rate()
        if tao_to_usd_rate:
            self.metagraph.set_tao_to_usd_rate(tao_to_usd_rate)
            return True
        return False

    def update_metagraph(self):
        if not self.refresh_allowed(self.interval_wait_time_ms):
            return

        if self.consecutive_failures > 0:
            if self.round_robin_enabled:
                # Use modularized round-robin switching
                bt.logging.warning(f"Switching to next network in round-robin due to consecutive failures")
                self._switch_to_next_network(cleanup_connection=False, create_new_subtensor=False)

            # Try to create new subtensor BEFORE cleaning up old one
            # This ensures we never leave self.subtensor in a broken state that breaks other components
            try:
                if self.running_unit_tests:
                    new_subtensor = self._create_mock_subtensor()
                else:
                    new_subtensor = bt.subtensor(config=self.config)

                # Only cleanup old connection after new one successfully created (prevents file descriptor leak)
                self._cleanup_subtensor_connection()
                self.subtensor = new_subtensor
                bt.logging.info("Successfully recreated subtensor connection after previous failures")

            except (ConnectionRefusedError, ConnectionError, OSError) as e:
                # Connection errors during subtensor creation - keep old subtensor and re-raise
                bt.logging.error(f"Failed to recreate subtensor connection (connection error): {e}")
                # Don't cleanup old connection - let it stay alive for other components (weight setting, etc.)
                # Re-raise so outer exception handler applies exponential backoff
                raise
            except Exception as e:
                # Other unexpected errors - still keep old subtensor but log differently
                bt.logging.error(f"Failed to recreate subtensor connection (unexpected error): {e}")
                # Don't cleanup old connection
                raise

        # Check if subtensor is available before attempting metagraph sync
        if self.subtensor is None:
            raise RuntimeError("Subtensor connection not available - cannot sync metagraph")

        recently_acked_miners = None
        recently_acked_validators = None
        if self.is_miner:
            recently_acked_validators = self.position_inspector.get_recently_acked_validators()
        else:
            # REMOVED: Expensive filesystem scan (127s) for unused log_metagraph_state() feature
            # if self.position_manager:
            #     recently_acked_miners = self.position_manager.get_recently_updated_miner_hotkeys()
            # else:
            #     recently_acked_miners = []
            recently_acked_miners = []

        hotkeys_before = set(self._metagraph_client.get_hotkeys())

        # Synchronize with weight setting operations to prevent WebSocket concurrency errors
        with get_subtensor_lock():
            metagraph_clone = self.subtensor.metagraph(self.config.netuid)

        assert hasattr(metagraph_clone, 'hotkeys'), "Metagraph clone does not have hotkeys attribute"
        bt.logging.info("Updating metagraph...")
        # metagraph_clone.sync(subtensor=self.subtensor) The call to subtensor.metagraph() already syncs the metagraph.
        hotkeys_after = set(metagraph_clone.hotkeys)
        lost_hotkeys = hotkeys_before - hotkeys_after
        gained_hotkeys = hotkeys_after - hotkeys_before
        if lost_hotkeys:
            bt.logging.info(f"metagraph has lost hotkeys: {lost_hotkeys}")
        if gained_hotkeys:
            bt.logging.info(f"metagraph has gained hotkeys: {gained_hotkeys}")
        if not lost_hotkeys and not gained_hotkeys:
            bt.logging.info(f"metagraph hotkeys remain the same. n = {len(hotkeys_after)}")

        # Use shared anomaly detection logic
        is_anomalous, percent_lost = is_anomalous_hotkey_loss(lost_hotkeys, len(hotkeys_before))
        # failsafe condition to reject new metagraph
        if is_anomalous:
            error_msg = (f"Too many hotkeys lost in metagraph update: {len(lost_hotkeys)} hotkeys lost, "
                         f"{percent_lost:.2f}% of total hotkeys. Rejecting new metagraph. ALERT A TEAM MEMBER ASAP...")
            bt.logging.error(error_msg)
            if self.slack_notifier:
                self.slack_notifier.send_message(
                    f"🚨 CRITICAL: {error_msg}",
                    level="error"
                )
            return  # Actually block the metagraph update

        # Gather validator-specific data (reserves and TAO/USD price) if needed
        tao_reserve_rao = None
        alpha_reserve_rao = None
        tao_to_usd_rate = None

        if self.is_validator:  # Only validators need reserves/prices for weight calculation
            tao_reserve_rao, alpha_reserve_rao = self._get_substrate_reserves(metagraph_clone)
            tao_to_usd_rate = self._get_tao_usd_rate()

        # Single atomic RPC call to update all metagraph fields
        # Much faster than multiple calls - all fields updated together under one lock
        self._metagraph_client.update_metagraph(
            neurons=list(metagraph_clone.neurons),
            uids=list(metagraph_clone.uids),
            hotkeys=list(metagraph_clone.hotkeys),  # Server will update cached set
            block_at_registration=list(metagraph_clone.block_at_registration),
            axons=list(metagraph_clone.axons) if self.is_miner else None,
            emission=list(metagraph_clone.emission),
            tao_reserve_rao=tao_reserve_rao,
            alpha_reserve_rao=alpha_reserve_rao,
            tao_to_usd_rate=tao_to_usd_rate
        )

        # Update local hotkeys cache atomically (no lock needed - set assignment is atomic)
        self._hotkeys_cache = set(metagraph_clone.hotkeys)

        if recently_acked_miners:
            self.update_likely_miners(recently_acked_miners)
        if recently_acked_validators:
            self.update_likely_validators(recently_acked_validators)

        # self.log_metagraph_state()
        self.set_last_update_time()


# len([x for x in self.metagraph.axons if '0.0.0.0' not in x.ip]), len([x for x in self.metagraph.neurons if '0.0.0.0' not in x.axon_info.for ip])
if __name__ == "__main__":
    from neurons.miner import Miner
    from miner_objects.position_inspector import PositionInspector
    from shared_objects.rpc.metagraph_server import MetagraphClient

    config = Miner.get_config()  # Must run this via commandline to populate correctly

    # Create MetagraphClient (not raw metagraph)
    metagraph_client = MetagraphClient()

    # Create PositionInspector with client
    position_inspector = PositionInspector(bt.wallet(config=config), metagraph_client, config)

    # Create MetagraphUpdater
    mgu = MetagraphUpdater(config, config.wallet.hotkey, is_miner=True, position_inspector=position_inspector)

    while True:
        mgu.update_metagraph()
        time.sleep(60)
