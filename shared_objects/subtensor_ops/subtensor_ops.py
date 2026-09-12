# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

import time
import traceback
import threading
import asyncio

from dataclasses import dataclass
from setproctitle import setproctitle

from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.subtensor_ops.metagraph_utils import is_anomalous_hotkey_loss
from shared_objects.locks.subtensor_lock import get_subtensor_lock
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from time_util.time_util import TimeUtil

import bittensor as bt
from bittensor.wallet import Wallet
import logging
from shared_objects.log import logger
from shared_objects.bt_config import make_subtensor, make_wallet


# ──────────────────────────────────────────────────────────────────────────────
# Bt11 metagraph adapters
#
# bt11 MetagraphNeuron presents a different interface than bt10 NeuronInfo:
#   - n.axon is "ip:port" string (not AxonInfo object)
#   - n.emission is Balance (not float)
#   - no mg.neurons, mg.uids, mg.axons etc. (iterate mg directly)
#   - no mg.pool (pool reserves queried separately)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AxonInfoAdapter:
    """Bt10-compatible AxonInfo adapter built from a bt11 'ip:port' axon string."""
    ip: str
    port: int
    hotkey: str = ""

    @classmethod
    def from_axon_str(cls, axon_str, hotkey: str = "") -> "AxonInfoAdapter":
        if axon_str and ':' in str(axon_str):
            parts = str(axon_str).rsplit(':', 1)
            try:
                return cls(ip=parts[0], port=int(parts[1]), hotkey=hotkey)
            except (ValueError, IndexError):
                pass
        return cls(ip='0.0.0.0', port=0, hotkey=hotkey)


class NeuronAdapter:
    """Bt10-compatible NeuronInfo adapter wrapping a bt11 MetagraphNeuron."""

    def __init__(self, neuron_bt11):
        self._n = neuron_bt11
        self.uid = int(neuron_bt11.uid)
        self.hotkey = neuron_bt11.hotkey
        self.validator_trust = float(getattr(neuron_bt11, 'validator_trust', 0.0))
        self.validator_permit = bool(getattr(neuron_bt11, 'validator_permit', False))
        self.incentive = float(getattr(neuron_bt11, 'incentive', 0.0))
        # bt11 stake is a Balance; expose as numeric for comparisons
        _stake = getattr(neuron_bt11, 'total_stake', getattr(neuron_bt11, 'alpha_stake', 0))
        self.stake = float(getattr(_stake, 'amount', _stake) or 0.0)
        # Build bt10-compatible axon_info from bt11's "ip:port" string
        self.axon_info = AxonInfoAdapter.from_axon_str(
            getattr(neuron_bt11, 'axon', None), hotkey=self.hotkey
        )


def _emission_float(emission_balance) -> float:
    """Convert a bt11 emission Balance (or plain number) to a Python float."""
    if emission_balance is None:
        return 0.0
    try:
        return float(emission_balance.amount)
    except AttributeError:
        pass
    try:
        return float(emission_balance)
    except (TypeError, ValueError):
        return 0.0


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
            "too soon to commit",
            "empty response from set_weights",
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

        self._metagraph_client = None

        # Initialize failure tracking BEFORE subtensor creation (needed if creation fails)
        self.consecutive_failures = 0

        # Create subtensor (mock if running unit tests)
        if running_unit_tests:
            self.subtensor = self._create_mock_subtensor()
        else:
            try:
                self.subtensor = make_subtensor(self.config)
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
        else:
            self._live_price_client = None
        # Round-robin on metagraph failure for known public networks.
        # 'subvortex' was removed: it is absent from bittensor's NETWORKS and
        # entrypoint-subvortex.opentensor.ai is NXDOMAIN, so rotating onto it
        # guaranteed a wasted retry cycle. With a single network the rotation
        # degenerates to a same-endpoint reconnect, which is the actual healing
        # mechanism (the recreate block in update_metagraph).
        self.round_robin_networks = ['finney']
        self.round_robin_enabled = False
        self.current_round_robin_index = 0
        if self.config.subtensor.network in self.round_robin_networks:
            # Guard: rotation rewrites chain_endpoint with the public
            # entrypoint template. An operator who passed a CUSTOM
            # --subtensor.chain_endpoint (e.g. a local node) keeps the default
            # network name ('finney'), and rotation would clobber their
            # endpoint irrecoverably on the first failure — so only enable
            # rotation when the configured endpoint IS a default entrypoint.
            configured_endpoint = getattr(self.config.subtensor, 'chain_endpoint', None)
            default_endpoints = {None, ''} | {
                f"wss://entrypoint-{n}.opentensor.ai:443" for n in self.round_robin_networks
            }
            if configured_endpoint in default_endpoints:
                logger.info(f"Using round-robin metagraph for network {self.config.subtensor.network}. ")
                self.round_robin_enabled = True
                self.current_round_robin_index = self.round_robin_networks.index(self.config.subtensor.network)
            else:
                logger.info(
                    f"Custom chain_endpoint configured ({configured_endpoint}); "
                    f"round-robin rotation disabled to preserve it. Connection "
                    f"recreation on failure remains active."
                )

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
        logger.info(f"SubtensorOpsManager initialized in {mode} mode, weight setting via RPC")

    def _create_mock_subtensor(self):
        """Create a mock subtensor for unit testing."""
        from unittest.mock import Mock

        mock_subtensor = Mock()

        # Mock metagraph() method to return empty bt11-compatible metagraph
        def mock_metagraph_func(netuid):
            mock_metagraph = Mock()
            mock_metagraph.hotkeys = []
            # bt11: metagraph is iterable (yields MetagraphNeuron objects)
            mock_metagraph.__iter__ = Mock(return_value=iter([]))

            return mock_metagraph

        # bt11: subtensor.subnets.metagraph(netuid) instead of subtensor.metagraph(netuid)
        mock_subtensor.subnets = Mock()
        mock_subtensor.subnets.metagraph = Mock(side_effect=mock_metagraph_func)
        # Keep old .metagraph for any remaining bt10-style callers
        mock_subtensor.metagraph = Mock(side_effect=mock_metagraph_func)

        # Mock execute() method for bt11 set_weights (bt.SetWeights intent)
        mock_response = Mock()
        mock_response.success = True
        mock_response.message = None
        mock_response.error = None
        mock_subtensor.execute = Mock(return_value=mock_response)
        # Keep old set_weights for any remaining bt10-style callers
        mock_subtensor.set_weights = Mock(return_value=mock_response)

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

        # Build bt11-compatible neuron objects for the mock metagraph
        class _MockBt11Neuron:
            """Minimal bt11 MetagraphNeuron for unit test mocking."""
            def __init__(self, uid, hotkey, validator_trust, axon_info):
                self.uid = uid
                self.hotkey = hotkey
                self.validator_trust = validator_trust
                self.validator_permit = validator_trust > 0
                self.incentive = 0.1
                self.emission = 1.0
                self.total_stake = 0.0
                self.alpha_stake = 0.0
                # axon as "ip:port" string
                self.axon = f"{axon_info.ip}:{axon_info.port}" if axon_info else None

        bt11_neurons = [
            _MockBt11Neuron(n.uid, n.hotkey, n.validator_trust, getattr(n, 'axon_info', None))
            for n in neurons
        ]

        # Update the mock metagraph function to return bt11-compatible data
        def mock_metagraph_func(netuid, _neurons=bt11_neurons, _hotkeys=hotkeys):
            mock_metagraph = Mock()
            mock_metagraph.hotkeys = list(_hotkeys)
            # bt11: metagraph is iterable
            mock_metagraph.__iter__ = Mock(return_value=iter(list(_neurons)))
            return mock_metagraph

        self.subtensor.subnets = Mock()
        self.subtensor.subnets.metagraph = Mock(side_effect=mock_metagraph_func)
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

    def broadcast_to_validators_rpc(self, synapse, validator_axons_list):
        """
        RPC method to broadcast a synapse to validators (called from ValidatorBroadcastBase).

        This method runs the broadcast using the SubtensorOpsManager's wallet and dendrite,
        allowing processes without direct subtensor/wallet access to send broadcasts.

        Args:
            synapse: The synapse object to broadcast (already validated as picklable)
            validator_axons_list: List of axon_info objects to broadcast to

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

            # Validate synapse object
            if not synapse or not hasattr(synapse, '__class__'):
                raise ValueError("Invalid synapse object")

            synapse_class_name = synapse.__class__.__name__

            # Create wallet from config
            wallet = make_wallet(self.config)

            target_hotkeys = [a.hotkey for a in validator_axons_list]
            logger.info(f"[BROADCAST RPC] Broadcasting {synapse_class_name} to {len(validator_axons_list)} validators: {target_hotkeys}")

            async def do_broadcast():
                """Send synapse to each validator axon using httpx + bt.http_auth.sign."""
                import httpx

                body = synapse.model_dump_json().encode()
                synapse_class_name = synapse.__class__.__name__
                path = f"/axon/{synapse_class_name}"

                success_count = 0
                errors = []
                successful_hotkeys = []

                async with httpx.AsyncClient(timeout=30.0) as client:
                    for axon in validator_axons_list:
                        axon_ip = getattr(axon, 'ip', None) or '0.0.0.0'
                        axon_port = getattr(axon, 'port', 8091)
                        axon_hotkey = getattr(axon, 'hotkey', '?')
                        if axon_ip in ('0.0.0.0', '', None):
                            errors.append(f"{axon_hotkey}: no valid IP")
                            continue
                        url = f"http://{axon_ip}:{axon_port}{path}"
                        try:
                            headers = bt.http_auth.sign(
                                wallet,
                                method="POST",
                                path=path,
                                body=body,
                                receiver_ss58=axon_hotkey if axon_hotkey != '?' else None,
                            )
                        except Exception as sign_err:
                            logger.warning(f"[BROADCAST RPC] http_auth.sign failed for {axon_hotkey}: {sign_err}; sending unsigned")
                            headers = {}
                        try:
                            resp = await client.post(url, content=body, headers=headers)
                            if resp.status_code == 200:
                                result_synapse = synapse.__class__.model_validate_json(resp.content)
                                if result_synapse.successfully_processed:
                                    success_count += 1
                                    successful_hotkeys.append(axon_hotkey)
                                else:
                                    errors.append(f"{axon_hotkey}: {result_synapse.error_message}")
                            else:
                                errors.append(f"{axon_hotkey}: HTTP {resp.status_code}")
                        except Exception as req_err:
                            errors.append(f"{axon_hotkey}: {req_err}")

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
                wallet = make_wallet(self.config)

            logger.info(f"[RPC] Processing weight setting request for {len(uids)} UIDs")

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
                    failure_type = self.weight_failure_tracker.classify_failure(error_msg)
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
    
    def _cleanup_subtensor_connection(self):
        """Safely close substrate connection to prevent file descriptor leaks"""
        if hasattr(self, 'subtensor') and self.subtensor:
            try:
                if hasattr(self.subtensor, 'substrate') and self.subtensor.substrate:
                    logger.debug("Cleaning up substrate connection")
                    self.subtensor.substrate.close()
            except Exception as e:
                logger.warning(f"Error during substrate cleanup: {e}")
    
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
                    rr_network = self.round_robin_networks[self.current_round_robin_index] if self.round_robin_enabled else "N/A"
                    logger.info(
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
        """Set weights with round-robin retry using existing subtensor"""
        # Check if subtensor is available before attempting weight setting
        if self.subtensor is None:
            error_msg = "Subtensor connection not available (initialization or reconnection in progress)"
            logger.error(error_msg)
            return False, error_msg

        max_retries = len(self.round_robin_networks) if self.round_robin_enabled else 1

        for attempt in range(max_retries):
            try:
                with get_subtensor_lock():
                    response = self.subtensor.execute(
                        bt.SetWeights(
                            netuid=netuid,
                            uids=list(uids),
                            weights=list(weights),
                            mechid=0,
                            version_key=version_key,
                        ),
                        wallet,
                    )

                success = response.success
                if not response.success:
                    err = getattr(response, 'error', None)
                    if err is not None:
                        remediation = getattr(err, 'remediation', None)
                        code = getattr(err, 'code', None)
                        error_msg = str(remediation or code or err)
                    else:
                        error_msg = getattr(response, 'message', None) or "set_weights returned failure"
                else:
                    error_msg = getattr(response, 'message', '') or ''
                logger.info(f"Weight setting attempt {attempt + 1}: success={success}, error={error_msg}")
                return success, error_msg

            except Exception as e:
                logger.warning(f"Weight setting failed (attempt {attempt + 1}): {e}")
                # Let the metagraph updater handle round-robin switching to avoid potential race conditions and rate limit issues
                #if self.round_robin_enabled and attempt < max_retries - 1:
                #    logger.info("Switching to next network for weight setting retry")
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
        
        logger.info(f"Switching to next network: {next_network}")
        
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
                self.subtensor = make_subtensor(self.config)
    
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

    def _get_substrate_reserves(self, netuid: int):
        """
        Get TAO and ALPHA reserve balances for the subnet.

        In bt11, metagraph.pool is gone.  We first try bt11 storage queries,
        then fall back to the alpha_price ratio read.

        Args:
            netuid: The subnet netuid

        Returns:
            tuple: (tao_reserve_rao, alpha_reserve_rao)
        """
        # bt11: try direct storage queries for pool reserves
        try:
            tao_in = self.subtensor.query(bt.storage.SubtensorModule.SubnetTaoIn, [netuid])
            alpha_in = self.subtensor.query(bt.storage.SubtensorModule.SubnetAlphaIn, [netuid])
            tao_reserve_rao = float(tao_in) * 1e9
            alpha_reserve_rao = float(alpha_in) * 1e9
            if alpha_reserve_rao == 0:
                raise ValueError("Alpha reserve is zero")
            logger.info(
                f"Got reserves via storage query: TAO={tao_reserve_rao/1e9:.2f}, "
                f"ALPHA={alpha_reserve_rao/1e9:.2f}"
            )
            return tao_reserve_rao, alpha_reserve_rao
        except Exception as storage_err:
            logger.debug(f"Storage query for reserves failed ({storage_err}), trying alpha_price")

        # Fallback: derive from alpha_price (TAO per 1 ALPHA)
        try:
            alpha_price = float(self.subtensor.prices.alpha_price(netuid=netuid))
            if alpha_price <= 0:
                raise ValueError(f"Invalid alpha_price: {alpha_price}")
            # Express as a unit ratio (alpha_reserve = 1.0 normalized)
            alpha_reserve_rao = 1.0
            tao_reserve_rao = alpha_price
            logger.info(f"Got reserves from alpha_price: {alpha_price} TAO/ALPHA")
            return tao_reserve_rao, alpha_reserve_rao
        except Exception as price_err:
            raise ValueError(
                f"Cannot determine subnet reserves: storage_err={storage_err}, price_err={price_err}"
            )

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

    def update_metagraph(self):
        if not self.refresh_allowed(self.interval_wait_time_ms):
            return

        if self.consecutive_failures > 0:
            if self.round_robin_enabled:
                # Use modularized round-robin switching
                logger.warning("Switching to next network in round-robin due to consecutive failures")
                self._switch_to_next_network(cleanup_connection=False, create_new_subtensor=False)

            # Try to create new subtensor BEFORE cleaning up old one
            # This ensures we never leave self.subtensor in a broken state that breaks other components
            try:
                if self.running_unit_tests:
                    new_subtensor = self._create_mock_subtensor()
                else:
                    new_subtensor = make_subtensor(self.config)

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
            # bt11: subtensor.subnets.metagraph(netuid) instead of subtensor.metagraph(netuid)
            metagraph_clone = self.subtensor.subnets.metagraph(self.config.netuid)

        assert hasattr(metagraph_clone, 'hotkeys'), "Metagraph clone does not have hotkeys attribute"
        logger.info("Updating metagraph...")
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

        # Build bt10-compatible neuron/axon adapter lists from bt11 metagraph
        # bt11 metagraph is iterable (yields MetagraphNeuron objects ordered by uid)
        neurons_bt11 = list(metagraph_clone)
        neurons = [NeuronAdapter(n) for n in neurons_bt11]
        uids = [n.uid for n in neurons]
        hotkeys = list(metagraph_clone.hotkeys)
        emission = [_emission_float(getattr(n, 'emission', 0.0)) for n in neurons_bt11]
        validator_permit = [bool(getattr(n, 'validator_permit', False)) for n in neurons_bt11]
        # block_at_registration not available in bt11 metagraph; preserve existing or use 0
        block_at_registration = [0] * len(hotkeys)
        axons = [n.axon_info for n in neurons] if self.is_miner else None

        # Gather validator-specific data (reserves and TAO/USD price) if needed
        tao_reserve_rao = None
        alpha_reserve_rao = None
        tao_to_usd_rate = None

        if self.is_validator and not self.running_unit_tests:
            tao_reserve_rao, alpha_reserve_rao = self._get_substrate_reserves(self.config.netuid)
            tao_to_usd_rate = self._get_tao_usd_rate()

        # Log validator hotkeys (those with validator_permit=True)
        validator_hotkeys = [hotkeys[i] for i, permit in enumerate(validator_permit) if permit]
        logger.info(f"Validators with permit ({len(validator_hotkeys)}): {validator_hotkeys}")

        # Single atomic RPC call to update all metagraph fields
        # Much faster than multiple calls - all fields updated together under one lock
        self._metagraph_client.update_metagraph(
            neurons=neurons,
            uids=uids,
            hotkeys=hotkeys,  # Server will update cached set
            block_at_registration=block_at_registration,
            axons=axons,
            emission=emission,
            tao_reserve_rao=tao_reserve_rao,
            alpha_reserve_rao=alpha_reserve_rao,
            tao_to_usd_rate=tao_to_usd_rate
        )

        # Update local hotkeys cache atomically (no lock needed - set assignment is atomic)
        self._hotkeys_cache = set(hotkeys)

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


# len([x for x in self.metagraph.axons if '0.0.0.0' not in x.ip]), len([x for x in self.metagraph.neurons if '0.0.0.0' not in x.axon_info.for ip])
if __name__ == "__main__":
    from neurons.miner import Miner

    config = Miner.get_config()  # Must run this via commandline to populate correctly

    # Create SubtensorOpsManager (no position_inspector needed!)
    mgu = SubtensorOpsManager(config, config.wallet.hotkey, is_miner=True)

    while True:
        mgu.update_metagraph()
        time.sleep(60)
