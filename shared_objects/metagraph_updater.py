# developer: jbonilla
# Copyright © 2024 Taoshi Inc

import time
import traceback
import threading
import queue
from setproctitle import setproctitle

from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.metagraph_utils import is_anomalous_hotkey_loss
from shared_objects.subtensor_lock import get_subtensor_lock
from time_util.time_util import TimeUtil

import bittensor as bt


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
    def __init__(self, config, metagraph, hotkey, is_miner, position_inspector=None, position_manager=None,
                 shutdown_dict=None, slack_notifier=None, weight_request_queue=None, live_price_fetcher=None):
        super().__init__(metagraph)
        self.config = config
        self.subtensor = bt.subtensor(config=self.config)
        self.live_price_fetcher = live_price_fetcher  # For TAO/USD price queries (validators only)
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
        if is_miner:
            assert position_inspector is not None, "Position inspector must be provided for miners"
        self.is_miner = is_miner
        self.interval_wait_time_ms = ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_MINER_MS if self.is_miner else \
            ValiConfig.METAGRAPH_UPDATE_REFRESH_TIME_VALIDATOR_MS
        self.position_inspector = position_inspector
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict  # Flag to control the loop
        self.slack_notifier = slack_notifier  # Add slack notifier for error reporting

        # Weight setting for validators only
        self.weight_request_queue = weight_request_queue if not is_miner else None
        self.last_weight_set = 0
        self.weight_failure_tracker = WeightFailureTracker() if not is_miner else None

        # Exponential backoff parameters
        self.min_backoff = 10 if self.round_robin_enabled else 120
        self.max_backoff = 43200  # 12 hours maximum (12 * 60 * 60)
        self.backoff_factor = 2  # Double the wait time on each retry
        self.current_backoff = self.min_backoff
        self.consecutive_failures = 0
        
        # Log mode
        mode = "miner" if is_miner else "validator"
        weight_mode = "enabled" if self.weight_request_queue else "disabled"
        bt.logging.info(f"MetagraphUpdater initialized in {mode} mode, weight setting: {weight_mode}")

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
        while not self.metagraph.get_hotkeys() and (time.time() - start_time) < max_wait_time:
            time.sleep(1)

        if not self.metagraph.get_hotkeys():
            error_msg = f"Failed to populate metagraph within {max_wait_time} seconds"
            bt.logging.error(error_msg)
            if slack_notifier:
                slack_notifier.send_message(f"❌ {error_msg}", level="error")
            exit()

        bt.logging.info(f"Metagraph populated with {len(self.metagraph.get_hotkeys())} hotkeys")
        return updater_thread

    def estimate_number_of_validators(self):
        # Filter out expired validators
        self.likely_validators = {k: v for k, v in self.likely_validators.items() if not self._is_expired(v)}
        hotkeys_with_v_trust = set() if self.is_miner else {self.hotkey}
        for neuron in self.metagraph.get_neurons():
            if neuron.validator_trust > 0:
                hotkeys_with_v_trust.add(neuron.hotkey)
        return len(hotkeys_with_v_trust.union(set(self.likely_validators.keys())))

    def run_update_loop(self):
        mode_name = "miner" if self.is_miner else "validator"
        setproctitle(f"metagraph_updater_{mode_name}_{self.hotkey}")
        bt.logging.enable_info()
        
        while not self.shutdown_dict:
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

    def run_weight_processing_loop(self):
        """
        Dedicated loop for processing weight requests using blocking queue.
        Runs in a separate thread. Blocks until a request arrives (efficient, no polling).
        """
        setproctitle(f"weight_processor_{self.hotkey}")
        bt.logging.enable_info()
        bt.logging.info("Starting dedicated weight processing loop")

        while not self.shutdown_dict:
            try:
                # Process weight requests if we're a validator
                if self.weight_request_queue:
                    self._process_weight_requests()
                else:
                    time.sleep(5)  # No queue configured, sleep

            except Exception as e:
                bt.logging.error(f"Error in weight processing loop: {e}")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)  # Longer sleep on error

        bt.logging.info("Weight processing loop shutting down")

    def _process_weight_requests(self):
        """
        Process pending weight setting requests (validators only).
        Uses blocking queue.get() to efficiently wait for requests without polling.
        """
        try:
            # Block until a request arrives (timeout 30s to check shutdown_dict periodically)
            try:
                request = self.weight_request_queue.get(timeout=30)
                self._handle_weight_request(request)

                # After processing first request, drain any additional pending requests
                # (non-blocking to avoid waiting if queue is empty)
                processed_count = 1
                while processed_count < 5:  # Process max 5 requests per cycle
                    try:
                        request = self.weight_request_queue.get_nowait()
                        self._handle_weight_request(request)
                        processed_count += 1
                    except queue.Empty:
                        break  # No more pending requests

                if processed_count > 1:
                    bt.logging.debug(f"Processed {processed_count} weight requests in batch")

            except queue.Empty:
                # Timeout reached, no requests - this is normal, just loop back
                pass

        except Exception as e:
            bt.logging.error(f"Error processing weight requests: {e}")
            bt.logging.error(traceback.format_exc())
    
    def _handle_weight_request(self, request):
        """Handle a single weight setting request (no response needed)"""
        try:
            uids = request['uids']
            weights = request['weights']
            version_key = request['version_key']
            
            # Use our own config for netuid
            netuid = self.config.netuid
            
            # Create wallet from our own config
            wallet = bt.wallet(config=self.config)
            
            bt.logging.info(f"Processing weight setting request for {len(uids)} UIDs")
            
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
                bt.logging.success("Weight setting completed successfully")
                
                # Track success and check for recovery alerts
                if self.weight_failure_tracker:
                    should_send_recovery = self.weight_failure_tracker.track_success()
                    if should_send_recovery and self.slack_notifier:
                        self._send_recovery_alert(wallet)
            else:
                bt.logging.warning(f"Weight setting failed: {error_msg}")
                
                # Track failure and send alerts
                if self.weight_failure_tracker:
                    failure_type = self.weight_failure_tracker.classify_failure(error_msg)
                    self.weight_failure_tracker.track_failure(error_msg, failure_type)
                    
                    if self.weight_failure_tracker.should_alert(failure_type, self.weight_failure_tracker.consecutive_failures):
                        self._send_weight_failure_alert(error_msg, failure_type, wallet)
                        self.weight_failure_tracker.last_alert_time = time.time()
            
        except Exception as e:
            bt.logging.error(f"Error handling weight request: {e}")
            bt.logging.error(traceback.format_exc())
    
    def _set_weights_with_retry(self, netuid, wallet, uids, weights, version_key):
        """Set weights with round-robin retry using existing subtensor"""
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
        for neuron in self.metagraph.get_neurons():
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
            f"{len(self.metagraph.get_hotkeys())}")

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
        return self.metagraph

    def refresh_substrate_reserves(self, metagraph_clone):
        """
        Refresh TAO and ALPHA reserve balances from metagraph.pool and store in shared metagraph.
        Uses built-in metagraph.pool data (verified to be identical to direct substrate queries).
        Fails fast - exceptions propagate to slack alert mechanism.

        Args:
            metagraph_clone: Freshly synced metagraph with pool data
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

        # Update shared metagraph (accessible from all processes via IPC)
        # Use .value accessor for manager.Value() thread-safe synchronization
        self.metagraph.tao_reserve_rao.value = tao_reserve_rao
        self.metagraph.alpha_reserve_rao.value = alpha_reserve_rao

        bt.logging.info(
            f"Updated reserves from metagraph.pool: TAO={tao_reserve_rao / 1e9:.2f} TAO "
            f"({tao_reserve_rao:.0f} RAO), ALPHA={alpha_reserve_rao / 1e9:.2f} ALPHA "
            f"({alpha_reserve_rao:.0f} RAO)"
        )

    def refresh_tao_usd_price(self):
        """
        Refresh TAO/USD price using live_price_fetcher and store in shared metagraph.
        Uses current timestamp to get latest available price.

        Non-blocking: If price refresh fails, logs error but continues metagraph update.
        Better to use a slightly stale TAO/USD price than block metagraph updates.

        Returns:
            bool: True if price was successfully updated, False otherwise
        """
        try:
            if not self.live_price_fetcher:
                bt.logging.warning(
                    "live_price_fetcher not available - cannot query TAO/USD price. "
                    "Using existing price from metagraph (may be stale)."
                )
                return False

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
                return False

            tao_to_usd_rate = float(price_source.close)

            # Validate price is reasonable
            if tao_to_usd_rate <= 0:
                bt.logging.warning(
                    f"Invalid TAO/USD price: ${tao_to_usd_rate}. "
                    f"Using existing price from metagraph (may be stale)."
                )
                return False

            # Update shared metagraph (accessible from all processes via IPC)
            self.metagraph.tao_to_usd_rate = tao_to_usd_rate

            bt.logging.info(
                f"Updated TAO/USD price: ${tao_to_usd_rate:.2f}/TAO "
                f"(timestamp: {current_time_ms})"
            )
            return True

        except Exception as e:
            bt.logging.error(
                f"Error refreshing TAO/USD price: {e}. "
                f"Using existing price from metagraph (may be stale)."
            )
            bt.logging.error(traceback.format_exc())
            return False

    def update_metagraph(self):
        if not self.refresh_allowed(self.interval_wait_time_ms):
            return

        if self.consecutive_failures > 0:
            if self.round_robin_enabled:
                # Use modularized round-robin switching
                bt.logging.warning(f"Switching to next network in round-robin due to consecutive failures")
                self._switch_to_next_network(cleanup_connection=False, create_new_subtensor=False)
            
            # CRITICAL: Close existing connection before creating new one to prevent file descriptor leak
            self._cleanup_subtensor_connection()
            self.subtensor = bt.subtensor(config=self.config)
        recently_acked_miners = None
        recently_acked_validators = None
        if self.is_miner:
            recently_acked_validators = self.position_inspector.get_recently_acked_validators()
        else:
            if self.position_manager:
                recently_acked_miners = self.position_manager.get_recently_updated_miner_hotkeys()
            else:
                recently_acked_miners = []

        hotkeys_before = set(self.metagraph.hotkeys)
        
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

        self.sync_lists(self.metagraph.neurons, list(metagraph_clone.neurons), brute_force=True)
        self.sync_lists(self.metagraph.uids, metagraph_clone.uids, brute_force=True)
        self.sync_lists(self.metagraph.hotkeys, metagraph_clone.hotkeys, brute_force=True)
        # Tuple doesn't support item assignment.
        self.sync_lists(self.metagraph.block_at_registration, metagraph_clone.block_at_registration,
                        brute_force=True)
        if self.is_miner:
            self.sync_lists(self.metagraph.axons, metagraph_clone.axons, brute_force=True)

        if recently_acked_miners:
            self.update_likely_miners(recently_acked_miners)
        if recently_acked_validators:
            self.update_likely_validators(recently_acked_validators)

        # Update shared emission data (TAO per tempo for each UID)
        self.sync_lists(self.metagraph.emission, metagraph_clone.emission, brute_force=True)

        # Refresh reserve data (TAO and ALPHA) from metagraph.pool for debt-based scoring
        # Also refresh TAO/USD price for USD-based payout calculations
        if not self.is_miner:  # Only validators need this for weight calculation
            self.refresh_substrate_reserves(metagraph_clone)
            self.refresh_tao_usd_price()

        # self.log_metagraph_state()
        self.set_last_update_time()


# len([x for x in self.metagraph.axons if '0.0.0.0' not in x.ip]), len([x for x in self.metagraph.neurons if '0.0.0.0' not in x.axon_info.for ip])
if __name__ == "__main__":
    from neurons.miner import Miner
    from miner_objects.position_inspector import PositionInspector

    config = Miner.get_config()  # Must run this via commandline to populate correctly
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    position_inspector = PositionInspector(bt.wallet(config=config), metagraph, config)
    mgu = MetagraphUpdater(config, metagraph, "test", is_miner=True, position_inspector=position_inspector)
    while True:
        mgu.update_metagraph()
        time.sleep(60)
