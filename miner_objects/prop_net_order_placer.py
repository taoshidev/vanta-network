# The MIT License (MIT)
# Copyright (c) 2024 Yuma Rao
# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
import asyncio
import json
import os
import threading
import time

from bittensor.core.synapse import Synapse
import bittensor as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

from miner_config import MinerConfig
from template.protocol import SendSignal
from vali_objects.vali_config import TradePair
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

REPO_VERSION = 'unknown'
with open(ValiBkpUtils.get_meta_json_path(), 'r') as f:
    REPO_VERSION = json.loads(f.read()).get("subnet_version", "unknown")


class SignalMetrics:
    def __init__(self, signal_uuid: str, trade_pair_id: str):
        self.signal_uuid = signal_uuid
        self.trade_pair_id = trade_pair_id
        self.network_start_time = None
        self.network_end_time = None
        self.validators_attempted = 0
        self.validators_succeeded = 0
        self.high_trust_total = 0
        self.high_trust_succeeded = 0
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.validator_errors: Dict[str, List[str]] = defaultdict(list)
        self.validator_response_times: Dict[str, float] = {}  # Only successful ones
        self.all_high_trust_succeeded = False
        self.exception = None

    def mark_network_start(self):
        self.network_start_time = time.time()

    def mark_network_end(self):
        self.network_end_time = time.time()

    def complete(self):
        if self.network_end_time is None:
            self.network_end_time = time.time()

    @property
    def processing_time(self) -> float:
        if self.network_start_time is None or self.network_end_time is None:
            return 0.0
        return self.network_end_time - self.network_start_time

    @property
    def total_retries(self) -> int:
        return sum(self.retry_counts.values())

    @property
    def average_response_time(self) -> float:
        if not self.validator_response_times:
            return 0
        return sum(self.validator_response_times.values()) / len(self.validator_response_times)

    def to_summary(self, miner_hotkey: str) -> Dict[str, Any]:
        return {
            "signal_uuid": self.signal_uuid,
            "trade_pair_id": self.trade_pair_id,
            "miner_hotkey": miner_hotkey,
            "validators_attempted": self.validators_attempted,
            "validators_succeeded": self.validators_succeeded,
            "high_trust_total": self.high_trust_total,
            "high_trust_succeeded": self.high_trust_succeeded,
            "all_high_trust_succeeded": self.all_high_trust_succeeded,
            "total_retries": self.total_retries,
            "processing_time": self.processing_time,
            "average_response_time": self.average_response_time,
            "validator_response_times": self.validator_response_times,
            "validator_errors": dict(self.validator_errors),
            "exception": str(self.exception) if self.exception else None,
            "timestamp": datetime.now().isoformat()
        }


class PropNetOrderPlacer:
    # Constants for retry logic with exponential backoff
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY_SECONDS = 20
    # Thread pool configuration
    MAX_WORKERS = 10
    THREAD_POOL_TIMEOUT = 300  # 5 minutes

    def __init__(self, wallet, metagraph_client, config, is_testnet, position_inspector=None, slack_notifier=None, running_unit_tests=False):
        self.wallet = wallet
        self.metagraph_client = metagraph_client
        self.config = config
        self.running_unit_tests = running_unit_tests
        self.recently_acked_validators = []
        self.is_testnet = is_testnet
        self.trade_pair_id_to_last_order_send = {tp.trade_pair_id: 0 for tp in TradePair}
        self.used_miner_uuids = set()
        self.position_inspector = position_inspector
        self.slack_notifier = slack_notifier
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.MAX_WORKERS,
            thread_name_prefix="signal_sender"
        )
        self._shutdown = False
        self._active_futures = set()
        self._lock = threading.Lock()

    def shutdown(self):
        """Gracefully shutdown the thread pool"""
        self._shutdown = True
        self.executor.shutdown(wait=True, cancel_futures=True)

    def send_signals(self, signals, signal_file_names, recently_acked_validators: list[str]):
        """
        Initiates the process of sending signals to all validators using a thread pool.
        """
        if self._shutdown:
            bt.logging.warning("PropNetOrderPlacer is shutting down, not accepting new signals")
            return

        self.recently_acked_validators = recently_acked_validators

        # Submit tasks to thread pool
        futures = []
        with self._lock:
            for (signal_data, signal_file_path) in zip(signals, signal_file_names):
                if self._shutdown:
                    break

                # Create a wrapper to run async function in thread with proper closure
                def run_async_signal(file_path, data):
                    return asyncio.run(self._safe_process_signal(file_path, data))
                
                future = self.executor.submit(run_async_signal, signal_file_path, signal_data)
                futures.append(future)
                self._active_futures.add(future)

        # Monitor futures asynchronously
        monitor_thread = threading.Thread(
            target=self._monitor_futures,
            args=(futures,),
            daemon=True
        )
        monitor_thread.start()

    async def _safe_process_signal(self, signal_file_path, signal_data):
        """Wrapper for process_a_signal with error handling and metrics"""
        signal_uuid = signal_file_path.split('/')[-1]
        trade_pair_id = signal_data.get('trade_pair', {}).get('trade_pair_id', 'Unknown')
        metrics = SignalMetrics(signal_uuid, trade_pair_id)

        try:
            result = await self.process_a_signal(signal_file_path, signal_data, metrics)
            metrics.complete()

            # Send summary to Slack
            if self.slack_notifier:
                summary = metrics.to_summary(self.wallet.hotkey.ss58_address)
                self.slack_notifier.send_signal_summary(summary)

            return result

        except Exception as e:
            bt.logging.error(f"Error processing signal {signal_file_path}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

            metrics.exception = e
            metrics.complete()

            # Send error notification to Slack
            if self.slack_notifier:
                summary = metrics.to_summary(self.wallet.hotkey.ss58_address)
                self.slack_notifier.send_signal_summary(summary)

                # Send additional detailed error message
                error_details = (f"❌ Signal Processing Exception:\n"
                                 f"Signal UUID: {signal_uuid}\n"
                                 f"Trade Pair: {trade_pair_id}\n"
                                 f"Error: {str(e)}\n"
                                 f"Traceback:\n{traceback.format_exc()[:1000]}")
                self.slack_notifier.send_message(error_details, level="error")

            return None
        finally:
            # Clean up resources if needed
            pass

    def _monitor_futures(self, futures):
        """Monitor futures for completion and handle results"""
        try:
            for future in as_completed(futures, timeout=self.THREAD_POOL_TIMEOUT):
                with self._lock:
                    self._active_futures.discard(future)

                try:
                    result = future.result()
                    if result:
                        bt.logging.debug(f"Successfully processed signal: {result}")
                except Exception as e:
                    bt.logging.error(f"Future resulted in exception: {e}")
        except TimeoutError:
            bt.logging.error(f"Some signal processing tasks timed out after {self.THREAD_POOL_TIMEOUT} seconds")
            # Cancel timed-out futures
            for future in futures:
                if not future.done():
                    future.cancel()
                    with self._lock:
                        self._active_futures.discard(future)

    def get_active_tasks_count(self):
        """Get the number of currently active signal processing tasks"""
        with self._lock:
            return len(self._active_futures)
    

    async def process_a_signal(self, signal_file_path, signal_data, metrics: SignalMetrics):
        """
        Processes a signal file by attempting to send it to the validators.
        """
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph_client.get_neurons()}
        axons_to_try = self.position_inspector.get_possible_validators()
        axons_to_try.sort(key=lambda validator: hotkey_to_v_trust[validator.hotkey], reverse=True)

        # Update metrics
        metrics.validators_attempted = len(axons_to_try)

        validator_hotkey_to_axon = {}
        for axon in axons_to_try:
            assert axon.hotkey not in validator_hotkey_to_axon, f"Duplicate hotkey {axon.hotkey} in axons"
            validator_hotkey_to_axon[axon.hotkey] = axon

        retry_status = {
            'retry_attempts': 0,
            'retry_delay_seconds': self.INITIAL_RETRY_DELAY_SECONDS,
            'validators_needing_retry': axons_to_try,
            'validator_error_messages': {},
            'created_orders': {}
        }

        # Track the high-trust validators
        high_trust_validators = self.get_high_trust_validators(axons_to_try, hotkey_to_v_trust)
        metrics.high_trust_total = len(high_trust_validators)

        miner_order_uuid = signal_file_path.split('/')[-1]

        # Thread-safe UUID check
        with self._lock:
            is_cancel_order = signal_data.get("execution_type", "MARKET") == "LIMIT_CANCEL"
            if miner_order_uuid in self.used_miner_uuids and not is_cancel_order:
                bt.logging.warning(f"Duplicate miner order uuid {miner_order_uuid}, skipping")
                return None
            self.used_miner_uuids.add(miner_order_uuid)

        send_signal_request = SendSignal(signal=signal_data, miner_order_uuid=miner_order_uuid,
                                         repo_version=REPO_VERSION)

        # Continue retrying until max retries reached
        while retry_status['retry_attempts'] < self.MAX_RETRIES and retry_status['validators_needing_retry']:
            if self._shutdown:
                bt.logging.warning("Shutting down, abandoning signal processing")
                return None
            await self.attempt_to_send_signal(send_signal_request, retry_status, high_trust_validators,
                                        validator_hotkey_to_axon, metrics)

        # After retries, check if all high-trust validators have processed the signal successfully
        high_trust_processed = True
        n_high_trust_validators = len(high_trust_validators)
        n_high_trust_validators_that_failed = 0
        for validator in high_trust_validators:
            if validator in retry_status['validators_needing_retry']:
                high_trust_processed = False
                n_high_trust_validators_that_failed += 1

        if self.is_testnet and retry_status['validator_error_messages']:
            high_trust_processed = False

        # Update metrics
        metrics.high_trust_succeeded = n_high_trust_validators - n_high_trust_validators_that_failed
        metrics.all_high_trust_succeeded = high_trust_processed
        metrics.validators_succeeded = len(retry_status['created_orders'])

        # Copy validator errors to metrics
        for hotkey, errors in retry_status['validator_error_messages'].items():
            metrics.validator_errors[hotkey] = errors

        # Process results
        if high_trust_processed:
            self.write_signal_to_processed_directory(signal_data, signal_file_path, retry_status)
        elif self.config.write_failed_signal_logs:
            v_trust_floor = min([hotkey_to_v_trust[validator.hotkey] for validator in high_trust_validators])
            error_msg = (f"Signal file {signal_file_path} was not successfully processed by "
                         f"{n_high_trust_validators_that_failed}/{n_high_trust_validators} high-trust validators. "
                         f"(floor {v_trust_floor})")
            bt.logging.error(error_msg)
            self.write_signal_to_failure_directory(signal_data, signal_file_path, retry_status)
        else:
            self.write_signal_to_processed_directory(signal_data, signal_file_path, retry_status)

        return signal_file_path

    def get_high_trust_validators(self, axons, hotkey_to_v_trust):
        """Returns a list of high-trust validators."""
        high_trust_validators = [ax for ax in axons if
                                 hotkey_to_v_trust[ax.hotkey] >= MinerConfig.HIGH_V_TRUST_THRESHOLD]
        if not high_trust_validators:
            if not self.is_testnet:
                bt.logging.error(
                    "No high-trust validators found. This is unexpected in mainnet. Please report to the team ASAP.")
            return axons
        else:
            return high_trust_validators


    async def attempt_to_send_signal(self, send_signal_request: SendSignal, retry_status: dict,
                               high_trust_validators: list, validator_hotkey_to_axon: dict,
                               metrics: SignalMetrics):
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph_client.get_neurons()}

        bt.logging.info(
            f"Attempt #{retry_status['retry_attempts']} for {send_signal_request.signal['trade_pair']['trade_pair_id']} "
            f"uuid {send_signal_request.miner_order_uuid}. "
            f"Sending order to {len(retry_status['validators_needing_retry'])} hotkeys...")

        if retry_status['retry_attempts'] != 0:
            await asyncio.sleep(retry_status['retry_delay_seconds'])
            retry_status['retry_delay_seconds'] *= 2

        # In test mode, skip network calls and create mock successful responses
        if self.running_unit_tests:
            from bittensor.core.synapse import TerminalInfo

            metrics.mark_network_start()
            # Create mock successful responses for all validators in test mode
            validator_responses = []
            for axon in retry_status['validators_needing_retry']:
                mock_response = SendSignal(signal=send_signal_request.signal, miner_order_uuid=send_signal_request.miner_order_uuid)
                mock_response.successfully_processed = True
                mock_response.validator_hotkey = axon.hotkey
                mock_response.order_json = json.dumps({"test": "order"})  # Must be JSON string, not dict
                mock_response.error_message = ""  # Empty string, not None
                # Mock process times with proper TerminalInfo instances
                mock_response.axon = TerminalInfo(process_time=0.001)
                mock_response.dendrite = TerminalInfo(process_time=0.001)
                validator_responses.append(mock_response)
            metrics.mark_network_end()
        else:
            # Production mode: make actual network calls
            # Use async context manager for automatic cleanup
            async with bt.dendrite(wallet=self.wallet) as dendrite:
                metrics.mark_network_start()
                validator_responses: list[Synapse] = await dendrite.aquery(retry_status['validators_needing_retry'], send_signal_request)
                metrics.mark_network_end()

        all_high_trust_validators_succeeded = True
        success_validators = set()

        for response in validator_responses:
            if response.successfully_processed and response.validator_hotkey:
                success_validators.add(response.validator_hotkey)
                retry_status['created_orders'][response.validator_hotkey] = response.order_json
                metrics.retry_counts[response.validator_hotkey] = retry_status['retry_attempts'] + 1

                # Extract true one way network time from response axon if available
                process_time_axon_ms = int(1000 * response.axon.process_time)
                process_time_dendrite_ms = int(1000 * response.dendrite.process_time)
                print(f"Process time axon: {process_time_axon_ms}ms, dendrite: {process_time_dendrite_ms}ms")
                metrics.validator_response_times[response.validator_hotkey] = process_time_axon_ms
            else:
                acked_axon = validator_hotkey_to_axon.get(response.validator_hotkey)
                if acked_axon and acked_axon in high_trust_validators:
                    all_high_trust_validators_succeeded = False

                if response.error_message:
                    vtrust = hotkey_to_v_trust.get(response.validator_hotkey)
                    msg = f"Error sending to {response.validator_hotkey} (v_trust: {vtrust}): {response.error_message}"
                    bt.logging.warning(msg)

                    if response.validator_hotkey not in retry_status['validator_error_messages']:
                        retry_status['validator_error_messages'][response.validator_hotkey] = []
                    retry_status['validator_error_messages'][response.validator_hotkey].append(response.error_message)
                    metrics.validator_errors[response.validator_hotkey].append(response.error_message)

        if all_high_trust_validators_succeeded:
            v_trust_floor = min([hotkey_to_v_trust[validator.hotkey] for validator in high_trust_validators])
            n_high_trust = len(high_trust_validators)
            bt.logging.success(f"Signal processed by {n_high_trust}/{n_high_trust} high-trust validators "
                               f"(min v_trust: {v_trust_floor})")

        def _allow_retry(axon):
            if axon.hotkey in success_validators:
                return False
            if axon.hotkey in self.recently_acked_validators:
                return True
            return hotkey_to_v_trust[axon.hotkey] > 0

        new_validators_to_retry = [axon for axon in retry_status['validators_needing_retry'] if _allow_retry(axon)]
        new_validators_to_retry.sort(key=lambda validator: hotkey_to_v_trust[validator.hotkey], reverse=True)

        retry_status['validators_needing_retry'] = new_validators_to_retry
        retry_status['retry_attempts'] += 1

    def write_signal_to_processed_directory(self, signal_data, signal_file_path: str, retry_status: dict):
        """Moves a processed signal file to the processed directory."""
        signal_copy = signal_data.copy()
        signal_copy['trade_pair'] = signal_copy['trade_pair']['trade_pair_id']
        data_to_write = {
            'signal_data': signal_copy,
            'created_orders': retry_status['created_orders'],
            'processing_timestamp': datetime.now().isoformat(),
            'retry_attempts': retry_status['retry_attempts']
        }
        self.write_signal_to_directory(MinerConfig.get_miner_processed_signals_dir(), signal_file_path, data_to_write,
                                       True)

    def write_signal_to_failure_directory(self, signal_data, signal_file_path: str, retry_status: dict):
        """Writes failed signal with detailed failure information"""
        validators_needing_retry = retry_status['validators_needing_retry']
        error_messages_dict = retry_status['validator_error_messages']
        created_orders = retry_status['created_orders']

        # Prepare validator data
        json_validator_data = [{
            'ip': validator.ip,
            'port': validator.port,
            'ip_type': validator.ip_type,
            'hotkey': validator.hotkey,
            'coldkey': validator.coldkey,
            'protocol': validator.protocol
        } for validator in validators_needing_retry]

        new_data = {
            'original_signal': signal_data,
            'validators_needing_retry': json_validator_data,
            'error_messages_dict': error_messages_dict,
            'created_orders': created_orders,
            'failure_timestamp': datetime.now().isoformat(),
            'retry_attempts': retry_status['retry_attempts']
        }

        # Move signal file to the failed directory
        self.write_signal_to_directory(MinerConfig.get_miner_failed_signals_dir(), signal_file_path, signal_data, False)

        # Overwrite with detailed failure data
        new_file_path = os.path.join(MinerConfig.get_miner_failed_signals_dir(), os.path.basename(signal_file_path))
        ValiBkpUtils.write_file(new_file_path, json.dumps(new_data))

        bt.logging.info(f"Signal file modified with failure info: {new_file_path}")

    def write_signal_to_directory(self, directory: str, signal_file_path, signal_data, success):
        """Write signal to specified directory"""
        ValiBkpUtils.make_dir(directory)
        new_path = os.path.join(directory, os.path.basename(signal_file_path))
        with open(new_path, 'w') as f:
            f.write(json.dumps(signal_data))

        msg = f"Signal file moved to {new_path}"
        if success:
            bt.logging.success(msg)
        else:
            bt.logging.error(msg)