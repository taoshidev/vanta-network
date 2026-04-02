# The MIT License (MIT)
# Copyright (c) 2024 Yuma Rao
# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
import asyncio
import json
import os
import threading
import time

import bittensor as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

from miner_config import MinerConfig
from template.protocol import SendSignal
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order_signal import Signal

REPO_VERSION = 'unknown'
with open(ValiBkpUtils.get_meta_json_path(), 'r') as f:
    REPO_VERSION = json.loads(f.read()).get("subnet_version", "unknown")

CONNECTION_ERROR_MSG = "Failed to connect to Vanta Network, please try again soon"

# DEPRECATED: No longer used by simplified retry logic
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
    # Constants for network retry logic (only retries on connection failures to mothership)
    MAX_NETWORK_RETRIES = 3
    NETWORK_RETRY_DELAY_SECONDS = 5
    # DEPRECATED: Thread pool used by file-based signal processing. Use REST server instead.
    MAX_WORKERS = 10
    THREAD_POOL_TIMEOUT = 300  # 5 minutes

    def __init__(self, wallet, metagraph_client, config, is_testnet, position_inspector=None, slack_notifier=None, running_unit_tests=False):
        self.wallet = wallet
        self.metagraph_client = metagraph_client
        self.config = config
        self.running_unit_tests = running_unit_tests
        self.is_testnet = is_testnet
        self.trade_pair_id_to_last_order_send = {tp.trade_pair_id: 0 for tp in TradePair}
        self.used_miner_uuids = set()
        self.position_inspector = position_inspector
        self.slack_notifier = slack_notifier
        # DEPRECATED: Thread pool for file-based signal processing. Use REST server instead.
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

    def send_signals(self, signals, signal_file_names):
        """
        DEPRECATED: File-based signal processing. Use REST server (process_a_signal_for_rest) instead.
        """
        if self._shutdown:
            bt.logging.warning("PropNetOrderPlacer is shutting down, not accepting new signals")
            return

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
        """Wrapper for process_a_signal with error handling and Slack notifications"""
        signal_uuid = signal_file_path.split('/')[-1]
        # Support both object {"trade_pair_id": "BTCUSD"} and string "BTCUSD"
        trade_pair = signal_data.get('trade_pair', 'Unknown')
        if isinstance(trade_pair, dict):
            trade_pair_id = trade_pair.get('trade_pair_id', 'Unknown')
        elif isinstance(trade_pair, str):
            trade_pair_id = trade_pair
        else:
            trade_pair_id = 'Unknown'

        try:
            result = await self.process_a_signal(signal_file_path, signal_data)

            # Send summary to Slack
            if self.slack_notifier and result:
                summary = {
                    "signal_uuid": signal_uuid,
                    "trade_pair_id": trade_pair_id,
                    "miner_hotkey": self.wallet.hotkey.ss58_address,
                    "validators_attempted": 1,
                    "validators_succeeded": 1 if result.get("success") else 0,
                    "all_high_trust_succeeded": result.get("success", False),
                    "average_response_time": 0,
                    "validator_errors": {},
                    "exception": None,
                    "timestamp": datetime.now().isoformat()
                }
                self.slack_notifier.send_signal_summary(summary)

            return result

        except Exception as e:
            bt.logging.error(f"Error processing signal {signal_file_path}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

            # Send error notification to Slack
            if self.slack_notifier:
                summary = {
                    "signal_uuid": signal_uuid,
                    "trade_pair_id": trade_pair_id,
                    "miner_hotkey": self.wallet.hotkey.ss58_address,
                    "validators_attempted": 1,
                    "validators_succeeded": 0,
                    "all_high_trust_succeeded": False,
                    "average_response_time": 0,
                    "validator_errors": {},
                    "exception": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.slack_notifier.send_signal_summary(summary)

                # Send additional detailed error message
                error_details = (f"Signal Processing Exception:\n"
                                 f"Signal UUID: {signal_uuid}\n"
                                 f"Trade Pair: {trade_pair_id}\n"
                                 f"Error: {str(e)}\n"
                                 f"Traceback:\n{traceback.format_exc()[:1000]}")
                self.slack_notifier.send_message(error_details, level="error")

            return None

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

    def _get_mothership_and_other_axons(self):
        """Get mothership axon and other validator axons (filtered by v_trust)."""
        mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY_TESTNET if self.is_testnet else ValiConfig.MOTHERSHIP_HOTKEY
        axons = self.position_inspector.get_possible_validators()
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self.metagraph_client.get_neurons()}
        mothership_axon = None
        other_axons = []
        for axon in axons:
            if axon.hotkey == mothership_hotkey:
                mothership_axon = axon
            elif hotkey_to_v_trust.get(axon.hotkey, 0) >= MinerConfig.HIGH_V_TRUST_THRESHOLD:
                other_axons.append(axon)

        # bt.logging.info(f"Validator hotkey -> v_trust: {hotkey_to_v_trust}")
        # bt.logging.info(f"Mothership validator: {mothership_axon.hotkey if mothership_axon else 'None'}")
        # bt.logging.info(f"Other validators ({len(other_axons)}): {[a.hotkey for a in other_axons]}")
        return mothership_axon, other_axons

    async def _send_order(self, synapse, mothership_axon, other_axons) -> dict:
        """
        Send order to mothership (await response) and other validators (fire-and-forget).
        Only retries on network connection failures or transient errors from mothership.

        Returns: {success: bool, order_json: str, error_message: str}
        """
        if self.running_unit_tests:
            return self._send_order_test_mode(synapse)

        dendrite = bt.dendrite(wallet=self.wallet)
        try:
            for attempt in range(self.MAX_NETWORK_RETRIES):
                try:
                    # Fire-and-forget to other validators on first attempt only
                    if attempt == 0 and other_axons:
                        thread = threading.Thread(
                            target=self._query_validators_sync,
                            args=(other_axons, synapse),
                            daemon=True
                        )
                        thread.start()

                    # Await mothership response only
                    responses = await dendrite.aquery([mothership_axon], synapse)
                    response = responses[0]

                    if response.successfully_processed:
                        return {"success": True, "order_json": response.order_json, "error_message": ""}

                    # Mothership rejected -- check should_retry
                    if not response.should_retry:
                        return {"success": False, "order_json": "", "error_message": response.error_message}

                    # should_retry=True but failed -- transient issue, retry with delay
                    bt.logging.warning(f"Mothership retryable error (attempt {attempt + 1}): {response.error_message}")
                    if attempt < self.MAX_NETWORK_RETRIES - 1:
                        await asyncio.sleep(self.NETWORK_RETRY_DELAY_SECONDS)

                except Exception as e:
                    bt.logging.warning(f"Network error to mothership (attempt {attempt + 1}/{self.MAX_NETWORK_RETRIES}): {e}")
                    if attempt < self.MAX_NETWORK_RETRIES - 1:
                        await asyncio.sleep(self.NETWORK_RETRY_DELAY_SECONDS)

            return {"success": False, "order_json": "", "error_message": CONNECTION_ERROR_MSG}
        finally:
            await dendrite.aclose_session()

    def _send_order_test_mode(self, synapse) -> dict:
        """Mock successful response for unit tests."""
        return {"success": True, "order_json": json.dumps({"test": "order"}), "error_message": ""}

    async def process_a_signal(self, signal_file_path, signal_data):
        """
        Processes a signal file by sending it to the mothership validator.
        Other validators receive fire-and-forget copies.
        """
        mothership_axon, other_axons = self._get_mothership_and_other_axons()

        if not mothership_axon and not self.running_unit_tests:
            error_msg = CONNECTION_ERROR_MSG
            bt.logging.error(error_msg)
            if self.config.write_failed_signal_logs:
                self.write_signal_to_failure_directory(signal_data, signal_file_path, error_msg)
            return {"success": False, "order_json": "", "error_message": error_msg}

        miner_order_uuid = signal_file_path.split('/')[-1]

        # Thread-safe UUID check
        with self._lock:
            execution_type = signal_data.get("execution_type", "MARKET")
            is_uuid_reuse_allowed = execution_type in ("LIMIT_CANCEL", "LIMIT_EDIT", "FLAT_ALL")
            if miner_order_uuid in self.used_miner_uuids and not is_uuid_reuse_allowed:
                bt.logging.warning(f"Duplicate miner order uuid {miner_order_uuid}, skipping")
                return None
            self.used_miner_uuids.add(miner_order_uuid)

        send_signal_request = SendSignal(
            signal=signal_data,
            miner_order_uuid=miner_order_uuid,
            repo_version=REPO_VERSION,
            subaccount_id=signal_data.get('subaccount_id'),
            successfully_processed=False,
            error_message="",
            should_retry=True,
            validator_hotkey="",
            order_json=""
        )

        result = await self._send_order(send_signal_request, mothership_axon, other_axons)

        # Archive result
        if result["success"]:
            self.write_signal_to_processed_directory(signal_data, signal_file_path, result["order_json"])
        elif self.config.write_failed_signal_logs:
            bt.logging.error(f"Signal file {signal_file_path} failed: {result['error_message']}")
            self.write_signal_to_failure_directory(signal_data, signal_file_path, result["error_message"])
        else:
            self.write_signal_to_processed_directory(signal_data, signal_file_path, result["order_json"])

        return result

    def process_a_signal_for_rest(self, order_uuid: str, signal: Signal, subaccount_id: Optional[int] = None) -> dict:
        """
        Process signal from REST API endpoint and send to validators.

        This is the main entry point for REST API signal processing. It handles the complete
        workflow: validator lookup, UUID deduplication, async order transmission, and result archiving.


        Args:
            order_uuid: UUID for this order
            signal: Validated Signal object with trade_pair, order_type, leverage, etc.
            subaccount_id: Optional subaccount ID for entity miners

        Returns:
            Dictionary with structure:
            {
                "success": bool,
                "order_uuid": str,
                "order_json": str or None,
                "error_message": str,
                "processing_time": float,
                "message": str
            }
        """
        start_time = time.time()

        try:
            mothership_axon, other_axons = self._get_mothership_and_other_axons()

            if not mothership_axon and not self.running_unit_tests:
                return {
                    "success": False,
                    "order_uuid": order_uuid,
                    "order_json": None,
                    "error_message": CONNECTION_ERROR_MSG,
                    "processing_time": time.time() - start_time,
                    "message": CONNECTION_ERROR_MSG
                }

            # Thread-safe UUID check
            with self._lock:
                execution_type = str(signal.execution_type)
                is_uuid_reuse_allowed = execution_type in ("LIMIT_CANCEL", "LIMIT_EDIT", "FLAT_ALL")
                if order_uuid in self.used_miner_uuids and not is_uuid_reuse_allowed:
                    bt.logging.warning(f"Duplicate miner order uuid {order_uuid}, skipping")
                    return {
                        "success": False,
                        "order_uuid": order_uuid,
                        "order_json": None,
                        "error_message": "Duplicate order UUID",
                        "processing_time": time.time() - start_time,
                        "message": "Order failed: duplicate order uuid"
                    }
                self.used_miner_uuids.add(order_uuid)

            # Convert Signal object to dict (mode='json' ensures enums are serialized)
            signal_data = signal.model_dump(mode='json')

            send_signal_request = SendSignal(
                signal=signal_data,
                miner_order_uuid=order_uuid,
                repo_version=REPO_VERSION,
                subaccount_id=subaccount_id,
                successfully_processed=False,
                error_message="",
                should_retry=True,
                validator_hotkey="",
                order_json=""
            )

            result = asyncio.run(self._send_order(send_signal_request, mothership_axon, other_axons))

            processing_time = time.time() - start_time

            # Archive result
            fake_signal_file_path = f"/rest-api/{order_uuid}"
            if result["success"]:
                self.write_signal_to_processed_directory(signal_data, fake_signal_file_path, result["order_json"])
                message = "Order successfully processed by Vanta Network"
            else:
                self.write_signal_to_failure_directory(signal_data, fake_signal_file_path, result["error_message"])
                message = f"Order failed on Vanta Network: {result['error_message']}"

            return {
                "success": result["success"],
                "order_uuid": order_uuid,
                "order_json": result["order_json"] or None,
                "error_message": result["error_message"],
                "processing_time": processing_time,
                "message": message
            }

        except Exception as e:
            bt.logging.error(f"Error processing REST order {order_uuid}: {e}")

            # Archive to failed_signals/ for audit trail
            fake_signal_file_path = f"/rest-api/{order_uuid}"
            if self.config.write_failed_signal_logs:
                try:
                    signal_data = signal.model_dump(mode='json')
                    self.write_signal_to_failure_directory(signal_data, fake_signal_file_path, str(e))
                except Exception:
                    pass

            return {
                "success": False,
                "order_uuid": order_uuid,
                "order_json": None,
                "error_message": str(e),
                "processing_time": time.time() - start_time,
                "message": f"Internal error: {str(e)}"
            }

    def _query_validators_sync(self, axons, send_signal_request):
        """Fire-and-forget query to validators (runs in separate thread)."""

        async def _query_validators_async(axons, send_signal_request):
            """Async helper for background validator queries."""
            dendrite = bt.dendrite(wallet=self.wallet)
            try:
                await dendrite.aquery(axons, send_signal_request)
            finally:
                await dendrite.aclose_session()

        try:
            asyncio.run(_query_validators_async(axons, send_signal_request))
        except Exception as e:
            bt.logging.debug(f"Background validator query error (non-critical): {e}")


    def write_signal_to_processed_directory(self, signal_data, signal_file_path: str, order_json: str):
        """Moves a processed signal file to the processed directory."""
        signal_copy = signal_data.copy()
        # trade_pair may not exist for LIMIT_CANCEL orders
        if 'trade_pair' in signal_copy and isinstance(signal_copy['trade_pair'], dict):
            signal_copy['trade_pair'] = signal_copy['trade_pair']['trade_pair_id']
        data_to_write = {
            'signal_data': signal_copy,
            'order_json': order_json,
            'processing_timestamp': datetime.now().isoformat(),
        }
        self.write_signal_to_directory(MinerConfig.get_miner_processed_signals_dir(), signal_file_path, data_to_write,
                                       True)

    def write_signal_to_failure_directory(self, signal_data, signal_file_path: str, error_message: str):
        """Writes failed signal with error information"""
        new_data = {
            'original_signal': signal_data,
            'error_message': error_message,
            'failure_timestamp': datetime.now().isoformat(),
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