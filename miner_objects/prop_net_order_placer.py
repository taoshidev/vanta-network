# The MIT License (MIT)
# Copyright (c) 2024 Yuma Rao
# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
import asyncio
import json
import os
import threading
import time

from datetime import datetime
from typing import Dict, Optional

from miner_config import MinerConfig
from shared_objects.http_client import HttpAuthClient
from vanta_protocol import SendSignalRequest, SendSignalResponse
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.order_signal import Signal
from shared_objects.log import logger

REPO_VERSION = 'unknown'
with open(ValiBkpUtils.get_meta_json_path(), 'r') as f:
    REPO_VERSION = json.loads(f.read()).get("subnet_version", "unknown")

CONNECTION_ERROR_MSG = "Failed to connect to Vanta Network, please try again soon"


class PropNetOrderPlacer:
    """Sends miner orders to validators. Signals arrive exclusively via the REST server
    (process_a_signal_for_rest) — file-based signal submission is no longer supported;
    every miner integration goes through the REST API."""

    # Constants for network retry logic (only retries on connection failures to mothership)
    MAX_NETWORK_RETRIES = 3
    NETWORK_RETRY_DELAY_SECONDS = 5
    ELIMINATION_CACHE_TTL_SECONDS = 600
    NO_TARGETS_WARNING_INTERVAL_S = 600

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
        self._lock = threading.Lock()
        self._elimination_cache: Dict[int, float] = {}
        self._last_no_targets_warning_s = 0.0

    def _is_subaccount_eliminated(self, subaccount_id: int) -> bool:
        expiry = self._elimination_cache.get(subaccount_id)
        if expiry is None:
            return False
        if time.time() >= expiry:
            self._elimination_cache.pop(subaccount_id, None)
            return False
        return True

    def _mark_subaccount_eliminated(self, subaccount_id: int) -> None:
        self._elimination_cache[subaccount_id] = time.time() + self.ELIMINATION_CACHE_TTL_SECONDS

    def _get_mothership_and_other_axons(self):
        """The mothership's axon, found by its hotkey, and the axons of the other validators by the stake rule (every
        served axon on testnet), with no validator-trust filter: under Null consensus only the weight setter has any."""
        mothership_hotkey = ValiConfig.MOTHERSHIP_HOTKEY_TESTNET if self.is_testnet else ValiConfig.MOTHERSHIP_HOTKEY
        mothership_axon = None
        other_axons = []
        for axon in self.position_inspector.get_possible_validators():
            if axon.hotkey == mothership_hotkey:
                mothership_axon = axon
            else:
                other_axons.append(axon)
        if mothership_axon is None:
            mothership_axon = next((n.axon_info for n in self.metagraph_client.get_neurons()
                                    if n.hotkey == mothership_hotkey and n.axon_info.ip != MinerConfig.AXON_NO_IP), None)
        if not other_axons:
            self._warn_no_other_validators()
        return mothership_axon, other_axons

    def _warn_no_other_validators(self):
        """Orders reach the mothership alone: the stake list is still empty after a failed metagraph sync, or no other
        validator serves an IP. At most once per NO_TARGETS_WARNING_INTERVAL_S."""
        now = time.time()
        if now - self._last_no_targets_warning_s >= self.NO_TARGETS_WARNING_INTERVAL_S:
            self._last_no_targets_warning_s = now
            logger.warning("No validator besides the mothership to send orders to: the metagraph's stake list may be "
                           "empty after a failed sync, or no other validator serves an IP")

    async def _send_order(self, send_signal_request: SendSignalRequest, mothership_axon, other_axons) -> dict:
        """
        Send order to mothership (await response) and other validators (fire-and-forget).
        Only retries on network connection failures or transient errors from mothership.

        Returns: {success: bool, order_json: str, error_message: str}
        """
        if self.running_unit_tests:
            return self._send_order_test_mode(send_signal_request)

        async with HttpAuthClient(self.wallet) as client:
            for attempt in range(self.MAX_NETWORK_RETRIES):
                try:
                    # Fire-and-forget to other validators on first attempt only
                    if attempt == 0 and other_axons:
                        thread = threading.Thread(
                            target=self._query_validators_sync,
                            args=(other_axons, send_signal_request),
                            daemon=True
                        )
                        thread.start()

                    # Await mothership response only
                    responses = await client.aquery([mothership_axon], "SendSignal", send_signal_request, SendSignalResponse)
                    response = responses[0]

                    if response.successfully_processed:
                        return {"success": True, "order_json": response.order_json, "error_message": ""}

                    # Mothership rejected -- check should_retry
                    if not response.should_retry:
                        return {"success": False, "order_json": "", "error_message": response.error_message}

                    # should_retry=True but failed -- transient issue, retry with delay
                    logger.warning(f"Mothership retryable error (attempt {attempt + 1}): {response.error_message}")
                    if attempt < self.MAX_NETWORK_RETRIES - 1:
                        await asyncio.sleep(self.NETWORK_RETRY_DELAY_SECONDS)

                except Exception as e:
                    logger.warning(f"Network error to mothership (attempt {attempt + 1}/{self.MAX_NETWORK_RETRIES}): {e}")
                    if attempt < self.MAX_NETWORK_RETRIES - 1:
                        await asyncio.sleep(self.NETWORK_RETRY_DELAY_SECONDS)

            return {"success": False, "order_json": "", "error_message": CONNECTION_ERROR_MSG}

    def _send_order_test_mode(self, send_signal_request: SendSignalRequest) -> dict:
        """Mock successful response for unit tests."""
        return {"success": True, "order_json": json.dumps({"test": "order"}), "error_message": ""}

    def _check_and_mark_uuid(self, order_uuid: str, execution_type: str) -> bool:
        """Thread-safe duplicate-order-uuid check. Returns True (and marks order_uuid
        used) if it's fresh; False if it's a disallowed duplicate (LIMIT_CANCEL/
        LIMIT_EDIT/FLAT_ALL may reuse one)."""
        is_uuid_reuse_allowed = execution_type in ("LIMIT_CANCEL", "LIMIT_EDIT", "FLAT_ALL")
        with self._lock:
            if order_uuid in self.used_miner_uuids and not is_uuid_reuse_allowed:
                return False
            self.used_miner_uuids.add(order_uuid)
            return True

    async def _send_signal(self, signal_data: dict, order_uuid: str, subaccount_id: Optional[int],
                            mothership_axon, other_axons) -> dict:
        """Builds the SendSignalRequest and sends it (awaited mothership response +
        fire-and-forget broadcast to other validators)."""
        send_signal_request = SendSignalRequest(
            signal=signal_data,
            miner_order_uuid=order_uuid,
            repo_version=REPO_VERSION,
            subaccount_id=subaccount_id,
        )
        return await self._send_order(send_signal_request, mothership_axon, other_axons)

    def process_a_signal_for_rest(self, order_uuid: str, signal: Signal, subaccount_id: Optional[int] = None) -> dict:
        """
        Process signal from REST API endpoint and send to validators.

        This is the entry point for signal processing — all miner integrations submit
        orders through the REST API. It handles the complete workflow: validator lookup,
        UUID deduplication, async order transmission, and result archiving.

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
            if subaccount_id is not None and self._is_subaccount_eliminated(subaccount_id):
                error_msg = "Subaccount is eliminated"
                return {
                    "success": False,
                    "order_uuid": order_uuid,
                    "order_json": None,
                    "error_message": error_msg,
                    "processing_time": time.time() - start_time,
                    "message": f"Order failed on Vanta Network: {error_msg}"
                }

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

            execution_type = str(signal.execution_type)
            if not self._check_and_mark_uuid(order_uuid, execution_type):
                logger.warning(f"Duplicate miner order uuid {order_uuid}, skipping")
                return {
                    "success": False,
                    "order_uuid": order_uuid,
                    "order_json": None,
                    "error_message": "Duplicate order UUID",
                    "processing_time": time.time() - start_time,
                    "message": "Order failed: duplicate order uuid"
                }

            # Convert Signal object to dict (mode='json' ensures enums are serialized)
            signal_data = signal.model_dump(mode='json', exclude_none=True)

            result = asyncio.run(self._send_signal(signal_data, order_uuid, subaccount_id,
                                                     mothership_axon, other_axons))

            processing_time = time.time() - start_time

            # Archive result
            fake_signal_file_path = f"/rest-api/{order_uuid}"
            if result["success"]:
                self.write_signal_to_processed_directory(signal_data, fake_signal_file_path, result["order_json"])
                message = "Order successfully processed by Vanta Network"
            else:
                if subaccount_id is not None and "eliminated" in (result["error_message"] or "").lower():
                    self._mark_subaccount_eliminated(subaccount_id)
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
            logger.error(f"Error processing REST order {order_uuid}: {e}")

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

    def _query_validators_sync(self, axons, send_signal_request: SendSignalRequest):
        """Fire-and-forget query to validators (runs in separate thread)."""

        async def _query_validators_async(axons, send_signal_request):
            """Async helper for background validator queries."""
            async with HttpAuthClient(self.wallet) as client:
                await client.aquery(axons, "SendSignal", send_signal_request, SendSignalResponse)

        try:
            asyncio.run(_query_validators_async(axons, send_signal_request))
        except Exception as e:
            logger.debug(f"Background validator query error (non-critical): {e}")

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

        logger.info(f"Signal file modified with failure info: {new_file_path}")

    def write_signal_to_directory(self, directory: str, signal_file_path, signal_data, success):
        """Write signal to specified directory"""
        ValiBkpUtils.make_dir(directory)
        new_path = os.path.join(directory, os.path.basename(signal_file_path))
        with open(new_path, 'w') as f:
            f.write(json.dumps(signal_data))

        msg = f"Signal file moved to {new_path}"
        if success:
            logger.info(msg)
        else:
            logger.error(msg)
