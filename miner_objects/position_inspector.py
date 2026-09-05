from collections import defaultdict

import bittensor as bt
import time
import asyncio
import json
import shutil
import threading
import os

from miner_config import MinerConfig
from template.protocol import GetPositions
from shared_objects.log import logger


class PositionInspector:
    MAX_RETRIES = 1
    INITIAL_RETRY_DELAY = 3  # seconds
    UPDATE_INTERVAL_S = 5 * 60  # 5 minutes

    def __init__(self, wallet, metagraph_client, config, running_unit_tests=False):
        self.wallet = wallet
        self._metagraph_client = metagraph_client
        self.config = config
        self.running_unit_tests = running_unit_tests
        self.last_update_time = 0
        self.recently_acked_validators = []
        self.stop_requested = False  # Flag to control the loop
        assert self.config.netuid in (8, 116, 171), "Taoshi runs on netuid 8 (mainnet), 116/171 (testnet)"
        self.is_testnet = self.config.netuid in (116, 171)


    async def run_update_loop(self):
        while not self.stop_requested:
            try:
                await self.log_validator_positions()
            except Exception as e:
                # Handle exceptions or log errors
                logger.error(f"Error during position inspector update: {e}. Please alert a team member ASAP!")
            await asyncio.sleep(1)  # Don't busy loop
    
    def run_update_loop_sync(self):
        """Synchronous wrapper for threading compatibility"""
        asyncio.run(self.run_update_loop())

    def stop_update_loop(self):
        self.stop_requested = True

    def get_recently_acked_validators(self):
        return self.recently_acked_validators

    def get_possible_validators(self):
        return [n.axon_info for n in self._metagraph_client.get_neurons()
                if n.validator_permit and n.axon_info.ip != MinerConfig.AXON_NO_IP]

    async def query_positions(self, validators, hotkey_to_positions):
        # In test mode, skip network calls
        if self.running_unit_tests:
            return []

        remaining_validators_to_query = [v for v in validators if v.hotkey not in hotkey_to_positions]

        # bt.Dendrite removed in bt11; position inspector falls back to REST API calls
        logger.warning(
            "Position inspector: bt.Dendrite not available in bt11. "
            "Validator axon queries are not supported; returning empty responses."
        )
        responses = [GetPositions() for _ in remaining_validators_to_query]

        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self._metagraph_client.get_neurons()}
        ret = []
        for validator, response in zip(remaining_validators_to_query, responses):
            v_trust = hotkey_to_v_trust.get(validator.hotkey, 0)
            if response.error_message and v_trust >= MinerConfig.HIGH_V_TRUST_THRESHOLD:
                logger.warning(f"Error getting positions from {validator}. v_trust {v_trust} Error message: {response.error_message}")
            if response.successfully_processed:
                ret.append((validator, response.positions))

        return ret

    def reconcile_validator_positions(self, hotkey_to_positions, validators):
        hotkey_to_validator = {v.hotkey: v for v in validators}
        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in self._metagraph_client.get_neurons()}
        orders_count = defaultdict(int)
        max_order_count = 0
        corresponding_positions = []
        corresponding_hotkey = None
        for hotkey, positions in hotkey_to_positions.items():
            hotkey_total_portfolio_leverage = 0
            for position in positions:
                orders_count[hotkey] += len(position['orders'])
                hotkey_total_portfolio_leverage += abs(position['net_leverage'])

            if hotkey_total_portfolio_leverage >= 10:
                logger.warning(
                    f"Validator {hotkey} has a total portfolio leverage of {hotkey_total_portfolio_leverage}. "
                    f"High leverage on crypto trade pairs comes with high fees which greatly increase your draw down.")

            if orders_count[hotkey] > max_order_count:
                max_order_count = orders_count[hotkey]
                corresponding_positions = positions
                corresponding_hotkey = hotkey


        unique_counts = set(orders_count.values())

        if len(unique_counts) > 1:
            logger.warning("Spilling hotkey to positions:")
            for hotkey, count in orders_count.items():
                axon_info = hotkey_to_validator[hotkey]
                logger.warning(f"Validator {hotkey} has {count} orders with v_trust {hotkey_to_v_trust[hotkey]}, axon: {axon_info}. "
                                   f"Validators may be mis-synced.")
                for i, position in enumerate(hotkey_to_positions[hotkey]):
                    logger.warning(f"Position {i}: {position}")


        # Return the position in hotkey_to_positions that has the most orders
        logger.info(f"Validator with the most orders: {corresponding_hotkey}, n_orders: {max_order_count}, v_trust:"
                        f" {hotkey_to_v_trust.get(corresponding_hotkey, 0)}. Corresponding positions: {corresponding_positions}")
        return corresponding_positions

    async def get_positions_with_retry(self, validators_to_query):
        attempts = 0
        delay = self.INITIAL_RETRY_DELAY
        hotkey_to_positions = {}
        while attempts < self.MAX_RETRIES and len(hotkey_to_positions) != len(validators_to_query):
            if attempts > 0:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

            positions = await self.query_positions(validators_to_query, hotkey_to_positions)
            for p in positions:
                hotkey_to_positions[p[0].hotkey] = p[1]

            attempts += 1

        logger.info(f"Got positions from {len(hotkey_to_positions)} possible validators")
        # We consider a validator acked if it successfully responded to the signal.
        # Note, a validator that has this miner blacklisted will not be added.
        self.recently_acked_validators = hotkey_to_positions.keys()
        position_most_orders = self.reconcile_validator_positions(hotkey_to_positions, validators_to_query)
        # Return the validator with the most orders
        return position_most_orders

    def refresh_allowed(self):
        return (time.time() - self.last_update_time) > self.UPDATE_INTERVAL_S

    async def log_validator_positions(self):
        """
        Sends signals to the validators to get their time-sorted positions for this miner.
        This method may be used directly in your own logic to attempt to "fix" validator positions.
        Note: The rate limiter on validators will prevent repeated calls from succeeding if they are too frequent.
        """
        # In test mode, skip network operations
        if self.running_unit_tests:
            return

        if not self.refresh_allowed():
            return

        validators_to_query = self.get_possible_validators()
        logger.info(f"Querying {len(validators_to_query)} possible validators for positions")
        result = await self.get_positions_with_retry(validators_to_query)

        if result:
            self.write_positions_to_disk(result)
        else:
            logger.info("No positions found.")

        self.last_update_time = time.time()
        logger.info("PositionInspector successfully completed signal processing.")

    def write_positions_to_disk(self, positions):
        """
        Atomically writes positions to disk.

        Args:
            positions: List of position dictionaries to save, or None/empty list
        """
        try:
            file_path = MinerConfig.get_position_file_location()
            # get_position_file_location() is process-global. That is correct for one
            # wallet per process, but a host running several miners in a single process
            # has every PositionInspector writing the same path: the file ends up holding
            # whichever wallet finished last and the rest are silently discarded. Keying
            # the destination by hotkey keeps each wallet's positions separate.
            hotkey = getattr(getattr(self.wallet, "hotkey", None), "ss58_address", None)
            if hotkey:
                root, ext = os.path.splitext(file_path)
                file_path = f"{root}-{hotkey}{ext}"
            # A shared temp name races the same way, and worse: the first shutil.move
            # renames it away, so every other writer fails with ENOENT and reports
            # "Failed to save positions to disk" while nothing is actually wrong.
            temp_path = f"{file_path}.tmp.{os.getpid()}.{threading.get_ident()}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(temp_path, 'w') as f:
                json.dump(positions if positions else [], f, indent=2)
            shutil.move(temp_path, file_path)
            logger.info(f"Successfully saved {len(positions) if positions else 0} positions to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save positions to disk: {e}")

