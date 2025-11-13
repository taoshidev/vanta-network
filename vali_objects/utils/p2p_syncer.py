import base64
import gzip
import json
import math
import statistics
import traceback
import asyncio
from collections import defaultdict
from typing import List, Set

import bittensor as bt
from bittensor import AxonInfo, NeuronInfo

import template
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair
from vali_objects.vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.validator_sync_base import ValidatorSyncBase

class P2PSyncer(ValidatorSyncBase):
    def __init__(self, wallet=None, metagraph=None, is_testnet=None, shutdown_dict=None, signal_sync_lock=None,
                 signal_sync_condition=None, n_orders_being_processed=None, running_unit_tests=False,
                 position_manager=None, ipc_manager=None):
        super().__init__(shutdown_dict, signal_sync_lock, signal_sync_condition, n_orders_being_processed,
                         running_unit_tests=running_unit_tests, position_manager=position_manager,
                         ipc_manager=ipc_manager)
        self.wallet = wallet
        self.metagraph = metagraph
        self.golden = None
        if self.wallet is not None:
            self.hotkey = self.wallet.hotkey.ss58_address
        # flag for testnet
        self.is_testnet = is_testnet
        self.created_golden = False
        self.last_signal_sync_time_ms = 0
        self.running_unit_tests = running_unit_tests

    def receive_checkpoint(self, synapse: template.protocol.ValidatorCheckpoint) -> template.protocol.ValidatorCheckpoint:
        """
        TODO: properly integrate with validator.py
        receive checkpoint request, and ensure that only requests received from valid validators are processed.
        """
        sender_hotkey = synapse.dendrite.hotkey

        # validator responds to poke from validator and attaches their checkpoint
        if sender_hotkey in [axon.hotkey for axon in self.get_validators()]:
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

    async def send_checkpoint_requests(self):
        """
        serializes checkpoint json and transmits to all validators via synapse
        """
        # only validators should request a checkpoint with a poke
        if not self.is_testnet and self.hotkey not in [axon.hotkey for axon in self.get_validators()]:
            bt.logging.info("Aborting send_checkpoint_poke; not a qualified validator")
            return

        # get axons to send checkpoints to
        validator_axons = self.get_largest_staked_validators(ValiConfig.TOP_N_STAKE)

        try:
            bt.logging.info(f"Validator {self.wallet.hotkey.ss58_address} requesting checkpoints")
            # create dendrite and transmit synapse
            checkpoint_synapse = template.protocol.ValidatorCheckpoint()
            async with bt.dendrite(wallet=self.wallet) as dendrite:
                validator_responses = await dendrite.aquery(axons=validator_axons,  synapse=checkpoint_synapse, timeout=60 * 5)

            n_failures = 0
            n_successful_checkpoints = 0
            hotkey_to_received_checkpoint = {}

            hotkey_to_v_trust = {}
            for neuron in self.metagraph.get_neurons():
                if neuron.validator_trust >= 0:
                    hotkey_to_v_trust[neuron.hotkey] = neuron.validator_trust

            for i, response in enumerate(validator_responses):
                if response.successfully_processed:
                    # Decode from base64 and decompress back into json
                    decoded = base64.b64decode(response.checkpoint)
                    decompressed = gzip.decompress(decoded).decode('utf-8')
                    recv_checkpoint = json.loads(decompressed)

                    hotkey = response.validator_receive_hotkey
                    hotkey_to_received_checkpoint[hotkey] = [hotkey_to_v_trust[hotkey], recv_checkpoint]

                    bt.logging.info(f"Successfully processed checkpoint from axon [{i+1}/{len(validator_responses)}]: {response.validator_receive_hotkey}")
                    n_successful_checkpoints += 1
                else:
                    n_failures += 1
                    if response.error_message:
                        bt.logging.info(
                            f"Checkpoint poke to axon [{i + 1}/{len(validator_responses)}] {response.axon.hotkey} errored: {response.error_message}")
                    else:
                        bt.logging.info(
                            f"Checkpoint poke to axon [{i + 1}/{len(validator_responses)}] {response.axon.hotkey} failed with status code: {response.axon.status_code}")

            bt.logging.info(f"{n_successful_checkpoints} responses succeeded. {n_failures} responses failed")

            if (n_successful_checkpoints > 0 and self.is_testnet) or n_successful_checkpoints >= ValiConfig.MIN_CHECKPOINTS_RECEIVED:
                # sort all our successful responses by validator_trust
                sorted_v_trust = sorted(hotkey_to_received_checkpoint.items(), key=lambda item: item[1][0], reverse=True)
                hotkey_to_received_checkpoint = {checkpoint[0]: checkpoint[1] for checkpoint in sorted_v_trust}

                bt.logging.info("Received enough checkpoints, now creating golden.")
                self.created_golden = self.create_golden(hotkey_to_received_checkpoint)
            else:
                bt.logging.info("Not enough checkpoints received to create a golden.")
                self.created_golden = False

        except Exception as e:
            bt.logging.info(f"Error generating golden with error [{e}]")

    def create_golden(self, trusted_checkpoints: dict) -> bool:
        """
        Create golden checkpoint from active validators (received order in last 10 hrs)
        """
        valid_checkpoints = {}

        # checkpoint is valid/not stale if the last order is recent
        for hotkey, checkpoint in trusted_checkpoints.items():
            # get the first 10 up-to-date checkpoints
            if len(valid_checkpoints) >= ValiConfig.TOP_N_CHECKPOINTS:
                break
            # add this checkpoint's data if the checkpoint is up-to-date
            latest_order_ms = self.last_order_time_in_checkpoint(checkpoint[1])
            if TimeUtil.now_in_millis() - latest_order_ms < 1000 * 60 * 60 * 10 or self.running_unit_tests:  # validators with no orders processed in 10 hrs are considered stale
                valid_checkpoints[hotkey] = checkpoint[1]
            else:
                bt.logging.info(f"Checkpoint from validator {hotkey} is stale with newest order timestamp {latest_order_ms}, {round((TimeUtil.now_in_millis() - latest_order_ms)/(1000 * 60 * 60))} hrs ago, Skipping.")

        if len(valid_checkpoints) == 0:
            bt.logging.info(f"All {len(trusted_checkpoints)} checkpoints are stale, unable to build golden.")
            return False
        else:
            bt.logging.info(f"Building golden from [{len(valid_checkpoints)}/{len(trusted_checkpoints)}] up-to-date checkpoints.")

        for hotkey, chk in valid_checkpoints.items():
            bt.logging.info(f"{hotkey} sent checkpoint {self.checkpoint_summary(chk)}")
            bt.logging.info("--------------------------------------------------")

        golden_eliminations = self.position_manager.elimination_manager.get_eliminations_from_memory()
        golden_positions = self.p2p_sync_positions(valid_checkpoints)
        golden_challengeperiod = self.p2p_sync_challengeperiod(valid_checkpoints)

        self.golden = {
            "created_timestamp_ms": TimeUtil.now_in_millis(),
            "hard_snap_cutoff_ms": TimeUtil.now_in_millis() - 1000 * 60 * 15,
            "eliminations": golden_eliminations,
            "positions": golden_positions,
            "challengeperiod": golden_challengeperiod
        }

        bt.logging.info(f"Created golden checkpoint: {self.checkpoint_summary(self.golden)}")
        return True

    def p2p_sync_challengeperiod(self, valid_checkpoints: dict):
        """
        hotkeys in challenge period determined by simple majority. use the median timestamp for each
        """
        challengeperiod_testing_data = defaultdict(list)        # {hotkey: [time]}
        challengeperiod_success_data = defaultdict(list)        # {hotkey: [time]}

        challengeperiod_testing = defaultdict(int)              # {hotkey: time}
        challengeperiod_success = defaultdict(int)              # {hotkey: time}

        for hotkey, checkpoint in valid_checkpoints.items():
            self.parse_checkpoint_challengeperiod(checkpoint, challengeperiod_testing_data, challengeperiod_success_data)

        threshold = self.consensus_threshold(len(valid_checkpoints))
        majority_testing = {hotkey for hotkey, times in challengeperiod_testing_data.items() if len(times) >= threshold}
        majority_success = {hotkey for hotkey, times in challengeperiod_success_data.items() if len(times) >= threshold}

        for hotkey in majority_testing:
            challengeperiod_testing[hotkey] = statistics.median_low(challengeperiod_testing_data[hotkey])
        for hotkey in majority_success:
            challengeperiod_success[hotkey] = statistics.median_low(challengeperiod_success_data[hotkey])

        return {"testing": challengeperiod_testing, "success": challengeperiod_success}

    def parse_checkpoint_challengeperiod(self, checkpoint: dict, testing_data: dict, success_data: dict):
        """
        parse checkpoint challengeperiod data
        testing_data = {hotkeys in challengeperiod test: [time]}
        success_data = {hotkeys in challengperiod success: [time]}
        """
        challengeperiod_testing = checkpoint.get("challengeperiod", {}).get("testing", {})
        for hotkey, timestamp in challengeperiod_testing.items():
            testing_data[hotkey].append(timestamp)

        challengeperiod_success = checkpoint.get("challengeperiod", {}).get("success", {})
        for hotkey, timestamp in challengeperiod_success.items():
            success_data[hotkey].append(timestamp)

    def p2p_sync_positions(self, valid_checkpoints: dict):
        """
        Simple majority approach
            If a position’s uuid exists on the majority of validators, that position is kept.
            If an order uuid exists in the majority of positions, that order is kept.
                Choose the order with the median price.
        """
        golden_positions = defaultdict(lambda: defaultdict(list))

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}
        miner_to_uuids = defaultdict(lambda: defaultdict(set))  # {miner_hotkey: {positions:[position_uuid], orders:[order_uuid]}
        miner_counts = defaultdict(int)                         # {miner_hotkey: count}

        positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        orders_matrix = defaultdict(lambda: defaultdict(list))                          # {position_uuid: {validator hotkey: [all orders on validator]}}

        # parse each checkpoint to count occurrences of each position and order
        for hotkey, checkpoint in valid_checkpoints.items():
            self.parse_checkpoint_positions(hotkey, checkpoint, position_counts, order_counts, order_data, miner_to_uuids, miner_counts, positions_matrix, orders_matrix)
        self.prune_position_orders(order_counts, orders_matrix)

        # miners who are still running legacy code. do not want to include them in checkpoint
        self.find_legacy_miners(len(valid_checkpoints), order_counts, miner_to_uuids, position_counts, order_data)

        # get the set of position_uuids that appear in the majority of checkpoints
        positions_threshold = self.consensus_threshold(len(valid_checkpoints))
        majority_positions = {position_uuid for position_uuid, count in position_counts.items() if count >= positions_threshold}
        seen_positions = set()
        seen_orders = set()

        for validator_hotkey, checkpoint in valid_checkpoints.items():
            positions = checkpoint.get("positions", {})
            for miner_hotkey, miner_positions in positions.items():
                if miner_counts[miner_hotkey] < positions_threshold:
                    continue

                # combinations where the position_uuid appears in the majority
                uuid_matched_positions = self.construct_positions_uuid_in_majority(miner_positions, majority_positions, seen_positions, seen_orders, position_counts, order_counts, order_data, orders_matrix, validator_hotkey)
                golden_positions[miner_hotkey]["positions"].extend(uuid_matched_positions)

        # combinations where the position_uuid does not appear in the majority, instead we use a heuristic match to combine positions
        for position in self.heuristic_resolve_positions(positions_matrix, len(valid_checkpoints), seen_positions):
            bt.logging.info(f"Position {position['position_uuid']} on miner {position['miner_hotkey']} matched, adding back in")
            miner_hotkey = position["miner_hotkey"]
            golden_positions[miner_hotkey]["positions"].append(position)

        # convert defaultdict to dict
        return {miner: dict(golden_positions[miner]) for miner in golden_positions}


    def construct_positions_uuid_in_majority(self, miner_positions: dict, majority_positions: Set[str], seen_positions: Set[str], seen_orders: Set[str], position_counts: dict, order_counts: dict, order_data: dict, orders_matrix: dict, validator_hotkey: str) -> List[dict]:
        """
        return the positions to add to golden, when the position_uuid appears in the majority of checkpoints.
        construct each position from its orders. if the order appears in the majority then it is taken, otherwise
        the order is attempted to be matched to other orders using a heuristic.

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}

        orders_matrix = defaultdict(lambda: defaultdict(list))  # {position_uuid: {validator hotkey: [all orders on validator]}}
        """
        uuid_matched_positions = []
        resolved_orders = set()  # separate from seen_orders, because we want to be able to match with seen orders

        for position in miner_positions["positions"]:
            position_uuid = position["position_uuid"]
            # position exists on majority of validators
            if position_uuid in majority_positions and position_uuid not in seen_positions:
                seen_positions.add(position_uuid)
                new_position = Position(**position)
                new_position.orders = []

                # get the set of order_uuids that appear in the majority of positions for a position_uuid
                orders_threshold = self.consensus_threshold(position_counts[position_uuid])
                majority_orders = {order_uuid for order_uuid, count in order_counts[position_uuid].items() if count >= orders_threshold}

                for order_uuid in order_counts[position_uuid].keys():
                    if order_uuid not in seen_orders:
                        # combinations where the order_uuid appears in the majority
                        if order_uuid in majority_orders:
                            orders = order_data[order_uuid]
                        # combinations where the order_uuid does not appear in the majority, instead we use a heuristic to combine orders
                        else:
                            orders = self.find_matching_orders(order_data[order_uuid][0], orders_matrix[position_uuid], resolved_orders)
                            # order has matched with another order that has already been inserted
                            if not set([o["order_uuid"] for o in orders]).isdisjoint(seen_orders):
                                seen_orders.update([o["order_uuid"] for o in orders])
                                continue

                            if len(orders) > self.consensus_threshold(position_counts[position_uuid], heuristic_match=True):
                                bt.logging.info(f"Order {order_uuid} with Position {position_uuid} on miner {position['miner_hotkey']} matched with {[o['order_uuid'] for o in orders]}, adding back in")
                            else:
                                bt.logging.info(f"Order {order_uuid} with Position {position_uuid} only matched [{len(orders)}/{position_counts[position_uuid]}] times on miner {position['miner_hotkey']} with with {[o['order_uuid'] for o in orders]}. Skipping")
                                continue

                        trade_pair = TradePair.from_trade_pair_id(position["trade_pair"][0])
                        median_order = self.get_median_order(orders, trade_pair)
                        new_position.orders.append(median_order)
                        seen_orders.update([o["order_uuid"] for o in orders])

                new_position.orders.sort(key=lambda o: o.processed_ms)
                try:
                    new_position.rebuild_position_with_updated_orders(self.position_manager.live_price_fetcher)
                    position_dict = json.loads(new_position.to_json_string())
                    uuid_matched_positions.append(position_dict)
                except ValueError as v:
                    bt.logging.info(f"Miner [{new_position.miner_hotkey}] Position [{new_position.position_uuid}] Orders {[o.order_uuid for o in new_position.orders]} ValueError {v}")
        return uuid_matched_positions

    def find_matching_orders(self, order: dict, validator_to_orders: dict, resolved_orders: Set[str]) -> List[dict] | None:
        """
        compare an order to all other orders associated with a position, and find all the matches using a heuristic.
        sort matches by order_uuid.

        validator_to_orders = defaultdict(list)  # {validator hotkey: [all orders on validator]}
        """
        if order["order_uuid"] in resolved_orders:
            return []
        matched_orders = []

        for validator_hotkey, order_list in validator_to_orders.items():
            for o in order_list:
                if o["order_uuid"] in resolved_orders:
                    continue

                # orders must have the same order_uuid or same leverage, order_type, and processed_ms
                if self.dict_orders_aligned(order, o):
                    matched_orders.append(o)
                    break

        for o in matched_orders:
            resolved_orders.add(o["order_uuid"])

        matched_orders.sort(key=lambda x: x["order_uuid"])
        return matched_orders

    def checkpoint_summary(self, checkpoint: dict) -> dict:
        """
        returns a summary of the checkpoint information
        {miner hotkey: [num of positions, num of orders]
        """
        summary = {}
        positions = checkpoint.get("positions", {})
        for miner_hotkey, miner_positions in positions.items():
            orders = 0
            for pos in miner_positions['positions']:
                orders += len(pos["orders"])
            pos_orders = [len(miner_positions['positions']), orders]
            summary[miner_hotkey] = pos_orders
        return summary

    def last_order_time_in_checkpoint(self, checkpoint: dict) -> int:
        """
        determine the latest order on each validator checkpoint
        """
        latest_order_ms = 0
        # get positions for each miner
        positions = checkpoint.get("positions", {})
        for miner_hotkey, miner_positions in positions.items():
            for position in miner_positions["positions"]:
                for order in position["orders"]:
                    latest_order_ms = max(latest_order_ms, order["processed_ms"])
        return latest_order_ms

    def parse_checkpoint_positions(self, validator_hotkey: str, checkpoint: dict, position_counts: dict, order_counts: dict, order_data: dict, miner_to_uuids: dict, miner_counts: dict, positions_matrix: dict, orders_matrix: dict):
        """
        parse checkpoint data

        position_counts = defaultdict(int)                      # {position_uuid: count}
        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        order_data = defaultdict(list)                          # {order_uuid: [{order}]}
        miner_to_uuids = defaultdict(lambda: defaultdict(set))  # {miner_hotkey: {positions:[position_uuid], orders:[order_uuid]}
        miner_counts = defaultdict(int)                         # {miner_hotkey: count}

        positions_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        orders_matrix = defaultdict(lambda: defaultdict(list))                          # {position_uuid: {validator_hotkey: [all orders]}}
        """
        # get positions for each miner
        positions = checkpoint.get("positions", {})
        for miner_hotkey, miner_positions in positions.items():
            miner_counts[miner_hotkey] += 1
            for position in miner_positions["positions"]:
                position_uuid = position["position_uuid"]
                position_counts[position_uuid] += 1
                miner_to_uuids[miner_hotkey]["positions"].add(position_uuid)

                # count and save orders
                for order in position["orders"]:
                    order_uuid = order["order_uuid"]
                    order_counts[position_uuid][order_uuid] += 1
                    order_data[order_uuid].append(dict(order))
                    miner_to_uuids[miner_hotkey]["orders"].add(order_uuid)
                    orders_matrix[position_uuid][validator_hotkey].append(order)

                orders_matrix[position_uuid][validator_hotkey].sort(key=lambda o: o["processed_ms"])
                positions_matrix[miner_hotkey][position["trade_pair"][0]][validator_hotkey].append(position)

    def prune_position_orders(self, order_counts: dict, orders_matrix: dict):
        """
        some validators may incorrectly combine multiple positions into one, so an order_uuid may appear under
        multiple positions. We will only keep the order_uuid in the position that it appears the most in.

        order_counts = defaultdict(lambda: defaultdict(int))    # {position_uuid: {order_uuid: count}}
        orders_matrix = defaultdict(lambda: defaultdict(list))  # {position_uuid: {validator hotkey: [all orders on validator]}}
        """
        order_max_count_appears_in_position = defaultdict(tuple)       # {order_uuid: (position_uuid, count)}
        # find the position_uuid with the max count for each order_uuid
        for position, orders in order_counts.items():
            for order, count in orders.items():
                if order not in order_max_count_appears_in_position:
                    order_max_count_appears_in_position[order] = (position, count)
                else:
                    if count > order_max_count_appears_in_position[order][1]:
                        order_max_count_appears_in_position[order] = (position, count)

        # remove order_uuid from all other position_uuids
        for position, orders in order_counts.items():
            for order in list(orders.keys()):
                max_position = order_max_count_appears_in_position[order][0]
                if position != max_position:
                    del orders[order]
                    order_counts[max_position][order] += 1

                    for vali_orders in orders_matrix[position].values():
                        vali_orders.remove(order) if order in vali_orders else None

    def find_legacy_miners(self, num_checkpoints: int, order_counts: dict, miner_to_uuids: dict, position_counts: dict, order_data: dict) -> Set[str]:
        """
        detect miners running legacy code. miner with legacy code will have all unique uuid's across validators
        so the counts of each position_uuid and order_uuid would be 1. Mark miners with some unique position/order
        uuid's as potentially running legacy code.
        """
        # create single dict of all orders and counts
        all_order_counts = defaultdict(int)  # {order_uuid: count}
        all_order_counts_list = list(order_counts.values())
        for order_count_dict in all_order_counts_list:
            for order_uuid, count in order_count_dict.items():
                all_order_counts[order_uuid] += count

        legacy_miners = set()            # position/order uuids are all unique across validators
        legacy_miner_candidates = set()  # at least one position/order uuid is unique across validators

        if num_checkpoints > 1:
            for miner_hotkey, uuids in miner_to_uuids.items():
                # number of repeated position_uuids and order_uuids
                num_repeated_pos = sum(1 for pos_uuid in uuids["positions"] if position_counts[pos_uuid] > 1)
                num_repeated_orders = sum(1 for order_uuid in uuids["orders"] if all_order_counts[order_uuid] > 1)
                newest_order_timestamp = max([order_data[order_uuid][0]["processed_ms"] for order_uuid in uuids["orders"]])
                newest_unique_order_timestamp = max([order_data[order_uuid][0]["processed_ms"] for order_uuid in uuids["orders"] if all_order_counts[order_uuid] == 1], default=-1)
                newest_unique_order_uuid = next((order_uuid for order_uuid in uuids["orders"]
                                                if all_order_counts[order_uuid] == 1 and order_data[order_uuid][0]["processed_ms"] == newest_unique_order_timestamp), "")

                # if there are positions or orders that only appear once across all validators
                if num_repeated_pos != len(uuids["positions"]) or num_repeated_orders != len(uuids["orders"]):
                    if (num_repeated_pos == 0) and (num_repeated_orders == 0):
                        legacy_miners.add(miner_hotkey)
                    elif newest_unique_order_timestamp == newest_order_timestamp:
                        legacy_miner_candidates.add(miner_hotkey)
                    bt.logging.info(
                        f"Miner {miner_hotkey} has [{(len(uuids['positions']) - num_repeated_pos)}/{len(uuids['positions'])} legacy positions, {(len(uuids['orders']) - num_repeated_orders)}/{len(uuids['orders'])} legacy orders]. Newest legacy order {newest_unique_order_uuid} at timestamp {newest_unique_order_timestamp}")
                else:
                    bt.logging.info(f"Miner {miner_hotkey} has 0 legacy positions or orders")
        bt.logging.info(f"legacy_miners: {legacy_miners}")
        bt.logging.info(f"legacy_miner_candidates: {legacy_miner_candidates}")
        return legacy_miners

    def heuristic_resolve_positions(self, positions_matrix: dict, num_checkpoints: int, seen_positions: set) -> List[dict]:
        """
        takes a matrix of unmatched positions, and returns a list of positions to add back in
        positions_matrix: {miner hotkey: {trade pair: {validator hotkey: [all positions on validator]}}}
        """
        resolved_position_uuids = set()

        matched_positions = []
        # want to resolve all the unmatched positions for the validators against each other
        for miner_hotkey, trade_pairs in positions_matrix.items():
            for trade_pair, validator in trade_pairs.items():
                for validator_hotkey, position_list in validator.items():
                    for position in position_list:
                        if position["position_uuid"] in seen_positions:
                            continue
                        matches = self.find_matching_positions(position, trade_pairs[trade_pair], resolved_position_uuids, validator_hotkey)

                        if (len(matches) > self.consensus_threshold(num_checkpoints, heuristic_match=True) and
                                set([match["position_uuid"] for match in matches]).isdisjoint(seen_positions)):
                            # median number of orders for matched positions
                            median_order_count = len(matches[len(matches) // 2]["orders"])
                            # greatest common number of orders by 2 or more positions
                            max_common_order_count = 0
                            for i in range(len(matches)-1):
                                if len(matches[i]["orders"]) == len(matches[i+1]["orders"]):
                                    max_common_order_count = len(matches[i]["orders"])
                                    break
                            goal_order_count = max(median_order_count, max_common_order_count)

                            matches_with_goal_order_count = [p for p in matches if len(p["orders"]) == goal_order_count]
                            bt.logging.info(f"Miner hotkey {miner_hotkey} has matches {[p['position_uuid'] for p in matches]}. goal_order_count: {goal_order_count}. matches_with_goal_order_count: {matches_with_goal_order_count}.")
                            matched_positions.append(matches_with_goal_order_count[0])
                        else:
                            bt.logging.info(f"Position {position['position_uuid']} only matched [{len(matches)}/{num_checkpoints}] times on miner {position['miner_hotkey']} with matches {[p['position_uuid'] for p in matches]}. Skipping")

                        seen_positions.update([p["position_uuid"] for p in matches])
        return matched_positions

    def find_matching_positions(self, position: dict, trade_pair_validator_positions: dict, resolved_positions: set, corresponding_validator_hotkey: str) -> List[dict]:
        """
        compares a position from corresponding_validator_hotkey to all other positions with matching trade pair from all the other validators.
        positions are matched with a heuristic, and returned in a list sorted by number of orders then position_uuid

        position match heuristic:
            same position type
            same # of orders
            opened and closed around SYNC_LOOK_AROUND_MS of each other

        trade_pair_validator_positions: {validator hotkey: [all positions on validator]}
        """
        if position["position_uuid"] in resolved_positions:
            return []

        matched_positions = [position]

        for validator_hotkey, position_list in trade_pair_validator_positions.items():
            if validator_hotkey == corresponding_validator_hotkey:
                continue

            for p in position_list:
                if p["position_uuid"] in resolved_positions:
                    continue

                # positions have same position_type, # of orders, and open/close_ms times
                # or positions contain orders that are matched or have the same order_uuid
                if (self.dict_positions_aligned(position, p, validate_num_orders=True) or
                        self.positions_order_aligned(position, p)):
                    matched_positions.append(p)
                    break

        for p in matched_positions:
            resolved_positions.add(p["position_uuid"])

        # we sort the matches by number of orders and then position_uuid
        matched_positions.sort(key=lambda x: (-len(x["orders"]), x["position_uuid"]))
        return matched_positions

    def consensus_threshold(self, total_items: int, heuristic_match: bool=False) -> int:
        """
        threshold for including a position or order in the golden
        """
        if heuristic_match:
            return math.floor(total_items / 2)
        else:
            return math.ceil(total_items / 2)

    def get_median_order(self, orders: List[dict], trade_pair: TradePair) -> Order:
        """
        select the order with the median price from a list of orders with the same order_uuid
        """
        sorted_orders = sorted(orders, key=lambda o: o["price"])
        median_order = sorted_orders[len(orders)//2]
        median_order["trade_pair"] = trade_pair
        return Order(**median_order)

    def get_validators(self, neurons: List[NeuronInfo]=None) -> List[AxonInfo]:
        """
        get a list of all validators. defined as:
        stake > 1000 and validator_trust > 0.5
        """
        if self.is_testnet:
            return self.metagraph.get_axons()
        if neurons is None:
            neurons = self.metagraph.get_neurons()
        validator_axons = [n.axon_info for n in neurons
                           if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                           and n.axon_info.ip != ValiConfig.AXON_NO_IP]
        return validator_axons

    def get_largest_staked_validators(self, top_n_validators: int, neurons: List[NeuronInfo]=None) -> List[AxonInfo]:
        """
        get a list of the trusted validators for checkpoint sending
        return top 20 neurons sorted by stake
        """
        if self.is_testnet:
            return self.get_validators()
        if neurons is None:
            neurons = self.metagraph.get_neurons()
        sorted_stake_neurons = sorted(neurons, key=lambda n: n.stake, reverse=True)

        return self.get_validators(sorted_stake_neurons)[:top_n_validators]

    async def sync_positions_with_cooldown(self):
        now_ms = TimeUtil.now_in_millis()
        # Already performed a sync recently
        if now_ms - self.last_signal_sync_time_ms < 1000 * 60 * 15:
            return
        datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
        # Check if the time is right to sync signals
        if False: #self.is_testnet:
            # every hour in testnet
            if not (47 < datetime_now.minute < 57):
                return
        else:
            # Check if we are between 7:09 AM and 7:19 AM UTC
            # Temp change time to 21:00 UTC so we can see the effects in shadow mode ASAP
            if not (datetime_now.hour == 1 and (18 < datetime_now.minute < 30)):
                return

        try:
            bt.logging.info("Calling send_checkpoint_requests")
            self.golden = None
            await self.send_checkpoint_requests()
            if self.created_golden:
                bt.logging.info("Calling apply_golden")
                # TODO guard sync_positions with the signal lock once we move on from shadow mode
                self.sync_positions(True, candidate_data=self.golden)
        except Exception as e:
            bt.logging.error(f"Error sending checkpoint: {e}")
            bt.logging.error(traceback.format_exc())

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()

if __name__ == "__main__":
    bt.logging.enable_default()
    position_syncer = P2PSyncer(is_testnet=True)
    asyncio.run(position_syncer.send_checkpoint_requests())
    if position_syncer.created_golden:
        position_syncer.sync_positions(True, candidate_data=position_syncer.golden)
