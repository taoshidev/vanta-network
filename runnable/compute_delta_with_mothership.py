import argparse
import time
import requests
import json

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.elimination_server import EliminationServer
from vali_objects.utils.position_manager import PositionManager
import bittensor as bt


def compute_delta(mothership_json, min_time_ms):

    bt.logging.info("Found mothership data with the following attributes:")
    # Log every key and value apir in the data except for positions, eliminations, and plagiarism scores
    for key, value in mothership_json.items():
        # Check is the value is of type dict or list. If so, print the size of the dict or list
        if isinstance(value, dict) or isinstance(value, list):
            # Log the size of the positions, eliminations, and plagiarism scores
            bt.logging.info(f"    {key}: {len(value)} entries")
        else:
            bt.logging.info(f"    {key}: {value}")
    backup_creation_time_ms = mothership_json['created_timestamp_ms']

    # EliminationServer creates its own RPC clients internally (forward compatibility pattern)
    elimination_manager = EliminationServer(running_unit_tests=True)
    position_manager = PositionManager(perform_order_corrections=True,
                                       challengeperiod_manager=None,
                                       elimination_manager=elimination_manager)

    # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
    # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup smallest timestamp is
    # older than the local smallest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
    smallest_disk_ms, largest_disk_ms = (
        position_manager.get_extreme_position_order_processed_on_disk_ms())
    smallest_backup_ms = mothership_json['youngest_order_processed_ms']
    largest_backup_ms = mothership_json['oldest_order_processed_ms']
    try:
        formatted_backup_creation_time = TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)
        formatted_disk_date_largest = TimeUtil.millis_to_formatted_date_str(largest_disk_ms)
        formatted_backup_date_largest = TimeUtil.millis_to_formatted_date_str(largest_backup_ms)
        formatted_disk_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_disk_ms)
        formatted_backup_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_backup_ms)
    except:  # noqa: E722
        formatted_backup_creation_time = backup_creation_time_ms
        formatted_disk_date_largest = largest_disk_ms
        formatted_backup_date_largest = largest_backup_ms
        formatted_disk_date_smallest = smallest_disk_ms
        formatted_backup_date_smallest = smallest_backup_ms

    bt.logging.info("Timestamp analysis of mothership vs disk (UTC):")
    bt.logging.info(f"    mothership_creation_time: {formatted_backup_creation_time}")
    bt.logging.info(f"    smallest_disk_order_timestamp: {formatted_disk_date_smallest}")
    bt.logging.info(f"    smallest_mothership_order_timestamp: {formatted_backup_date_smallest}")
    bt.logging.info(f"    oldest_disk_order_timestamp: {formatted_disk_date_largest}")
    bt.logging.info(f"    oldest_mothership_order_timestamp: {formatted_backup_date_largest}")


    n_existing_position = position_manager.get_number_of_miners_with_any_positions()
    n_existing_eliminations = position_manager.get_number_of_eliminations()
    msg = (f"Detected {n_existing_position} existing hotkeys with positions, {n_existing_eliminations} existing eliminations")
    bt.logging.info(msg)

    bt.logging.info("Computing delta positions")

    delta_positions = []
    position_uuids_added = set()
    delta_orders = []
    delta_order_positions = []
    for hotkey in position_manager.get_miner_hotkeys_with_at_least_one_position():
        # sort positions by close_ms otherwise, writing a closed position after an open position for the same
        # trade pair will delete the open position
        remote_positions = mothership_json['positions'].get(hotkey, [])
        if remote_positions:
            remote_positions = [Position(**json_dict) for json_dict in remote_positions['positions']]

        my_positions = position_manager.get_positions_for_one_hotkey(hotkey)
        if min_time_ms:
            remote_positions = [p for p in remote_positions if any(o.processed_ms >= min_time_ms for o in p.orders)]
            my_positions = [p for p in my_positions if any(o.processed_ms >= min_time_ms for o in p.orders)]
        for mp in my_positions:
            corresponding_position = None
            for rp in remote_positions:
                if mp.position_uuid == rp.position_uuid:
                    corresponding_position = rp
                    break
            if corresponding_position is None:
                delta_positions.append(mp)
                position_uuids_added.add(mp.position_uuid)
                continue

            for mo in mp.orders:
                corresponding_order = None
                for ro in corresponding_position.orders:
                    if mo.order_uuid == ro.order_uuid:
                        corresponding_order = ro
                        break
                if corresponding_order is None:
                    delta_orders.append((corresponding_position.orders, mp.orders))
                    if mp.position_uuid not in position_uuids_added:
                        delta_order_positions.append(mp)
                        position_uuids_added.add(mp.position_uuid)
                    break

    print(f"Found {len(delta_positions)} positions to snap to and {len(delta_orders)} order deltas to snap to ")
    ans = ',\n\n'.join([p.to_copyable_str() for p in delta_positions + delta_order_positions])
    print(ans)

def get_mothership_checkpoint(url, api_key):
    data = {
        'api_key': api_key
    }
    json_data = json.dumps(data)
    headers = {
        'Content-Type': 'application/json',
    }
    test = requests.get(url, data=json_data, headers=headers)
    return test.json()


if __name__ == "__main__":
    bt.logging.enable_info()
    parser = argparse.ArgumentParser(description="Compute a delta with mothership in a format that can be used with PM.")
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--url', type=str, default=None)
    parser.add_argument('--min-time-ms', type=int, default=None)

    t0 = time.time()
    # Parse command-line arguments
    args = parser.parse_args()



    # Use the disable_backup argument to control backup
    api_key = args.api_key
    url = args.url
    min_time_ms = args.min_time_ms
    bt.logging.info(f"Computing delta. api_key: {api_key}, url: {url}, min_time_ms: {min_time_ms}")

    mothership_json = get_mothership_checkpoint(url, api_key)
    #with open('validator_checkpoint.json', 'r') as f:
    #    mothership_json = json.loads(f.read())

    compute_delta(mothership_json, min_time_ms)
    bt.logging.info("Delta complete in %.2f seconds" % (time.time() - t0))
