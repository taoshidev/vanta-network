import argparse
import json
import gzip
import os
import shutil
import time

import traceback
from datetime import datetime

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.limit_order_manager import LimitOrderManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.asset_selection_manager import AssetSelectionManager
import bittensor as bt
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

DEBUG = 0

def backup_validation_directory():
    dir_to_backup = ValiBkpUtils.get_vali_dir()
    # Write to the backup location. Make sure it is a function of the date. No dashes. Days and months get 2 digits.
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_location = ValiBkpUtils.get_vali_bkp_dir() + date_str + '/'
    # Sync directory to the backup location using python shutil
    shutil.copytree(dir_to_backup, backup_location)
    bt.logging.info(f"backed up {dir_to_backup} to {backup_location}")


def force_validator_to_restore_from_checkpoint(validator_hotkey, metagraph, config, secrets):
    try:
        time_ms = TimeUtil.now_in_millis()
        if time_ms > 1716644087000 + 1000 * 60 * 60 * 2:  # Only perform under a targeted time as checkpoint goes stale quickly.
            return

        if "mothership" in secrets:
            bt.logging.warning(f"Validator {validator_hotkey} is the mothership. Not forcing restore.")
            return

        #if config.subtensor.network == "test":  # Only need do this in mainnet
        #    bt.logging.warning("Not forcing validator to restore from checkpoint in testnet.")
        #    return

        hotkey_to_v_trust = {neuron.hotkey: neuron.validator_trust for neuron in metagraph.neurons}
        my_trust = hotkey_to_v_trust.get(validator_hotkey)
        if my_trust is None:
            bt.logging.warning(f"Validator {validator_hotkey} not found in metagraph. Cannot determine trust.")
            return

        # Good enough
        #if my_trust > 0.5:
        #    return

        bt.logging.warning(f"Validator {validator_hotkey} trust is {my_trust}. Forcing restore.")
        regenerate_miner_positions(perform_backup=True, backup_from_data_dir=True, ignore_timestamp_checks=True)
        bt.logging.info('Successfully forced validator to restore from checkpoint.')

    except Exception as e:
        bt.logging.error(f"Error forcing validator to restore from checkpoint: {e}")
        bt.logging.error(traceback.format_exc())


def regenerate_miner_positions(perform_backup=True, backup_from_data_dir=False, ignore_timestamp_checks=False):
    # Check for compressed version first, then fallback to uncompressed for backward compatibility
    compressed_path = ValiBkpUtils.get_validator_checkpoint_path(use_data_dir=backup_from_data_dir)
    uncompressed_path = ValiBkpUtils.get_backup_file_path(use_data_dir=backup_from_data_dir)
    
    try:
        if os.path.exists(compressed_path):
            bt.logging.info(f"Found compressed checkpoint file: {compressed_path}")
            with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        elif os.path.exists(uncompressed_path):
            bt.logging.info(f"Found uncompressed checkpoint file: {uncompressed_path}")
            data = json.loads(ValiBkpUtils.get_file(uncompressed_path))
            if isinstance(data, str):
                data = json.loads(data)
        else:
            raise FileNotFoundError(f"No checkpoint file found at {uncompressed_path} or {compressed_path}")
            
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful guidance for common misnamed file scenarios
        if "Not a gzipped file" in error_msg:
            bt.logging.error(f"File {compressed_path} has .gz extension but contains uncompressed data.")
            bt.logging.error("Solution: Remove the .gz extension and rename to validator_checkpoint.json")
        elif "invalid start byte" in error_msg or "'utf-8' codec can't decode" in error_msg:
            bt.logging.error(f"File {uncompressed_path} appears to contain compressed data but lacks .gz extension.")
            bt.logging.error("Solution: Add .gz extension and rename to validator_checkpoint.json.gz")
        else:
            bt.logging.error(f"Unable to read validator checkpoint file. {error_msg}")
        
        return False

    bt.logging.info("Found validator backup file with the following attributes:")
    # Log every key and value apir in the data except for positions, eliminations, and plagiarism scores
    for key, value in data.items():
        # Check is the value is of type dict or list. If so, print the size of the dict or list
        if isinstance(value, dict) or isinstance(value, list):
            # Log the size of the positions, eliminations, and plagiarism scores
            bt.logging.info(f"    {key}: {len(value)} entries")
        else:
            bt.logging.info(f"    {key}: {value}")
    backup_creation_time_ms = data['created_timestamp_ms']

    elimination_manager = EliminationManager(None, None, None)
    position_manager = PositionManager(perform_order_corrections=True,
                                       challengeperiod_manager=None,
                                       elimination_manager=elimination_manager)
    contract_manager = ValidatorContractManager(config=None, running_unit_tests=False)
    challengeperiod_manager = ChallengePeriodManager(metagraph=None, position_manager=position_manager)
    perf_ledger_manager = PerfLedgerManager(None)
    asset_selection_manager = AssetSelectionManager()
    limit_order_manager = LimitOrderManager(position_manager, None)

    if DEBUG:
        position_manager.pre_run_setup()

    # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
    # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup smallest timestamp is
    # older than the local smallest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
    smallest_disk_ms, largest_disk_ms = (
        position_manager.get_extreme_position_order_processed_on_disk_ms())
    smallest_backup_ms = data['youngest_order_processed_ms']
    largest_backup_ms = data['oldest_order_processed_ms']
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

    bt.logging.info("Timestamp analysis of backup vs disk (UTC):")
    bt.logging.info(f"    backup_creation_time: {formatted_backup_creation_time}")
    bt.logging.info(f"    smallest_disk_order_timestamp: {formatted_disk_date_smallest}")
    bt.logging.info(f"    smallest_backup_order_timestamp: {formatted_backup_date_smallest}")
    bt.logging.info(f"    oldest_disk_order_timestamp: {formatted_disk_date_largest}")
    bt.logging.info(f"    oldest_backup_order_timestamp: {formatted_backup_date_largest}")

    if ignore_timestamp_checks:
        checkpoint_file = compressed_path if os.path.exists(compressed_path) else uncompressed_path
        bt.logging.info(f'Forcing validator restore no timestamp checks from: {checkpoint_file}')
        pass
    elif smallest_disk_ms >= smallest_backup_ms and largest_disk_ms <= backup_creation_time_ms:
        pass  # Ready for update!
    elif largest_disk_ms > backup_creation_time_ms:
        bt.logging.error(f"Please re-pull the backup file before restoring. Backup {formatted_backup_creation_time} appears to be older than the disk {formatted_disk_date_largest}.")
        return False
    elif smallest_disk_ms < smallest_backup_ms:
        #bt.logging.error("Your local filesystem has older orders than the backup. Please reach out to the team ASAP before regenerating. You may be holding irrecoverable data!")
        #return False
        pass  # Deregistered miners can trip this check. We will allow the regeneration to proceed.
    else:
        bt.logging.error("Problem with backup file detected. Please reach out to the team ASAP")
        return False


    n_existing_position = position_manager.get_number_of_miners_with_any_positions()
    n_existing_eliminations = position_manager.get_number_of_eliminations()
    msg = (f"Detected {n_existing_position} hotkeys with positions, {n_existing_eliminations} eliminations")
    bt.logging.info(msg)

    bt.logging.info("Overwriting all existing positions, eliminations, and plagiarism scores.")
    if perform_backup:
        backup_validation_directory()

    bt.logging.info(f"regenerating {len(data['positions'].keys())} hotkeys")
    position_manager.clear_all_miner_positions()
    for hotkey, json_positions in data['positions'].items():
        # sort positions by close_ms otherwise, writing a closed position after an open position for the same
        # trade pair will delete the open position
        positions = [Position(**json_positions_dict) for json_positions_dict in json_positions['positions']]
        if not positions:
            continue
        assert len(positions) > 0, f"no positions for hotkey {hotkey}"
        positions.sort(key=PositionManager.sort_by_close_ms)
        ValiBkpUtils.make_dir(ValiBkpUtils.get_miner_all_positions_dir(hotkey))
        for p_obj in positions:
            #bt.logging.info(f'creating position {p_obj}')
            position_manager.save_miner_position(p_obj)

        # Validate that the positions were written correctly
        disk_positions = position_manager.get_positions_for_one_hotkey(hotkey, sort_positions=True)
        #bt.logging.info(f'disk_positions: {disk_positions}, positions: {positions}')
        n_disk_positions = len(disk_positions)
        n_memory_positions = len(positions)
        memory_p_uuids = set([p.position_uuid for p in positions])
        disk_p_uuids = set([p.position_uuid for p in disk_positions])
        assert n_disk_positions == n_memory_positions, f"n_disk_positions: {n_disk_positions}, n_memory_positions: {n_memory_positions}"
        assert memory_p_uuids == disk_p_uuids, f"memory_p_uuids: {memory_p_uuids}, disk_p_uuids: {disk_p_uuids}"


    bt.logging.info(f"regenerating {len(data['eliminations'])} eliminations")
    position_manager.elimination_manager.write_eliminations_to_disk(data['eliminations'])

    perf_ledgers = data.get('perf_ledgers', {})
    bt.logging.info(f"regenerating {len(perf_ledgers)} perf ledgers")
    perf_ledger_manager.save_perf_ledgers(perf_ledgers)

    ## Now sync challenge period with the disk
    challengeperiod = data.get('challengeperiod', {})
    challengeperiod_manager.sync_challenge_period_data(challengeperiod)

    ## Sync miner account sizes with the disk
    miner_account_sizes = data.get('miner_account_sizes', {})
    if miner_account_sizes:
        bt.logging.info(f"syncing {len(miner_account_sizes)} miner account size records")
        contract_manager.sync_miner_account_sizes_data(miner_account_sizes)
    else:
        bt.logging.info("No miner account sizes found in backup data")

    challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

    limit_orders = data.get('limit_orders', {})
    limit_order_manager.sync_limit_orders(limit_orders)
    
    ## Restore asset selections
    asset_selections_data = data.get('asset_selections', {})
    if asset_selections_data:
        bt.logging.info(f"syncing {len(asset_selections_data)} miner asset selection records")
        asset_selection_manager.sync_miner_asset_selection_data(asset_selections_data)
    else:
        bt.logging.info("No asset selections found in backup data")

    return True

if __name__ == "__main__":
    bt.logging.enable_info()
    t0 = time.time()
    # Check commandline arg "disable_backup" to disable backup.
    parser = argparse.ArgumentParser(description="Regenerate miner positions with optional backup disabling.")
    # Add disable_backup argument, default is 0 (False), change type to int
    parser.add_argument('--backup', type=int, default=0,
                        help='Set to 1 to enable backup during regeneration process.')

    # Parse command-line arguments
    args = parser.parse_args()

    # Use the disable_backup argument to control backup
    perform_backup = bool(args.backup)
    bt.logging.info("regenerating miner positions")
    if not perform_backup:
        bt.logging.warning("backup disabled")
    passed = regenerate_miner_positions(perform_backup, ignore_timestamp_checks=True)
    if passed:
        bt.logging.info("regeneration complete in %.2f seconds" % (time.time() - t0))
    else:
        bt.logging.error("regeneration failed")
