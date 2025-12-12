import argparse
import json
import gzip
import os
import shutil
import time

import traceback
from datetime import datetime

from shared_objects.rpc.common_data_server import CommonDataServer
from shared_objects.rpc.metagraph_server import MetagraphServer
from shared_objects.rpc.rpc_client_base import RPCClientBase
from shared_objects.rpc.rpc_server_base import RPCServerBase
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.position import Position
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.challenge_period import ChallengePeriodServer
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.utils.elimination.elimination_server import EliminationServer
from vali_objects.utils.limit_order.limit_order_server import LimitOrderServer
from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.position_management.position_manager_server import PositionManagerServer
from vali_objects.contract.contract_server import ContractServer
from vali_objects.contract.contract_client import ContractClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from vali_objects.utils.asset_selection.asset_selection_server import AssetSelectionServer
from entitiy_management.entity_server import EntityServer
from entitiy_management.entity_client import EntityClient
import bittensor as bt

from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_server import PerfLedgerServer
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
import time as time_module

DEBUG = 0

def start_servers_for_restore():
    """
    Start all required RPC servers in background threads for restore operation.
    Returns dict of server instances that can be shut down later.
    """
    bt.logging.info("Starting RPC servers for restore operation...")
    servers = {}

    servers['common_data'] = CommonDataServer()
    servers['metagraph_server'] = MetagraphServer()
    # Start servers in dependency order
    # 1. Base servers with no dependencies
    servers['position'] = PositionManagerServer(
        running_unit_tests=True,
        is_backtesting=False,
        start_server=True,
        start_daemon=False,
        load_from_disk=False,  # Don't load existing positions (we're restoring from backup)
        split_positions_on_disk_load=False  # CRITICAL: Disable position splitting during restore
    )

    servers['contract'] = ContractServer(
        start_server=True,
        running_unit_tests=True
    )

    servers['perf_ledger'] = PerfLedgerServer(
        start_server=True,
        running_unit_tests=True
    )

    servers['challengeperiod'] = ChallengePeriodServer(
        start_server=True,
        running_unit_tests=True
    )

    # 2. Elimination server (needed by LimitOrderManager)
    servers['elimination'] = EliminationServer(
        start_server=True,
        running_unit_tests=True
    )

    # Give servers a moment to start listening
    time_module.sleep(2)

    # 3. Servers that depend on other servers
    servers['limit_order'] = LimitOrderServer(
        start_server=True,
        running_unit_tests=True,
        serve=False  # Don't start market order manager
    )

    servers['asset_selection'] = AssetSelectionServer(
        start_server=True,
        running_unit_tests=True
    )

    servers['entity'] = EntityServer(
        start_server=True,
        running_unit_tests=True
    )

    # Give all servers time to fully initialize
    time_module.sleep(1)
    bt.logging.success("All RPC servers started successfully")

    return servers

def shutdown_all_servers_and_clients():
    """
    Shutdown all RPC servers and clients using proper cleanup methods.

    This ensures complete cleanup and prevents the script from hanging.
    """
    bt.logging.info("Shutting down all RPC clients and servers...")

    # Step 1: Disconnect all clients first (prevents clients from holding connections)
    RPCClientBase.disconnect_all()
    bt.logging.info("  ✓ All RPC clients disconnected")

    # Step 2: Shutdown all servers and force-kill any processes still using RPC ports
    RPCServerBase.shutdown_all(force_kill_ports=True)
    bt.logging.success("  ✓ All RPC servers shut down and ports cleaned up")

    bt.logging.success("All servers and clients shut down successfully")

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

    # Load checkpoint file - fail fast if file is missing or corrupt
    if os.path.exists(compressed_path):
        bt.logging.info(f"Found compressed checkpoint file: {compressed_path}")
        try:
            with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            if "Not a gzipped file" in str(e):
                bt.logging.error(f"File {compressed_path} has .gz extension but contains uncompressed data.")
                bt.logging.error("Solution: Remove the .gz extension and rename to validator_checkpoint.json")
            raise RuntimeError(f"Failed to load compressed checkpoint: {e}") from e
    elif os.path.exists(uncompressed_path):
        bt.logging.info(f"Found uncompressed checkpoint file: {uncompressed_path}")
        try:
            data = json.loads(ValiBkpUtils.get_file(uncompressed_path))
            if isinstance(data, str):
                data = json.loads(data)
        except Exception as e:
            if "invalid start byte" in str(e) or "'utf-8' codec can't decode" in str(e):
                bt.logging.error(f"File {uncompressed_path} appears to contain compressed data but lacks .gz extension.")
                bt.logging.error("Solution: Add .gz extension and rename to validator_checkpoint.json.gz")
            raise RuntimeError(f"Failed to load uncompressed checkpoint: {e}") from e
    else:
        raise FileNotFoundError(f"No checkpoint file found at {uncompressed_path} or {compressed_path}")

    bt.logging.info("Found validator backup file with the following attributes:")
    # Log every key and value pair in the data except for positions, eliminations, and plagiarism scores
    for key, value in data.items():
        # Check is the value is of type dict or list. If so, print the size of the dict or list
        if isinstance(value, dict) or isinstance(value, list):
            # Log the size of the positions, eliminations, and plagiarism scores
            bt.logging.info(f"    {key}: {len(value)} entries")
        else:
            bt.logging.info(f"    {key}: {value}")
    backup_creation_time_ms = data['created_timestamp_ms']

    # Start RPC servers (tests production code paths)
    servers = start_servers_for_restore()

    try:
        # Create RPC clients to connect to the servers
        # This tests the actual production RPC communication paths
        position_client = PositionManagerClient(running_unit_tests=True)
        elimination_client = EliminationClient()
        contract_client = ContractClient()
        perf_ledger_client = PerfLedgerClient(running_unit_tests=True)
        challengeperiod_client = ChallengePeriodClient(running_unit_tests=True)
        limit_order_client = LimitOrderClient(running_unit_tests=True)
        asset_selection_client = AssetSelectionClient(running_unit_tests=True)
        entity_client = EntityClient(running_unit_tests=True)

        if DEBUG:
            position_client.pre_run_setup()

        # We want to get the smallest processed_ms timestamp across all positions in the backup and then compare this to
        # the smallest processed_ms timestamp across all orders on the local filesystem. If the backup smallest timestamp is
        # older than the local smallest timestamp, we will not regenerate the positions. Similarly for the oldest timestamp.
        smallest_disk_ms, largest_disk_ms = position_client.get_extreme_position_order_processed_on_disk_ms()
        smallest_backup_ms = data['youngest_order_processed_ms']
        largest_backup_ms = data['oldest_order_processed_ms']

        # Check if disk is empty (returns inf/0 when no positions exist)
        disk_is_empty = smallest_disk_ms == float('inf') or largest_disk_ms == 0

        # Format timestamps for display - fail fast if data is corrupt
        formatted_backup_creation_time = TimeUtil.millis_to_formatted_date_str(backup_creation_time_ms)
        formatted_backup_date_largest = TimeUtil.millis_to_formatted_date_str(largest_backup_ms)
        formatted_backup_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_backup_ms)

        if disk_is_empty:
            formatted_disk_date_largest = "N/A (no positions on disk)"
            formatted_disk_date_smallest = "N/A (no positions on disk)"
        else:
            formatted_disk_date_largest = TimeUtil.millis_to_formatted_date_str(largest_disk_ms)
            formatted_disk_date_smallest = TimeUtil.millis_to_formatted_date_str(smallest_disk_ms)

        bt.logging.info("Timestamp analysis of backup vs disk (UTC):")
        bt.logging.info(f"    backup_creation_time: {formatted_backup_creation_time}")
        bt.logging.info(f"    smallest_disk_order_timestamp: {formatted_disk_date_smallest}")
        bt.logging.info(f"    smallest_backup_order_timestamp: {formatted_backup_date_smallest}")
        bt.logging.info(f"    oldest_disk_order_timestamp: {formatted_disk_date_largest}")
        bt.logging.info(f"    oldest_backup_order_timestamp: {formatted_backup_date_largest}")

        # Validate timestamp consistency - fail fast on data integrity issues
        if ignore_timestamp_checks:
            checkpoint_file = compressed_path if os.path.exists(compressed_path) else uncompressed_path
            bt.logging.warning(f'SKIPPING TIMESTAMP CHECKS - Forcing restore from: {checkpoint_file}')
        elif disk_is_empty:
            bt.logging.info("✓ Disk is empty - proceeding with fresh restore")
        elif smallest_disk_ms >= smallest_backup_ms and largest_disk_ms <= backup_creation_time_ms:
            bt.logging.info("✓ Timestamp validation passed - backup is newer than disk data")
        elif largest_disk_ms > backup_creation_time_ms:
            raise ValueError(
                f"BACKUP TOO OLD: Backup created at {formatted_backup_creation_time} "
                f"but disk has data as recent as {formatted_disk_date_largest}. "
                f"Please re-pull a newer backup file before restoring."
            )
        elif smallest_disk_ms < smallest_backup_ms:
            # Deregistered miners can trip this check - allow to proceed but warn
            bt.logging.warning(
                f"Disk has older data ({formatted_disk_date_smallest}) than backup ({formatted_backup_date_smallest}). "
                f"This may be from deregistered miners. Proceeding with restore."
            )
        else:
            raise ValueError(
                f"TIMESTAMP VALIDATION FAILED: Unexpected timestamp relationship detected. "
                f"Backup: {formatted_backup_creation_time}, Disk range: {formatted_disk_date_smallest} to {formatted_disk_date_largest}"
            )


        n_existing_position = len(position_client.get_all_hotkeys())
        n_existing_eliminations = len(elimination_client.get_eliminations_from_memory())
        msg = (f"Detected {n_existing_position} hotkeys with positions, {n_existing_eliminations} eliminations")
        bt.logging.info(msg)

        bt.logging.info("Overwriting all existing positions, eliminations, and plagiarism scores.")
        if perform_backup:
            backup_validation_directory()

        # Calculate global statistics
        total_positions_in_backup = sum(len(json_positions['positions']) for json_positions in data['positions'].values())
        num_hotkeys = len(data['positions'].keys())

        bt.logging.info(f"=" * 80)
        bt.logging.info(f"RESTORE SUMMARY:")
        bt.logging.info(f"  Total hotkeys: {num_hotkeys}")
        bt.logging.info(f"  Total positions: {total_positions_in_backup}")
        bt.logging.info(f"  Average positions per hotkey: {total_positions_in_backup / num_hotkeys:.1f}")
        bt.logging.info(f"=" * 80)

        # CRITICAL: Clear both memory AND disk to avoid stale positions from previous runs
        # Without this, old positions on disk can trigger deletion logic during restore
        position_client.clear_all_miner_positions_and_disk()

        total_saved = 0
        for hotkey, json_positions in data['positions'].items():
            # Sort positions by close_ms to save in chronological order
            # (closed positions first, then open positions with close_ms=None → inf)
            positions = [Position(**json_positions_dict) for json_positions_dict in json_positions['positions']]
            if not positions:
                continue
            assert len(positions) > 0, f"no positions for hotkey {hotkey}"

            # Check for duplicate trade pairs BEFORE saving
            trade_pair_to_positions = {}
            for p in positions:
                tp_id = p.trade_pair.trade_pair_id
                if tp_id not in trade_pair_to_positions:
                    trade_pair_to_positions[tp_id] = []
                trade_pair_to_positions[tp_id].append(p)

            duplicates = {tp: ps for tp, ps in trade_pair_to_positions.items() if len(ps) > 1}
            if duplicates:
                # Show which trade pairs have multiple positions and the breakdown
                duplicate_summary = ', '.join([f"{tp}({len(ps)})" for tp, ps in duplicates.items()])
                hotkey_short = hotkey[-8:] if len(hotkey) > 8 else hotkey
                bt.logging.warning(f"...{hotkey_short}: {len(duplicates)} trade pairs with multiple positions: {duplicate_summary}")
                bt.logging.warning(f"  Total: {len(positions)} positions (all will be preserved)")

            positions.sort(key=lambda p: p.close_ms if p.close_ms is not None else float('inf'))
            ValiBkpUtils.make_dir(ValiBkpUtils.get_miner_all_positions_dir(hotkey))
            for p_obj in positions:
                #bt.logging.info(f'creating position {p_obj}')
                # CRITICAL: Pass delete_open_position_if_exists=False to preserve ALL positions from backup
                # Without this, later closed positions would delete earlier open positions for same trade pair
                position_client.save_miner_position(p_obj, delete_open_position_if_exists=False)

            # Validate that the positions were written correctly
            disk_positions = position_client.get_positions_for_one_hotkey(hotkey)
            n_disk_positions = len(disk_positions)
            n_memory_positions = len(positions)

            # During restore, we save closed positions FIRST (due to sort order), then open positions.
            # Since closed positions are saved first, the deletion logic in save_miner_position doesn't
            # find any existing open positions to delete. Therefore, ALL positions are kept.
            # The sort order specifically prevents deletions during restore (see comment above sort).
            expected_disk_count = n_memory_positions

            if n_disk_positions != expected_disk_count:
                memory_p_uuids = set([p.position_uuid for p in positions])
                disk_p_uuids = set([p.position_uuid for p in disk_positions])
                missing_uuids = memory_p_uuids - disk_p_uuids
                extra_uuids = disk_p_uuids - memory_p_uuids

                bt.logging.error(f"UNEXPECTED position mismatch for hotkey {hotkey}:")
                bt.logging.error(f"  Expected: {expected_disk_count} positions")
                bt.logging.error(f"  Got: {n_disk_positions} positions")

                if missing_uuids:
                    bt.logging.error(f"  Missing {len(missing_uuids)} positions from disk:")
                    for uuid in list(missing_uuids)[:5]:  # Limit to first 5 for brevity
                        missing_pos = next((p for p in positions if p.position_uuid == uuid), None)
                        if missing_pos:
                            bt.logging.error(f"    - {uuid}: trade_pair={missing_pos.trade_pair.trade_pair_id}, "
                                           f"is_open={missing_pos.is_open_position}")

                if extra_uuids:
                    bt.logging.error(f"  Found {len(extra_uuids)} unexpected positions on disk:")
                    bt.logging.error(f"  POSSIBLE CAUSE: Position splitting may have occurred during save operations")
                    for uuid in list(extra_uuids)[:5]:  # Limit to first 5 for brevity
                        extra_pos = next((p for p in disk_positions if p.position_uuid == uuid), None)
                        if extra_pos:
                            bt.logging.error(f"    + {uuid}: trade_pair={extra_pos.trade_pair.trade_pair_id}, "
                                           f"is_open={extra_pos.is_open_position}, "
                                           f"open_ms={extra_pos.open_ms}, close_ms={extra_pos.close_ms}")
                            # Check if this looks like a split position (has fewer orders than original)
                            bt.logging.error(f"      orders: {len(extra_pos.orders)}")

                raise AssertionError(f"Unexpected position count: expected {expected_disk_count}, got {n_disk_positions}")

            # Log success (only reached if validation passed)
            if duplicates:
                hotkey_short = hotkey[-8:] if len(hotkey) > 8 else hotkey
                bt.logging.info(f"  ✓ ...{hotkey_short}: Saved {n_memory_positions} positions (with overlaps)")

            total_saved += n_memory_positions

        # Log final global statistics and validate - fail fast on mismatch
        bt.logging.info(f"=" * 80)
        bt.logging.info(f"POSITION RESTORE COMPLETE:")
        bt.logging.info(f"  Expected to save: {total_positions_in_backup} positions")
        bt.logging.info(f"  Actually saved: {total_saved} positions")
        if total_saved == total_positions_in_backup:
            bt.logging.success(f"  ✓ All positions successfully restored!")
        else:
            bt.logging.error(f"  ✗ Mismatch: {total_positions_in_backup - total_saved} positions missing")
            raise AssertionError(
                f"GLOBAL POSITION COUNT MISMATCH: Expected {total_positions_in_backup} positions, "
                f"but saved {total_saved}. Missing {total_positions_in_backup - total_saved} positions."
            )
        bt.logging.info(f"=" * 80)

        bt.logging.info(f"regenerating {len(data['eliminations'])} eliminations")
        elimination_client.write_eliminations_to_disk(data['eliminations'])

        perf_ledgers = data.get('perf_ledgers', {})
        bt.logging.info(f"regenerating {len(perf_ledgers)} perf ledgers")
        perf_ledger_client.save_perf_ledgers(perf_ledgers)

        ## Now sync challenge period with the disk
        challengeperiod = data.get('challengeperiod', {})
        challengeperiod_client.sync_challenge_period_data(challengeperiod)

        ## Sync miner account sizes with the disk
        miner_account_sizes = data.get('miner_account_sizes', {})
        if miner_account_sizes:
            bt.logging.info(f"syncing {len(miner_account_sizes)} miner account size records")
            contract_client.sync_miner_account_sizes_data(miner_account_sizes)
        else:
            bt.logging.info("No miner account sizes found in backup data")

        challengeperiod_client._write_challengeperiod_from_memory_to_disk()

        limit_orders = data.get('limit_orders', {})
        limit_order_client.sync_limit_orders(limit_orders)

        ## Restore asset selections
        asset_selections_data = data.get('asset_selections', {})
        if asset_selections_data:
            bt.logging.info(f"syncing {len(asset_selections_data)} miner asset selection records")
            asset_selection_client.sync_miner_asset_selection_data(asset_selections_data)
        else:
            bt.logging.info("No asset selections found in backup data")

        ## Restore entity data
        entities_data = data.get('entities', {})
        if entities_data:
            bt.logging.info(f"syncing {len(entities_data)} entity records")
            entity_client.sync_entity_data(entities_data)
            bt.logging.success(f"✓ Restored {len(entities_data)} entities")
        else:
            bt.logging.info("No entity data found in backup data")

        bt.logging.success("✓ RESTORE COMPLETED SUCCESSFULLY - All data validated and saved")

    finally:
        # Always shutdown servers and clients, even if restore fails
        # This prevents the script from hanging after completion
        shutdown_all_servers_and_clients()

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

    try:
        regenerate_miner_positions(perform_backup, ignore_timestamp_checks=True)
        bt.logging.success(f"regeneration complete in {time.time() - t0:.2f} seconds")
    except Exception as e:
        bt.logging.error(f"RESTORE FAILED: {e}")
        bt.logging.error(traceback.format_exc())
        raise  # Re-raise to exit with error code
