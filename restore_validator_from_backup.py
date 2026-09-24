"""
Restore validator state from a validator checkpoint (validator_checkpoint.json[.gz]).

The restore overwrites everything the checkpoint covers: positions, archived positions, limit
orders, eliminations, perf ledgers, challenge period, miner accounts, asset selections, entities
and weekly seals. Each service's manager is created in-process in LOCAL mode with no RPC servers,
so a running validator's ports are never touched. Never run this against the mothership.

Always writes to validation/ and backs it up first. Stop the validator before running.
"""
import json
import gzip
import os
import shutil
import time

import traceback
from datetime import datetime

from time_util.time_util import TimeUtil
from vali_objects.vali_config import RPCConnectionMode
from vali_objects.vali_dataclasses.position import Position
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination.elimination_manager import EliminationManager
from vali_objects.utils.limit_order.limit_order_manager import LimitOrderManager
from vali_objects.miner_account.miner_account_manager import MinerAccountManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.asset_selection.asset_selection_manager import AssetSelectionManager
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_manager import PerfLedgerManager
from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import WeeklySealLedger
from entity_management.entity_manager import EntityManager
import logging
from shared_objects.log import logger

# Managers run in test mode (no wallets, secrets or broadcasts); ValiBkpUtils.use_production_paths
# still points their files at validation/ rather than tests/validation/.
RUNNING_UNIT_TESTS = True
ValiBkpUtils.use_production_paths = True
CONNECTION_MODE = RPCConnectionMode.LOCAL

# Per-hotkey directories under miners/ that the checkpoint fully describes
MINER_SUBDIRS_TO_OVERWRITE = ("positions", "limit_orders", "archived_positions")


def backup_validation_directory():
    dir_to_backup = ValiBkpUtils.get_vali_dir(running_unit_tests=RUNNING_UNIT_TESTS)
    if not os.path.exists(dir_to_backup):
        logger.info(f"Nothing to back up at {dir_to_backup}")
        return
    # Write to the backup location. Make sure it is a function of the date. No dashes. Days and months get 2 digits.
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_location = ValiBkpUtils.get_vali_bkp_dir() + date_str + '/'
    # Sync directory to the backup location using python shutil
    shutil.copytree(dir_to_backup, backup_location)
    logger.info(f"backed up {dir_to_backup} to {backup_location}")


def load_checkpoint() -> dict:
    # Check for compressed version first, then fallback to uncompressed for backward compatibility
    compressed_path = ValiBkpUtils.get_validator_checkpoint_path()
    uncompressed_path = ValiBkpUtils.get_backup_file_path()

    # Load checkpoint file - fail fast if file is missing or corrupt
    if os.path.exists(compressed_path):
        logger.info(f"Found compressed checkpoint file: {compressed_path}")
        try:
            with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if "Not a gzipped file" in str(e):
                logger.error(f"File {compressed_path} has .gz extension but contains uncompressed data.")
                logger.error("Solution: Remove the .gz extension and rename to validator_checkpoint.json")
            raise RuntimeError(f"Failed to load compressed checkpoint: {e}") from e
    elif os.path.exists(uncompressed_path):
        logger.info(f"Found uncompressed checkpoint file: {uncompressed_path}")
        try:
            data = json.loads(ValiBkpUtils.get_file(uncompressed_path))
            if isinstance(data, str):
                data = json.loads(data)
            return data
        except Exception as e:
            if "invalid start byte" in str(e) or "'utf-8' codec can't decode" in str(e):
                logger.error(f"File {uncompressed_path} appears to contain compressed data but lacks .gz extension.")
                logger.error("Solution: Add .gz extension and rename to validator_checkpoint.json.gz")
            raise RuntimeError(f"Failed to load uncompressed checkpoint: {e}") from e
    else:
        raise FileNotFoundError(f"No checkpoint file found at {uncompressed_path} or {compressed_path}")


def clear_restore_targets():
    """
    Delete the state the checkpoint replaces, so each manager starts empty and the result matches
    the checkpoint exactly. Several syncs merge into existing state (challenge period, entities,
    weekly seals), so clearing first is what makes the restore an overwrite.
    """
    miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=RUNNING_UNIT_TESTS)
    if os.path.exists(miner_dir):
        for hotkey in ValiBkpUtils.get_directories_in_dir(miner_dir):
            for subdir in MINER_SUBDIRS_TO_OVERWRITE:
                shutil.rmtree(os.path.join(miner_dir, hotkey, subdir), ignore_errors=True)

    rut = RUNNING_UNIT_TESTS
    for file_path in (
        ValiBkpUtils.get_eliminations_dir(rut),
        ValiBkpUtils.get_perf_ledgers_path(rut),
        ValiBkpUtils.get_perf_ledgers_path_pkl(rut),
        ValiBkpUtils.get_perf_ledgers_path_legacy(rut),
        ValiBkpUtils.get_frozen_perf_ledgers_path(rut),
        ValiBkpUtils.get_challengeperiod_file_location(rut),
        ValiBkpUtils.get_miner_account_sizes_file_location(rut),
        ValiBkpUtils.get_asset_selections_file_location(rut),
        ValiBkpUtils.get_entity_file_location(rut),
        ValiBkpUtils.get_weekly_seal_ledger_file_location(rut),
    ):
        if os.path.exists(file_path):
            os.remove(file_path)
    logger.info(f"Cleared restore targets under {ValiBkpUtils.get_vali_dir(running_unit_tests=rut)}")


def write_positions(hotkey_to_dashboard: dict, archived: bool) -> tuple[int, int]:
    """
    Write every position in the checkpoint straight to disk, one file per position, in the same
    layout PositionManager uses. Returns (n_written, n_skipped).
    """
    n_written = 0
    n_skipped = 0
    for hotkey, dashboard in hotkey_to_dashboard.items():
        if archived:
            base_dir = ValiBkpUtils.get_miner_archived_positions_dir(hotkey, running_unit_tests=RUNNING_UNIT_TESTS)
        else:
            base_dir = ValiBkpUtils.get_miner_all_positions_dir(hotkey, running_unit_tests=RUNNING_UNIT_TESTS)
        for position_dict in dashboard['positions']:
            try:
                position = Position(**position_dict)
            except Exception as e:
                tp = position_dict.get('trade_pair')
                tp_id = tp[0] if isinstance(tp, list) else tp
                logger.warning(f"Skipping position for hotkey {hotkey[-8:]} trade_pair={tp_id}: {e}")
                n_skipped += 1
                continue
            status_dir = "open" if position.is_open_position else "closed"
            file_path = os.path.join(base_dir, position.trade_pair.trade_pair_id, status_dir, position.position_uuid)
            ValiBkpUtils.write_file(file_path, position)
            n_written += 1
    return n_written, n_skipped


def regenerate_miner_positions():
    data = load_checkpoint()

    logger.info("Found validator backup file with the following attributes:")
    # Log every key and value pair in the data, with sizes for dicts and lists
    for key, value in data.items():
        if isinstance(value, dict) or isinstance(value, list):
            logger.info(f"    {key}: {len(value)} entries")
        else:
            logger.info(f"    {key}: {value}")
    logger.info(f"    backup_creation_time: {TimeUtil.millis_to_formatted_date_str(data['created_timestamp_ms'])}")

    backup_validation_directory()

    clear_restore_targets()

    # Positions and archived positions: written directly, no manager needed
    total_in_backup = sum(len(d['positions']) for d in data['positions'].values())
    n_written, n_skipped = write_positions(data['positions'], archived=False)
    logger.info(f"Restored {n_written}/{total_in_backup} positions for {len(data['positions'])} hotkeys")

    archived_positions = data.get('archived_positions', {})
    total_archived_in_backup = sum(len(d['positions']) for d in archived_positions.values())
    n_archived_written, n_archived_skipped = write_positions(archived_positions, archived=True)
    logger.info(f"Restored {n_archived_written}/{total_archived_in_backup} archived positions")

    if n_skipped or n_archived_skipped:
        logger.warning(f"Skipped {n_skipped} positions and {n_archived_skipped} archived positions "
                       f"(unresolvable dynamic trade pairs)")
    if n_written + n_skipped != total_in_backup or n_archived_written + n_archived_skipped != total_archived_in_backup:
        raise AssertionError("Position count mismatch between checkpoint and restored files")

    # Managers load from the now-empty targets, then take the checkpoint's data
    kwargs = dict(running_unit_tests=RUNNING_UNIT_TESTS, connection_mode=CONNECTION_MODE)

    eliminations = data['eliminations']
    logger.info(f"regenerating {len(eliminations)} eliminations")
    EliminationManager(**kwargs).write_eliminations_to_disk(eliminations)

    perf_ledger_manager = PerfLedgerManager(enable_rss=False, **kwargs)
    perf_ledgers = data.get('perf_ledgers', {})
    logger.info(f"regenerating {len(perf_ledgers)} perf ledgers")
    perf_ledger_manager.save_perf_ledgers(perf_ledgers)
    frozen_perf_ledgers = data.get('frozen_perf_ledgers', {})
    logger.info(f"regenerating {len(frozen_perf_ledgers)} frozen perf ledgers")
    perf_ledger_manager.sync_frozen_ledgers(frozen_perf_ledgers)

    challengeperiod = data.get('challengeperiod', {})
    logger.info(f"syncing {len(challengeperiod)} challenge period records")
    ChallengePeriodManager(**kwargs).sync_challenge_period_data(challengeperiod)

    # Asset selections before miner accounts: the account sync reads asset selections from disk
    asset_selections = data.get('asset_selections', {})
    logger.info(f"syncing {len(asset_selections)} miner asset selection records")
    AssetSelectionManager(**kwargs).sync_miner_asset_selection_data(asset_selections)

    miner_account_sizes = data.get('miner_account_sizes', {})
    logger.info(f"syncing {len(miner_account_sizes)} miner account size records")
    MinerAccountManager(**kwargs).sync_miner_account_sizes_data(miner_account_sizes)

    limit_orders = data.get('limit_orders', {})
    logger.info(f"syncing limit orders for {len(limit_orders)} trade pairs")
    LimitOrderManager(serve=False, **kwargs).sync_limit_orders(limit_orders)

    entities = data.get('entities', {})
    logger.info(f"syncing {len(entities)} entity records")
    EntityManager(**kwargs).sync_entity_data(entities)

    weekly_seals = data.get('weekly_seals', {})
    logger.info("syncing weekly seal records")
    WeeklySealLedger(running_unit_tests=RUNNING_UNIT_TESTS).sync_from_checkpoint(weekly_seals)

    logger.info("== RESTORE COMPLETED SUCCESSFULLY ==")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    t0 = time.time()
    logger.warning(f"Restoring into {ValiBkpUtils.get_vali_dir(running_unit_tests=RUNNING_UNIT_TESTS)}")

    try:
        regenerate_miner_positions()
        logger.info(f"regeneration complete in {time.time() - t0:.2f} seconds")
    except Exception as e:
        logger.error(f"RESTORE FAILED: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to exit with error code
