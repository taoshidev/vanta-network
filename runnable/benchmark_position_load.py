#!/usr/bin/env python3
"""
Benchmark the validator's position disk load from a cold cache: the original serial loader vs the
current parallel one (PositionManager._load_positions_from_disk), both loading live and archived
positions from the production miner directory.

Before each load the position files are evicted from the OS page cache, so each run measures a
real boot rather than a re-read from RAM:
  - /proc/sys/vm/drop_caches when writable (root on Linux; also drops dentry/inode caches)
  - otherwise posix_fadvise(DONTNEED) per file (Linux, no root; dentry/inode caches stay warm)

Usage:
    sudo -E venv/bin/python runnable/benchmark_position_load.py
"""

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shared_objects.log import logger
from time_util.time_util import StageTimer
from vali_objects.position_management.position_manager import DISK_LOAD_READ_WORKERS, PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.position import Position

DROP_CACHES_PATH = "/proc/sys/vm/drop_caches"


def evict(miner_dir: str) -> str:
    """Evict the position files from the page cache. Returns the method used."""
    os.sync()
    if os.access(DROP_CACHES_PATH, os.W_OK):
        with open(DROP_CACHES_PATH, "w") as f:
            f.write("3\n")
        return "drop_caches"
    if not hasattr(os, "posix_fadvise"):
        raise SystemExit("Can't evict the page cache here (no drop_caches access, no posix_fadvise). "
                         "Run on the Linux validator host, ideally as root.")
    for root, _, files in os.walk(miner_dir):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    return "fadvise"


def bare_position_manager() -> PositionManager:
    """A PositionManager with only the state the disk load touches (no RPC clients or daemons)."""
    pm = PositionManager.__new__(PositionManager)
    pm.load_from_disk = True
    pm.running_unit_tests = False
    pm.is_backtesting = False
    pm.hotkey_to_positions = {}
    pm.hotkey_to_open_positions = {}
    pm.hotkey_to_archived_positions = {}
    return pm


def load_serial(pm: PositionManager, timer: StageTimer):
    """The original loader (main before the parallel change), with the same stage laps."""
    base_dir = Path(ValiBkpUtils.get_miner_dir())

    for hotkey_dir in base_dir.iterdir():
        if not hotkey_dir.is_dir():
            continue
        hotkey = hotkey_dir.name
        all_files = ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_miner_all_positions_dir(hotkey))
        positions_dict = {}
        for position_file in all_files:
            try:
                position = Position.model_validate_json(ValiBkpUtils.get_file(position_file))
                positions_dict[position.position_uuid] = position
            except Exception as e:
                logger.error(f"Error loading position file {position_file} for {hotkey}: {e}")
        if positions_dict:
            pm.hotkey_to_positions[hotkey] = positions_dict
    timer.lap("positions")

    pm._rebuild_open_index()
    timer.lap("open_index")

    for hotkey_dir in base_dir.iterdir():
        if not hotkey_dir.is_dir():
            continue
        hotkey = hotkey_dir.name
        all_files = ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_miner_archived_positions_dir(hotkey))
        archived_dict = {}
        for position_file in all_files:
            try:
                position = Position.model_validate_json(ValiBkpUtils.get_file(position_file))
                archived_dict[position.position_uuid] = position
            except Exception as e:
                logger.error(f"Error loading archived position file {position_file} for {hotkey}: {e}")
        if archived_dict:
            pm.hotkey_to_archived_positions[hotkey] = archived_dict
    timer.lap("archived_positions")


def load_parallel(pm: PositionManager, timer: StageTimer):
    """The production loader. It logs its own per-stage timings."""
    pm._load_positions_from_disk()
    timer.lap("load")


def load_open_only(pm: PositionManager, timer: StageTimer):
    """Parallel reads of positions/<trade_pair>/open/ only: no closed or archived positions."""
    base_dir = Path(ValiBkpUtils.get_miner_dir())
    hotkeys = [hotkey_dir.name for hotkey_dir in base_dir.iterdir() if hotkey_dir.is_dir()]
    timer.lap("list_hotkeys")

    def read_hotkey(hotkey):
        files = []
        positions_base = ValiBkpUtils.get_miner_all_positions_dir(hotkey)
        if not os.path.isdir(positions_base):
            return files
        for trade_pair_dir in os.scandir(positions_base):
            open_dir = os.path.join(trade_pair_dir.path, "open")
            if not os.path.isdir(open_dir):
                continue
            for name in os.listdir(open_dir):
                position_file = os.path.join(open_dir, name)
                files.append((position_file, ValiBkpUtils.get_file(position_file)))
        return files

    with ThreadPoolExecutor(max_workers=DISK_LOAD_READ_WORKERS) as executor:
        for hotkey, files in zip(hotkeys, executor.map(read_hotkey, hotkeys)):
            positions_dict = {}
            for position_file, file_string in files:
                try:
                    position = Position.model_validate_json(file_string)
                    positions_dict[position.position_uuid] = position
                except Exception as e:
                    logger.error(f"Error loading position file {position_file} for {hotkey}: {e}")
            if positions_dict:
                pm.hotkey_to_positions[hotkey] = positions_dict
    timer.lap("open_positions")

    pm._rebuild_open_index()
    timer.lap("open_index")


def main():
    miner_dir = ValiBkpUtils.get_miner_dir()
    if not Path(miner_dir).is_dir():
        raise SystemExit(f"No miner directory at {miner_dir}")

    results = []
    for name, loader in (("serial", load_serial), ("parallel", load_parallel), ("open_only", load_open_only)):
        eviction = evict(miner_dir)
        gc.collect()
        pm = bare_position_manager()
        timer = StageTimer()
        loader(pm, timer)
        n_positions = sum(len(d) for d in pm.hotkey_to_positions.values())
        n_archived = sum(len(d) for d in pm.hotkey_to_archived_positions.values())
        results.append((name, n_positions, n_archived, timer.summary()))
        print(f"{name}: positions={n_positions} archived={n_archived} eviction={eviction} | {timer.summary()}",
              flush=True)
        del pm

    print()
    for name, n_positions, n_archived, summary in results:
        print(f"{name:<9} positions={n_positions} archived={n_archived} | {summary}")


if __name__ == "__main__":
    main()
