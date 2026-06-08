"""
Shared helpers for migration scripts under runnable/migrations/.

Migration scripts run once each, alphabetically, right before the validator
restarts after an update (see runnable/migrations/README.md). They run as
standalone Python processes — no RPC servers are available — so they must
operate directly on on-disk state.

Helpers exposed:

  MigrationUtils.load_all_positions  — walks validation/miners/<hotkey>/<pair>/<status>
                                       and returns hotkey -> [Position] for every
                                       open and closed position on disk.
  MigrationUtils.save_position       — persists a single Position back to disk
                                       under its OPEN or CLOSED partition dir.

This file lives at runnable/migration_utils.py (NOT under runnable/migrations/)
because run_migrations.py scans runnable/migrations/ for .py files and treats
any without a main() as a failed migration. A helper module belongs outside
that scan.
"""

from collections import defaultdict
import os
from typing import Dict, List

import bittensor as bt

from vali_objects.enums.misc import OrderStatus
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.position import Position


class MigrationUtils:
    """Static helpers for migration scripts. Do NOT instantiate."""

    @staticmethod
    def load_all_positions(running_unit_tests: bool = False) -> Dict[str, List[Position]]:
        """Load every open and closed Position on disk, grouped by hotkey.

        Walks the partitioned position layout used by PositionManager:
            validation/miners/<hotkey>/<trade_pair_id>/{open,closed}/<position_uuid>

        Returns:
            { hotkey: [Position, ...], ... }  — empty if the base dir is missing.
        """
        all_positions: Dict[str, List[Position]] = defaultdict(list)

        base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
        if not os.path.exists(base_dir):
            bt.logging.error(f"Positions directory not found: {base_dir}")
            return all_positions

        for hotkey in os.listdir(base_dir):
            hotkey_path = os.path.join(base_dir, hotkey)
            if not os.path.isdir(hotkey_path):
                continue

            for trade_pair in TradePair:
                for status in (OrderStatus.OPEN, OrderStatus.CLOSED):
                    dir_path = ValiBkpUtils.get_partitioned_miner_positions_dir(
                        hotkey,
                        trade_pair.trade_pair_id,
                        order_status=status,
                        running_unit_tests=running_unit_tests,
                    )
                    if not os.path.exists(dir_path):
                        continue
                    for filename in os.listdir(dir_path):
                        filepath = os.path.join(dir_path, filename)
                        try:
                            file_string = ValiBkpUtils.get_file(filepath)
                            position = Position.model_validate_json(file_string)
                            all_positions[hotkey].append(position)
                        except Exception as e:
                            bt.logging.warning(f"Failed to load {filepath}: {e}")

        total = sum(len(v) for v in all_positions.values())
        print(f"Loaded {total} positions from {len(all_positions)} hotkeys")
        return all_positions

    @staticmethod
    def save_position(position: Position, running_unit_tests: bool = False) -> None:
        """Persist a single Position back to disk under its correct OPEN/CLOSED dir."""
        miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
            position.miner_hotkey,
            position.trade_pair.trade_pair_id,
            order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
            running_unit_tests=running_unit_tests,
        )
        ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)
