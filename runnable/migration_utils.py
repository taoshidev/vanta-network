"""
Shared helpers for migration scripts under runnable/migrations/.

Migration scripts run once each, alphabetically, right before the validator
restarts after an update (see runnable/migrations/README.md). They run as
standalone Python processes — no RPC servers are available — so they must
operate directly on on-disk state.

Helpers exposed:

  MigrationUtils.load_all_positions    — walks validation/miners/<hotkey>/<pair>/<status>
                                         and returns hotkey -> [Position] for every
                                         open and closed position on disk.
  MigrationUtils.save_position         — persists a single Position back to disk
                                         under its OPEN or CLOSED partition dir.
  MigrationUtils.load_asset_selections — returns dict from validation/asset_selections.json
  MigrationUtils.save_asset_selections — writes dict back to asset_selections.json
  MigrationUtils.load_miner_accounts   — returns dict from validation/miner_account_sizes.json
  MigrationUtils.save_miner_accounts   — writes dict back to miner_account_sizes.json
  MigrationUtils.load_entities         — returns dict from validation/entities.json
  MigrationUtils.save_entities         — writes dict back to entities.json

This file lives at runnable/migration_utils.py (NOT under runnable/migrations/)
because run_migrations.py scans runnable/migrations/ for .py files and treats
any without a main() as a failed migration. A helper module belongs outside
that scan.
"""

from collections import defaultdict
import json
import os
from typing import Dict, List


from vali_objects.enums.misc import OrderStatus
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.position import Position
from shared_objects.log import logger


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
            logger.error(f"Positions directory not found: {base_dir}")
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
                            logger.warning(f"Failed to load {filepath}: {e}")

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

    # ------------------------------------------------------------------ #
    # JSON file loaders / savers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_json(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON at {path}; returning empty dict.")
                return {}

    @staticmethod
    def _save_json(path: str, data: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_asset_selections(running_unit_tests: bool = False) -> dict:
        """Load validation/asset_selections.json. Returns {} if missing."""
        path = ValiBkpUtils.get_asset_selections_file_location(running_unit_tests=running_unit_tests)
        return MigrationUtils._load_json(path)

    @staticmethod
    def save_asset_selections(data: dict, running_unit_tests: bool = False) -> None:
        """Write validation/asset_selections.json."""
        path = ValiBkpUtils.get_asset_selections_file_location(running_unit_tests=running_unit_tests)
        MigrationUtils._save_json(path, data)

    @staticmethod
    def load_miner_accounts(running_unit_tests: bool = False):
        """Load validation/miner_account_sizes.json as typed MinerAccount objects.

        Uses MinerAccountManager._parse_accounts_dict so the asset_class field
        comes from asset_selections.json (the source of truth) — matches the
        validator boot path. Returns {} if the on-disk file is missing.
        """
        from vali_objects.miner_account.miner_account_manager import MinerAccountManager

        path = ValiBkpUtils.get_miner_account_sizes_file_location(running_unit_tests=running_unit_tests)
        raw = MigrationUtils._load_json(path)
        raw.pop("_cost_per_theta", None)  # legacy top-level key; never written back
        selections = MigrationUtils.load_asset_selections(running_unit_tests=running_unit_tests)
        return MinerAccountManager._parse_accounts_dict(raw, selections)

    @staticmethod
    def save_miner_accounts(accounts, running_unit_tests: bool = False) -> None:
        """Persist Dict[hotkey, MinerAccount] back to miner_account_sizes.json.

        Mirrors MinerAccountManager.accounts_dict serialization: each hotkey's
        value is a list of CollateralRecord dicts followed by the account
        summary dict from MinerAccount.to_dict.
        """
        data: dict = {}
        for hotkey, account in accounts.items():
            records_list = [vars(record).copy() for record in account.collateral_records]
            records_list.append(account.to_dict(include_collateral_records=False))
            data[hotkey] = records_list

        path = ValiBkpUtils.get_miner_account_sizes_file_location(running_unit_tests=running_unit_tests)
        ValiBkpUtils.write_file(path, data)

    @staticmethod
    def load_entities(running_unit_tests: bool = False):
        """Load validation/entities.json as Dict[hotkey, EntityData] (typed)."""
        from entity_management.entity_manager import EntityManager

        path = ValiBkpUtils.get_entity_file_location(running_unit_tests=running_unit_tests)
        raw = MigrationUtils._load_json(path)
        if not raw:
            return {}
        return EntityManager.parse_checkpoint_dict(raw)

    @staticmethod
    def save_entities(entities, running_unit_tests: bool = False) -> None:
        """Persist Dict[hotkey, EntityData] back to entities.json via model_dump."""
        data = {hotkey: entity.model_dump() for hotkey, entity in entities.items()}
        path = ValiBkpUtils.get_entity_file_location(running_unit_tests=running_unit_tests)
        ValiBkpUtils.write_file(path, data)
