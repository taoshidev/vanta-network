"""
Migrate miner_account_sizes.json from legacy list format to the new dict format.

The legacy format stored each account as a list of CollateralRecord dicts with
account-level fields stapled onto the last entry. The new format stores each account
as a single dict with collateral_records as a top-level list.

After this migration runs, the legacy parsing code can be deleted from prod.

Usage:
    python runnable/migrations/migrate_account_checkpoint_format.py
    python runnable/migrations/migrate_account_checkpoint_format.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

from vali_objects.miner_account.miner_account import (
    CollateralRecord, DailyOpenSnapshot, MinerAccount,
)
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import TradePairCategory
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


def _parse_legacy_account(hotkey: str, account_data: list) -> MinerAccount | None:
    """Parse a single account from the legacy list format.

    Legacy format: list of CollateralRecord dicts with account-level fields
    stapled onto the last entry.
    """
    if not isinstance(account_data, list):
        return None

    collateral_records = []

    if account_data and isinstance(account_data[-1], dict):
        last = account_data[-1]
        total_realized_pnl = last.get("total_realized_pnl")
        capital_used = last.get("capital_used")
        total_borrowed = last.get("total_borrowed_amount", 0.0)
        total_fees_paid = last.get("total_fees_paid", 0.0)
        total_dividend_income = last.get("total_dividend_income", 0.0)
        miner_bucket_str = last.get("miner_bucket")
        asset_class_str = last.get("asset_class")
        hl_address = last.get("hl_address")
        max_return = last.get("max_return", 1.0)
        unrealized_pnl = last.get("unrealized_pnl", 0.0)
        capital_used_by_class_raw = last.get("capital_used_by_class", {})
        daily_open_snapshot_raw = last.get("daily_open_snapshot")
    else:
        total_realized_pnl = None
        capital_used = None
        total_borrowed = 0.0
        total_fees_paid = 0.0
        total_dividend_income = 0.0
        miner_bucket_str = None
        asset_class_str = None
        hl_address = None
        max_return = 1.0
        unrealized_pnl = 0.0
        capital_used_by_class_raw = {}
        daily_open_snapshot_raw = None

    capital_used_by_class: dict[TradePairCategory, float] = {}
    for cat_str, amount in (capital_used_by_class_raw or {}).items():
        try:
            capital_used_by_class[TradePairCategory(cat_str)] = float(amount)
        except ValueError:
            print(f"  Warning: unknown TradePairCategory '{cat_str}' for {hotkey}; skipping")

    for r in account_data:
        if isinstance(r, dict) and "account_size" in r and "update_time_ms" in r:
            collateral_records.append(CollateralRecord(
                r["account_size"],
                r.get("account_size_theta", 0),
                r["update_time_ms"],
            ))

    return MinerAccount(
        miner_hotkey=hotkey,
        total_realized_pnl=total_realized_pnl if total_realized_pnl is not None else 0.0,
        capital_used=capital_used if capital_used is not None else 0.0,
        total_borrowed_amount=total_borrowed,
        total_fees_paid=total_fees_paid,
        total_dividend_income=total_dividend_income,
        asset_class=MinerAssetClass(asset_class_str) if asset_class_str else None,
        collateral_records=collateral_records,
        miner_bucket=MinerBucket(miner_bucket_str) if miner_bucket_str else None,
        hl_address=hl_address,
        max_return=max_return,
        unrealized_pnl=unrealized_pnl,
        capital_used_by_class=capital_used_by_class,
        daily_open_snapshot=DailyOpenSnapshot.from_dict(daily_open_snapshot_raw) if daily_open_snapshot_raw else None,
    )


def _parse_legacy_checkpoint(raw: dict) -> dict[str, MinerAccount]:
    parsed = {}
    for hotkey, entry in raw.items():
        if isinstance(entry, list):
            account = _parse_legacy_account(hotkey, entry)
            if account is not None:
                parsed[hotkey] = account
    return parsed


def _migrate(dry_run: bool = False, running_unit_tests: bool = False) -> bool:
    prefix = "[DRY RUN] " if dry_run else ""

    accounts_file = ValiBkpUtils.get_miner_account_sizes_file_location(
        running_unit_tests=running_unit_tests
    )
    raw = ValiUtils.get_vali_json_file_dict(accounts_file)
    raw.pop("_cost_per_theta", None)

    if not raw:
        print("miner_account_sizes.json not found or empty — nothing to migrate.")
        return True

    legacy = sum(1 for v in raw.values() if isinstance(v, list))

    print(f"Found {len(raw)} accounts, {legacy} in legacy list format.")

    if legacy == 0:
        print("No legacy accounts found — nothing to do.")
        return True

    parsed = _parse_legacy_checkpoint(raw)
    new_data = {hotkey: account.to_dict(include_computed=False) for hotkey, account in parsed.items()}

    print(f"{prefix}Rewriting {accounts_file} with {len(new_data)} accounts in new format.")

    if not dry_run:
        ValiBkpUtils.write_file(accounts_file, new_data)
        print("Migration complete.")
    else:
        print("Dry run complete — no files written.")

    return True


def main() -> bool:
    return _migrate(dry_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate miner_account_sizes.json to new checkpoint format.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    args = parser.parse_args()
    sys.exit(0 if _migrate(dry_run=args.dry_run) else 1)
