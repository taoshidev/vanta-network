"""
Migration script to backfill max_return (HWM) and rebuild regular miner accounts.

For regular hotkeys:
- Rebuilds total_realized_pnl and capital_used from ALL positions (no cutoff)
- Reads max_return from perf_ledgers.json.gz portfolio ledger

For synthetic hotkeys:
- Keeps existing account data unchanged
- Computes max_return as balance / account_size

Usage:
    python runnable/migrations/_migrate_max_return.py [--dry-run]

Options:
    --dry-run, -n    Test migration without modifying files
"""

import gzip
import json
import os
import sys

from vali_objects.vali_config import ValiConfig, TradePair
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

DRY_RUN = False
for arg in sys.argv[1:]:
    if arg in ['--dry-run', '-n']:
        DRY_RUN = True
        print("*** DRY RUN MODE - No files will be modified ***\n")

ACCOUNTS_FILE = ValiConfig.BASE_DIR + "/validation/miner_account_sizes.json"
PERF_LEDGERS_FILE = ValiConfig.BASE_DIR + "/validation/perf_ledgers.json.gz"
TP_ID_PORTFOLIO = "portfolio"


def is_synthetic_hotkey(hotkey: str) -> bool:
    if '_' not in hotkey:
        return False
    suffix = hotkey.rsplit('_', 1)[-1]
    return suffix.isdigit()


def load_positions(status: str) -> dict[str, list[Position]]:
    """Load positions from disk. status is 'open' or 'closed'."""
    all_positions: dict[str, list[Position]] = {}
    base_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)

    if not os.path.exists(base_dir):
        print(f"Positions directory not found: {base_dir}")
        return all_positions

    for hotkey in os.listdir(base_dir):
        hotkey_path = os.path.join(base_dir, hotkey)
        if not os.path.isdir(hotkey_path):
            continue

        for trade_pair in TradePair:
            pos_dir = os.path.join(
                base_dir, hotkey, "positions", trade_pair.trade_pair_id, status
            )
            if not os.path.exists(pos_dir):
                continue

            for filename in os.listdir(pos_dir):
                filepath = os.path.join(pos_dir, filename)
                try:
                    file_string = ValiBkpUtils.get_file(filepath)
                    position = Position.model_validate_json(file_string)
                    if hotkey not in all_positions:
                        all_positions[hotkey] = []
                    all_positions[hotkey].append(position)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")

    total = sum(len(positions) for positions in all_positions.values())
    print(f"Loaded {total} {status} positions from {len(all_positions)} hotkeys")
    return all_positions


def load_perf_ledger_max_returns() -> dict[str, float]:
    result = {}
    try:
        with gzip.open(PERF_LEDGERS_FILE, 'rt') as f:
            data = json.load(f)

        for hotkey, bundle in data.items():
            if not isinstance(bundle, dict):
                continue
            portfolio = bundle.get(TP_ID_PORTFOLIO)
            if not isinstance(portfolio, dict):
                continue
            max_return = portfolio.get('max_return', 1.0)
            if max_return > 1.0:
                result[hotkey] = max_return

        print(f"Loaded perf ledger max_return for {len(result)} hotkeys")
    except Exception as e:
        print(f"Failed to load perf ledgers: {e}")

    return result


def main() -> bool:
    print(f"Reading {ACCOUNTS_FILE}...")
    with open(ACCOUNTS_FILE, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} miner accounts")

    print(f"\nReading {PERF_LEDGERS_FILE}...")
    perf_max_returns = load_perf_ledger_max_returns()

    print("\nLoading positions for regular miners...")
    open_positions = load_positions("open")
    closed_positions = load_positions("closed")

    updated = 0
    skipped = 0
    results = []

    for hotkey, records in data.items():
        if not isinstance(records, list) or not records:
            skipped += 1
            continue

        last_record = records[-1]
        if not isinstance(last_record, dict):
            skipped += 1
            continue

        account_size = last_record.get('account_size', 0)
        if account_size <= 0:
            skipped += 1
            continue

        if is_synthetic_hotkey(hotkey):
            # Synthetic: keep existing account, compute max_return from balance
            total_realized_pnl = last_record.get('total_realized_pnl', 0)
            total_interest_paid = last_record.get('total_interest_paid', 0)
            total_fees_paid = last_record.get('total_fees_paid', 0)
            balance = account_size + total_realized_pnl - total_interest_paid - total_fees_paid

            max_return = max(1.0, balance / account_size)
        else:
            # Regular: rebuild account from ALL positions, max_return from perf ledger
            hotkey_open = open_positions.get(hotkey, [])
            hotkey_closed = closed_positions.get(hotkey, [])

            total_realized_pnl = 0.0
            capital_used = 0.0
            for position in closed_positions.get(hotkey, []):
                total_realized_pnl += position.realized_pnl
            for position in open_positions.get(hotkey, []):
                total_realized_pnl += position.realized_pnl  # partial closes
                capital_used += abs(position.net_value)

            total_interest_paid = last_record.get('total_interest_paid', 0)
            total_fees_paid = last_record.get('total_fees_paid', 0)
            balance = account_size + total_realized_pnl - total_interest_paid - total_fees_paid

            if not DRY_RUN:
                last_record['total_realized_pnl'] = total_realized_pnl
                last_record['capital_used'] = capital_used

            max_return = perf_max_returns.get(hotkey, 1.0)

        changed = False
        if max_return > 1.0:
            if not DRY_RUN:
                last_record['max_return'] = max_return
            changed = True

        if not is_synthetic_hotkey(hotkey):
            # Always count regular miners as updated (account rebuilt)
            changed = True

        if changed:
            label = "SYNTH" if is_synthetic_hotkey(hotkey) else "REG"
            results.append((hotkey, max_return, balance, label))
            updated += 1
        else:
            skipped += 1

    # Print results
    results.sort(key=lambda x: x[1], reverse=True)
    for hotkey, max_val, bal, label in results:
        print(f"[{hotkey}] [{label}] max_return: {max_val:.6f}, balance: ${bal:,.2f}")

    # Save
    if not DRY_RUN and updated > 0:
        with open(ACCOUNTS_FILE, 'w') as f:
            json.dump(data, f)
        print(f"\nWrote {ACCOUNTS_FILE}")

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Updated:  {updated}")
    print(f"Skipped:  {skipped}")

    if DRY_RUN:
        print("\n[DRY RUN] No files were modified")
    else:
        print("\nMigration completed.")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
