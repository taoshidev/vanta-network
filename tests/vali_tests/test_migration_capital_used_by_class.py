"""
Unit tests for runnable/migrations/migrate_capital_used_by_class.py.

The migration backfills MinerAccount.capital_used_by_class on disk from each
miner's open positions so per-class portfolio caps bind from the moment the
validator comes up after upgrade.

Tests stub out disk I/O (file read / write / position loader) and exercise
main() end-to-end on synthetic in-memory state.
"""

import os
import unittest
from unittest.mock import patch

from runnable.migrations import migrate_capital_used_by_class as mig
from vali_objects.vali_config import TradePair, TradePairCategory
from vali_objects.vali_dataclasses.position import Position


def _make_position(hotkey, trade_pair, net_value, closed=False):
    return Position(
        miner_hotkey=hotkey,
        position_uuid=f"{hotkey}_{trade_pair.trade_pair_id}_{net_value}",
        open_ms=0,
        trade_pair=trade_pair,
        net_value=net_value,
        is_closed_position=closed,
    )


def _make_account_record(capital_used=0.0, capital_used_by_class=None):
    """Minimal account-state record matching the shape MinerAccount.to_dict() produces.

    Note: production account-state records contain `account_size` but NOT
    `update_time_ms` (the latter is only on collateral records). The parser
    distinguishes them by the presence of `update_time_ms`, so adding it here
    would make the record look like a collateral record too.
    """
    record = {
        "account_size": 50_000,
        "capital_used": capital_used,
    }
    if capital_used_by_class is not None:
        record["capital_used_by_class"] = capital_used_by_class
    return record


class MigrationFixture:
    """Aggregates all the patchers a test needs into a single context manager."""

    def __init__(self, accounts_data, positions_by_hotkey, file_exists=True):
        self._accounts_data = accounts_data
        self._positions_by_hotkey = positions_by_hotkey
        self._file_exists = file_exists
        self.written = None  # captures (path, data) passed to ValiBkpUtils.write_file
        self._patches = []

    def __enter__(self):
        # File presence + read
        self._patches.append(patch.object(os.path, "exists", return_value=self._file_exists))
        self._patches.append(patch(
            "runnable.migrations.migrate_capital_used_by_class.ValiUtils.get_vali_json_file_dict",
            return_value=self._accounts_data,
        ))
        # Position loader
        self._patches.append(patch(
            "runnable.migrations.migrate_capital_used_by_class.MigrationUtils.load_all_positions",
            return_value=self._positions_by_hotkey,
        ))

        # Capture writes instead of touching disk
        def _capture_write(path, data):
            self.written = (path, data)

        self._patches.append(patch(
            "runnable.migrations.migrate_capital_used_by_class.ValiBkpUtils.write_file",
            side_effect=_capture_write,
        ))

        for p in self._patches:
            p.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        for p in reversed(self._patches):
            p.stop()


class TestMigrationNoFile(unittest.TestCase):

    def test_returns_true_when_accounts_file_missing(self):
        with MigrationFixture(accounts_data={}, positions_by_hotkey={}, file_exists=False) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        self.assertIsNone(fx.written)


class TestMigrationEmptyAccounts(unittest.TestCase):

    def test_empty_accounts_dict_is_noop(self):
        with MigrationFixture(accounts_data={}, positions_by_hotkey={}) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        self.assertIsNone(fx.written)


class TestMigrationCore(unittest.TestCase):

    def test_single_class_crypto_account_gets_per_class_populated(self):
        accounts = {
            "hk_crypto": [_make_account_record(capital_used=150.0)],
        }
        positions = {
            "hk_crypto": [
                _make_position("hk_crypto", TradePair.BTCUSD, 100.0),
                _make_position("hk_crypto", TradePair.ETHUSD, 50.0),
            ],
        }
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        self.assertIsNotNone(fx.written)
        path, written = fx.written
        self.assertEqual(
            written["hk_crypto"][-1]["capital_used_by_class"],
            {"crypto": 150.0},
        )

    def test_multi_class_account_per_class_breakdown_correct(self):
        accounts = {
            "hk_hl_all": [_make_account_record(capital_used=380.0)],
        }
        positions = {
            "hk_hl_all": [
                _make_position("hk_hl_all", TradePair.BTCUSDC, 100.0),     # CRYPTO
                _make_position("hk_hl_all", TradePair.ETHUSDC, 50.0),      # CRYPTO
                _make_position("hk_hl_all", TradePair.NVDAUSDC, 80.0),     # EQUITIES
                _make_position("hk_hl_all", TradePair.GOLDUSDC, 30.0),     # COMMODITIES
                _make_position("hk_hl_all", TradePair.SP500USDC, 120.0),   # INDICES
            ],
        }
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        _, written = fx.written
        per_class = written["hk_hl_all"][-1]["capital_used_by_class"]
        self.assertEqual(per_class["crypto"], 150.0)
        self.assertEqual(per_class["equities"], 80.0)
        self.assertEqual(per_class["commodities"], 30.0)
        self.assertEqual(per_class["indices"], 120.0)
        self.assertNotIn("forex", per_class)  # No FOREX positions

    def test_closed_positions_excluded(self):
        accounts = {
            "hk": [_make_account_record(capital_used=100.0)],
        }
        positions = {
            "hk": [
                _make_position("hk", TradePair.BTCUSD, 100.0),
                _make_position("hk", TradePair.ETHUSD, 200.0, closed=True),
            ],
        }
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            mig.main(dry_run=False)
        _, written = fx.written
        self.assertEqual(
            written["hk"][-1]["capital_used_by_class"],
            {"crypto": 100.0},
        )

    def test_account_with_no_open_positions_gets_empty_dict(self):
        accounts = {
            "hk_idle": [_make_account_record(capital_used=0.0)],
        }
        positions = {}  # no positions for this hotkey
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            mig.main(dry_run=False)
        _, written = fx.written
        self.assertEqual(written["hk_idle"][-1]["capital_used_by_class"], {})

    def test_account_with_only_closed_positions_gets_empty_dict(self):
        accounts = {
            "hk_done": [_make_account_record(capital_used=0.0)],
        }
        positions = {
            "hk_done": [_make_position("hk_done", TradePair.BTCUSD, 100.0, closed=True)],
        }
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            mig.main(dry_run=False)
        _, written = fx.written
        self.assertEqual(written["hk_done"][-1]["capital_used_by_class"], {})

    def test_negative_net_value_uses_absolute(self):
        """Short positions (negative net_value) contribute their absolute size."""
        accounts = {
            "hk_short": [_make_account_record(capital_used=100.0)],
        }
        positions = {
            "hk_short": [_make_position("hk_short", TradePair.BTCUSD, -100.0)],
        }
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            mig.main(dry_run=False)
        _, written = fx.written
        self.assertEqual(written["hk_short"][-1]["capital_used_by_class"], {"crypto": 100.0})


class TestMigrationIdempotency(unittest.TestCase):

    def test_already_correct_per_class_is_noop(self):
        """Re-running on a hotkey whose per-class breakdown already matches positions
        should detect the match and skip the write."""
        accounts = {
            "hk": [_make_account_record(
                capital_used=100.0,
                capital_used_by_class={"crypto": 100.0},
            )],
        }
        positions = {
            "hk": [_make_position("hk", TradePair.BTCUSD, 100.0)],
        }
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        # write_file is only called when updated > 0; idempotent run should not write
        self.assertIsNone(fx.written)

    def test_dry_run_never_writes(self):
        accounts = {
            "hk": [_make_account_record(capital_used=100.0)],
        }
        positions = {
            "hk": [_make_position("hk", TradePair.BTCUSD, 100.0)],
        }
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            result = mig.main(dry_run=True)
        self.assertTrue(result)
        self.assertIsNone(fx.written)


class TestMigrationEdgeRecords(unittest.TestCase):

    def test_record_list_not_a_list_is_skipped(self):
        accounts = {"hk_corrupt": "not_a_list"}  # type: ignore[dict-item]
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey={}) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        # No writes — only failure case for the file would be data corruption upstream
        self.assertIsNone(fx.written)

    def test_empty_record_list_is_skipped(self):
        accounts = {"hk_empty": []}
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey={}) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        self.assertIsNone(fx.written)

    def test_last_record_not_a_dict_is_skipped(self):
        accounts = {"hk_bad": ["not_a_dict_string"]}
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey={}) as fx:
            result = mig.main(dry_run=False)
        self.assertTrue(result)
        self.assertIsNone(fx.written)

    def test_cost_per_theta_legacy_key_dropped(self):
        """The migration drops the legacy top-level _cost_per_theta key the same way
        MinerAccountManager.load does. On the next write, the key won't reappear."""
        accounts = {
            "_cost_per_theta": 12.34,
            "hk": [_make_account_record(capital_used=100.0)],
        }
        positions = {"hk": [_make_position("hk", TradePair.BTCUSD, 100.0)]}
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            mig.main(dry_run=False)
        _, written = fx.written
        self.assertNotIn("_cost_per_theta", written)


class TestMigrationOnlyTouchesPerClassField(unittest.TestCase):
    """Migration is additive: never modifies other fields on the last record."""

    def test_other_fields_preserved(self):
        accounts = {
            "hk": [_make_account_record(capital_used=100.0)],
        }
        # Add other fields a real record would have
        accounts["hk"][-1].update({
            "total_realized_pnl": 42.0,
            "total_fees_paid": 1.0,
            "total_borrowed_amount": 0.0,
            "asset_class": "hl_all",
            "miner_bucket": "SUBACCOUNT_FUNDED",
            "hl_address": "0xabc",
            "max_return": 1.05,
        })
        positions = {"hk": [_make_position("hk", TradePair.BTCUSD, 100.0)]}
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            mig.main(dry_run=False)
        _, written = fx.written
        rec = written["hk"][-1]
        # Touched
        self.assertEqual(rec["capital_used_by_class"], {"crypto": 100.0})
        # Untouched
        self.assertEqual(rec["capital_used"], 100.0)
        self.assertEqual(rec["total_realized_pnl"], 42.0)
        self.assertEqual(rec["total_fees_paid"], 1.0)
        self.assertEqual(rec["total_borrowed_amount"], 0.0)
        self.assertEqual(rec["asset_class"], "hl_all")
        self.assertEqual(rec["miner_bucket"], "SUBACCOUNT_FUNDED")
        self.assertEqual(rec["hl_address"], "0xabc")
        self.assertEqual(rec["max_return"], 1.05)

    def test_collateral_records_preserved(self):
        accounts = {
            "hk": [
                {"account_size": 50_000, "update_time_ms": 1000},  # collateral record 1
                {"account_size": 60_000, "update_time_ms": 2000},  # collateral record 2
                _make_account_record(capital_used=100.0),          # final account state record
            ],
        }
        positions = {"hk": [_make_position("hk", TradePair.BTCUSD, 100.0)]}
        with MigrationFixture(accounts_data=accounts, positions_by_hotkey=positions) as fx:
            mig.main(dry_run=False)
        _, written = fx.written
        # All three entries preserved; only the last gets capital_used_by_class added
        self.assertEqual(len(written["hk"]), 3)
        self.assertEqual(written["hk"][0]["account_size"], 50_000)
        self.assertEqual(written["hk"][1]["account_size"], 60_000)
        self.assertEqual(written["hk"][-1]["capital_used_by_class"], {"crypto": 100.0})


if __name__ == "__main__":
    unittest.main()
