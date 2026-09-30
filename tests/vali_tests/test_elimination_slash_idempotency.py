# developer: Taoshidev
# Copyright (c) 2026 Taoshi Inc
"""
An elimination must slash collateral at most once.

append_elimination_row slashes on-chain before writing the elimination row, and the row is the
guard against slashing again. Regression context: when the slash outlasted the RPC timeout the
exception skipped the row write, so the next challenge-period pass slashed a second time.

Runs the real append_elimination_row on a bare EliminationManager (object.__new__, no heavy
constructor) with mocked RPC clients.
"""
import threading
import unittest
from unittest.mock import MagicMock

from shared_objects.rpc.rpc_client_base import RPCCallTimeoutError
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.elimination.elimination_manager import EliminationManager

HOTKEY = "5MinerHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
ELIM_KWARGS = dict(
    reason=EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN,
    elimination_drawdown_pct=5.0,
    intraday_drawdown_pct=5.0,
    eod_drawdown_pct=1.0,
    bucket_at_elimination=MinerBucket.MAINCOMP,
)


def _slash_timeout():
    return RPCCallTimeoutError("ContractManager", "slash_miner_collateral_proportion_rpc", 600.0)


class TestEliminationSlashIdempotency(unittest.TestCase):

    def setUp(self):
        m = EliminationManager.__new__(EliminationManager)
        m.eliminations = {}
        m.eliminations_lock = threading.Lock()
        m._eliminations_in_progress = set()
        m._slashed_before_row = {}
        m._save_eliminations_to_disk = MagicMock()
        m._challenge_period_client = MagicMock()
        m._entity_collateral_client = MagicMock()
        m._limit_order_client = MagicMock()
        m._position_client = MagicMock()
        m._miner_account_client = MagicMock()
        m._contract_client = MagicMock()
        m._contract_client.slash_miner_collateral_proportion.return_value = True
        self.m = m

    def _slash_calls(self):
        return self.m._contract_client.slash_miner_collateral_proportion.call_count

    def test_slash_timeout_still_writes_the_row(self):
        self.m._contract_client.slash_miner_collateral_proportion.side_effect = _slash_timeout()

        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)

        self.assertIn(HOTKEY, self.m.eliminations)
        # Outcome unknown -> recorded as slashed so a later withdrawal doesn't slash again
        self.assertTrue(self.m.eliminations[HOTKEY].collateral_slashed)
        self.m._position_client.close_all_positions.assert_called_once()
        self.assertNotIn(HOTKEY, self.m._eliminations_in_progress)
        self.assertNotIn(HOTKEY, self.m._slashed_before_row)

    def test_next_pass_after_slash_timeout_does_not_slash_again(self):
        self.m._contract_client.slash_miner_collateral_proportion.side_effect = _slash_timeout()
        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.assertEqual(self._slash_calls(), 1)

    def test_later_step_failure_retry_reuses_the_slash(self):
        # The slash succeeds, then the position close times out before the row is written.
        self.m._position_client.close_all_positions.side_effect = [
            RPCCallTimeoutError("PositionManagerServer", "close_all_positions_rpc", 60.0),
            True,
        ]
        with self.assertRaises(RPCCallTimeoutError):
            self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.assertNotIn(HOTKEY, self.m.eliminations)
        self.assertNotIn(HOTKEY, self.m._eliminations_in_progress)

        # Next pass: positions get closed and the row written, without a second slash.
        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.assertEqual(self._slash_calls(), 1)
        self.assertEqual(self.m._position_client.close_all_positions.call_count, 2)
        self.assertIn(HOTKEY, self.m.eliminations)
        self.assertTrue(self.m.eliminations[HOTKEY].collateral_slashed)
        self.assertNotIn(HOTKEY, self.m._slashed_before_row)

    def test_concurrent_call_while_in_progress_does_not_slash(self):
        self.m._eliminations_in_progress.add(HOTKEY)
        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.assertEqual(self._slash_calls(), 0)
        self.assertNotIn(HOTKEY, self.m.eliminations)
        # The second caller must not release the first caller's claim.
        self.assertIn(HOTKEY, self.m._eliminations_in_progress)

    def test_slash_error_records_not_slashed(self):
        self.m._contract_client.slash_miner_collateral_proportion.side_effect = RuntimeError("contract down")
        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.assertIn(HOTKEY, self.m.eliminations)
        self.assertFalse(self.m.eliminations[HOTKEY].collateral_slashed)

    def test_already_eliminated_is_a_no_op(self):
        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.m.append_elimination_row(HOTKEY, **ELIM_KWARGS)
        self.assertEqual(self._slash_calls(), 1)
        self.m._position_client.close_all_positions.assert_called_once()


if __name__ == "__main__":
    unittest.main()
