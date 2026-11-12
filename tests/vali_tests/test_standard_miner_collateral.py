"""
The collateral of a standard miner's account: withdrawals above the locked theta, the withdrawal that closes the
account and what it slashes, the slash each elimination reason takes from the locked theta, the close of the record
on the mothership and its reopening by an admin, and the queries run through the test servers.
"""
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, patch

from flask import Flask

from shared_objects.rpc.rpc_client_base import RPCCallTimeoutError
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from tests.vali_tests.test_standard_miner_opening import BLOCKS, CUTOFF, HOTKEYS, NEW, RAO, STAKE_WEIGHTS, TIER
from time_util.time_util import TimeUtil
from vali_objects.contract.validator_contract_manager import ValidatorContractManager
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.account_size_lock import AccountSizeLock
from vali_objects.miner_account.miner_account_manager import MinerAccount, MinerAccountManager
from vali_objects.miner_account.standard_miner import (StandardMinerAccount, closing_slash_proportion,
                                                       elimination_slash_proportion)
from vali_objects.utils.elimination.elimination_manager import EliminationManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from vanta_api.validator_rest_server import ValidatorRestServer

HOTKEY = "standard_miner_hotkey"
LEGACY = "legacy_miner_hotkey"
SUBACCOUNT = "entity_hotkey_1"
MOTHERSHIP = "mothership_hotkey"
COLDKEY = "miner_coldkey"
NOW_MS = 1_700_000_000_000
LOCKED = 70.0
BALANCE = 100.0
OPENING = dict(opened_at_ms=NOW_MS, opening_leverage_tier=2, asset_class="crypto", account_size=50_000.0,
               locked_theta=LOCKED, opening_price_usd=7.857)
# Down 2% today against a 3% daily loss limit, 4% below the end-of-day high-water mark against a 5% limit: 80%
STANDARD_LIMITS = {"intraday_drawdown_threshold": 0.03, "eod_drawdown_threshold": 0.05}
EIGHTY_PERCENT = {"intraday_drawdown_pct": 2.0, "trailing_drawdown_pct": 4.0, **STANDARD_LIMITS}
# While the static rules apply, the other drawdown is the one from the starting balance against the 5% static limit:
# down 2% today and 4% below the starting balance is 80% too, whatever the 9% from the high-water mark says
STATIC_LIMITS = {**STANDARD_LIMITS, "static_drawdown_threshold": 0.05, "static_rules_apply": True}
STATIC_EIGHTY_PERCENT = {"intraday_drawdown_pct": 2.0, "static_drawdown_pct": 4.0, "trailing_drawdown_pct": 9.0,
                         **STATIC_LIMITS}
CLOSING_SLASH = 0.8 * LOCKED
ACCOUNT_CLOSED = EliminationReason.ACCOUNT_CLOSED


def record(**overrides) -> StandardMinerAccount:
    return StandardMinerAccount(**{**OPENING, **overrides})


def rao(theta: float) -> int:
    return int(theta * 10 ** 9)


def elimination_row(reason=ACCOUNT_CLOSED, collateral_slashed=True, elimination_drawdown_pct=4.0, intraday_dd=2.0,
                    eod_dd=4.0) -> dict:
    """An elimination row as the elimination client serves it."""
    return {"hotkey": HOTKEY, "reason": reason.value, "collateral_slashed": collateral_slashed,
            "elimination_drawdown_pct": elimination_drawdown_pct, "dd": elimination_drawdown_pct,
            "intraday_dd": intraday_dd, "eod_dd": eod_dd, "bucket_at_elimination": MinerBucket.SUBACCOUNT_FUNDED.value,
            "elimination_initiated_time_ms": NOW_MS, "positions_closed": True, "row_added_ms": NOW_MS}


class TestClosingSlash(TestBase):

    def test_the_worse_of_the_two_drawdowns_against_its_limit_capped_at_everything(self):
        self.assertAlmostEqual(closing_slash_proportion(EIGHTY_PERCENT), 0.8)
        self.assertAlmostEqual(closing_slash_proportion({**EIGHTY_PERCENT, "intraday_drawdown_pct": 2.5}), 2.5 / 3)
        self.assertAlmostEqual(closing_slash_proportion({**EIGHTY_PERCENT, "trailing_drawdown_pct": 4.9}), 0.98)
        self.assertEqual(closing_slash_proportion({**EIGHTY_PERCENT, "intraday_drawdown_pct": 3.5}), 1.0)
        # The limits come with the stats, so the pro funded bucket's limits are read the same way
        pro = {"intraday_drawdown_pct": 2.0, "trailing_drawdown_pct": 4.0,
               "intraday_drawdown_threshold": 0.05, "eod_drawdown_threshold": 0.08}
        self.assertAlmostEqual(closing_slash_proportion(pro), 0.5)

    def test_under_the_static_rules_the_other_drawdown_is_the_one_from_the_starting_balance(self):
        self.assertAlmostEqual(closing_slash_proportion(STATIC_EIGHTY_PERCENT), 0.8)
        # Above the starting balance only today's loss counts
        self.assertAlmostEqual(closing_slash_proportion({**STATIC_EIGHTY_PERCENT, "static_drawdown_pct": -3.0}), 2.0 / 3)
        # Out of the static rules (the pro buckets) the high-water mark counts again
        self.assertEqual(closing_slash_proportion({**STATIC_EIGHTY_PERCENT, "static_rules_apply": False}), 1.0)

    def test_a_gain_or_no_stats_slashes_nothing(self):
        self.assertEqual(closing_slash_proportion({**EIGHTY_PERCENT, "intraday_drawdown_pct": -1.0,
                                                   "trailing_drawdown_pct": -2.0}), 0.0)
        self.assertEqual(closing_slash_proportion({"intraday_drawdown_pct": 2.0, "trailing_drawdown_pct": 4.0}), 0.0)
        self.assertEqual(closing_slash_proportion({"intraday_drawdown_pct": None, **STANDARD_LIMITS}), 0.0)
        self.assertEqual(closing_slash_proportion(None), 0.0)
        self.assertEqual(closing_slash_proportion({}), 0.0)

    def test_what_each_elimination_reason_takes(self):
        for reason, drawdown_pct, drawdown, expected in (
            (EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN, 3.1, None, 1.0),
            (EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN, 5.2, EIGHTY_PERCENT, 1.0),
            (EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN, 5.1, None, 1.0),
            (EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN, 8.3, None, 1.0),
            (EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_DRAWDOWN, 5.1, None, 1.0),
            (EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN, 5.1, STATIC_EIGHTY_PERCENT, 1.0),
            (EliminationReason.INACTIVE, 7.0, EIGHTY_PERCENT, 0.8),
            (EliminationReason.INACTIVE, None, STATIC_EIGHTY_PERCENT, 0.8),
            (EliminationReason.ZOMBIE, None, None, 0.0),
            (ACCOUNT_CLOSED, 4.0, EIGHTY_PERCENT, 0.8),
            (EliminationReason.PLAGIARISM, 5.0, EIGHTY_PERCENT, 0.5),
            (EliminationReason.DEREGISTERED, 12.0, EIGHTY_PERCENT, 0.8),
            (EliminationReason.DEREGISTERED, 12.0, None, 0.0),
            (EliminationReason.MARTINGALE, None, None, 0.0),
        ):
            with self.subTest(reason=reason):
                self.assertAlmostEqual(elimination_slash_proportion(reason, drawdown_pct, drawdown), expected)


class TestRecordReopening(TestBase):

    def test_a_later_reopening_outranks_a_close_and_a_later_stage_wins_otherwise(self):
        opened, closed = record(), record(closed_at_ms=NOW_MS)
        reopened = record(reopened_at_ms=NOW_MS + 1)
        closed_again = record(closed_at_ms=NOW_MS + 2, reopened_at_ms=NOW_MS + 1)
        reopened_twice = record(reopened_at_ms=NOW_MS + 3)
        pro = record(pro_account_size=150_000.0)
        for later, earlier in ((closed, opened), (pro, opened), (reopened, closed), (closed_again, reopened),
                               (reopened_twice, closed_again), (record(pro_account_size=150_000.0, reopened_at_ms=1), pro)):
            with self.subTest(later=later, earlier=earlier):
                self.assertTrue(later.supersedes(earlier))
                self.assertFalse(earlier.supersedes(later))
        for same in (opened, closed, reopened, closed_again):
            self.assertFalse(same.supersedes(same))
        self.assertEqual(StandardMinerAccount.from_dict(closed_again.to_dict()), closed_again)
        with self.assertRaises(ValueError):
            record(reopened_at_ms=1.5)


class TestCloseAndReopenOnTheAccount(TestBase):
    """The mothership alone closes or reopens an account and tells the other validators; a peer waits for its word,
    a write from a copy read earlier keeps the close, and a synced snapshot never undoes a close unless the account was
    reopened after it."""

    def setUp(self):
        super().setUp()
        mothership_hotkey = patch.object(ValiConfig, "MOTHERSHIP_HOTKEY_TESTNET", MOTHERSHIP)
        mothership_hotkey.start()
        self.addCleanup(mothership_hotkey.stop)
        self.mothership = self._manager(is_mothership=True)
        self.mothership._broadcast_to_validators = Mock()
        self.peer = self._manager(is_mothership=False)

    def _manager(self, is_mothership) -> MinerAccountManager:
        manager = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL, is_testnet=True)
        manager.is_mothership = is_mothership
        manager.sync_miner_account_sizes_data({})
        self.addCleanup(manager.sync_miner_account_sizes_data, {})
        return manager

    def _broadcast_records(self):
        return [call.kwargs["synapse_factory"]().record["record"]
                for call in self.mothership._broadcast_to_validators.call_args_list]

    def test_the_mothership_closes_and_reopens_and_broadcasts_each(self):
        self.mothership.open_standard_account(HOTKEY, record())
        self.assertTrue(self.mothership.close_standard_account(HOTKEY, NOW_MS + 5))
        self.assertEqual(self.mothership.get_standard_account(HOTKEY), record(closed_at_ms=NOW_MS + 5))
        # Closing a closed account changes nothing
        self.assertTrue(self.mothership.close_standard_account(HOTKEY, NOW_MS + 6))
        self.assertEqual(self.mothership.get_standard_account(HOTKEY), record(closed_at_ms=NOW_MS + 5))
        self.assertTrue(self.mothership.reopen_standard_account(HOTKEY, NOW_MS + 7))
        self.assertEqual(self.mothership.get_standard_account(HOTKEY), record(reopened_at_ms=NOW_MS + 7))
        self.assertTrue(self.mothership.reopen_standard_account(HOTKEY, NOW_MS + 8))
        self.assertEqual(self._broadcast_records(),
                         [record(closed_at_ms=NOW_MS + 5).to_dict(), record(reopened_at_ms=NOW_MS + 7).to_dict()])
        self.assertFalse(self.mothership.close_standard_account(LEGACY, NOW_MS))
        self.assertFalse(self.mothership.reopen_standard_account(LEGACY, NOW_MS))
        self.assertIsNone(self.mothership.get_account(LEGACY))
        self.assertEqual(self.mothership.get_dashboard(HOTKEY)["standard_account"],
                         record(reopened_at_ms=NOW_MS + 7).to_dict())

    def test_a_write_from_a_copy_read_earlier_keeps_the_close_and_the_reopening(self):
        self.mothership.open_standard_account(HOTKEY, record())
        stale = self.mothership.get_standard_account(HOTKEY)
        self.mothership.close_standard_account(HOTKEY, NOW_MS + 5)
        # A bucket move sizing the copy it read before the close
        self.mothership.set_standard_account(HOTKEY, replace(stale, trading_account_size=50_000.0))
        closed = record(trading_account_size=50_000.0, closed_at_ms=NOW_MS + 5)
        self.assertEqual(self.mothership.get_standard_account(HOTKEY), closed)
        self.assertEqual(self._broadcast_records()[-1], closed.to_dict())
        # The reopening passes, at a time past the close whatever the clock says, and a copy read before it keeps it
        self.assertTrue(self.mothership.reopen_standard_account(HOTKEY, NOW_MS + 7))
        reopened = replace(closed, closed_at_ms=None, reopened_at_ms=NOW_MS + 7)
        self.assertEqual(self.mothership.get_standard_account(HOTKEY), reopened)
        self.mothership.set_standard_account(HOTKEY, closed)
        self.assertEqual(self.mothership.get_standard_account(HOTKEY), reopened)
        self.mothership.close_standard_account(HOTKEY, NOW_MS + 9)
        self.assertTrue(self.mothership.reopen_standard_account(HOTKEY, NOW_MS + 7))
        self.assertEqual(self.mothership.get_standard_account(HOTKEY), replace(reopened, reopened_at_ms=NOW_MS + 8))

    def test_a_peer_takes_a_close_and_a_reopening_from_the_mothership_only(self):
        self.peer.open_standard_account(HOTKEY, record())
        self.assertFalse(self.peer.close_standard_account(HOTKEY, NOW_MS + 5))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), record())
        closed = record(closed_at_ms=NOW_MS + 5)
        self.assertTrue(self.peer.receive_standard_miner_record_update(
            {"hotkey": HOTKEY, "record": closed.to_dict()}, sender_hotkey=MOTHERSHIP))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), closed)
        self.assertTrue(self.peer.close_standard_account(HOTKEY, NOW_MS + 6))
        self.assertFalse(self.peer.reopen_standard_account(HOTKEY, NOW_MS + 7))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), closed)
        reopened = record(reopened_at_ms=NOW_MS + 7)
        self.assertTrue(self.peer.receive_standard_miner_record_update(
            {"hotkey": HOTKEY, "record": reopened.to_dict()}, sender_hotkey=MOTHERSHIP))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), reopened)

    def test_a_synced_snapshot_undoes_a_close_only_through_a_later_reopening(self):
        self.peer.open_standard_account(HOTKEY, record())
        closed = record(closed_at_ms=NOW_MS + 5)
        reopened = record(reopened_at_ms=NOW_MS + 7)
        closed_again = record(closed_at_ms=NOW_MS + 9, reopened_at_ms=NOW_MS + 7)
        for held, synced, kept in (
            (closed, record(), closed),
            (closed, reopened, reopened),
            (reopened, closed, reopened),
            (closed_again, reopened, closed_again),
            (reopened, closed_again, closed_again),
            (closed_again, record(reopened_at_ms=NOW_MS + 11), record(reopened_at_ms=NOW_MS + 11)),
        ):
            with self.subTest(held=held, synced=synced):
                self.peer.sync_miner_account_sizes_data({})
                self.peer.open_standard_account(HOTKEY, held)
                snapshot = self.peer.accounts_dict()
                snapshot[HOTKEY][-1]["standard_account"] = synced.to_dict()
                self.peer.sync_miner_account_sizes_data(snapshot)
                self.assertEqual(self.peer.get_standard_account(HOTKEY), kept)


class TestWithdrawals(TestBase):
    """query_withdrawal_request and process_withdrawal_request on the mothership, with the chain and the services
    mocked: the unlocked part leaves freely, a withdrawal into the locked theta closes the account through its
    elimination, and an eliminated account whose slash failed pays it on the way out."""

    def setUp(self):
        super().setUp()
        m = object.__new__(ValidatorContractManager)
        m.is_mothership = True
        m.is_testnet = False  # max_theta is then MAX_COLLATERAL_BALANCE_THETA
        m.running_unit_tests = False
        m._account_locks = {}
        m._account_locks_guard = threading.Lock()
        m.wallet = MagicMock()
        m.collateral_manager = MagicMock()
        m.collateral_manager.withdraw.side_effect = lambda amount, **kwargs: SimpleNamespace(rao=amount)
        m.get_miner_collateral_balance = MagicMock(return_value=BALANCE)
        m._set_miner_account_size = MagicMock()
        m.force_deposit = MagicMock()
        m._miner_account_client = MagicMock()
        m._miner_account_client.get_standard_account.return_value = record()
        m._miner_account_client.get_account.return_value = None
        m._position_client = MagicMock()
        m._position_client.get_positions_for_one_hotkey.return_value = []
        m._challenge_period_client = MagicMock()
        m._challenge_period_client.get_drawdown_stats.return_value = EIGHTY_PERCENT
        m._elimination_client = MagicMock()
        m._elimination_client.get_elimination.return_value = None
        m._elimination_client.append_elimination_row.side_effect = self._elimination_lands
        m._entity_collateral_client = MagicMock()
        m._entity_collateral_client.compute_entity_required_collateral.return_value = 0.0
        m._perf_ledger_client = MagicMock()
        m._perf_ledger_client.get_perf_ledger_for_hotkey.return_value = None
        secret = patch.object(ValiUtils, "get_secret", return_value="secret")
        secret.start()
        self.addCleanup(secret.stop)
        self.m = m
        self.landed_row = elimination_row()

    def _elimination_lands(self, **kwargs):
        """The elimination server slashes the closing amount, closes the record and writes the row."""
        self.m._elimination_client.get_elimination.return_value = self.landed_row
        self.m._miner_account_client.get_standard_account.return_value = record(closed_at_ms=NOW_MS)
        self.m.get_miner_collateral_balance.return_value -= kwargs["slash_theta"]

    def _open_position(self):
        return SimpleNamespace(is_open_position=True, trade_pair=SimpleNamespace(trade_pair_id="BTCUSD"))

    def _withdrawn_theta(self):
        return self.m.collateral_manager.withdraw.call_args.kwargs["amount"] / 10 ** 9

    def _error(self, amount=31.0, hotkey=HOTKEY, close_account=True):
        processed = self.m.process_withdrawal_request(amount, COLDKEY, hotkey, close_account)
        self.assertFalse(processed["successfully_processed"])
        self.assertEqual(processed.get("returned_amount", 0.0), 0.0)
        return processed["error_message"]

    def _close(self, amount=31.0):
        return self.m.process_withdrawal_request(amount, COLDKEY, HOTKEY, close_account=True)

    def test_the_unlocked_part_leaves_without_a_slash_whatever_the_positions(self):
        self.m._position_client.get_positions_for_one_hotkey.return_value = [self._open_position()]
        for amount in (30.0, 30.0 + 1e-10):
            with self.subTest(amount=amount):
                result = self.m.query_withdrawal_request(amount, HOTKEY)
                self.assertTrue(result["successfully_processed"], result)
                self.assertEqual((result["closes_account"], result["slashed_amount"], result["locked_theta"]),
                                 (False, 0.0, LOCKED))
                self.assertAlmostEqual(result["withdrawal_amount"], amount)
                self.assertAlmostEqual(result["new_balance"], BALANCE - amount)
        self.m._position_client.get_positions_for_one_hotkey.assert_not_called()
        self.m._challenge_period_client.get_drawdown_stats.assert_not_called()
        processed = self.m.process_withdrawal_request(30.0, COLDKEY, HOTKEY)
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertEqual((processed["returned_amount"], processed["account_closed"]), (30.0, False))
        self.m.collateral_manager.slash.assert_not_called()
        self.m._elimination_client.append_elimination_row.assert_not_called()

    def test_a_withdrawal_into_the_locked_theta_needs_every_position_closed(self):
        self.m._position_client.get_positions_for_one_hotkey.return_value = [self._open_position()]
        result = self.m.query_withdrawal_request(31.0, HOTKEY)
        self.assertFalse(result["successfully_processed"])
        self.assertIn("closes it", result["error_message"])
        self.assertIn("BTCUSD", result["error_message"])
        self.assertIn("closes it", self._error())
        self.m._elimination_client.append_elimination_row.assert_not_called()
        self.m.collateral_manager.withdraw.assert_not_called()

    def test_a_withdrawal_into_the_locked_theta_takes_everything_less_the_closing_slash(self):
        for amount in (31.0, 100.0):
            with self.subTest(amount=amount):
                result = self.m.query_withdrawal_request(amount, HOTKEY)
                self.assertTrue(result["successfully_processed"], result)
                self.assertEqual((result["closes_account"], result["locked_theta"], result["new_balance"]),
                                 (True, LOCKED, 0.0))
                self.assertAlmostEqual(result["slash_proportion"], 0.8)
                self.assertAlmostEqual(result["slashed_amount"], CLOSING_SLASH)
                self.assertAlmostEqual(result["withdrawal_amount"], BALANCE - CLOSING_SLASH)
                self.assertEqual((result["intraday_drawdown_pct"], result["trailing_drawdown_pct"], result["drawdown"]),
                                 (2.0, 4.0, 0.04))
        # No drawdown stats, or a gain: nothing is slashed and the whole balance comes out
        for stats in (None, {**EIGHTY_PERCENT, "intraday_drawdown_pct": -1.0, "trailing_drawdown_pct": -1.0}):
            self.m._challenge_period_client.get_drawdown_stats.return_value = stats
            result = self.m.query_withdrawal_request(31.0, HOTKEY)
            self.assertEqual((result["slashed_amount"], result["withdrawal_amount"], result["drawdown"]), (0.0, BALANCE, 0.0))
        # The slash never exceeds the balance
        self.m._challenge_period_client.get_drawdown_stats.return_value = {**EIGHTY_PERCENT, "intraday_drawdown_pct": 9.0}
        self.m.get_miner_collateral_balance.return_value = 50.0
        result = self.m.query_withdrawal_request(31.0, HOTKEY)
        self.assertEqual((result["slashed_amount"], result["withdrawal_amount"]), (50.0, 0.0))
        # A record closed without its elimination row has the close recorded again, whatever the amount, and is
        # never slashed again
        self.m.get_miner_collateral_balance.return_value = BALANCE
        self.m._miner_account_client.get_standard_account.return_value = record(closed_at_ms=NOW_MS)
        self.m._challenge_period_client.get_drawdown_stats.reset_mock()
        result = self.m.query_withdrawal_request(1.0, HOTKEY)
        self.assertEqual((result["closes_account"], result["slashed_amount"], result["withdrawal_amount"]),
                         (True, 0.0, BALANCE))
        self.m._challenge_period_client.get_drawdown_stats.assert_not_called()
        processed = self._close(1.0)
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertEqual(self.m._elimination_client.append_elimination_row.call_args.kwargs["slash_theta"], 0.0)
        self.assertEqual(processed["returned_amount"], BALANCE)

    def test_the_closing_slash_reads_the_balance_as_it_is_now(self):
        # The cached stats say 2% and 4% down, but the positions just closed left the balance 3.5% below both marks
        self.m._challenge_period_client.get_drawdown_stats.return_value = {
            **EIGHTY_PERCENT, "daily_open_balance": 1.0, "daily_open_equity": 1.0, "eod_hwm": 1.0}
        self.m._miner_account_client.get_account.return_value = SimpleNamespace(
            balance=48_250.0, get_account_size=lambda: 50_000.0)
        result = self.m.query_withdrawal_request(31.0, HOTKEY)
        self.assertEqual((result["slash_proportion"], result["slashed_amount"]), (1.0, LOCKED))
        self.assertAlmostEqual(result["intraday_drawdown_pct"], 3.5)
        # And a recovery the cache has not seen yet lowers the slash
        self.m._miner_account_client.get_account.return_value = SimpleNamespace(
            balance=49_750.0, get_account_size=lambda: 50_000.0)
        result = self.m.query_withdrawal_request(31.0, HOTKEY)
        self.assertAlmostEqual(result["slash_proportion"], 0.5 / 3)
        self.assertAlmostEqual(result["slashed_amount"], LOCKED * 0.5 / 3)

    def test_under_the_static_rules_the_closing_slash_reads_the_drawdown_from_the_starting_balance(self):
        # The balance 4% below the starting balance and 2% below the day's open: 80%, though it stands 9% below the
        # high-water mark
        self.m._challenge_period_client.get_drawdown_stats.return_value = {
            **STATIC_EIGHTY_PERCENT, "daily_open_balance": 0.96 / 0.98, "eod_hwm": 0.96 / 0.91}
        self.m._miner_account_client.get_account.return_value = SimpleNamespace(
            balance=48_000.0, get_account_size=lambda: 50_000.0)
        result = self.m.query_withdrawal_request(31.0, HOTKEY)
        self.assertAlmostEqual(result["slash_proportion"], 0.8)
        self.assertAlmostEqual(result["slashed_amount"], CLOSING_SLASH)
        self.assertAlmostEqual(result["intraday_drawdown_pct"], 2.0)
        self.assertAlmostEqual(result["static_drawdown_pct"], 4.0)
        self.assertAlmostEqual(result["drawdown"], 0.04)
        self.assertNotIn("trailing_drawdown_pct", result)
        # Above the starting balance only today's loss counts
        self.m._challenge_period_client.get_drawdown_stats.return_value = {
            **STATIC_EIGHTY_PERCENT, "daily_open_balance": 1.02 / 0.98, "eod_hwm": 1.02 / 0.91}
        self.m._miner_account_client.get_account.return_value = SimpleNamespace(
            balance=51_000.0, get_account_size=lambda: 50_000.0)
        result = self.m.query_withdrawal_request(31.0, HOTKEY)
        self.assertAlmostEqual(result["slash_proportion"], 2.0 / 3)
        self.assertEqual(result["static_drawdown_pct"], 0.0)
        self.assertAlmostEqual(result["drawdown"], 0.02)

    def test_a_withdrawal_that_closes_the_account_needs_the_consent(self):
        processed = self.m.process_withdrawal_request(31.0, COLDKEY, HOTKEY)
        self.assertFalse(processed["successfully_processed"])
        self.assertIn("close_account", processed["error_message"])
        self.assertEqual((processed["closes_account"], processed["slashed_amount"], processed["returned_amount"]),
                         (True, CLOSING_SLASH, 0.0))
        self.m._elimination_client.append_elimination_row.assert_not_called()
        self.m.collateral_manager.withdraw.assert_not_called()
        self.assertTrue(self._close()["successfully_processed"])
        self.m._elimination_client.append_elimination_row.assert_called_once()

    def test_under_the_static_rules_the_closing_row_keeps_the_drawdown_from_the_starting_balance(self):
        self.m._challenge_period_client.get_drawdown_stats.return_value = STATIC_EIGHTY_PERCENT
        processed = self._close()
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertEqual(processed["returned_amount"], BALANCE - CLOSING_SLASH)
        self.m._elimination_client.append_elimination_row.assert_called_once_with(
            hotkey=HOTKEY, reason=ACCOUNT_CLOSED, elimination_drawdown_pct=4.0, intraday_drawdown_pct=2.0,
            eod_drawdown_pct=4.0, elimination_time_ms=ANY, slash_theta=CLOSING_SLASH)

    def test_closing_runs_the_elimination_then_pays_out_what_is_left(self):
        calls = Mock()
        calls.attach_mock(self.m._elimination_client, "elimination")
        calls.attach_mock(self.m.collateral_manager, "chain")
        processed = self._close()
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertEqual((processed["returned_amount"], processed["account_closed"]), (BALANCE - CLOSING_SLASH, True))
        # The elimination takes the closing slash on the locked theta; nothing is slashed here
        self.m._elimination_client.append_elimination_row.assert_called_once_with(
            hotkey=HOTKEY, reason=ACCOUNT_CLOSED, elimination_drawdown_pct=4.0, intraday_drawdown_pct=2.0,
            eod_drawdown_pct=4.0, elimination_time_ms=ANY, slash_theta=CLOSING_SLASH)
        self.m.collateral_manager.slash.assert_not_called()
        self.m.force_deposit.assert_not_called()
        names = [name for name, args, kwargs in calls.mock_calls]
        self.assertLess(names.index("elimination.append_elimination_row"), names.index("chain.withdraw"))
        self.assertAlmostEqual(self._withdrawn_theta(), BALANCE - CLOSING_SLASH)
        # Once closed and slashed, the rest comes out as for any eliminated miner, with nothing slashed
        self.m._challenge_period_client.get_miner_bucket.return_value = MinerBucket.ELIMINATED
        self.m.get_miner_collateral_balance.return_value = 44.0
        result = self.m.query_withdrawal_request(44.0, HOTKEY)
        self.assertTrue(result["successfully_processed"], result)
        self.assertEqual((result["slashed_amount"], result["withdrawal_amount"], result.get("closes_account")),
                         (0, 44.0, None))

    def test_a_close_that_slashed_everything_pays_nothing_out_and_still_succeeds(self):
        self.m._challenge_period_client.get_drawdown_stats.return_value = {**EIGHTY_PERCENT, "intraday_drawdown_pct": 9.0}
        self.m.get_miner_collateral_balance.return_value = LOCKED
        processed = self._close(LOCKED)
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertEqual((processed["returned_amount"], processed["account_closed"]), (0.0, True))
        self.assertEqual(self.m._elimination_client.append_elimination_row.call_args.kwargs["slash_theta"], LOCKED)
        self.m.collateral_manager.withdraw.assert_not_called()

    def test_a_close_cut_short_pays_nothing_out(self):
        def lands_without_the_slash_on_chain(**kwargs):
            self.m._elimination_client.get_elimination.return_value = elimination_row()

        for failure, expected in (
            ("stays open", RuntimeError("The closing slash of 56.0 theta failed; the account stays open")),
            ("still being processed", RPCCallTimeoutError("EliminationManager", "append_elimination_row_rpc", 600.0)),
            ("not recorded", lambda **kwargs: None),
            ("eliminated (INACTIVE)", elimination_row(reason=EliminationReason.INACTIVE)),
            ("still owed", elimination_row(collateral_slashed=False)),
            ("still settling", lands_without_the_slash_on_chain),
        ):
            with self.subTest(failure=failure):
                self.m._elimination_client.reset_mock()
                self.m.collateral_manager.reset_mock()
                self.m._elimination_client.get_elimination.return_value = None
                self.m._miner_account_client.get_standard_account.return_value = record()
                self.m.get_miner_collateral_balance.return_value = BALANCE
                if isinstance(expected, Exception):
                    self.m._elimination_client.append_elimination_row.side_effect = expected
                elif callable(expected):
                    self.m._elimination_client.append_elimination_row.side_effect = expected
                else:
                    self.landed_row = expected
                    self.m._elimination_client.append_elimination_row.side_effect = self._elimination_lands
                self.assertIn(failure, self._error())
                self.m._elimination_client.append_elimination_row.assert_called_once()
                self.m.collateral_manager.withdraw.assert_not_called()
                self.m.collateral_manager.slash.assert_not_called()
                self.m.force_deposit.assert_not_called()

    def test_a_payout_that_fails_after_the_close_keeps_the_slash(self):
        self.m.collateral_manager.withdraw.side_effect = RuntimeError("transfer failed")
        self.assertIn("transfer failed", self._error())
        self.m._elimination_client.append_elimination_row.assert_called_once()
        self.m.force_deposit.assert_not_called()
        # The rest comes out on the next try, as for any eliminated miner
        self.m.collateral_manager.withdraw.side_effect = lambda amount, **kwargs: SimpleNamespace(rao=amount)
        self.m._challenge_period_client.get_miner_bucket.return_value = MinerBucket.ELIMINATED
        processed = self.m.process_withdrawal_request(BALANCE - CLOSING_SLASH, COLDKEY, HOTKEY)
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertEqual((processed["returned_amount"], processed["account_closed"]), (BALANCE - CLOSING_SLASH, False))
        self.m._elimination_client.append_elimination_row.assert_called_once()
        self.m.collateral_manager.slash.assert_not_called()

    def test_an_account_eliminated_under_the_static_rules_pays_on_the_drawdown_from_the_starting_balance(self):
        # The row's eod_dd holds the drawdown from the starting balance; read against the 8% EOD limit instead, the
        # inactive account would owe only two thirds
        self.m._miner_account_client.get_standard_account.return_value = record(closed_at_ms=NOW_MS)
        self.m._challenge_period_client.get_drawdown_stats.return_value = {**STATIC_LIMITS, "eod_drawdown_threshold": 0.08}
        for row, slash in (
            (elimination_row(reason=EliminationReason.INACTIVE, collateral_slashed=False), CLOSING_SLASH),
            (elimination_row(reason=EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN, collateral_slashed=False,
                             elimination_drawdown_pct=5.1), LOCKED),
        ):
            with self.subTest(reason=row["reason"]):
                self.m._elimination_client.get_elimination.return_value = row
                result = self.m.query_withdrawal_request(1.0, HOTKEY)
                self.assertTrue(result["successfully_processed"], result)
                self.assertAlmostEqual(result["slashed_amount"], slash)
                self.assertAlmostEqual(result["withdrawal_amount"], BALANCE - slash)

    def test_an_eliminated_account_whose_slash_failed_pays_it_on_the_way_out(self):
        self.m._miner_account_client.get_standard_account.return_value = record(closed_at_ms=NOW_MS)
        self.m._challenge_period_client.get_drawdown_stats.return_value = STANDARD_LIMITS
        for row, slash in (
            (elimination_row(reason=EliminationReason.INACTIVE, collateral_slashed=False), CLOSING_SLASH),
            (elimination_row(reason=EliminationReason.ZOMBIE, collateral_slashed=False, intraday_dd=-1.0, eod_dd=-1.0), 0.0),
            (elimination_row(reason=EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN, collateral_slashed=False,
                             elimination_drawdown_pct=5.2), LOCKED),
            (elimination_row(reason=EliminationReason.PLAGIARISM, collateral_slashed=False, elimination_drawdown_pct=5.0), 0.5 * LOCKED),
        ):
            with self.subTest(reason=row["reason"]):
                self.m.collateral_manager.reset_mock()
                self.m._elimination_client.reset_mock()
                self.m._elimination_client.get_elimination.return_value = row
                result = self.m.query_withdrawal_request(1.0, HOTKEY)
                self.assertTrue(result["successfully_processed"], result)
                self.assertEqual((result["closes_account"], result["new_balance"], result["locked_theta"]), (False, 0.0, LOCKED))
                self.assertAlmostEqual(result["slashed_amount"], slash)
                self.assertAlmostEqual(result["withdrawal_amount"], BALANCE - slash)
                processed = self.m.process_withdrawal_request(1.0, COLDKEY, HOTKEY)
                self.assertTrue(processed["successfully_processed"], processed)
                self.assertAlmostEqual(processed["returned_amount"], BALANCE - slash)
                if slash > 0:
                    self.assertEqual(self.m.collateral_manager.slash.call_args.kwargs["amount"], rao(slash))
                    # The row then says the slash was taken, so a later deposit is not slashed again
                    self.m._elimination_client.set_collateral_slashed.assert_called_once_with(HOTKEY, True)
                else:
                    self.m.collateral_manager.slash.assert_not_called()
                    self.m._elimination_client.set_collateral_slashed.assert_not_called()
                self.m._elimination_client.append_elimination_row.assert_not_called()
        # The slash never exceeds the balance, and a transfer that fails restores it
        self.m._elimination_client.get_elimination.return_value = elimination_row(
            reason=EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN, collateral_slashed=False)
        self.m.get_miner_collateral_balance.return_value = 50.0
        self.m.collateral_manager.reset_mock()
        self.m.collateral_manager.withdraw.side_effect = RuntimeError("transfer failed")
        processed = self.m.process_withdrawal_request(1.0, COLDKEY, HOTKEY)
        self.assertTrue(processed["successfully_processed"], processed)  # nothing left to transfer
        self.assertEqual(processed["returned_amount"], 0.0)
        self.assertEqual(self.m.collateral_manager.slash.call_args.kwargs["amount"], rao(50.0))
        self.m.get_miner_collateral_balance.return_value = BALANCE
        self.assertIn("transfer failed", self._error(1.0))
        self.m.force_deposit.assert_called_once_with(LOCKED, HOTKEY)
        # A row that could not be marked does not fail a withdrawal that paid out
        self.m._elimination_client.set_collateral_slashed.side_effect = RuntimeError("elimination server down")
        self.m.collateral_manager.withdraw.side_effect = lambda amount, **kwargs: SimpleNamespace(rao=amount)
        processed = self.m.process_withdrawal_request(1.0, COLDKEY, HOTKEY)
        self.assertTrue(processed["successfully_processed"], processed)
        self.m._elimination_client.set_collateral_slashed.side_effect = None
        # A slash that fails pays nothing out and restores nothing
        self.m._elimination_client.reset_mock()
        self.m.collateral_manager.reset_mock()
        self.m.force_deposit.reset_mock()
        self.m.collateral_manager.slash.side_effect = RuntimeError("chain down")
        self.assertIn("owed by the account failed", self._error(1.0))
        self.m.collateral_manager.withdraw.assert_not_called()
        self.m.force_deposit.assert_not_called()
        self.m._elimination_client.set_collateral_slashed.assert_not_called()

    def test_a_closing_withdrawal_is_processed_by_the_mothership_only(self):
        self.m.is_mothership = False
        self.assertTrue(self.m.query_withdrawal_request(31.0, HOTKEY)["closes_account"])
        self.assertIn("mothership", self._error())
        self.m._elimination_client.append_elimination_row.assert_not_called()
        self.m.collateral_manager.withdraw.assert_not_called()
        self.assertTrue(self.m.process_withdrawal_request(30.0, COLDKEY, HOTKEY)["successfully_processed"])

    def test_one_account_withdraws_one_request_at_a_time(self):
        second = {}
        thread = threading.Thread(target=lambda: second.update(result=self._close()), daemon=True)

        def withdraw(amount, **kwargs):
            thread.start()
            thread.join(0.5)
            self.assertTrue(thread.is_alive())  # the second request waits for this one
            self.m.get_miner_collateral_balance.return_value = 0.0
            return SimpleNamespace(rao=amount)

        self.m.collateral_manager.withdraw.side_effect = withdraw
        processed = self._close()
        self.assertTrue(processed["successfully_processed"], processed)
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertFalse(second["result"]["successfully_processed"])
        self.assertIn("Insufficient collateral balance", second["result"]["error_message"])
        self.m._elimination_client.append_elimination_row.assert_called_once()
        self.m.collateral_manager.withdraw.assert_called_once()

    def test_any_other_miner_still_has_a_failed_withdrawal_restore_its_slash(self):
        self.m._miner_account_client.get_standard_account.return_value = None
        self.m._miner_account_client.get_account.return_value = SimpleNamespace(balance=95_000.0, account_size=100_000.0, max_return=1.0)
        self.m._challenge_period_client.get_miner_bucket.return_value = MinerBucket.MAINCOMP
        self.m.get_miner_collateral_balance.return_value = 1_500.0
        result = self.m.query_withdrawal_request(1_000.0, LEGACY)
        self.assertTrue(result["successfully_processed"], result)
        self.assertNotIn("closes_account", result)
        # 5% drawdown against the 8% funded limit, on the 500 theta above the penalty-free balance
        expected_slash = 500.0 * 0.05 / ValiConfig.FUNDED_EOD_DRAWDOWN_THRESHOLD
        self.assertAlmostEqual(result["slashed_amount"], expected_slash)
        self.m.collateral_manager.withdraw.side_effect = RuntimeError("transfer failed")
        self.assertIn("transfer failed", self._error(1_000.0, LEGACY))
        self.m.force_deposit.assert_called_once()
        self.assertAlmostEqual(self.m.force_deposit.call_args.args[0], expected_slash)
        self.m._elimination_client.append_elimination_row.assert_not_called()
        # Nothing is restored for a slash that never ran
        self.m.force_deposit.reset_mock()
        self.m.collateral_manager.slash.side_effect = RuntimeError("chain down")
        self.assertIn("transfer failed", self._error(1_000.0, LEGACY))
        self.m.force_deposit.assert_not_called()


class TestEliminationSlash(TestBase):
    """append_elimination_row on a bare EliminationManager with the services mocked: what each reason takes from a
    standard miner's locked theta, and that every elimination closes the record and the account bucket."""

    def setUp(self):
        super().setUp()
        m = EliminationManager.__new__(EliminationManager)
        m.eliminations = {}
        m.eliminations_lock = threading.Lock()
        m._eliminations_in_progress = set()
        m._slashed_before_row = {}
        m._save_eliminations_to_disk = MagicMock()
        m._perf_ledger_client = MagicMock()
        m._perf_ledger_client.filtered_ledger_for_scoring.return_value = {}
        m._challenge_period_client = MagicMock()
        m._challenge_period_client.get_miner_bucket.return_value = MinerBucket.SUBACCOUNT_FUNDED
        m._challenge_period_client.get_drawdown_stats.return_value = EIGHTY_PERCENT
        m._entity_collateral_client = MagicMock()
        m._limit_order_client = MagicMock()
        m._position_client = MagicMock()
        m._miner_account_client = MagicMock()
        m._miner_account_client.get_standard_account.side_effect = lambda hotkey: record() if hotkey == HOTKEY else None
        m._contract_client = MagicMock()
        m._contract_client.slash_miner_collateral.return_value = True
        m._contract_client.slash_miner_collateral_proportion.return_value = True
        self.m = m

    def _slashed_theta(self):
        calls = self.m._contract_client.slash_miner_collateral.call_args_list
        return [call.args[1] for call in calls]

    def _eliminate(self, reason, hotkey=HOTKEY, **kwargs):
        self.m.append_elimination_row(hotkey, reason, **kwargs)
        return self.m.eliminations[hotkey]

    def _nothing_happened(self):
        self.assertEqual(self.m.eliminations, {})
        self.assertEqual(self.m._slashed_before_row, {})
        self.m._miner_account_client.close_standard_account.assert_not_called()
        self.m._miner_account_client.set_miner_bucket.assert_not_called()

    def test_what_each_reason_takes_from_the_locked_theta(self):
        cases = [
            (EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN, dict(elimination_drawdown_pct=3.1), LOCKED),
            (EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN, dict(elimination_drawdown_pct=5.2), LOCKED),
            (EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN, dict(elimination_drawdown_pct=5.1), LOCKED),
            (EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN, dict(elimination_drawdown_pct=8.3), LOCKED),
            (EliminationReason.INACTIVE, {}, CLOSING_SLASH),
            (EliminationReason.ZOMBIE, {}, CLOSING_SLASH),
            (EliminationReason.PLAGIARISM, dict(elimination_drawdown_pct=5.0), 0.5 * LOCKED),
            (EliminationReason.DEREGISTERED, dict(elimination_drawdown_pct=12.0), CLOSING_SLASH),
            (EliminationReason.MARTINGALE, dict(elimination_drawdown_pct=0.0), None),
            (ACCOUNT_CLOSED, dict(elimination_drawdown_pct=4.0, slash_theta=CLOSING_SLASH), CLOSING_SLASH),
            (ACCOUNT_CLOSED, dict(elimination_drawdown_pct=0.0, slash_theta=0.0), None),
        ]
        for reason, kwargs, expected in cases:
            with self.subTest(reason=reason, kwargs=kwargs):
                self.m.eliminations.clear()
                self.m._contract_client.reset_mock()
                self.m._miner_account_client.reset_mock()
                row = self._eliminate(reason, **kwargs)
                slashed = self._slashed_theta()
                if expected is None:
                    self.assertEqual(slashed, [])
                else:
                    self.assertEqual(len(slashed), 1)
                    self.assertAlmostEqual(slashed[0], expected)
                self.assertTrue(row.collateral_slashed)
                self.assertEqual((row.reason, row.bucket_at_elimination), (reason.value, MinerBucket.SUBACCOUNT_FUNDED))
                self.m._contract_client.slash_miner_collateral_proportion.assert_not_called()
                self.m._miner_account_client.close_standard_account.assert_called_once_with(HOTKEY, row.elimination_initiated_time_ms)
                self.m._miner_account_client.set_miner_bucket.assert_called_once_with(HOTKEY, MinerBucket.ELIMINATED)
                if reason in (EliminationReason.INACTIVE, EliminationReason.ZOMBIE, EliminationReason.DEREGISTERED):
                    # The row keeps the drawdown the closing share was measured on
                    self.assertEqual((row.intraday_drawdown_pct, row.eod_drawdown_pct), (2.0, 4.0))
        # Inactive with no drawdown on record: the whole locked theta stays with the miner
        self.m.eliminations.clear()
        self.m._contract_client.reset_mock()
        self.m._challenge_period_client.get_drawdown_stats.return_value = None
        row = self._eliminate(EliminationReason.INACTIVE)
        self.assertTrue(row.collateral_slashed)
        self.assertEqual(self._slashed_theta(), [])
        self.assertIsNone(row.intraday_drawdown_pct)

    def test_under_the_static_rules_a_breach_takes_everything_and_a_closing_share_reads_the_starting_balance(self):
        self.m._challenge_period_client.get_drawdown_stats.return_value = STATIC_EIGHTY_PERCENT
        for reason, kwargs, expected in (
            (EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_DRAWDOWN, dict(elimination_drawdown_pct=5.1), LOCKED),
            (EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN, dict(elimination_drawdown_pct=5.1), LOCKED),
            (EliminationReason.INACTIVE, {}, CLOSING_SLASH),
        ):
            with self.subTest(reason=reason):
                self.m.eliminations.clear()
                self.m._contract_client.reset_mock()
                row = self._eliminate(reason, **kwargs)
                slashed = self._slashed_theta()
                self.assertEqual(len(slashed), 1)
                self.assertAlmostEqual(slashed[0], expected)
        # The row keeps the drawdown from the starting balance that the closing share was measured on
        self.assertEqual((row.intraday_drawdown_pct, row.eod_drawdown_pct), (2.0, 4.0))

    def test_a_record_closed_by_an_attempt_a_restart_forgot_is_not_slashed_again(self):
        self.m._miner_account_client.get_standard_account.side_effect = lambda hotkey: record(closed_at_ms=NOW_MS)
        for reason, kwargs in ((EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN, dict(elimination_drawdown_pct=5.2)),
                               (EliminationReason.INACTIVE, {}),
                               (ACCOUNT_CLOSED, dict(elimination_drawdown_pct=4.0, slash_theta=CLOSING_SLASH))):
            with self.subTest(reason=reason):
                self.m.eliminations.clear()
                self.m._contract_client.reset_mock()
                row = self._eliminate(reason, **kwargs)
                self.assertTrue(row.collateral_slashed)
                self.assertEqual(self._slashed_theta(), [])
                self.m._miner_account_client.close_standard_account.assert_called()

    def test_a_closing_slash_that_fails_leaves_the_account_open(self):
        for failure in (False, RuntimeError("chain down")):
            with self.subTest(failure=failure):
                self.m._contract_client.slash_miner_collateral.side_effect = failure if isinstance(failure, Exception) else None
                self.m._contract_client.slash_miner_collateral.return_value = failure
                with self.assertRaises(RuntimeError) as raised:
                    self.m.append_elimination_row(HOTKEY, ACCOUNT_CLOSED, elimination_drawdown_pct=4.0, slash_theta=CLOSING_SLASH)
                self.assertIn("stays open", str(raised.exception))
                self._nothing_happened()
                self.assertEqual(self.m._eliminations_in_progress, set())
        # An outcome unknown is taken as slashed, as for any elimination
        self.m._contract_client.slash_miner_collateral.side_effect = RPCCallTimeoutError("ContractManager", "slash_miner_collateral_rpc", 600.0)
        row = self._eliminate(ACCOUNT_CLOSED, elimination_drawdown_pct=4.0, slash_theta=CLOSING_SLASH)
        self.assertTrue(row.collateral_slashed)
        self.m._miner_account_client.close_standard_account.assert_called_once()

    def test_a_slash_that_timed_out_or_failed_is_recorded_as_for_any_other_miner(self):
        self.m._contract_client.slash_miner_collateral.side_effect = RPCCallTimeoutError("ContractManager", "slash_miner_collateral_rpc", 600.0)
        self.assertTrue(self._eliminate(EliminationReason.INACTIVE).collateral_slashed)
        self.m.eliminations.clear()
        self.m._contract_client.slash_miner_collateral.side_effect = RuntimeError("chain down")
        self.assertFalse(self._eliminate(EliminationReason.INACTIVE).collateral_slashed)
        self.m.eliminations.clear()
        self.m._contract_client.slash_miner_collateral.side_effect = None
        self.m._contract_client.slash_miner_collateral.return_value = False
        self.assertFalse(self._eliminate(EliminationReason.INACTIVE).collateral_slashed)

    def test_a_retry_after_a_failure_past_the_slash_does_not_slash_again(self):
        for reason, kwargs in ((EliminationReason.INACTIVE, {}),
                               (ACCOUNT_CLOSED, dict(elimination_drawdown_pct=4.0, slash_theta=CLOSING_SLASH))):
            with self.subTest(reason=reason):
                self.m.eliminations.clear()
                self.m._contract_client.reset_mock()
                self.m._miner_account_client.reset_mock()
                self.m._position_client.close_all_positions.side_effect = RuntimeError("position server down")
                with self.assertRaises(RuntimeError):
                    self.m.append_elimination_row(HOTKEY, reason, **kwargs)
                self.assertNotIn(HOTKEY, self.m.eliminations)
                # The account is closed as soon as its slash is taken, so nothing trades on meanwhile
                self.m._miner_account_client.close_standard_account.assert_called_once()
                self.m._miner_account_client.set_miner_bucket.assert_called_once_with(HOTKEY, MinerBucket.ELIMINATED)
                self.m._position_client.close_all_positions.side_effect = None
                self.assertTrue(self._eliminate(reason, **kwargs).collateral_slashed)
                self.assertEqual(self._slashed_theta(), [CLOSING_SLASH])
        # A close that fails right after the slash is made on the retry, which slashes nothing more
        self.m.eliminations.clear()
        self.m._contract_client.reset_mock()
        self.m._miner_account_client.reset_mock()
        self.m._miner_account_client.close_standard_account.side_effect = RuntimeError("account server down")
        with self.assertRaises(RuntimeError):
            self.m.append_elimination_row(HOTKEY, EliminationReason.INACTIVE)
        self.m._miner_account_client.close_standard_account.side_effect = None
        self.assertTrue(self._eliminate(EliminationReason.INACTIVE).collateral_slashed)
        self.assertEqual(self.m._miner_account_client.close_standard_account.call_count, 2)
        self.m._miner_account_client.set_miner_bucket.assert_called_once_with(HOTKEY, MinerBucket.ELIMINATED)
        self.assertEqual(self._slashed_theta(), [CLOSING_SLASH])

    def test_the_row_records_a_slash_taken_later(self):
        row = self._eliminate(EliminationReason.INACTIVE)
        self.assertTrue(self.m.set_collateral_slashed(HOTKEY, False))
        self.assertFalse(self.m.get_elimination(HOTKEY)["collateral_slashed"])
        self.assertTrue(self.m.set_collateral_slashed(HOTKEY, True))
        self.assertTrue(row.collateral_slashed)
        self.assertFalse(self.m.set_collateral_slashed(LEGACY, True))
        self.assertEqual(self.m._save_eliminations_to_disk.call_count, 3)

    def test_other_miners_keep_their_own_branch(self):
        row = self._eliminate(EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN, hotkey=LEGACY,
                              elimination_drawdown_pct=4.0, bucket_at_elimination=MinerBucket.MAINCOMP)
        self.assertTrue(row.collateral_slashed)
        self.m._contract_client.slash_miner_collateral_proportion.assert_called_once_with(LEGACY, 0.5)
        self.m._contract_client.slash_miner_collateral.assert_not_called()
        self.m._miner_account_client.close_standard_account.assert_not_called()
        self.m._miner_account_client.set_miner_bucket.assert_not_called()
        self._eliminate(EliminationReason.INACTIVE, hotkey=SUBACCOUNT)
        self.m._contract_client.slash_miner_collateral_proportion.assert_called_once()
        self.m._contract_client.slash_miner_collateral.assert_not_called()
        self.m._miner_account_client.close_standard_account.assert_not_called()


class TestAdminEndpoints(TestBase):
    """The admin endpoints that revert an elimination reopen a standard miner's account, never one its miner closed;
    the one that eliminates never records a close."""

    def setUp(self):
        super().setUp()
        server = object.__new__(ValidatorRestServer)
        server._get_api_key_safe = MagicMock(return_value="key")
        server.is_valid_api_key = MagicMock(return_value=True)
        server.can_access_tier = MagicMock(return_value=True)
        for name in ("_elimination_client", "_challenge_period_client", "_entity_client", "_miner_account_client",
                     "_perf_ledger_client", "_debt_ledger_client", "_position_client", "_limit_order_client"):
            setattr(server, name, MagicMock())
        server._elimination_client.get_elimination.return_value = None
        server._challenge_period_client.get_miner_bucket.return_value = MinerBucket.SUBACCOUNT_FUNDED
        server._challenge_period_client.get_bucket_history.return_value = []
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.route("/admin/revert-elimination/<hotkey>", methods=["POST"])(server.revert_elimination)
        app.route("/admin/reset/<hotkey>", methods=["POST"])(server.reset_hotkey)
        app.route("/admin/eliminate/<hotkey>", methods=["POST"])(server.eliminate_hotkey)
        self.server, self.client = server, app.test_client()

    def test_reverting_or_resetting_reopens_a_standard_miner_only(self):
        for path in ("/admin/revert-elimination/", "/admin/reset/"):
            for hotkey, reopened in ((HOTKEY, True), (LEGACY, True), (SUBACCOUNT, False)):
                with self.subTest(path=path, hotkey=hotkey):
                    self.server._miner_account_client.reset_mock()
                    self.server._entity_client.reset_mock()
                    response = self.client.post(path + hotkey, json={})
                    self.assertEqual(response.status_code, 200, response.get_json())
                    self.server._challenge_period_client.revert_elimination.assert_called_with(hotkey)
                    if reopened:
                        # The account manager answers for a hotkey without a record, so the call is made for every
                        # real hotkey
                        self.server._miner_account_client.reopen_standard_account.assert_called_once_with(hotkey, ANY)
                        self.server._entity_client.restore_subaccount.assert_not_called()
                    else:
                        self.server._miner_account_client.reopen_standard_account.assert_not_called()
                        self.server._entity_client.restore_subaccount.assert_called_once_with(hotkey)

    def test_an_account_its_miner_closed_is_not_reopened(self):
        self.server._elimination_client.get_elimination.return_value = elimination_row()
        for path in ("/admin/revert-elimination/", "/admin/reset/"):
            with self.subTest(path=path):
                self.server._elimination_client.reset_mock()
                self.server._challenge_period_client.reset_mock()
                self.server._miner_account_client.reset_mock()
                response = self.client.post(path + HOTKEY, json={})
                self.assertEqual(response.status_code, 400)
                self.assertIn("closed by its miner", response.get_json()["error"])
                self.server._elimination_client.remove_elimination.assert_not_called()
                self.server._challenge_period_client.revert_elimination.assert_not_called()
                self.server._miner_account_client.reopen_standard_account.assert_not_called()
                self.server._miner_account_client.reset_account.assert_not_called()

    def test_a_fixed_size_account_its_miner_closed_is_not_reopened(self):
        # Closed by its miner's withdrawal, row or not; an open fixed size is reverted as usual
        for closed_at_ms, status in ((NOW_MS, 400), (None, 200)):
            with self.subTest(closed=closed_at_ms is not None):
                self.server._miner_account_client.get_account.return_value = MinerAccount(
                    miner_hotkey=LEGACY, size_lock=AccountSizeLock(NOW_MS, 150_000.0, 300.0, closed_at_ms))
                response = self.client.post("/admin/revert-elimination/" + LEGACY, json={})
                self.assertEqual(response.status_code, status, response.get_json())
                if status == 400:
                    self.assertIn("closed by its miner", response.get_json()["error"])

    def test_an_admin_elimination_never_records_a_close(self):
        response = self.client.post("/admin/eliminate/" + HOTKEY, json={"reason": ACCOUNT_CLOSED.value})
        self.assertEqual(response.status_code, 400)
        self.server._elimination_client.append_elimination_row.assert_not_called()
        response = self.client.post("/admin/eliminate/" + HOTKEY, json={"reason": EliminationReason.PLAGIARISM.value})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.server._elimination_client.append_elimination_row.assert_called_once_with(HOTKEY, EliminationReason.PLAGIARISM)


class TestWithdrawEndpoint(TestBase):
    """POST /collateral/withdraw signs the consent to closing the account with the request."""

    def setUp(self):
        super().setUp()
        from bittensor_wallet import Keypair
        from vanta_api.nonce_manager import NonceManager
        self.coldkey, self.hotkey = Keypair.create_from_uri("//Alice"), Keypair.create_from_uri("//Bob")
        server = object.__new__(ValidatorRestServer)
        server._contract_client = MagicMock()
        server._contract_client.process_withdrawal_request.return_value = {"successfully_processed": True}
        server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
        server.check_vanta_cli_version = MagicMock(return_value=None)
        server.nonce_manager = NonceManager()
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.route("/collateral/withdraw", methods=["POST"])(server.withdraw_collateral)
        self.server, self.client = server, app.test_client()

    def _post(self, signed_close=None, sent_close=None):
        import json
        import uuid
        fields = {"amount": 31.0, "miner_coldkey": self.coldkey.ss58_address, "miner_hotkey": self.hotkey.ss58_address,
                  "nonce": uuid.uuid4().hex, "timestamp": TimeUtil.now_in_millis()}
        to_sign = {**fields, "close_account": True} if signed_close else fields
        signature = self.coldkey.sign(json.dumps(to_sign, sort_keys=True).encode("utf-8")).hex()
        body = {**fields, "signature": signature, "version": ValiConfig.VANTA_CLI_MINIMUM_VERSION}
        if sent_close is not None:
            body["close_account"] = sent_close
        response = self.client.post("/collateral/withdraw", json=body)
        return response.status_code, response.get_json()

    def test_the_consent_is_part_of_the_signature(self):
        self.assertEqual(self._post()[0], 200)
        self.assertEqual(self.server.contract_manager.process_withdrawal_request.call_args.kwargs["close_account"], False)
        self.assertEqual(self._post(signed_close=True, sent_close=True)[0], 200)
        self.assertEqual(self.server.contract_manager.process_withdrawal_request.call_args.kwargs["close_account"], True)
        # Consent that was not signed, or is not a boolean, is refused
        self.assertEqual(self._post(signed_close=False, sent_close=True)[0], 401)
        self.assertEqual(self._post(signed_close=True, sent_close="yes")[0], 400)
        self.assertEqual(self.server.contract_manager.process_withdrawal_request.call_count, 2)


class TestThroughTheServers(TestBase):
    """An opened account on the test servers: the withdrawal queries on the drawdown the challenge period holds, and
    an elimination that marks the account bucket while a validator that is not the mothership leaves the record to the
    mothership's broadcast."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.contract_client = cls.orchestrator.get_client('contract')
        cls.account_client = cls.orchestrator.get_client('miner_account')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')
        cls.elimination_client = cls.orchestrator.get_client('elimination')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()
        self.contract_client.clear_test_collateral_balances()
        self.metagraph_client.set_hotkeys(HOTKEYS, block_at_registration=BLOCKS, stake_weight=STAKE_WEIGHTS,
                                          standard_miner_cutoff_block=CUTOFF)
        self.asset_selection_client.process_asset_selection_request(MinerAssetClass.CRYPTO.value, NEW)
        self.contract_client.set_test_collateral_balance(NEW, int(5.5 * RAO))
        opened, message, _ = self.contract_client.open_standard_miner_account(NEW, TIER)
        self.assertTrue(opened, message)
        self.locked = self.account_client.get_standard_account(NEW).locked_theta

    def tearDown(self):
        self.contract_client.clear_test_collateral_balances()
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def _down_two_percent_today_and_four_from_the_start(self):
        # Equity 0.96 of the starting balance against a day open of 0.96 / 0.98: the static rules read 4% and 2%. The
        # high-water mark of 1.2 puts it 20% below, which would take everything if it were read
        today_ms = TimeUtil.get_start_of_day_ms(TimeUtil.now_in_millis())
        set_ok, message, _ = self.challenge_period_client.set_miner_drawdown_stats(
            NEW, {"current_equity": 0.96, "current_balance": 0.96, "daily_open_equity": 0.96 / 0.98,
                  "eod_hwm": 1.2, "last_eod_checked_ms": today_ms}, force=True)
        self.assertTrue(set_ok, message)

    def test_the_withdrawal_queries_of_an_opened_account(self):
        self.contract_client.set_test_collateral_balance(NEW, int((self.locked + 1.0) * RAO))
        unlocked = self.contract_client.query_withdrawal_request(1.0, NEW)
        self.assertTrue(unlocked["successfully_processed"], unlocked)
        self.assertEqual((unlocked["closes_account"], unlocked["slashed_amount"], unlocked["withdrawal_amount"]),
                         (False, 0.0, 1.0))
        self.assertAlmostEqual(unlocked["new_balance"], self.locked)
        closing = self.contract_client.query_withdrawal_request(1.5, NEW)
        self.assertTrue(closing["successfully_processed"], closing)
        self.assertEqual((closing["closes_account"], closing["locked_theta"], closing["new_balance"]),
                         (True, self.locked, 0.0))
        # Nothing traded yet: no drawdown, nothing slashed, the whole balance comes out
        self.assertEqual((closing["slashed_amount"], closing["slash_proportion"]), (0.0, 0.0))
        self.assertAlmostEqual(closing["withdrawal_amount"], self.locked + 1.0)
        # An opened account runs the static rules: under 3% a day and 5% below the starting balance, 2% and 4% down
        # costs 80%. The balance carries the 4% loss and the day opened 2% above it
        self._down_two_percent_today_and_four_from_the_start()
        self.account_client.process_order_sell(NEW, entry_value_usd=0, realized_pnl=-0.04 * 50_000.0, loan_repaid=0)
        closing = self.contract_client.query_withdrawal_request(1.5, NEW)
        self.assertTrue(closing["successfully_processed"], closing)
        self.assertAlmostEqual(closing["slash_proportion"], 0.8)
        self.assertAlmostEqual(closing["slashed_amount"], 0.8 * self.locked)
        self.assertAlmostEqual(closing["withdrawal_amount"], self.locked + 1.0 - 0.8 * self.locked)
        self.assertAlmostEqual(closing["intraday_drawdown_pct"], 2.0)
        self.assertAlmostEqual(closing["static_drawdown_pct"], 4.0)

    def test_an_elimination_marks_the_account_and_leaves_the_record_to_the_mothership(self):
        self._down_two_percent_today_and_four_from_the_start()
        self.elimination_client.append_elimination_row(NEW, EliminationReason.INACTIVE,
                                                       elimination_time_ms=TimeUtil.now_in_millis())
        row = self.elimination_client.get_elimination(NEW)
        self.assertEqual((row["reason"], row["collateral_slashed"]), (EliminationReason.INACTIVE.value, True))
        self.assertAlmostEqual(row["intraday_dd"], 2.0)
        self.assertAlmostEqual(row["eod_dd"], 4.0)
        self.assertEqual(self.account_client.get_account(NEW).miner_bucket, MinerBucket.ELIMINATED)
        self.assertFalse(self.account_client.get_standard_account(NEW).is_closed)
        self.assertTrue(self.elimination_client.set_collateral_slashed(NEW, False))
        self.assertFalse(self.elimination_client.get_elimination(NEW)["collateral_slashed"])
