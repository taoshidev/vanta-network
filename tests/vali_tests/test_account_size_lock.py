"""
A miner on the existing rules keeps a fixed account size from the lock start: deposits never raise it, collateral
above the theta behind it leaves freely, and a withdrawal into that theta closes the account.
"""
import contextlib
import pickle
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import template.protocol
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from shared_objects.rpc.server_registry import ServerRegistry
from vali_objects.miner_account.miner_account_server import MinerAccountServer
from tests.vali_tests.base_objects.test_base import TestBase
from tests.vali_tests.test_standard_miner_pipeline import CLIENT_PATHS, record
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.contract.validator_contract_manager import ValidatorContractManager
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.account_size_lock import AccountSizeLock
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount, MinerAccountManager
from vali_objects.utils.elimination.elimination_manager import EliminationManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode, ValiConfig

LEGACY = "legacy_miner_hotkey"
LATE = "late_legacy_hotkey"
STANDARD = "standard_miner_hotkey"
SUBACCOUNT = "entity_hotkey_1"
COLDKEY = "miner_coldkey"
DAY = ValiConfig.DAILY_MS
HOUR = 60 * 60 * 1000
START = 1_700_000_000_000  # the lock start
USD_PER_THETA = ValiConfig.COST_PER_THETA
LOCKED = 70.0
BALANCE = 100.0
ACCOUNT_CLOSED = EliminationReason.ACCOUNT_CLOSED
# Down 2% today against the main competition's 5% daily limit, 4% below its high-water mark against 8%: half
FIFTY_PERCENT = {"intraday_drawdown_pct": 2.0, "trailing_drawdown_pct": 4.0,
                 "intraday_drawdown_threshold": 0.05, "eod_drawdown_threshold": 0.08}


def deposit(theta, at_ms, first=False):
    return CollateralRecord(theta * USD_PER_THETA, theta, at_ms, first)


def locked_account(hotkey=LEGACY, closed_at_ms=None, theta=LOCKED):
    lock = AccountSizeLock(START, theta * USD_PER_THETA, theta, closed_at_ms)
    return MinerAccount(miner_hotkey=hotkey, size_lock=lock)


class TestAccountSizeLock(TestBase):

    def test_the_lock_a_collateral_balance_gives(self):
        for theta, size, locked_theta in ((0.0, ValiConfig.MIN_CAPITAL, 0.0), (5.0, ValiConfig.MIN_CAPITAL, 5.0),
                                          (300.0, 150_000.0, 300.0), (1_200.0, 500_000.0, 1_000.0), (None, 5_000.0, 0.0)):
            with self.subTest(theta=theta):
                lock = AccountSizeLock.from_collateral(theta, START)
                self.assertEqual((lock.locked_ms, lock.account_size, lock.locked_theta), (START, size, locked_theta))
        lock = AccountSizeLock.from_collateral(300.0, START)
        self.assertEqual(AccountSizeLock.from_dict({**lock.to_dict(), "unknown": 1}), lock)
        closed = lock.closed(START + 1)
        self.assertEqual((closed.is_closed, closed.closed(START + 2).closed_at_ms), (True, START + 1))
        for bad in (dict(locked_ms=START, account_size=0.0, locked_theta=1.0),
                    dict(locked_ms=START, account_size=5_000.0, locked_theta=-1.0)):
            with self.assertRaises(ValueError):
                AccountSizeLock.from_dict(bad)

    def test_the_size_is_fixed_from_the_lock_time_on(self):
        account = MinerAccount(miner_hotkey=LEGACY, collateral_records=[
            deposit(300, START - 10 * DAY, first=True), deposit(600, START - 5 * DAY)])
        account.size_lock = account.size_lock_now(START)
        self.assertEqual((account.size_lock.account_size, account.size_lock.locked_theta), (300_000.0, 600.0))
        account.add_collateral_record(deposit(900, START + DAY))
        account.add_collateral_record(deposit(100, START + 3 * DAY))
        for at_ms in (None, START, START + 2 * DAY, START + 5 * DAY):
            self.assertEqual(account.get_account_size(at_ms), 300_000.0)
        # Before the lock, the size of the time
        self.assertEqual(account.get_account_size(START - 6 * DAY), 150_000.0)
        self.assertEqual(account.get_account_size(START - 1), 300_000.0)
        self.assertEqual(account.collateral_account_size(), 50_000.0)

    def test_the_lock_is_the_size_in_force(self):
        account = MinerAccount(miner_hotkey=LEGACY, collateral_records=[
            deposit(300, START - 10 * DAY, first=True), deposit(900, START - HOUR)])
        self.assertEqual(account.size_lock_now(START).account_size, account.get_account_size())
        self.assertEqual(account.size_lock_now(START).account_size, 450_000.0)
        self.assertEqual(MinerAccount(miner_hotkey=LEGACY).size_lock_now(START).account_size, ValiConfig.MIN_CAPITAL)

    def test_a_standard_miner_reads_its_record_whatever_the_lock(self):
        account = MinerAccount(miner_hotkey=STANDARD, standard_account=record(), size_lock=AccountSizeLock(0, 1.0, 0.0))
        self.assertEqual(account.get_account_size(), record().size_traded)


class TestLockingThroughTheManager(TestBase):

    def setUp(self):
        super().setUp()
        start = patch.object(ValiConfig, "ACCOUNT_SIZE_LOCK_START_MS", START)
        start.start()
        self.addCleanup(start.stop)
        self.manager = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)
        self.manager.sync_miner_account_sizes_data({})
        self.manager.set_miner_account_size(LEGACY, 300.0, START - 10 * DAY)

    def _lock(self, hotkeys, now_ms=START + 6 * HOUR):
        return self.manager.lock_account_sizes(hotkeys, now_ms=now_ms)

    def test_nothing_is_locked_before_the_start_or_while_it_is_unset(self):
        self.assertEqual(self._lock([LEGACY], now_ms=START - 1), {})
        with patch.object(ValiConfig, "ACCOUNT_SIZE_LOCK_START_MS", None):
            self.assertEqual(self._lock([LEGACY]), {})
        self.assertIsNone(self.manager.accounts[LEGACY].size_lock)

    def test_a_miner_is_locked_at_the_size_in_force_when_first_offered(self):
        self.manager.set_miner_account_size(LEGACY, 600.0, START + HOUR)
        locks = self._lock([LEGACY], now_ms=START + 2 * HOUR)
        self.assertEqual(locks, {LEGACY: AccountSizeLock(START + 2 * HOUR, 300_000.0, 600.0)})
        self.assertEqual(self.manager.accounts[LEGACY].to_dashboard()["size_lock"], locks[LEGACY].to_dict())
        # Locked once: later deposits and later calls change nothing
        self.manager.set_miner_account_size(LEGACY, 1_000.0, START + DAY)
        self.assertEqual(self._lock([LEGACY]), {})
        self.assertEqual(self.manager.get_miner_account_size(LEGACY), 300_000.0)

    def test_subaccounts_standard_miners_and_unknown_accounts_are_not_locked(self):
        self.manager.set_miner_account_size(SUBACCOUNT, 300.0, START - DAY)
        self.manager.open_standard_account(STANDARD, record())
        self.assertEqual(self._lock([SUBACCOUNT, STANDARD, "unknown"]), {})

    def test_the_lock_rides_the_checkpoint_and_a_reset(self):
        lock = self._lock([LEGACY])[LEGACY]
        snapshot = self.manager.accounts_dict()
        self.assertEqual(MinerAccountManager._parse_accounts_dict(snapshot)[LEGACY].size_lock, lock)
        other = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)
        other.sync_miner_account_sizes_data(snapshot)
        other.set_miner_account_size(LEGACY, 900.0, START + 2 * DAY)
        self.assertEqual(other.get_miner_account_size(LEGACY), 150_000.0)
        self.manager.reset_account(LEGACY, MinerBucket.MAINCOMP)
        self.assertEqual(self.manager.accounts[LEGACY].size_lock, lock)
        # An unreadable lock loses the lock, not the account
        snapshot[LEGACY][-1]["size_lock"] = {"locked_ms": START, "account_size": -1.0, "locked_theta": 1.0}
        parsed = MinerAccountManager._parse_accounts_dict(snapshot)[LEGACY]
        self.assertIsNone(parsed.size_lock)

    def test_a_snapshot_never_unlocks_and_its_own_lock_wins(self):
        lock = self._lock([LEGACY])[LEGACY]
        snapshot = self.manager.accounts_dict()
        snapshot[LEGACY][-1].pop("size_lock")
        self.manager.sync_miner_account_sizes_data(snapshot)
        self.assertEqual(self.manager.accounts[LEGACY].size_lock, lock)
        theirs = AccountSizeLock(START, 120_000.0, 240.0)
        snapshot[LEGACY][-1]["size_lock"] = theirs.to_dict()
        self.manager.sync_miner_account_sizes_data(snapshot)
        self.assertEqual(self.manager.accounts[LEGACY].size_lock, theirs)
        # A snapshot taken before a close never reopens the account
        self.manager.close_locked_account(LEGACY, START + DAY)
        self.manager.sync_miner_account_sizes_data(snapshot)
        self.assertEqual(self.manager.accounts[LEGACY].size_lock, theirs.closed(START + DAY))

    def test_a_peer_adopts_the_mothership_locks_and_closes(self):
        self.manager.is_mothership = True
        self.manager._broadcast_to_validators = MagicMock()
        lock = self._lock([LEGACY])[LEGACY]
        sent = self.manager._broadcast_to_validators.call_args.kwargs
        self.assertEqual((sent["broadcast_name"], sent["synapse_factory"]().size_locks), ("AccountSizeLocks", {LEGACY: lock.to_dict()}))
        self.assertTrue(self.manager.close_locked_account(LEGACY, START + DAY))
        closed = lock.closed(START + DAY)
        self.assertEqual(self.manager._broadcast_to_validators.call_args.kwargs["synapse_factory"]().size_locks, {LEGACY: closed.to_dict()})
        peer = MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)
        peer.sync_miner_account_sizes_data({})
        peer.get_or_create(LEGACY).size_lock = AccountSizeLock(START + HOUR, 250_000.0, 500.0)  # its own, different
        with patch.object(peer, "verify_broadcast_sender", return_value=True):
            self.assertTrue(peer.receive_size_locks_update({LEGACY: lock.to_dict()}, sender_hotkey="mothership"))
            self.assertEqual(peer.accounts[LEGACY].size_lock, lock)
            self.assertTrue(peer.receive_size_locks_update({LEGACY: closed.to_dict()}, sender_hotkey="mothership"))
            self.assertEqual(peer.accounts[LEGACY].size_lock, closed)
            # A late open lock never reopens a closed account
            self.assertTrue(peer.receive_size_locks_update({LEGACY: lock.to_dict()}, sender_hotkey="mothership"))
            self.assertEqual(peer.accounts[LEGACY].size_lock, closed)
        with patch.object(peer, "verify_broadcast_sender", return_value=False):
            self.assertFalse(peer.receive_size_locks_update({LATE: lock.to_dict()}, sender_hotkey="someone"))
        self.assertNotIn(LATE, peer.accounts)

    def test_a_balance_below_the_locked_theta_shrinks_an_open_lock(self):
        # A withdrawal still in flight when the lock was written lands after it
        self.manager.is_mothership = True
        self.manager._broadcast_to_validators = MagicMock()
        self._lock([LEGACY])
        self.manager.set_miner_account_size(LEGACY, 10.0, START + 7 * HOUR)
        shrunk = AccountSizeLock(START + 7 * HOUR, ValiConfig.MIN_CAPITAL, 10.0)
        self.assertEqual(self.manager.accounts[LEGACY].size_lock, shrunk)
        self.assertEqual(self.manager._broadcast_to_validators.call_args.kwargs["synapse_factory"]().size_locks,
                         {LEGACY: shrunk.to_dict()})
        # A deposit afterwards does not raise it again
        self.manager.set_miner_account_size(LEGACY, 300.0, START + 8 * HOUR)
        self.assertEqual(self.manager.get_miner_account_size(LEGACY), ValiConfig.MIN_CAPITAL)
        # A closed lock stays as it was closed, and the broadcast record path shrinks too
        self.manager.close_locked_account(LEGACY, START + 9 * HOUR)
        self.manager.set_miner_account_size(LEGACY, 0.0, START + 10 * HOUR)
        self.assertEqual(self.manager.accounts[LEGACY].size_lock, shrunk.closed(START + 9 * HOUR))
        self.manager.set_miner_account_size(LATE, 400.0, START - DAY)
        self._lock([LATE])
        with patch.object(self.manager, "verify_broadcast_sender", return_value=True):
            # A record made before the lock, arriving late, changes nothing
            self.assertTrue(self.manager.receive_collateral_record_update(
                {"hotkey": LATE, "account_size": 50_000.0, "account_size_theta": 100.0, "update_time_ms": START},
                sender_hotkey="mothership"))
            self.assertEqual(self.manager.accounts[LATE].size_lock.locked_theta, 400.0)
            self.assertTrue(self.manager.receive_collateral_record_update(
                {"hotkey": LATE, "account_size": 50_000.0, "account_size_theta": 100.0, "update_time_ms": START + 11 * HOUR},
                sender_hotkey="mothership"))
        self.assertEqual(self.manager.accounts[LATE].size_lock, AccountSizeLock(START + 11 * HOUR, 50_000.0, 100.0))

    def test_the_synapse_reaches_the_manager_and_pickles(self):
        with patch.object(ServerRegistry, "register"):
            server = MinerAccountServer(running_unit_tests=True, start_server=False, connection_mode=RPCConnectionMode.LOCAL)
        server._manager.sync_miner_account_sizes_data({})
        lock = AccountSizeLock(START, 150_000.0, 300.0)
        synapse = pickle.loads(pickle.dumps(template.protocol.AccountSizeLocks(size_locks={LEGACY: lock.to_dict()})))
        synapse.dendrite.hotkey = "mothership"
        with patch.object(server._manager, "verify_broadcast_sender", return_value=True) as verify:
            answered = server.receive_size_locks_rpc(synapse)
        self.assertTrue(answered.successfully_processed, answered.error_message)
        self.assertEqual(verify.call_args.args[0], "mothership")
        self.assertEqual(server._manager.accounts[LEGACY].size_lock, lock)

    def test_the_minimum_collateral_rule_reads_the_collateral(self):
        # Trading on the free account when the lock starts, then posting the 300 theta: the penalty clears, not the size
        self.assertEqual(self._lock(["fresh_hotkey"]), {})  # no account yet
        self.manager.get_or_create("fresh_hotkey")
        self.assertEqual(self._lock(["fresh_hotkey"])["fresh_hotkey"].account_size, ValiConfig.MIN_CAPITAL)
        self.manager.set_miner_account_size("fresh_hotkey", 300.0, START + DAY)
        self.assertEqual(self.manager.get_miner_account_size("fresh_hotkey"), ValiConfig.MIN_CAPITAL)
        self.assertEqual(self.manager.get_collateral_account_size("fresh_hotkey"), 150_000.0)
        self.assertEqual(self.manager.get_all_collateral_account_sizes()["fresh_hotkey"], 150_000.0)
        self.assertEqual(ValidatorContractManager.min_collateral_penalty(150_000.0), 1)
        self.assertIsNone(self.manager.get_collateral_account_size("unknown_hotkey"))
        # A standard miner posts no minimum: the statistics read it as meeting it
        self.manager.open_standard_account(STANDARD, record())
        self.assertEqual(ValidatorContractManager.min_collateral_penalty(self.manager.get_collateral_account_size(STANDARD)), 1)
        self.assertEqual(ValidatorContractManager.min_collateral_penalty(
            self.manager.get_all_collateral_account_sizes()[STANDARD]), 1)

    def test_closing_a_locked_account(self):
        self._lock([LEGACY])
        self.assertTrue(self.manager.close_locked_account(LEGACY, START + DAY))
        self.assertTrue(self.manager.close_locked_account(LEGACY, START + 2 * DAY))
        self.assertEqual(self.manager.accounts[LEGACY].size_lock.closed_at_ms, START + DAY)
        self.assertFalse(self.manager.close_locked_account("unknown_hotkey", START))


class TestLockingInTheRefresh(TestBase):
    """The challenge period manager offers the account server every unlocked miner on the existing rules that has
    traded."""

    def setUp(self):
        super().setUp()
        stack = contextlib.ExitStack()
        for path in CLIENT_PATHS:
            stack.enter_context(patch(path))
        self.addCleanup(stack.close)
        self.manager = ChallengePeriodManager(is_backtesting=True)
        stack.enter_context(patch.object(self.manager, "_save_to_disk"))
        self.client = self.manager._miner_account_client
        for hotkey, bucket in ((LEGACY, MinerBucket.CHALLENGE), (LATE, MinerBucket.MAINCOMP),
                               ("locked_hotkey", MinerBucket.MAINCOMP), ("no_orders", MinerBucket.MAINCOMP),
                               (STANDARD, MinerBucket.SUBACCOUNT_CHALLENGE), (SUBACCOUNT, MinerBucket.SUBACCOUNT_FUNDED)):
            self.manager.set_miner_bucket(hotkey, bucket, START - DAY)
        self.accounts = {hotkey: MinerAccount(miner_hotkey=hotkey) for hotkey in (LEGACY, LATE, "no_orders", SUBACCOUNT)}
        self.accounts["locked_hotkey"] = locked_account("locked_hotkey")
        self.accounts[STANDARD] = MinerAccount(miner_hotkey=STANDARD, standard_account=record())
        self.first_orders = {hotkey: START - DAY for hotkey in self.accounts if hotkey != "no_orders"}

    def test_every_unlocked_miner_on_the_existing_rules_is_offered(self):
        lock = AccountSizeLock(START, 150_000.0, 300.0)
        self.client.lock_account_sizes.return_value = {LEGACY: lock}
        self.manager._lock_account_sizes(self.accounts, self.first_orders)
        self.client.lock_account_sizes.assert_called_once_with([LEGACY, LATE])
        self.assertEqual(self.accounts[LEGACY].size_lock, lock)
        self.assertIsNone(self.accounts[LATE].size_lock)

    def test_the_existing_rules_buckets_are_pushed_to_the_accounts(self):
        self.manager._push_existing_rules_buckets(self.accounts)
        self.client.set_miner_buckets.assert_called_once_with({
            LEGACY: MinerBucket.CHALLENGE, LATE: MinerBucket.MAINCOMP, "locked_hotkey": MinerBucket.MAINCOMP,
            "no_orders": MinerBucket.MAINCOMP})
        self.assertEqual(self.accounts[LEGACY].miner_bucket, MinerBucket.CHALLENGE)
        self.client.set_miner_buckets.reset_mock()
        self.manager._push_existing_rules_buckets(self.accounts)
        self.client.set_miner_buckets.assert_not_called()

    def test_a_subaccount_or_standard_miner_flagged_for_plagiarism_keeps_its_bucket(self):
        self.accounts[SUBACCOUNT].miner_bucket = MinerBucket.SUBACCOUNT_FUNDED
        self.accounts[STANDARD].miner_bucket = MinerBucket.SUBACCOUNT_CHALLENGE
        self.manager.sync_plagiarism_miners([LEGACY, SUBACCOUNT, STANDARD], START)
        self.manager._push_existing_rules_buckets(self.accounts)
        pushed = self.client.set_miner_buckets.call_args.args[0]
        self.assertEqual(pushed[LEGACY], MinerBucket.PLAGIARISM)
        self.assertNotIn(SUBACCOUNT, pushed)
        self.assertNotIn(STANDARD, pushed)

        self.client.set_miner_buckets.reset_mock()
        self.manager.sync_plagiarism_miners([], START + HOUR)
        self.manager._push_existing_rules_buckets(self.accounts)
        self.client.set_miner_buckets.assert_called_once_with({LEGACY: MinerBucket.CHALLENGE})
        self.assertEqual(self.accounts[SUBACCOUNT].miner_bucket, MinerBucket.SUBACCOUNT_FUNDED)
        self.assertEqual(self.accounts[STANDARD].miner_bucket, MinerBucket.SUBACCOUNT_CHALLENGE)

    def test_nothing_is_asked_without_candidates_and_a_failure_never_stops_the_refresh(self):
        self.accounts[LEGACY].size_lock = self.accounts[LATE].size_lock = AccountSizeLock(START, 5_000.0, 0.0)
        self.manager._lock_account_sizes(self.accounts, self.first_orders)
        self.client.lock_account_sizes.assert_not_called()
        self.accounts[LEGACY].size_lock = None
        self.client.lock_account_sizes.side_effect = RuntimeError("account server down")
        self.manager._lock_account_sizes(self.accounts, self.first_orders)
        self.assertIsNone(self.accounts[LEGACY].size_lock)


class TestLockedWithdrawals(TestBase):
    """query_withdrawal_request and process_withdrawal_request for a locked account on the mothership, with the chain
    and the services mocked."""

    def setUp(self):
        super().setUp()
        m = object.__new__(ValidatorContractManager)
        m.is_mothership = True
        m.is_testnet = False
        m.running_unit_tests = False
        m._account_locks = {}
        m._account_locks_guard = threading.Lock()
        m.wallet = MagicMock()
        m.collateral_manager = MagicMock()
        m.collateral_manager.withdraw.side_effect = lambda amount, **kwargs: SimpleNamespace(rao=amount)
        m.get_miner_collateral_balance = MagicMock(return_value=BALANCE)
        m._set_miner_account_size = MagicMock()
        m._miner_account_client = MagicMock()
        m._miner_account_client.get_standard_account.return_value = None
        m._miner_account_client.get_account.return_value = locked_account()
        m._position_client = MagicMock()
        m._position_client.get_positions_for_one_hotkey.return_value = []
        m._challenge_period_client = MagicMock()
        m._challenge_period_client.get_drawdown_stats.return_value = FIFTY_PERCENT
        m._challenge_period_client.get_miner_bucket.return_value = MinerBucket.MAINCOMP
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

    def _elimination_lands(self, **kwargs):
        """The elimination server slashes the closing amount, closes the lock and writes the row."""
        self.m._elimination_client.get_elimination.return_value = {
            "hotkey": LEGACY, "reason": ACCOUNT_CLOSED.value, "collateral_slashed": True,
            "bucket_at_elimination": MinerBucket.MAINCOMP.value}
        self.m._miner_account_client.get_account.return_value = locked_account(closed_at_ms=START)
        self.m.get_miner_collateral_balance.return_value -= kwargs["slash_theta"]

    def _open_position(self):
        return SimpleNamespace(is_open_position=True, trade_pair=SimpleNamespace(trade_pair_id="BTCUSD"))

    def _withdrawn_theta(self):
        return self.m.collateral_manager.withdraw.call_args.kwargs["amount"] / 10 ** 9

    def test_the_collateral_above_the_minimum_leaves_without_a_slash_whatever_the_positions(self):
        self.m.get_miner_collateral_balance.return_value = 400.0
        self.m._position_client.get_positions_for_one_hotkey.return_value = [self._open_position()]
        result = self.m.query_withdrawal_request(100.0, LEGACY)
        self.assertTrue(result["successfully_processed"], result)
        self.assertEqual((result["closes_account"], result["slashed_amount"], result["locked_theta"]), (False, 0.0, LOCKED))
        processed = self.m.process_withdrawal_request(100.0, COLDKEY, LEGACY)
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertEqual((processed["returned_amount"], processed["account_closed"]), (100.0, False))
        self.m._elimination_client.append_elimination_row.assert_not_called()

    def test_the_collateral_below_the_minimum_leaves_flat_and_slashed_on_the_drawdown(self):
        # 4% below the high-water mark against the main competition's 8%: half of the part below 300 theta
        account = locked_account()
        account.total_realized_pnl = -0.04 * account.get_account_size()
        self.m._miner_account_client.get_account.return_value = account
        self.m.get_miner_collateral_balance.return_value = 400.0
        self.m._position_client.get_positions_for_one_hotkey.return_value = [self._open_position()]
        self.assertIn("close all positions", self.m.query_withdrawal_request(250.0, LEGACY)["error_message"])
        self.m._position_client.get_positions_for_one_hotkey.return_value = []
        result = self.m.query_withdrawal_request(250.0, LEGACY)
        self.assertNotIn("closes_account", result)
        self.assertAlmostEqual(result["slashed_amount"], 0.5 * 150.0)
        processed = self.m.process_withdrawal_request(250.0, COLDKEY, LEGACY)
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertAlmostEqual(self.m.collateral_manager.slash.call_args.kwargs["amount"] / 10 ** 9, 75.0)
        self.assertAlmostEqual(self._withdrawn_theta(), 250.0 - 75.0)
        self.m._elimination_client.append_elimination_row.assert_not_called()

    def test_a_withdrawal_into_the_locked_theta_closes_the_account(self):
        self.m._position_client.get_positions_for_one_hotkey.return_value = [self._open_position()]
        self.assertIn("close all positions", self.m.query_withdrawal_request(31.0, LEGACY)["error_message"])
        self.m._position_client.get_positions_for_one_hotkey.return_value = []
        result = self.m.query_withdrawal_request(31.0, LEGACY)
        # Below 300 theta the closing slash takes in the whole balance, not just the locked theta
        self.assertEqual((result["closes_account"], result["slash_proportion"]), (True, 0.5))
        self.assertAlmostEqual(result["slashed_amount"], 0.5 * BALANCE)
        self.assertAlmostEqual(result["withdrawal_amount"], BALANCE - 0.5 * BALANCE)
        # Without the consent nothing happens
        refused = self.m.process_withdrawal_request(31.0, COLDKEY, LEGACY)
        self.assertFalse(refused["successfully_processed"])
        self.m._elimination_client.append_elimination_row.assert_not_called()
        processed = self.m.process_withdrawal_request(31.0, COLDKEY, LEGACY, close_account=True)
        self.assertTrue(processed["successfully_processed"], processed)
        self.assertTrue(processed["account_closed"])
        kwargs = self.m._elimination_client.append_elimination_row.call_args.kwargs
        self.assertEqual((kwargs["reason"], kwargs["slash_theta"]), (ACCOUNT_CLOSED, 0.5 * BALANCE))
        self.assertAlmostEqual(self._withdrawn_theta(), BALANCE - 0.5 * BALANCE)
        self.m.collateral_manager.slash.assert_not_called()

    def test_a_lock_of_at_least_the_minimum_frees_everything_above_it_and_slashes_only_it_on_closing(self):
        self.m._miner_account_client.get_account.return_value = locked_account(theta=350.0)
        self.m.get_miner_collateral_balance.return_value = 400.0
        self.m._position_client.get_positions_for_one_hotkey.return_value = [self._open_position()]
        free = self.m.query_withdrawal_request(50.0, LEGACY)
        self.assertEqual((free["closes_account"], free["slashed_amount"]), (False, 0.0))
        self.m._position_client.get_positions_for_one_hotkey.return_value = []
        closing = self.m.query_withdrawal_request(60.0, LEGACY)
        self.assertTrue(closing["closes_account"])
        self.assertAlmostEqual(closing["slashed_amount"], 0.5 * 350.0)

    def test_a_lock_closed_without_its_row_closes_again_without_a_slash(self):
        self.m._miner_account_client.get_account.return_value = locked_account(closed_at_ms=START)
        result = self.m.query_withdrawal_request(1.0, LEGACY)
        self.assertEqual((result["closes_account"], result["slashed_amount"]), (True, 0.0))
        self.m._challenge_period_client.get_drawdown_stats.assert_not_called()

    def test_an_eliminated_locked_account_withdraws_as_every_other_miner(self):
        self.m._elimination_client.get_elimination.return_value = {
            "hotkey": LEGACY, "reason": EliminationReason.INACTIVE.value, "collateral_slashed": True,
            "bucket_at_elimination": MinerBucket.MAINCOMP.value}
        result = self.m.query_withdrawal_request(31.0, LEGACY)
        self.assertTrue(result["successfully_processed"], result)
        self.assertNotIn("closes_account", result)
        self.assertEqual(result["slashed_amount"], 0)


class TestClosingElimination(TestBase):
    """append_elimination_row for a locked account on the existing rules, on a bare EliminationManager."""

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
        m._challenge_period_client.get_miner_bucket.return_value = MinerBucket.MAINCOMP
        m._entity_collateral_client = MagicMock()
        m._limit_order_client = MagicMock()
        m._position_client = MagicMock()
        m._miner_account_client = MagicMock()
        m._miner_account_client.get_standard_account.return_value = None
        m._miner_account_client.get_account.return_value = locked_account()
        m._contract_client = MagicMock()
        m._contract_client.slash_miner_collateral.return_value = True
        m._contract_client.slash_miner_collateral_proportion.return_value = True
        self.m = m

    def _slashed_theta(self):
        return [call.args[1] for call in self.m._contract_client.slash_miner_collateral.call_args_list]

    def test_the_close_takes_the_closing_slash_and_closes_the_lock(self):
        self.m.append_elimination_row(LEGACY, ACCOUNT_CLOSED, elimination_drawdown_pct=4.0, slash_theta=35.0)
        row = self.m.eliminations[LEGACY]
        self.assertEqual((row.reason, row.collateral_slashed), (ACCOUNT_CLOSED.value, True))
        self.assertEqual(self._slashed_theta(), [35.0])
        self.m._contract_client.slash_miner_collateral_proportion.assert_not_called()
        self.m._miner_account_client.close_locked_account.assert_called_once_with(LEGACY, row.elimination_initiated_time_ms)
        self.m._miner_account_client.set_miner_bucket.assert_called_once_with(LEGACY, MinerBucket.ELIMINATED)

    def test_a_closed_lock_is_never_slashed_again(self):
        self.m._miner_account_client.get_account.return_value = locked_account(closed_at_ms=START)
        for reason, kwargs in ((ACCOUNT_CLOSED, dict(slash_theta=35.0)), (EliminationReason.INACTIVE, {})):
            with self.subTest(reason=reason):
                self.m.eliminations.clear()
                self.m.append_elimination_row(LEGACY, reason, elimination_drawdown_pct=4.0, **kwargs)
                self.assertEqual(self._slashed_theta(), [])
                self.m._contract_client.slash_miner_collateral_proportion.assert_not_called()

    def test_a_failed_closing_slash_leaves_the_account_open(self):
        self.m._contract_client.slash_miner_collateral.return_value = False
        with self.assertRaises(RuntimeError):
            self.m.append_elimination_row(LEGACY, ACCOUNT_CLOSED, elimination_drawdown_pct=4.0, slash_theta=35.0)
        self.assertEqual(self.m.eliminations, {})
        self.m._miner_account_client.close_locked_account.assert_not_called()

    def test_a_close_with_no_account_to_close_raises(self):
        self.m._miner_account_client.get_account.return_value = MinerAccount(miner_hotkey=LEGACY)
        with self.assertRaises(RuntimeError):
            self.m.append_elimination_row(LEGACY, ACCOUNT_CLOSED, elimination_drawdown_pct=4.0, slash_theta=35.0)
        self.assertEqual(self.m.eliminations, {})
        self.assertEqual(self._slashed_theta(), [])
        self.m._contract_client.slash_miner_collateral_proportion.assert_not_called()

    def test_any_other_reason_slashes_as_for_every_other_miner(self):
        self.m.append_elimination_row(LEGACY, EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN,
                                      elimination_drawdown_pct=4.0)
        self.m._contract_client.slash_miner_collateral_proportion.assert_called_once_with(LEGACY, 0.5)
        self.assertEqual(self._slashed_theta(), [])
        self.m._miner_account_client.close_locked_account.assert_not_called()


class TestThroughTheServers(TestBase):

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.account_client = cls.orchestrator.get_client('miner_account')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()

    def tearDown(self):
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def test_the_calls_round_trip(self):
        self.account_client.set_miner_account_size(LEGACY, 300.0, timestamp_ms=START)
        # The lock start is unset in the servers' configuration: nothing is locked
        self.assertEqual(self.account_client.lock_account_sizes([LEGACY]), {})
        self.assertFalse(self.account_client.close_locked_account(LEGACY, START))
        self.assertEqual(self.account_client.get_collateral_account_size(LEGACY), 150_000.0)
        self.assertEqual(self.account_client.get_all_collateral_account_sizes()[LEGACY], 150_000.0)
