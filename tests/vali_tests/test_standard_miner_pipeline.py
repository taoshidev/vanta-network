"""
A standard miner on the standard account pipeline under its own hotkey: the daily loss rule and the auto-promotion
candidacy read from its record, the funded leverage tier, the pro size the mothership writes to the record and the
peers follow, the size traded at each bucket move, the record broadcast, the penalties it is spared, and the hops
run through the test servers.
"""
import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from tests.vali_tests.test_standard_miner_opening import BLOCKS, CUTOFF, HOTKEYS, NEW, RAO, STAKE_WEIGHTS, TIER
from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_manager import AutoPromotionStats, ChallengePeriodManager
from vali_objects.contract.validator_contract_manager import ValidatorContractManager
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import MinerAccount, MinerAccountManager
from vali_objects.miner_account.standard_miner import SizingRejected, StandardMinerAccount, bucket_sizing
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from vali_objects.vali_dataclasses.ledger.penalty.penalty_ledger import PenaltyLedgerManager
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger

HOTKEY = "standard_miner_hotkey"
OTHER = "other_standard_miner_hotkey"
LEGACY = "legacy_miner_hotkey"
SUBACCOUNT = "entity_hotkey_1"
MOTHERSHIP = "mothership_hotkey"
PEER = "peer_validator_hotkey"
NOW_MS = 1_700_000_000_000
SIZE = 50_000.0
PRO_SIZE = 3 * SIZE
FUNDED_TIER = ValiConfig.STANDARD_MINER_FUNDED_LEVERAGE_TIER
CLIENT_PATHS = [
    "vali_objects.challenge_period.challengeperiod_manager." + name for name in (
        "PerfLedgerClient", "PositionManagerClient", "EliminationClient", "PlagiarismClient", "MinerAccountClient",
        "CommonDataClient", "AssetSelectionClient", "DebtLedgerClient", "LimitOrderClient", "EntityClient")
]


def record(**overrides) -> StandardMinerAccount:
    values = dict(opened_at_ms=NOW_MS - 90 * ValiConfig.DAILY_MS, opening_leverage_tier=1, asset_class="crypto",
                  account_size=SIZE, locked_theta=3.5, opening_price_usd=100.0)
    values.update(overrides)
    return StandardMinerAccount(**values)


def account(hotkey=HOTKEY, standard_account=None, hl_address=None) -> MinerAccount:
    return MinerAccount(miner_hotkey=hotkey, standard_account=standard_account, hl_address=hl_address,
                        leverage_tier=1)


class TestBucketSizing(TestBase):
    """The record after each bucket move, with no fee."""

    def test_the_pro_track_records_the_size_given_and_trades_it_past_the_transition(self):
        granted = bucket_sizing(record(), MinerBucket.PRO_CHALLENGE_TRANSITION, PRO_SIZE)
        self.assertEqual((granted.pro_account_size, granted.trading_account_size, granted.size_traded),
                         (PRO_SIZE, None, SIZE))
        live = bucket_sizing(granted, MinerBucket.PRO_CHALLENGE_FROM_STANDARD)
        self.assertEqual((live.pro_account_size, live.trading_account_size, live.size_traded),
                         (PRO_SIZE, PRO_SIZE, PRO_SIZE))
        self.assertEqual(bucket_sizing(live, MinerBucket.PRO_FUNDED), live)
        back = bucket_sizing(live, MinerBucket.SUBACCOUNT_FUNDED)
        self.assertEqual((back.pro_account_size, back.trading_account_size, back.size_traded), (PRO_SIZE, None, SIZE))
        self.assertEqual(bucket_sizing(record(), MinerBucket.SUBACCOUNT_FUNDED), record())
        # A size given on a later hop replaces the recorded one
        self.assertEqual(bucket_sizing(granted, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 5 * SIZE).trading_account_size,
                         5 * SIZE)

    def test_what_the_pro_track_refuses(self):
        for bucket in (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
                       MinerBucket.PRO_FUNDED):
            with self.subTest(bucket=bucket), self.assertRaises(SizingRejected):
                bucket_sizing(record(), bucket)  # no pro size granted yet
        for size in (SIZE, 1.5 * SIZE, 0.0, -1.0, True, "150000", float("nan"), ValiConfig.MAX_PRO_ACCOUNT_SIZE + 1):
            with self.subTest(size=size), self.assertRaises(SizingRejected):
                bucket_sizing(record(), MinerBucket.PRO_CHALLENGE_TRANSITION, size)


class TestPipelineOnARealHotkey(TestBase):
    """The challenge period manager's reads and moves for a standard miner, next to a subaccount's and a regular
    miner's on the existing rules, with every client mocked."""

    def setUp(self):
        super().setUp()
        stack = contextlib.ExitStack()
        for path in CLIENT_PATHS:
            stack.enter_context(patch(path))
        self.addCleanup(stack.close)
        self.manager = ChallengePeriodManager(is_backtesting=True)
        stack.enter_context(patch.object(self.manager, "_sync_buckets_to_accounts"))
        stack.enter_context(patch.object(self.manager, "_save_to_disk"))
        self.accounts = self.manager._miner_account_client
        self.entity = self.manager._entity_client
        self.records = {HOTKEY: record()}
        self.accounts.get_standard_account.side_effect = self.records.get
        self.accounts.set_standard_account.side_effect = self.records.__setitem__
        self.entity.get_subaccount_info_for_synthetic.side_effect = (
            lambda hotkey: {"initial_bucket": None} if hotkey == SUBACCOUNT else None)
        self.entity.apply_bucket_account_size.return_value = (True, "account size set")
        self.entity.snapshot_bucket_account_size.return_value = {"account_size": SIZE}
        self.entity.is_mothership.return_value = True

    def _in(self, hotkey, bucket, since_ms=NOW_MS - ValiConfig.DAILY_MS):
        self.manager.set_miner_bucket(hotkey, bucket, since_ms)
        return self.manager.miner_states[hotkey]

    def _bucket(self, hotkey):
        return self.manager.miner_states[hotkey].current_bucket

    def test_the_daily_loss_is_measured_from_the_balance_in_the_standard_buckets(self):
        for bucket, expected in ((MinerBucket.SUBACCOUNT_CHALLENGE, True), (MinerBucket.SUBACCOUNT_FUNDED, True),
                                 (MinerBucket.PRO_CHALLENGE_TRANSITION, True),
                                 (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, False), (MinerBucket.PRO_FUNDED, False)):
            with self.subTest(bucket=bucket):
                self._in(HOTKEY, bucket)
                self.assertEqual(self.manager._daily_loss_from_balance(HOTKEY, account(standard_account=record())), expected)
        self._in(LEGACY, MinerBucket.CHALLENGE)
        self.assertFalse(self.manager._daily_loss_from_balance(LEGACY, account(LEGACY)))
        self._in(SUBACCOUNT, MinerBucket.SUBACCOUNT_FUNDED)
        self.assertTrue(self.manager._daily_loss_from_balance(SUBACCOUNT, account(SUBACCOUNT)))
        self.assertFalse(self.manager._daily_loss_from_balance(SUBACCOUNT, account(SUBACCOUNT, hl_address="0xabc")))

    def test_candidacy_for_auto_promotion(self):
        self.assertTrue(self.manager._is_auto_promotion_candidate(HOTKEY, account(standard_account=record())))
        self.assertFalse(self.manager._is_auto_promotion_candidate(LEGACY, account(LEGACY)))
        self.assertTrue(self.manager._is_auto_promotion_candidate(SUBACCOUNT, account(SUBACCOUNT)))
        self.entity.get_subaccount_info_for_synthetic.side_effect = lambda hotkey: {"initial_bucket": "SUBACCOUNT_FUNDED"}
        self.manager._initial_buckets.clear()
        self.assertFalse(self.manager._is_auto_promotion_candidate(SUBACCOUNT, account(SUBACCOUNT)))

    def test_passing_the_challenge_puts_a_standard_miner_on_the_funded_tier(self):
        self._in(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE)
        self._in(SUBACCOUNT, MinerBucket.SUBACCOUNT_CHALLENGE)
        self.manager.promote_hotkeys({HOTKEY: MinerBucket.SUBACCOUNT_CHALLENGE,
                                      SUBACCOUNT: MinerBucket.SUBACCOUNT_CHALLENGE}, NOW_MS)
        self.assertEqual((self._bucket(HOTKEY), self._bucket(SUBACCOUNT)),
                         (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_FUNDED))
        self.accounts.set_leverage_tier.assert_called_once_with(HOTKEY, FUNDED_TIER)
        self.assertEqual(self.records[HOTKEY], record())  # nothing to resize
        self.accounts.reset_account.assert_any_call(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED)
        self.entity.apply_bucket_account_size.assert_called_once_with(
            SUBACCOUNT, MinerBucket.SUBACCOUNT_FUNDED, waive_promotion_fee=False)

    def test_an_admin_move_into_funded_puts_a_standard_miner_on_the_funded_tier_too(self):
        self._in(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE)
        self._in(LEGACY, MinerBucket.CHALLENGE)
        self.assertTrue(self.manager.admin_set_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS)[0])
        self.assertTrue(self.manager.admin_set_bucket(LEGACY, MinerBucket.MAINCOMP, NOW_MS)[0])
        self.accounts.set_leverage_tier.assert_called_once_with(HOTKEY, FUNDED_TIER)

    def test_the_mothership_grants_the_pro_size_on_the_record_and_a_peer_waits_for_it(self):
        state = self._in(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED)
        stats = AutoPromotionStats(calmar=2.0, daily_consistency=0.3, max_drawdown=0.97, trading_days=60)
        # A peer moves nothing until the record carries the mothership's pro size
        self.entity.is_mothership.return_value = False
        self.assertFalse(self.manager._move_to_auto_promotion(
            HOTKEY, account(standard_account=record()), stats, ("steady_track", 5), NOW_MS))
        self.assertEqual((self._bucket(HOTKEY), self.records[HOTKEY]), (MinerBucket.SUBACCOUNT_FUNDED, record()))
        # The mothership sizes it from the path's multiple and the move follows
        self.entity.is_mothership.return_value = True
        self.manager._mothership = None  # the answer is read once, so forget the peer's
        self.assertTrue(self.manager._move_to_auto_promotion(
            HOTKEY, account(standard_account=record()), stats, ("steady_track", 5), NOW_MS))
        self.assertEqual(self._bucket(HOTKEY), MinerBucket.PRO_CHALLENGE_TRANSITION)
        self.assertEqual(self.records[HOTKEY], record(pro_account_size=5 * SIZE))
        self.assertEqual(state.auto_promotion.pro_account_size, 5 * SIZE)
        self.entity.apply_bucket_account_size.assert_not_called()
        self.entity.snapshot_bucket_account_size.assert_not_called()
        # The peer, once the record has the size, moves on it without sizing anything itself
        self.records[OTHER] = record(pro_account_size=PRO_SIZE)
        self._in(OTHER, MinerBucket.SUBACCOUNT_FUNDED)
        self.entity.is_mothership.return_value = False
        self.manager._mothership = None
        self.assertTrue(self.manager._move_to_auto_promotion(
            OTHER, account(OTHER, standard_account=self.records[OTHER]), stats, ("fast_track", 3), NOW_MS))
        self.assertEqual(self._bucket(OTHER), MinerBucket.PRO_CHALLENGE_TRANSITION)
        self.assertEqual(self.records[OTHER], record(pro_account_size=PRO_SIZE))

    def test_leaving_the_transition_trades_the_pro_size_on_all_markets(self):
        self.records[HOTKEY] = record(pro_account_size=PRO_SIZE)
        self._in(HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION, NOW_MS - 10 * ValiConfig.DAILY_MS)
        self.manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_TRANSITION}, NOW_MS)
        self.assertEqual(self._bucket(HOTKEY), MinerBucket.PRO_CHALLENGE_FROM_STANDARD)
        self.assertEqual(self.records[HOTKEY], record(pro_account_size=PRO_SIZE, trading_account_size=PRO_SIZE))
        self.manager._asset_selection_client.process_asset_selection_request.assert_called_once_with(
            MinerAssetClass.ALL_MARKETS.value, HOTKEY, overwrite=True)
        self.accounts.reset_account.assert_called_once_with(HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD)
        self.entity.apply_bucket_account_size.assert_not_called()
        # Passing the pro challenge keeps the account and its sizing
        self.manager.promote_hotkeys({HOTKEY: MinerBucket.PRO_CHALLENGE_FROM_STANDARD}, NOW_MS)
        self.assertEqual(self._bucket(HOTKEY), MinerBucket.PRO_FUNDED)
        self.assertEqual(self.records[HOTKEY], record(pro_account_size=PRO_SIZE, trading_account_size=PRO_SIZE))

    def test_a_failure_before_the_positions_are_closed_leaves_the_record_and_the_tier_alone(self):
        self.records[HOTKEY] = record(pro_account_size=PRO_SIZE)
        self._in(HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION, NOW_MS - 10 * ValiConfig.DAILY_MS)
        self._in(OTHER, MinerBucket.SUBACCOUNT_CHALLENGE)
        self.records[OTHER] = record()
        self.manager._position_client.close_all_positions.side_effect = RuntimeError("position server unavailable")
        for hotkey, bucket in ((HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION), (OTHER, MinerBucket.SUBACCOUNT_CHALLENGE)):
            with self.subTest(hotkey=hotkey), self.assertRaises(RuntimeError):
                self.manager.promote_hotkeys({hotkey: bucket}, NOW_MS)
            self.assertEqual(self._bucket(hotkey), bucket)
        self.assertEqual(self.records[HOTKEY], record(pro_account_size=PRO_SIZE))
        self.manager._asset_selection_client.process_asset_selection_request.assert_not_called()
        self.accounts.set_leverage_tier.assert_not_called()
        self.accounts.set_standard_account.assert_not_called()

    def test_the_refresh_promotes_a_standard_miner_on_the_returns_target(self):
        from vali_objects.challenge_period.challengeperiod_manager import DrawdownStats
        self._in(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE)
        self.manager.miner_states[HOTKEY].drawdown = DrawdownStats(
            current_equity=1.12, current_balance=1.12, eod_hwm=1.0, last_eod_equity=1.0,
            last_eod_checked_ms=NOW_MS - ValiConfig.DAILY_MS)
        self.manager._position_client.get_first_order_times.return_value = {}
        self.manager._position_client.filtered_positions_for_scoring.return_value = ({}, {})
        self.manager._position_client.get_positions_for_hotkeys.return_value = {}
        self.manager._elimination_client.get_eliminated_hotkeys.return_value = []
        self.manager._plagiarism_client.get_plagiarism_miners.return_value = []
        self.accounts.get_accounts.return_value = {HOTKEY: account(standard_account=record())}
        self.manager._perf_ledger_client.filtered_ledger_for_scoring.return_value = {}
        self.manager._asset_selection_client.get_asset_selections.return_value = {HOTKEY: MinerAssetClass.CRYPTO}
        with patch.object(self.manager, "_refresh_drawdown_cache"), patch.object(self.manager, "_refresh_rank_cache"):
            self.manager.refresh(current_time_ms=NOW_MS)
        self.assertEqual(self._bucket(HOTKEY), MinerBucket.SUBACCOUNT_FUNDED)
        self.accounts.set_leverage_tier.assert_called_once_with(HOTKEY, FUNDED_TIER)
        self.assertEqual(self.records[HOTKEY], record())

    def test_a_pro_bucket_needs_a_granted_size_and_a_regular_miner_needs_no_sizing(self):
        self._in(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED)
        moved, message = self.manager.admin_set_bucket(HOTKEY, MinerBucket.PRO_FUNDED, NOW_MS)
        self.assertFalse(moved)
        self.assertIn("pro account size", message)
        self.assertEqual(self._bucket(HOTKEY), MinerBucket.SUBACCOUNT_FUNDED)
        self._in(LEGACY, MinerBucket.MAINCOMP)
        self.assertTrue(self.manager.admin_set_bucket(LEGACY, MinerBucket.PRO_FUNDED, NOW_MS)[0])
        self.accounts.set_standard_account.assert_not_called()

    def test_a_standard_miner_may_enter_the_transition_and_a_failed_move_restores_its_record(self):
        self._in(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED)
        self.assertEqual(self.manager.can_admin_set_bucket(HOTKEY, MinerBucket.PRO_CHALLENGE_TRANSITION), (True, ""))
        with patch.object(self.manager, "admin_set_bucket", return_value=(False, "refused")):
            moved, _ = self.manager.promote_subaccount(HOTKEY, NOW_MS, PRO_SIZE, from_bucket=MinerBucket.SUBACCOUNT_FUNDED)
        self.assertFalse(moved)
        self.assertEqual(self.records[HOTKEY], record())
        self.assertEqual(self.accounts.set_standard_account.call_count, 2)  # the grant, then the record put back


class TestRecordBroadcast(TestBase):
    """The mothership tells the other validators every record it writes; a peer takes it as it comes."""

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

    def _receive(self, data, sender=MOTHERSHIP) -> bool:
        return self.peer.receive_standard_miner_record_update(data, sender_hotkey=sender)

    def test_every_record_write_on_the_mothership_is_broadcast(self):
        self.mothership.set_standard_account(HOTKEY, record(pro_account_size=PRO_SIZE))
        self.mothership._broadcast_to_validators.assert_called_once()
        kwargs = self.mothership._broadcast_to_validators.call_args.kwargs
        self.assertEqual(kwargs["broadcast_name"], "StandardMinerRecord")
        self.assertEqual(kwargs["synapse_factory"]().record,
                         {"hotkey": HOTKEY, "record": record(pro_account_size=PRO_SIZE).to_dict()})
        # An opening has its own broadcast
        self.mothership.open_standard_account(OTHER, record())
        self.mothership._broadcast_to_validators.assert_called_once()

    def test_a_peer_takes_the_record_as_the_mothership_wrote_it(self):
        self.peer.set_standard_account(HOTKEY, record())
        granted = record(pro_account_size=PRO_SIZE)
        self.assertTrue(self._receive({"hotkey": HOTKEY, "record": granted.to_dict()}))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), granted)
        closed = record(pro_account_size=PRO_SIZE, closed_at_ms=NOW_MS)
        self.assertTrue(self._receive({"hotkey": HOTKEY, "record": closed.to_dict()}))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), closed)
        # Even one at an earlier stage: the mothership's write is the latest word
        self.assertTrue(self._receive({"hotkey": HOTKEY, "record": record().to_dict()}))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), record())
        # An account the peer did not have yet gets the opening tier and bucket, and trades the opening size
        live = record(opening_leverage_tier=2, pro_account_size=PRO_SIZE, trading_account_size=PRO_SIZE)
        self.assertTrue(self._receive({"hotkey": OTHER, "record": live.to_dict()}))
        other = self.peer.get_account(OTHER)
        self.assertEqual((other.leverage_tier, other.miner_bucket, other.standard_account),
                         (2, MinerBucket.SUBACCOUNT_CHALLENGE, record(opening_leverage_tier=2, pro_account_size=PRO_SIZE)))

    def test_a_peer_keeps_the_size_it_trades_until_its_own_move(self):
        # The mothership leaves the transition first and broadcasts the pro size as traded; the peer, still in the
        # transition, keeps trading the opening size or its drawdown would read the resize as a loss
        self.peer.set_standard_account(HOTKEY, record(pro_account_size=PRO_SIZE))
        live = record(pro_account_size=PRO_SIZE, trading_account_size=PRO_SIZE)
        self.assertTrue(self._receive({"hotkey": HOTKEY, "record": live.to_dict()}))
        self.assertEqual(self.peer.get_standard_account(HOTKEY), record(pro_account_size=PRO_SIZE))
        self.assertEqual(self.peer.get_miner_account_size(HOTKEY), SIZE)
        # Its own move sets the size, and a later record from the mothership leaves it
        self.peer.set_standard_account(HOTKEY, live)
        closed = record(pro_account_size=PRO_SIZE, closed_at_ms=NOW_MS)
        self.assertTrue(self._receive({"hotkey": HOTKEY, "record": closed.to_dict()}))
        self.assertEqual(self.peer.get_standard_account(HOTKEY),
                         record(pro_account_size=PRO_SIZE, trading_account_size=PRO_SIZE, closed_at_ms=NOW_MS))

    def test_what_a_peer_refuses(self):
        for data, sender in (
            ({"hotkey": HOTKEY, "record": record().to_dict()}, PEER),
            ({"hotkey": HOTKEY, "record": record().to_dict()}, None),
            ({"record": record().to_dict()}, MOTHERSHIP),
            ({"hotkey": HOTKEY, "record": "not a record"}, MOTHERSHIP),
            ({"hotkey": HOTKEY, "record": {"opened_at_ms": NOW_MS}}, MOTHERSHIP),
            ({"hotkey": SUBACCOUNT, "record": record().to_dict()}, MOTHERSHIP),
        ):
            with self.subTest(data=data, sender=sender):
                self.assertFalse(self._receive(data, sender))
        self.assertIsNone(self.peer.get_standard_account(HOTKEY))
        self.assertIsNone(self.peer.get_account(SUBACCOUNT))
        self.assertFalse(self.mothership.receive_standard_miner_record_update(
            {"hotkey": HOTKEY, "record": record().to_dict()}, sender_hotkey=MOTHERSHIP))


class TestPenalties(TestBase):
    """A standard miner is spared the minimum collateral penalty and, like a subaccount, the per-checkpoint product."""

    CP_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    def _checkpoints(self, hotkey, standard_account, bucket):
        manager = object.__new__(PenaltyLedgerManager)
        manager.running_unit_tests = True
        manager.penalty_ledgers = {}
        start_ms = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 14 * ValiConfig.DAILY_MS
        cps = [PerfCheckpoint(last_update_ms=start_ms + (i + 1) * self.CP_MS, prev_portfolio_ret=1.0,
                              accum_ms=self.CP_MS, gain=0.001, loss=0.0, mdd=1.0) for i in range(4)]
        for cp in cps:
            cp.prev_portfolio_realized_pnl = 1_000.0
        ledger = PerfLedger(initialization_time_ms=start_ms, cps=cps)
        manager._perf_ledger_client = SimpleNamespace(get_perf_ledgers=lambda: {hotkey: ledger})
        manager._position_client = SimpleNamespace(get_positions_for_all_miners=lambda: {hotkey: []})
        manager._miner_account_client = SimpleNamespace(get_miner_account_size=lambda _hk: SIZE,
                                                        get_standard_account=lambda _hk: standard_account)
        manager._challengeperiod_client = SimpleNamespace(to_checkpoint_dict=lambda: {hotkey: {
            "entries": [{"bucket": bucket.value, "start_time_ms": start_ms}],
            "pro_stats": {"max_drawdown": 0.99, "soft_breach_days": []},
        }})
        manager.save_to_disk = lambda *_a, **_k: None
        manager.build_penalty_ledgers(delta_update=False)
        return manager.penalty_ledgers[hotkey].checkpoints

    def test_a_standard_miner_is_spared_the_minimum_collateral_and_the_checkpoint_product(self):
        spared = self._checkpoints(HOTKEY, record(), MinerBucket.SUBACCOUNT_FUNDED)
        self.assertTrue(spared)
        for cp in spared:
            self.assertEqual((cp.min_collateral_penalty, cp.total_penalty), (1.0, 1))
        expected = ValidatorContractManager.min_collateral_penalty(SIZE)
        self.assertLess(expected, 1.0)
        for cp in self._checkpoints(LEGACY, None, MinerBucket.MAINCOMP):
            self.assertEqual(cp.min_collateral_penalty, expected)
            self.assertLess(cp.total_penalty, 1.0)
        # A subaccount keeps its waiver: the product is 1 whatever the minimum collateral penalty computes
        for cp in self._checkpoints(SUBACCOUNT, None, MinerBucket.SUBACCOUNT_FUNDED):
            self.assertEqual((cp.min_collateral_penalty, cp.total_penalty), (expected, 1))


class TestDebtLedgerDeletion(TestBase):
    """An account switch deletes the debt ledger of a subaccount or a standard miner, never a regular miner's."""

    def test_only_an_account_on_the_pipeline_has_its_ledger_deleted(self):
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_manager import DebtLedgerManager
        manager = object.__new__(DebtLedgerManager)
        manager.debt_ledgers = {HOTKEY: "ledger", LEGACY: "ledger", SUBACCOUNT: "ledger"}
        manager.penalty_ledger_manager = Mock()
        manager._miner_account_client = Mock()
        manager._miner_account_client.get_standard_account.side_effect = lambda hk: record() if hk == HOTKEY else None
        manager.save_to_disk = Mock()
        self.assertTrue(manager.delete_ledger(HOTKEY))
        self.assertFalse(manager.delete_ledger(LEGACY))
        self.assertTrue(manager.delete_ledger(SUBACCOUNT))
        self.assertEqual(set(manager.debt_ledgers), {LEGACY})
        looked_up = [call.args[0] for call in manager._miner_account_client.get_standard_account.call_args_list]
        self.assertEqual(looked_up, [HOTKEY, LEGACY])  # a subaccount is known without asking


class TestPipelineThroughTheServers(TestBase):
    """The hops of an opened account through the test servers: the funded tier, the pro size on the record, the
    size traded and the asset class once the pro account is live."""

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

    def tearDown(self):
        self.contract_client.clear_test_collateral_balances()
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def test_the_hops_of_an_opened_account(self):
        now_ms = TimeUtil.now_in_millis()
        moved, message = self.challenge_period_client.admin_set_bucket(NEW, MinerBucket.SUBACCOUNT_FUNDED, now_ms)
        self.assertTrue(moved, message)
        funded = self.account_client.get_account(NEW)
        self.assertEqual((funded.leverage_tier, funded.miner_bucket), (FUNDED_TIER, MinerBucket.SUBACCOUNT_FUNDED))
        self.assertAlmostEqual(self.account_client.get_miner_account_size(NEW), 50_000.0)

        moved, message = self.challenge_period_client.promote_subaccount(NEW, now_ms, pro_account_size=250_000.0)
        self.assertTrue(moved, message)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(NEW), MinerBucket.PRO_CHALLENGE_TRANSITION)
        self.assertEqual(self.account_client.get_standard_account(NEW).pro_account_size, 250_000.0)
        self.assertAlmostEqual(self.account_client.get_miner_account_size(NEW), 50_000.0)
        self.assertEqual(self.asset_selection_client.get_asset_selection(NEW), MinerAssetClass.CRYPTO)

        moved, message = self.challenge_period_client.admin_set_bucket(NEW, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, now_ms)
        self.assertTrue(moved, message)
        live = self.account_client.get_standard_account(NEW)
        self.assertEqual((live.pro_account_size, live.trading_account_size), (250_000.0, 250_000.0))
        self.assertAlmostEqual(self.account_client.get_miner_account_size(NEW), 250_000.0)
        self.assertEqual(self.asset_selection_client.get_asset_selection(NEW), MinerAssetClass.ALL_MARKETS)
        self.assertEqual(self.account_client.get_account(NEW).asset_class, MinerAssetClass.ALL_MARKETS)
