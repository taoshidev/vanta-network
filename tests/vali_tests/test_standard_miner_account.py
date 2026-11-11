"""
The record of the account a standard miner opens: its persistence through the accounts file, the
checkpoint and auto sync, and the pipeline account view shared with entity subaccounts.
"""
import dataclasses
import itertools

from entity_management.entity_manager import SubaccountInfo
from entity_management.entity_utils import is_created_through_challenge, is_instant_funded, pro_payout_scale
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import MinerAccountManager
from vali_objects.miner_account.standard_miner import StandardMinerAccount, PipelineAccountInfo
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode, ValiConfig

HOTKEY = "standard_miner_hotkey"
LEGACY = "legacy_miner_hotkey"
OPENING = dict(opened_at_ms=1_700_000_000_000, opening_leverage_tier=2, asset_class="crypto", account_size=50_000.0,
               locked_theta=70.0, opening_price_usd=7.857)


def record(**overrides) -> StandardMinerAccount:
    return StandardMinerAccount(**{**OPENING, **overrides})


def subaccount(**overrides) -> dict:
    values = dict(subaccount_id=1, subaccount_uuid="uuid-1", synthetic_hotkey="entity_1", created_at_ms=1,
                  account_size=100_000.0, asset_class="crypto")
    values.update(overrides)
    return SubaccountInfo(**values).model_dump()


def manager() -> MinerAccountManager:
    return MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL)


class TestStandardMinerAccount(TestBase):

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

    def test_record_round_trips_and_ignores_unknown_keys(self):
        full = record(pro_account_size=250_000.0, payout_scale=1.2, closed_at_ms=5)
        self.assertEqual(StandardMinerAccount.from_dict(full.to_dict()), full)
        self.assertTrue(full.is_closed)
        opened = StandardMinerAccount.from_dict({**OPENING, "unknown_field": 1})
        self.assertEqual(opened, record())
        self.assertFalse(opened.is_closed)
        self.assertIsNone(opened.pro_account_size)

    def test_record_is_frozen_and_grows_by_replacement(self):
        opened = record()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            opened.pro_account_size = 250_000.0
        promoted = dataclasses.replace(opened, pro_account_size=250_000.0)
        self.assertEqual(promoted.pro_account_size, 250_000.0)
        self.assertEqual(promoted.account_size, opened.account_size)

    def test_record_converts_numbers_and_rejects_missing_ones(self):
        loaded = StandardMinerAccount.from_dict({**OPENING, "account_size": "50000", "opened_at_ms": "1700000000000"})
        self.assertEqual(loaded, record())
        self.assertIsInstance(loaded.account_size, float)
        self.assertIsInstance(loaded.opened_at_ms, int)
        self.assertEqual(record(opening_leverage_tier=2.0).opening_leverage_tier, 2)
        for bad in (dict(locked_theta=None), dict(account_size="not a number"), dict(opening_leverage_tier=True),
                    dict(opening_leverage_tier=2.5), dict(account_size=float("nan")), dict(locked_theta=-1.0),
                    dict(opening_price_usd=0.0), dict(pro_account_size=float("inf"))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                record(**bad)
        with self.assertRaises(TypeError):
            StandardMinerAccount.from_dict({"opened_at_ms": 1})

    def test_record_lives_on_the_account_and_survives_a_reload_and_an_account_reset(self):
        m = manager()
        m.set_miner_account_size(LEGACY, collateral_balance_theta=300.0, timestamp_ms=TimeUtil.now_in_millis())
        self.assertIsNone(m.get_standard_account(LEGACY))
        m.set_standard_account(HOTKEY, record())
        self.assertEqual(m.get_standard_account(HOTKEY), record())
        self.assertEqual(m.accounts_dict()[HOTKEY][-1]["standard_account"], record().to_dict())
        self.assertIsNone(m.accounts_dict()[LEGACY][-1]["standard_account"])
        reloaded = manager()
        self.assertEqual(reloaded.get_standard_account(HOTKEY), record())
        self.assertIsNone(reloaded.get_standard_account(LEGACY))
        # The record outlives the account reset that a bucket change runs
        m.reset_account(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED)
        self.assertEqual(m.get_standard_account(HOTKEY), record())
        # An unreadable record drops only itself from what is loaded, and a sync then keeps the record held
        broken = m.accounts_dict()
        broken[HOTKEY][-1]["standard_account"] = {"opened_at_ms": "not a record"}
        parsed = MinerAccountManager._parse_accounts_dict(broken)
        self.assertIsNone(parsed[HOTKEY].standard_account)
        m.sync_miner_account_sizes_data(broken)
        self.assertEqual(m.get_standard_account(HOTKEY), record())

    def test_a_synced_snapshot_only_moves_a_record_forward(self):
        m = manager()
        now_ms = TimeUtil.now_in_millis()
        m.set_miner_account_size(LEGACY, collateral_balance_theta=300.0, timestamp_ms=now_ms)
        m.set_miner_account_size("other_legacy_hotkey", collateral_balance_theta=100.0, timestamp_ms=now_ms)
        m.open_standard_account(HOTKEY, record())  # the record, the opening tier and the bucket
        opened = (record(), 2, MinerBucket.SUBACCOUNT_CHALLENGE)

        def account_state(hotkey=HOTKEY):
            account = m.get_account(hotkey)
            return account.standard_account, account.leverage_tier, account.miner_bucket

        # A snapshot cut before the opening has no account for the hotkey, or one without the record, tier and bucket;
        # accounts without a record are taken from the snapshot as they are
        snapshot = {LEGACY: m.accounts_dict()[LEGACY]}
        snapshot[LEGACY][-1]["total_realized_pnl"] = 123.0
        m.sync_miner_account_sizes_data(snapshot)
        self.assertEqual(account_state(), opened)
        self.assertEqual(m.get_account(LEGACY).total_realized_pnl, 123.0)
        self.assertIsNone(m.get_account("other_legacy_hotkey"))
        without_record = m.accounts_dict()
        for field in ("standard_account", "leverage_tier", "miner_bucket"):
            without_record[HOTKEY][-1][field] = None
        m.sync_miner_account_sizes_data(without_record)
        self.assertEqual(account_state(), opened)
        # A snapshot behind the record held leaves it; one ahead of it or at the same stage replaces it. The size
        # traded is not a stage: each validator sets it at its own bucket move, so the snapshot's value stands
        pro = record(pro_account_size=250_000.0)
        trading_pro = record(pro_account_size=250_000.0, trading_account_size=250_000.0)
        for held, synced, kept in (
            (pro, record(), pro),
            (record(closed_at_ms=5), trading_pro, record(closed_at_ms=5)),
            (record(), pro, pro),
            (trading_pro, pro, pro),
            (pro, trading_pro, trading_pro),
            (trading_pro, record(pro_account_size=250_000.0, closed_at_ms=5), record(pro_account_size=250_000.0, closed_at_ms=5)),
            (record(), record(payout_scale=1.2), record(payout_scale=1.2)),
        ):
            with self.subTest(held=held, synced=synced):
                m.get_account(HOTKEY).standard_account = held  # whatever this validator holds, a close included
                snapshot = m.accounts_dict()
                snapshot[HOTKEY][-1]["standard_account"] = synced.to_dict()
                m.sync_miner_account_sizes_data(snapshot)
                self.assertEqual(m.get_standard_account(HOTKEY), kept)

    def test_only_a_record_can_be_stored(self):
        m = manager()
        with self.assertRaises(TypeError):
            m.set_standard_account(HOTKEY, record().to_dict())
        self.assertIsNone(m.get_account(HOTKEY))

    def test_record_rides_the_checkpoint_and_auto_sync(self):
        m = manager()
        m.set_standard_account(HOTKEY, record(pro_account_size=250_000.0))
        checkpoint = m.accounts_dict()  # what the validator checkpoint carries
        self.account_client.sync_miner_account_sizes_data(checkpoint)  # what auto sync applies
        self.assertEqual(self.account_client.get_standard_account(HOTKEY), record(pro_account_size=250_000.0))
        self.assertEqual(self.account_client.get_account(HOTKEY).standard_account,
                         record(pro_account_size=250_000.0))
        # A checkpoint cut before the pro sizing does not roll it back
        checkpoint[HOTKEY][-1]["standard_account"] = record().to_dict()
        self.account_client.sync_miner_account_sizes_data(checkpoint)
        self.assertEqual(self.account_client.get_standard_account(HOTKEY), record(pro_account_size=250_000.0))

    def test_record_is_set_and_read_over_rpc(self):
        self.account_client.set_standard_account(HOTKEY, record())
        self.assertEqual(self.account_client.get_standard_account(HOTKEY), record())
        self.assertIsNone(self.account_client.get_standard_account(LEGACY))

    def test_pipeline_view_of_a_subaccount_matches_the_reads_it_replaces(self):
        buckets = (None, "SUBACCOUNT_CHALLENGE", "SUBACCOUNT_FUNDED", "PRO_CHALLENGE_FROM_STANDARD")
        for initial_bucket, account_type, payout_scale, standard, pro in itertools.product(
                buckets, ("standard", "pro"), (None, 0.0, 1.0), (None, 0.0, 100_000.0), (None, 0.0, 500_000.0)):
            info = subaccount(initial_bucket=initial_bucket, account_type=account_type, payout_scale=payout_scale,
                              standard_account_size=standard, pro_account_size=pro)
            view = PipelineAccountInfo.from_subaccount(info)
            with self.subTest(info=info):
                self.assertEqual(view.created_through_challenge, is_created_through_challenge(info.get("initial_bucket")))
                self.assertEqual(view.instant_funded, is_instant_funded(info.get("initial_bucket")))
                # The peer step of auto-promotion: a pro size granted by the mothership
                self.assertEqual(view.account_type == AccountType.PRO.value and bool(view.pro_account_size),
                                 bool(info.get("account_type") == AccountType.PRO.value and info.get("pro_account_size")))
                self.assertEqual(pro_payout_scale(view.standard_account_size, view.pro_account_size, view.payout_scale),
                                 pro_payout_scale(info.get("standard_account_size"), info.get("pro_account_size"),
                                                  info.get("payout_scale")))

    def test_pipeline_view_of_a_standard_miner(self):
        opened = PipelineAccountInfo.from_standard_miner(record())
        self.assertTrue(opened.created_through_challenge)
        self.assertFalse(opened.instant_funded)
        self.assertEqual(opened.account_type, AccountType.STANDARD.value)
        self.assertIsNone(opened.standard_account_size)
        self.assertIsNone(opened.pro_account_size)
        self.assertEqual(opened.payout_scale, ValiConfig.grow_payout_multiplier())
        self.assertEqual(pro_payout_scale(opened.standard_account_size, opened.pro_account_size, opened.payout_scale), 1.0)

        # Once auto-promotion grants a pro size the view reads as a subaccount on the pro track does
        grown = PipelineAccountInfo.from_standard_miner(record(pro_account_size=250_000.0))
        self.assertEqual(grown.account_type, AccountType.PRO.value)
        self.assertEqual(grown.standard_account_size, 50_000.0)
        self.assertTrue(grown.account_type == AccountType.PRO.value and bool(grown.pro_account_size))
        self.assertEqual(pro_payout_scale(grown.standard_account_size, grown.pro_account_size, grown.payout_scale),
                         ValiConfig.grow_payout_multiplier() * 50_000.0 / 250_000.0)
        pinned = PipelineAccountInfo.from_standard_miner(record(payout_scale=1.0))
        self.assertEqual(pinned.payout_scale, 1.0)
