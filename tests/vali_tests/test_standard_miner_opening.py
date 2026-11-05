"""
Opening a standard miner's account: the size and locked theta from the collateral at today's price, the record,
the opening tier and SUBACCOUNT_CHALLENGE with the standard miner limits; orders rejected before the opening;
a size that collateral never changes; the challenge period's first sight of a real hotkey; and the transport of
an opening from the mothership to its peers.
"""
import contextlib
import dataclasses
import pickle
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock, patch

import template.protocol
from shared_objects.rpc.rpc_client_base import RPCClientBase
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from shared_objects.rpc.server_registry import ServerRegistry
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.contract.contract_client import ContractClient
from vali_objects.contract.validator_contract_manager import ValidatorContractManager
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.miner_account.miner_account_manager import MinerAccount
from vali_objects.miner_account.miner_account_server import MinerAccountServer
from vali_objects.miner_account.standard_miner import StandardMinerAccount, OpeningRejected, open_account
from vali_objects.utils.order_processor import OrderProcessor
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode, TradePair, ValiConfig

CUTOFF = 5_000
NEW = "miner_registered_after_cutoff"
LEGACY = "miner_registered_at_cutoff"
VALIDATOR = "validator_registered_after_cutoff"
ENTITY = "entity_registered_after_cutoff"
HOTKEYS = [NEW, LEGACY, VALIDATOR, ENTITY]
BLOCKS = [CUTOFF + 1, CUTOFF, CUTOFF + 1, CUTOFF + 1]
STAKE_WEIGHTS = [0.0, 0.0, ValiConfig.STAKE_MIN, 0.0]
PRICE = 100.0  # set_hotkeys publishes equal reserves and a $100 TAO price, so one theta is worth $100
RAO = 10 ** 9
TIER = 2
PCT = ValiConfig.STANDARD_MINER_COLLATERAL_PCT_BY_TIER[TIER]
MIN_SIZE = ValiConfig.STANDARD_MINER_MIN_ACCOUNT_SIZE
MAX_SIZE = ValiConfig.STANDARD_MINER_MAX_ACCOUNT_SIZE
NOW_MS = 1_700_000_000_000
MOTHERSHIP = "mothership_hotkey"
PEER = "peer_validator_hotkey"
REAL_TESTNET_MOTHERSHIP = ValiConfig.MOTHERSHIP_HOTKEY_TESTNET  # what the test servers, in their own processes, verify against
CLIENT_PATHS = [
    "vali_objects.challenge_period.challengeperiod_manager." + name for name in (
        "PerfLedgerClient", "PositionManagerClient", "EliminationClient", "PlagiarismClient", "MinerAccountClient",
        "CommonDataClient", "AssetSelectionClient", "DebtLedgerClient", "LimitOrderClient", "EntityClient")
]


def record(**overrides) -> StandardMinerAccount:
    values = dict(opened_at_ms=NOW_MS, opening_leverage_tier=TIER, asset_class="crypto", account_size=50_000.0,
                  locked_theta=5.5, opening_price_usd=PRICE)
    values.update(overrides)
    return StandardMinerAccount(**values)


class TestOpenAccount(TestBase):
    """The pure sizing: collateral USD value over the tier's collateral percentage, within the size bounds."""

    def test_size_follows_the_collateral_at_the_tier_percentage(self):
        opened = open_account(5.5, PRICE, TIER, "crypto", NOW_MS)
        self.assertAlmostEqual(opened.account_size, 50_000.0)
        self.assertAlmostEqual(opened.locked_theta, 5.5)
        self.assertEqual((opened.opening_leverage_tier, opened.asset_class, opened.opening_price_usd, opened.opened_at_ms),
                         (TIER, "crypto", PRICE, NOW_MS))
        for tier, pct in ValiConfig.STANDARD_MINER_COLLATERAL_PCT_BY_TIER.items():
            with self.subTest(tier=tier):
                opened = open_account(5.0, PRICE, tier, "forex", NOW_MS)
                self.assertAlmostEqual(opened.account_size, 5.0 * PRICE / pct)
                self.assertAlmostEqual(opened.locked_theta * PRICE, opened.account_size * pct)
                self.assertLessEqual(opened.locked_theta, 5.0)

    def test_size_is_capped_and_the_collateral_beyond_the_locked_theta_stays_free(self):
        opened = open_account(50.0, PRICE, TIER, "crypto", NOW_MS)
        self.assertEqual(opened.account_size, MAX_SIZE)
        self.assertAlmostEqual(opened.locked_theta, MAX_SIZE * PCT / PRICE)
        self.assertLess(opened.locked_theta, 50.0)
        at_minimum = open_account(MIN_SIZE * PCT / PRICE, PRICE, TIER, "crypto", NOW_MS)
        self.assertAlmostEqual(at_minimum.account_size, MIN_SIZE)

    def test_rejections(self):
        for collateral, price, tier in ((0.5, PRICE, TIER), (5.5, 0.0, TIER), (5.5, None, TIER), (5.5, PRICE, 4),
                                        (5.5, PRICE, True), (5.5, PRICE, 2.0), (5.5, PRICE, [2]), (0.0, PRICE, TIER),
                                        (None, PRICE, TIER)):
            with self.subTest(collateral=collateral, price=price, tier=tier), self.assertRaises(OpeningRejected):
                open_account(collateral, price, tier, "crypto", NOW_MS)

    def test_the_account_size_comes_from_the_record_whatever_the_bucket(self):
        for bucket in (None, MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.PRO_FUNDED, MinerBucket.ELIMINATED):
            with self.subTest(bucket=bucket):
                opened = MinerAccount(miner_hotkey=NEW, miner_bucket=bucket, standard_account=record())
                self.assertEqual(opened.get_account_size(), 50_000.0)
                self.assertEqual(opened.get_account_size(timestamp_ms=NOW_MS), 50_000.0)
                grown = MinerAccount(miner_hotkey=NEW, miner_bucket=bucket,
                                     standard_account=record(trading_account_size=250_000.0, pro_account_size=250_000.0))
                self.assertEqual(grown.get_account_size(), 250_000.0)
                self.assertEqual(grown.balance, 250_000.0)


class TestOpening(TestBase):
    """The opening through the contract manager, with the chain and the price mocked by the test servers."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.contract_client = cls.orchestrator.get_client('contract')
        cls.account_client = cls.orchestrator.get_client('miner_account')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.elimination_client = cls.orchestrator.get_client('elimination')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()
        self.contract_client.clear_test_collateral_balances()
        self.metagraph_client.set_hotkeys(HOTKEYS, block_at_registration=BLOCKS, stake_weight=STAKE_WEIGHTS,
                                          standard_miner_cutoff_block=CUTOFF)
        self.account_client.update_asset_selection(NEW, MinerAssetClass.CRYPTO)
        self.contract_client.set_test_collateral_balance(NEW, int(5.5 * RAO))

    def tearDown(self):
        self.contract_client.clear_test_collateral_balances()
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def _open(self, hotkey=NEW, tier=TIER):
        return self.contract_client.open_standard_miner_account(hotkey, tier)

    def _rejection(self, hotkey=NEW, tier=TIER):
        opened, message, record_dict = self._open(hotkey, tier)
        self.assertFalse(opened, message)
        self.assertIsNone(record_dict)
        return message

    def test_opening_fixes_the_size_and_starts_the_challenge(self):
        opened, message, record_dict = self._open()
        self.assertTrue(opened, message)
        opened_record = StandardMinerAccount.from_dict(record_dict)
        self.assertAlmostEqual(opened_record.account_size, 50_000.0)
        self.assertAlmostEqual(opened_record.locked_theta, 5.5)
        self.assertEqual(opened_record.opening_price_usd, PRICE)
        self.assertEqual(self.account_client.get_standard_account(NEW), opened_record)
        account = self.account_client.get_account(NEW)
        self.assertEqual((account.leverage_tier, account.miner_bucket), (TIER, MinerBucket.SUBACCOUNT_CHALLENGE))
        self.assertAlmostEqual(self.account_client.get_miner_account_size(NEW), 50_000.0)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(NEW), MinerBucket.SUBACCOUNT_CHALLENGE)
        self.assertEqual(self.challenge_period_client.get_miner_start_time(NEW), opened_record.opened_at_ms)
        state = self.challenge_period_client.to_checkpoint_dict()[NEW]
        self.assertEqual(state["intraday_drawdown_threshold_override"], ValiConfig.STANDARD_MINER_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(state["eod_hwm_threshold_override"], ValiConfig.STANDARD_MINER_EOD_HWM_THRESHOLD)

        # Deposits afterwards are recorded at the fixed size, once per balance
        self.contract_client.set_test_collateral_balance(NEW, int(40 * RAO))
        self.assertTrue(self.contract_client.refresh_miner_account_size(NEW))
        self.assertTrue(self.contract_client.refresh_miner_account_size(NEW))
        self.assertAlmostEqual(self.account_client.get_miner_account_size(NEW), 50_000.0)
        records = self.account_client.get_account(NEW).collateral_records
        self.assertEqual((records[-1].account_size, records[-1].account_size_theta), (50_000.0, 40.0))
        self.assertEqual(len(records), 1)

        # A second request changes nothing and reports the account already opened
        opened, message, record_dict = self._open(tier=3)
        self.assertTrue(opened)
        self.assertIn("already opened", message)
        self.assertEqual(StandardMinerAccount.from_dict(record_dict), opened_record)
        self.assertEqual(self.account_client.get_account(NEW).leverage_tier, TIER)

    def test_a_request_for_an_opened_account_puts_a_missing_state_back(self):
        # As an opening interrupted before its challenge period state leaves the account
        self.account_client.set_standard_account(NEW, record())
        self.assertIsNone(self.challenge_period_client.get_miner_bucket(NEW))
        opened, message, _ = self._open()
        self.assertTrue(opened, message)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(NEW), MinerBucket.SUBACCOUNT_CHALLENGE)
        self.assertEqual(self.challenge_period_client.get_miner_start_time(NEW), NOW_MS)
        self.assertEqual(self.account_client.get_account(NEW).leverage_tier, TIER)

    def test_who_cannot_open(self):
        self.assertIn("existing rules", self._rejection(LEGACY))
        self.assertIn("not in the last metagraph sync", self._rejection("unregistered_hotkey"))
        self.assertIn("validator", self._rejection(VALIDATOR))
        registered, message = self.entity_client.register_entity(entity_hotkey=ENTITY)
        self.assertTrue(registered, message)
        self.assertIn("entity", self._rejection(ENTITY))
        self.assertIn("leverage tier", self._rejection(tier=7))
        self.contract_client.set_test_collateral_balance(NEW, int(0.5 * RAO))
        self.assertIn("needs", self._rejection())
        self.contract_client.set_test_collateral_balance(NEW, int(5.5 * RAO))
        self.metagraph_client.update_metagraph(tao_to_usd_rate=0.0)
        self.assertIn("price", self._rejection())
        self.assertIsNone(self.account_client.get_standard_account(NEW))
        self.assertIsNone(self.challenge_period_client.get_miner_bucket(NEW))

    def test_an_asset_class_the_pipeline_supports_comes_first(self):
        self.account_client.sync_miner_account_sizes_data({})
        self.contract_client.set_test_collateral_balance(NEW, int(5.5 * RAO))
        self.assertIn("asset class", self._rejection())
        self.account_client.update_asset_selection(NEW, MinerAssetClass.HL_ALL)
        self.assertIn("hl_all", self._rejection())

    def test_a_closed_eliminated_or_already_trading_hotkey_cannot_open(self):
        self.account_client.set_standard_account(NEW, record(closed_at_ms=NOW_MS))
        self.assertIn("closed", self._rejection())
        self.account_client.sync_miner_account_sizes_data({})
        self.account_client.update_asset_selection(NEW, MinerAssetClass.CRYPTO)
        self.challenge_period_client.set_miner_bucket(NEW, MinerBucket.CHALLENGE, NOW_MS)
        self.assertIn("existing rules", self._rejection())
        self.orchestrator.clear_all_test_data()
        self.metagraph_client.set_hotkeys(HOTKEYS, block_at_registration=BLOCKS, stake_weight=STAKE_WEIGHTS,
                                          standard_miner_cutoff_block=CUTOFF)
        self.account_client.update_asset_selection(NEW, MinerAssetClass.CRYPTO)
        self.elimination_client.append_elimination_row(NEW, EliminationReason.INACTIVE)
        self.assertIn("eliminated", self._rejection())

    def test_an_opened_miner_cannot_register_as_an_entity(self):
        opened, message, _ = self._open()
        self.assertTrue(opened, message)
        registered, message = self.entity_client.register_entity(entity_hotkey=NEW)
        self.assertFalse(registered)
        self.assertIn("already used as a miner", message)

    def test_an_eliminated_hotkey_cannot_reopen_or_complete_its_account(self):
        opened, message, _ = self._open()
        self.assertTrue(opened, message)
        self.elimination_client.append_elimination_row(NEW, EliminationReason.INACTIVE)
        self.assertIn("eliminated", self._rejection())

    def test_an_opened_account_is_kept_when_the_hotkey_later_counts_as_a_validator(self):
        opened, message, record_dict = self._open()
        self.assertTrue(opened, message)
        self.metagraph_client.set_hotkeys(HOTKEYS, block_at_registration=BLOCKS, standard_miner_cutoff_block=CUTOFF,
                                          stake_weight=[ValiConfig.STAKE_MIN, 0.0, ValiConfig.STAKE_MIN, 0.0])
        opened, message, again = self._open(tier=3)
        self.assertTrue(opened, message)
        self.assertIn("already opened", message)
        self.assertEqual(again, record_dict)
        self.assertAlmostEqual(self.account_client.get_miner_account_size(NEW), 50_000.0)


class TestOrdersBeforeAndAfterTheOpening(TestBase):
    """The order gate and the size floor, on an in-process account server whose cutoff can be set."""

    def setUp(self):
        super().setUp()
        with mock.patch.object(ServerRegistry, "register"):
            self.server = MinerAccountServer(running_unit_tests=True, start_server=False, connection_mode=RPCConnectionMode.LOCAL)
        self.client = MinerAccountClient(connection_mode=RPCConnectionMode.LOCAL, running_unit_tests=True)
        self.client.set_direct_server(self.server)
        self.manager = self.server._manager
        self.manager.sync_miner_account_sizes_data({})
        self.addCleanup(self.manager.sync_miner_account_sizes_data, {})
        self.addCleanup(self.manager.set_registered_after_cutoff, None, {})
        self.processor = OrderProcessor(limit_order_client=Mock(), market_order_client=Mock(), miner_account_client=self.client)
        cutoff = mock.patch.object(ValiConfig, "STANDARD_MINER_CUTOFF_BLOCK_TESTNET", CUTOFF)
        cutoff.start()
        self.addCleanup(cutoff.stop)
        self.manager.set_registered_after_cutoff(CUTOFF, {NEW: True, LEGACY: False})

    def _validate(self, hotkey, execution_type=ExecutionType.MARKET, order_type=OrderType.LONG):
        trade_pair = None if execution_type in (ExecutionType.FLAT_ALL, ExecutionType.LIMIT_CANCEL) else TradePair.BTCUSDC
        return self.processor.validate(hotkey, execution_type, trade_pair, order_type)

    def test_orders_wait_for_the_opening(self):
        ok, message, _ = self._validate(NEW)
        self.assertFalse(ok)
        self.assertIn("open an account", message)
        ok, message, _ = self._validate("unregistered_hotkey")
        self.assertFalse(ok)
        self.assertIn("not known yet", message)
        for hotkey in (LEGACY, ValiConfig.DEVELOPMENT_HOTKEY, f"{ENTITY}_1"):
            ok, message, _ = self._validate(hotkey)
            self.assertFalse(ok)
            self.assertIn("not yet initialized", message)  # the existing path: no account yet, not the gate

        self.manager.update_asset_selection(NEW, MinerAssetClass.CRYPTO)
        self.manager.set_standard_account(NEW, record())
        ok, message, _ = self._validate(NEW)
        self.assertTrue(ok, message)

        self.manager.set_standard_account(NEW, record(closed_at_ms=NOW_MS))
        ok, message, _ = self._validate(NEW)
        self.assertFalse(ok)
        self.assertIn("closed", message)
        # A closed account can still close what it holds
        for execution_type, order_type in ((ExecutionType.FLAT_ALL, None), (ExecutionType.MARKET, OrderType.FLAT),
                                           (ExecutionType.LIMIT_CANCEL, None)):
            ok, message, _ = self._validate(NEW, execution_type, order_type)
            self.assertNotIn("closed", message)

    def test_closing_orders_pass_while_the_answer_is_pending_or_the_account_unopened(self):
        self.manager.update_asset_selection(LEGACY, MinerAssetClass.CRYPTO)
        for hotkey in (NEW, "unregistered_hotkey", LEGACY):
            with self.subTest(hotkey=hotkey):
                ok, message, _ = self._validate(hotkey, ExecutionType.FLAT_ALL, None)
                self.assertNotIn("open an account", message)
                self.assertNotIn("not known yet", message)
                ok, message, _ = self._validate(hotkey, ExecutionType.MARKET, OrderType.FLAT)
                self.assertNotIn("open an account", message)
                self.assertNotIn("not known yet", message)
        ok, message, _ = self._validate(LEGACY)
        self.assertTrue(ok, message)

    def test_no_account_and_no_floor_before_the_opening(self):
        self.manager.get_or_create(NEW)  # as a collateral deposit does
        self.assertIsNone(self.manager.get_miner_account_size(NEW, use_account_floor=True))
        self.assertNotIn(NEW, self.manager.get_all_miner_account_sizes())
        self.assertEqual(self.manager.get_miner_account_size(LEGACY, use_account_floor=True), ValiConfig.MIN_CAPITAL)
        self.manager.set_standard_account(NEW, record())
        self.assertEqual(self.manager.get_miner_account_size(NEW), 50_000.0)
        self.assertEqual(self.manager.get_all_miner_account_sizes()[NEW], 50_000.0)

    def test_opening_on_the_account_is_one_step_and_idempotent(self):
        stored = self.manager.open_standard_account(NEW, record())
        account = self.manager.get_account(NEW)
        self.assertEqual((stored, account.standard_account, account.leverage_tier, account.miner_bucket),
                         (record(), record(), TIER, MinerBucket.SUBACCOUNT_CHALLENGE))
        again = self.manager.open_standard_account(NEW, record(opening_leverage_tier=3, account_size=70_000.0))
        self.assertEqual(again, record())
        self.assertEqual(self.manager.get_account(NEW).leverage_tier, TIER)
        self.manager.set_leverage_tier(NEW, None)
        self.manager.open_standard_account(NEW, record())
        self.assertEqual(self.manager.get_account(NEW).leverage_tier, TIER)
        with self.assertRaises(ValueError):
            self.manager.open_standard_account(f"{ENTITY}_1", record())
        self.assertIsNone(self.manager.get_account(f"{ENTITY}_1"))


class TestFirstSight(TestBase):
    """A real hotkey with positions but no challenge period state."""

    def setUp(self):
        super().setUp()
        stack = contextlib.ExitStack()
        for path in CLIENT_PATHS:
            stack.enter_context(patch(path))
        self.addCleanup(stack.close)
        self.manager = ChallengePeriodManager(is_backtesting=True)
        self.accounts = self.manager._miner_account_client
        self.accounts.get_standard_account.return_value = None

    def _first_sight(self, hotkey, after_cutoff, opened_record=None, leverage_tier=TIER):
        self.accounts.is_registered_after_cutoff.return_value = after_cutoff
        self.accounts.get_standard_account.return_value = opened_record
        self.accounts.get_account.return_value = MinerAccount(miner_hotkey=hotkey, leverage_tier=leverage_tier)
        return self.manager._sync_positions([hotkey], [], {hotkey: NOW_MS + 1}, NOW_MS)

    def test_a_legacy_hotkey_starts_in_challenge(self):
        self.assertTrue(self._first_sight(LEGACY, False))
        state = self.manager.miner_states[LEGACY]
        self.assertEqual((state.current_bucket, state.current_bucket_start_ms), (MinerBucket.CHALLENGE, NOW_MS + 1))

    def test_an_opened_hotkey_starts_in_the_standard_challenge_from_its_record(self):
        self.assertTrue(self._first_sight(NEW, None, record(), leverage_tier=None))  # the record decides, not the registration
        state = self.manager.miner_states[NEW]
        self.assertEqual((state.current_bucket, state.current_bucket_start_ms), (MinerBucket.SUBACCOUNT_CHALLENGE, NOW_MS))
        self.assertEqual(state.intraday_drawdown_threshold_override, ValiConfig.STANDARD_MINER_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(state.eod_hwm_threshold_override, ValiConfig.STANDARD_MINER_EOD_HWM_THRESHOLD)
        self.accounts.set_leverage_tier.assert_called_once_with(NEW, TIER)

    def test_nothing_before_the_opening_or_while_the_registration_is_unknown(self):
        self.assertFalse(self._first_sight(NEW, True, None))
        self.assertFalse(self._first_sight(NEW, True, record(closed_at_ms=NOW_MS)))
        self.assertFalse(self._first_sight(NEW, None))
        self.assertNotIn(NEW, self.manager.miner_states)


class TestOpeningTransport(TestBase):
    """The mothership's broadcast of an opening and a peer's application of it, on in-process contract managers
    whose clients reach the test servers, and the synapse handler through the test contract server."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.contract_client = cls.orchestrator.get_client('contract')
        cls.account_client = cls.orchestrator.get_client('miner_account')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.elimination_client = cls.orchestrator.get_client('elimination')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()
        self.metagraph_client.set_hotkeys(HOTKEYS, block_at_registration=BLOCKS, stake_weight=STAKE_WEIGHTS,
                                          standard_miner_cutoff_block=CUTOFF)
        self.account_client.update_asset_selection(NEW, MinerAssetClass.CRYPTO)
        mothership_hotkey = patch.object(ValiConfig, "MOTHERSHIP_HOTKEY_TESTNET", MOTHERSHIP)
        mothership_hotkey.start()
        self.addCleanup(mothership_hotkey.stop)
        self.mothership = self._manager(MOTHERSHIP, is_mothership=True)
        self.mothership.set_test_collateral_balance(NEW, int(5.5 * RAO))
        self.mothership._broadcast_to_validators = Mock()
        self.peer = self._manager(PEER, is_mothership=False)

    def tearDown(self):
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    @staticmethod
    def _manager(hotkey, is_mothership):
        config = SimpleNamespace(netuid=116, wallet=SimpleNamespace(hotkey=hotkey),
                                 subtensor=SimpleNamespace(network="test"))
        manager = ValidatorContractManager(config=config, running_unit_tests=True)
        manager.is_mothership = is_mothership
        return manager

    def _broadcast(self) -> dict:
        """What the mothership broadcast: the payload of the synapse its one broadcast builds."""
        self.mothership._broadcast_to_validators.assert_called_once()
        kwargs = self.mothership._broadcast_to_validators.call_args.kwargs
        self.assertEqual(kwargs["broadcast_name"], "StandardMinerOpening")
        synapse = kwargs["synapse_factory"]()
        self.assertIsInstance(synapse, template.protocol.StandardMinerOpening)
        return pickle.loads(pickle.dumps(synapse)).opening

    def _receive(self, opening, sender=MOTHERSHIP) -> bool:
        return self.peer.receive_standard_miner_opening_update(opening, sender_hotkey=sender)

    def test_a_fresh_opening_is_broadcast_once_with_its_record(self):
        opened, message, record_dict = self.mothership.open_standard_miner_account(NEW, TIER)
        self.assertTrue(opened, message)
        self.assertEqual(self._broadcast(), {"hotkey": NEW, "record": record_dict})
        self.assertEqual(StandardMinerAccount.from_dict(record_dict), self.account_client.get_standard_account(NEW))
        # A repeated request is answered from the record, not broadcast again
        opened, message, again = self.mothership.open_standard_miner_account(NEW, 3)
        self.assertTrue(opened)
        self.assertIn("already opened", message)
        self.assertEqual(again, record_dict)
        self.mothership._broadcast_to_validators.assert_called_once()
        # Nor is an eliminated hotkey's
        self.elimination_client.append_elimination_row(NEW, EliminationReason.INACTIVE)
        opened, message, _ = self.mothership.open_standard_miner_account(NEW, TIER)
        self.assertFalse(opened)
        self.assertIn("eliminated", message)
        self.mothership._broadcast_to_validators.assert_called_once()

    def test_a_failure_after_the_record_is_written_still_broadcasts_it(self):
        self.mothership._challenge_period_client = Mock()
        self.mothership._challenge_period_client.has_miner.return_value = False
        self.mothership._challenge_period_client.set_miner_bucket.side_effect = RuntimeError("challenge period server unavailable")
        with self.assertRaises(RuntimeError):
            self.mothership.open_standard_miner_account(NEW, TIER)
        self.assertEqual(self._broadcast()["hotkey"], NEW)
        self.assertIsNotNone(self.account_client.get_standard_account(NEW))
        self.assertIsNone(self.challenge_period_client.get_miner_bucket(NEW))

    def test_an_opening_interrupted_before_its_state_is_completed_and_broadcast_on_retry(self):
        self.account_client.set_standard_account(NEW, record())
        opened, message, record_dict = self.mothership.open_standard_miner_account(NEW, TIER)
        self.assertTrue(opened, message)
        self.assertIn("Opened", message)
        self.assertEqual(self._broadcast(), {"hotkey": NEW, "record": record().to_dict()})
        self.assertEqual(self.challenge_period_client.get_miner_bucket(NEW), MinerBucket.SUBACCOUNT_CHALLENGE)
        opened, message, _ = self.mothership.open_standard_miner_account(NEW, TIER)
        self.assertIn("already opened", message)
        self.mothership._broadcast_to_validators.assert_called_once()

    def test_only_the_mothership_opens_accounts(self):
        # The guard is skipped under unit tests, where no server is the mothership
        self.peer.running_unit_tests = False
        try:
            opened, message, record_dict = self.peer.open_standard_miner_account(NEW, TIER)
        finally:
            self.peer.running_unit_tests = True
        self.assertFalse(opened)
        self.assertIn("mothership", message)
        self.assertIsNone(record_dict)
        self.assertIsNone(self.account_client.get_standard_account(NEW))
        self.assertIsNone(self.challenge_period_client.get_miner_bucket(NEW))

    def test_a_peer_applies_the_broadcast_record_as_it_is(self):
        # The peer has no collateral balance for the hotkey and does not need one: nothing is recomputed
        self.assertTrue(self._receive({"hotkey": NEW, "record": record().to_dict()}))
        self.assertEqual(self.account_client.get_standard_account(NEW), record())
        account = self.account_client.get_account(NEW)
        self.assertEqual((account.leverage_tier, account.miner_bucket), (TIER, MinerBucket.SUBACCOUNT_CHALLENGE))
        self.assertAlmostEqual(self.account_client.get_miner_account_size(NEW), 50_000.0)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(NEW), MinerBucket.SUBACCOUNT_CHALLENGE)
        self.assertEqual(self.challenge_period_client.get_miner_start_time(NEW), NOW_MS)
        state = self.challenge_period_client.to_checkpoint_dict()[NEW]
        self.assertEqual(state["intraday_drawdown_threshold_override"], ValiConfig.STANDARD_MINER_INTRADAY_DRAWDOWN_THRESHOLD)
        self.assertEqual(state["eod_hwm_threshold_override"], ValiConfig.STANDARD_MINER_EOD_HWM_THRESHOLD)
        # A repeat, even with other values, changes nothing
        self.assertTrue(self._receive({"hotkey": NEW, "record": record(opening_leverage_tier=3, opened_at_ms=NOW_MS + 1).to_dict()}))
        self.assertEqual(self.account_client.get_standard_account(NEW), record())
        self.assertEqual(self.account_client.get_account(NEW).leverage_tier, TIER)
        self.assertEqual(self.challenge_period_client.get_miner_start_time(NEW), NOW_MS)

    def test_a_peer_keeps_a_record_further_along(self):
        self.account_client.set_standard_account(NEW, record(pro_account_size=250_000.0))
        self.assertTrue(self._receive({"hotkey": NEW, "record": record().to_dict()}))
        self.assertEqual(self.account_client.get_standard_account(NEW), record(pro_account_size=250_000.0))

    def test_a_closed_account_gets_no_challenge_period_state_on_a_peer(self):
        self.account_client.set_standard_account(NEW, record(closed_at_ms=NOW_MS))
        self.assertTrue(self._receive({"hotkey": NEW, "record": record().to_dict()}))
        self.assertEqual(self.account_client.get_standard_account(NEW), record(closed_at_ms=NOW_MS))
        self.assertIsNone(self.challenge_period_client.get_miner_bucket(NEW))

    def test_the_axon_handler_keeps_the_ordinary_call_timeout(self):
        client = ContractClient(running_unit_tests=True)
        client._invoke_rpc = Mock(return_value="answered")
        synapse = template.protocol.StandardMinerOpening(opening={"hotkey": NEW})
        self.assertEqual(client.receive_standard_miner_opening(synapse), "answered")
        client._invoke_rpc.assert_called_once_with("receive_standard_miner_opening_rpc", args=(synapse,),
                                                   timeout_s=RPCClientBase.RPC_CALL_TIMEOUT_S)
        self.assertGreater(ContractClient.RPC_CALL_TIMEOUT_S, RPCClientBase.RPC_CALL_TIMEOUT_S)

    def test_what_a_peer_refuses(self):
        for opening, sender in (
            ({"hotkey": NEW, "record": record().to_dict()}, PEER),  # not the mothership
            ({"hotkey": NEW, "record": record().to_dict()}, None),
            ({"record": record().to_dict()}, MOTHERSHIP),
            ({"hotkey": NEW, "record": "not a record"}, MOTHERSHIP),
            ({"hotkey": NEW, "record": {"opened_at_ms": NOW_MS}}, MOTHERSHIP),  # fields missing
            ({"hotkey": NEW, "record": {**record().to_dict(), "account_size": -1.0}}, MOTHERSHIP),
            ({"hotkey": f"{ENTITY}_1", "record": record().to_dict()}, MOTHERSHIP),  # a subaccount
        ):
            with self.subTest(opening=opening, sender=sender):
                self.assertFalse(self._receive(opening, sender))
        self.assertIsNone(self.account_client.get_standard_account(NEW))
        self.assertIsNone(self.account_client.get_account(f"{ENTITY}_1"))
        self.assertIsNone(self.challenge_period_client.get_miner_bucket(NEW))
        # The mothership ignores its own broadcast
        self.assertFalse(self.mothership.receive_standard_miner_opening_update(
            {"hotkey": NEW, "record": record().to_dict()}, sender_hotkey=MOTHERSHIP))
        self.assertIsNone(self.account_client.get_standard_account(NEW))

    def test_the_synapse_handler_applies_the_opening_and_reports_the_outcome(self):
        synapse = template.protocol.StandardMinerOpening(opening={"hotkey": NEW, "record": record().to_dict()})
        synapse.dendrite.hotkey = PEER
        answered = self.contract_client.receive_standard_miner_opening(synapse)
        self.assertFalse(answered.successfully_processed)
        self.assertTrue(answered.error_message)
        self.assertIsNone(self.account_client.get_standard_account(NEW))
        # The test contract server verifies the sender in its own process, against the configured testnet mothership
        synapse.dendrite.hotkey = REAL_TESTNET_MOTHERSHIP
        answered = self.contract_client.receive_standard_miner_opening(synapse)
        self.assertTrue(answered.successfully_processed, answered.error_message)
        self.assertEqual(answered.error_message, "")
        self.assertEqual(self.account_client.get_standard_account(NEW), record())
        self.assertEqual(self.challenge_period_client.get_miner_bucket(NEW), MinerBucket.SUBACCOUNT_CHALLENGE)
