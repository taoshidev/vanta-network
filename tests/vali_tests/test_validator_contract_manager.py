# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Integration tests for ValidatorContractManager using server/client architecture.
Tests end-to-end contract management scenarios with real server infrastructure.

Data injection pattern (polygon_data_service.py): Tests inject collateral balances via
set_test_collateral_balance() instead of mocking CollateralManager. This avoids network
calls to the blockchain while maintaining proper multiprocess isolation.
"""
import time

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.miner_account.miner_account_manager import CollateralRecord
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig


class TestValidatorContractManager(TestBase):
    """
    Integration tests for ValidatorContractManager using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    contract_client = None
    miner_account_client = None

    # Test constants
    MINER_1 = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    MINER_2 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    DAY_MS = 1000 * 60 * 60 * 24

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.contract_client = cls.orchestrator.get_client('contract')
        cls.miner_account_client = cls.orchestrator.get_client('miner_account')

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No cleanup needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # Clear all test data (includes contract-specific cleanup as of orchestrator update)
        self.orchestrator.clear_all_test_data()

        # Inject default test balances using data injection pattern (like polygon_data_service.py)
        self.contract_client.set_test_collateral_balance(self.MINER_1, 1000000)  # 1M rao
        self.contract_client.set_test_collateral_balance(self.MINER_2, 500000)   # 500K rao

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        # Clear all test data (includes contract-specific cleanup)
        self.orchestrator.clear_all_test_data()

    def test_collateral_record_creation(self):
        """Test CollateralRecord creation and properties"""
        timestamp_ms = int(time.time() * 1000)
        account_size = 10000.0
        account_size_theta = 10000.0 / ValiConfig.COST_PER_THETA

        record = CollateralRecord(account_size, account_size_theta, timestamp_ms)

        self.assertEqual(record.account_size, account_size)
        self.assertEqual(record.update_time_ms, timestamp_ms)
        self.assertIsInstance(record.valid_date_timestamp, int)
        self.assertIsInstance(record.valid_date_str, str)

        # Test date string format
        self.assertRegex(record.valid_date_str, r'^\d{4}-\d{2}-\d{2}$')

    def test_set_and_get_miner_account_size(self):
        """Test setting and getting miner account sizes"""
        current_time = int(time.time() * 1000)

        # Initially should return None for non-existent miner
        self.assertIsNone(self.miner_account_client.get_miner_account_size(self.MINER_1))

        # Set account size via MinerAccountClient (pass collateral balance in theta directly)
        # 1M rao = 0.001 theta
        collateral_theta_1 = 1000000 / 10**9
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta_1, current_time)

        # First record should be available SAME day!
        account_size = self.miner_account_client.get_miner_account_size(self.MINER_1, current_time)
        self.assertIsNotNone(account_size)

        # Set for second miner (500K rao = 0.0005 theta)
        collateral_theta_2 = 500000 / 10**9
        self.miner_account_client.set_miner_account_size(self.MINER_2, collateral_theta_2, current_time)
        account_size_2 = self.miner_account_client.get_miner_account_size(self.MINER_2, current_time)
        self.assertIsNotNone(account_size_2)

    def test_account_size_persistence(self):
        """Test that account sizes are saved to and loaded from disk"""
        current_time = int(time.time() * 1000)

        # Set account size via MinerAccountClient (1M rao = 0.001 theta)
        collateral_theta = 1000000 / 10**9
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta, current_time)

        # Verify it was set (first record should be available same day)
        account_size = self.miner_account_client.get_miner_account_size(self.MINER_1, current_time)
        self.assertIsNotNone(account_size)

        # Test the disk persistence by checking via accounts_dict
        # accounts_dict returns [collateral_records..., account_summary_dict]
        account_sizes_dict = self.miner_account_client.accounts_dict()
        self.assertIn(self.MINER_1, account_sizes_dict)
        self.assertEqual(len(account_sizes_dict[self.MINER_1]), 2)  # 1 collateral record + 1 account summary

    def test_multiple_account_size_records(self):
        """Test that multiple records are stored and sorted correctly"""
        base_time = int(time.time() * 1000)

        # Set different collateral balances directly via MinerAccountClient
        collateral_theta_1 = 1_000_000 / 10**9  # 0.001 theta
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta_1, base_time)

        collateral_theta_2 = 2_000_000 / 10**9  # 0.002 theta
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta_2, base_time + 1000)

        collateral_theta_3 = 3_000_000 / 10**9  # 0.003 theta
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta_3, base_time + 2000)

        # Verify records are stored
        # accounts_dict returns [collateral_records..., account_summary_dict]
        account_sizes_dict = self.miner_account_client.accounts_dict()
        records = account_sizes_dict[self.MINER_1]
        self.assertEqual(len(records), 4)  # 3 collateral records + 1 account summary

        # Verify collateral records are sorted by update_time_ms (exclude account summary at end)
        collateral_records = records[:-1]
        for i in range(1, len(collateral_records)):
            self.assertGreaterEqual(collateral_records[i]['update_time_ms'], collateral_records[i-1]['update_time_ms'])

    def test_no_duplicate_account_size_records(self):
        """Test that duplicate records are ignored"""
        base_time = int(time.time() * 1000)

        # Use same balance for all calls (1M rao = 0.001 theta)
        # Add multiple records with different timestamps but same balance
        collateral_theta = 1_000_000 / 10**9
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta, base_time)
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta, base_time + 1000)
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta, base_time + 2000)

        # Verify only one record is stored (duplicates are skipped)
        # accounts_dict returns [collateral_records..., account_summary_dict]
        account_sizes_dict = self.miner_account_client.accounts_dict()
        records = account_sizes_dict[self.MINER_1]
        self.assertEqual(len(records), 2)  # 1 collateral record + 1 account summary

    def test_sync_miner_account_sizes_data(self):
        """Test syncing miner account sizes from external data"""
        # Create test data in the format expected by sync method
        test_data = {
            self.MINER_1: [
                {
                    "account_size": 15000.0,
                    "account_size_theta": 1000.0,
                    "update_time_ms": int(time.time() * 1000) - 1000,
                    "valid_date_timestamp": CollateralRecord.valid_from_ms(int(time.time() * 1000) - 1000)
                }
            ],
            self.MINER_2: [
                {
                    "account_size": 25000.0,
                    "account_size_theta": 2000.0,
                    "update_time_ms": int(time.time() * 1000),
                    "valid_date_timestamp": CollateralRecord.valid_from_ms(int(time.time() * 1000))
                }
            ]
        }

        # Sync the data via MinerAccountClient
        self.miner_account_client.sync_miner_account_sizes_data(test_data)

        # Verify data was synced correctly
        account_sizes_dict = self.miner_account_client.accounts_dict()
        self.assertIn(self.MINER_1, account_sizes_dict)
        self.assertIn(self.MINER_2, account_sizes_dict)

        # Check the records
        # accounts_dict returns [collateral_records..., account_summary_dict]
        miner1_records = account_sizes_dict[self.MINER_1]
        miner2_records = account_sizes_dict[self.MINER_2]

        self.assertEqual(len(miner1_records), 2)  # 1 collateral record + 1 account summary
        self.assertEqual(len(miner2_records), 2)  # 1 collateral record + 1 account summary
        self.assertEqual(miner1_records[0]['account_size'], 15000.0)
        self.assertEqual(miner2_records[0]['account_size'], 25000.0)

    def test_to_checkpoint_dict(self):
        """Test converting account sizes to checkpoint dictionary format"""
        current_time = int(time.time() * 1000)

        # Set account sizes via MinerAccountClient
        collateral_theta_1 = 1000000 / 10**9  # 0.001 theta
        collateral_theta_2 = 500000 / 10**9   # 0.0005 theta
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta_1, current_time)
        self.miner_account_client.set_miner_account_size(self.MINER_2, collateral_theta_2, current_time)

        # Get checkpoint dict
        checkpoint_dict = self.miner_account_client.accounts_dict()

        # Verify structure
        self.assertIsInstance(checkpoint_dict, dict)
        self.assertIn(self.MINER_1, checkpoint_dict)
        self.assertIn(self.MINER_2, checkpoint_dict)

        # Verify record structure
        # accounts_dict returns [collateral_records..., account_summary_dict]
        # Only collateral records have update_time_ms, account summary (last item) doesn't
        for hotkey, records in checkpoint_dict.items():
            self.assertIsInstance(records, list)
            # Check collateral records (all but last)
            for record in records[:-1]:
                self.assertIn('account_size', record)
                self.assertIn('update_time_ms', record)
                self.assertIn('valid_date_timestamp', record)
            # Check account summary (last record)
            account_summary = records[-1]
            self.assertIn('account_size', account_summary)
            self.assertIn('miner_hotkey', account_summary)

    def test_collateral_balance_retrieval(self):
        """Test getting collateral balance for miners"""
        # Inject test balance
        self.contract_client.set_test_collateral_balance(self.MINER_1, 1500000)  # 1.5M rao

        # Get balance (should return test balance via data injection)
        balance = self.contract_client.get_miner_collateral_balance(self.MINER_1)
        self.assertIsNotNone(balance)

        # Verify conversion to theta (1.5M rao = 0.0015 theta)
        expected_theta = 0.0015
        self.assertAlmostEqual(balance, expected_theta, places=4)

    def test_query_withdrawal_request_slash_formula_accuracy(self):
        """
        Test the drawdown -> slash formula exposed by query_withdrawal_request across a range
        of drawdowns. This is the only production path that turns an equity drawdown into a
        slashed_amount without mutating state (no chain calls, no elimination triggered), so it's
        the right surface to pin the formula against -- there is no standalone
        ContractClient.compute_slash_amount() method.

        With no collateral records, MinerAccount.account_size is fixed at ValiConfig.MIN_CAPITAL,
        so crediting a known realized loss via process_order_sell produces an exact,
        reproducible current_return = 1 - drawdown_pct/100 (max_return stays at the 1.0 default
        since only update_unrealized_pnl advances the high-water mark). With no bucket and no
        elimination on file, query_withdrawal_request falls back to a drawdown_threshold of
        (1 - ValiConfig.MAX_TOTAL_DRAWDOWN) == 0.10, matching the 10%-drawdown-> 100%-slash cases
        below.
        """
        balance_theta = 1000.0
        balance_rao = int(balance_theta * 10 ** 9)
        self.contract_client.set_test_collateral_balance(self.MINER_1, balance_rao)
        account_size = ValiConfig.MIN_CAPITAL

        # Test cases: (drawdown_pct, expected_slash_proportion)
        test_cases = [
            (0.0, 0.0),    # 0% drawdown -> 0% slash
            (1.0, 0.1),    # 1% drawdown -> 10% slash
            (2.0, 0.2),    # 2% drawdown -> 20% slash
            (3.0, 0.3),    # 3% drawdown -> 30% slash
            (4.0, 0.4),    # 4% drawdown -> 40% slash
            (5.0, 0.5),    # 5% drawdown -> 50% slash
            (6.0, 0.6),    # 6% drawdown -> 60% slash
            (7.0, 0.7),    # 7% drawdown -> 70% slash
            (8.0, 0.8),    # 8% drawdown -> 80% slash
            (9.0, 0.9),    # 9% drawdown -> 90% slash
            (10.0, 1.0),   # 10% drawdown -> 100% slash (elimination threshold)
        ]

        for dd_pct, expected_proportion in test_cases:
            with self.subTest(drawdown_pct=dd_pct):
                self.miner_account_client.reset_account(self.MINER_1)
                loss = (dd_pct / 100.0) * account_size
                if loss > 0:
                    self.miner_account_client.process_order_sell(
                        self.MINER_1, entry_value_usd=0, realized_pnl=-loss, loan_repaid=0)

                result = self.contract_client.query_withdrawal_request(balance_theta, self.MINER_1)
                self.assertTrue(result["successfully_processed"], result.get("error_message"))

                expected_slash = balance_theta * expected_proportion
                self.assertAlmostEqual(result["drawdown"] * 100, dd_pct, places=4)
                self.assertAlmostEqual(result["slashed_amount"], expected_slash, places=1,
                                       msg=f"{dd_pct}% drawdown should slash {expected_proportion*100}% of balance. "
                                           f"Expected {expected_slash}, got {result['slashed_amount']}")

    def test_get_all_miner_account_sizes(self):
        """Test getting all miner account sizes at a specific timestamp"""
        current_time = int(time.time() * 1000)

        # Set account sizes for multiple miners via MinerAccountClient
        collateral_theta_1 = 1000000 / 10**9  # 0.001 theta
        collateral_theta_2 = 500000 / 10**9   # 0.0005 theta
        self.miner_account_client.set_miner_account_size(self.MINER_1, collateral_theta_1, current_time)
        self.miner_account_client.set_miner_account_size(self.MINER_2, collateral_theta_2, current_time)

        # Get all account sizes (first records should be available same day)
        all_sizes = self.miner_account_client.get_all_miner_account_sizes(timestamp_ms=current_time)

        # Verify both miners are present
        self.assertIn(self.MINER_1, all_sizes)
        self.assertIn(self.MINER_2, all_sizes)
        self.assertIsNotNone(all_sizes[self.MINER_1])
        self.assertIsNotNone(all_sizes[self.MINER_2])


class TestBoundedOwnershipQuery(TestBase):
    """verify_coldkey_owns_hotkey must never pin its caller on a hung chain
    endpoint: the subtensor fallback runs on a bounded executor with a timeout,
    and failures enter a cooldown so a down chain can't stack threads."""

    def _make_manager(self, query_side_effect):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import MagicMock
        from vali_objects.contract.validator_contract_manager import ValidatorContractManager

        m = object.__new__(ValidatorContractManager)
        m._coldkey_hotkey_cache_lock = threading.Lock()
        m._coldkey_hotkey_cache = {}
        m._ownership_failure_ts = {}
        m._ownership_query_executor = ThreadPoolExecutor(max_workers=2)
        m.OWNERSHIP_QUERY_TIMEOUT_S = 0.2
        m.OWNERSHIP_FAILURE_COOLDOWN_S = 30.0
        m._metagraph_client = MagicMock()
        m._metagraph_client.get_neurons.return_value = []  # force the chain fallback
        m.collateral_manager = MagicMock()
        m.collateral_manager.subtensor_api.queries.query_subtensor.side_effect = query_side_effect
        return m

    def test_hung_chain_query_returns_false_within_timeout(self):
        import time as _time

        def hang(*a, **k):
            _time.sleep(5)
            return "5ColdKey"

        m = self._make_manager(hang)
        start = _time.monotonic()
        result = m.verify_coldkey_owns_hotkey("5ColdKey", "5HotKey")
        elapsed = _time.monotonic() - start
        self.assertFalse(result)
        self.assertLess(elapsed, 2.0, f"caller was pinned for {elapsed:.1f}s")
        self.assertIn(("5ColdKey", "5HotKey"), m._ownership_failure_ts)

    def test_failure_cooldown_short_circuits_chain(self):
        m = self._make_manager(RuntimeError("chain down"))
        self.assertFalse(m.verify_coldkey_owns_hotkey("5CK", "5HK"))
        calls_after_first = m.collateral_manager.subtensor_api.queries.query_subtensor.call_count
        # Within the cooldown the chain must NOT be queried again.
        self.assertFalse(m.verify_coldkey_owns_hotkey("5CK", "5HK"))
        self.assertEqual(
            m.collateral_manager.subtensor_api.queries.query_subtensor.call_count,
            calls_after_first,
        )
        # Once the cooldown lapses the query is attempted again.
        m.OWNERSHIP_FAILURE_COOLDOWN_S = 0.0
        self.assertFalse(m.verify_coldkey_owns_hotkey("5CK", "5HK"))
        self.assertGreater(
            m.collateral_manager.subtensor_api.queries.query_subtensor.call_count,
            calls_after_first,
        )

    def test_successful_query_caches_and_clears_failure(self):
        m = self._make_manager(lambda *a, **k: "5Owner")
        m._ownership_failure_ts[("5Owner", "5HK")] = 0.0  # stale failure entry
        self.assertTrue(m.verify_coldkey_owns_hotkey("5Owner", "5HK"))
        self.assertEqual(m._coldkey_hotkey_cache[("5Owner", "5HK")], True)
        self.assertNotIn(("5Owner", "5HK"), m._ownership_failure_ts)
        # Cached: no further chain calls.
        calls = m.collateral_manager.subtensor_api.queries.query_subtensor.call_count
        self.assertTrue(m.verify_coldkey_owns_hotkey("5Owner", "5HK"))
        self.assertEqual(m.collateral_manager.subtensor_api.queries.query_subtensor.call_count, calls)

    def test_definitive_negative_is_cached(self):
        m = self._make_manager(lambda *a, **k: "5SomeoneElse")
        self.assertFalse(m.verify_coldkey_owns_hotkey("5CK", "5HK"))
        self.assertEqual(m._coldkey_hotkey_cache[("5CK", "5HK")], False)


class TestSlashBurnProportion(TestBase):
    """slash_miner_collateral debits the full slash from the miner but burns only
    SLASH_BURN_PROPORTION of it from the vault stake."""

    def _make_manager(self, balance_theta):
        from unittest.mock import MagicMock
        from vali_objects.contract.validator_contract_manager import ValidatorContractManager

        m = object.__new__(ValidatorContractManager)
        m.is_mothership = True
        m.is_testnet = False
        m.wallet = MagicMock()
        m.collateral_manager = MagicMock()
        m.get_miner_collateral_balance = MagicMock(return_value=balance_theta)
        return m

    def test_burns_proportion_of_slash(self):
        from unittest.mock import patch

        m = self._make_manager(500.0)
        with patch.object(ValiUtils, "get_secret", return_value="secret"):
            self.assertTrue(m.slash_miner_collateral("5HK", 100.0))

        slash_rao = int(100.0 * 10 ** 9)
        self.assertEqual(m.collateral_manager.slash.call_args.kwargs["amount"], slash_rao)
        self.assertEqual(
            m.collateral_manager.burn.call_args.kwargs["amount"],
            int(slash_rao * ValiConfig.SLASH_BURN_PROPORTION),
        )

    def test_burn_failure_still_reports_slash_success(self):
        from unittest.mock import patch

        m = self._make_manager(500.0)
        m.collateral_manager.burn.side_effect = RuntimeError("burn failed")
        with patch.object(ValiUtils, "get_secret", return_value="secret"):
            self.assertTrue(m.slash_miner_collateral("5HK", 100.0))
        m.collateral_manager.slash.assert_called_once()

    def test_non_mothership_slash_is_dry_run(self):
        m = self._make_manager(500.0)
        m.is_mothership = False
        self.assertTrue(m.slash_miner_collateral("5HK", 100.0))
        self.assertTrue(m.slash_miner_collateral_proportion("5HK", 0.5))
        m.collateral_manager.slash.assert_not_called()
        m.collateral_manager.burn.assert_not_called()

    def test_non_mothership_slash_still_validates(self):
        m = self._make_manager(0.0)
        m.is_mothership = False
        self.assertFalse(m.slash_miner_collateral("5HK", 100.0))
        self.assertFalse(m.slash_miner_collateral("5HK", -1.0))
        self.assertFalse(m.slash_miner_collateral_proportion("5HK", 1.5))
        m.collateral_manager.slash.assert_not_called()


class TestCollateralEndpoints(TestBase):
    """--collateral.chain_endpoint routes collateral traffic (substrate and EVM) through a self-hosted
    node while it is healthy; everything else, and collateral whenever the node is unhealthy, stays on
    the public endpoints."""

    NODE = "ws://10.0.0.5:9944"

    def _make_manager(self, network="finney", collateral_endpoint=None, subtensor_endpoint=None,
                      problem=None, node_constructor_error=None):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from shared_objects.rpc.test_mock_factory import TestMockFactory
        from vali_objects.vali_config import RPCConnectionMode
        import vali_objects.contract.validator_contract_manager as vcm

        config = TestMockFactory.create_mock_config(netuid=8, network=network)
        if subtensor_endpoint is not None:
            config.subtensor.chain_endpoint = subtensor_endpoint
        if collateral_endpoint is not None:
            config.collateral = SimpleNamespace(chain_endpoint=collateral_endpoint)

        def build(network, **kwargs):
            if kwargs.get("evm_endpoint") and node_constructor_error:
                raise node_constructor_error
            return MagicMock(name="node" if kwargs.get("evm_endpoint") else "public")

        slack = MagicMock()
        with patch.object(vcm, "CollateralManager", side_effect=build) as collateral_manager_cls, \
                patch.object(vcm.ValidatorContractManager, "_collateral_node_problem", return_value=problem):
            m = vcm.ValidatorContractManager(config=config, running_unit_tests=True,
                                             connection_mode=RPCConnectionMode.LOCAL, slack_notifier=slack)
        return m, collateral_manager_cls, slack

    def test_no_flag_uses_public_endpoints_only(self):
        from collateral_sdk import Network

        m, cls, slack = self._make_manager()
        self.assertEqual([c.args for c in cls.call_args_list], [(Network.MAINNET,)])
        self.assertEqual(cls.call_args.kwargs, {})
        self.assertIs(m.collateral_manager, m._public_collateral_manager)
        slack.send_message.assert_not_called()

    def test_subtensor_chain_endpoint_does_not_route_collateral(self):
        m, cls, _ = self._make_manager(subtensor_endpoint="ws://127.0.0.1:9944")
        self.assertEqual(cls.call_count, 1)
        self.assertEqual(cls.call_args.kwargs, {})
        self.assertIs(m.collateral_manager, m._public_collateral_manager)

    def test_healthy_node_carries_collateral(self):
        from collateral_sdk import Network

        m, cls, slack = self._make_manager(collateral_endpoint=self.NODE)
        self.assertEqual(cls.call_args.args, (Network.MAINNET,))
        self.assertEqual(cls.call_args.kwargs, {"evm_endpoint": "http://10.0.0.5:9944", "subtensor_endpoint": self.NODE})
        self.assertIs(m.collateral_manager, m._node_collateral_manager)
        slack.send_message.assert_not_called()

    def test_wss_endpoint_maps_to_https(self):
        _, cls, _ = self._make_manager(collateral_endpoint="wss://node.example.com:443")
        self.assertEqual(cls.call_args.kwargs["evm_endpoint"], "https://node.example.com:443")
        self.assertEqual(cls.call_args.kwargs["subtensor_endpoint"], "wss://node.example.com:443")

    def test_testnet_keeps_testnet(self):
        from collateral_sdk import Network

        m, cls, _ = self._make_manager(network="test", collateral_endpoint="ws://127.0.0.1:9945")
        self.assertTrue(all(c.args == (Network.TESTNET,) for c in cls.call_args_list))
        self.assertIs(m.collateral_manager, m._node_collateral_manager)

    def test_unhealthy_node_at_startup_falls_back_and_alerts(self):
        m, cls, slack = self._make_manager(collateral_endpoint=self.NODE, problem="unreachable: refused")
        self.assertEqual(cls.call_count, 1)  # only the public manager
        self.assertIsNone(m._node_collateral_manager)
        self.assertIs(m.collateral_manager, m._public_collateral_manager)
        self.assertIn("unreachable", slack.send_message.call_args.args[0])

    def test_node_constructor_failure_falls_back(self):
        m, _, slack = self._make_manager(collateral_endpoint=self.NODE,
                                         node_constructor_error=ValueError("serves chain id 945, expected 964"))
        self.assertIs(m.collateral_manager, m._public_collateral_manager)
        self.assertIn("945", slack.send_message.call_args.args[0])

    def test_non_websocket_endpoint_uses_public_endpoints(self):
        m, cls, _ = self._make_manager(collateral_endpoint="http://10.0.0.5:9944")
        self.assertEqual(cls.call_count, 1)
        self.assertIs(m.collateral_manager, m._public_collateral_manager)

    def _run_monitor(self, m, problems):
        """Run the monitor for len(problems) checks, the probe returning each in turn; returns the
        max_lag_blocks the probe was called with."""
        from unittest.mock import MagicMock, patch
        import vali_objects.contract.validator_contract_manager as vcm
        from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator

        with patch.object(vcm.time, "sleep"), \
                patch.object(ShutdownCoordinator, "is_shutdown", side_effect=[False] * len(problems) + [True]), \
                patch.object(vcm, "CollateralManager", return_value=MagicMock(name="rebuilt node")), \
                patch.object(m, "_collateral_node_problem", side_effect=problems) as probe:
            m._monitor_collateral_endpoint()
        return [c.args[0] for c in probe.call_args_list]

    def test_monitor_fails_over_and_recovers(self):
        m, _, slack = self._make_manager(collateral_endpoint=self.NODE)
        node = m._node_collateral_manager
        stale = "stale: block 1, public endpoint at 100"

        # One failed check is not enough to leave; the second is
        lags = self._run_monitor(m, [stale])
        self.assertIs(m.collateral_manager, node)
        self.assertEqual(lags, [m.COLLATERAL_ENDPOINT_MAX_LAG_BLOCKS])
        lags = self._run_monitor(m, [stale, stale])
        self.assertIs(m.collateral_manager, m._public_collateral_manager)
        self.assertIn("moved to the public endpoints", slack.send_message.call_args.args[0])
        self.assertTrue(slack.send_message.call_args.kwargs["bypass_cooldown"])

        # Returning needs three passes in a row, each at the stricter lag
        lags = self._run_monitor(m, [None, None, stale, None, None, None])
        self.assertEqual(lags, [m.COLLATERAL_ENDPOINT_RETURN_LAG_BLOCKS] * 6)
        self.assertIs(m.collateral_manager, node)
        self.assertIn("moved back", slack.send_message.call_args.args[0])
        self.assertTrue(slack.send_message.call_args.kwargs["bypass_cooldown"])

    def test_monitor_does_not_flap_on_alternating_results(self):
        m, _, slack = self._make_manager(collateral_endpoint=self.NODE)
        self._run_monitor(m, ["stale: x", None] * 5)
        self.assertIs(m.collateral_manager, m._node_collateral_manager)
        slack.send_message.assert_not_called()

    def test_monitor_ignores_unmeasurable_lag(self):
        m, _, slack = self._make_manager(collateral_endpoint=self.NODE)
        unknown = m.COLLATERAL_NODE_LAG_UNKNOWN
        # Unknown neither counts as a failure nor breaks a failing streak
        self._run_monitor(m, ["stale: x", unknown, unknown])
        self.assertIs(m.collateral_manager, m._node_collateral_manager)
        self._run_monitor(m, ["stale: x", unknown, "stale: x"])
        self.assertIs(m.collateral_manager, m._public_collateral_manager)

        # On the public endpoints an unknown lag never moves collateral back to the node
        self._run_monitor(m, [unknown] * 5)
        self.assertIs(m.collateral_manager, m._public_collateral_manager)

    def test_monitor_builds_node_manager_when_startup_failed(self):
        m, _, _ = self._make_manager(collateral_endpoint=self.NODE, problem="unreachable: refused")
        self._run_monitor(m, [None] * m.COLLATERAL_ENDPOINT_PASSES_TO_RETURN)
        self.assertIsNotNone(m._node_collateral_manager)
        self.assertIs(m.collateral_manager, m._node_collateral_manager)

    def test_monitor_without_change_does_not_alert(self):
        m, _, slack = self._make_manager(collateral_endpoint=self.NODE)
        self._run_monitor(m, [None] * 3)
        self.assertIs(m.collateral_manager, m._node_collateral_manager)
        slack.send_message.assert_not_called()

    def test_startup_uses_node_when_only_lag_is_unknown(self):
        from types import SimpleNamespace
        import vali_objects.contract.validator_contract_manager as vcm

        m, _, slack = self._make_manager(collateral_endpoint=self.NODE,
                                         problem=vcm.ValidatorContractManager.COLLATERAL_NODE_LAG_UNKNOWN)
        self.assertIs(m.collateral_manager, m._node_collateral_manager)
        slack.send_message.assert_not_called()

    def test_startup_alert_bypasses_slack_cooldown(self):
        _, _, slack = self._make_manager(collateral_endpoint=self.NODE, problem="no peers")
        self.assertTrue(slack.send_message.call_args.kwargs["bypass_cooldown"])


class TestCollateralNodeProblem(TestBase):
    """The health probe behind collateral failover."""

    def _probe(self, node, public_block=1000, max_lag_blocks=20):
        from unittest.mock import MagicMock, patch
        from collateral_sdk import Network
        import vali_objects.contract.validator_contract_manager as vcm

        m = object.__new__(vcm.ValidatorContractManager)
        m._collateral_network = Network.MAINNET
        m._collateral_evm_endpoint = "http://10.0.0.5:9944"
        m._public_collateral_manager = MagicMock(evm_endpoint="https://lite.chain.opentensor.ai")

        def post(url, json, timeout):
            if url == m._collateral_evm_endpoint:
                if isinstance(node, Exception):
                    raise node
                result = node[json["method"]]
            elif public_block is None:
                raise ConnectionError("public down")
            else:
                result = {"number": hex(public_block)}
            return MagicMock(json=MagicMock(return_value={"result": result}), raise_for_status=MagicMock())

        with patch.object(vcm.requests, "post", side_effect=post):
            return m._collateral_node_problem(max_lag_blocks)

    def _node(self, **overrides):
        from collateral_sdk import Network
        import vali_objects.contract.validator_contract_manager as vcm

        node = {
            "chain_getBlockHash": vcm.ValidatorContractManager.GENESIS_HASHES[Network.MAINNET],
            "eth_chainId": hex(964),
            "system_health": {"isSyncing": False, "peers": 10},
            "chain_getHeader": {"number": hex(995)},
        }
        node.update(overrides)
        return node

    def test_healthy(self):
        self.assertIsNone(self._probe(self._node()))

    def test_wrong_genesis(self):
        from collateral_sdk import Network
        import vali_objects.contract.validator_contract_manager as vcm

        problem = self._probe(self._node(chain_getBlockHash=vcm.ValidatorContractManager.GENESIS_HASHES[Network.TESTNET]))
        self.assertIn("wrong chain", problem)

    def test_wrong_evm_chain_id(self):
        self.assertIn("EVM chain id 945", self._probe(self._node(eth_chainId=hex(945))))

    def test_syncing(self):
        self.assertEqual(self._probe(self._node(system_health={"isSyncing": True, "peers": 3})), "still syncing")

    def test_stale(self):
        self.assertIn("stale", self._probe(self._node(chain_getHeader={"number": hex(900)})))

    def test_unreachable(self):
        self.assertIn("unreachable", self._probe(ConnectionError("refused")))

    def test_public_endpoint_down_leaves_lag_unknown(self):
        import vali_objects.contract.validator_contract_manager as vcm

        self.assertEqual(self._probe(self._node(), public_block=None),
                         vcm.ValidatorContractManager.COLLATERAL_NODE_LAG_UNKNOWN)

    def test_public_endpoint_down_still_reports_node_problems(self):
        problem = self._probe(self._node(system_health={"isSyncing": False, "peers": 0, "shouldHavePeers": True}),
                              public_block=None)
        self.assertEqual(problem, "no peers")

    def test_no_peers(self):
        self.assertEqual(self._probe(self._node(system_health={"isSyncing": False, "peers": 0, "shouldHavePeers": True})),
                         "no peers")

    def test_lag_threshold_is_the_callers(self):
        # 5 blocks behind: fine to stay on (20), not yet to return to (2)
        self.assertIsNone(self._probe(self._node(), max_lag_blocks=20))
        self.assertIn("stale", self._probe(self._node(), max_lag_blocks=2))
