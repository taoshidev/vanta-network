# developer: jbonilla
# Copyright © 2024 Taoshi Inc
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
from vali_objects.contract.validator_contract_manager import CollateralRecord
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
        day_after_current_time = self.DAY_MS + current_time

        # Initially should return None for non-existent miner
        self.assertIsNone(self.contract_client.get_miner_account_size(self.MINER_1))

        # Set account size (ValidatorContractManager calculates account size from collateral)
        # Test balance already injected in setUp()
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)

        # Verify retrieval - should return the calculated account size
        account_size = self.contract_client.get_miner_account_size(self.MINER_1, day_after_current_time)
        self.assertIsNotNone(account_size)

        # Set for second miner (balance already injected in setUp())
        self.contract_client.set_miner_account_size(self.MINER_2, current_time)
        account_size_2 = self.contract_client.get_miner_account_size(self.MINER_2, day_after_current_time)
        self.assertIsNotNone(account_size_2)

    def test_account_size_persistence(self):
        """Test that account sizes are saved to and loaded from disk"""
        current_time = int(time.time() * 1000)
        day_after_current_time = self.DAY_MS + current_time

        # Set account size (balance already injected in setUp())
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)

        # Verify it was set
        account_size = self.contract_client.get_miner_account_size(self.MINER_1, day_after_current_time)
        self.assertIsNotNone(account_size)

        # Test the disk persistence by checking via miner_account_sizes_dict
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()
        self.assertIn(self.MINER_1, account_sizes_dict)
        self.assertEqual(len(account_sizes_dict[self.MINER_1]), 1)

    def test_multiple_account_size_records(self):
        """Test that multiple records are stored and sorted correctly"""
        base_time = int(time.time() * 1000)

        # Inject different collateral balances for each call
        self.contract_client.set_test_collateral_balance(self.MINER_1, 1_000_000)  # First call
        self.contract_client.set_miner_account_size(self.MINER_1, base_time)

        self.contract_client.set_test_collateral_balance(self.MINER_1, 2_000_000)  # Second call
        self.contract_client.set_miner_account_size(self.MINER_1, base_time + 1000)

        self.contract_client.set_test_collateral_balance(self.MINER_1, 3_000_000)  # Third call
        self.contract_client.set_miner_account_size(self.MINER_1, base_time + 2000)

        # Verify records are stored
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()
        records = account_sizes_dict[self.MINER_1]
        self.assertEqual(len(records), 3)

        # Verify records are sorted by update_time_ms
        for i in range(1, len(records)):
            self.assertGreaterEqual(records[i]['update_time_ms'], records[i-1]['update_time_ms'])

    def test_no_duplicate_account_size_records(self):
        """Test that duplicate records are ignored"""
        base_time = int(time.time() * 1000)

        # Use same balance for all calls (already injected in setUp() as 1M rao)
        # Add multiple records with different timestamps but same balance
        self.contract_client.set_miner_account_size(self.MINER_1, base_time)
        self.contract_client.set_miner_account_size(self.MINER_1, base_time + 1000)
        self.contract_client.set_miner_account_size(self.MINER_1, base_time + 2000)

        # Verify only one record is stored (duplicates are skipped)
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()
        records = account_sizes_dict[self.MINER_1]
        self.assertEqual(len(records), 1)

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

        # Sync the data

        self.contract_client.sync_miner_account_sizes_data(test_data)

        # Verify data was synced correctly
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()
        self.assertIn(self.MINER_1, account_sizes_dict)
        self.assertIn(self.MINER_2, account_sizes_dict)

        # Check the records
        miner1_records = account_sizes_dict[self.MINER_1]
        miner2_records = account_sizes_dict[self.MINER_2]

        self.assertEqual(len(miner1_records), 1)
        self.assertEqual(len(miner2_records), 1)
        self.assertEqual(miner1_records[0]['account_size'], 15000.0)
        self.assertEqual(miner2_records[0]['account_size'], 25000.0)

    def test_to_checkpoint_dict(self):
        """Test converting account sizes to checkpoint dictionary format"""
        current_time = int(time.time() * 1000)

        # Set account sizes (balances already injected in setUp())
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)
        self.contract_client.set_miner_account_size(self.MINER_2, current_time)

        # Get checkpoint dict
        checkpoint_dict = self.contract_client.miner_account_sizes_dict()

        # Verify structure
        self.assertIsInstance(checkpoint_dict, dict)
        self.assertIn(self.MINER_1, checkpoint_dict)
        self.assertIn(self.MINER_2, checkpoint_dict)

        # Verify record structure
        for hotkey, records in checkpoint_dict.items():
            self.assertIsInstance(records, list)
            for record in records:
                self.assertIn('account_size', record)
                self.assertIn('update_time_ms', record)
                self.assertIn('valid_date_timestamp', record)

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

    def test_compute_slash_amount_formula_accuracy(self):
        """Test the exact formula for slash calculation across range of drawdowns"""
        # Set up real collateral balance via data injection
        balance_theta = 1000.0
        balance_rao = int(balance_theta * 10 ** 9)
        self.contract_client.set_test_collateral_balance(self.MINER_1, balance_rao)

        # Test cases: (drawdown, expected_drawdown_percentage, expected_slash_proportion)
        test_cases = [
            (1.0, 0.0, 0.0),      # 0% drawdown -> 0% slash
            (0.99, 1.0, 0.1),     # 1% drawdown -> 10% slash
            (0.98, 2.0, 0.2),     # 2% drawdown -> 20% slash
            (0.97, 3.0, 0.3),     # 3% drawdown -> 30% slash
            (0.96, 4.0, 0.4),     # 4% drawdown -> 40% slash
            (0.95, 5.0, 0.5),     # 5% drawdown -> 50% slash
            (0.94, 6.0, 0.6),     # 6% drawdown -> 60% slash
            (0.93, 7.0, 0.7),     # 7% drawdown -> 70% slash
            (0.92, 8.0, 0.8),     # 8% drawdown -> 80% slash
            (0.91, 9.0, 0.9),     # 9% drawdown -> 90% slash
            (0.90, 10.0, 1.0),    # 10% drawdown -> 100% slash (elimination)
        ]

        for drawdown, dd_pct, expected_proportion in test_cases:
            with self.subTest(drawdown_pct=dd_pct):
                # Test through production RPC path
                slash_amount = self.contract_client.compute_slash_amount(self.MINER_1, drawdown=drawdown)
                expected_slash = balance_theta * expected_proportion

                self.assertAlmostEqual(slash_amount, expected_slash, places=1,
                                       msg=f"{dd_pct}% drawdown should slash {expected_proportion*100}% of balance. "
                                           f"Expected {expected_slash}, got {slash_amount}")

    def test_get_all_miner_account_sizes(self):
        """Test getting all miner account sizes at a specific timestamp"""
        current_time = int(time.time() * 1000)

        # Set account sizes for multiple miners (balances already injected in setUp())
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)
        self.contract_client.set_miner_account_size(self.MINER_2, current_time)

        # Get all account sizes
        all_sizes = self.contract_client.get_all_miner_account_sizes(timestamp_ms=current_time + self.DAY_MS)

        # Verify both miners are present
        self.assertIn(self.MINER_1, all_sizes)
        self.assertIn(self.MINER_2, all_sizes)
        self.assertIsNotNone(all_sizes[self.MINER_1])
        self.assertIsNotNone(all_sizes[self.MINER_2])
