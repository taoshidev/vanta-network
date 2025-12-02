# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Integration tests for ValidatorContractManager using server/client architecture.
Tests end-to-end contract management scenarios with real server infrastructure.
"""
import time
import threading
from unittest.mock import patch, MagicMock

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.validator_contract_manager import CollateralRecord
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

        # Mock the CollateralManager at the class level to avoid actual contract calls
        # This is appropriate since CollateralManager interacts with external blockchain
        cls.mock_collateral_manager_patcher = patch(
            'vali_objects.utils.validator_contract_manager.CollateralManager'
        )
        cls.mock_collateral_manager_class = cls.mock_collateral_manager_patcher.start()
        cls.mock_collateral_manager_instance = MagicMock()
        cls.mock_collateral_manager_class.return_value = cls.mock_collateral_manager_instance

        # Set up mock collateral balances
        cls.mock_balances = {
            cls.MINER_1: 1000000,  # 1M rao
            cls.MINER_2: 500000    # 500K rao
        }
        cls.mock_collateral_manager_instance.balance_of.side_effect = lambda hotkey: cls.mock_balances.get(hotkey, 0)

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: Clean up mocks.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        cls.mock_collateral_manager_patcher.stop()

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # Clear all standard test data (positions, perf_ledger, eliminations, etc.)
        self.orchestrator.clear_all_test_data()

        # Contract-specific clears (not in orchestrator.clear_all_test_data())
        self.contract_client.sync_miner_account_sizes_data({})  # Clear account sizes
        self.contract_client.re_init_account_sizes()  # Reload from disk after clearing

        # Reset mock side_effect (in case a test overrode it)
        self.mock_collateral_manager_instance.balance_of.side_effect = lambda hotkey: self.mock_balances.get(hotkey, 0)

        # Reset mock balances to defaults
        self.mock_balances[self.MINER_1] = 1000000  # 1M rao
        self.mock_balances[self.MINER_2] = 500000   # 500K rao

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        # Clear all standard test data
        self.orchestrator.clear_all_test_data()

        # Contract-specific clears
        self.contract_client.sync_miner_account_sizes_data({})
        self.contract_client.re_init_account_sizes()  # Reload from disk after clearing

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
        self.mock_balances[self.MINER_1] = 1000000  # 1M rao
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)

        # Verify retrieval - should return the calculated account size
        account_size = self.contract_client.get_miner_account_size(self.MINER_1, day_after_current_time)
        self.assertIsNotNone(account_size)

        # Set for second miner
        self.mock_balances[self.MINER_2] = 500000  # 500K rao
        self.contract_client.set_miner_account_size(self.MINER_2, current_time)
        account_size_2 = self.contract_client.get_miner_account_size(self.MINER_2, day_after_current_time)
        self.assertIsNotNone(account_size_2)

    def test_account_size_persistence(self):
        """Test that account sizes are saved to and loaded from disk"""
        current_time = int(time.time() * 1000)
        day_after_current_time = self.DAY_MS + current_time

        # Set account size
        self.mock_balances[self.MINER_1] = 1000000  # 1M rao
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

        # Mock collateral balance for consistent account size calculation
        self.mock_collateral_manager_instance.balance_of.side_effect = [
            1_000_000,  # First call
            2_000_000,  # Second call
            3_000_000,  # Third call
        ]

        # Add multiple records with different timestamps
        self.contract_client.set_miner_account_size(self.MINER_1, base_time)
        self.contract_client.set_miner_account_size(self.MINER_1, base_time + 1000)
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

        # Mock collateral balance for consistent account size calculation
        self.mock_balances[self.MINER_1] = 1000000  # 1M rao

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

        # Set account sizes
        self.mock_balances[self.MINER_1] = 1000000  # 1M rao
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)

        self.mock_balances[self.MINER_2] = 500000  # 500K rao
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
        # Mock different balances
        self.mock_balances[self.MINER_1] = 1500000  # 1.5M rao

        # Get balance
        balance = self.contract_client.get_miner_collateral_balance(self.MINER_1)
        self.assertIsNotNone(balance)

        # Verify the mock was called
        self.mock_collateral_manager_instance.balance_of.assert_called_with(self.MINER_1)

    def test_compute_slash_amount_formula_accuracy(self):
        """Test the exact formula for slash calculation across range of drawdowns"""
        # Set up real collateral balance via RPC (not mocking!)
        balance_theta = 1000.0
        balance_rao = int(balance_theta * 10 ** 9)
        self.mock_balances[self.MINER_1] = balance_rao

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

        # Set account sizes for multiple miners
        self.mock_balances[self.MINER_1] = 1000000
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)

        self.mock_balances[self.MINER_2] = 500000
        self.contract_client.set_miner_account_size(self.MINER_2, current_time)

        # Get all account sizes
        all_sizes = self.contract_client.get_all_miner_account_sizes(timestamp_ms=current_time + self.DAY_MS)

        # Verify both miners are present
        self.assertIn(self.MINER_1, all_sizes)
        self.assertIn(self.MINER_2, all_sizes)
        self.assertIsNotNone(all_sizes[self.MINER_1])
        self.assertIsNotNone(all_sizes[self.MINER_2])

    # ==================== RACE CONDITION TESTS ====================
    # These tests demonstrate concurrency bugs in ValidatorContractManager.
    # They model actual access patterns from the production codebase.
    # EXPECTED TO FAIL until proper locking is implemented!

    def test_race_concurrent_list_append_to_same_miner(self):
        """
        Race Condition Test: Concurrent appends to same miner's record list

        Scenario: Multiple deposits/withdrawals happening simultaneously for the same miner.
        This simulates the real-world case where a miner makes multiple collateral changes
        in quick succession (e.g., multiple deposits, or deposit + withdrawal).

        Bug: set_miner_account_size() does not acquire lock before appending to list.
        Line 853: self.miner_account_sizes[hotkey] = self.miner_account_sizes[hotkey] + [record]

        Expected Failure: Lost updates - some records will be overwritten instead of appended.
        With 100 concurrent appends, we should get 100 records, but will likely get fewer.
        """
        current_time = int(time.time() * 1000)
        num_concurrent_updates = 100
        results = []

        # Create unique balances for each update to ensure unique records
        balance_sequence = list(range(1_000_000, 1_000_000 + num_concurrent_updates * 1000, 1000))
        balance_index = [0]  # Use list to allow modification in nested function
        balance_lock = threading.Lock()

        def get_next_balance(hotkey):
            """Return next unique balance for testing (thread-safe)"""
            with balance_lock:
                idx = balance_index[0]
                balance_index[0] += 1
                return balance_sequence[idx]

        # Override balance_of to return unique values
        self.mock_collateral_manager_instance.balance_of.side_effect = get_next_balance

        def concurrent_set_account_size(thread_id):
            """Each thread sets account size for same miner"""
            try:
                timestamp = current_time + thread_id  # Unique timestamps
                result = self.contract_client.set_miner_account_size(self.MINER_1, timestamp)
                results.append((thread_id, result))
            except Exception as e:
                results.append((thread_id, f"ERROR: {e}"))

        # Launch concurrent threads
        threads = []
        for i in range(num_concurrent_updates):
            thread = threading.Thread(target=concurrent_set_account_size, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()
        actual_record_count = len(account_sizes_dict.get(self.MINER_1, []))

        # EXPECTED FAILURE: Due to race condition, we lose updates
        # We launched 100 updates, but without proper locking, some will be lost
        self.assertEqual(
            actual_record_count,
            num_concurrent_updates,
            f"RACE CONDITION DETECTED: Expected {num_concurrent_updates} records but got {actual_record_count}. "
            f"Lost {num_concurrent_updates - actual_record_count} updates due to unlocked list append!"
        )

    def test_race_concurrent_append_different_miners(self):
        """
        Race Condition Test: Concurrent appends to different miners' lists

        Scenario: Multiple miners depositing collateral simultaneously.
        This simulates the real-world case of concurrent deposit processing.

        Bug: set_miner_account_size() + _save_miner_account_sizes_to_disk() both unlocked.
        Multiple threads can write to disk simultaneously, causing corruption or lost writes.

        Expected Failure: Missing records or file corruption due to concurrent disk writes.
        """
        current_time = int(time.time() * 1000)
        num_miners = 50

        # Generate unique miner hotkeys
        test_miners = [f"5Miner{i:0>44}" for i in range(num_miners)]

        # Set up unique balances for each miner
        for i, miner in enumerate(test_miners):
            self.mock_balances[miner] = (i + 1) * 100000  # 100K, 200K, 300K, etc.

        results = []

        def concurrent_set_for_different_miners(miner_hotkey):
            """Each thread sets account size for a different miner"""
            try:
                result = self.contract_client.set_miner_account_size(miner_hotkey, current_time)
                results.append((miner_hotkey, result))
            except Exception as e:
                results.append((miner_hotkey, f"ERROR: {e}"))

        # Launch concurrent threads (one per miner)
        threads = []
        for miner in test_miners:
            thread = threading.Thread(target=concurrent_set_for_different_miners, args=(miner,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all miners have records
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()
        missing_miners = [m for m in test_miners if m not in account_sizes_dict]

        # EXPECTED FAILURE: Due to unlocked disk writes, some miners may be missing
        self.assertEqual(
            len(missing_miners),
            0,
            f"RACE CONDITION DETECTED: {len(missing_miners)} miners missing from disk due to concurrent writes! "
            f"Missing miners: {missing_miners[:5]}..."  # Show first 5
        )

    def test_race_iterator_invalidation_during_get_all(self):
        """
        Race Condition Test: Iterator invalidation during get_all_miner_account_sizes

        Scenario: One thread retrieves all account sizes while another syncs (clears + repopulates).
        This simulates the real-world case of validator checkpoint restoration during normal operation.

        Bug: get_all_miner_account_sizes() iterates dict without lock (line 927).
        sync_miner_account_sizes_data() clears dict (line 316).

        Expected Failure: RuntimeError: dictionary changed size during iteration, or missing data.
        """
        current_time = int(time.time() * 1000)

        # Set up initial data with many miners
        num_initial_miners = 100
        test_miners = [f"5Init{i:0>45}" for i in range(num_initial_miners)]
        for i, miner in enumerate(test_miners):
            self.mock_balances[miner] = (i + 1) * 50000
            self.contract_client.set_miner_account_size(miner, current_time)

        # Prepare sync data (different set of miners)
        sync_data = {}
        for i in range(50):
            hotkey = f"5Sync{i:0>45}"
            sync_data[hotkey] = [{
                "account_size": 10000.0 * (i + 1),
                "account_size_theta": 100.0 * (i + 1),
                "update_time_ms": current_time,
                "valid_date_timestamp": CollateralRecord.valid_from_ms(current_time)
            }]

        errors = []
        get_all_results = []

        def continuous_get_all():
            """Continuously call get_all_miner_account_sizes"""
            for _ in range(20):  # Try 20 times
                try:
                    result = self.contract_client.get_all_miner_account_sizes()
                    get_all_results.append(len(result))
                except RuntimeError as e:
                    errors.append(f"RuntimeError: {e}")
                except Exception as e:
                    errors.append(f"Unexpected error: {e}")
                time.sleep(0.001)  # Small delay

        def sync_data_repeatedly():
            """Repeatedly sync data (clear + repopulate)"""
            for _ in range(10):  # Sync 10 times
                try:
                    self.contract_client.sync_miner_account_sizes_data(sync_data)
                except Exception as e:
                    errors.append(f"Sync error: {e}")
                time.sleep(0.002)  # Small delay

        # Run both operations concurrently
        thread1 = threading.Thread(target=continuous_get_all)
        thread2 = threading.Thread(target=sync_data_repeatedly)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # EXPECTED FAILURE: Should see RuntimeError or inconsistent results
        self.assertEqual(
            len(errors),
            0,
            f"RACE CONDITION DETECTED: Iterator invalidation occurred! Errors: {errors[:3]}"
        )

    def test_race_read_during_sync_clear(self):
        """
        Race Condition Test: Reading account size while sync is clearing dict

        Scenario: One thread reads miner account size while another is syncing (clear in progress).
        This simulates checkpoint restoration happening during normal query operations.

        Bug: sync_miner_account_sizes_data() holds lock during clear (line 316),
        but get_miner_account_size() doesn't acquire lock (line 892).

        Expected Failure: get_miner_account_size returns None temporarily during sync,
        even though miner should have data before and after.
        """
        current_time = int(time.time() * 1000)

        # Set initial account size
        self.mock_balances[self.MINER_1] = 1_000_000
        self.contract_client.set_miner_account_size(self.MINER_1, current_time)

        # Prepare sync data (same miner, same balance)
        sync_data = {
            self.MINER_1: [{
                "account_size": 1000.0,
                "account_size_theta": 1.0,
                "update_time_ms": current_time,
                "valid_date_timestamp": CollateralRecord.valid_from_ms(current_time)
            }]
        }

        none_results = []  # Track when get returns None

        def continuous_read():
            """Continuously read account size"""
            for _ in range(100):
                result = self.contract_client.get_miner_account_size(
                    self.MINER_1,
                    timestamp_ms=current_time + self.DAY_MS,
                    most_recent=True
                )
                if result is None:
                    none_results.append("Got None during read")
                time.sleep(0.0001)  # Very small delay

        def sync_repeatedly():
            """Repeatedly sync data"""
            for _ in range(20):
                self.contract_client.sync_miner_account_sizes_data(sync_data)
                time.sleep(0.001)

        # Run concurrently
        thread1 = threading.Thread(target=continuous_read)
        thread2 = threading.Thread(target=sync_repeatedly)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # EXPECTED FAILURE: Should see None results during clear phase
        self.assertEqual(
            len(none_results),
            0,
            f"RACE CONDITION DETECTED: Got None {len(none_results)} times during sync! "
            f"Data temporarily disappeared during clear+repopulate."
        )

    def test_race_concurrent_disk_writes(self):
        """
        Race Condition Test: Concurrent disk writes without lock

        Scenario: Multiple account size updates triggering simultaneous disk saves.
        This simulates heavy deposit/withdrawal activity.

        Bug: _save_miner_account_sizes_to_disk() doesn't acquire _disk_lock (even though it exists!).
        Multiple threads call ValiBkpUtils.write_file() on same file simultaneously.

        Expected Failure: File corruption, last-write-wins, or incomplete data on disk.
        """
        current_time = int(time.time() * 1000)
        num_concurrent_writes = 50

        # Set up unique miners and balances
        test_miners = [f"5Disk{i:0>45}" for i in range(num_concurrent_writes)]
        for i, miner in enumerate(test_miners):
            self.mock_balances[miner] = (i + 1) * 100000

        def concurrent_write(miner_hotkey):
            """Each thread triggers a disk write"""
            self.contract_client.set_miner_account_size(miner_hotkey, current_time)

        # Launch all threads at once
        threads = []
        for miner in test_miners:
            thread = threading.Thread(target=concurrent_write, args=(miner,))
            threads.append(thread)

        # Start all threads simultaneously
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Reload from disk to verify integrity
        self.contract_client.re_init_account_sizes()
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()

        # Check if all miners are present after disk reload
        missing_from_disk = [m for m in test_miners if m not in account_sizes_dict]

        # EXPECTED FAILURE: Some miners missing due to disk write races
        self.assertEqual(
            len(missing_from_disk),
            0,
            f"RACE CONDITION DETECTED: {len(missing_from_disk)} miners lost due to concurrent disk writes! "
            f"File corruption or last-write-wins. Missing: {missing_from_disk[:5]}"
        )

    def test_race_chronological_order_violation(self):
        """
        Race Condition Test: Chronological order maintained under concurrent updates

        Scenario: Multiple concurrent updates with varying balances.
        With proper locking and timestamp generation inside the lock, records
        should maintain chronological order based on when they acquire the lock.

        NOTE: This test does NOT pass explicit timestamps, allowing timestamps
        to be generated inside the lock, which ensures chronological ordering.
        """
        num_updates = 50

        # Use incrementing balances (some may be duplicates due to RPC retry logic, that's OK)
        balance_counter = [0]
        balance_lock = threading.Lock()

        def get_incrementing_balance(hotkey):
            """Return incrementing balance - simulates changing collateral"""
            with balance_lock:
                balance_counter[0] += 1
                # Return balances that vary enough to avoid duplicate detection
                return 1_000_000 + (balance_counter[0] % 10) * 100000

        # Override balance_of to return incrementing balances
        self.mock_collateral_manager_instance.balance_of.side_effect = get_incrementing_balance

        def concurrent_update(thread_id):
            """Update WITHOUT explicit timestamp - let it be generated inside the lock"""
            # NO timestamp parameter = generated inside lock = chronological order guaranteed
            self.contract_client.set_miner_account_size(self.MINER_1)

        # Launch threads concurrently
        threads = []
        for i in range(num_updates):
            thread = threading.Thread(target=concurrent_update, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify results
        account_sizes_dict = self.contract_client.miner_account_sizes_dict()
        records = account_sizes_dict.get(self.MINER_1, [])

        # We should have at least some records (not all may be saved due to duplicate detection)
        self.assertGreater(
            len(records),
            0,
            "Should have at least one record from 50 concurrent updates"
        )

        # Critical check: Records MUST be in chronological order
        # Since timestamps are generated INSIDE the lock, they're guaranteed to be chronological
        if len(records) > 1:
            is_sorted = all(
                records[i]['update_time_ms'] <= records[i+1]['update_time_ms']
                for i in range(len(records) - 1)
            )

            self.assertTrue(
                is_sorted,
                f"RACE CONDITION DETECTED: Records are not in chronological order! "
                f"This breaks get_miner_account_size logic which assumes chronological ordering. "
                f"Records (timestamps): {[r['update_time_ms'] for r in records]}"
            )