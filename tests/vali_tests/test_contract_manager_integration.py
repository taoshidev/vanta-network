# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Integration tests for ValidatorContractManager focusing on deposit/withdrawal operations.

Tests the wallet reference change from vault_wallet to self.wallet and ensures collateral
operations work correctly with the new architecture. Uses running_unit_tests flag to branch
logic on network calls (similar to miner.py / test_miner_integration.py pattern).
"""
import time
from unittest.mock import MagicMock, patch, Mock

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils


class TestContractManagerIntegration(TestBase):
    """
    Integration tests for ValidatorContractManager deposit/withdrawal operations.

    Follows the same pattern as test_miner_integration.py:
    - Uses ServerOrchestrator singleton
    - running_unit_tests flag prevents network calls
    - Mocks external dependencies (CollateralManager)
    - Tests end-to-end flows
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    contract_client = None
    position_client = None
    metagraph_client = None
    perf_ledger_client = None

    # Test constants
    MINER_HOTKEY = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    MINER_COLDKEY = "5CJL9JdYdTYAy6xBjHdHivF7XzKcCE4KGgSXEQM5Lk9GJGcJ"
    DEPOSIT_AMOUNT_THETA = 1000.0  # 1000 theta
    WITHDRAWAL_AMOUNT_THETA = 500.0  # 500 theta

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.contract_client = cls.orchestrator.get_client('contract')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No cleanup needed.

        Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # Clear all test data (includes contract-specific cleanup)
        self.orchestrator.clear_all_test_data()

        # Set default test collateral balance (1000 theta = 1000 * 10^9 rao)
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, 1_000_000_000_000)

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    # ============================================================
    # DEPOSIT TESTS
    # ============================================================

    def test_deposit_request_wallet_reference(self):
        """
        Test that deposit operations work correctly after wallet reference change.
        This verifies the wallet reference change from vault_wallet to self.wallet.

        We verify through client API that:
        1. Account size can be set (requires wallet to fetch balance)
        2. Operations that would use wallet internally succeed
        """
        # Inject test balance
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, 1_000_000_000_000)

        # Set account size - this internally uses the wallet to fetch balance
        timestamp_ms = int(time.time() * 1000)
        success = self.contract_client.set_miner_account_size(self.MINER_HOTKEY, timestamp_ms)

        # Verify success - this proves wallet reference is working
        self.assertTrue(success, "Account size should be set successfully with wallet reference")

        # Get account size to verify it was stored
        account_size = self.contract_client.get_miner_account_size(self.MINER_HOTKEY, timestamp_ms, most_recent=True)
        self.assertIsNotNone(account_size)
        self.assertGreater(account_size, 0)

    def test_deposit_operations_work_with_new_wallet(self):
        """
        Test that deposit-related operations work correctly with new wallet architecture.
        Uses client API to verify internal wallet reference is correctly configured.
        """
        # Test 1: Verify collateral balance retrieval works
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, 2_000_000_000_000)
        balance = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertEqual(balance, 2000.0, "Balance retrieval should work with wallet")

        # Test 2: Verify account size operations work
        success = self.contract_client.set_miner_account_size(self.MINER_HOTKEY)
        self.assertTrue(success, "Account size operations should work with wallet")

        # Test 3: Verify get all account sizes works
        all_sizes = self.contract_client.get_all_miner_account_sizes()
        self.assertIn(self.MINER_HOTKEY, all_sizes, "Should retrieve all account sizes")

    def test_deposit_max_balance_concept(self):
        """
        Test that max balance concept exists and can be queried.
        (Full deposit validation would require mocking CollateralManager which isn't
        feasible in TESTING mode, so we verify the concept through other means)
        """
        # Verify we can set and retrieve balances up to reasonable limits
        test_balance = 5_000_000_000_000  # 5000 theta (well under max)
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, test_balance)

        balance = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertEqual(balance, 5000.0)

        # Set account size should work with valid balance
        success = self.contract_client.set_miner_account_size(self.MINER_HOTKEY)
        self.assertTrue(success, "Account size should work with valid balance")

    def test_error_handling_exists(self):
        """
        Test that error handling mechanisms exist in the contract manager.
        We verify this by checking that invalid operations return proper results.
        """
        # Trying to get balance for non-existent miner should return None
        self.contract_client.clear_test_collateral_balances()
        balance = self.contract_client.get_miner_collateral_balance("non_existent_hotkey")
        self.assertIsNone(balance, "Should return None for non-existent miner")

    # ============================================================
    # WITHDRAWAL TESTS
    # ============================================================

    def test_withdrawal_query_works(self):
        """
        Test that withdrawal query operations work correctly.
        This verifies the wallet reference is working through the query API.
        """
        # Set test balance
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, 1_000_000_000_000)

        # Query withdrawal (doesn't execute, just returns preview)
        # With no positions, drawdown will be 1.0 (no drawdown) so slashed_amount will be 0
        result = self.contract_client.query_withdrawal_request(
            amount=500.0,
            miner_hotkey=self.MINER_HOTKEY
        )

        # Verify result structure
        self.assertTrue(result["successfully_processed"])
        self.assertIn("drawdown", result)
        self.assertIn("slashed_amount", result)
        self.assertIn("withdrawal_amount", result)

        # With no positions/drawdown, drawdown should be 1.0
        self.assertEqual(result["drawdown"], 1.0)

        # Verify withdrawal_amount = amount (no slashing when drawdown is 1.0)
        self.assertEqual(result["withdrawal_amount"], 500.0)

        # Verify new_balance = current_balance - amount
        self.assertEqual(result["new_balance"], 500.0)

    def test_withdrawal_operations_use_wallet_correctly(self):
        """
        Test that withdrawal-related operations work correctly with wallet reference.
        We verify through client API that wallet is correctly configured.
        """
        # Set test balance for withdrawal operations
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, 2_000_000_000_000)

        # Verify we can query withdrawal (this uses wallet internally)
        result = self.contract_client.query_withdrawal_request(
            amount=100.0,
            miner_hotkey=self.MINER_HOTKEY
        )

        # Verify query succeeded - this confirms wallet is working
        self.assertTrue(result["successfully_processed"])
        self.assertGreaterEqual(result["new_balance"], 0)
        self.assertEqual(result["new_balance"], 1900.0)  # 2000 - 100

    def test_withdrawal_calculates_slashing(self):
        """
        Test that withdrawal slashing calculation framework exists and returns correct structure.

        Note: Full drawdown testing requires creating positions + performance ledgers, which
        belongs in position_manager/perf_ledger test suites. These contract tests verify the
        slashing calculation framework works with the default case (no positions = no drawdown).
        """
        # Set test balance
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, 1_000_000_000_000)

        # Query withdrawal
        result = self.contract_client.query_withdrawal_request(
            amount=500.0,
            miner_hotkey=self.MINER_HOTKEY
        )

        # Verify slashing calculation framework exists
        self.assertTrue(result["successfully_processed"])
        self.assertIn("slashed_amount", result)
        self.assertIn("drawdown", result)

        # With no positions (default case), drawdown should be 1.0 (no drawdown)
        self.assertEqual(result["drawdown"], 1.0, "Default drawdown is 1.0 with no positions")

        # Slashed amount should be 0 when there's no drawdown
        self.assertEqual(result["slashed_amount"], 0.0, "No slashing when drawdown = 1.0")

    # ============================================================
    # QUERY WITHDRAWAL TESTS
    # ============================================================

    def test_query_withdrawal_returns_slash_preview(self):
        """
        Test query_withdrawal_request returns slash amount without executing.
        Verifies slash calculation logic in running_unit_tests mode.

        Note: Full drawdown testing requires creating positions + performance ledgers.
        This test verifies the query mechanism works correctly with the default case.
        """
        # Set test collateral balance (1000 theta = 1000 * 10^9 rao)
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, 1_000_000_000_000)

        # Query withdrawal (doesn't execute, just returns preview)
        result = self.contract_client.query_withdrawal_request(
            amount=500.0,
            miner_hotkey=self.MINER_HOTKEY
        )

        # Verify result structure
        self.assertTrue(result["successfully_processed"])
        self.assertIn("drawdown", result)
        self.assertIn("slashed_amount", result)
        self.assertIn("withdrawal_amount", result)
        self.assertIn("new_balance", result)

        # With no positions, drawdown should be 1.0 (no drawdown)
        self.assertEqual(result["drawdown"], 1.0)

        # Verify slashed_amount is 0 with no drawdown
        self.assertEqual(result["slashed_amount"], 0.0)

        # Verify withdrawal_amount = amount - slashed_amount
        expected_withdrawal = result["withdrawal_amount"]
        self.assertEqual(expected_withdrawal, 500.0 - result["slashed_amount"])
        self.assertEqual(expected_withdrawal, 500.0)

        # Verify new_balance = current_balance - amount
        self.assertEqual(result["new_balance"], 1000.0 - 500.0)

    # ============================================================
    # ACCOUNT SIZE TESTS (Integration with Collateral)
    # ============================================================

    def test_set_account_size_uses_test_balance(self):
        """
        Test that set_miner_account_size uses test collateral balance injection.
        Verifies running_unit_tests pattern for avoiding blockchain calls.
        """
        # Inject test balance (2000 theta = 2000 * 10^9 rao)
        test_balance_rao = 2_000_000_000_000  # 2000 theta
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, test_balance_rao)

        # Set account size
        timestamp_ms = int(time.time() * 1000)
        success = self.contract_client.set_miner_account_size(self.MINER_HOTKEY, timestamp_ms)

        # Verify success
        self.assertTrue(success)

        # Get account size and verify it was calculated from test balance
        account_size = self.contract_client.get_miner_account_size(
            self.MINER_HOTKEY,
            timestamp_ms=timestamp_ms + (24 * 60 * 60 * 1000),  # Next day
            most_recent=False
        )

        # Verify account size is not None
        self.assertIsNotNone(account_size)
        # Account size should be based on 2000 theta (but capped at MAX_COLLATERAL_BALANCE_THETA)
        self.assertGreater(account_size, 0)

    def test_collateral_balance_test_injection_pattern(self):
        """
        Test the collateral balance injection pattern (like polygon_data_service.py).
        Verifies running_unit_tests flag prevents blockchain calls.
        """
        # Clear any existing test balances
        self.contract_client.clear_test_collateral_balances()

        # Initially, get_miner_collateral_balance should return None in test mode (no balance set)
        balance = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertIsNone(balance, "Should return None when no test balance is set")

        # Inject test balance (5000 theta = 5000 * 10^9 rao)
        test_balance_rao = 5_000_000_000_000  # 5000 theta
        self.contract_client.set_test_collateral_balance(self.MINER_HOTKEY, test_balance_rao)

        # Now get_miner_collateral_balance should return the injected balance
        balance = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertIsNotNone(balance)
        self.assertEqual(balance, 5000.0)  # Should return in theta (5000)

    def test_collateral_balance_queue_pattern(self):
        """
        Test the collateral balance queue pattern for race condition testing.
        Verifies queue_test_collateral_balance works correctly.
        """
        # Clear existing balances
        self.contract_client.clear_test_collateral_balances()

        # Queue multiple balances for the same miner (FIFO)
        # Note: 1000 theta = 1000 * 10^9 rao = 1_000_000_000_000
        self.contract_client.queue_test_collateral_balance(self.MINER_HOTKEY, 1_000_000_000_000)  # 1000 theta
        self.contract_client.queue_test_collateral_balance(self.MINER_HOTKEY, 2_000_000_000_000)  # 2000 theta
        self.contract_client.queue_test_collateral_balance(self.MINER_HOTKEY, 3_000_000_000_000)  # 3000 theta

        # First call should return first queued value
        balance1 = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertEqual(balance1, 1000.0)

        # Second call should return second queued value
        balance2 = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertEqual(balance2, 2000.0)

        # Third call should return third queued value
        balance3 = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertEqual(balance3, 3000.0)

        # Fourth call should return None (queue exhausted)
        balance4 = self.contract_client.get_miner_collateral_balance(self.MINER_HOTKEY)
        self.assertIsNone(balance4)
