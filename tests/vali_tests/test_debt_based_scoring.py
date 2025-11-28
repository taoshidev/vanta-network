"""
Unit tests for debt-based scoring algorithm with emission projection
"""

import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone
from vali_objects.vali_dataclasses.debt_ledger import DebtLedger, DebtCheckpoint
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig


class TestDebtBasedScoring(unittest.TestCase):
    """Test debt-based scoring functionality"""

    def setUp(self):
        """Set up mock dependencies"""
        # Mock metagraph
        self.mock_metagraph = Mock()
        # metagraph.emission is in TAO per tempo (360 blocks)
        # To get 10 TAO/block total, we need 10 * 360 = 3600 TAO per tempo
        self.mock_metagraph.emission = [360] * 10  # 10 miners, 360 TAO per tempo each = 1 TAO/block each
        # Create hotkeys list for burn address testing
        self.mock_metagraph.hotkeys = [f"hotkey_{i}" for i in range(256)]
        self.mock_metagraph.hotkeys[229] = "burn_address_mainnet"
        self.mock_metagraph.hotkeys[5] = "burn_address_testnet"

        # Mock substrate reserves (IPC manager.Value objects)
        # Using Mock objects that have .value attribute
        # Set reserves to achieve 2.0 ALPHA/TAO conversion rate for testing
        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9  # 1M TAO in RAO
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9  # 2M ALPHA in RAO (2.0 ALPHA per TAO)
        self.mock_metagraph.tao_reserve_rao = mock_tao_reserve
        self.mock_metagraph.alpha_reserve_rao = mock_alpha_reserve

        # Mock TAO/USD price (set by MetagraphUpdater via live_price_fetcher)
        self.mock_metagraph.tao_to_usd_rate = 500.0  # $500/TAO

        # Use static dust value from config
        self.expected_dynamic_dust = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT

        # Mock challengeperiod_manager
        self.mock_challengeperiod_manager = Mock()
        # Default to MAINCOMP for all miners
        def mock_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            mock_bucket.value = MinerBucket.MAINCOMP.value
            return mock_bucket
        self.mock_challengeperiod_manager.get_miner_bucket = Mock(side_effect=mock_get_miner_bucket)

        # Mock contract_manager (for collateral-aware weight assignment)
        self.mock_contract_manager = Mock()
        # Default: return 0 collateral (USD) for all miners (can be overridden in specific tests)
        def mock_get_miner_account_size(hotkey, most_recent=False):
            return 0.0
        self.mock_contract_manager.get_miner_account_size = Mock(side_effect=mock_get_miner_account_size)

    def test_empty_ledgers(self):
        """Test with no ledgers returns burn address with weight 1.0"""
        result = DebtBasedScoring.compute_results(
            {},
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            is_testnet=False
        )
        # With no miners, burn address gets all weight
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("burn_address_mainnet", 1.0))

    def test_single_miner(self):
        """Test with single miner gets dust weight, burn address gets remainder"""
        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])
        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            is_testnet=False
        )
        # Single miner with no performance gets dust weight
        # Burn address gets the rest
        self.assertEqual(len(result), 2)
        weights_dict = dict(result)
        self.assertIn("test_hotkey", weights_dict)
        self.assertIn("burn_address_mainnet", weights_dict)
        # Verify sum is 1.0
        total_weight = sum(w for _, w in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

    def test_before_activation_date(self):
        """Test that only dust weights + burn address before December 2025"""
        # Use November 2025 as current time (previous month is October 2025, before December)
        current_time = datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create ledgers with different statuses
        prev_checkpoint = datetime(2025, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_checkpoint_ms = int(prev_checkpoint.timestamp() * 1000)

        ledger1 = DebtLedger(hotkey="hotkey1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger2 = DebtLedger(hotkey="hotkey2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))

        ledgers = {"hotkey1": ledger1, "hotkey2": ledger2}

        # Create custom mock challengeperiod_manager for this test
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            if hotkey == "hotkey1":
                mock_bucket.value = MinerBucket.MAINCOMP.value
            elif hotkey == "hotkey2":
                mock_bucket.value = MinerBucket.CHALLENGE.value
            else:
                mock_bucket.value = MinerBucket.UNKNOWN.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False
        )

        # Should have 3 entries: 2 miners + burn address
        self.assertEqual(len(result), 3)

        # Verify dust weights based on status
        weights_dict = dict(result)
        dust = self.expected_dynamic_dust
        self.assertAlmostEqual(weights_dict["hotkey1"], 3 * dust)  # MAINCOMP = 3x dust
        self.assertAlmostEqual(weights_dict["hotkey2"], 1 * dust)  # CHALLENGE = 1x dust

        # Verify burn address gets excess (sum should be 1.0)
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Verify burn address is present
        burn_hotkey = "burn_address_mainnet"
        self.assertIn(burn_hotkey, weights_dict)

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0"""
        # Use January 2026 as current time (previous month is December 2025, after activation)
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create ledgers with some performance data
        # Previous month (December 2025) data
        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Current month (January 2026) data
        current_month_checkpoint = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        current_month_checkpoint_ms = int(current_month_checkpoint.timestamp() * 1000)

        # Miner 1: Good performance, needs payout
        ledger1 = DebtLedger(hotkey="hotkey1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=5000.0,
            unrealized_pnl=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=100.0,  # Received 100 ALPHA so far
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: Better performance, needs more payout
        ledger2 = DebtLedger(hotkey="hotkey2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=10000.0,
            unrealized_pnl=-2000.0,  # net_pnl = 8000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=200.0,  # Received 200 ALPHA so far
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"hotkey1": ledger1, "hotkey2": ledger2}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False
        )

        # Check that weights sum to 1.0
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Check that miner2 has higher weight (better performance)
        weights_dict = dict(result)
        self.assertGreater(weights_dict["hotkey2"], weights_dict["hotkey1"])

    def test_minimum_weights_by_status(self):
        """Test that minimum weights are enforced based on challenge period status when sum < 1.0"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create miners with different statuses and ZERO/NEGATIVE performance
        # This ensures remaining_payout = 0, so minimum weights dominate
        # Sum of weights will be 1*dust + 2*dust + 3*dust = 6*dust << 1.0
        ledger_challenge = DebtLedger(hotkey="challenge_miner", checkpoints=[])
        ledger_challenge.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative PnL -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))

        ledger_probation = DebtLedger(hotkey="probation_miner", checkpoints=[])
        ledger_probation.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative PnL -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.PROBATION.value
        ))

        ledger_maincomp = DebtLedger(hotkey="maincomp_miner", checkpoints=[])
        ledger_maincomp.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative PnL -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {
            "challenge_miner": ledger_challenge,
            "probation_miner": ledger_probation,
            "maincomp_miner": ledger_maincomp
        }

        # Create custom mock challengeperiod_manager for this test
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            if hotkey == "challenge_miner":
                mock_bucket.value = MinerBucket.CHALLENGE.value
            elif hotkey == "probation_miner":
                mock_bucket.value = MinerBucket.PROBATION.value
            elif hotkey == "maincomp_miner":
                mock_bucket.value = MinerBucket.MAINCOMP.value
            else:
                mock_bucket.value = MinerBucket.UNKNOWN.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False,
            verbose=True
        )

        # Should have 4 entries: 3 miners + burn address (since sum < 1.0)
        self.assertEqual(len(result), 4)

        weights_dict = dict(result)

        # Verify minimum weights are enforced (guaranteed when sum < 1.0)
        weight_challenge = weights_dict.get("challenge_miner", 0)
        weight_probation = weights_dict.get("probation_miner", 0)
        weight_maincomp = weights_dict.get("maincomp_miner", 0)

        # Check that weights are exactly the minimum (since remaining_payout = 0)
        self.assertAlmostEqual(weight_challenge, dust, places=10)
        self.assertAlmostEqual(weight_probation, 2 * dust, places=10)
        self.assertAlmostEqual(weight_maincomp, 3 * dust, places=10)

        # Check ratio (probation should be exactly 2x challenge, maincomp exactly 3x challenge)
        self.assertAlmostEqual(weight_probation / weight_challenge, 2.0, places=5)
        self.assertAlmostEqual(weight_maincomp / weight_challenge, 3.0, places=5)

        # Burn address should get most of the weight (1.0 - 6*dust)
        burn_hotkey = "burn_address_mainnet"
        self.assertIn(burn_hotkey, weights_dict)
        self.assertGreater(weights_dict[burn_hotkey], 0.99)  # Should be ~0.9999+

    def test_burn_address_mainnet(self):
        """Test burn address receives excess weight on mainnet when sum < 1.0"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_checkpoint_ms = int(prev_checkpoint.timestamp() * 1000)

        # Create TWO miners with minimal weight (need 2+ to avoid single-miner bypass)
        # Very small performance will cause sum < 1.0
        ledger1 = DebtLedger(hotkey="test_hotkey_1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            realized_pnl=0.0001,
            unrealized_pnl=0.0,  # net_pnl = 0.0001 (tiny)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger2 = DebtLedger(hotkey="test_hotkey_2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            realized_pnl=0.00005,
            unrealized_pnl=0.0,  # net_pnl = 0.00005 (tiny)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        result = DebtBasedScoring.compute_results(
            {"test_hotkey_1": ledger1, "test_hotkey_2": ledger2},
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False
        )

        # Should have 3 entries: 2 miners + burn address
        self.assertEqual(len(result), 3)

        weights_dict = dict(result)

        # Burn address should be mainnet (uid 229)
        burn_hotkey = "burn_address_mainnet"
        self.assertIn(burn_hotkey, weights_dict)

        # Total should sum to 1.0
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Burn address should have non-zero weight (excess)
        self.assertGreater(weights_dict[burn_hotkey], 0.0)

    def test_burn_address_testnet(self):
        """Test burn address receives excess weight on testnet with correct UID when sum < 1.0"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        prev_checkpoint_ms = int(prev_checkpoint.timestamp() * 1000)

        # Create TWO miners with minimal weight (need 2+ to avoid single-miner bypass)
        # Very small performance will cause sum < 1.0
        ledger1 = DebtLedger(hotkey="test_hotkey_1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            realized_pnl=0.0001,
            unrealized_pnl=0.0,  # net_pnl = 0.0001 (tiny)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger2 = DebtLedger(hotkey="test_hotkey_2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_checkpoint_ms,
            realized_pnl=0.00005,
            unrealized_pnl=0.0,  # net_pnl = 0.00005 (tiny)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        result = DebtBasedScoring.compute_results(
            {"test_hotkey_1": ledger1, "test_hotkey_2": ledger2},
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=True  # TESTNET
        )

        weights_dict = dict(result)

        # Burn address should be testnet (uid 5)
        burn_hotkey = "hotkey_220"
        self.assertIn(burn_hotkey, weights_dict)

        # Total should sum to 1.0
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Burn address should have non-zero weight (excess)
        self.assertGreater(weights_dict[burn_hotkey], 0.0)

    def test_negative_performance_gets_minimum_weight(self):
        """Test that miners with negative performance get minimum dust weight when sum < 1.0"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Miner with negative performance (gets minimum dust weight)
        ledger_negative = DebtLedger(hotkey="negative_miner", checkpoints=[])
        ledger_negative.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=1000.0,
            unrealized_pnl=-5000.0,  # net_pnl = -4000 (negative) -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner with small positive performance (keep sum < 1.0)
        ledger_positive = DebtLedger(hotkey="positive_miner", checkpoints=[])
        ledger_positive.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.001,
            unrealized_pnl=-0.0005,  # net_pnl = 0.0005 (very small positive)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"negative_miner": ledger_negative, "positive_miner": ledger_positive}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False,
            verbose=True
        )

        # Should have 3 entries: 2 miners + burn address (sum < 1.0)
        self.assertEqual(len(result), 3)

        weights_dict = dict(result)

        # Negative miner should get minimum dust weight (3x for MAINCOMP)
        # Positive miner should get higher weight based on remaining payout
        self.assertGreater(weights_dict["positive_miner"], weights_dict["negative_miner"])

        # Negative miner gets exactly minimum (since remaining_payout = 0)
        self.assertAlmostEqual(weights_dict["negative_miner"], 3 * dust, places=10)

        # Positive miner gets max(remaining_payout, 3*dust) = max(0.0005, 3*dust) = 0.0005
        self.assertAlmostEqual(weights_dict["positive_miner"], 0.0005, places=10)

        # Burn address gets the rest (1.0 - 0.0005 - 3*dust)
        burn_hotkey = "burn_address_mainnet"
        self.assertIn(burn_hotkey, weights_dict)
        expected_burn = 1.0 - 0.0005 - (3 * dust)
        self.assertAlmostEqual(weights_dict[burn_hotkey], expected_burn, places=5)

    def test_penalty_reduces_needed_payout(self):
        """Test that penalties reduce the needed payout"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Miner 1: No penalty
        ledger1 = DebtLedger(hotkey="no_penalty", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=5000.0,
            unrealized_pnl=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,  # No penalty
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: 50% penalty (same PnL but lower needed payout)
        ledger2 = DebtLedger(hotkey="with_penalty", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=5000.0,
            unrealized_pnl=-1000.0,  # net_pnl = 4000
            total_penalty=0.5,  # 50% penalty
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"no_penalty": ledger1, "with_penalty": ledger2}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False
        )

        # Miner with no penalty should get higher weight
        weights_dict = dict(result)
        self.assertGreater(weights_dict["no_penalty"], weights_dict["with_penalty"])

        # Ratio should be approximately 2:1 (4000 vs 2000 needed payout)
        ratio = weights_dict["no_penalty"] / weights_dict["with_penalty"]
        self.assertAlmostEqual(ratio, 2.0, places=1)

    def test_emission_projection_calculation(self):
        """Test that emission projection is calculated correctly"""
        # Use mocked subtensor with known emission rate
        days_until_target = 10

        projected_alpha = DebtBasedScoring._estimate_alpha_emissions_until_target(
            metagraph=self.mock_metagraph,
            days_until_target=days_until_target,
            verbose=True
        )

        # Expected calculation:
        # - 10 miners * 1 TAO/block = 10 TAO/block
        # - 7200 blocks/day * 10 days = 72000 blocks
        # - 10 TAO/block * 72000 blocks = 720,000 TAO
        # - 720,000 TAO / 0.5 (ALPHA to TAO rate) = 1,440,000 ALPHA

        expected_alpha = 10 * 7200 * 10 / 0.5  # 1,440,000
        self.assertAlmostEqual(projected_alpha, expected_alpha, places=0)

    def test_aggressive_payout_strategy(self):
        """Test that aggressive payout strategy is applied correctly"""
        # Test day 1 - should use 4-day buffer (aggressive)
        current_time_day1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms_day1 = int(current_time_day1.timestamp() * 1000)

        # Create simple ledger with remaining payout
        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=10000.0,
            unrealized_pnl=-2000.0,  # net_pnl = 8000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Run compute_results and check projection uses 4-day window
        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms_day1,
                        is_testnet=False,
            verbose=True
        )

        # Verify weight is assigned (single miner gets 1.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("test_hotkey", 1.0))

        # Test day 23 - should use 3-day buffer (actual remaining is 3)
        current_time_day23 = datetime(2026, 1, 23, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms_day23 = int(current_time_day23.timestamp() * 1000)

        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms_day23,
                        is_testnet=False,
            verbose=True
        )

        # Should still return weight
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("test_hotkey", 1.0))

    def test_only_earning_periods_counted(self):
        """Test that only MAINCOMP/PROBATION checkpoints count for earnings"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create checkpoints in previous month with different statuses
        challenge_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        challenge_checkpoint_ms = int(challenge_checkpoint.timestamp() * 1000)

        maincomp_checkpoint = datetime(2025, 12, 30, 12, 0, 0, tzinfo=timezone.utc)
        maincomp_checkpoint_ms = int(maincomp_checkpoint.timestamp() * 1000)

        ledger = DebtLedger(hotkey="test_hotkey", checkpoints=[])

        # CHALLENGE checkpoint (should NOT count)
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=challenge_checkpoint_ms,
            realized_pnl=5000.0,
            unrealized_pnl=-1000.0,  # net_pnl = 4000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))

        # MAINCOMP checkpoint (SHOULD count)
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=maincomp_checkpoint_ms,
            realized_pnl=10000.0,  # Cumulative
            unrealized_pnl=-2000.0,  # Cumulative, net_pnl = 8000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        result = DebtBasedScoring.compute_results(
            {"test_hotkey": ledger},
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False,
            verbose=True
        )

        # Should only use MAINCOMP checkpoint for earnings calculation
        # (net_pnl = 8000, not 4000 from CHALLENGE period)
        self.assertEqual(len(result), 1)
        # With only one miner, weight should be 1.0
        self.assertEqual(result[0][1], 1.0)

    def test_iterative_payouts_approach_target_by_day_25(self):
        """Test that iterative weight setting causes payouts to approach required payout by day 25"""
        # Setup: 3 miners with different needed payouts from previous month
        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Miner 1: Needs $50000 USD payout (net_pnl in USD)
        ledger1 = DebtLedger(hotkey="miner1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=50000.0,
            unrealized_pnl=0.0,  # net_pnl = $50000 USD
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: Needs $100000 USD payout (2x miner1)
        ledger2 = DebtLedger(hotkey="miner2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=100000.0,
            unrealized_pnl=0.0,  # net_pnl = $100000 USD
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 3: Needs $75000 USD payout (1.5x miner1)
        ledger3 = DebtLedger(hotkey="miner3", checkpoints=[])
        ledger3.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=75000.0,
            unrealized_pnl=0.0,  # net_pnl = $75000 USD
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {
            "miner1": ledger1,
            "miner2": ledger2,
            "miner3": ledger3
        }

        # Total needed payout: $225,000 USD
        # Emissions in ALPHA, converted to USD via: ALPHA * 250 = USD
        # Aggressive 4-day projection: 144K ALPHA/day = $36M USD/day (enough to cover needed payout)
        # Available emissions over 25 days: 144K ALPHA/day * 25 = 3.6M ALPHA = $900M USD
        total_needed_payout = 225000.0  # USD

        # Simulate emissions per day (based on mocked emission rate)
        # metagraph.emission = [360] * 10 = 3600 TAO per tempo for subnet
        # 3600 / 360 = 10 TAO per block
        # 10 TAO/block * 7200 blocks/day = 72000 TAO/day
        # 72000 TAO / 0.5 (alpha_to_tao_rate) = 144000 ALPHA/day
        alpha_per_day = 144000.0

        # Track cumulative payouts for each miner
        cumulative_payouts = {
            "miner1": 0.0,
            "miner2": 0.0,
            "miner3": 0.0
        }

        # Track weights over time for verification
        weights_over_time = []

        # Simulate days 1-25 of January 2026
        for day in range(1, 26):
            current_time = datetime(2026, 1, day, 12, 0, 0, tzinfo=timezone.utc)
            current_time_ms = int(current_time.timestamp() * 1000)

            # Compute weights for this day
            result = DebtBasedScoring.compute_results(
                ledgers,
                self.mock_metagraph,
                self.mock_challengeperiod_manager,
                self.mock_contract_manager,
                                current_time_ms=current_time_ms,
                                is_testnet=False,
                verbose=False
            )

            weights_dict = dict(result)

            # Record weights
            weights_over_time.append({
                "day": day,
                "miner1": weights_dict.get("miner1", 0.0),
                "miner2": weights_dict.get("miner2", 0.0),
                "miner3": weights_dict.get("miner3", 0.0)
            })

            # Simulate daily emissions distributed according to weights
            for hotkey in ["miner1", "miner2", "miner3"]:
                daily_payout = alpha_per_day * weights_dict.get(hotkey, 0.0)
                cumulative_payouts[hotkey] += daily_payout

                # Add checkpoint to ledger for cumulative emissions
                # Convert ALPHA to USD using mocked conversion rates:
                # ALPHA → TAO: 0.5 (1M TAO / 2M ALPHA)
                # TAO → USD: 500.0 (fallback)
                # Total: ALPHA → USD = ALPHA * 250
                alpha_to_usd_rate = 250.0
                current_month_checkpoint_ms = int(datetime(2026, 1, day + 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
                ledgers[hotkey].checkpoints.append(DebtCheckpoint(
                    timestamp_ms=current_month_checkpoint_ms,
                    chunk_emissions_alpha=cumulative_payouts[hotkey],
                    chunk_emissions_usd=cumulative_payouts[hotkey] * alpha_to_usd_rate,
                    challenge_period_status=MinerBucket.MAINCOMP.value
                ))

        # Assertions

        # 1. Verify proportional distribution (2:1.5:1 ratio) - THIS IS CRITICAL
        # The algorithm should maintain proportional distribution regardless of exact amounts
        ratio_2_to_1 = cumulative_payouts["miner2"] / cumulative_payouts["miner1"]
        ratio_3_to_1 = cumulative_payouts["miner3"] / cumulative_payouts["miner1"]
        self.assertAlmostEqual(ratio_2_to_1, 2.0, delta=0.05)  # Should be exactly 2.0
        self.assertAlmostEqual(ratio_3_to_1, 1.5, delta=0.05)  # Should be exactly 1.5

        # 2. Verify all miners received payouts (positive emissions)
        # Aggressive strategy may overpay, but amounts should be in right ballpark (within 50%)
        self.assertGreater(cumulative_payouts["miner1"], 25000.0)  # At least 50% of needed
        self.assertLess(cumulative_payouts["miner1"], 100000.0)  # At most 2x needed
        self.assertGreater(cumulative_payouts["miner2"], 50000.0)
        self.assertLess(cumulative_payouts["miner2"], 200000.0)
        self.assertGreater(cumulative_payouts["miner3"], 37500.0)
        self.assertLess(cumulative_payouts["miner3"], 150000.0)

        # 3. Verify weights decrease over time
        # Weights should be highest at day 1 and decrease as payouts are fulfilled
        day_1_sum = weights_over_time[0]["miner1"] + weights_over_time[0]["miner2"] + weights_over_time[0]["miner3"]
        day_10_sum = weights_over_time[9]["miner1"] + weights_over_time[9]["miner2"] + weights_over_time[9]["miner3"]
        day_20_sum = weights_over_time[19]["miner1"] + weights_over_time[19]["miner2"] + weights_over_time[19]["miner3"]
        day_25_sum = weights_over_time[24]["miner1"] + weights_over_time[24]["miner2"] + weights_over_time[24]["miner3"]

        # Weights should decrease monotonically (or stay at minimum dust)
        self.assertGreaterEqual(day_1_sum, day_10_sum)
        self.assertGreaterEqual(day_10_sum, day_20_sum)
        self.assertGreaterEqual(day_20_sum, day_25_sum)

        # 4. Verify weights approach zero (or minimum dust) by day 25
        # By day 25, remaining payouts should be close to zero, so weights should be minimal
        dust = self.expected_dynamic_dust
        expected_minimum_sum = 3 * (3 * dust)  # 3 miners * 3x dust (MAINCOMP)

        # Day 25 weights should be close to minimum (within 10%)
        self.assertLess(day_25_sum, expected_minimum_sum * 1.1)

        # 5. Verify early aggressive payout (more weight early on)
        # Days 1-10 should receive more total emissions than days 11-20
        early_payout_sum = sum(
            weights_over_time[i]["miner1"] + weights_over_time[i]["miner2"] + weights_over_time[i]["miner3"]
            for i in range(0, 10)
        )
        mid_payout_sum = sum(
            weights_over_time[i]["miner1"] + weights_over_time[i]["miner2"] + weights_over_time[i]["miner3"]
            for i in range(10, 20)
        )

        # Early period should have higher total weights (aggressive payout)
        self.assertGreater(early_payout_sum, mid_payout_sum)

    def test_high_payouts_normalize_without_burn(self):
        """Test that when payouts exceed network capacity (sum >= 1.0), we normalize without burn address"""
        current_time = datetime(2026, 1, 25, 12, 0, 0, tzinfo=timezone.utc)  # Late in month
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        current_month_checkpoint = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        current_month_checkpoint_ms = int(current_month_checkpoint.timestamp() * 1000)

        # Create 3 miners with high performance (high remaining payouts)
        # With high payouts and few days remaining, sum will exceed 1.0
        ledger1 = DebtLedger(hotkey="high_performer_1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=50000.0,
            unrealized_pnl=-10000.0,  # net_pnl = 40000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=1000.0,  # Received some emissions
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger2 = DebtLedger(hotkey="high_performer_2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=60000.0,
            unrealized_pnl=-10000.0,  # net_pnl = 50000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=1200.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger3 = DebtLedger(hotkey="high_performer_3", checkpoints=[])
        ledger3.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=40000.0,
            unrealized_pnl=-10000.0,  # net_pnl = 30000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger3.checkpoints.append(DebtCheckpoint(
            timestamp_ms=current_month_checkpoint_ms,
            chunk_emissions_alpha=800.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {
            "high_performer_1": ledger1,
            "high_performer_2": ledger2,
            "high_performer_3": ledger3
        }

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
                        current_time_ms=current_time_ms,
                        is_testnet=False,
            verbose=True
        )

        # Should have exactly 3 entries (NO burn address)
        self.assertEqual(len(result), 3)

        weights_dict = dict(result)

        # Verify NO burn address is present
        self.assertNotIn("burn_address_mainnet", weights_dict)
        self.assertNotIn("burn_address_testnet", weights_dict)

        # Verify all 3 miners are present
        self.assertIn("high_performer_1", weights_dict)
        self.assertIn("high_performer_2", weights_dict)
        self.assertIn("high_performer_3", weights_dict)

        # Total should sum to exactly 1.0 (normalized)
        total_weight = sum(weight for _, weight in result)
        self.assertAlmostEqual(total_weight, 1.0, places=10)

        # Verify proportional distribution is maintained
        # high_performer_2 has highest PnL (50000), should have highest weight
        # high_performer_1 has medium PnL (40000), should have medium weight
        # high_performer_3 has lowest PnL (30000), should have lowest weight
        self.assertGreater(weights_dict["high_performer_2"], weights_dict["high_performer_1"])
        self.assertGreater(weights_dict["high_performer_1"], weights_dict["high_performer_3"])

        # Check approximate ratio (should be 5:4:3 based on net PnL)
        ratio_2_to_3 = weights_dict["high_performer_2"] / weights_dict["high_performer_3"]
        ratio_1_to_3 = weights_dict["high_performer_1"] / weights_dict["high_performer_3"]
        self.assertAlmostEqual(ratio_2_to_3, 50000.0 / 30000.0, places=1)  # ~1.67
        self.assertAlmostEqual(ratio_1_to_3, 40000.0 / 30000.0, places=1)  # ~1.33


    # ========================================================================
    # DYNAMIC DUST TESTS
    # ========================================================================

    def test_dynamic_dust_enabled_by_default(self):
        """Test that dynamic dust is always enabled (miners with same PnL get same dynamic weight)"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create miners with different PnL in MAINCOMP bucket
        ledger1 = DebtLedger(hotkey="miner1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative PnL -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger2 = DebtLedger(hotkey="miner2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative PnL -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"miner1": ledger1, "miner2": ledger2}

        # Call compute_results (dynamic dust always enabled)
        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Both miners have 0 PnL (negative floored at 0), should get floor weight (3x dust for MAINCOMP)
        self.assertAlmostEqual(weights_dict["miner1"], 3 * dust, places=10)
        self.assertAlmostEqual(weights_dict["miner2"], 3 * dust, places=10)

    def test_dynamic_dust_within_bucket_scaling(self):
        """Test that dynamic dust properly scales weights within bucket based on 30-day PnL"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create checkpoint within 30-day window (10 days ago, in CURRENT month)
        # This ensures it's used for dynamic dust but NOT for previous month payout
        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        # For main scoring: previous month checkpoint (OUTSIDE earning period)
        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create 3 miners in MAINCOMP bucket with different 30-day PnL
        # Use a single checkpoint within 30-day window for clarity

        # Miner 1: Best performer (10,000 PnL)
        ledger1 = DebtLedger(hotkey="best_miner", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=10000.0,
            unrealized_pnl=0.0,  # net_pnl = 10000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        # Prev month checkpoint for main scoring (negative to ensure 0 remaining payout)
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: Middle performer (5,000 PnL)
        ledger2 = DebtLedger(hotkey="middle_miner", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=5000.0,
            unrealized_pnl=0.0,  # net_pnl = 5000
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 3: Worst performer (0 PnL)
        ledger3 = DebtLedger(hotkey="worst_miner", checkpoints=[])
        ledger3.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=0.0,
            unrealized_pnl=0.0,  # net_pnl = 0
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger3.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {
            "best_miner": ledger1,
            "middle_miner": ledger2,
            "worst_miner": ledger3
        }

        # Call compute_results (dynamic dust always enabled)
        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # MAINCOMP floor = 3x dust, ceiling = 4x dust
        floor = 3 * dust
        ceiling = 4 * dust

        # Verify scaling:
        # - Best performer should get ceiling (4x dust)
        # - Worst performer should get floor (3x dust)
        # - Middle performer should get between floor and ceiling
        self.assertAlmostEqual(weights_dict["best_miner"], ceiling, places=10)
        self.assertAlmostEqual(weights_dict["worst_miner"], floor, places=10)

        # Middle miner should be exactly halfway between floor and ceiling
        expected_middle = floor + 0.5 * (ceiling - floor)
        self.assertAlmostEqual(weights_dict["middle_miner"], expected_middle, places=10)

        # Verify ordering
        self.assertGreater(weights_dict["best_miner"], weights_dict["middle_miner"])
        self.assertGreater(weights_dict["middle_miner"], weights_dict["worst_miner"])

    def test_dynamic_dust_cross_bucket_hierarchy(self):
        """Test that cross-bucket hierarchy is maintained with dynamic dust"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Use CURRENT month for dynamic dust (not previous month)
        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create worst MAINCOMP (0 PnL) and best PROBATION (high PnL)
        # Worst MAINCOMP should still get >= best PROBATION due to bucket floors

        # Worst MAINCOMP miner (0 PnL)
        ledger_maincomp = DebtLedger(hotkey="worst_maincomp", checkpoints=[])
        ledger_maincomp.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=0.0,
            unrealized_pnl=0.0,  # net_pnl = 0
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger_maincomp.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Best PROBATION miner (10,000 PnL)
        ledger_probation = DebtLedger(hotkey="best_probation", checkpoints=[])
        ledger_probation.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=10000.0,
            unrealized_pnl=0.0,  # net_pnl = 10000 (excellent)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.PROBATION.value
        ))
        ledger_probation.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.PROBATION.value
        ))

        ledgers = {
            "worst_maincomp": ledger_maincomp,
            "best_probation": ledger_probation
        }

        # Create custom mock challengeperiod_manager
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            if hotkey == "worst_maincomp":
                mock_bucket.value = MinerBucket.MAINCOMP.value
            elif hotkey == "best_probation":
                mock_bucket.value = MinerBucket.PROBATION.value
            else:
                mock_bucket.value = MinerBucket.UNKNOWN.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        # Mock adequate collateral for all miners
        mock_cm = Mock()
        def custom_get_collateral(hotkey):
            return 1000.0  # 1000 theta = adequate collateral
        mock_cm.get_miner_collateral_balance = Mock(side_effect=custom_get_collateral)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            mock_cm,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Verify:
        # - Worst MAINCOMP gets floor = 3x dust
        # - Best PROBATION gets ceiling = 3x dust
        # - They should be EQUAL (bucket floors/ceilings align)
        maincomp_floor = 3 * dust
        probation_ceiling = 3 * dust  # 2x + 1x = 3x

        self.assertAlmostEqual(weights_dict["worst_maincomp"], maincomp_floor, places=10)
        self.assertAlmostEqual(weights_dict["best_probation"], probation_ceiling, places=10)

        # Verify they're equal (hierarchy preserved at boundaries)
        self.assertAlmostEqual(
            weights_dict["worst_maincomp"],
            weights_dict["best_probation"],
            places=10
        )

    def test_dynamic_dust_all_miners_zero_pnl(self):
        """Test that all miners with 0 PnL get floor weight"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create 3 miners with all 0 PnL
        ledgers = {}
        for i in range(3):
            ledger = DebtLedger(hotkey=f"miner{i}", checkpoints=[])
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=within_window_ms,
                realized_pnl=0.0,
                unrealized_pnl=0.0,  # net_pnl = 0
                total_penalty=1.0,
                challenge_period_status=MinerBucket.MAINCOMP.value
            ))
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=prev_month_checkpoint_ms,
                realized_pnl=0.0,
                unrealized_pnl=-1.0,  # Negative -> 0 remaining payout
                total_penalty=1.0,
                challenge_period_status=MinerBucket.MAINCOMP.value
            ))
            ledgers[f"miner{i}"] = ledger

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # All miners should get exactly floor weight (3x dust for MAINCOMP)
        floor = 3 * dust
        for i in range(3):
            self.assertAlmostEqual(weights_dict[f"miner{i}"], floor, places=10)

    def test_dynamic_dust_negative_pnl_floored_at_zero(self):
        """Test that negative PnL is floored at 0 for dust calculation"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create 2 miners: one with negative PnL, one with 0 PnL
        ledger_negative = DebtLedger(hotkey="negative_miner", checkpoints=[])
        ledger_negative.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=-1000.0,
            unrealized_pnl=-1.0,  # net_pnl = -4000 (negative)
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger_negative.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-5000.0,  # Negative -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledger_zero = DebtLedger(hotkey="zero_miner", checkpoints=[])
        ledger_zero.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=0.0,
            unrealized_pnl=0.0,  # net_pnl = 0
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger_zero.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"negative_miner": ledger_negative, "zero_miner": ledger_zero}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Both should get floor weight (negative PnL floored at 0)
        floor = 3 * dust
        self.assertAlmostEqual(weights_dict["negative_miner"], floor, places=10)
        self.assertAlmostEqual(weights_dict["zero_miner"], floor, places=10)

    def test_dynamic_dust_30_day_lookback_window(self):
        """Test that only checkpoints within 30-day window are considered"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # 2 months ago (OUTSIDE 30-day window AND outside previous month)
        old_checkpoint = datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
        old_checkpoint_ms = int(old_checkpoint.timestamp() * 1000)

        # 20 days ago (INSIDE 30-day window)
        recent_checkpoint = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
        recent_checkpoint_ms = int(recent_checkpoint.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Miner 1: Has old checkpoint with high PnL (should be IGNORED)
        ledger1 = DebtLedger(hotkey="miner1", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=old_checkpoint_ms,
            realized_pnl=10000.0,  # High PnL but too old
            unrealized_pnl=0.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: Has recent checkpoint with high PnL (should be USED for dynamic dust)
        # Use CHALLENGE status so it doesn't count for previous month payout (only for dynamic dust)
        ledger2 = DebtLedger(hotkey="miner2", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=recent_checkpoint_ms,
            realized_pnl=10000.0,  # High PnL within window
            unrealized_pnl=0.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.CHALLENGE.value  # CHALLENGE = not earning status
        ))
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,  # Negative -> 0 remaining payout
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {"miner1": ledger1, "miner2": ledger2}

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Miner 1 should get floor (old checkpoint ignored)
        # Miner 2 should get ceiling (recent checkpoint used)
        floor = 3 * dust
        ceiling = 4 * dust

        self.assertAlmostEqual(weights_dict["miner1"], floor, places=10)
        self.assertAlmostEqual(weights_dict["miner2"], ceiling, places=10)

    def test_dynamic_dust_penalty_applied_to_pnl(self):
        """Test that penalties are applied to PnL in dynamic dust calculation"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2025, 12, 26, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Miner 1: 10,000 PnL with no penalty
        ledger1 = DebtLedger(hotkey="no_penalty", checkpoints=[])
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=10000.0,
            unrealized_pnl=0.0,  # net_pnl = 10000
            total_penalty=1.0,  # No penalty
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger1.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 2: 10,000 PnL with 50% penalty (effective PnL = 5000)
        ledger2 = DebtLedger(hotkey="with_penalty", checkpoints=[])
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=10000.0,
            unrealized_pnl=0.0,  # net_pnl = 10000
            total_penalty=0.5,  # 50% penalty -> effective PnL = 5000
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger2.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        # Miner 3: 5,000 PnL with no penalty (for comparison)
        ledger3 = DebtLedger(hotkey="half_pnl", checkpoints=[])
        ledger3.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=5000.0,
            unrealized_pnl=0.0,  # net_pnl = 5000
            total_penalty=1.0,  # No penalty
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))
        ledger3.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.MAINCOMP.value
        ))

        ledgers = {
            "no_penalty": ledger1,
            "with_penalty": ledger2,
            "half_pnl": ledger3
        }

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            self.mock_challengeperiod_manager,
            self.mock_contract_manager,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Miner with penalty should have SAME weight as miner with half the PnL
        # (10000 * 0.5 = 5000 effective PnL)
        self.assertAlmostEqual(
            weights_dict["with_penalty"],
            weights_dict["half_pnl"],
            places=10
        )

        # Miner with no penalty should have higher weight
        self.assertGreater(weights_dict["no_penalty"], weights_dict["with_penalty"])

    # ========================================================================
    # CALCULATE_DYNAMIC_DUST UNIT TESTS
    # ========================================================================

    def test_calculate_dynamic_dust_success(self):
        """Test successful dynamic dust calculation with valid metagraph data"""
        # Expected calculation with mocked data:
        # - Total TAO per tempo: 10 * 360 = 3600 TAO
        # - TAO per block: 3600 / 360 = 10 TAO
        # - TAO per day: 10 * 7200 = 72,000 TAO
        # - ALPHA per day: 72,000 / 0.5 = 144,000 ALPHA
        # - $0.01 in TAO: 0.01 / 500 = 0.00002 TAO
        # - $0.01 in ALPHA: 0.00002 / 0.5 = 0.00004 ALPHA
        # - Dust weight: 0.00004 / 144,000 = 2.777...e-10

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=self.mock_metagraph,
            target_daily_usd=0.01,
            verbose=True
        )

        expected_dust = 0.00004 / 144000.0  # 2.777...e-10
        self.assertAlmostEqual(dust, expected_dust, places=12)

        # Verify dust is in reasonable range
        self.assertGreater(dust, 0)
        self.assertLess(dust, 0.001)

    def test_calculate_dynamic_dust_missing_emission_attr(self):
        """Test fallback when metagraph is missing emission attribute"""
        mock_metagraph = Mock()
        # Don't set emission attribute

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        # Should fallback to static dust
        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_emission_none(self):
        """Test fallback when emission is None"""
        mock_metagraph = Mock()
        mock_metagraph.emission = None

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_emission_not_summable(self):
        """Test fallback when emission cannot be summed"""
        mock_metagraph = Mock()
        mock_metagraph.emission = "not_a_list"  # Will fail on sum()

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_zero_emissions(self):
        """Test fallback when total emissions are zero"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [0] * 10  # All zeros

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_negative_emissions(self):
        """Test fallback when total emissions are negative"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [-100]  # Negative (shouldn't happen but test anyway)

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_missing_reserves(self):
        """Test fallback when reserve attributes are missing"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10
        # Don't set tao_reserve_rao or alpha_reserve_rao

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_zero_reserves(self):
        """Test fallback when reserves are zero"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        # Set reserves to zero
        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 0.0
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 0.0
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_invalid_alpha_to_tao_rate(self):
        """Test fallback when ALPHA-to-TAO rate is > 1.0"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        # Set reserves so alpha_to_tao_rate > 1.0 (invalid)
        # alpha_to_tao_rate = tao_reserve / alpha_reserve
        # To get > 1.0: tao_reserve > alpha_reserve
        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 2_000_000 * 1e9  # 2M TAO
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 1_000_000 * 1e9  # 1M ALPHA (rate = 2.0, invalid)
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_missing_tao_usd_price(self):
        """Test fallback when TAO/USD price is missing"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve
        # Don't set tao_to_usd_rate

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_zero_tao_usd_price(self):
        """Test fallback when TAO/USD price is zero"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve
        mock_metagraph.tao_to_usd_rate = 0.0  # Zero price

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_negative_tao_usd_price(self):
        """Test fallback when TAO/USD price is negative"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve
        mock_metagraph.tao_to_usd_rate = -100.0  # Negative price

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_tao_price_out_of_range_low(self):
        """Test fallback when TAO/USD price is below $1"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve
        mock_metagraph.tao_to_usd_rate = 0.5  # Below $1

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_tao_price_out_of_range_high(self):
        """Test fallback when TAO/USD price is above $10,000"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve
        mock_metagraph.tao_to_usd_rate = 15000.0  # Above $10,000

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_invalid_tao_usd_type(self):
        """Test fallback when TAO/USD price has invalid type"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve
        mock_metagraph.tao_to_usd_rate = "not_a_number"  # Invalid type

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_weight_exceeds_maximum(self):
        """Test fallback when calculated dust weight exceeds 0.001"""
        mock_metagraph = Mock()
        # Very low emissions to create high dust weight (> 0.001)
        # With emission = [0.0005], dust will be 0.002 which exceeds 0.001
        mock_metagraph.emission = [0.0005]  # Extremely low to create dust > 0.001

        mock_tao_reserve = Mock()
        mock_tao_reserve.value = 1_000_000 * 1e9
        mock_alpha_reserve = Mock()
        mock_alpha_reserve.value = 2_000_000 * 1e9
        mock_metagraph.tao_reserve_rao = mock_tao_reserve
        mock_metagraph.alpha_reserve_rao = mock_alpha_reserve
        mock_metagraph.tao_to_usd_rate = 500.0

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    def test_calculate_dynamic_dust_different_target_amounts(self):
        """Test that dynamic dust scales linearly with target amount"""
        # Calculate dust for $0.01
        dust_1_cent = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=self.mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        # Calculate dust for $0.02 (should be exactly 2x)
        dust_2_cent = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=self.mock_metagraph,
            target_daily_usd=0.02,
            verbose=False
        )

        # Should be exactly 2x
        self.assertAlmostEqual(dust_2_cent / dust_1_cent, 2.0, places=10)

    def test_calculate_dynamic_dust_market_responsive(self):
        """Test that dust adjusts when TAO price changes"""
        # Calculate with $500/TAO
        dust_high_price = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=self.mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        # Create metagraph with $250/TAO (half the price)
        mock_metagraph_low_price = Mock()
        mock_metagraph_low_price.emission = self.mock_metagraph.emission
        mock_metagraph_low_price.tao_reserve_rao = self.mock_metagraph.tao_reserve_rao
        mock_metagraph_low_price.alpha_reserve_rao = self.mock_metagraph.alpha_reserve_rao
        mock_metagraph_low_price.tao_to_usd_rate = 250.0  # Half the price

        dust_low_price = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph_low_price,
            target_daily_usd=0.01,
            verbose=False
        )

        # Lower TAO price means need more ALPHA for same USD amount
        # So dust weight should be higher (approximately 2x)
        self.assertGreater(dust_low_price, dust_high_price)
        self.assertAlmostEqual(dust_low_price / dust_high_price, 2.0, places=1)

    def test_calculate_dynamic_dust_reserve_value_extraction(self):
        """Test that dust calculation works with direct float values (no .value accessor)"""
        mock_metagraph = Mock()
        mock_metagraph.emission = [360] * 10

        # Use direct float values instead of Mock with .value
        mock_metagraph.tao_reserve_rao = 1_000_000 * 1e9  # Direct float
        mock_metagraph.alpha_reserve_rao = 2_000_000 * 1e9  # Direct float
        mock_metagraph.tao_to_usd_rate = 500.0

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        # Should work and return valid dust (not fallback)
        self.assertNotEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)
        self.assertGreater(dust, 0)
        self.assertLess(dust, 0.001)

    def test_calculate_dynamic_dust_exception_handling(self):
        """Test fallback when unexpected exception occurs"""
        mock_metagraph = Mock()
        # Make emission raise an exception when accessed
        mock_metagraph.emission = Mock(side_effect=RuntimeError("Unexpected error"))

        dust = DebtBasedScoring.calculate_dynamic_dust(
            metagraph=mock_metagraph,
            target_daily_usd=0.01,
            verbose=False
        )

        self.assertEqual(dust, ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT)

    # ========================================================================
    # CHALLENGE BUCKET TESTS (Bottom 25% get 0 weight, capped at 10 miners)
    # ========================================================================

    def test_challenge_bucket_bottom_25_percent_gets_zero_weight(self):
        """Test that bottom 25% of CHALLENGE miners get 0 weight"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        # Create checkpoint within 30-day window for dynamic dust
        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create 20 CHALLENGE miners with varying PnL (bottom 5 should get 0 weight = 25% of 20)
        ledgers = {}
        for i in range(20):
            ledger = DebtLedger(hotkey=f"challenge_miner_{i}", checkpoints=[])
            # Distribute PnL from 0 to 19000 (miner_0 has lowest, miner_19 has highest)
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=within_window_ms,
                realized_pnl=float(i * 1000),
                unrealized_pnl=0.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=prev_month_checkpoint_ms,
                realized_pnl=0.0,
                unrealized_pnl=-1.0,  # Negative -> 0 remaining payout
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledgers[f"challenge_miner_{i}"] = ledger

        # Create custom mock challengeperiod_manager (all CHALLENGE)
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            mock_bucket.value = MinerBucket.CHALLENGE.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        # Mock adequate collateral for all miners (so PnL-based ranking applies)
        mock_cm = Mock()
        def custom_get_account_size(hotkey, most_recent=False):
            return 175000.0  # $175k USD = adequate collateral (> $99,925 MIN_COLLATERAL_VALUE)
        mock_cm.get_miner_account_size = Mock(side_effect=custom_get_account_size)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            mock_cm,  # Use custom mock with adequate collateral
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Filter out burn address from weights_dict for testing
        miner_weights = {k: v for k, v in weights_dict.items() if not k.startswith("burn_address") and not k.startswith("hotkey_")}

        # Bottom 5 miners (0-4) should have 0 weight (PnL-based, all have adequate collateral)
        for i in range(5):
            self.assertEqual(miner_weights[f"challenge_miner_{i}"], 0.0,
                           f"Miner {i} should have 0 weight (bottom 25%)")

        # Remaining miners (5-19) should have non-zero weight
        for i in range(5, 20):
            self.assertGreater(miner_weights[f"challenge_miner_{i}"], 0.0,
                             f"Miner {i} should have non-zero weight")

        # Verify miner 5 has weight based on its normalized PnL
        # PnL = 5000, max_pnl = 19000, normalized = 5000/19000
        # weight = floor + (normalized * (ceiling - floor))
        floor = dust
        ceiling = 2 * dust
        expected_miner_5 = floor + (5000.0 / 19000.0) * (ceiling - floor)
        self.assertAlmostEqual(miner_weights["challenge_miner_5"], expected_miner_5, places=10)

        # Verify highest miner gets ceiling (floor + dust)
        self.assertAlmostEqual(miner_weights["challenge_miner_19"], ceiling, places=10)

    def test_challenge_bucket_cap_at_10_miners(self):
        """Test that maximum 10 CHALLENGE miners get 0 weight even with > 40 miners"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create 50 CHALLENGE miners (25% = 12.5, but capped at 10)
        ledgers = {}
        for i in range(50):
            ledger = DebtLedger(hotkey=f"challenge_miner_{i}", checkpoints=[])
            # Distribute PnL from 0 to 49000
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=within_window_ms,
                realized_pnl=float(i * 1000),
                unrealized_pnl=0.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=prev_month_checkpoint_ms,
                realized_pnl=0.0,
                unrealized_pnl=-1.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledgers[f"challenge_miner_{i}"] = ledger

        # Create custom mock challengeperiod_manager
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            mock_bucket.value = MinerBucket.CHALLENGE.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        # Mock adequate collateral for all miners (so PnL-based ranking applies)
        mock_cm = Mock()
        def custom_get_account_size(hotkey, most_recent=False):
            return 175000.0  # $175k USD = adequate collateral (> $99,925 MIN_COLLATERAL_VALUE)
        mock_cm.get_miner_account_size = Mock(side_effect=custom_get_account_size)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            mock_cm,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Filter out burn address
        miner_weights = {k: v for k, v in weights_dict.items() if not k.startswith("burn_address") and not k.startswith("hotkey_")}

        # Count how many miners have 0 weight
        zero_weight_count = sum(1 for weight in miner_weights.values() if weight == 0.0)

        # Should be exactly 10 (capped at max)
        self.assertEqual(zero_weight_count, 10,
                        "Should have exactly 10 miners with 0 weight (cap)")

        # Bottom 10 miners (0-9) should have 0 weight
        for i in range(10):
            self.assertEqual(miner_weights[f"challenge_miner_{i}"], 0.0,
                           f"Miner {i} should have 0 weight")

        # Miner 10 onwards should have non-zero weight
        for i in range(10, 50):
            self.assertGreater(miner_weights[f"challenge_miner_{i}"], 0.0,
                             f"Miner {i} should have non-zero weight")

    def test_challenge_bucket_all_zero_pnl_lexicographic_selection(self):
        """Test that when all CHALLENGE miners have 0 PnL, bottom 25% (capped at 10) get 0 weight by lexicographic order"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create 15 CHALLENGE miners all with 0 PnL
        # Use specific hotkey names to test lexicographic order
        hotkeys = [
            "hotkey_alice", "hotkey_bob", "hotkey_charlie", "hotkey_david",
            "hotkey_eve", "hotkey_frank", "hotkey_grace", "hotkey_henry",
            "hotkey_iris", "hotkey_jack", "hotkey_kate", "hotkey_leo",
            "hotkey_mary", "hotkey_nick", "hotkey_olivia"
        ]

        ledgers = {}
        for hotkey in hotkeys:
            ledger = DebtLedger(hotkey=hotkey, checkpoints=[])
            # All have 0 PnL
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=within_window_ms,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=prev_month_checkpoint_ms,
                realized_pnl=0.0,
                unrealized_pnl=-1.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledgers[hotkey] = ledger

        # Create custom mock challengeperiod_manager
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            mock_bucket.value = MinerBucket.CHALLENGE.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        # Mock adequate collateral for all miners (so PnL-based ranking applies)
        mock_cm = Mock()
        def custom_get_account_size(hotkey, most_recent=False):
            return 175000.0  # $175k USD = adequate collateral (> $99,925 MIN_COLLATERAL_VALUE)
        mock_cm.get_miner_account_size = Mock(side_effect=custom_get_account_size)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            mock_cm,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Filter out burn address
        miner_weights = {k: v for k, v in weights_dict.items() if k in hotkeys}

        # Sort hotkeys lexicographically to determine which get 0 weight
        # With 15 miners, 25% = 3.75 → 3 miners get 0 weight (capped at 10)
        sorted_hotkeys = sorted(hotkeys)
        num_zero = min(int(len(hotkeys) * 0.25), 10)  # Should be 3
        zero_weight_hotkeys = sorted_hotkeys[:num_zero]  # First 3 lexicographically
        non_zero_weight_hotkeys = sorted_hotkeys[num_zero:]  # Remaining 12

        # First 3 (lexicographically) should have 0 weight
        for hotkey in zero_weight_hotkeys:
            self.assertEqual(miner_weights[hotkey], 0.0,
                           f"{hotkey} should have 0 weight (bottom 25%)")

        # Remaining 12 should have floor weight
        for hotkey in non_zero_weight_hotkeys:
            self.assertAlmostEqual(miner_weights[hotkey], dust, places=10,
                                 msg=f"{hotkey} should have floor weight")

    def test_challenge_bucket_small_group_all_zero_pnl(self):
        """Test that with 8 CHALLENGE miners all at 0 PnL, bottom 2 get 0 weight (25% of 8)"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Create 8 miners with 0 PnL (25% = 2 miners)
        hotkeys = ["miner_a", "miner_b", "miner_c", "miner_d", "miner_e", "miner_f", "miner_g", "miner_h"]
        ledgers = {}
        for hotkey in hotkeys:
            ledger = DebtLedger(hotkey=hotkey, checkpoints=[])
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=within_window_ms,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=prev_month_checkpoint_ms,
                realized_pnl=0.0,
                unrealized_pnl=-1.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledgers[hotkey] = ledger

        # Create custom mock challengeperiod_manager
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            mock_bucket.value = MinerBucket.CHALLENGE.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        # Mock adequate collateral for all miners (so PnL-based ranking applies)
        mock_cm = Mock()
        def custom_get_account_size(hotkey, most_recent=False):
            return 175000.0  # $175k USD = adequate collateral (> $99,925 MIN_COLLATERAL_VALUE)
        mock_cm.get_miner_account_size = Mock(side_effect=custom_get_account_size)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            mock_cm,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Filter out burn address
        miner_weights = {k: v for k, v in weights_dict.items() if k in hotkeys}

        # Count zero weight miners
        zero_weight_count = sum(1 for weight in miner_weights.values() if weight == 0.0)
        self.assertEqual(zero_weight_count, 2, "Should have exactly 2 miners with 0 weight (25% of 8)")

        # Lexicographically first 2 should have 0 weight
        sorted_hotkeys = sorted(hotkeys)
        self.assertEqual(miner_weights[sorted_hotkeys[0]], 0.0)
        self.assertEqual(miner_weights[sorted_hotkeys[1]], 0.0)

        # Remaining 6 should have floor weight
        for i in range(2, 8):
            self.assertAlmostEqual(miner_weights[sorted_hotkeys[i]], dust, places=10)

    def test_challenge_bucket_single_miner_zero_pnl_gets_floor_weight(self):
        """Test that single CHALLENGE miner with 0 PnL gets floor weight (not 0)"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        dust = self.expected_dynamic_dust

        # Single CHALLENGE miner with 0 PnL
        ledger = DebtLedger(hotkey="solo_challenge_miner", checkpoints=[])
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=within_window_ms,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))
        ledger.checkpoints.append(DebtCheckpoint(
            timestamp_ms=prev_month_checkpoint_ms,
            realized_pnl=0.0,
            unrealized_pnl=-1.0,
            total_penalty=1.0,
            challenge_period_status=MinerBucket.CHALLENGE.value
        ))

        # Create custom mock challengeperiod_manager
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            mock_bucket.value = MinerBucket.CHALLENGE.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        # Mock adequate collateral for the miner
        mock_cm = Mock()
        def custom_get_account_size(hotkey, most_recent=False):
            return 175000.0  # $175k USD = adequate collateral (> $99,925 MIN_COLLATERAL_VALUE)
        mock_cm.get_miner_account_size = Mock(side_effect=custom_get_account_size)

        result = DebtBasedScoring.compute_results(
            {"solo_challenge_miner": ledger},
            self.mock_metagraph,
            mock_cpm,
            mock_cm,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Single miner should get floor weight (1x dust for CHALLENGE), NOT 0
        # Due to burn address logic, result will have 2 entries (miner + burn address)
        # But the key assertion is that the miner gets floor weight, not 0
        self.assertIn("solo_challenge_miner", weights_dict)
        self.assertAlmostEqual(weights_dict["solo_challenge_miner"], dust, places=10,
                             msg="Single CHALLENGE miner with 0 PnL should get floor weight, not 0")

    def test_challenge_bucket_threshold_boundary(self):
        """Test that miners exactly at the threshold get non-zero weight"""
        current_time = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time_ms = int(current_time.timestamp() * 1000)

        within_window = datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        within_window_ms = int(within_window.timestamp() * 1000)

        prev_month_checkpoint = datetime(2025, 12, 10, 12, 0, 0, tzinfo=timezone.utc)
        prev_month_checkpoint_ms = int(prev_month_checkpoint.timestamp() * 1000)

        # Create 12 miners (25% = 3, so bottom 3 get 0 weight)
        # Create specific PnL distribution to test boundary
        ledgers = {}
        pnl_values = [100, 200, 300, 400, 400, 500, 600, 700, 800, 900, 1000, 1100]

        for i, pnl in enumerate(pnl_values):
            ledger = DebtLedger(hotkey=f"miner_{i}", checkpoints=[])
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=within_window_ms,
                realized_pnl=float(pnl),
                unrealized_pnl=0.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledger.checkpoints.append(DebtCheckpoint(
                timestamp_ms=prev_month_checkpoint_ms,
                realized_pnl=0.0,
                unrealized_pnl=-1.0,
                total_penalty=1.0,
                challenge_period_status=MinerBucket.CHALLENGE.value
            ))
            ledgers[f"miner_{i}"] = ledger

        # Create custom mock challengeperiod_manager
        mock_cpm = Mock()
        def custom_get_miner_bucket(hotkey):
            mock_bucket = Mock()
            mock_bucket.value = MinerBucket.CHALLENGE.value
            return mock_bucket
        mock_cpm.get_miner_bucket = Mock(side_effect=custom_get_miner_bucket)

        # Mock adequate collateral for all miners (so PnL-based ranking applies)
        mock_cm = Mock()
        def custom_get_account_size(hotkey, most_recent=False):
            return 175000.0  # $175k USD = adequate collateral (> $99,925 MIN_COLLATERAL_VALUE)
        mock_cm.get_miner_account_size = Mock(side_effect=custom_get_account_size)

        result = DebtBasedScoring.compute_results(
            ledgers,
            self.mock_metagraph,
            mock_cpm,
            mock_cm,
            current_time_ms=current_time_ms,
            is_testnet=False,
            verbose=True
        )

        weights_dict = dict(result)

        # Filter out burn address
        miner_weights = {k: v for k, v in weights_dict.items() if k.startswith("miner_")}

        # Sort by PnL to identify threshold
        # Bottom 3 should have 0 weight: miner_0 (100), miner_1 (200), miner_2 (300)
        # Threshold is at sorted_pnls[3] = 400
        # Miners at exactly 400 (miner_3, miner_4) should have non-zero weight
        self.assertEqual(miner_weights["miner_0"], 0.0)
        self.assertEqual(miner_weights["miner_1"], 0.0)
        self.assertEqual(miner_weights["miner_2"], 0.0)

        # Miners at threshold should NOT have 0 weight
        self.assertGreater(miner_weights["miner_3"], 0.0,
                         "Miner at threshold should have non-zero weight")
        self.assertGreater(miner_weights["miner_4"], 0.0,
                         "Miner at threshold should have non-zero weight")


if __name__ == '__main__':
    unittest.main()
