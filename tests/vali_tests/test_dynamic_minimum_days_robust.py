"""
Robust test suite for dynamic minimum participation days calculation.
Uses production code paths and tests for exact expected values.
"""

import unittest
from typing import Dict, List

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.vali_config import ValiConfig, TradePairCategory, TradePair
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger, PerfCheckpoint, TP_ID_PORTFOLIO


class TestDynamicMinimumDaysRobust(TestBase):
    """Robust test suite using production code paths for dynamic minimum days calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.checkpoint_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS  # 12 hours
        self.daily_ms = ValiConfig.DAILY_MS
        
    def create_checkpoint_sequence(
        self, 
        start_time_ms: int, 
        num_days: int
    ) -> List[PerfCheckpoint]:
        """
        Create a precise sequence of checkpoints that will produce exactly num_days 
        from daily_return_log_by_date().
        
        Based on the production logic:
        - Only full cells (accum_ms == TARGET_CHECKPOINT_DURATION_MS) count
        - Only complete days (exactly 2 checkpoints per day) count
        - Checkpoints are grouped by their START time (last_update_ms - accum_ms)
        
        Args:
            start_time_ms: Starting timestamp (should be start of UTC day)
            num_days: Number of complete days to create
        
        Returns:
            List of PerfCheckpoint objects that produce exactly num_days
        """
        checkpoints = []
        checkpoints_per_day = int(ValiConfig.DAILY_CHECKPOINTS)  # Should be 2
        checkpoint_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS  # 12 hours
        
        # Start at beginning of a UTC day (midnight)
        # Ensure start_time_ms is aligned to UTC day boundary
        day_ms = ValiConfig.DAILY_MS
        aligned_start = (start_time_ms // day_ms) * day_ms
        
        current_start_time = aligned_start
        
        for day in range(num_days):
            for checkpoint_in_day in range(checkpoints_per_day):
                # Calculate the end time for this checkpoint
                end_time = current_start_time + checkpoint_duration
                
                # Create checkpoint with precise timing
                checkpoint = PerfCheckpoint(
                    last_update_ms=end_time,  # End time of checkpoint
                    prev_portfolio_ret=1.0 + (day * 0.001) + (checkpoint_in_day * 0.0001),
                    accum_ms=checkpoint_duration,  # MUST be exactly TARGET_CHECKPOINT_DURATION_MS
                    open_ms=checkpoint_duration,   # Full checkpoint duration
                    n_updates=5,
                    gain=0.005 + (day * 0.0001),   # Small progression
                    loss=0.002,
                    mdd=0.99
                )
                checkpoints.append(checkpoint)
                current_start_time += checkpoint_duration
        
        return checkpoints
    
    def create_production_ledger(self, num_days: int, start_time_ms: int = 1000000000000) -> PerfLedger:
        """
        Create a PerfLedger using production patterns that will yield exactly num_days 
        when processed by LedgerUtils.daily_return_log().
        
        Args:
            num_days: Exact number of days this ledger should represent
            start_time_ms: Starting timestamp
        
        Returns:
            PerfLedger that produces exactly num_days from daily_return_log()
        """
        checkpoints = self.create_checkpoint_sequence(start_time_ms, num_days)
        
        ledger = PerfLedger(
            target_cp_duration_ms=self.checkpoint_duration_ms,
            target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS,
            cps=checkpoints
        )
        
        return ledger
    
    def create_production_ledger_dict(
        self, 
        miner_participation: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, PerfLedger]]:
        """
        Create a production-style ledger dictionary with realistic structure.
        
        Args:
            miner_participation: Dict mapping miner_hotkey -> trade_pair_id -> num_days
        
        Returns:
            Production-style ledger dictionary
        """
        ledger_dict = {}
        base_time = 1000000000000  # Base timestamp
        
        for miner_hotkey, trade_pairs in miner_participation.items():
            miner_ledgers = {}
            
            # Create portfolio ledger (aggregate of all positions)
            max_days = max(trade_pairs.values()) if trade_pairs else 0
            if max_days > 0:
                miner_ledgers[TP_ID_PORTFOLIO] = self.create_production_ledger(max_days, base_time)
            else:
                miner_ledgers[TP_ID_PORTFOLIO] = PerfLedger()
            
            # Create individual trade pair ledgers
            for trade_pair_id, days in trade_pairs.items():
                if days > 0:
                    # Stagger start times slightly to be realistic
                    start_time = base_time + hash(miner_hotkey + trade_pair_id) % (24 * 60 * 60 * 1000)
                    miner_ledgers[trade_pair_id] = self.create_production_ledger(days, start_time)
                else:
                    miner_ledgers[trade_pair_id] = PerfLedger()
            
            ledger_dict[miner_hotkey] = miner_ledgers
        
        return ledger_dict
    
    def verify_ledger_produces_expected_days(self, ledger: PerfLedger, expected_days: int):
        """Verify that a ledger produces exactly the expected number of days."""
        actual_days = len(LedgerUtils.daily_return_log(ledger))
        self.assertEqual(
            actual_days, 
            expected_days, 
            f"Ledger should produce exactly {expected_days} days, got {actual_days}"
        )
    
    def test_empty_ledger_dict_exact(self):
        """Test with empty ledger dictionary returns exact ceil value."""
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            {}, [TradePairCategory.CRYPTO]
        )
        self.assertEqual(result_dict[TradePairCategory.CRYPTO], ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL)
    
    def test_no_participating_miners_exact(self):
        """Test when no miners participate in the asset class returns exact floor."""
        # Create miners that only trade forex (not crypto)
        miner_participation = {
            "forex_trader_001": {"EURUSD": 30, "GBPUSD": 25},
            "forex_trader_002": {"USDJPY": 40, "USDCHF": 35},
            "forex_trader_003": {"AUDUSD": 20, "NZDUSD": 15}
        }
        
        ledger_dict = self.create_production_ledger_dict(miner_participation)
        
        # Test for crypto - should find no participants
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO]
        )

        self.assertEqual(result_dict[TradePairCategory.CRYPTO], ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
    
    def test_fewer_than_percentile_rank_miners_exact(self):
        """Test with fewer than PERCENTILE_RANK miners uses exact minimum participation."""
        # Create fewer than PERCENTILE_RANK miners trading crypto with known participation days
        percentile_rank = ValiConfig.DYNAMIC_MIN_DAYS_NUM_MINERS
        # Create fewer miners than the percentile rank (e.g., 15 if percentile rank is 20)
        num_miners = percentile_rank - 5
        participation_days = list(range(60, 60 - num_miners, -1))  # Descending participation
        miner_participation = {}
        
        for i, days in enumerate(participation_days):
            miner_participation[f"crypto_trader_{i:03d}"] = {
                "BTCUSD": days,
                "ETHUSD": days - 1 if days > 1 else days  # Slightly different participation
            }
        
        ledger_dict = self.create_production_ledger_dict(miner_participation)
        
        # Verify our ledgers produce expected days
        for i, expected_days in enumerate(participation_days):
            miner_key = f"crypto_trader_{i:03d}"
            btc_ledger = ledger_dict[miner_key]["BTCUSD"]
            self.verify_ledger_produces_expected_days(btc_ledger, expected_days)
        
        # Test the function
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO]
        )
        result = result_dict[TradePairCategory.CRYPTO]

        # Get actual aggregated participation after asset segmentation
        segmentation = AssetSegmentation(ledger_dict)
        crypto_ledgers = segmentation.segmentation(TradePairCategory.CRYPTO)
        
        actual_participation = []
        for miner_hotkey, ledger in crypto_ledgers.items():
            if ledger and ledger.cps:
                days = len(LedgerUtils.daily_return_log(ledger))
                actual_participation.append(days)
        
        # With fewer than 20 miners, should return floor regardless of actual participation
        self.assertEqual(result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
    
    def test_exactly_percentile_rank_miners_exact(self):
        """Test with exactly DYNAMIC_MIN_DAYS_PERCENTILE_RANK miners uses exact percentile."""
        # Create exactly DYNAMIC_MIN_DAYS_PERCENTILE_RANK miners with incremental participation days
        percentile_rank = ValiConfig.DYNAMIC_MIN_DAYS_NUM_MINERS
        participation_days = list(range(5, 5 + percentile_rank * 5, 5))  # [5, 10, 15, ..., up to percentile_rank * 5]
        self.assertEqual(len(participation_days), percentile_rank)
        
        miner_participation = {}
        for i, days in enumerate(participation_days):
            miner_participation[f"crypto_trader_{i:03d}"] = {
                "BTCUSD": days,
                "ETHUSD": days
            }
        
        ledger_dict = self.create_production_ledger_dict(miner_participation)
        
        # Verify our ledgers produce expected days
        for i, expected_days in enumerate(participation_days):
            miner_key = f"crypto_trader_{i:03d}"
            btc_ledger = ledger_dict[miner_key]["BTCUSD"]
            self.verify_ledger_produces_expected_days(btc_ledger, expected_days)
        
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO]
        )
        result = result_dict[TradePairCategory.CRYPTO]
        
        # Sorted descending, the DYNAMIC_MIN_DAYS_PERCENTILE_RANK-th element (index percentile_rank-1) = 5, but floor is 7
        sorted_desc = sorted(participation_days, reverse=True)
        expected_raw = sorted_desc[percentile_rank - 1]  # Last element in our case
        expected = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR, expected_raw)
        self.assertEqual(result, expected)
    
    def test_more_than_percentile_rank_miners_exact(self):
        """Test with more than DYNAMIC_MIN_DAYS_PERCENTILE_RANK miners uses exact percentile."""
        # Create more miners than the percentile rank with known participation pattern
        percentile_rank = ValiConfig.DYNAMIC_MIN_DAYS_NUM_MINERS
        # Create 1.5x the percentile rank miners (e.g., 30 if percentile rank is 20)
        total_miners = int(percentile_rank * 1.5)
        
        # Top third: high participation
        top_third = total_miners // 3
        # Middle third: medium participation  
        middle_third = total_miners // 3
        # Bottom third: low participation
        bottom_third = total_miners - top_third - middle_third
        
        participation_days = (
            list(range(90, 90 + top_third))  +     # Top miners: 90+ days
            list(range(40, 40 + middle_third)) +   # Middle miners: 40+ days  
            list(range(10, 10 + bottom_third))     # Bottom miners: 10+ days
        )
        self.assertEqual(len(participation_days), total_miners)
        
        miner_participation = {}
        for i, days in enumerate(participation_days):
            miner_participation[f"crypto_trader_{i:03d}"] = {
                "BTCUSD": days,
                "ETHUSD": days
            }
        
        ledger_dict = self.create_production_ledger_dict(miner_participation)
        
        # Verify all ledgers produce expected days
        for i, expected_days in enumerate(participation_days):
            miner_key = f"crypto_trader_{i:03d}"
            btc_ledger = ledger_dict[miner_key]["BTCUSD"]
            self.verify_ledger_produces_expected_days(btc_ledger, expected_days)
        
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO]
        )
        result = result_dict[TradePairCategory.CRYPTO]
        
        # The AssetSegmentation.aggregate_miner_subledgers() function will aggregate
        # BTCUSD and ETHUSD ledgers for each miner, creating a combined participation
        # We need to get the actual result by testing the actual segmentation
        segmentation = AssetSegmentation(ledger_dict)
        crypto_ledgers = segmentation.segmentation(TradePairCategory.CRYPTO)
        
        # Get actual participation days after aggregation
        actual_participation = []
        for miner_hotkey, ledger in crypto_ledgers.items():
            if ledger and ledger.cps:
                days = len(LedgerUtils.daily_return_log(ledger))
                actual_participation.append(days)
        
        actual_participation.sort(reverse=True)
        
        # DYNAMIC_MIN_DAYS_PERCENTILE_RANK-th element (index percentile_rank-1) after aggregation
        expected_element_index = percentile_rank - 1
        expected_percentile_value = actual_participation[expected_element_index]
        self.assertEqual(result, expected_percentile_value)
    
    def test_floor_ceiling_boundaries_exact(self):
        """Test exact floor and ceiling boundary conditions."""
        # Test floor boundary - create miners with very low participation
        low_participation = list(range(1, 21))  # 1 to 20 days
        miner_participation_low = {}
        for i, days in enumerate(low_participation):
            miner_participation_low[f"low_trader_{i:03d}"] = {
                "BTCUSD": days,
                "ETHUSD": days
            }
        
        ledger_dict_low = self.create_production_ledger_dict(miner_participation_low)
        result_dict_low = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict_low, [TradePairCategory.CRYPTO]
        )
        result_low = result_dict_low[TradePairCategory.CRYPTO]
        
        # DYNAMIC_MIN_DAYS_PERCENTILE_RANK-th percentile would be 1, but should be floored at 7
        self.assertEqual(result_low, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
        
        # Test ceiling boundary - create miners with high participation
        high_participation = list(range(65, 90))  # 65 to 89 days (25 miners)
        miner_participation_high = {}
        for i, days in enumerate(high_participation):
            miner_participation_high[f"high_trader_{i:03d}"] = {
                "BTCUSD": days,
                "ETHUSD": days
            }
        
        ledger_dict_high = self.create_production_ledger_dict(miner_participation_high)
        result_dict_high = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict_high, [TradePairCategory.CRYPTO]
        )
        result_high = result_dict_high[TradePairCategory.CRYPTO]
        
        # DYNAMIC_MIN_DAYS_PERCENTILE_RANK-th percentile would be 69, but should be capped at 60
        # Calculate the actual percentile value dynamically
        sorted_desc = sorted(high_participation, reverse=True)
        percentile_rank = ValiConfig.DYNAMIC_MIN_DAYS_NUM_MINERS
        raw_percentile_value = sorted_desc[percentile_rank - 1] if len(sorted_desc) >= percentile_rank else min(sorted_desc)
        expected_high = min(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL, raw_percentile_value)
        self.assertEqual(result_high, expected_high)
    
    def test_different_asset_classes_exact(self):
        """Test that different asset classes return exact different values."""
        # Create miners with mixed participation across different assets
        miner_participation = {
            # Crypto specialists (high participation)
            "crypto_001": {"BTCUSD": 80, "ETHUSD": 75},
            "crypto_002": {"BTCUSD": 70, "ETHUSD": 65},
            "crypto_003": {"BTCUSD": 60, "ETHUSD": 55},
            
            # Crypto alts specialists (medium participation)
            "crypto_alt_001": {"SOLUSD": 50, "XRPUSD": 45},
            "crypto_alt_002": {"SOLUSD": 40, "XRPUSD": 35},
            "crypto_alt_003": {"SOLUSD": 30, "XRPUSD": 25},
            
            # Forex G1 specialists (low participation)
            "forex_g1_001": {"EURUSD": 25, "GBPUSD": 20},
            "forex_g1_002": {"EURUSD": 15, "GBPUSD": 12},
            "forex_g1_003": {"EURUSD": 10, "GBPUSD": 8},
            
            # Mixed traders
            "mixed_001": {"BTCUSD": 45, "SOLUSD": 40, "EURUSD": 35},
            "mixed_002": {"ETHUSD": 35, "XRPUSD": 30, "GBPUSD": 25}
        }
        
        ledger_dict = self.create_production_ledger_dict(miner_participation)
        
        # Calculate all asset classes at once
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO, TradePairCategory.FOREX]
        )

        # Extract results for each asset class
        crypto_result = result_dict[TradePairCategory.CRYPTO]
        forex_result = result_dict[TradePairCategory.FOREX]

        # With fewer than 20 miners in each asset class, all should return floor value
        self.assertEqual(crypto_result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
        self.assertEqual(forex_result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)

        # All results should be within bounds
        for result in [crypto_result, forex_result]:
            self.assertGreaterEqual(result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
            self.assertLessEqual(result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL)
    
    def test_production_trade_pair_categorization(self):
        """Test that actual TradePair categorization works correctly."""
        # Verify our understanding of trade pair categories
        self.assertEqual(TradePair.BTCUSD.trade_pair_category, TradePairCategory.CRYPTO)
        self.assertEqual(TradePair.ETHUSD.trade_pair_category, TradePairCategory.CRYPTO)
        self.assertEqual(TradePair.SOLUSD.trade_pair_category, TradePairCategory.CRYPTO)
        self.assertEqual(TradePair.EURUSD.trade_pair_category, TradePairCategory.FOREX)
        self.assertEqual(TradePair.GBPUSD.trade_pair_category, TradePairCategory.FOREX)
        self.assertEqual(TradePair.USDJPY.trade_pair_category, TradePairCategory.FOREX)
        
        # Create miners with specific trade pairs
        miner_participation = {
            "btc_trader": {"BTCUSD": 50},  # Crypto
            "sol_trader": {"SOLUSD": 40},  # Crypto
            "eur_trader": {"EURUSD": 30},  # Forex
            "jpy_trader": {"USDJPY": 20},  # Forex
        }

        ledger_dict = self.create_production_ledger_dict(miner_participation)

        # Test each asset class finds exactly the expected miners
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO, TradePairCategory.FOREX]
        )

        crypto_result = result_dict[TradePairCategory.CRYPTO]
        # Both btc_trader and sol_trader participate, but with fewer than 20 miners, should return floor
        self.assertEqual(crypto_result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)

        forex_result = result_dict[TradePairCategory.FOREX]
        # Both eur_trader and jpy_trader participate, but with fewer than 20 miners, should return floor
        self.assertEqual(forex_result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
    
    def test_realistic_production_scenario(self):
        """Test with a realistic production-like scenario with exact expectations."""
        # Simulate realistic miner distribution
        miner_participation = {}
        
        # Top-tier crypto major traders (5 miners, 80-100 days)
        for i in range(5):
            miner_participation[f"top_crypto_{i}"] = {
                "BTCUSD": 100 - i * 2,  # 100, 98, 96, 94, 92
                "ETHUSD": 95 - i * 2    # 95, 93, 91, 89, 87
            }
        
        # Mid-tier crypto major traders (10 miners, 40-60 days)
        for i in range(10):
            miner_participation[f"mid_crypto_{i}"] = {
                "BTCUSD": 60 - i * 2,   # 60, 58, 56, ..., 42
                "ETHUSD": 55 - i * 2    # 55, 53, 51, ..., 37
            }
        
        # New crypto major traders (10 miners, 10-30 days)
        for i in range(10):
            miner_participation[f"new_crypto_{i}"] = {
                "BTCUSD": 30 - i * 2,   # 30, 28, 26, ..., 12
                "ETHUSD": 25 - i * 2    # 25, 23, 21, ..., 7
            }
        
        ledger_dict = self.create_production_ledger_dict(miner_participation)
        
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO]
        )
        result = result_dict[TradePairCategory.CRYPTO]
        
        # The AssetSegmentation.aggregate_miner_subledgers() function will aggregate
        # BTCUSD and ETHUSD ledgers for each miner, creating a combined participation
        # We need to calculate the expected result by testing the actual segmentation
        
        segmentation = AssetSegmentation(ledger_dict)
        crypto_ledgers = segmentation.segmentation(TradePairCategory.CRYPTO)
        
        # Get actual participation days after aggregation
        actual_participation = []
        for miner_hotkey, ledger in crypto_ledgers.items():
            if ledger and ledger.cps:
                days = len(LedgerUtils.daily_return_log(ledger))
                actual_participation.append(days)
        
        actual_participation.sort(reverse=True)
        
        # Should have 25 miners (5 top + 10 mid + 10 new)
        expected_total_miners = 25
        self.assertEqual(len(actual_participation), expected_total_miners)
        
        # DYNAMIC_MIN_DAYS_PERCENTILE_RANK-th element (index percentile_rank-1) after aggregation
        percentile_rank = ValiConfig.DYNAMIC_MIN_DAYS_NUM_MINERS
        if len(actual_participation) >= percentile_rank:
            expected_percentile = actual_participation[percentile_rank - 1]
        else:
            expected_percentile = min(actual_participation) if actual_participation else ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR
        expected_final = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR, min(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL, expected_percentile))
        self.assertEqual(result, expected_final)
    
    def test_exception_handling_exact(self):
        """Test that invalid data is gracefully handled and treated as no participants."""
        # Create invalid ledger structure - these entries will be filtered out gracefully
        invalid_ledger_dict = {
            "miner_001": None,  # Invalid structure (filtered out)
            "miner_002": "not_a_dict",  # Invalid type (filtered out)
        }

        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            invalid_ledger_dict, [TradePairCategory.CRYPTO]
        )

        # Invalid entries are filtered out, resulting in 0 participants -> returns floor
        self.assertEqual(result_dict[TradePairCategory.CRYPTO], ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
    
    def test_production_asset_segmentation_integration(self):
        """Test integration with production AssetSegmentation logic."""
        # Create comprehensive ledger structure
        miner_participation = {
            "diversified_trader": {
                "BTCUSD": 50,   # Crypto major
                "SOLUSD": 45,   # Crypto alt  
                "EURUSD": 40,   # Forex G1
                "USDJPY": 35,   # Forex G2
                "GBPAUD": 30,   # Forex G4
            },
            "crypto_only": {
                "BTCUSD": 60,
                "ETHUSD": 55,
                "SOLUSD": 50,
                "XRPUSD": 45,
            },
            "forex_only": {
                "EURUSD": 70,
                "GBPUSD": 65,
                "USDJPY": 60,
                "USDCHF": 55,
            }
        }
        
        ledger_dict = self.create_production_ledger_dict(miner_participation)
        
        # Verify AssetSegmentation correctly segments each asset class
        segmentation = AssetSegmentation(ledger_dict)
        
        # Test crypto segmentation
        crypto_ledgers = segmentation.segmentation(TradePairCategory.CRYPTO)
        
        # Should include diversified_trader and crypto_only, not forex_only
        self.assertIn("diversified_trader", crypto_ledgers)
        self.assertIn("crypto_only", crypto_ledgers) 
        self.assertIn("forex_only", crypto_ledgers)  # forex_only doesn't have crypto, so empty ledger
        
        # Verify the segmented ledgers have data
        self.assertIsInstance(crypto_ledgers["diversified_trader"], PerfLedger)
        self.assertIsInstance(crypto_ledgers["crypto_only"], PerfLedger)
        
        # Test the dynamic minimum calculation uses this segmentation
        result_dict = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            ledger_dict, [TradePairCategory.CRYPTO]
        )
        result = result_dict[TradePairCategory.CRYPTO]
        
        # Should be within expected bounds
        self.assertGreaterEqual(result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR)
        self.assertLessEqual(result, ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL)


if __name__ == '__main__':
    unittest.main()
