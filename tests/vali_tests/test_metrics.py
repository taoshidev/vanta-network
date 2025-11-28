import copy
import math
import random
import time


import numpy as np

from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.test_utilities import create_daily_checkpoints_with_pnl
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, PerfCheckpoint
from vali_objects.vali_config import ValiConfig



class TestMetrics(TestBase):
    def test_return_no_positions(self):
        self.assertEqual(Metrics.base_return([]), 0.0)

    def test_negative_returns(self):
        """Test that the returns scoring function works properly for only negative returns"""
        log_returns = [-0.2, -0.1, -0.3, -0.2, -0.1, -0.3]

        base_return = Metrics.base_return(log_returns)
        self.assertLess(base_return, 0.0)

    def test_positive_returns(self):
        """Test that the returns scoring function works properly for only positive returns"""
        log_returns = [0.2, 0.1, 0.3, 0.2, 0.1, 0.3]

        base_return = Metrics.base_return(log_returns)
        self.assertGreater(base_return, 0.0)

    def test_typical_omega(self):
        """Test that the omega function works as expected for only positive returns"""
        log_returns = [0.003, -0.002] * 50  # Sum = 0.015
        omega = Metrics.omega(log_returns)

        # Should always be greater or equal to 0
        self.assertGreater(omega, 0.0)

    def test_negative_omega(self):
        """Test that the omega function works as expected for only negative returns"""
        log_returns = [-0.003, -0.001] * 50  # Sum = -0.06
        omega = Metrics.omega(log_returns)

        # Should always be less or equal to 0
        self.assertEqual(omega, 0.0)

    def test_positive_omega(self):
        """Test that the omega function works as expected for only positive returns - the default loss works"""
        log_returns = [0.002, 0.001] * 50  # Sum = 0.045
        omega = Metrics.omega(log_returns)

        # Should always be greater or equal to 0, cannot be massive
        self.assertGreater(omega, 0.0)
        self.assertLess(omega, np.inf)

    def test_positive_omega_small_loss(self):
        """Test that the omega function works as expected for only positive returns - the default loss works"""
        log_returns = [0.002, -0.001, 0.001] * 50  # Sum = 0.01
        omega = Metrics.omega(log_returns)

        # Should always be greater or equal to 0, cannot be massive
        self.assertGreater(omega, 0.0)
        self.assertLess(omega, np.inf)

    def test_omega_no_returns(self):
        """Test that the omega function works as expected for no returns"""
        returns = []

        omega = Metrics.omega(returns)

        # Expected value is zero
        self.assertEqual(omega, ValiConfig.OMEGA_NOCONFIDENCE_VALUE)
    def test_omega_weighting(self):
        #No positive returns means empty numerator
        returns = [-0.01] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL + 1)
        omega = Metrics.omega(returns, weighting=True)
        #Numerator is 0 so will be 0
        self.assertEqual(omega, 0)

        #No negative returns means empty denominator
        returns = [0.01] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL + 1)
        omega = Metrics.omega(returns, weighting=True)
        #Should be small in magnitude, but positive
        self.assertGreater(omega, 0)





    def test_sharpe_positive(self):
        """Test that the sharpe function is positive when all returns are positive"""
        log_returns = [0.002] * 100  # Sum = 0.06
        sharpe = Metrics.sharpe(log_returns)

        # Should always be greater or equal to 0
        self.assertGreater(sharpe, 0.0)

    def test_sharpe_no_returns_no_variance(self):
        """Test that the sharpe function is 0 with 0 returns"""
        log_returns = [0.0] * 100  # Sum = 0.0
        sharpe = Metrics.sharpe(log_returns)

        # Expected value is zero
        self.assertLess(sharpe, 0.0)

    def test_sharpe_no_returns(self):
        """Test that the sharpe function works as expected"""
        returns = []

        sharpe = Metrics.sharpe(returns)

        # Expected value is zero
        self.assertEqual(sharpe, ValiConfig.SHARPE_NOCONFIDENCE_VALUE)

    def test_sharpe_perfect_positive_year(self):
        """Test that the sharpe function works for 365 days of returns"""
        log_returns = [0.10/ValiConfig.DAYS_IN_YEAR_CRYPTO for _ in range(ValiConfig.DAYS_IN_YEAR_CRYPTO)]

        sharpe = Metrics.sharpe(log_returns)

        # Expected value is between zero and 10
        self.assertGreater(sharpe, 0.0)
        self.assertLess(sharpe, 10)

    def test_sortino_no_returns(self):
        """Test that the Sortino function returns 0.0 when there are no returns"""
        log_returns = []

        sortino = Metrics.sortino(log_returns)

        # Expected value is zero
        self.assertEqual(sortino, ValiConfig.SORTINO_NOCONFIDENCE_VALUE)

    def test_sortino_noconfidence_limit(self):
        """Test that the Sortino function returns 0.0 when there are no returns"""
        log_returns = [0.1] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL - 1)

        sortino = Metrics.sortino(log_returns)

        # Expected value is zero
        self.assertEqual(sortino, ValiConfig.SORTINO_NOCONFIDENCE_VALUE)

    def test_sortino_no_returns_no_variance(self):
        """Test that the Sortino function returns 0.0 when there are no returns and no variance"""
        log_returns = [0.0] * 100
        sortino = Metrics.sortino(log_returns)

        # Expected value will be the annual tbill rate, if there is no variance
        self.assertAlmostEqual(sortino, -ValiConfig.ANNUAL_RISK_FREE_PERCENTAGE)

    def test_sortino_no_losses(self):
        """Test that the Sortino function returns 0.0 when there are no losses"""
        log_returns = [0.002] * 100
        sortino = Metrics.sortino(log_returns)

        # Expected value is zero, as there are no losses and volatility is inf
        self.assertEqual(sortino, 0.0)

    def test_sortino_only_losses(self):
        """Test that the Sortino function returns 0.0 when there are no losses"""
        log_returns = [-0.002] * 100

        downside_volatility = Metrics.ann_downside_volatility(log_returns)
        volatility = Metrics.ann_volatility(log_returns)

        self.assertEqual(downside_volatility, volatility)

    def test_sortino_general(self):
        """Test that the Sortino function returns a positive value for general returns"""
        log_returns = [0.003, -0.002] * 50
        sortino = Metrics.sortino(log_returns)
        self.assertGreater(sortino, 0.0)

    def test_sortino_negative(self):
        """Test that the Sortino function returns a negative value for negative log returns"""
        log_returns = [-0.001, 0.001] * 60
        log_returns.append(-0.015)
        sortino = Metrics.sortino(log_returns)

        # Expected value less than zero for negative returns
        self.assertLess(sortino, 0.0)
        self.assertGreater(sortino, -10)

    def test_statistical_confidence_no_returns(self):
        """Test that the statistical confidence function returns 0.0 when there are no returns"""
        log_returns = []

        confidence = Metrics.statistical_confidence(log_returns)

        # Expected value is zero
        self.assertEqual(confidence, ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE)

    def test_statistical_confidence_noconfidence_limit(self):
        """Test that the statistical confidence function returns 0.0 when there are no returns"""
        log_returns = [0.1] * (ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL - 1)

        confidence = Metrics.statistical_confidence(log_returns)

        # Expected value is zero
        self.assertEqual(confidence, ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE)

    def test_statistical_confidence_general(self):
        """Test that the statistical confidence function returns a positive value for general returns"""
        log_returns = [0.003, -0.002] * 50
        confidence = Metrics.statistical_confidence(log_returns)
        self.assertGreater(confidence, 0.0)

    def test_statistical_confidence_negative(self):
        """Test that the statistical confidence function returns a negative value for negative log returns"""
        log_returns = [0.001, -0.002] * 50
        confidence = Metrics.statistical_confidence(log_returns)

        # Expected value less than zero for negative returns
        self.assertLess(confidence, 0.0)

    def test_statistical_confidence_monotonic(self):
        """Test that the statistical confidence function is monotonic"""
        log_returns = [0.003, -0.002] * 50
        confidence = Metrics.statistical_confidence(log_returns)

        log_returns = [0.003, -0.002] * 70
        confidence_new = Metrics.statistical_confidence(log_returns)

        log_returns = [0.003, -0.002] * 100
        confidence_newest = Metrics.statistical_confidence(log_returns)

        self.assertLess(confidence, confidence_new)
        self.assertLess(confidence_new, confidence_newest)

    def test_ann_volatility(self):
        a = [9/252, 10/252, 11/252]
        b = [8/252, 10/252, 12/252]
        self.assertGreater(Metrics.ann_volatility(b), Metrics.ann_volatility(a))

    def test_ann_downside_volatility(self):
        a = [9/252, 10/252, 11/252]
        b = [8/252, 10/252, 12/252]
        self.assertEqual(Metrics.ann_downside_volatility(b), Metrics.ann_downside_volatility(a))
        self.assertEqual(Metrics.ann_downside_volatility(a), np.inf)

        c = [-9/252, -10/252, -11/252]
        d = [-8/252, -10/252, -12/252]
        self.assertGreater(Metrics.ann_downside_volatility(d), Metrics.ann_downside_volatility(c))

        e = copy.deepcopy(c)
        e.append(12/252)
        self.assertEqual(Metrics.ann_downside_volatility(e), Metrics.ann_downside_volatility(c))

    def test_weighting(self):
        log_returns = []
        empty_distribution = Metrics.weighting_distribution(log_returns).tolist()
        self.assertEqual(log_returns, empty_distribution)

        self.assertEqual(log_returns, Metrics.weighted_log_returns(log_returns))


        log_returns = [float('-inf')]
        one_entry_dist = Metrics.weighting_distribution(log_returns).tolist()
        self.assertEqual([1], one_entry_dist)

        #Test distribution
        log_returns = [random.uniform(-1, 1) for _ in range(ValiConfig.TARGET_LEDGER_WINDOW_DAYS)]
        large_distribution = Metrics.weighting_distribution(log_returns).tolist()
        for value in large_distribution:
            self.assertGreaterEqual(value, ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN)
            self.assertLessEqual(value, ValiConfig.WEIGHTED_AVERAGE_DECAY_MAX)
        #first day should be minimum
        self.assertAlmostEqual(large_distribution[0], ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN, places=2)
        #most recent day should be maximum
        self.assertAlmostEqual(large_distribution[-1], ValiConfig.WEIGHTED_AVERAGE_DECAY_MAX, places=2)

    def test_variance(self):
        log_returns = []
        self.assertEqual(0, Metrics.variance(log_returns))

        #Case for degrees of freedom
        log_returns = [1]
        self.assertEqual(np.inf, Metrics.variance(log_returns))

        log_returns = [1, 2]
        self.assertNotEqual(np.inf, Metrics.variance(log_returns))
        self.assertNotEqual(0, Metrics.variance(log_returns))

    def test_average(self):
        log_returns = []
        self.assertEqual(0, Metrics.average(log_returns))
        self.assertEqual(0, Metrics.average(log_returns, weighting=True))
        log_returns = [0.01, -20, 0.01]

        #The negative returns shouldn't be included in the average
        self.assertGreaterEqual(Metrics.average(log_returns, weighting=True, indices=[0, 2]), 0)

        #Without indices it should be negative
        self.assertLessEqual(Metrics.average(log_returns, weighting=True, indices=None), 0)

    def test_daily_max_drawdown_empty(self):
        """Test that daily_max_drawdown returns 0.0 for empty input"""
        # Test with empty list
        self.assertEqual(Metrics.daily_max_drawdown([]), 0.0)

    def test_daily_max_drawdown_constant(self):
        """Test that daily_max_drawdown returns 0.0 for constant returns"""
        # All zeros
        log_returns = [0.0] * 10
        self.assertEqual(Metrics.daily_max_drawdown(log_returns), 0.0)

        # All same positive value
        log_returns = [0.01] * 10
        self.assertEqual(Metrics.daily_max_drawdown(log_returns), 0.0)

    def test_daily_max_drawdown_monotonic(self):
        """Test daily_max_drawdown with monotonically increasing/decreasing returns"""
        # Monotonically increasing returns should have zero drawdown
        log_returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        self.assertEqual(Metrics.daily_max_drawdown(log_returns), 0.0)

        # Monotonically decreasing returns should have drawdown
        log_returns = [0.05, -0.02, -0.03, -0.04, -0.05]
        mdd = Metrics.daily_max_drawdown(log_returns)
        self.assertGreater(mdd, 0.0)

    def test_daily_max_drawdown_recovery(self):
        """Test daily_max_drawdown with a pattern that drops and recovers"""
        # Peak, drawdown, and recovery
        log_returns = [0.1, 0.1, -0.15, -0.1, 0.05, 0.2, 0.1]
        mdd = Metrics.daily_max_drawdown(log_returns)

        # Should have a significant drawdown
        self.assertGreater(mdd, 0.0)
        # But less than 100%
        self.assertLess(mdd, 1.0)

    def test_daily_max_drawdown_detailed(self):
        """Test daily_max_drawdown with a specific pattern to confirm exact calculation"""
        # Design a specific scenario with known drawdown
        # Initial value 100, goes to 110, drops to 77, recovers to 99
        # Drawdown should be (77/110) - 1 = -30%
        log_returns = [
            math.log(1.1),  # Day 1: +10% (value = 110)
            math.log(0.7),  # Day 2: -30% (value = 77)
            math.log(1.1),  # Day 3: +10% (value = 84.7)
            math.log(1.1),  # Day 4: +10% (value = 93.17)
            math.log(1.06),  # Day 5: +6% (value = 98.76)
        ]

        mdd = Metrics.daily_max_drawdown(log_returns)
        self.assertAlmostEqual(mdd, 0.3, delta=0.01,
                              msg="Maximum drawdown should be approximately 30%")

    def test_daily_max_drawdown_all_negative(self):
        """Test daily_max_drawdown with all negative returns"""
        log_returns = [-0.01, -0.02, -0.03, -0.04, -0.05]
        mdd = Metrics.daily_max_drawdown(log_returns)

        # Should have a significant drawdown
        self.assertGreater(mdd, 0.0)
        # For small negative returns, drawdown should be less than 15%
        self.assertLess(mdd, 0.15)

    def test_daily_max_drawdown_extreme(self):
        """Test daily_max_drawdown with extreme market crash scenario"""
        # Simulate a market crash with -50% in one day
        log_returns = [0.01, 0.02, math.log(0.5), 0.01, 0.02]
        mdd = Metrics.daily_max_drawdown(log_returns)

        # Max drawdown should be close to 50%
        self.assertAlmostEqual(mdd, 0.5, delta=0.02)

    def test_pnl_score_empty_ledger(self):
        """Test pnl_score with empty or None ledger"""
        # Test with None ledger
        result = Metrics.pnl_score([], None)
        self.assertEqual(result, ValiConfig.PNL_NOCONFIDENCE_VALUE)
        
        # Test with empty ledger
        empty_ledger = PerfLedger(cps=[])
        result = Metrics.pnl_score([], empty_ledger)
        self.assertEqual(result, ValiConfig.PNL_NOCONFIDENCE_VALUE)

    def test_pnl_score_no_daily_pnl(self):
        """Test pnl_score when daily_pnl returns empty list"""
        # Create a ledger with incomplete checkpoints that won't count as complete days
        current_time_ms = int(time.time() * 1000)
        ledger = PerfLedger(cps=[
            PerfCheckpoint(
                last_update_ms=current_time_ms,
                prev_portfolio_ret=1.0,
                accum_ms=100,  # Partial accumulation - won't count as complete day
                pnl_gain=10,
                pnl_loss=0,
                gain=0.01,
                loss=0.0,
                mdd=0.95
            )
        ])
        
        result = Metrics.pnl_score([], ledger)
        self.assertEqual(result, ValiConfig.PNL_NOCONFIDENCE_VALUE)

    def test_pnl_score_with_positive_pnl(self):
        """Test pnl_score with positive daily PnL values"""
        # Create ledger with positive PnL values
        realized_pnl = [100.0, 150.0, 120.0]
        unrealized_pnl = [0.0, 0.0, 80.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        result = Metrics.pnl_score([], ledger, weighting=False)

        # Should return average of the daily PnL values
        expected = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(result, expected)

    def test_pnl_score_with_negative_pnl(self):
        """Test pnl_score with negative daily PnL values"""
        # Create ledger with negative PnL values
        realized_pnl = [-50.0, -75.0, -15.0]
        unrealized_pnl = [0.0, 0.0, -10.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        result = Metrics.pnl_score([], ledger, weighting=False)

        # Should return average of the daily PnL values
        expected = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(result, expected)

    def test_pnl_score_with_mixed_pnl(self):
        """Test pnl_score with mixed positive and negative daily PnL values"""
        # Create ledger with mixed PnL values
        realized_pnl = [100.0, -50.0, 75.0, -15.0]
        unrealized_pnl = [0.0, 0.0, 0.0, -10.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        result = Metrics.pnl_score([], ledger, weighting=False)

        # Should return average of the daily PnL values
        expected = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(result, expected)

    def test_pnl_score_with_zero_pnl(self):
        """Test pnl_score with zero daily PnL values"""
        # Create ledger with zero PnL values
        realized_pnl = [0.0, 0.0, 0.0]
        unrealized_pnl = [0.0, 0.0, 0.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        result = Metrics.pnl_score([], ledger, weighting=False)
        
        # Should return 0.0
        self.assertEqual(result, 0.0)

    def test_pnl_score_without_weighting(self):
        """Test pnl_score without time weighting"""
        # Create ledger with specific PnL values
        realized_pnl = [10.0, 20.0, 18.0]
        unrealized_pnl = [0.0, 0.0, 12.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        result = Metrics.pnl_score([], ledger, weighting=False)

        # Should return average of the daily PnL values
        expected = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(result, expected)

    def test_pnl_score_with_weighting(self):
        """Test pnl_score with time weighting enabled"""
        # Create ledger with ascending PnL pattern to test time weighting
        realized_pnl = [10.0, 20.0, 18.0]
        unrealized_pnl = [0.0, 0.0, 12.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        weighted_result = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_result = Metrics.pnl_score([], ledger, weighting=False)

        # Unweighted should be the simple average
        expected_unweighted = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(unweighted_result, expected_unweighted)

        # Weighted should be higher than unweighted due to recent higher values
        self.assertGreater(weighted_result, unweighted_result)

    def test_pnl_score_single_day(self):
        """Test pnl_score with single day of PnL data"""
        # Create ledger with single day of PnL data
        realized_pnl = [25.5]
        unrealized_pnl = [17.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        result = Metrics.pnl_score([], ledger, weighting=False)

        # Should return the single value
        self.assertEqual(result, 25.5)

    def test_pnl_score_large_pnl_values(self):
        """Test pnl_score with large PnL values"""
        # Create ledger with large PnL values
        realized_pnl = [10000.0, 15000.0, 7200.0]
        unrealized_pnl = [0.0, 0.0, 4800.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        result = Metrics.pnl_score([], ledger, weighting=False)

        # Should handle large values correctly
        expected = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(result, expected)

    def test_pnl_score_bypass_confidence_parameter(self):
        """Test that pnl_score accepts bypass_confidence parameter (though it may not use it)"""
        # Create ledger with specific PnL values
        realized_pnl = [50.0, 36.0]
        unrealized_pnl = [0.0, 24.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        # Test with bypass_confidence=True
        result1 = Metrics.pnl_score([], ledger, bypass_confidence=True, weighting=False)

        # Test with bypass_confidence=False
        result2 = Metrics.pnl_score([], ledger, bypass_confidence=False, weighting=False)

        # Should return same result regardless (as noted in function comment)
        expected = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(result1, expected)
        self.assertEqual(result2, expected)

    def test_pnl_score_log_returns_parameter_unused(self):
        """Test that pnl_score doesn't use the log_returns parameter"""
        # Create ledger with specific PnL values
        realized_pnl = [30.0, 24.0]
        unrealized_pnl = [0.0, 16.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        # Test with different log_returns values - should not affect result
        result1 = Metrics.pnl_score([0.1, 0.2, 0.3], ledger, weighting=False)
        result2 = Metrics.pnl_score([], ledger, weighting=False)
        result3 = Metrics.pnl_score([100, 200], ledger, weighting=False)

        # All should return same result since log_returns is unused
        expected = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertEqual(result1, expected)
        self.assertEqual(result2, expected)
        self.assertEqual(result3, expected)

    def test_pnl_score_time_weighted_recent_low_reduces_historical_high(self):
        """Test that recent low PnL brings down historically large PnL with time weighting"""
        # Pattern: [1000, 1000, 1000, 1000, -500, -500, -500]
        # Recent values should have higher weight and bring down the average
        realized_pnl = [1000.0, 1000.0, 1000.0, 1000.0, -500.0, -500.0, -300.0]
        unrealized_pnl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -200.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        # Calculate both weighted and unweighted scores
        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)

        # Unweighted average should be: (4*1000 + 3*(-500)) / 7 = 2500/7 ≈ 357.14
        expected_unweighted = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertAlmostEqual(unweighted_score, expected_unweighted, places=1)
        
        # With time weighting, recent negative values should reduce the score significantly
        self.assertLess(weighted_score, unweighted_score,
                       "Time-weighted score should be lower due to recent poor performance")
        
        # The weighted score should be lower than unweighted
        self.assertLess(weighted_score, expected_unweighted,
                       "Time weighting should  reduce score due to recent losses")

    def test_pnl_score_time_weighted_recent_high_increases_score(self):
        """Test that recent high PnL increases score with historically low PnL"""
        # Pattern: [-500, -500, -500, -500, 1000, 1000, 1000]
        realized_pnl = [-500.0, -500.0, -500.0, -500.0, 1000.0, 1000.0, 600.0]
        unrealized_pnl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        # Calculate both weighted and unweighted scores
        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)

        # Unweighted average should be: (4*(-500) + 3*1000) / 7 = 1000/7 ≈ 142.86
        expected_unweighted = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertAlmostEqual(unweighted_score, expected_unweighted, places=1)
        
        # With time weighting, recent positive values should increase the score
        self.assertGreater(weighted_score, unweighted_score,
                          "Time-weighted score should be higher due to recent strong performance")

    def test_pnl_score_time_weighted_consistent_pattern(self):
        """Test time weighting with consistent PnL pattern"""
        # Consistent positive PnL pattern
        realized_pnl = [100.0] * 9 + [100.0]
        unrealized_pnl = [0.0] * 9 + [40.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        # With consistent values, weighted and unweighted should be very similar
        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)
        
        # Both should equal 100.0
        self.assertAlmostEqual(weighted_score, 100.0, places=1)
        self.assertAlmostEqual(unweighted_score, 100.0, places=1)
        self.assertAlmostEqual(weighted_score, unweighted_score, places=1)

    def test_pnl_score_time_weighted_gradual_decline(self):
        """Test time weighting with gradual PnL decline"""
        # Gradual decline: [500, 400, 300, 200, 100, 0, -100]
        realized_pnl = [500.0, 400.0, 300.0, 200.0, 100.0, 0.0, -60.0]
        unrealized_pnl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -40.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)

        # Unweighted average: sum(pnl_pattern) / len = 1300/7 ≈ 185.71
        expected_unweighted = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertAlmostEqual(unweighted_score, expected_unweighted, places=1)
        
        # With time weighting, recent lower values should reduce the weighted average
        self.assertLess(weighted_score, unweighted_score,
                       "Time weighting should emphasize recent decline")

    def test_pnl_score_time_weighted_volatile_pattern(self):
        """Test time weighting with highly volatile PnL pattern"""
        # Volatile pattern: [1000, -800, 600, -400, 200, -100, 50]
        realized_pnl = [1000.0, -800.0, 600.0, -400.0, 200.0, -100.0, 30.0]
        unrealized_pnl = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)
        
        # Both scores should be finite numbers
        self.assertNotEqual(weighted_score, float('inf'))
        self.assertNotEqual(weighted_score, float('-inf'))
        self.assertNotEqual(unweighted_score, float('inf'))
        self.assertNotEqual(unweighted_score, float('-inf'))
        
        # Verify the function handles volatile data correctly
        self.assertTrue(isinstance(weighted_score, (int, float)))
        self.assertTrue(isinstance(unweighted_score, (int, float)))

    def test_pnl_score_time_weighted_extreme_recent_values(self):
        """Test time weighting with extreme recent values"""
        # Pattern with extreme recent loss: [100, 100, 100, 100, 100, -10000]
        realized_pnl = [100.0, 100.0, 100.0, 100.0, 100.0, -6000.0]
        unrealized_pnl = [0.0, 0.0, 0.0, 0.0, 0.0, -4000.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)
        
        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)

        # Unweighted: (5*100 + 1*(-10000)) / 6 = -9500/6 ≈ -1583.33
        expected_unweighted = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertAlmostEqual(unweighted_score, expected_unweighted, places=1)
        
        # Weighted score should be even more negative due to recency weighting
        self.assertLess(weighted_score, unweighted_score,
                       "Extreme recent loss should be heavily weighted")

    def test_pnl_score_time_weighted_single_value(self):
        """Test time weighting with single PnL value"""
        realized_pnl = [150.0]
        unrealized_pnl = [100.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)

        # With single value, both should return the same result
        self.assertEqual(weighted_score, 150.0)
        self.assertEqual(unweighted_score, 150.0)
        self.assertEqual(weighted_score, unweighted_score)

    def test_pnl_score_time_weighted_large_dataset(self):
        """Test time weighting with large PnL dataset"""
        # Create large dataset with declining trend
        # First 20 days: positive, last 10 days: negative
        realized_pnl = [50.0] * 20 + [-100.0] * 9 + [-60.0]
        unrealized_pnl = [0.0] * 20 + [0.0] * 9 + [-40.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        weighted_score = Metrics.pnl_score([], ledger, weighting=True)
        unweighted_score = Metrics.pnl_score([], ledger, weighting=False)

        # Unweighted: (20*50 + 10*(-100)) / 30 = 0/30 = 0
        expected_unweighted = sum([r + min(0.0, u) for r, u in zip(realized_pnl, unrealized_pnl)]) / len(realized_pnl)
        self.assertAlmostEqual(unweighted_score, expected_unweighted, places=1)

        # Time weighting should make the score more negative due to recent losses
        self.assertLess(weighted_score, unweighted_score,
                       "Recent negative values should dominate with time weighting")

    def test_pnl_score_70_percent_weight_in_first_30_days(self):
        """Test that roughly 70% of the weight for PnL metric is in the most recent 30 days"""
        # Create a 100-day dataset where:
        # - First 70 days have PnL = 0
        # - Last 30 days have PnL = 100
        # If ~70% of weight is in the last 30 days, weighted average should be ~70
        realized_pnl = [0.0] * 90 + [100.0] * 29 + [100.0]
        unrealized_pnl = [0.0] * 90 + [0.0] * 29 + [40.0]
        ledger = create_daily_checkpoints_with_pnl(realized_pnl, unrealized_pnl)

        weighted_score = Metrics.pnl_score([], ledger, weighting=True)

        # With 70% weight on recent 30 days (value=100) and 30% on older 70 days (value=0):
        # Expected weighted score ≈ 70
        # Using a tolerance of ±5 to account for the decay distribution
        self.assertAlmostEqual(weighted_score, 70.0, delta=1.0,
                             msg=f"Expected weighted score ~70 if 70% weight in recent 30 days, got {weighted_score:.2f}")
