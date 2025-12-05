
import math
import numpy as np
from scipy.stats import ttest_1samp
from typing import Union

#from vali_objects.utils.contract_manager import CollateralRecord
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger, TP_ID_PORTFOLIO

class Metrics:

    @staticmethod
    def log_risk_free_rate(days_in_year: int) -> float:
        if days_in_year is None or days_in_year <= 0:
            return np.inf
        return math.log(1 + ValiConfig.ANNUAL_RISK_FREE_DECIMAL) / days_in_year

    @staticmethod
    def weighted_log_returns(log_returns: list[float]) -> list[float]:
        if len(log_returns) < 1:
            return []

        weighting_distribution = Metrics.weighting_distribution(log_returns)
        weighted_returns = np.multiply(np.array(log_returns), weighting_distribution) / np.sum(weighting_distribution)
        return list(weighted_returns)

    @staticmethod
    def weighting_distribution(log_returns: Union[list[float], np.ndarray],
                               max_weight: float = ValiConfig.WEIGHTED_AVERAGE_DECAY_MAX,
                               min_weight: float = ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN,
                               decay_rate: float = ValiConfig.WEIGHTED_AVERAGE_DECAY_RATE) -> np.ndarray:
        """
        Returns the weighting distribution that decays from max_weight to min_weight
        using the configured decay rate
        """

        if len(log_returns) < 1:
            return np.ones(0)

        weighting_distribution_days = np.arange(0, len(log_returns))

        # Calculate decay from max to min
        weight_range = max_weight - min_weight
        decay_values = min_weight + (weight_range * np.exp(-decay_rate * weighting_distribution_days))

        return decay_values[::-1][-len(log_returns):]

    @staticmethod
    def average(log_returns: Union[list[float], np.ndarray], weighting=False,
                indices: Union[list[int], None] = None,
                max_weight: float = ValiConfig.WEIGHTED_AVERAGE_DECAY_MAX,
                min_weight: float = ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN,
                decay_rate: float = ValiConfig.WEIGHTED_AVERAGE_DECAY_RATE) -> float:
        """
        Returns the mean of the log returns
        """
        if len(log_returns) == 0:
            return 0.0

        weighting_distribution = Metrics.weighting_distribution(log_returns, max_weight=max_weight, min_weight=min_weight, decay_rate=decay_rate)

        if indices is not None and len(indices) != 0:
            indices = [i for i in indices if i in range(len(log_returns))]
            log_returns = np.array(log_returns)[indices]
            weighting_distribution = np.array(weighting_distribution)[indices]

        if weighting:
            avg_value = np.average(log_returns, weights=weighting_distribution)
        else:
            avg_value = np.mean(log_returns)

        return float(avg_value)

    @staticmethod
    def variance(log_returns: list[float], ddof: int = 1, weighting=False, indices: Union[list[int], None] = None) -> float:
        """
        Returns the standard deviation of the log returns
        """
        if len(log_returns) == 0:
            return 0.0

        window = len(indices) if indices is not None else len(log_returns)
        if window < ddof + 1:
            return np.inf

        return Metrics.average((np.array(log_returns) - Metrics.average(log_returns, weighting=weighting, indices=indices)) ** 2, weighting=weighting, indices=indices)

    @staticmethod
    def ann_excess_return(log_returns: list[float], weighting=False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO) -> float:
        """
        Calculates annualized excess return using mean daily log returns and mean daily 1yr risk free rate.
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        days_in_year int: Number of trading days in a year for annualization.
        """
        annual_risk_free_rate = ValiConfig.ANNUAL_RISK_FREE_DECIMAL

        if len(log_returns) == 0:
            return 0.0

        # Annualize the mean daily excess returns
        annualized_excess_return = (Metrics.average(log_returns, weighting=weighting) * days_in_year) - annual_risk_free_rate
        return annualized_excess_return

    @staticmethod
    def ann_volatility(log_returns: list[float], ddof: int = 1, weighting=False, indices: list[int]=None, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO) -> float:
        """
        Calculates annualized volatility ASSUMING DAILY OBSERVATIONS
        Parameters:
        log_returns list[float]: Daily Series of log returns.
        ddof int: Delta Degrees of Freedom. The divisor used in the calculation is N - ddof, where N represents the number of elements.
        weighting bool: Whether to use weighted average.
        indices list[int]: The indices of the log returns to consider.
        days_in_year int: Number of trading days in a year for annualization.
        """
        if indices is None:
            indices = list(range(len(log_returns)))
            
        # Annualize volatility of the daily log returns assuming sample variance
        window = len(indices)
        if window < ddof + 1:
            return np.inf

        annualized_volatility = np.sqrt(Metrics.variance(log_returns, ddof=ddof, weighting=weighting, indices=indices) * days_in_year)

        return float(annualized_volatility)

    @staticmethod
    def ann_downside_volatility(log_returns: list[float], target: int = None, weighting=False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO) -> float:
        """
        Args:
            log_returns: list[float] - Daily Series of log returns.
            target: int: The target return (default: 0).
            weighting: bool: Whether to use weighted average.
            days_in_year int: Number of trading days in a year for annualization.

        Returns:
            The downside annualized volatility as a float assuming sample variance
        """
        if target is None:
            target = Metrics.log_risk_free_rate(days_in_year=days_in_year)

        indices = [i for i, log_return in enumerate(log_returns) if log_return < target]
        return Metrics.ann_volatility(log_returns, weighting=weighting, indices=indices, days_in_year=days_in_year)

    @staticmethod
    def base_return_log(log_returns: list[float], weighting=False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            weighting: whether to use weighted average
            days_in_year: number of trading days in a year for annualization

        Returns:
             The aggregate total return of the miner as a log total
        """
        if len(log_returns) == 0:
            return 0.0

        return float(Metrics.average(log_returns, weighting=weighting)) * days_in_year

    @staticmethod
    def base_return_log_percentage(log_returns: list[float], weighting=False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            weighting: whether to use weighted average
            days_in_year: number of trading days in a year for annualization

        Returns:
             The aggregate total return of the miner as a percentage log total
                """
        if len(log_returns) == 0:
            return 0.0

        return Metrics.average(log_returns, weighting=weighting) * days_in_year * 100

    @staticmethod
    def base_return(log_returns: list[float], weighting: bool = False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            weighting: whether to use weighted average
            days_in_year: number of trading days in a year for annualization

        Returns:
             The aggregate total return of the miner as a percentage
        """
        return float((math.exp(Metrics.base_return_log(log_returns, weighting=weighting, days_in_year=days_in_year)) - 1) * 100)

    @staticmethod
    def daily_max_drawdown(log_returns: list[float]) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner

        Returns:
            The daily drawdowns of the miner
        """
        if len(log_returns) == 0:
            return 0.0

        # More efficient implementation using cumulative sum of log returns
        cumulative_log_returns = np.cumsum(log_returns)
        
        # Maximum cumulative log return at each point
        running_max_log = np.maximum.accumulate(cumulative_log_returns)
        
        # Drawdown = 1 - exp(current - peak)
        # This gives us the percentage decline from the peak
        drawdowns = 1 - np.exp(cumulative_log_returns - running_max_log)
        
        # Find the maximum drawdown
        max_drawdown = np.max(drawdowns)
        
        return max_drawdown

    @staticmethod
    def calmar(log_returns: list[float], ledger: PerfLedger, bypass_confidence: bool = False, weighting: bool = False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO, min_days: int = None, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            ledger: the ledger of the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
            days_in_year: number of trading days in a year for annualization
            min_days: dynamic minimum number of days for statistical confidence
        """

        # Use dynamic minimum if provided, otherwise fall back to global config
        minimum_n = min_days if min_days is not None else ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL

        # Positional Component
        if len(log_returns) < minimum_n:
            if not bypass_confidence:
                return ValiConfig.CALMAR_NOCONFIDENCE_VALUE

        base_return_percentage = Metrics.base_return_log_percentage(log_returns, weighting=weighting, days_in_year=days_in_year)
        drawdown_normalization_factor = LedgerUtils.risk_normalization(ledger)

        return float(base_return_percentage * drawdown_normalization_factor)

    @staticmethod
    def sharpe(log_returns: list[float], bypass_confidence: bool = False, weighting: bool = False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO, min_days: int = None, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
            days_in_year: number of trading days in a year for annualization
            min_days: dynamic minimum number of days for statistical confidence
        """
        # Use dynamic minimum if provided, otherwise fall back to global config
        minimum_n = min_days if min_days is not None else ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL

        # Need a large enough sample size
        if len(log_returns) < minimum_n:
            if not bypass_confidence:
                return ValiConfig.SHARPE_NOCONFIDENCE_VALUE

        # Hyperparameter
        min_std_dev = ValiConfig.SHARPE_STDDEV_MINIMUM

        excess_return = Metrics.ann_excess_return(log_returns, weighting=weighting, days_in_year=days_in_year)
        volatility = Metrics.ann_volatility(log_returns, weighting=weighting, days_in_year=days_in_year)
        
        return float(excess_return / max(volatility, min_std_dev))

    @staticmethod
    def omega(log_returns: list[float], bypass_confidence: bool = False, weighting: bool = False, min_days: int = None, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
            min_days: dynamic minimum number of days for statistical confidence
        """
        # Use dynamic minimum if provided, otherwise fall back to global config
        minimum_n = min_days if min_days is not None else ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL

        # Need a large enough sample size
        if len(log_returns) < minimum_n:
            if not bypass_confidence:
                return ValiConfig.OMEGA_NOCONFIDENCE_VALUE

        if weighting:
            weighing_array = Metrics.weighting_distribution(log_returns)

            positive_indices = []
            negative_indices = []
            product_sum_positive = product_sum_negative = 0
            sum_of_weights_positive = sum_of_weights_negative = ValiConfig.OMEGA_LOSS_MINIMUM

            for c,log_return in enumerate(log_returns):
                if log_return > 0:
                    positive_indices.append(c)
                else:
                    negative_indices.append(c)

            log_return_arr = np.array(log_returns)

            if len(positive_indices) > 0:
                positive_indices_arr = np.array(positive_indices)
                sum_of_weights_positive = max(np.sum(weighing_array[positive_indices_arr]),
                                              ValiConfig.OMEGA_LOSS_MINIMUM)

                product_sum_positive = np.sum(np.multiply(log_return_arr[positive_indices_arr], weighing_array[positive_indices_arr]))

            if len(negative_indices) > 0:
                negative_indices_arr = np.array(negative_indices)
                sum_of_weights_negative = max(np.sum(weighing_array[negative_indices_arr]),
                                              ValiConfig.OMEGA_LOSS_MINIMUM)
                product_sum_negative = np.sum(np.multiply(log_return_arr[negative_indices_arr], weighing_array[negative_indices_arr]))

            positive_sum = product_sum_positive * sum_of_weights_negative
            negative_sum = product_sum_negative * sum_of_weights_positive

        else:
            positive_sum = 0
            negative_sum = 0

            for log_return in log_returns:
                if log_return > 0:
                    positive_sum += log_return
                else:
                    negative_sum += log_return

        numerator = positive_sum
        denominator = max(abs(negative_sum), ValiConfig.OMEGA_LOSS_MINIMUM)

        return float(numerator / denominator)

    @staticmethod
    def statistical_confidence(log_returns: list[float], bypass_confidence: bool = False, min_days: int = None, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
            min_days: dynamic minimum number of days for statistical confidence
        """
        # Use dynamic minimum if provided, otherwise fall back to global config
        minimum_n = min_days if min_days is not None else ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL

        # Impose a minimum sample size on the miner
        if len(log_returns) < minimum_n:
            if not bypass_confidence or len(log_returns) < 2:
                return ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE

        # Also now check for zero variance condition
        zero_variance_condition = bool(np.isclose(np.var(log_returns), 0))
        if zero_variance_condition:
            return ValiConfig.STATISTICAL_CONFIDENCE_NOCONFIDENCE_VALUE

        res = ttest_1samp(log_returns, 0, alternative='greater')
        return float(res.statistic)

    @staticmethod
    def sortino(log_returns: list[float], bypass_confidence: bool = False, weighting: bool = False, days_in_year: int = ValiConfig.DAYS_IN_YEAR_CRYPTO, min_days: int = None, **kwargs) -> float:
        """
        Args:
            log_returns: list of daily log returns from the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use weighted average
            days_in_year: number of trading days in a year for annualization
            min_days: dynamic minimum number of days for statistical confidence
        """
        # Use dynamic minimum if provided, otherwise fall back to global config
        minimum_n = min_days if min_days is not None else ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL

        # Need a large enough sample size
        if len(log_returns) < minimum_n:
            if not bypass_confidence:
                return ValiConfig.SORTINO_NOCONFIDENCE_VALUE

        # Hyperparameter
        min_downside = ValiConfig.SORTINO_DOWNSIDE_MINIMUM

        # Sortino ratio is calculated as the mean of the returns divided by the standard deviation of the negative returns
        excess_return = Metrics.ann_excess_return(log_returns, weighting=weighting, days_in_year=days_in_year)
        downside_volatility = Metrics.ann_downside_volatility(log_returns, weighting=weighting, days_in_year=days_in_year)

        return float(excess_return / max(downside_volatility, min_downside))

    @staticmethod
    def time_weighted_scores(ledgers: dict[str, PerfLedger], miner_scores: list[str, float]):
        """
        Scales scores by the number of trading days each miner has
        Args:
            ledgers: list of daily ledgers
            miner_scores: list of tuples as (hotkey, score)
        """

        time_weighted_scores = []
        LEDGER_WINDOW_DAYS = ValiConfig.TARGET_LEDGER_WINDOW_DAYS
        if LEDGER_WINDOW_DAYS <= 0:
            return []

        if not ledgers or not miner_scores:
            return []

        for miner, weight in miner_scores:
            if isinstance(ledgers.get(miner), dict):
                miner_ledger = ledgers.get(miner).get(TP_ID_PORTFOLIO)
            else:
                miner_ledger = ledgers.get(miner)
            miner_trading_days = LedgerUtils.get_trading_days(miner_ledger)
            time_weighted_scores.append((miner, weight * math.sqrt(miner_trading_days / LEDGER_WINDOW_DAYS)))

        return time_weighted_scores

    @staticmethod
    def pnl_score(log_returns: list[float], ledger: PerfLedger, bypass_confidence: bool = False,
                   weighting: bool = False, **kwargs) -> float:
        """
        Calculate PnL score from daily PnL values in the ledger.
        
        Args:
            log_returns: list of daily log returns from the miner (unused but kept for consistency)
            ledger: PerfLedger - the ledger of the miner
            bypass_confidence: whether to use default value if not enough trading days
            weighting: whether to use time-weighted average
            
        Returns:
            float: Average daily PnL score
        """
        if not ledger:
            return ValiConfig.PNL_NOCONFIDENCE_VALUE
            
        # Get daily PnL values for complete days only
        daily_pnl_values = LedgerUtils.daily_pnl(ledger)

        if not daily_pnl_values:
            return ValiConfig.PNL_NOCONFIDENCE_VALUE
            
        # Apply time weighting if requested
        return Metrics.average(daily_pnl_values, weighting=weighting, min_weight=ValiConfig.WEIGHTED_AVERAGE_DECAY_MIN_PNL)

