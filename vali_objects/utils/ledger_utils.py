# developer: trdougherty
import math
import statistics

import numpy as np
import copy
from datetime import datetime, timezone, timedelta, date
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
from vali_objects.vali_config import ValiConfig, TradePair
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
from vali_objects.utils.asset_segmentation import AssetSegmentation
from time_util.time_util import ForexHolidayCalendar
import bittensor as bt


class LedgerUtils:
    forex_holiday_calendar = ForexHolidayCalendar()

    @staticmethod
    def daily_returns(ledger: PerfLedger) -> list[float]:
        """
        Calculate daily returns from performance checkpoints, only including full days
        :param ledger: PerfLedger - the ledger of the miner
        :return: list[float] - list of daily returns for complete days as a percentage
        """
        daily_returns_logged = LedgerUtils.daily_return_log(ledger)
        daily_returns_percentage = [(math.exp(x) - 1) * 100 for x in daily_returns_logged]
        return daily_returns_percentage

    @staticmethod
    def daily_returns_by_date(ledger: PerfLedger, return_type: str) -> dict[date, float]:
        """
        Calculate daily returns from performance checkpoints, only including full days
        :param ledger: PerfLedger - the ledger of the miner
        :param return_type: str - either 'simple' or 'log' to specify return type
        :return: dict[datetime.date, float] - dictionary mapping dates to daily returns 
                 (both as decimal values, not percentages)
        """
        if return_type not in ['simple', 'log']:
            raise ValueError("return_type must be either 'simple' or 'log'")
        
        date_return_map = LedgerUtils.daily_return_log_by_date(ledger)
        
        if return_type == 'log':
            return date_return_map
        else:  # simple
            return {date: math.exp(log_return) - 1
                    for date, log_return in date_return_map.items()}

    @staticmethod
    def daily_return_ratio_by_date(ledger: PerfLedger, return_type: str) -> dict[datetime.date, float]:
        """
        Calculate daily returns from performance checkpoints, with date keys as datetime.date objects.

        :param ledger: PerfLedger - the ledger of the miner
        :param return_type: str - either 'simple' or 'log' to specify return type
        :return: dict[datetime.date, float] - dictionary mapping dates to daily returns 
                 (both as decimal values, not percentages)
        """
        if return_type not in ['simple', 'log']:
            raise ValueError("return_type must be either 'simple' or 'log'")
            
        if not ledger or not ledger.cps:
            return {}

        daily_cps = {}

        # Group checkpoints by date
        for cp in ledger.cps:
            # if cp's `last_update_ms` is at day T 00:00:00, it represents the ending value for day T-1
            cp_dt = datetime.fromtimestamp(cp.last_update_ms / 1000, tz=timezone.utc)
            if cp_dt.time() == datetime.min.time():   # 00:00:00 UTC
                running_date = (cp_dt - timedelta(days=1)).date()
                daily_cps[running_date] = cp

        ans = {}
        prev_day_end_value = 1.0  # First day begins with portfolio value of 1
        
        # Iterating in chronological order since python dicts are sorted by insertion order
        for date, cp in daily_cps.items():
            # For day 'date':
            # - begin_value = portfolio value at beginning of day (00:00:00)
            # - end_value = portfolio value at end of day (next day 00:00:00)
            begin_value = prev_day_end_value
            end_value = cp.prev_portfolio_ret
            
            try:
                if return_type == 'log':
                    ans[date] = math.log(end_value / begin_value)
                else:  # simple
                    ans[date] = (end_value / begin_value) - 1
            except (ZeroDivisionError, ValueError):
                ans[date] = None   # fallback if begin_value is 0 or invalid
                
            prev_day_end_value = end_value

        return ans
                
    @staticmethod
    def daily_returns_by_date_json(ledger: PerfLedger, return_type: str) -> dict[str, float]:
        """
        Calculate daily returns from performance checkpoints, with date keys as strings (YYYY-MM-DD)
        to ensure JSON compatibility.
        
        :param ledger: PerfLedger - the ledger of the miner
        :param return_type: str - either 'simple' or 'log' to specify return type
        :return: dict[str, float] - dictionary mapping date strings to daily returns 
                 (both as decimal values, not percentages, rounded to 6 decimal places)
        """
        date_return_map = LedgerUtils.daily_returns_by_date(ledger, return_type)
        return {date.isoformat(): round(return_value, 6)
                for date, return_value in date_return_map.items()}

    @staticmethod
    def daily_return_log(ledger: PerfLedger) -> list[float]:
        """
        Calculate daily returns from performance checkpoints, only including full days
        with complete data and correct total accumulated time.

        Args:
            ledger: PerfLedger - the ledger of the miner
        Returns:
            List[float] - list of daily returns for complete days
        """
        if ledger is None or not ledger.cps:
            return []

        date_return_map = LedgerUtils.daily_return_log_by_date(ledger)
        return list(date_return_map.values())

    @staticmethod
    def daily_return_log_by_date(ledger: PerfLedger) -> dict[date, float]:
        """
        Calculate daily returns from performance checkpoints, only including full days
        with complete data and correct total accumulated time.
        Returns results as a dictionary mapping dates to returns.

        Args:
            ledger: PerfLedger - the ledger of the miner
        Returns:
            dict[date, float] - dictionary mapping dates to daily log returns
        """
        complete_days = LedgerUtils._group_checkpoints_by_complete_days(ledger)
        
        date_return_map = {}
        for running_date, day_checkpoints in sorted(complete_days.items()):
            daily_return = sum(cp.gain + cp.loss for cp in day_checkpoints)
            date_return_map[running_date] = daily_return

        return date_return_map

    @staticmethod
    def _group_checkpoints_by_complete_days(ledger: PerfLedger) -> dict[datetime.date, list]:
        """
        Helper function to group checkpoints by date for complete days only.
        
        Args:
            ledger: PerfLedger - the ledger of the miner
            
        Returns:
            dict[datetime.date, list] - dictionary mapping dates to lists of checkpoints for complete days
        """
        if not ledger or not ledger.cps:
            return {}

        checkpoints = ledger.cps
        daily_groups = {}
        n_checkpoints_per_day = int(ValiConfig.DAILY_CHECKPOINTS)

        # Group checkpoints by date
        for cp in checkpoints:
            # Need to use the beginning of the cell, otherwise it may bleed into the next day
            start_time = (cp.last_update_ms - cp.accum_ms)
            full_cell = cp.accum_ms == ValiConfig.TARGET_CHECKPOINT_DURATION_MS

            running_date = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).date()
            if not LedgerUtils.is_valid_trading_day(ledger, running_date):
                continue
            if full_cell:
                if running_date not in daily_groups:
                    daily_groups[running_date] = []
                daily_groups[running_date].append(cp)

        # Filter to only include complete days
        complete_days = {}
        for running_date, day_checkpoints in daily_groups.items():
            if len(day_checkpoints) == n_checkpoints_per_day:
                complete_days[running_date] = day_checkpoints

        return complete_days

    @staticmethod
    def raw_pnl(ledger: PerfLedger) -> float:
        """
        Calculate total pnl from tracked PnL values in perf ledgers.
        Uses the payout formula: realized_pnl + min(0, unrealized_pnl)

        Args:
            ledger: PerfLedger - the ledger of the miner

        Returns:
            float - total payout pnl of the ledger (penalizes unrealized losses, ignores unrealized gains)
        """

        if ledger is None or not ledger.cps:
            return 0
        total_pnl = 0
        for cp in ledger.cps:
            # Payout formula: realized + unrealized (only if negative)
            total_pnl += cp.realized_pnl + min(0, cp.unrealized_pnl)

        return total_pnl

    @staticmethod
    def daily_pnl_by_date(ledger: PerfLedger) -> dict[datetime.date, float]:
        """
        Calculate daily PnL from performance checkpoints, only including full days
        with complete data and correct total accumulated time.
        Uses the payout formula: realized_pnl + min(0, unrealized_pnl)

        Args:
            ledger: PerfLedger - the ledger of the miner

        Returns:
            dict[datetime.date, float] - dictionary mapping dates to payout PnL
        """
        complete_days = LedgerUtils._group_checkpoints_by_complete_days(ledger)

        date_pnl_map = {}
        for running_date, day_checkpoints in sorted(complete_days.items()):
            # Payout formula: realized + unrealized (only if negative)
            total_pnl = sum(cp.realized_pnl + min(0, cp.unrealized_pnl) for cp in day_checkpoints)
            date_pnl_map[running_date] = total_pnl

        return date_pnl_map

    @staticmethod
    def daily_pnl(ledger: PerfLedger) -> list[float]:
        """
        Calculate daily PnL from performance checkpoints, only including full days
        with complete data and correct total accumulated time.

        Args:
            ledger: PerfLedger - the ledger of the miner
        Returns:
            List[float] - list of daily PnL values for complete days
        """
        if ledger is None or not ledger.cps:
            return []

        date_pnl_map = LedgerUtils.daily_pnl_by_date(ledger)
        return list(date_pnl_map.values())

    @staticmethod
    def is_valid_trading_day(ledger: PerfLedger, testing_date: date) -> bool:
        """
        Verifies if a particular forex day has polygon data at the start and end of a day.
        If the fx market is closed for an entire day (Saturdays with UTC), we don't
        want to track daily returns for that day.

        Args:
            ledger: PerfLedger - the ledger of the miner
            testing_date: datetime - the date at which to check for valid forex days
        Returns:
            bool - True if a valid forex day is found, False otherwise
        """
        if ledger is None:
            bt.logging.info("ledger is None, returning False")
            return False
        
        if testing_date is None or not isinstance(testing_date, date):
            bt.logging.info(f"testing_date is invalid, returning False: {testing_date}")
            return False
        
        asset_id = ledger.tp_id
        # TODO We may need to revisit this if portfolio ledgers become asset specific
        if asset_id == TP_ID_PORTFOLIO:
            return True

        trade_pair = TradePair.from_trade_pair_id(asset_id)
        if trade_pair is None:
            bt.logging.info(f"trade_pair not found for asset_id: {asset_id}, returning False")
            return False
            
        if trade_pair.is_forex and LedgerUtils.forex_holiday_calendar.is_forex_market_closed_full_day(testing_date):
            return False

        return True

    @staticmethod
    def ledger_drawdowns(ledger: PerfLedger) -> list[float]:
        """
        Extracts all drawdown values from a ledger.
        
        Args:
            ledger: PerfLedger - the ledger of the miner
            
        Returns:
            list[float]: List of drawdown values from all checkpoints
        """
        if not ledger or not ledger.cps:
            return []

        drawdowns = []
        for cp in ledger.cps:
            drawdowns.append(cp.mdd)

        return drawdowns

    @staticmethod
    def daily_return_percentage(ledger: PerfLedger) -> list[float]:
        """
        Calculate daily returns as percentages.
        
        Args:
            ledger: PerfLedger - the ledger of the miner
        
        Returns:
            list[float] - list of daily returns for complete days as percentages
        """
        return [(math.exp(x)-1) * 100 if x != 0 else 0 for x in LedgerUtils.daily_return_log(ledger)]

    @staticmethod
    def ledger_returns(ledger: dict[str, PerfLedger]) -> dict[str, list[float]]:
        """
        Calculate percentage returns for multiple miners.
        
        Args:
            ledger: dict[str, PerfLedger] - the ledger of the miners
            
        Returns:
            dict[str, list[float]] - Dictionary mapping miner hotkeys to daily returns as percentages
        """
        miner_returns = {}

        for miner, miner_ledger in ledger.items():
            miner_returns[miner] = LedgerUtils.daily_return_percentage(miner_ledger if miner_ledger else PerfLedger())

        return miner_returns

    @staticmethod
    def ledger_returns_log(ledger: dict[str, PerfLedger]) -> dict[str, list[float]]:
        """
        Calculate logarithmic returns for multiple miners.
        
        Args:
            ledger: dict[str, PerfLedger] - the ledger of the miners
            
        Returns:
            dict[str, list[float]] - Dictionary mapping miner hotkeys to daily returns as logarithmic values
        """
        if not ledger:
            return {}

        miner_returns = {}

        for miner, miner_ledger in ledger.items():
            miner_returns[miner] = LedgerUtils.daily_return_log(miner_ledger)

        return miner_returns

    @staticmethod
    def drawdown_percentage(drawdown_decimal: float) -> float:
        """
        Args:
            drawdown_decimal: float - the drawdown value

        Returns:
            float - the drawdown percentage
        """
        if drawdown_decimal >= 1:
            return 0

        if drawdown_decimal <= 0:
            return 100

        return np.clip((1 - drawdown_decimal) * 100, 0, 100)

    @staticmethod
    def mdd_lower_augmentation(drawdown_percent: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if drawdown_percent <= 0:
            return 0

        if drawdown_percent > 100:
            return 0

        drawdown_minvalue = ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE

        # Drawdown value
        if drawdown_percent <= drawdown_minvalue:
            return 0

        return 1

    @staticmethod
    def mdd_upper_augmentation(drawdown_percent: float) -> float:
        """
        Should only look at the upper region of the drawdown
        """
        if drawdown_percent <= 0:
            return 0

        if drawdown_percent > 100:
            return 0

        drawdown_maxvalue = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE
        drawdown_scaling = ValiConfig.DRAWDOWN_UPPER_SCALING

        upper_penalty = (-drawdown_percent + drawdown_maxvalue) / drawdown_scaling
        return float(np.clip(upper_penalty, 0, 1))

    @staticmethod
    def mdd_base_augmentation(drawdown_percent: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if drawdown_percent <= 0:
            return 0

        if drawdown_percent > 100:
            return 0

        return float(1 / drawdown_percent)

    @staticmethod
    def mdd_augmentation(drawdown: float) -> float:
        """
        Args: mdd: float - the maximum drawdown of the miner
        """
        if drawdown <= 0 or drawdown > 1:
            return 0

        recent_drawdown_percentage = LedgerUtils.drawdown_percentage(drawdown)
        # if recent_drawdown_percentage <= ValiConfig.DRAWDOWN_MINVALUE_PERCENTAGE:
        #     return 0

        if recent_drawdown_percentage >= ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE:
            return 0

        base_augmentation = LedgerUtils.mdd_base_augmentation(recent_drawdown_percentage)
        # lower_augmentation = LedgerUtils.mdd_lower_augmentation(recent_drawdown_percentage)
        # upper_augmentation = LedgerUtils.mdd_upper_augmentation(recent_drawdown_percentage)

        drawdown_penalty = base_augmentation # * lower_augmentation * upper_augmentation
        return float(drawdown_penalty)

    @staticmethod
    def max_drawdown_threshold_penalty(ledger: PerfLedger) -> float:
        """
        Args:
            ledger: PerfLedger - the ledger of the miner
        """
        if not ledger:
            return 1
        drawdown_limit = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE

        effective_drawdown = LedgerUtils.instantaneous_max_drawdown(ledger)
        effective_drawdown_percentage = LedgerUtils.drawdown_percentage(effective_drawdown)

        if effective_drawdown_percentage >= drawdown_limit:
            return 0

        return 1

    @staticmethod
    def daily_max_drawdown(ledger: PerfLedger) -> float:
        """
        Args:
            ledger: PerfLedger - the ledger of the miner

        Returns:
            float - the maximum drawdown percentage for the ledger
        """
        if not ledger:
            return 0
        checkpoints = ledger.cps
        if len(checkpoints) == 0:
            return 0

        # First collect daily returns
        derivative_account_values = LedgerUtils.daily_return_log(ledger)
        
        # Handle case where there are no complete daily returns
        if len(derivative_account_values) == 0:
            return 0

        # We are now looking at the cumulative account values as a percentage relative to 1 (baseline)
        cumulative_account_values = np.exp(np.cumsum(derivative_account_values))

        # Minimum points in the future relative to the current point
        running_max = np.maximum.accumulate(cumulative_account_values)
        drawdowns = (cumulative_account_values - running_max) / running_max
        drawdowns_numeric = 1 + drawdowns  # drawdowns should all be negative

        return min(drawdowns_numeric)

    @staticmethod
    def instantaneous_max_drawdown(ledger: PerfLedger) -> float:
        """
        Args:
            ledger: PerfLedger - the ledger of the miner
        """
        if not ledger:
            return 0
        checkpoints = ledger.cps
        if len(checkpoints) == 0:
            return 0

        # Compute the drawdown of the checkpoints
        drawdowns = [checkpoint.mdd for checkpoint in checkpoints]
        effective_drawdown = np.min(drawdowns)
        final_drawdown = np.clip(effective_drawdown, 0, 1.0)

        return final_drawdown

    @staticmethod
    def is_beyond_max_drawdown(ledger_element: PerfLedger):
        """Checks if the maximum drawdown percentage is surpassed"""
        if ledger_element is None:
            return False, 0

        if len(ledger_element.cps) == 0:
            return False, 0

        maximum_drawdown_percent = ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE

        max_drawdown = LedgerUtils.instantaneous_max_drawdown(ledger_element)
        recorded_drawdown_percentage = LedgerUtils.drawdown_percentage(max_drawdown)

        # Drawdown is less than our maximum permitted drawdown
        max_drawdown_criteria = recorded_drawdown_percentage >= maximum_drawdown_percent

        # Drawdown already checked, round for display purposes
        recorded_drawdown_percentage = float(round(recorded_drawdown_percentage, 2))

        return max_drawdown_criteria, recorded_drawdown_percentage

    @staticmethod
    def risk_normalization(ledger: PerfLedger) -> float:
        """
        Args:
            ledger: PerfLedger - the ledger of the miner
        """
        # recent_drawdown = LedgerUtils.recent_drawdown(checkpoints)
        approximate_drawdown = LedgerUtils.instantaneous_max_drawdown(ledger)
        drawdown_penalty = LedgerUtils.mdd_augmentation(approximate_drawdown)
        return drawdown_penalty

    @staticmethod
    def cumulative(ledger: PerfLedger) -> PerfLedger:
        """
        Adds the cumulative return of the ledger to each checkpoint.
        Args:
            ledger: dict[str, PerfLedger] - the ledger of the miners

        Returns:
            dict[str, dict] - the cumulative return of the ledger
        """
        ledger_dict = ledger.to_dict()
        ledger_copy = copy.deepcopy(ledger_dict)

        # for miner, miner_ledger in ledger_copy.items():
        return_overall = 1.0
        if len(ledger_copy['cps']) > 0:
            for cp in ledger_copy['cps']:
                return_value = math.exp(cp['gain'] + cp['loss'])
                return_overall *= return_value
                cp['overall_returns'] = return_overall

            ledger_copy = PerfLedger.from_dict(ledger_copy)

        return ledger_copy

    @staticmethod
    def get_trading_days(ledger: PerfLedger) -> int:
        """
        Get the number of trading days for a ledger.
        Args:
            ledger: PerfLedger - the ledger of the miners
        Returns:
            int - the number of trading days
        """

        if ledger is None:
            return 0
        miner_returns = LedgerUtils.daily_return_log(ledger)

        return len(miner_returns)

    @staticmethod
    def calculate_dynamic_minimum_days_for_asset_classes(
        ledger_dict: dict[str, dict[str, PerfLedger]],
        asset_classes: list
    ) -> dict:
        """
        Calculates the dynamic minimum participation days for specific asset classes.
        Returns the number of days that the Nth longest participating miner has (where N is
        configured by DYNAMIC_MIN_DAYS_PERCENTILE_RANK), capped at 60 days and floored at 7 days.

        Args:
            ledger_dict: Dictionary mapping hotkeys to their full ledger data
            asset_classes: List of asset classes (TradePairCategory) to calculate dynamic minimum days for

        Returns:
            dict: Dictionary mapping asset class to min days requirement (between 7-60 days)
        """
        asset_class_min_days = {asset_class: ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL for asset_class in asset_classes}

        if not ledger_dict:
            return asset_class_min_days

        try:
            # Create asset segmentation to get miners participating in this asset class
            segmentation_machine = AssetSegmentation(ledger_dict)

            for asset_class in asset_classes:
                asset_ledger = segmentation_machine.segmentation(asset_class)

                # Calculate participation days for each miner in this asset class
                miner_participation_days = []
                for hotkey, ledger in asset_ledger.items():
                    if ledger is not None:
                        days_participating = LedgerUtils.get_trading_days(ledger)
                        if days_participating > 0:
                            miner_participation_days.append(days_participating)

                # Sort in descending order (longest participation first)
                miner_participation_days.sort(reverse=True)

                if len(miner_participation_days) < ValiConfig.DYNAMIC_MIN_DAYS_NUM_MINERS:
                    minimum_days = ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR  # Not enough participating miners, return floor
                else:
                    # Use the shorter of Nth longest participating miner (index N-1), or median of all participating miners
                    minimum_days = min(miner_participation_days[ValiConfig.DYNAMIC_MIN_DAYS_NUM_MINERS - 1], int(statistics.median(miner_participation_days)))

                # Apply bounds: floor of 7 days, cap of 60 days
                asset_class_min_days[asset_class] = max(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_FLOOR, min(ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL, minimum_days))
            return asset_class_min_days
        except Exception as e:
            bt.logging.warning(f"Error calculating dynamic minimum days: {e}")
            return asset_class_min_days
