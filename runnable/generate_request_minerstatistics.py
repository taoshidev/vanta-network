import os
import json
import time
import gzip

import bittensor as bt
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from datetime import datetime
from shared_objects.mock_metagraph import MockMetagraph
from time_util.time_util import TimeUtil
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import ValiConfig, TradePair
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.position_penalties import PositionPenalties
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, TP_ID_PORTFOLIO
from vali_objects.utils.risk_profiling import RiskProfiling
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
from vali_objects.position import Position

# ---------------------------------------------------------------------------
# Enums and Dataclasses
# ---------------------------------------------------------------------------
class ScoreType(Enum):
    """Enum for different types of scores that can be calculated"""
    BASE = "base"
    AUGMENTED = "augmented"


@dataclass
class ScoreMetric:
    """Class to hold metric calculation configuration"""
    name: str
    metric_func: callable
    weight: float = 1.0
    requires_penalties: bool = False
    requires_weighting: bool = False
    bypass_confidence: bool = False

class ScoreResult:
    """Class to hold score calculation results"""

    def __init__(self, value: float, rank: int, percentile: float, overall_contribution: float = 0):
        self.value = value
        self.rank = rank
        self.percentile = percentile
        self.overall_contribution = overall_contribution

    def to_dict(self) -> Dict[str, float]:
        return {
            "value": self.value,
            "rank": self.rank,
            "percentile": self.percentile,
            "overall_contribution": self.overall_contribution
        }


# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------
class MetricsCalculator:
    """Class to handle all metrics calculations"""

    def __init__(self, metrics=None):
        # Add or remove metrics as desired. Excluding short-term metrics as requested.
        if metrics is None:

            self.metrics = {
                "omega": ScoreMetric(
                    name="omega",
                    metric_func=Metrics.omega,
                    weight=ValiConfig.SCORING_OMEGA_WEIGHT,
                ),
                "sharpe": ScoreMetric(
                    name="sharpe",
                    metric_func=Metrics.sharpe,
                    weight=ValiConfig.SCORING_SHARPE_WEIGHT
                ),
                "sortino": ScoreMetric(
                    name="sortino",
                    metric_func=Metrics.sortino,
                    weight=ValiConfig.SCORING_SORTINO_WEIGHT
                ),
                "statistical_confidence": ScoreMetric(
                    name="statistical_confidence",
                    metric_func=Metrics.statistical_confidence,
                    weight=ValiConfig.SCORING_STATISTICAL_CONFIDENCE_WEIGHT
                ),
                "calmar": ScoreMetric(
                    name="calmar",
                    metric_func=Metrics.calmar,
                    weight=ValiConfig.SCORING_CALMAR_WEIGHT
                ),
                "return": ScoreMetric(
                    name="return",
                    metric_func=Metrics.base_return_log_percentage,
                    weight=ValiConfig.SCORING_RETURN_WEIGHT
                ),
                "pnl": ScoreMetric(
                    name="pnl",
                    metric_func=Metrics.pnl_score,
                    weight=ValiConfig.SCORING_PNL_WEIGHT
                )
            }
        else:
            self.metrics = metrics

    def calculate_metric(
        self,
        metric: ScoreMetric,
        data: Dict[str, Dict[str, Any]],
        weighting: bool = False
    ) -> list[tuple[str, float]]:
        """
        Calculate a single metric for all miners.
        """
        scores = {}
        for hotkey, miner_data in data.items():
            log_returns = miner_data.get("log_returns", [])
            ledger = miner_data.get("ledger", {}).get(TP_ID_PORTFOLIO)
            value = metric.metric_func(
                log_returns=log_returns,
                ledger=ledger,
                weighting=weighting,
                bypass_confidence=metric.bypass_confidence
            )

            scores[hotkey] = value

        return list(scores.items())


# ---------------------------------------------------------------------------
# MinerStatisticsManager
# ---------------------------------------------------------------------------
class MinerStatisticsManager:
    def __init__(
        self,
        position_manager: PositionManager,
        subtensor_weight_setter: SubtensorWeightSetter,
        plagiarism_detector: PlagiarismDetector,
        contract_manager: ValidatorContractManager,
        metrics: Dict[str, MetricsCalculator] = None,
        ipc_manager = None
    ):
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.elimination_manager = position_manager.elimination_manager
        self.challengeperiod_manager = position_manager.challengeperiod_manager
        self.subtensor_weight_setter = subtensor_weight_setter
        self.plagiarism_detector = plagiarism_detector
        self.contract_manager = contract_manager

        self.metrics_calculator = MetricsCalculator(metrics=metrics)
        if ipc_manager:
            self.miner_statistics = ipc_manager.dict()
        else:
            self.miner_statistics = {}

    # -------------------------------------------
    # Ranking / Percentile Helpers
    # -------------------------------------------
    def rank_dictionary(self, d: list[tuple[str, float]], ascending: bool = False) -> list[tuple[str, int]]:
        """Rank the values in a dictionary (descending by default)."""
        sorted_items = sorted(d, key=lambda item: item[1], reverse=not ascending)
        return {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}

    def percentile_rank_dictionary(self, d: list[tuple[str, float]], ascending: bool = False) -> list[tuple[str, float]]:
        """Calculate percentile ranks for dictionary values."""
        percentiles = Scoring.miner_scores_percentiles(d)
        return dict(percentiles)
    # -------------------------------------------
    # Gather Extra Stats (drawdowns, volatility, etc.)
    # -------------------------------------------
    def gather_extra_data(self, hotkey: str, miner_ledger: PerfLedger, positions_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gathers additional data such as volatility, drawdowns, engagement stats,
        ignoring short-term metrics.
        """
        miner_cps = miner_ledger.cps if miner_ledger else []
        miner_positions = positions_dict.get(hotkey, [])
        miner_returns = LedgerUtils.daily_return_log(miner_ledger)

        # Volatility
        ann_volatility = min(Metrics.ann_volatility(miner_returns), 100)
        ann_downside_volatility = min(Metrics.ann_downside_volatility(miner_returns), 100)

        # Drawdowns
        instantaneous_mdd = LedgerUtils.instantaneous_max_drawdown(miner_ledger)
        daily_mdd = LedgerUtils.daily_max_drawdown(miner_ledger)

        # Engagement: positions
        n_positions = len(miner_positions)
        pos_duration = PositionUtils.total_duration(miner_positions)
        percentage_profitable = self.position_manager.get_percent_profitable_positions(miner_positions)

        # Engagement: checkpoints
        n_checkpoints = len([cp for cp in miner_cps if cp.open_ms > 0])
        checkpoint_durations = sum(cp.open_ms for cp in miner_cps)

        # Minimum days boolean
        meets_min_days = (len(miner_returns) >= ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL)

        return {
            "annual_volatility": ann_volatility,
            "annual_downside_volatility": ann_downside_volatility,
            "instantaneous_max_drawdown": instantaneous_mdd,
            "daily_max_drawdown": daily_mdd,
            "positions_info": {
                "n_positions": n_positions,
                "positional_duration": pos_duration,
                "percentage_profitable": percentage_profitable
            },
            "checkpoints_info": {
                "n_checkpoints": n_checkpoints,
                "checkpoint_durations": checkpoint_durations
            },
            "minimum_days_boolean": meets_min_days
        }

    # -------------------------------------------
    # Prepare data for metric calculations
    # -------------------------------------------
    def prepare_miner_data(self, hotkey: str, filtered_ledger: Dict[str, Any], filtered_positions: Dict[str, Any], time_now: int) -> Dict[str, Any]:
        """
        Combines the minimal fields needed for the metrics plus the extra data.
        """
        miner_ledger: Dict = filtered_ledger.get(hotkey, {})
        if not miner_ledger:
            return {}
        overall_miner_ledger = miner_ledger.get(TP_ID_PORTFOLIO)
        cumulative_miner_returns_ledger: PerfLedger = LedgerUtils.cumulative(overall_miner_ledger)
        miner_daily_returns: list[float] = LedgerUtils.daily_return_log(overall_miner_ledger)
        miner_positions: list[Position] = filtered_positions.get(hotkey, [])

        extra_data = self.gather_extra_data(hotkey, overall_miner_ledger, filtered_positions)

        return {
            "positions": miner_positions,
            "ledger": miner_ledger,
            "log_returns": miner_daily_returns,
            "cumulative_ledger": cumulative_miner_returns_ledger,
            "extra_data": extra_data
        }

    # -------------------------------------------
    # Penalties: store them individually so we can show them
    # -------------------------------------------
    def calculate_penalties_breakdown(self, miner_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Returns a dict:
            {
               hotkey: {
                  "drawdown_threshold": ...,
                  "risk_profile": ...,
                  "total": ...
               }
            }
        """
        results = {}
        for hotkey, data in miner_data.items():
            ledger = data.get("ledger", {}).get(TP_ID_PORTFOLIO)
            positions = data.get("positions", [])

            # For functions that still require checkpoints directly
            drawdown_threshold_penalty = LedgerUtils.max_drawdown_threshold_penalty(ledger)
            risk_profile_penalty = PositionPenalties.risk_profile_penalty(positions)

            total_penalty = drawdown_threshold_penalty * risk_profile_penalty

            results[hotkey] = {
                "drawdown_threshold": drawdown_threshold_penalty,
                "risk_profile": risk_profile_penalty,
                "total": total_penalty
            }
        return results

    # -------------------------------------------
    def calculate_penalties(self, miner_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        breakdown = self.calculate_penalties_breakdown(miner_data)
        return {hk: breakdown[hk]["total"] for hk in breakdown}

    # -------------------------------------------
    # Main scoring wrapper
    # -------------------------------------------
    def calculate_all_scores(
            self,
            miner_data: Dict[str, Dict[str, Any]],
            asset_class_min_days: dict[str, int],
            score_type: ScoreType = ScoreType.BASE,
            bypass_confidence: bool = False,
            time_now: int = None
    ) -> Dict[str, Dict[str, ScoreResult]]:
        """Calculate all metrics for all miners (BASE, AUGMENTED) for all asset classes."""
        ledgers = {}
        positions = {}
        for hotkey, data in miner_data.items():
            ledgers[hotkey] = data.get("ledger", None)
            positions[hotkey] = data.get("positions", [])
        # Initialize flags
        weighting = False

        # Reset all flags first
        for metric in self.metrics_calculator.metrics.values():
            metric.requires_penalties = False
            metric.requires_weighting = False
            metric.bypass_confidence = bypass_confidence

        if score_type == ScoreType.AUGMENTED:
            weighting = True
            for metric in self.metrics_calculator.metrics.values():
                metric.requires_weighting = True
        all_miner_account_sizes = self.contract_manager.get_all_miner_account_sizes(timestamp_ms=time_now)
        asset_class_scores = Scoring.score_miners(
            ledger_dict=ledgers,
            positions=positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=time_now,
            weighting=weighting,
            scoring_config=self.extract_scoring_config(self.metrics_calculator.metrics),
            all_miner_account_sizes=all_miner_account_sizes
        )

        metric_results = {asset_class.value: {} for asset_class in asset_class_scores.keys()}
        asset_class_scores["overall"] = {"metrics": self.metrics_calculator.metrics}
        metric_results["overall"] = {}


        for asset_class, scoring_dict in asset_class_scores.items():
            for metric_name, metric_data in scoring_dict['metrics'].items():
                if asset_class == "overall":
                    numeric_scores = self.metrics_calculator.calculate_metric(
                        self.metrics_calculator.metrics.get(metric_name, {}),
                        miner_data,
                        weighting=weighting
                    )
                else:
                    numeric_scores = metric_data.get("scores", [])
                ranks = self.rank_dictionary(numeric_scores)
                percentiles = self.percentile_rank_dictionary(numeric_scores)
                numeric_dict = dict(numeric_scores)

                # Build ScoreResult objects
                metric_results[asset_class][metric_name] = {
                    hotkey: ScoreResult(
                        value=numeric_dict[hotkey],
                        rank=ranks[hotkey],
                        percentile=percentiles[hotkey],
                        overall_contribution=percentiles[hotkey] * self.metrics_calculator.metrics.get(metric_name, {}).weight
                    )
                    for hotkey in numeric_dict
                }

        return metric_results

    # -------------------------------------------
    # Daily Returns
    # -------------------------------------------
    def calculate_all_daily_returns(self, filtered_ledger: dict[str, dict[str, PerfLedger]], return_type: str) -> dict[str, list[float]]:
        """Calculate daily returns for all miners.
        
        Args:
            filtered_ledger: Dictionary of miner ledgers
            return_type: 'simple' or 'log' to specify return type
        
        Returns:
            Dictionary mapping hotkeys to daily returns
        """
        return {
            hotkey: LedgerUtils.daily_returns_by_date_json(ledgers.get(TP_ID_PORTFOLIO), return_type=return_type)
            for hotkey, ledgers in filtered_ledger.items()
        }

    # -------------------------------------------
    # Risk Profile
    # -------------------------------------------
    def calculate_risk_profile(
        self,
        miner_data: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Computes all statistics associated with the risk profiling system"""
        miner_data_positions = {hk: data.get("positions", []) for hk, data in miner_data.items()}

        risk_score = RiskProfiling.risk_profile_score(miner_data_positions)
        risk_penalty = RiskProfiling.risk_profile_penalty(miner_data_positions)

        risk_dictionary = {
            hotkey: {
                "risk_profile_score": risk_score.get(hotkey),
                "risk_profile_penalty": risk_penalty.get(hotkey)
            } for hotkey in miner_data_positions.keys()
        }

        return risk_dictionary

    def calculate_risk_report(
        self,
        miner_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Computes all statistics associated with the risk profiling system"""
        miner_data_positions = {hk: data.get("positions", []) for hk, data in miner_data.items()}

        miner_risk_report = {}
        for hotkey, positions in miner_data_positions.items():
            risk_report = RiskProfiling.risk_profile_reporting(positions, verbose=False)
            miner_risk_report[hotkey] = risk_report

        return miner_risk_report

    def extract_scoring_config(self, scoremetric_dict):
        scoring_config = {}

        for key, metric in scoremetric_dict.items():
            # Skip "return" if not needed in the final config
            if key == "return":
                continue

            scoring_config[key] = {
                "function": metric.metric_func,
                "weight": metric.weight
            }

        return scoring_config

    # -------------------------------------------
    # Current Account Size
    # -------------------------------------------

    def prepare_account_sizes(self, filtered_ledger, now_ms):
        """Calculates percentiles for most recent account size"""
        if now_ms is None:
            now_ms = TimeUtil.now_in_millis()

        account_sizes = []
        account_size_object = self.contract_manager.miner_account_sizes_dict()

        # Calculate raw PnL for each miner
        for hotkey, _ in filtered_ledger.items():

            # Fetch most recent account size even if it isn't valid yet for scoring
            account_size = self.contract_manager.get_miner_account_size(hotkey, now_ms, most_recent=True)
            if account_size is None:
                account_size = ValiConfig.MIN_CAPITAL
            else:
                account_size = max(account_size, ValiConfig.MIN_CAPITAL)
            account_sizes.append((hotkey, account_size))

        account_size_ranks = self.rank_dictionary(account_sizes)
        account_size_percentiles = self.percentile_rank_dictionary(account_sizes)
        account_sizes_dict = dict(account_sizes)


        # Build result dictionary
        result = {}
        for hotkey in account_sizes_dict:
            result[hotkey] = {
                "account_size_statistics": {
                    "value": account_sizes_dict.get(hotkey),
                    "rank": account_size_ranks.get(hotkey),
                    "percentile": account_size_percentiles.get(hotkey),
                "account_sizes": account_size_object.get(hotkey, [])
                }
            }

        return result

    # -------------------------------------------
    # Raw PnL Calculation
    # -------------------------------------------
    def calculate_pnl_info(self, filtered_ledger: Dict[str, Dict[str, PerfLedger]]) -> Dict[str, Dict[str, float]]:
        """Calculate raw PnL values, rankings and percentiles for all miners."""

        raw_pnl_values = []
        # Calculate raw PnL for each miner
        for hotkey, ledgers in filtered_ledger.items():
            portfolio_ledger = ledgers.get(TP_ID_PORTFOLIO)
            if portfolio_ledger:
                raw_pnl = LedgerUtils.raw_pnl(portfolio_ledger)
                raw_pnl_values.append((hotkey, raw_pnl))
            else:
                raw_pnl_values.append((hotkey, 0.0))
        
        # Calculate rankings and percentiles
        ranks = self.rank_dictionary(raw_pnl_values)
        percentiles = self.percentile_rank_dictionary(raw_pnl_values)
        values_dict = dict(raw_pnl_values)
        
        # Build result dictionary
        result = {}
        for hotkey in values_dict:
            result[hotkey] = {
                "raw_pnl": {
                    "value": values_dict.get(hotkey),
                    "rank": ranks.get(hotkey),
                    "percentile": percentiles.get(hotkey)
                }
            }
        
        return result

    # -------------------------------------------
    # Asset Class Performance
    # -------------------------------------------
    def miner_asset_class_scores(
            self,
            hotkey: str,
            asset_softmaxed_scores: dict[str, dict[str, float]],
            asset_class_weights: dict[str, float] = None
    ) -> dict[str, dict[str, float]]:
        """
        Extract individual miner's scores and rankings for each asset class.

        Args:
            hotkey: The miner's hotkey
            asset_softmaxed_scores: A dictionary with softmax scores for each miner within each asset class
            asset_class_weights: A dictionary with emission weights for each asset class

        Returns:
            asset_class_data: dict with asset class as key and score/rank/percentile info as value
        """
        asset_class_data = {}

        for asset_class, miner_scores in asset_softmaxed_scores.items():
            if hotkey in miner_scores:
                asset_class_percentiles = self.percentile_rank_dictionary(miner_scores.items())
                asset_class_ranks = self.rank_dictionary(miner_scores.items())

                # Score is the only one directly impacted by the asset class weighting, each score element should show the overall scoring contribution
                miner_score = miner_scores.get(hotkey)
                asset_emission = asset_class_weights.get(asset_class, 0.0)
                aggregated_score = miner_score * asset_emission

                asset_class_data[asset_class] = {
                    "score": aggregated_score,
                    "rank": asset_class_ranks.get(hotkey, 0),
                    "percentile": asset_class_percentiles.get(hotkey, 0.0) * 100
                }

        return asset_class_data

    # -------------------------------------------
    # Generate final data
    # -------------------------------------------
    def generate_miner_statistics_data(
        self,
        time_now: int = None,
        checkpoints: bool = True,
        risk_report: bool = False,
        selected_miner_hotkeys: List[str] = None,
        final_results_weighting = True,
        bypass_confidence: bool = False
    ) -> Dict[str, Any]:

        if time_now is None:
            time_now = TimeUtil.now_in_millis()

        # ChallengePeriod: success + testing
        challengeperiod_testing_dict = self.challengeperiod_manager.get_testing_miners()
        challengeperiod_success_dict = self.challengeperiod_manager.get_success_miners()
        challengeperiod_probation_dict = self.challengeperiod_manager.get_probation_miners()
        challengeperiod_plagiarism_dict = self.challengeperiod_manager.get_plagiarism_miners()

        sorted_challengeperiod_testing = dict(sorted(challengeperiod_testing_dict.items(), key=lambda x: x[1]))
        sorted_challengeperiod_success = dict(sorted(challengeperiod_success_dict.items(), key=lambda x: x[1]))
        sorted_challengeperiod_probation = dict(sorted(challengeperiod_probation_dict.items(), key=lambda x: x[1]))
        sorted_challengeperiod_plagiarism = dict(sorted(challengeperiod_plagiarism_dict.items(), key=lambda x: x[1]))

        challengeperiod_testing_hotkeys = list(sorted_challengeperiod_testing.keys())
        challengeperiod_success_hotkeys = list(sorted_challengeperiod_success.keys())
        challengeperiod_probation_hotkeys = list(sorted_challengeperiod_probation.keys())
        challengeperiod_plagiarism_hotkeys = list(sorted_challengeperiod_plagiarism.keys())

        challengeperiod_eval_hotkeys  = challengeperiod_testing_hotkeys + challengeperiod_probation_hotkeys + challengeperiod_plagiarism_hotkeys

        all_miner_hotkeys = list(set(challengeperiod_testing_hotkeys + challengeperiod_success_hotkeys + challengeperiod_probation_hotkeys + challengeperiod_plagiarism_hotkeys))
        if selected_miner_hotkeys is None:
            selected_miner_hotkeys = all_miner_hotkeys

        # Filter ledger/positions
        filtered_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=all_miner_hotkeys)
        filtered_positions, _ = self.position_manager.filtered_positions_for_scoring(all_miner_hotkeys)

        maincomp_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=[*challengeperiod_success_hotkeys, *challengeperiod_probation_hotkeys])  # ledger of all miners in maincomp, including probation
        asset_classes = list(AssetSegmentation.distill_asset_classes(ValiConfig.ASSET_CLASS_BREAKDOWN))
        asset_class_min_days = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            maincomp_ledger, asset_classes
        )
        bt.logging.info(f"generate_minerstats asset_class_min_days: {asset_class_min_days}")
        all_miner_account_sizes = self.contract_manager.get_all_miner_account_sizes(timestamp_ms=time_now)
        success_competitiveness, asset_softmaxed_scores = Scoring.score_miner_asset_classes(
            ledger_dict=filtered_ledger,
            positions=filtered_positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=time_now,
            weighting=final_results_weighting,
            all_miner_account_sizes=all_miner_account_sizes
        ) # returns asset competitiveness dict, asset softmaxed scores

        # Get asset class weights from config
        asset_class_weights = {
            asset_class: config.get('emission', 0.0)
            for asset_class, config in ValiConfig.ASSET_CLASS_BREAKDOWN.items()
        }
        asset_aggregated_scores = Scoring.asset_class_score_aggregation(
            asset_softmaxed_scores
        )

        # For weighting logic: gather "successful" checkpoint-based results
        successful_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=challengeperiod_success_hotkeys)
        successful_positions, _ = self.position_manager.filtered_positions_for_scoring(challengeperiod_success_hotkeys)

        # Compute the checkpoint-based weighting for successful miners
        checkpoint_results = Scoring.compute_results_checkpoint(
            successful_ledger,
            successful_positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=time_now,
            verbose=False,
            weighting=final_results_weighting,
            metrics=self.extract_scoring_config(self.metrics_calculator.metrics),
            all_miner_account_sizes=all_miner_account_sizes
        )  # returns list of (hotkey, weightVal)

        # Only used for testing weight calculation
        testing_ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=challengeperiod_eval_hotkeys)
        testing_positions, _ = self.position_manager.filtered_positions_for_scoring(challengeperiod_eval_hotkeys)

        # Compute testing miner scores
        testing_checkpoint_results = Scoring.compute_results_checkpoint(
            testing_ledger,
            testing_positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=time_now,
            verbose=False,
            weighting=final_results_weighting,
            metrics= self.extract_scoring_config( self.metrics_calculator.metrics),
            all_miner_account_sizes=all_miner_account_sizes
        )

        challengeperiod_scores = Scoring.score_testing_miners(testing_ledger, testing_checkpoint_results)

        # Combine them
        combined_weights_list = checkpoint_results + challengeperiod_scores

        ######## TEMPORARY LOGIC FOR BLOCK REMOVALS ON MINERS - REMOVE WHEN CLEARED
        dtao_registration_bug_registrations = set(['5Dvep8Psc5ASQf6jGJHz5qsi8x1HS2sefRbkKxNNjPcQYPfH', '5DnViSacXqrP8FnQMtpAFGyahUPvU2A6pbrX7wcexb3bmVjb', '5Grgb5e4aHrGzhAd1ZSFQwUHQSM5yaJw5Dp7T7ss7yLY17jB',
         '5FbaR3qjbbnYpkDCkuh4TUqqen1UMSscqjmhoDWQgGRh189o', '5FqSBwa7KXvv8piHdMyVbcXQwNWvT9WjHZGHAQwtoGVQD3vo', '5F25maVPbzV4fojdABw5Jmawr43UAc5uNRJ3VjgKCUZrYFQh',
         '5DjqgrgQcKdrwGDg7RhSkxjnAVWwVgYTBodAdss233s3zJ6T', '5FpypsPpSFUBpByFXMkJ34sV88PRjAKSSBkHkmGXMqFHR19Q', '5CXsrszdjWooHK3tfQH4Zk6spkkSsduFrEHzMemxU7P2wh7H',
         '5EFbAfq4dsGL6Fu6Z4jMkQUF3WiGG7XczadUvT48b9U7gRYW', '5GyBmAHFSFRca5BYY5yHC3S8VEcvZwgamsxyZTXep5prVz9f', '5EXWvBCADJo1JVv6jHZPTRuV19YuuJBnjG3stBm3bF5cR9oy',
         '5HDjwdba5EvQy27CD6HksabaHaPP4NSHLLaH2o9CiD3aA5hv', '5EWSKDmic7fnR89AzVmqLL14YZbJK53pxSc6t3Y7qbYm5SaV', '5DQ1XPp8KuDEwGP1eC9eRacpLoA1RBLGX22kk5vAMBtp3kGj',
         '5ERorZ39jVQJ7cMx8j8osuEV8dAHHCbpx8kGZP4Ygt5dxf93', '5GsNcT3ENpxQdNnM2LTSC5beBneEddZjpUhNVCcrdUbicp1w'])

        combined_weights_dict = dict(combined_weights_list)
        for hotkey, w_val in combined_weights_dict.items():
            if hotkey in dtao_registration_bug_registrations:
                combined_weights_dict[hotkey] = 0.0

        # Rebuild the list
        combined_weights_list = list(combined_weights_dict.items())
        #################################

        weights_dict = dict(combined_weights_list)
        weights_rank = self.rank_dictionary(combined_weights_list)
        weights_percentile = self.percentile_rank_dictionary(combined_weights_list)

        # Load plagiarism once
        plagiarism_scores = self.plagiarism_detector.get_plagiarism_scores_from_disk()

        # Prepare data for each miner
        miner_data = {}
        for hotkey in selected_miner_hotkeys:
            miner_data[hotkey] = self.prepare_miner_data(hotkey, filtered_ledger, filtered_positions, time_now)

        # Compute the base and augmented scores
        base_scores = self.calculate_all_scores(miner_data, asset_class_min_days, ScoreType.BASE, bypass_confidence, time_now)
        augmented_scores = self.calculate_all_scores(miner_data, asset_class_min_days, ScoreType.AUGMENTED, bypass_confidence, time_now)

        # For visualization
        daily_returns_dict = self.calculate_all_daily_returns(filtered_ledger, return_type='simple')

        # Calculate raw PnL values with rankings and percentiles
        raw_pnl_dict = self.calculate_pnl_info(filtered_ledger)

        # Gather account sizes
        account_size_dict = self.prepare_account_sizes(filtered_ledger, now_ms=time_now)

        # Also compute penalty breakdown (for display in final "penalties" dict).
        penalty_breakdown = self.calculate_penalties_breakdown(miner_data)

        # Risk profiling
        risk_profile_dict = self.calculate_risk_profile(miner_data)
        risk_profile_report = self.calculate_risk_report(miner_data)

        # Build the final list
        results = []
        for hotkey in selected_miner_hotkeys:

            # ChallengePeriod info
            challengeperiod_info = {}
            if hotkey in sorted_challengeperiod_testing:
                cp_start = sorted_challengeperiod_testing[hotkey]
                cp_end = cp_start + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
                remaining = cp_end - time_now
                challengeperiod_info = {
                    "status": "testing",
                    "start_time_ms": cp_start,
                    "remaining_time_ms": max(remaining, 0)
                }
            elif hotkey in sorted_challengeperiod_success:
                cp_start = sorted_challengeperiod_success[hotkey]
                challengeperiod_info = {
                    "status": "success",
                    "start_time_ms": cp_start
                }
            elif hotkey in sorted_challengeperiod_probation:
                bucket_start = sorted_challengeperiod_probation[hotkey]
                bucket_end = bucket_start + ValiConfig.PROBATION_MAXIMUM_MS
                remaining = bucket_end - time_now
                challengeperiod_info = {
                    "status": "probation",
                    "start_time_ms": bucket_start,
                    "remaining_time_ms": max(remaining, 0)
                }
            elif hotkey in sorted_challengeperiod_plagiarism:
                bucket_start = sorted_challengeperiod_plagiarism[hotkey]
                bucket_end = bucket_start + ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS
                remaining = bucket_end - time_now
                challengeperiod_info = {
                    "status": "plagiarism",
                    "start_time_ms": bucket_start,
                    "remaining_time_ms": max(remaining, 0)
                }

            # Build a small function to extract ScoreResult -> dict for each metric
            def build_scores_dict(metric_set: Dict[str, Dict[str, ScoreResult]]) -> Dict[str, Dict[str, float]]:
                out = {}
                for subcategory, metric_scores in metric_set.items():
                    out[subcategory] = {}
                    for metric_name, hotkey_map in metric_scores.items():
                        sr = hotkey_map.get(hotkey)
                        if sr is not None:
                            out[subcategory][metric_name] = sr.to_dict()
                        else:
                            out[subcategory][metric_name] = {}
                return out

            base_dict = build_scores_dict(base_scores)
            augmented_dict = build_scores_dict(augmented_scores)

            # Extra data
            extra = miner_data[hotkey].get("extra_data", {})

            # Volatility
            volatility_subdict = {
                "annual": extra.get("annual_volatility"),
                "annual_downside": extra.get("annual_downside_volatility"),
            }
            # Drawdowns
            drawdowns_subdict = {
                "instantaneous_max_drawdown": extra.get("instantaneous_max_drawdown"),
                "daily_max_drawdown": extra.get("daily_max_drawdown"),
            }
            # Engagement
            engagement_subdict = {
                "n_checkpoints": extra.get("checkpoints_info", {}).get("n_checkpoints"),
                "n_positions": extra.get("positions_info", {}).get("n_positions"),
                "position_duration": extra.get("positions_info", {}).get("positional_duration"),
                "checkpoint_durations": extra.get("checkpoints_info", {}).get("checkpoint_durations"),
                "minimum_days_boolean": extra.get("minimum_days_boolean"),
                "percentage_profitable": extra.get("positions_info", {}).get("percentage_profitable"),
            }
            # Raw PnL
            raw_pnl_info = raw_pnl_dict.get(hotkey)

            # Account Size
            account_sizes = account_size_dict.get(hotkey)

            # Plagiarism
            plagiarism_val = plagiarism_scores.get(hotkey)

            # Weight
            w_val = weights_dict.get(hotkey)
            w_rank = weights_rank.get(hotkey)
            w_pct = weights_percentile.get(hotkey)

            # Penalties breakdown for display
            pen_break = penalty_breakdown.get(hotkey, {})

            # Purely for visualization purposes
            daily_returns = daily_returns_dict.get(hotkey, {})
            daily_returns_list = [{"date": date, "value": value * 100} for date, value in daily_returns.items()]

            # Risk Profile
            risk_profile_single_dict = risk_profile_dict.get(hotkey, {})

            # Asset Class Performance
            asset_class_performance = self.miner_asset_class_scores(
                hotkey,
                asset_softmaxed_scores,
                asset_class_weights
            )

            final_miner_dict = {
                "hotkey": hotkey,
                "challengeperiod": challengeperiod_info,
                "scores": base_dict,
                "augmented_scores": augmented_dict,
                "daily_returns": daily_returns_list,
                "volatility": volatility_subdict,
                "drawdowns": drawdowns_subdict,
                "plagiarism": plagiarism_val,
                "engagement": engagement_subdict,
                "risk_profile": risk_profile_single_dict,
                "asset_class_performance": asset_class_performance,
                "pnl_info": raw_pnl_info,
                "account_size_info": account_sizes,
                "penalties": {
                    "drawdown_threshold": pen_break.get("drawdown_threshold", 1.0),
                    "risk_profile": pen_break.get("risk_profile", 1.0),
                    "total": pen_break.get("total", 1.0),
                },
                "weight": {
                    "value": w_val,
                    "rank": w_rank,
                    "percentile": w_pct,
                },
            }

            if risk_report:
                final_miner_dict["risk_profile_report"] = risk_profile_report.get(hotkey, {})

            # Optionally attach actual checkpoints (like the original first script)
            if checkpoints:
                ledger_obj = miner_data[hotkey].get("cumulative_ledger")
                if ledger_obj and hasattr(ledger_obj, "cps"):
                    final_miner_dict["checkpoints"] = ledger_obj.cps

            results.append(final_miner_dict)

        # (Optional) sort by weight rank if you want the final data sorted in that manner:
        # Filter out any miners lacking a weight, then sort.
        # If you want to keep them all, remove this filtering:
        results_with_weight = [r for r in results if r["weight"]["rank"] is not None]
        # Sort by ascending rank
        results_sorted = sorted(results_with_weight, key=lambda x: x["weight"]["rank"])

        # network level data
        network_data_dict = {
            "asset_competitiveness": success_competitiveness
        }

        # If you'd prefer not to filter out those without weight, you could keep them at the end
        # Or you can unify them in a single list. For simplicity, let's keep it consistent:
        final_dict = {
            'version': ValiConfig.VERSION,
            'created_timestamp_ms': time_now,
            'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
            'data': results_sorted,
            'constants': self.get_printable_config(),
            'network_data': network_data_dict
        }
        return final_dict

    # -------------------------------------------
    # Printable config
    # -------------------------------------------
    def get_printable_config(self) -> Dict[str, Any]:
        """Get printable configuration values."""
        config_data = dict(ValiConfig.__dict__)
        printable_config = {
            key: value for key, value in config_data.items()
            if isinstance(value, (int, float, str))
               and key not in ['BASE_DIR', 'base_directory']
        }
        
        # Add asset class breakdown with subcategory weights
        printable_config['asset_class_breakdown'] = ValiConfig.ASSET_CLASS_BREAKDOWN
        printable_config['trade_pairs_by_subcategory'] = TradePair.subcategories()
        
        return printable_config

    # -------------------------------------------
    # Write to disk, memory
    # -------------------------------------------
    def generate_request_minerstatistics(self, time_now: int, checkpoints: bool = True, risk_report: bool = False, bypass_confidence: bool = False, custom_output_path=None):
        final_dict = self.generate_miner_statistics_data(time_now, checkpoints=checkpoints, risk_report=risk_report, bypass_confidence=bypass_confidence)
        if custom_output_path:
            output_file_path = custom_output_path
        else:
            output_file_path = ValiBkpUtils.get_miner_stats_dir()
        ValiBkpUtils.write_file(output_file_path, final_dict)
        
        # Create version without checkpoints for API optimization
        final_dict_no_checkpoints = self._create_statistics_without_checkpoints(final_dict)
        
        # Store compressed JSON payloads for immediate API response (memory efficient)
        json_with_checkpoints = json.dumps(final_dict, cls=CustomEncoder)
        json_without_checkpoints = json.dumps(final_dict_no_checkpoints, cls=CustomEncoder)
        
        # Compress both versions for efficient storage and transfer
        compressed_with_checkpoints = gzip.compress(json_with_checkpoints.encode('utf-8'))
        compressed_without_checkpoints = gzip.compress(json_without_checkpoints.encode('utf-8'))
        
        # Only store compressed payloads - saves ~22MB of uncompressed data per validator
        self.miner_statistics['stats_compressed_with_checkpoints'] = compressed_with_checkpoints
        self.miner_statistics['stats_compressed_without_checkpoints'] = compressed_without_checkpoints

    def _create_statistics_without_checkpoints(self, stats_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create a copy of statistics with checkpoints removed from all miner data."""
        import copy
        stats_no_checkpoints = copy.deepcopy(stats_dict)
        
        # Remove checkpoints from each miner's data
        for element in stats_no_checkpoints.get("data", []):
            element.pop("checkpoints", None)
        
        return stats_no_checkpoints
    
    def get_compressed_statistics(self, include_checkpoints: bool = True) -> bytes:
        """Get pre-compressed statistics payload for immediate API response."""
        if include_checkpoints:
            return self.miner_statistics.get('stats_compressed_with_checkpoints', None)
        else:
            return self.miner_statistics.get('stats_compressed_without_checkpoints', None)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bt.logging.enable_info()
    all_hotkeys = ValiBkpUtils.get_directories_in_dir(ValiBkpUtils.get_miner_dir())
    print('N hotkeys:', len(all_hotkeys))
    metagraph = MockMetagraph(all_hotkeys)

    perf_ledger_manager = PerfLedgerManager(metagraph)
    elimination_manager = EliminationManager(metagraph, None, None)
    position_manager = PositionManager(
        metagraph, None,
        elimination_manager=elimination_manager,
        challengeperiod_manager=None,
        perf_ledger_manager=perf_ledger_manager
    )
    challengeperiod_manager = ChallengePeriodManager(metagraph, None, position_manager=position_manager)
    contract_manager = ValidatorContractManager(config=None, position_manager=position_manager)

    # Cross-wire references
    elimination_manager.position_manager = position_manager
    position_manager.challengeperiod_manager = challengeperiod_manager
    elimination_manager.challengeperiod_manager = challengeperiod_manager
    challengeperiod_manager.position_manager = position_manager
    perf_ledger_manager.position_manager = position_manager
    perf_ledger_manager.contract_manager = contract_manager

    subtensor_weight_setter = SubtensorWeightSetter(
        metagraph=metagraph,
        running_unit_tests=False,
        position_manager=position_manager,
        contract_manager=contract_manager,
    )
    plagiarism_detector = PlagiarismDetector(metagraph, None, position_manager=position_manager)

    msm = MinerStatisticsManager(position_manager, subtensor_weight_setter, plagiarism_detector, contract_manager=contract_manager)
    pwd = os.getcwd()
    custom_output_path = os.path.join(pwd, 'debug_miner_statistics.json')
    msm.generate_request_minerstatistics(TimeUtil.now_in_millis(), True, custom_output_path=custom_output_path)
    # Confirm output path and ability to read file
    if os.path.exists(custom_output_path):
        with open(custom_output_path, 'r') as f:
            data = json.load(f)
            print('Generated miner statistics:', custom_output_path)
    else:
        print(f"Output file not found at {custom_output_path}")

