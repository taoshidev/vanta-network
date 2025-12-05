# developer: trdougherty

from dataclasses import dataclass
import math
from typing import List, Tuple, Callable

from vali_objects.enums.misc import PenaltyInputType
from vali_objects.vali_dataclasses.position import Position
import copy
from collections import defaultdict

import numpy as np
from scipy.stats import percentileofscore

from vali_objects.contract.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger, TP_ID_PORTFOLIO
from time_util.time_util import TimeUtil
from vali_objects.position_management.position_utils import PositionFiltering
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.utils.metrics import Metrics
from vali_objects.position_management.position_utils import PositionPenalties
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.vali_config import TradePairCategory
import bittensor as bt


@dataclass
class PenaltyConfig:
    function: Callable
    input_type: PenaltyInputType


class Scoring:
    # Set the scoring configuration
    scoring_config = {
        'calmar': {
            'function': Metrics.calmar,
            'weight': ValiConfig.SCORING_CALMAR_WEIGHT
        },
        'sharpe': {
            'function': Metrics.sharpe,
            'weight': ValiConfig.SCORING_SHARPE_WEIGHT
        },
        'omega': {
            'function': Metrics.omega,
            'weight': ValiConfig.SCORING_OMEGA_WEIGHT
        },
        'sortino': {
            'function': Metrics.sortino,
            'weight': ValiConfig.SCORING_SORTINO_WEIGHT
        },
        'statistical_confidence': {
            'function': Metrics.statistical_confidence,
            'weight': ValiConfig.SCORING_STATISTICAL_CONFIDENCE_WEIGHT
        },
        'pnl': {
            'function': Metrics.pnl_score,
            'weight': ValiConfig.SCORING_PNL_WEIGHT
        }
    }

    # Define the configuration with input types
    penalties_config = {
        'drawdown_threshold': PenaltyConfig(
            function=LedgerUtils.max_drawdown_threshold_penalty,
            input_type=PenaltyInputType.LEDGER
        ),
        'risk_profile': PenaltyConfig(
            function=PositionPenalties.risk_profile_penalty,
            input_type=PenaltyInputType.POSITIONS
        ),
        'min_collateral': PenaltyConfig(
            function=ValidatorContractManager.min_collateral_penalty,
            input_type=PenaltyInputType.COLLATERAL
        )
    }

    @staticmethod
    def compute_results_checkpoint(
            ledger_dict: dict[str, dict[str, PerfLedger]],
            full_positions: dict[str, list[Position]],
            asset_class_min_days: dict[str, int],
            evaluation_time_ms: int = None,
            verbose=True,
            weighting=False,
            metrics=None,
            all_miner_account_sizes: dict[str, float]=None
    ) -> List[Tuple[str, float]]:
        bt.logging.info(f"compute_results_checkpoint called with {len(ledger_dict)} miners")

        if len(ledger_dict) == 0:
            bt.logging.debug("No results to compute, returning empty list")
            return []

        if len(ledger_dict) == 1:
            miner = list(ledger_dict.keys())[0]
            bt.logging.info(
                f"compute_results_checkpoint - Only one miner: {miner}, returning 1.0 for the solo miner weight")
            return [(miner, 1.0)]

        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        filtered_positions = PositionFiltering.filter(
            full_positions,
            evaluation_time_ms=evaluation_time_ms
        )
        if metrics is not None:
            Scoring.scoring_config = metrics
        # Compute miner penalties
        miner_penalties = Scoring.miner_penalties(filtered_positions, ledger_dict, all_miner_account_sizes)

        # Miners with full penalty
        full_penalty_miner_scores: list[tuple[str, float]] = [
            (miner, 0) for miner, penalty in miner_penalties.items() if penalty == 0
        ]

        # Run scoring functions for each miner in each asset class
        _, asset_softmaxed_scores = Scoring.score_miner_asset_classes(
            ledger_dict=ledger_dict,
            positions=full_positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=evaluation_time_ms,
            weighting=weighting,
            all_miner_account_sizes=all_miner_account_sizes
        )
        bt.logging.info(f"asset_softmaxed_scores has {len(asset_softmaxed_scores)} asset classes")

        # Now combine the percentile scores using asset class emission weights
        asset_aggregated_scores = Scoring.asset_class_score_aggregation(asset_softmaxed_scores)
        bt.logging.info(f"asset_aggregated_scores has {len(asset_aggregated_scores)} miners")

        # Force good performance of all error metrics
        combined_weighed = asset_aggregated_scores + full_penalty_miner_scores
        bt.logging.info(f"combined_weighed has {len(combined_weighed)} entries (aggregated: {len(asset_aggregated_scores)}, penalties: {len(full_penalty_miner_scores)})")
        combined_scores = dict(combined_weighed)

        # Normalize the scores
        normalized_scores = Scoring.normalize_scores(combined_scores)
        bt.logging.info(f"normalized_scores has {len(normalized_scores)} miners, returning results")
        return sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def score_miner_asset_classes(
            ledger_dict: dict[str, dict[str, PerfLedger]],
            positions: dict[str, list[Position]],
            asset_class_min_days: dict[TradePairCategory, int],
            evaluation_time_ms: int = None,
            weighting=False,
            all_miner_account_sizes: dict[str, float]=None
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """
        returns:
        asset_competitiveness: dictionary with asset classes as keys and their competitiveness as values.
        asset_miner_softmaxed_scores: A dictionary with softmax scores for each miner within each asset class
        """
        bt.logging.info(f"score_miner_asset_classes called with {len(ledger_dict)} miners")
        if len(ledger_dict) <= 1:
            bt.logging.info("score_miner_asset_classes: <= 1 miner, returning empty dicts (no competition)")
            return {}, {}

        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()

        # Run all scoring functions
        asset_penalized_scores_dict = Scoring.score_miners(
            ledger_dict=ledger_dict,
            positions=positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=evaluation_time_ms,
            weighting=weighting,
            all_miner_account_sizes=all_miner_account_sizes
        )

        # Combine and penalize scores
        asset_combined_scores = Scoring.combine_scores(asset_penalized_scores_dict)
        asset_competitiveness = AssetSegmentation.asset_competitiveness_dictionary(asset_combined_scores)
        bt.logging.debug(f"Asset competitiveness: {asset_competitiveness}")

        # Now we probably want to apply the softmax to the asset combined scores
        asset_miner_softmaxed_scores = Scoring.softmax_by_asset(asset_combined_scores)

        return asset_competitiveness, asset_miner_softmaxed_scores

    @staticmethod
    def score_miners(
            ledger_dict: dict[str, dict[str, PerfLedger]],
            positions: dict[str, list[Position]],
            asset_class_min_days: dict[TradePairCategory, int],
            evaluation_time_ms: int = None,
            weighting: bool = False,
            scoring_config: dict[str, dict[str, float]] = None,
            all_miner_account_sizes: dict[str, float] = None
    ) -> dict[TradePairCategory, dict]:
        """
        Scores the miners based on their ledger and positions.
        Args:
            ledger_dict:
            positions:
            asset_class_min_days:
            evaluation_time_ms:
            weighting:

        Returns:
            dict[TradePairCategory, dict]: A dictionary where keys are asset classes and values are dictionaries containing scores and penalties.

        """
        if ledger_dict is None or len(ledger_dict) == 0:
            bt.logging.warning("No ledger provided for scoring, returning empty scores")

        if positions is None or len(positions) == 0:
            bt.logging.warning("No positions provided for scoring, returning empty scores")

        if evaluation_time_ms is None:
            evaluation_time_ms = TimeUtil.now_in_millis()
        if scoring_config is None:
            scoring_config = Scoring.scoring_config

        filtered_positions = PositionFiltering.filter(
            positions,
            evaluation_time_ms=evaluation_time_ms
        )

        # Compute miner penalties
        miner_penalties = Scoring.miner_penalties(filtered_positions, ledger_dict, all_miner_account_sizes)

        # Miners with full penalty
        full_penalty_miners = set([
            miner for miner, penalty in miner_penalties.items() if penalty == 0
        ])

        # We now want to track miner incentive between each asset class
        asset_class_breakdown = ValiConfig.ASSET_CLASS_BREAKDOWN
        asset_classes = AssetSegmentation.distill_asset_classes(asset_class_breakdown)

        segmentation_machine = AssetSegmentation(ledger_dict)

        # This is going to track miner scores on each asset class
        miner_asset_benefit = {}

        for asset_class in asset_classes:
            asset_ledger = segmentation_machine.segmentation(asset_class)
            filtered_ledger_returns = LedgerUtils.ledger_returns_log(asset_ledger)
            days_in_year = segmentation_machine.days_in_year_from_asset_category(asset_class)

            scores_dict = {"metrics": {}}
            for config_name, config in scoring_config.items():
                scores = []
                for miner, returns in filtered_ledger_returns.items():
                    # Get the miner ledger
                    ledger = asset_ledger.get(miner, PerfLedger())

                    # Check if the miner has full penalty - if not include them in the scoring competition
                    if miner in full_penalty_miners:
                        #bt.logging.info(f"Skipping {miner} in {asset_class.value}/{config_name} (full penalty)")
                        continue

                    score = config['function'](
                        log_returns=returns,
                        ledger=ledger,
                        weighting=weighting,
                        days_in_year=days_in_year,
                        min_days=asset_class_min_days[asset_class],
                    )

                    scores.append((miner, float(score)))

                scores_dict["metrics"][config_name] = {
                    "scores": scores[:],
                    "weight": config["weight"]
                }

            scores_dict["penalties"] = copy.deepcopy(miner_penalties)
            miner_asset_benefit[asset_class] = scores_dict

        return miner_asset_benefit

    @staticmethod
    def combine_scores(scoring_dict: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
        """
        Combines scores and penalties for each of the asset classes into a single score for each asset class.
        Args:
            scoring_dict:

        Returns:

        """

        asset_combined_scores = {}
        for asset_class, asset_scores in scoring_dict.items():
            combined_scores = {}
            for config_name, config in asset_scores["metrics"].items():

                percentile_scores = Scoring.miner_scores_percentiles(config["scores"])

                for miner, percentile_rank in percentile_scores:
                    if miner not in combined_scores:
                        combined_scores[miner] = 0
                    combined_scores[miner] += config['weight'] * percentile_rank  # + (1 - config['weight'])

            # Now applying the penalties post scoring
            for miner, penalty in asset_scores["penalties"].items():
                if miner in combined_scores:
                    combined_scores[miner] *= penalty

            asset_combined_scores[asset_class] = combined_scores

        return asset_combined_scores

    @staticmethod
    def miner_penalties(
            hotkey_positions: dict[str, list[Position]],
            ledger_dict: dict[str, dict[str, PerfLedger]],
            all_miner_account_sizes: dict[str, float]
    ) -> dict[str, float]:
        # Compute miner penalties
        miner_penalties = {}

        empty_ledger_miners = []
        for miner, ledger in ledger_dict.items():
            positions = hotkey_positions.get(miner, [])

            if not ledger and len(positions):
                empty_ledger_miners.append((miner, len(positions)))
                continue

            portfolio_ledger = ledger.get(TP_ID_PORTFOLIO) if ledger else PerfLedger()

            miner_account_size = all_miner_account_sizes.get(miner, 0)
            if miner_account_size is None:
                miner_account_size = 0

            cumulative_penalty = 1
            for penalty_name, penalty_config in Scoring.penalties_config.items():
                # Apply penalty based on its input type
                penalty = 1
                if penalty_config.input_type == PenaltyInputType.LEDGER:
                    penalty = penalty_config.function(portfolio_ledger)
                elif penalty_config.input_type == PenaltyInputType.POSITIONS:
                    penalty = penalty_config.function(positions)
                elif penalty_config.input_type == PenaltyInputType.COLLATERAL:
                    penalty = penalty_config.function(miner_account_size)

                cumulative_penalty *= penalty

            miner_penalties[miner] = cumulative_penalty

        if empty_ledger_miners:
            bt.logging.warning(
                f"Unexpectedly skipping miners with empty ledgers [(hk, n_positions)]: {empty_ledger_miners}")

        return miner_penalties

    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """
        Args: scores: dict[str, float] - the scores of the miner returns
        """
        # bt.logging.info(f"Normalizing scores: {scores}")
        if len(scores) == 0:
            bt.logging.info("No scores to normalize, returning empty list")
            return {}

        sum_scores = sum(scores.values())
        if sum_scores == 0:
            bt.logging.info("sum_scores is 0, returning empty list")
            return {}

        normalized_scores = {
            miner: (score / sum_scores) for miner, score in scores.items()
        }
        # normalized_scores = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
        return normalized_scores

    @staticmethod
    def base_return(positions: list[Position]) -> float:
        """
        Args:
            positions: list of positions from the miner
        """
        if len(positions) == 0:
            return 0.0

        positional_returns = [math.log(
            max(position.return_at_close, .00001))  # Prevent math domain error
            for position in positions]

        aggregate_return = 0.0
        for positional_return in positional_returns:
            aggregate_return += positional_return

        return aggregate_return

    @staticmethod
    def class_aggregation(asset_miner_scores: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Aggregates the scores of miners across different asset classes.

        Args:
            asset_miner_scores (dict[str, dict[str, float]]): A dictionary where keys are asset classes and values are dictionaries of miner scores.

        Returns:
            dict[str, float]: A dictionary with aggregated scores for each miner across all asset classes.
        """
        asset_class_breakdown: dict[str, float] = ValiConfig.ASSET_CLASS_BREAKDOWN
        aggregated_scores = defaultdict(float)

        for asset_class, miner_scores in asset_miner_scores.items():
            for miner, score in miner_scores.items():
                aggregated_scores[miner] += score * asset_class_breakdown.get(asset_class, 0)

        return dict(aggregated_scores)

    @staticmethod
    def softmax_by_asset(
            asset_miner_scores: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """
        Applies softmax to the scores of miners within each asset class.

        Args:
            asset_miner_scores (dict[str, dict[str, float]]): A dictionary where keys are asset classes and values are dictionaries of miner scores.

        Returns:
            dict[str, dict[str, float]]: A dictionary with softmax scores for each miner within each asset class.
        """
        softmaxed_scores = {}
        for asset_class, miner_scores in asset_miner_scores.items():
            sorted_returns = sorted(miner_scores.items(), key=lambda x: x[1], reverse=True)
            softmaxed_scores[asset_class] = dict(Scoring.softmax_scores(sorted_returns))

        return softmaxed_scores

    #TODO Remove this since scoring will change
    @staticmethod
    def asset_class_score_aggregation(
            miner_asset_scores: dict[TradePairCategory, dict[str, float]]
    ) -> list[tuple[str, float]]:
        """
        Aggregates the softmax scores of miners across different asset classes using emission weights.

        Args:
            miner_asset_scores (dict[TradePairCategory, dict[str, float]]): A dictionary where keys are asset classes
                and values are dictionaries of miner scores.

        Returns:
            list[tuple[str, float]]: A list of tuples with miner names and their aggregated softmax scores.
        """
        asset_class_breakdown = ValiConfig.ASSET_CLASS_BREAKDOWN
        aggregated_scores = defaultdict(float)

        # Aggregate scores weighted by asset class emission
        for asset_class, scores in miner_asset_scores.items():
            asset_class_emission = asset_class_breakdown.get(asset_class, {}).get('emission', 0.0)

            if asset_class_emission == 0:
                bt.logging.warning(f"Asset class {asset_class} has no emission. Please report this issue!")

            for miner, score in scores.items():
                if miner not in aggregated_scores:
                    aggregated_scores[miner] = 0.0

                aggregated_scores[miner] += score * asset_class_emission
        return sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def softmax_scores(returns: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Assign weights to the returns based on their relative position and apply softmax with a temperature parameter.

        The softmax function is used to convert the scores into probabilities that sum to 1.
        Subtracting the max value from the scores before exponentiation improves numerical stability.

        Parameters:
        returns (list[tuple[str, float]]): List of tuples with miner names and their scores.
        temperature (float): Temperature parameter to control the sharpness of the softmax distribution. Default is 1.0.

        Returns:
        list[tuple[str, float]]: List of tuples with miner names and their softmax weights.
        """
        epsilon = ValiConfig.EPSILON
        temperature = ValiConfig.SOFTMAX_TEMPERATURE

        if not returns:
            bt.logging.debug("No returns to score, returning empty list")
            return []

        if len(returns) == 1:
            bt.logging.info("softmax_scores - Only one miner, returning 1.0 for the solo miner weight")
            return [(returns[0][0], 1.0)]

        # Extract scores and apply softmax with temperature
        names, scores = zip(*returns)
        scores = np.array(scores, dtype=np.float64)

        # Mask for non-zero scores
        nonzero_mask = scores != 0
        valid_scores = scores[nonzero_mask]

        # Softmax only on non-zero scores
        if valid_scores.size == 0:
            # All scores are zero, return zero weights
            return [(name, 0.0) for name in names]

        # Numerically stable softmax
        shifted_scores = (valid_scores - np.max(valid_scores)) / temperature
        exp_scores = np.exp(shifted_scores)
        softmax_scores = exp_scores / (np.sum(exp_scores) + epsilon)

        # Reinsert weights with 0.0 for zero-score items
        result = []
        softmax_iter = iter(softmax_scores)
        for name, is_nonzero in zip(names, nonzero_mask):
            weight = next(softmax_iter) if is_nonzero else 0.0
            result.append((name, float(weight)))

        return result

    @staticmethod
    def miner_scores_percentiles(miner_scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Args: miner_scores: list[tuple[str, float]] - the scores of the miners
        """
        if len(miner_scores) == 0:
            bt.logging.debug("No miner scores to compute percentiles, returning empty list")
            return []

        if len(miner_scores) == 1:
            miner, score = miner_scores[0]
            bt.logging.info(
                f"miner_scores_percentiles - Only one miner: {miner}, returning 1.0 for the solo miner weight")
            return [(miner, 1.0)]

        miner_hotkeys = []
        scores = []

        for miner, score in miner_scores:
            miner_hotkeys.append(miner)
            scores.append(score)

        percentiles = percentileofscore(scores, scores, kind='strict') / 100

        miner_percentiles = list(zip(miner_hotkeys, percentiles))

        return miner_percentiles

    @staticmethod
    def score_testing_miners(ledgers, miner_scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Applies time weighting and distributes challenge weights for prioritization
        Args:
            ledgers: list[tuple[str, float]] - the scores of the miners
            miner_scores: list[tuple[str, float]] - the scores of the miners
        Returns:
            list[tuple[str, float]] - the final weights of the miners
        """

        MIN_WEIGHT = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
        MAX_WEIGHT = ValiConfig.CHALLENGE_PERIOD_MAX_WEIGHT

        if not ledgers or not miner_scores:
            bt.logging.info(f"Ledgers: {ledgers} and miner scores: {miner_scores}, returning empty list")
            return []

        time_weighted = sorted(
            Metrics.time_weighted_scores(ledgers, miner_scores),
            key=lambda x: x[1],
            reverse=True
        )

        num_miners = len(time_weighted)

        distributed = np.linspace(MAX_WEIGHT, MIN_WEIGHT, num=num_miners)

        final_scores = [(miner, float(score)) for (miner, _), score in zip(time_weighted, distributed)]

        return sorted(final_scores, key=lambda x: x[1], reverse=True)
