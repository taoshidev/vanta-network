# developer: trdougherty
import copy
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.vali_config import TradePairCategory

from tests.shared_objects.test_utilities import generate_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.scoring.scoring import Scoring
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import TP_ID_PORTFOLIO


class TestWeights(TestBase):

    def setUp(self):
        super().setUp()

        self.DEFAULT_MINER_HOTKEY = "test_miner"
        self.DEFAULT_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = 1000
        self.DEFAULT_CLOSE_MS = 2000
        self.EVALUATION_TIME_MS = self.DEFAULT_CLOSE_MS + 1

        self.DEFAULT_ORDER_MS = 1000
        self.MS_IN_DAY = 86400000
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD
        self.DEFAULT_POSITION = Position(
            position_type=OrderType.LONG,
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
        )
        # self.DEFAULT_SUBCATEGORY = ForexSubcategory.G1
        self.DEFAULT_ASSET_CLASS = TradePairCategory.FOREX
        self.DEFAULT_ASSET_SCORES = {
            TradePairCategory.FOREX: {
                "miner1": 0.6,
                "miner2": 0.3,
                "miner3": 0.1,
            },
            TradePairCategory.CRYPTO: {
                "miner1": 0.2,
                "miner2": 0.7,
                "miner3": 0.1,
            }
        }
        self.DEFAULT_SCORING_DICT = {
            self.DEFAULT_ASSET_CLASS: {
                "metrics": {
                    "sharpe": {
                        "scores": [("miner1", 1.5), ("miner2", 1.0)],
                        "weight": 0.5
                    },
                    "calmar": {
                        "scores": [("miner1", 2.0), ("miner2", 1.5)],
                        "weight": 0.3
                    }
                },
                "penalties": {
                    "miner1": 1.0,
                    "miner2": 0.9
                }
            }
        }

        asset_classes = list(AssetSegmentation.distill_asset_classes(ValiConfig.ASSET_CLASS_BREAKDOWN))
        self.SUBCATEGORY_MIN_DAYS = {asset_class: ValiConfig.STATISTICAL_CONFIDENCE_MINIMUM_N_CEIL for asset_class in asset_classes}

        self.DEFAULT_LEDGER = generate_ledger(0.1)

    def test_transform_and_scale_results_defaults(self):
        """Test that the transform and scale results works as expected"""
        ledger = {}
        miner_positions = {}
        for i in range(10):
            ledger[f"miner{i}"] = generate_ledger(0.1)
            miner_positions[f"miner{i}"] = [copy.deepcopy(self.DEFAULT_POSITION)]

        # Test the default values
        scaled_transformed_list = Scoring.compute_results_checkpoint(
            ledger,
            miner_positions,
            asset_class_min_days=self.SUBCATEGORY_MIN_DAYS,
            evaluation_time_ms=self.EVALUATION_TIME_MS,
            all_miner_account_sizes={}
        )

        # Check that the result is a list of tuples with string and float elements
        self.assertIsInstance(scaled_transformed_list, list)
        for item in scaled_transformed_list:
            self.assertIsInstance(item, tuple)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

        # Check that the values are sorted in descending order
        values = [x[1] for x in scaled_transformed_list]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_return_no_positions(self):
        self.assertEqual(Scoring.base_return([]), 0.0)

    def test_negative_returns(self):
        """Test that the returns scoring function works properly for only negative returns"""
        positional_returns = [0.8, 0.9, 0.7, 0.8, 0.9, 0.7]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        # Switch to log returns
        base_return = Scoring.base_return(positions)
        self.assertLess(base_return, 0.0)

    def test_positive_returns(self):
        """Test that the returns scoring function works properly for only positive returns"""
        positional_returns = [1.2, 1.1, 1.3, 1.2, 1.1, 1.3]
        positions = []

        for positional_return in positional_returns:
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = positional_return
            positions.append(p)

        # Switch to log returns
        base_return = Scoring.base_return(positions)
        self.assertGreater(base_return, 0.0)

    def test_swing_miners(self):
        """Test that the base_return function works as expected"""
        m1 = []

        # First miner spreads returns equally across all positions
        total_return_1 = 0.2  # 20% total return
        n_positions_1 = 10
        per_position_return_1 = (1 + total_return_1) ** (1 / n_positions_1) - 1

        for i in range(n_positions_1):
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = 1 + per_position_return_1
            m1.append(p)

        # Second miner has small returns in most positions and a large return in one
        m2 = []

        small_return_2 = 0.001
        n_positions_2 = 10
        # Calculate the large return to ensure the total return matches m1
        large_return_2 = (1 + total_return_1) / (1 + small_return_2) ** (n_positions_2 - 1) - 1

        for i in range(n_positions_2 - 1):
            p = copy.deepcopy(self.DEFAULT_POSITION)
            p.return_at_close = 1 + small_return_2
            m2.append(p)

        high_return_position = copy.deepcopy(self.DEFAULT_POSITION)
        high_return_position.return_at_close = 1 + large_return_2
        m2.append(high_return_position)

        # Assertions to compare performance metrics
        self.assertAlmostEqual(Scoring.base_return(m1), Scoring.base_return(m2), places=2)

    def test_no_miners(self):
        """Test when there are no miners in the list"""
        miner_scores = []
        result = Scoring.miner_scores_percentiles(miner_scores)
        self.assertEqual(result, [])

    def test_one_miner(self):
        """Test when there is only one miner in the list"""
        miner_scores = [("miner1", 10.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)
        self.assertEqual(result, [("miner1", 1.0)])

    def test_all_same_scores(self):
        """Test when all miners have the same scores"""
        miner_scores = [("miner1", 10.0), ("miner2", 10.0), ("miner3", 10.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)
        # When all scores are the same, strict percentile gives 0.0 for all miners
        expected_result = [("miner1", 0.0), ("miner2", 0.0), ("miner3", 0.0)]

        for i in range(len(result)):
            self.assertAlmostEqual(result[i][1], expected_result[i][1], places=3)

    def test_zero_value_conditions(self):
        """Test when all scores are zero"""
        miner_scores = [("miner1", 0.0), ("miner2", 0.0), ("miner3", 0.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)
        # When all scores are zero (same), strict percentile gives 0.0 for all miners
        expected_result = [("miner1", 0.0), ("miner2", 0.0), ("miner3", 0.0)]
        for i in range(len(result)):
            self.assertAlmostEqual(result[i][1], expected_result[i][1], places=3)

    def test_typical_conditions(self):
        """Test when miners have different scores"""
        miner_scores = [("miner1", 20.0), ("miner2", 30.0), ("miner3", 10.0), ("miner4", 40.0)]
        result = Scoring.miner_scores_percentiles(miner_scores)

        # Expected percentiles with strict percentile ranking:
        # "miner3" with score 10.0 -> 0.0 (lowest score)
        # "miner1" with score 20.0 -> 0.25 (25th percentile)
        # "miner2" with score 30.0 -> 0.50 (50th percentile)
        # "miner4" with score 40.0 -> 0.75 (75th percentile)
        expected_result = [
            ("miner1", 0.25),
            ("miner2", 0.50),
            ("miner3", 0.0),
            ("miner4", 0.75)
        ]

        self.assertEqual(result, expected_result)
    def test_no_miners_softmax(self):
       """Test when there are no miners in the list"""
       miner_scores = []
       result = Scoring.softmax_scores(miner_scores)
       self.assertEqual(result, [])

    def test_one_miner_softmax(self):
        """Test when there is only one miner in the list"""
        miner_scores = [("miner1", 10.0)]
        result = Scoring.softmax_scores(miner_scores)
        self.assertEqual(result, [("miner1", 1.0)])

    def test_ordering_softmax(self):
        returns = [("miner1", 10.0), ("miner2", 5.0), ("miner3", 1.0), ("miner4", 15.0)]
        result = Scoring.softmax_scores(returns)

        #Sort the list by order of softmax output values
        result.sort(key=lambda x: x[1])
        ordered_keys = [s[0] for s in result]
        self.assertEqual(ordered_keys, ["miner3", "miner2", "miner1", "miner4"])

    def test_sum_to_one_softmax(self):

        returns = [("miner1", 10.0), ("miner2", 5.0), ("miner3", 1.0), ("miner4", 15.0),("miner5", 15.0)]
        result = Scoring.softmax_scores(returns)
        values = [v[1] for v in result]
        self.assertAlmostEqual(sum(values), 1.0, places=3)

    def test_challenge_scoring_no_values(self):

        partial_window = ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS // 3
        full_window = ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS

        ledgers = {
            "miner1": None,
            "miner2": generate_ledger(gain=0.001, loss=0, end_time=partial_window)[TP_ID_PORTFOLIO],
            "miner3": generate_ledger(gain=0.001, loss=0, end_time=full_window)[TP_ID_PORTFOLIO],
        }
        miner_scores = [("miner1", 0.1), ("miner2", 0.5), ("miner3", 0.49)]

        # Test 1: Different combinations of bad input
        self.assertEqual(len(Scoring.score_testing_miners({}, [])), 0)
        self.assertEqual(len(Scoring.score_testing_miners({}, None)), 0)
        self.assertEqual(len(Scoring.score_testing_miners({}, miner_scores)), 0)

        self.assertEqual(len(Scoring.score_testing_miners(None, [])), 0)
        self.assertEqual(len(Scoring.score_testing_miners(None, None)), 0)
        self.assertEqual(len(Scoring.score_testing_miners(None, miner_scores)), 0)

        self.assertEqual(len(Scoring.score_testing_miners(ledgers, [])), 0)
        self.assertEqual(len(Scoring.score_testing_miners(ledgers, None)), 0)

    def test_challenge_scoring_general(self):

        partial_window = ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS // 3
        full_window = ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS

        ledgers = {
            "miner1": None,
            "miner2": generate_ledger(gain=0.001, loss=0, end_time=partial_window)[TP_ID_PORTFOLIO],
            "miner3": generate_ledger(gain=0.001, loss=0, end_time=full_window)[TP_ID_PORTFOLIO],
        }
        miner_scores = [("miner1", 0.1), ("miner2", 0.5), ("miner3", 0.49)]

        final_scores = dict(Scoring.score_testing_miners(ledgers, miner_scores))

        self.assertEqual(len(final_scores), 3)

        # No ledger should result in the minimum score
        self.assertLess(final_scores["miner1"], final_scores["miner2"])
        self.assertLess(final_scores["miner1"], final_scores["miner3"])

        # Time weighting should make miner2 have a lower score
        self.assertLess(final_scores["miner2"], final_scores["miner3"])

        MIN_WEIGHT = ValiConfig.CHALLENGE_PERIOD_MIN_WEIGHT
        MAX_WEIGHT = ValiConfig.CHALLENGE_PERIOD_MAX_WEIGHT

        self.assertTrue(
            all(MIN_WEIGHT <= weight <= MAX_WEIGHT for weight in final_scores.values()),
            f"All scores must be between {MIN_WEIGHT} and {MAX_WEIGHT}",
        )

        self.assertEqual(final_scores["miner3"], MAX_WEIGHT)
        self.assertEqual(final_scores["miner1"], MIN_WEIGHT)

    def test_asset_class_score_aggregation_empty_input(self):
        """Test asset_class_score_aggregation with empty input"""
        result = Scoring.asset_class_score_aggregation({})
        self.assertEqual(result, [])

    def test_asset_class_score_aggregation_single_asset(self):
        """Test asset_class_score_aggregation with single asset class"""

        asset_scores = {self.DEFAULT_ASSET_CLASS: self.DEFAULT_ASSET_SCORES[self.DEFAULT_ASSET_CLASS]}
        
        result = Scoring.asset_class_score_aggregation(asset_scores)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        
        # Check that results are sorted in descending order
        self.assertGreaterEqual(result[0][1], result[1][1])
        
        # Check that all miners are present
        miner_names = [r[0] for r in result]
        self.assertIn("miner1", miner_names)
        self.assertIn("miner2", miner_names)
        self.assertIn("miner3", miner_names)

    # def test_subclass_score_aggregation_multiple_assets(self):
    #     """Test subclass_score_aggregation with multiple asset classes"""
    #     asset_scores = self.DEFAULT_ASSET_SCORES
    #     asset_weights = {sub_category : 0.5 for sub_category in self.DEFAULT_ASSET_SCORES.keys()}
    #     result = Scoring.subclass_score_aggregation(asset_scores, asset_weights)
    #     self.assertIsInstance(result, list)
    #     self.assertEqual(len(result), 3)
        
    #     # Check that results are sorted in descending order
    #     for i in range(len(result) - 1):
    #         self.assertGreaterEqual(result[i][1], result[i + 1][1])

    def test_softmax_by_asset_empty_input(self):
        """Test softmax_by_asset with empty input"""
        result = Scoring.softmax_by_asset({})
        self.assertEqual(result, {})

    def test_softmax_by_asset_single_asset(self):
        """Test softmax_by_asset with single asset class"""
        asset_scores = {self.DEFAULT_ASSET_CLASS: self.DEFAULT_ASSET_SCORES[self.DEFAULT_ASSET_CLASS]}

        result = Scoring.softmax_by_asset(asset_scores)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertIn(self.DEFAULT_ASSET_CLASS, result)
        
        softmax_scores = result[self.DEFAULT_ASSET_CLASS]
        self.assertEqual(len(softmax_scores), 3)
        
        # Check that softmax scores sum to approximately 1.0
        total_score = sum(softmax_scores.values())
        self.assertAlmostEqual(total_score, 1.0, places=5)
        
        # Check that higher original scores get higher softmax scores
        self.assertGreater(softmax_scores["miner1"], softmax_scores["miner2"])
        self.assertGreater(softmax_scores["miner2"], softmax_scores["miner3"])

    def test_softmax_by_asset_multiple_assets(self):
        """Test softmax_by_asset with multiple asset classes"""
        asset_scores = self.DEFAULT_ASSET_SCORES
        
        result = Scoring.softmax_by_asset(asset_scores)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        
        # Check that each asset class has softmax scores that sum to 1.0
        for asset_class, scores in result.items():
            total_score = sum(scores.values())
            self.assertAlmostEqual(total_score, 1.0, places=5)

    def test_score_miners_empty_ledger(self):
        """Test score_miners with empty ledger"""
        result = Scoring.score_miners({}, {}, self.EVALUATION_TIME_MS)

        # Function output should retain structure, but have no scores
        for subcategory, score_dict in result.items():
            subcategory_metrics = score_dict["metrics"]
            for metric, metric_score_dict in subcategory_metrics.items():
                self.assertListEqual(metric_score_dict["scores"], [])

    def test_score_miners_single_miner(self):
        """Test score_miners with single miner"""
        ledger = {"miner1": self.DEFAULT_LEDGER}
        positions = {"miner1": [self.DEFAULT_POSITION]}
        
        result = Scoring.score_miners(ledger, positions, self.SUBCATEGORY_MIN_DAYS, self.EVALUATION_TIME_MS, all_miner_account_sizes={})
        self.assertIsInstance(result, dict)
        
        # Check that result contains asset subcategories
        self.assertGreater(len(result), 0)
        
        # Check structure of result
        for asset_class, asset_data in result.items():
            self.assertIn("metrics", asset_data)
            self.assertIn("penalties", asset_data)
            self.assertIsInstance(asset_data["penalties"], dict)

    def test_combine_scores_empty_input(self):
        """Test combine_scores with empty input"""
        result = Scoring.combine_scores({})
        self.assertEqual(result, {})

    def test_combine_scores_single_asset(self):
        """Test combine_scores with single asset class"""
        scoring_dict = self.DEFAULT_SCORING_DICT

        result = Scoring.combine_scores(scoring_dict)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertIn(self.DEFAULT_ASSET_CLASS, result)

        combined_scores = result[self.DEFAULT_ASSET_CLASS]
        self.assertIn("miner1", combined_scores)
        self.assertIn("miner2", combined_scores)

    def test_miner_penalties_empty_input(self):
        """Test miner_penalties with empty input"""
        result = Scoring.miner_penalties({}, {}, {})
        self.assertEqual(result, {})

    def test_miner_penalties_with_ledger(self):
        """Test miner_penalties with valid ledger"""
        positions = {"miner1": [self.DEFAULT_POSITION]}
        ledger = {"miner1": self.DEFAULT_LEDGER}
        
        result = Scoring.miner_penalties(positions, ledger, {})
        self.assertIsInstance(result, dict)
        self.assertIn("miner1", result)
        self.assertIsInstance(result["miner1"], float)
        self.assertGreaterEqual(result["miner1"], 0.0)
        self.assertLessEqual(result["miner1"], 1.0)

    def test_miner_penalties_empty_ledger(self):
        """Test miner_penalties with empty ledger"""
        positions = {"miner1": [self.DEFAULT_POSITION]}
        ledger = {"miner1": None}
        
        result = Scoring.miner_penalties(positions, ledger, {})
        self.assertEqual(result, {})

    def test_normalize_scores_empty_input(self):
        """Test normalize_scores with empty input"""
        result = Scoring.normalize_scores({})
        self.assertEqual(result, {})

    def test_normalize_scores_zero_sum(self):
        """Test normalize_scores with zero sum"""
        scores = {"miner1": 0.0, "miner2": 0.0}
        result = Scoring.normalize_scores(scores)
        self.assertEqual(result, {})

    def test_normalize_scores_valid_input(self):
        """Test normalize_scores with valid input"""
        scores = {"miner1": 0.6, "miner2": 0.4}
        result = Scoring.normalize_scores(scores)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        
        # Check that scores sum to 1.0
        total_score = sum(result.values())
        self.assertAlmostEqual(total_score, 1.0, places=5)
        
        # Check that relative proportions are maintained
        self.assertAlmostEqual(result["miner1"], 0.6, places=5)
        self.assertAlmostEqual(result["miner2"], 0.4, places=5)

    def test_normalize_scores_greater_than_one(self):
        """Test normalize_scores with greater than one"""
        scores = {"miner1": 1.0, "miner2": 1.0}
        result = Scoring.normalize_scores(scores)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)

        # Check that scores sum to 1.0
        total_score = sum(result.values())
        self.assertAlmostEqual(total_score, 1.0, places=5)

