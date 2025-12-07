from vali_objects.utils.elimination.elimination_server import EliminationServer
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.plagiarism.plagiarism_detector import PlagiarismDetector
from vali_objects.position_management.position_manager import PositionManager
from vali_objects.scoring.weight_calculator_manager import WeightCalculatorManager
from time_util.time_util import TimeUtil
from vali_objects.challenge_period import ChallengePeriodManager
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_manager import PerfLedgerManager

if __name__ == "__main__":
    logger = LoggerUtils.init_logger("run challenge review")

    current_time = TimeUtil.now_in_millis()

    perf_ledger_manager = PerfLedgerManager(None)
    # EliminationServer creates its own RPC clients internally (forward compatibility pattern)
    elimination_manager = EliminationServer(running_unit_tests=True)
    position_manager = PositionManager(None, None, elimination_manager=elimination_manager,
                                       challengeperiod_manager=None,
                                       perf_ledger_manager=perf_ledger_manager)
    challengeperiod_manager = ChallengePeriodManager(None, None, position_manager=position_manager)

    elimination_manager.position_manager = position_manager
    position_manager.challengeperiod_manager = challengeperiod_manager
    elimination_manager.challengeperiod_manager = challengeperiod_manager
    challengeperiod_manager.position_manager = position_manager
    perf_ledger_manager.position_manager = position_manager
    # Note: This script uses legacy API and may need updates for RPC architecture
    subtensor_weight_setter = WeightCalculatorManager(
        running_unit_tests=False,
        is_backtesting=True,
        is_mainnet=False
    )
    plagiarism_detector = PlagiarismDetector(None, None, position_manager=position_manager)

    ## Collect the ledger
    ledger = subtensor_weight_setter.perf_ledger_manager.get_perf_ledgers()


    inspection_hotkeys_dict = challengeperiod_manager.challengeperiod_testing

    ## filter the ledger for the miners that passed the challenge period
    success_hotkeys = list(inspection_hotkeys_dict.keys())
    filtered_ledger = perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=success_hotkeys, portfolio_only=False)

    # Get all possible positions, even beyond the lookback range
    success, demoted, eliminations = challengeperiod_manager.inspect(
        ledger=filtered_ledger,
        inspection_hotkeys=inspection_hotkeys_dict,
        current_time=current_time,
    )

    prior_challengeperiod_miners = set(inspection_hotkeys_dict.keys())
    success_miners = set(success)
    eliminated_miners = set(eliminations.keys())

    post_challengeperiod_miners = prior_challengeperiod_miners - eliminated_miners - success_miners

    logger.info(f"{len(prior_challengeperiod_miners)} prior_challengeperiod_miners [{prior_challengeperiod_miners}]")
    logger.info(f"{len(success_miners)} success_miners [{success_miners}]")
    logger.info(f"{len(eliminated_miners)} challengeperiod_eliminations [{eliminations}]")
    logger.info(f"{len(post_challengeperiod_miners)} challenge period remaining [{post_challengeperiod_miners}]")
