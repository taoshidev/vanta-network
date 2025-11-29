import time
import traceback
import argparse
from multiprocessing import Process

from setproctitle import setproctitle

from runnable.generate_request_core import RequestCoreManager
from runnable.generate_request_minerstatistics import MinerStatisticsManager

from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from time_util.time_util import TimeUtil
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
import bittensor as bt

class RequestOutputGenerator:
    def __init__(self, running_deprecated=False, rcm=None, msm=None, checkpoints=True, risk_report=False):
        self.running_deprecated = running_deprecated
        self.last_write_time_s = 0
        self.n_updates = 0
        self.msm_refresh_interval_ms = 15 * 1000
        self.rcm_refresh_interval_ms = 15 * 1000
        self.rcm = rcm
        self.msm = msm
        self.checkpoints = checkpoints
        self.risk_report = risk_report


    def run_deprecated_loop(self):
        bt.logging.info(f'Running RequestOutputGenerator. running_deprecated: {self.running_deprecated}')
        while True:
            self.log_deprecation_message()
            current_time_ms = TimeUtil.now_in_millis()
            self.repull_data_from_disk()
            self.rcm.generate_request_core(write_and_upload_production_files=True)
            self.msm.generate_request_minerstatistics(
                time_now=current_time_ms,
                checkpoints=self.checkpoints,
                risk_report=self.risk_report
            )

            time_to_wait_ms = (self.msm_refresh_interval_ms + self.rcm_refresh_interval_ms) - \
                             (TimeUtil.now_in_millis() - current_time_ms)
            if time_to_wait_ms > 0:
                time.sleep(time_to_wait_ms / 1000)

    def start_generation(self):
        if self.running_deprecated:
            dp = Process(target=self.run_deprecated_loop, daemon=True)
            dp.start()
        else:
            rcm_process = Process(target=self.run_rcm_loop, daemon=True)
            msm_process = Process(target=self.run_msm_loop, daemon=True)
            # Start both processes
            rcm_process.start()
            msm_process.start()

        while True:   # "Don't Die"
            time.sleep(100)

    def log_deprecation_message(self):
        bt.logging.warning("The generate script is no longer managed by pm2. Please update your repo and relaunch the "
                           "run.sh script with (same arguments). This will prevent this pm2 process from being "
                           "spawned and allow significant efficiency improvements by running this code from the "
                           "main validator loop.")

    def repull_data_from_disk(self):
        contract_manager = ValidatorContractManager()
        perf_ledger_manager = PerfLedgerManager(metagraph=None)
        elimination_manager = EliminationManager(None, None, None)
        self.position_manager = PositionManager(None, None,
                                                elimination_manager=elimination_manager,
                                                challengeperiod_manager=None,
                                                perf_ledger_manager=perf_ledger_manager)
        challengeperiod_manager = ChallengePeriodManager(None, None,
                                                         position_manager=self.position_manager)
        elimination_manager.position_manager = self.position_manager
        self.position_manager.challengeperiod_manager = challengeperiod_manager
        elimination_manager.challengeperiod_manager = challengeperiod_manager
        challengeperiod_manager.position_manager = self.position_manager
        perf_ledger_manager.position_manager = self.position_manager
        self.subtensor_weight_setter = SubtensorWeightSetter(
            metagraph=None,
            running_unit_tests=False,
            position_manager=self.position_manager,
            contract_manager=contract_manager,
        )
        self.plagiarism_detector = PlagiarismDetector(None, None, position_manager=self.position_manager)
        self.rcm = RequestCoreManager(self.position_manager, self.subtensor_weight_setter, self.plagiarism_detector, contract_manager=contract_manager)
        self.msm = MinerStatisticsManager(self.position_manager, self.subtensor_weight_setter, self.plagiarism_detector, contract_manager=contract_manager)


    def run_rcm_loop(self):
        setproctitle("vali_RequestCoreManager")
        bt.logging.enable_info()
        bt.logging.info("Running RequestCoreManager process.")
        last_update_time_ms = 0
        n_updates = 0
        while True:
            try:
                current_time_ms = TimeUtil.now_in_millis()
                if current_time_ms - last_update_time_ms < self.rcm_refresh_interval_ms:
                    time.sleep(1)
                    continue
                self.rcm.generate_request_core(write_and_upload_production_files=True)
                n_updates += 1
                tf = TimeUtil.now_in_millis()
                if n_updates % 5 == 0:
                    bt.logging.success(f"RequestCoreManager completed a cycle in {tf - current_time_ms} ms.")
                last_update_time_ms = tf
            except Exception as e:
                bt.logging.error(f"RCM Error: {str(e)}")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)

    def run_msm_loop(self):
        setproctitle("vali_MinerStatisticsManager")
        bt.logging.enable_info()
        bt.logging.info("Running MinerStatisticsManager process.")
        last_update_time_ms = 0
        n_updates = 0
        while True:
            try:
                current_time_ms = TimeUtil.now_in_millis()
                if current_time_ms - last_update_time_ms < self.msm_refresh_interval_ms:
                    time.sleep(1)
                    continue
                self.msm.generate_request_minerstatistics(time_now=current_time_ms, checkpoints=self.checkpoints)
                n_updates += 1
                tf = TimeUtil.now_in_millis()
                if n_updates % 5 == 0:
                    bt.logging.success(f"MinerStatisticsManager completed a cycle in {tf - current_time_ms} ms.")
                last_update_time_ms = tf
            except Exception as e:
                bt.logging.error(f"MSM Error: {str(e)}")
                bt.logging.error(traceback.format_exc())
                time.sleep(10)

if __name__ == "__main__":
    bt.logging.enable_info()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        default=True,
        help="Flag indicating if generation should be with checkpoints (default: True)."
    )
    parser.add_argument(
        "--no-checkpoints",
        dest="checkpoints",
        action="store_false",
        help="If present, disables checkpoints."
    )
    parser.add_argument(
        "--risk-report",
        action="store_true",
        default=False,
        help="Flag indicating if generation should be with risk report report (default: False)."
    )

    args = parser.parse_args()
    rog = RequestOutputGenerator(running_deprecated=True, checkpoints=args.checkpoints, risk_report=args.risk_report)
    rog.start_generation()
