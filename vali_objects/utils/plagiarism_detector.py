# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import os
import shutil

from setproctitle import setproctitle

from time_util.time_util import TimeUtil
from vali_objects.utils.plagiarism_definitions import FollowPercentage, LagDetection, CopySimilarity, TwoCopySimilarity, \
    ThreeCopySimilarity
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
import time
import traceback

import bittensor as bt

from vali_objects.utils.plagiarism_pipeline import PlagiarismPipeline

class PlagiarismDetector(CacheController):
    def __init__(self, metagraph, running_unit_tests=False, shutdown_dict=None,
                 position_manager: PositionManager=None):
        super().__init__(metagraph, running_unit_tests=running_unit_tests)
        self.plagiarism_data = {}
        self.plagiarism_raster = {}
        self.plagiarism_positions = {}
        self.plagiarism_classes = [FollowPercentage,
                                   LagDetection,
                                   CopySimilarity,
                                   TwoCopySimilarity,
                                   ThreeCopySimilarity]
        self.position_manager = position_manager if position_manager else PositionManager(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.plagiarism_pipeline = PlagiarismPipeline(self.plagiarism_classes)
        self.shutdown_dict = shutdown_dict

        plagiarism_dir = ValiBkpUtils.get_plagiarism_dir(running_unit_tests=self.running_unit_tests)
        if not os.path.exists(plagiarism_dir):
            ValiBkpUtils.make_dir(ValiBkpUtils.get_plagiarism_dir(running_unit_tests=self.running_unit_tests))
            ValiBkpUtils.make_dir(ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests))

    def run_update_loop(self):
        setproctitle(f"vali_{self.__class__.__name__}")
        bt.logging.enable_info()
        while not self.shutdown_dict:
            try:
                if self.refresh_allowed(ValiConfig.PLAGIARISM_REFRESH_TIME_MS):
                    self.detect(hotkeys=self.position_manager.metagraph.get_hotkeys())
                    self.set_last_update_time(skip_message=False)  # TODO: set True

            except Exception as e:
                # Handle exceptions or log errors
                bt.logging.error(f"Error during plagiarism update: {e}. Please alert a team member ASAP!")
                bt.logging.error(traceback.format_exc())
                time.sleep(30)
            time.sleep(1)

    def detect(self, hotkeys = None, hotkey_positions = None) -> None:
        """
        Kick off the plagiarism detection process.
        """
        if self.running_unit_tests:
            current_time = ValiConfig.PLAGIARISM_LOOKBACK_RANGE_MS
        else:
            current_time = TimeUtil.now_in_millis()
        if hotkeys is None:
            hotkeys = self.metagraph.get_hotkeys()
            assert hotkeys, f"No hotkeys found in metagraph {self.metagraph}"
        if hotkey_positions is None:
            hotkey_positions = self.position_manager.get_positions_for_hotkeys(
                hotkeys,
                eliminations=self.position_manager.elimination_manager.get_eliminations_from_memory(),
            )

        bt.logging.info("Starting Plagiarism Detection")
        #bt.logging.error(
        #    f'$$$$$$$ {len(hotkey_positions)} {len(self.position_manager.elimination_manager.get_eliminations_from_memory())} {len(self.metagraph.get_hotkeys())} {id(self.metagraph)} {type(self.metagraph)} {self.metagraph}')

        plagiarism_data, raster_positions, positions = self.plagiarism_pipeline.run_reporting(positions=hotkey_positions, current_time=current_time)


        self.write_plagiarism_scores_to_disk(plagiarism_data)
        self.write_plagiarism_raster_to_disk(raster_positions)
        self.write_plagiarism_positions_to_disk(positions)

        bt.logging.info("Plagiarism Detection Complete")

    def clear_plagiarism_from_disk(self, target_hotkey=None):
        # Clear all files and directories in the directory specified by dir
        dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        for file in os.listdir(dir):
            if target_hotkey and file != target_hotkey:
                continue
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def write_plagiarism_scores_to_disk(self, plagiarism_data):
        for plagiarist in plagiarism_data:
            self.write_plagiarism_score_to_disk(plagiarist["plagiarist"], plagiarist)

    def write_plagiarism_score_to_disk(self, hotkey, plagiarism_data):
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_plagiarism_score_file_location(hotkey=hotkey, running_unit_tests=self.running_unit_tests),
            plagiarism_data)

    def write_plagiarism_raster_to_disk(self, raster_positions):
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_plagiarism_raster_file_location(running_unit_tests=self.running_unit_tests),
            raster_positions)

    def write_plagiarism_positions_to_disk(self, plagiarism_positions):
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_plagiarism_positions_file_location(running_unit_tests=self.running_unit_tests),
            plagiarism_positions)

    def get_plagiarism_scores_from_disk(self):

        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(plagiarist_dir)

        # Retrieve hotkeys from plagiarism file names
        all_hotkeys = ValiBkpUtils.get_hotkeys_from_file_name(all_files)

        plagiarism_data = {hotkey: self.get_miner_plagiarism_data_from_disk(hotkey) for hotkey in all_hotkeys}
        plagiarism_scores = {}
        for hotkey in plagiarism_data:
            plagiarism_scores[hotkey] = plagiarism_data[hotkey].get("overall_score", 0)

        bt.logging.trace(f"Loaded [{len(plagiarism_scores)}] plagiarism scores from disk. Dir: {plagiarist_dir}")
        return plagiarism_scores

    def get_plagiarism_data_from_disk(self):
        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        all_files = ValiBkpUtils.get_all_files_in_dir(plagiarist_dir)

        # Retrieve hotkeys from plagiarism file names
        all_hotkeys = ValiBkpUtils.get_hotkeys_from_file_name(all_files)

        plagiarism_data = {hotkey: self.get_miner_plagiarism_data_from_disk(hotkey) for hotkey in all_hotkeys}

        bt.logging.trace(f"Loaded [{len(plagiarism_data)}] plagiarism scores from disk. Dir: {plagiarist_dir}")
        return plagiarism_data

    def get_miner_plagiarism_data_from_disk(self, hotkey):
        plagiarist_dir = ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=self.running_unit_tests)
        file_path = os.path.join(plagiarist_dir, f"{hotkey}.json")

        if os.path.exists(file_path):
            data = ValiUtils.get_vali_json_file(file_path)
            return data
        else:
            return {}

    """
    def _refresh_plagiarism_scores_in_memory_and_disk(self):
        # Filters out miners that have already been deregistered. (Not in the metagraph)
        # This allows the miner to participate again once they re-register
        cached_miner_plagiarism = self.get_plagiarism_scores_from_disk()

        blocklist_dict = ValiUtils.get_vali_json_file(ValiBkpUtils.get_plagiarism_blocklist_file_location())
        blocklist_scores = {key['miner_id']: 1 for key in blocklist_dict}

        self.miner_plagiarism_scores = {mch: mc for mch, mc in cached_miner_plagiarism.items() if mch in self.metagraph.get_hotkeys()}

        self.miner_plagiarism_scores = {
            **self.miner_plagiarism_scores,
            **blocklist_scores
        }

        bt.logging.trace(f"Loaded [{len(self.miner_plagiarism_scores)}] miner plagiarism scores from disk.")

        self._write_updated_plagiarism_scores_from_memory_to_disk()
    """

    def _update_plagiarism_scores_in_memory(self):
        raster_positions_location = ValiBkpUtils.get_plagiarism_raster_file_location(
            running_unit_tests=self.running_unit_tests)
        self.plagiarism_raster = ValiUtils.get_vali_json_file(raster_positions_location)

        positions_location = ValiBkpUtils.get_plagiarism_positions_file_location(
            running_unit_tests=self.running_unit_tests)
        self.plagiarism_positions = ValiUtils.get_vali_json_file(positions_location)

        self.plagiarism_data = self.get_plagiarism_data_from_disk()
        

        

