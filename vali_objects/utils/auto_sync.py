import gzip
import io
import json
import traceback
import zipfile

import requests

from time_util.time_util import TimeUtil
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.validator_sync_base import ValidatorSyncBase
import bittensor as bt
#from restore_validator_from_backup import regenerate_miner_positions
#from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


class PositionSyncer(ValidatorSyncBase):
    def __init__(self, shutdown_dict=None, signal_sync_lock=None, signal_sync_condition=None,
                 n_orders_being_processed=None, running_unit_tests=False, position_manager=None,
                 ipc_manager=None, auto_sync_enabled=False, enable_position_splitting=False, verbose=False,
                 contract_manager=None, live_price_fetcher=None, asset_selection_manager=None,
                 sync_in_progress=None, sync_epoch=None):
        super().__init__(shutdown_dict, signal_sync_lock, signal_sync_condition, n_orders_being_processed,
                         running_unit_tests=running_unit_tests, position_manager=position_manager,
                         ipc_manager=ipc_manager, enable_position_splitting=enable_position_splitting, verbose=verbose,
                         contract_manager=contract_manager, live_price_fetcher=live_price_fetcher, asset_selection_manager=asset_selection_manager)

        self.force_ran_on_boot = True
        self.sync_in_progress = sync_in_progress
        self.sync_epoch = sync_epoch
        print(f'PositionSyncer: auto_sync_enabled: {auto_sync_enabled}')
        """
        time_now_ms = TimeUtil.now_in_millis()
        if auto_sync_enabled and time_now_ms < 1736697619000 + 3 * 1000 * 60 * 60:
            response = requests.get(self.fname_to_url('validator_checkpoint.json'))
            response.raise_for_status()
            output_path = ValiBkpUtils.get_restore_file_path()
            print(f'writing {response.content[:100]} to {output_path}')
            with open(output_path, 'wb') as f:
                f.write(response.content)
            regenerate_miner_positions(False, ignore_timestamp_checks=True)
        """

    def fname_to_url(self, fname):
        return f"https://storage.googleapis.com/validator_checkpoint/{fname}"

    def read_validator_checkpoint_from_gcloud_zip(self, fname="validator_checkpoint.json.gz"):
        # URL of the zip file
        url = self.fname_to_url(fname)
        try:
            # Send HTTP GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Read the content of the gz file from the response
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
                # Decode the gzip content to a string
                json_bytes = gz_file.read()
                json_str = json_bytes.decode('utf-8')

                # Load JSON data from the string
                json_data = json.loads(json_str)
                return json_data

        except requests.HTTPError as e:
            bt.logging.error(f"HTTP Error: {e}")
        except zipfile.BadZipFile:
            bt.logging.error("The downloaded file is not a zip file or it is corrupted.")
        except json.JSONDecodeError:
            bt.logging.error("Error decoding JSON from the file.")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e}")
        return None

    def perform_sync(self):
        with self.signal_sync_lock:
            while self.n_orders_being_processed[0] > 0:
                self.signal_sync_condition.wait()

            # Wrap everything in try/finally to guarantee sync_in_progress is always reset
            # This prevents deadlock if an exception occurs anywhere after setting the flag
            try:
                # CRITICAL ORDERING: Set flag BEFORE incrementing epoch to prevent race condition
                # 1. Set sync_in_progress FIRST to block new iterations from starting
                if self.sync_in_progress:
                    self.sync_in_progress.value = True

                # 2. THEN increment sync epoch to invalidate in-flight iterations
                # This ensures no new iteration can start with the new epoch before sync completes
                if self.sync_epoch:
                    old_epoch = self.sync_epoch.value
                    self.sync_epoch.value += 1
                    bt.logging.info(f"Incrementing sync epoch {old_epoch} -> {self.sync_epoch.value}")

                candidate_data = self.read_validator_checkpoint_from_gcloud_zip()
                if not candidate_data:
                    bt.logging.error("Unable to read validator checkpoint file. Sync canceled")
                else:
                    self.sync_positions(False, candidate_data=candidate_data)
            except Exception as e:
                bt.logging.error(f"Error syncing positions: {e}")
                bt.logging.error(traceback.format_exc())
            finally:
                # CRITICAL: Always clear sync_in_progress flag to prevent deadlock
                # This executes even if exception occurs before sync starts
                if self.sync_in_progress:
                    self.sync_in_progress.value = False

        self.last_signal_sync_time_ms = TimeUtil.now_in_millis()

    def sync_positions_with_cooldown(self, auto_sync_enabled:bool):
        if not auto_sync_enabled:
            return

        if self.force_ran_on_boot == False:  # noqa: E712
            self.perform_sync()
            self.force_ran_on_boot = True

        # Check if the time is right to sync signals
        now_ms = TimeUtil.now_in_millis()
        # Already performed a sync recently
        if now_ms - self.last_signal_sync_time_ms < 1000 * 60 * 30:
            return

        datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
        if not (datetime_now.hour == 21 and (7 < datetime_now.minute < 17)):
            return

        self.perform_sync()


if __name__ == "__main__":
    bt.logging.enable_info()
    elimination_manager = EliminationManager(None, [], None, None)
    position_manager = PositionManager({}, elimination_manager=elimination_manager, challengeperiod_manager=None)
    challengeperiod_manager = ChallengePeriodManager(metagraph=None, position_manager=position_manager)
    contract_manager = ValidatorContractManager(config=None, running_unit_tests=False)
    position_manager.challengeperiod_manager = challengeperiod_manager
    position_syncer = PositionSyncer(position_manager=position_manager, contract_manager=contract_manager)
    candidate_data = position_syncer.read_validator_checkpoint_from_gcloud_zip()
    position_syncer.sync_positions(False, candidate_data=candidate_data)
