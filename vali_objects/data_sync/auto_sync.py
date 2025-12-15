import gzip
import io
import json
import traceback
import zipfile

import requests

from shared_objects.rpc.common_data_server import CommonDataServer
from shared_objects.rpc.metagraph_server import MetagraphServer
from time_util.time_util import TimeUtil
from vali_objects.utils.asset_selection.asset_selection_server import AssetSelectionServer
from vali_objects.challenge_period.challengeperiod_server import ChallengePeriodServer
from vali_objects.contract.contract_server import ContractServer
from vali_objects.utils.elimination.elimination_server import EliminationServer
from vali_objects.position_management.position_manager_server import PositionManagerServer
from vali_objects.data_sync.validator_sync_base import ValidatorSyncBase
from entity_management.entity_server import EntityServer
import bittensor as bt

from vali_objects.vali_config import RPCConnectionMode
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_server import PerfLedgerServer


#from restore_validator_from_backup import regenerate_miner_positions
#from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


class PositionSyncer(ValidatorSyncBase):
    def __init__(self, order_sync=None, running_unit_tests=False,
                 auto_sync_enabled=False, enable_position_splitting=False, verbose=False,
                 connection_mode=RPCConnectionMode.RPC, is_mothership=False):
        # ValidatorSyncBase creates its own LivePriceFetcherClient, PerfLedgerClient, AssetSelectionClient,
        # LimitOrderClient, and ContractClient internally (forward compatibility)
        super().__init__(order_sync=order_sync,
                         running_unit_tests=running_unit_tests,
                         enable_position_splitting=enable_position_splitting, verbose=verbose,
                         is_mothership=is_mothership)
        self.order_sync = order_sync

        # Create own CommonDataClient (forward compatibility - no parameter passing)
        from shared_objects.rpc.common_data_client import CommonDataClient
        self._common_data_client = CommonDataClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self.force_ran_on_boot = True
        print(f'PositionSyncer: auto_sync_enabled: {auto_sync_enabled}')

    # ==================== Common Data Properties ====================

    @property
    def sync_in_progress(self):
        """Get sync_in_progress flag from CommonDataClient."""
        return self._common_data_client.get_sync_in_progress()

    @property
    def sync_epoch(self):
        """Get sync_epoch from CommonDataClient."""
        return self._common_data_client.get_sync_epoch()

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
        # Wait for in-flight orders and set sync_waiting flag (context manager handles this)
        with self.order_sync.begin_sync():
            # Wrap everything in try/finally to guarantee sync_in_progress is always reset
            # This prevents deadlock if an exception occurs anywhere after setting the flag
            try:
                # CRITICAL ORDERING: Set flag BEFORE incrementing epoch to prevent race condition
                # 1. Set sync_in_progress FIRST to block new iterations from starting
                self._common_data_client.set_sync_in_progress(True)

                # 2. THEN increment sync epoch to invalidate in-flight iterations
                # This ensures no new iteration can start with the new epoch before sync completes
                old_epoch = self.sync_epoch
                new_epoch = self._common_data_client.increment_sync_epoch()
                bt.logging.info(f"Incrementing sync epoch {old_epoch} -> {new_epoch}")

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
                self._common_data_client.set_sync_in_progress(False)

                # Update timestamp
                self.last_signal_sync_time_ms = TimeUtil.now_in_millis()
        # Context manager auto-clears sync_waiting flag on exit

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
        if not (datetime_now.hour == 0 and (7 < datetime_now.minute < 17)):
            return

        self.perform_sync()


if __name__ == "__main__":
    bt.logging.enable_info()
    # EliminationServer creates its own RPC clients internally (forward compatibility pattern)
    cds = CommonDataServer()
    ms = MetagraphServer()
    es = EliminationServer()
    cs = ChallengePeriodServer()
    ps = PositionManagerServer()
    pls = PerfLedgerServer()
    vs = ContractServer()
    ass = AssetSelectionServer()
    ent_server = EntityServer()
    # ValidatorSyncBase creates its own ContractClient and LimitOrderClient internally (forward compatibility)
    position_syncer = PositionSyncer()
    candidate_data = position_syncer.read_validator_checkpoint_from_gcloud_zip()
    position_syncer.sync_positions(False, candidate_data=candidate_data)
