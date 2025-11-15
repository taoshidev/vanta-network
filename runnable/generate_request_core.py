import copy
import gzip
import json
import os
import hashlib

from google.cloud import storage

from time_util.time_util import TimeUtil
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.limit_order_manager import LimitOrderManager
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import ValiConfig
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager

from vali_objects.utils.validator_sync_base import AUTO_SYNC_ORDER_LAG_MS

# no filters,... , max filter
PERCENT_NEW_POSITIONS_TIERS = [100, 50, 30, 0]
assert sorted(PERCENT_NEW_POSITIONS_TIERS, reverse=True) == PERCENT_NEW_POSITIONS_TIERS, 'needs to be sorted for efficient pruning'

class RequestCoreManager:
    def __init__(self, position_manager, subtensor_weight_setter, plagiarism_detector, contract_manager=None,
                 ipc_manager=None, asset_selection_manager=None, limit_order_manager=None):
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.elimination_manager = position_manager.elimination_manager
        self.challengeperiod_manager = position_manager.challengeperiod_manager
        self.subtensor_weight_setter = subtensor_weight_setter
        self.plagiarism_detector = plagiarism_detector
        self.contract_manager = contract_manager
        self.live_price_fetcher = None
        self.asset_selection_manager = asset_selection_manager

        # Initialize IPC-managed dictionary for validator checkpoint caching
        if ipc_manager:
            self.validator_checkpoint_cache = ipc_manager.dict()
        else:
            self.validator_checkpoint_cache = {}
        self.limit_order_manager = limit_order_manager

    def hash_string_to_int(self, s: str) -> int:
        # Create a SHA-256 hash object
        hash_object = hashlib.sha256()
        # Update the hash object with the bytes of the string
        hash_object.update(s.encode('utf-8'))
        # Get the hexadecimal digest of the hash
        hex_digest = hash_object.hexdigest()
        # Convert the hexadecimal digest to an integer
        hash_int = int(hex_digest, 16)
        return hash_int

    def filter_new_positions_random_sample(self, percent_new_positions_keep: float, hotkey_to_positions: dict[str:[dict]], time_of_position_read_ms:int) -> None:
        """
        candidate_data['positions'][hk]['positions'] = [json.loads(str(p), cls=GeneralizedJSONDecoder) for p in positions_orig]
        """
        def filter_orders(p: Position) -> bool:
            nonlocal stale_date_threshold_ms
            if p.is_closed_position and p.close_ms < stale_date_threshold_ms:
                return False
            if p.is_open_position and p.orders[-1].processed_ms < stale_date_threshold_ms:
                return False
            if percent_new_positions_keep == 100:
                return False
            if percent_new_positions_keep and self.hash_string_to_int(p.position_uuid) % 100 < percent_new_positions_keep:
                return False
            return True

        def truncate_position(position_to_truncate: Position) -> Position:
            nonlocal stale_date_threshold_ms
            if not self.live_price_fetcher:
                secrets = ValiUtils.get_secrets()
                self.live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)

            new_orders = []
            for order in position_to_truncate.orders:
                if order.processed_ms < stale_date_threshold_ms:
                    new_orders.append(order)

            if len(new_orders):
                position_to_truncate.orders = new_orders
                position_to_truncate.rebuild_position_with_updated_orders(self.live_price_fetcher)
                return position_to_truncate
            else:  # no orders left. erase position
                return None

        assert percent_new_positions_keep in PERCENT_NEW_POSITIONS_TIERS
        stale_date_threshold_ms = time_of_position_read_ms - AUTO_SYNC_ORDER_LAG_MS
        for hotkey, positions in hotkey_to_positions.items():
            new_positions = []
            positions_deserialized = [Position(**json_positions_dict) for json_positions_dict in positions['positions']]
            for position in positions_deserialized:
                if filter_orders(position):
                    truncated_position = truncate_position(position)
                    if truncated_position:
                        new_positions.append(truncated_position)
                else:
                    new_positions.append(position)

            # Turn the positions back into json dicts. Note we are overwriting the original positions
            positions['positions'] = [json.loads(str(p), cls=GeneralizedJSONDecoder) for p in new_positions]

    def compress_dict(self, data: dict) -> bytes:
        str_to_write = json.dumps(data, cls=CustomEncoder)
        # Encode the JSON string to bytes and then compress it using gzip
        compressed = gzip.compress(str_to_write.encode("utf-8"))
        return compressed

    def decompress_dict(self, compressed_data: bytes) -> dict:
        # Decompress the compressed data
        decompressed = gzip.decompress(compressed_data)
        # Decode the decompressed data to a JSON string and then load it into a dictionary
        data = json.loads(decompressed.decode("utf-8"))
        return data

    def store_checkpoint_in_memory(self, checkpoint_data: dict):
        """Store compressed validator checkpoint data in IPC memory cache."""
        try:
            compressed_data = self.compress_dict(checkpoint_data)
            self.validator_checkpoint_cache['checkpoint'] = {
                'data': compressed_data,
                'timestamp_ms': TimeUtil.now_in_millis()
            }
        except Exception as e:
            print(f"Error storing checkpoint in memory: {e}")

    def get_compressed_checkpoint_from_memory(self) -> bytes | None:
        """Retrieve compressed validator checkpoint data directly from memory cache."""
        try:
            cached_entry = self.validator_checkpoint_cache.get('checkpoint', {})
            if not cached_entry or 'data' not in cached_entry:
                return None
            
            return cached_entry['data']
        except Exception as e:
            print(f"Error retrieving compressed checkpoint from memory: {e}")
            return None

    def upload_checkpoint_to_gcloud(self, final_dict):
        """
        The idea is to upload a zipped, time lagged validator checkpoint to google cloud for auto restoration
        on other validators as well as transparency with the community.

        Positions are already time-filtered from the code called before this function.
        """
        datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
        #if not (datetime_now.hour == 6 and datetime_now.minute < 9 and datetime_now.second < 30):
        if not (datetime_now.minute == 24):
            return

        # check if file exists
        KEY_PATH = ValiConfig.BASE_DIR + '/gcloud_new.json'
        if not os.path.exists(KEY_PATH):
            return

        # Path to your service account key file
        key_path = KEY_PATH
        key_info = json.load(open(key_path))

        # Initialize a storage client using your service account key
        client = storage.Client.from_service_account_info(key_info)

        # Name of the bucket you want to write to
        bucket_name = 'validator_checkpoint'

        # Get the bucket
        bucket = client.get_bucket(bucket_name)

        # Name for the new blob
        # blob_name = 'validator_checkpoint.json'
        blob_name = 'validator_checkpoint.json.gz'

        # Create a new blob and upload data
        blob = bucket.blob(blob_name)

        # Create a zip file in memory
        zip_buffer = self.compress_dict(final_dict)
        # Upload the content of the zip_buffer to Google Cloud Storage
        blob.upload_from_string(zip_buffer)
        print(f'Uploaded {blob_name} to {bucket_name}')

    def create_and_upload_production_files(self, eliminations, ord_dict_hotkey_position_map, time_now,
                                           youngest_order_processed_ms, oldest_order_processed_ms,
                                           challengeperiod_dict, miner_account_sizes_dict, limit_orders_dict):

        perf_ledgers = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)

        # Get asset selections if available
        asset_selections = {}
        if self.asset_selection_manager:
            asset_selections = self.asset_selection_manager._to_dict()

        final_dict = {
            'version': ValiConfig.VERSION,
            'created_timestamp_ms': time_now,
            'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
            'challengeperiod': challengeperiod_dict,
            'miner_account_sizes': miner_account_sizes_dict,
            'eliminations': eliminations,
            'youngest_order_processed_ms': youngest_order_processed_ms,
            'oldest_order_processed_ms': oldest_order_processed_ms,
            'positions': ord_dict_hotkey_position_map,
            'perf_ledgers': perf_ledgers,
            'asset_selections': asset_selections,
            'limit_orders': limit_orders_dict
        }

        # Write compressed checkpoint only - saves disk space and bandwidth
        compressed_data = self.compress_dict(final_dict)
        
        # Write compressed file directly
        compressed_path = ValiBkpUtils.get_vcp_output_path()
        with open(compressed_path, 'wb') as f:
            f.write(compressed_data)
        #print(f"Wrote compressed checkpoint to {compressed_path}")
        
        # Store compressed checkpoint data in IPC memory cache
        self.store_checkpoint_in_memory(final_dict)

        # Write positions data (sellable via RN) at the different tiers. Each iteration, the number of orders (possibly) decreases
        for t in PERCENT_NEW_POSITIONS_TIERS:
            if t == 100:  # no filtering
                # Write legacy location as well. no compression
                ValiBkpUtils.write_file(
                    ValiBkpUtils.get_miner_positions_output_path(suffix_dir=None),
                    ord_dict_hotkey_position_map,
                )
            else:
                self.filter_new_positions_random_sample(t, ord_dict_hotkey_position_map, time_now)

            # "v2" add a tier. compress the data. This is a location in a subdir
            for hotkey, dat in ord_dict_hotkey_position_map.items():
                dat['tier'] = t

            compressed_positions = self.compress_dict(ord_dict_hotkey_position_map)
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_miner_positions_output_path(suffix_dir=str(t)),
                compressed_positions, is_binary=True
            )

        # Max filtering
        self.upload_checkpoint_to_gcloud(final_dict)

    def generate_request_core(self, get_dash_data_hotkey: str | None = None, write_and_upload_production_files=False) -> dict:
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        try:
            if not os.path.exists(ValiBkpUtils.get_miner_dir()):
                raise FileNotFoundError
        except FileNotFoundError:
            raise Exception(
                f"directory for miners doesn't exist "
                f"[{ValiBkpUtils.get_miner_dir()}]. Skip run for now."
            )

        if get_dash_data_hotkey:
            all_miner_hotkeys: list = [get_dash_data_hotkey]
        else:
            all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(ValiBkpUtils.get_miner_dir())

        # we won't be able to query for eliminated hotkeys from challenge period
        hotkey_positions = self.position_manager.get_positions_for_hotkeys(
            all_miner_hotkeys,
            sort_positions=True
        )

        time_now_ms = TimeUtil.now_in_millis()

        dict_hotkey_position_map = {}

        youngest_order_processed_ms = float("inf")
        oldest_order_processed_ms = 0

        for k, original_positions in hotkey_positions.items():
            dict_hotkey_position_map[k] = self.position_manager.positions_to_dashboard_dict(original_positions, time_now_ms)
            for p in original_positions:
                youngest_order_processed_ms = min(youngest_order_processed_ms,
                                                  min(p.orders, key=lambda o: o.processed_ms).processed_ms)
                oldest_order_processed_ms = max(oldest_order_processed_ms,
                                                max(p.orders, key=lambda o: o.processed_ms).processed_ms)

        ord_dict_hotkey_position_map = dict(
            sorted(
                dict_hotkey_position_map.items(),
                key=lambda item: item[1]["thirty_day_returns"],
                reverse=True,
            )
        )

        # unfiltered positions dict for checkpoints
        unfiltered_positions = copy.deepcopy(ord_dict_hotkey_position_map)

        n_orders_original = 0
        for positions in hotkey_positions.values():
            n_orders_original += sum([len(position.orders) for position in positions])

        n_positions_new = 0
        for data in ord_dict_hotkey_position_map.values():
            positions = data['positions']
            n_positions_new += sum([len(p['orders']) for p in positions])

        assert n_orders_original == n_positions_new, f"n_orders_original: {n_orders_original}, n_positions_new: {n_positions_new}"

        challengeperiod_dict = self.challengeperiod_manager.to_checkpoint_dict()

        # Get miner account sizes if contract manager is available
        miner_account_sizes_dict = {}
        if self.contract_manager:
            miner_account_sizes_dict = self.contract_manager.miner_account_sizes_dict()

        if write_and_upload_production_files:
            limit_orders_dict = self.limit_order_manager.get_limit_orders()
            self.create_and_upload_production_files(eliminations, ord_dict_hotkey_position_map, time_now_ms,
                                           youngest_order_processed_ms, oldest_order_processed_ms,
                                           challengeperiod_dict, miner_account_sizes_dict, limit_orders_dict)

        checkpoint_dict = {
            'challengeperiod': challengeperiod_dict,
            'miner_account_sizes': miner_account_sizes_dict,
            'positions': unfiltered_positions
        }
        return checkpoint_dict

if __name__ == "__main__":
    contract_manager = ValidatorContractManager()
    perf_ledger_manager = PerfLedgerManager(None, {}, [])
    elimination_manager = EliminationManager(None, [],None, None)
    position_manager = PositionManager(None, None, elimination_manager=elimination_manager,
                                       challengeperiod_manager=None,
                                       perf_ledger_manager=perf_ledger_manager)
    challengeperiod_manager = ChallengePeriodManager(None, None, position_manager=position_manager)

    elimination_manager.position_manager = position_manager
    position_manager.challengeperiod_manager = challengeperiod_manager
    elimination_manager.challengeperiod_manager = challengeperiod_manager
    challengeperiod_manager.position_manager = position_manager
    perf_ledger_manager.position_manager = position_manager
    subtensor_weight_setter = SubtensorWeightSetter(
        metagraph=None,
        running_unit_tests=False,
        position_manager=position_manager,
        contract_manager=contract_manager
    )
    plagiarism_detector = PlagiarismDetector(None, None, position_manager=position_manager)
    limit_order_manager = LimitOrderManager(position_manager, None)

    rcm = RequestCoreManager(position_manager, subtensor_weight_setter, plagiarism_detector, limit_order_manager, contract_manager=contract_manager)

    rcm.generate_request_core(write_and_upload_production_files=True)
