# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import json
import os
import shutil
import pickle
import uuid
from multiprocessing.managers import DictProxy

import bittensor as bt
from pydantic import BaseModel

from vali_objects.vali_config import ValiConfig
from vali_objects.position import Position
from vali_objects.vali_dataclasses.order import OrderStatus
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_config import TradePair


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TradePair) or isinstance(obj, OrderType):
            return obj.__json__()
        elif isinstance(obj, BaseModel):
            return obj.dict()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, DictProxy):
            return dict(obj)

        return json.JSONEncoder.default(self, obj)

class ValiBkpUtils:
    @staticmethod
    def get_miner_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/miners/"

    @staticmethod
    def get_temp_file_path():
        return ValiConfig.BASE_DIR + "/validation/tmp/"

    @staticmethod
    def get_validator_checkpoint_path(use_data_dir=False):
        """Get path for compressed validator checkpoint file (input/restore).
        
        Args:
            use_data_dir: If True, use data/ subdirectory, else use root directory
        
        Returns:
            Full path to compressed validator checkpoint file (.gz)
        """
        base_path = ValiConfig.BASE_DIR + ("/data/validator_checkpoint.json" if use_data_dir else "/validator_checkpoint.json")
        return base_path + ".gz"
    
    @staticmethod
    def get_backup_file_path(use_data_dir=False):
        """Legacy method for backward compatibility. Use get_validator_checkpoint_path instead.
        Note: Returns uncompressed path for backward compatibility with existing checkpoint files."""
        base_path = ValiConfig.BASE_DIR + ("/data/validator_checkpoint.json" if use_data_dir else "/validator_checkpoint.json")
        return base_path

    @staticmethod
    def get_api_keys_file_path():
        return ValiConfig.BASE_DIR + "/vanta_api/api_keys.json"

    @staticmethod
    def get_sequence_number_file_path():
        return ValiConfig.BASE_DIR + "/vanta_api/sequence_number.json"


    @staticmethod
    def get_positions_override_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/data/positions_overrides/"

    @staticmethod
    def get_miner_all_positions_dir(miner_hotkey, running_unit_tests=False) -> str:
        return f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}{miner_hotkey}/positions/"

    @staticmethod
    def get_eliminations_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/eliminations.json"

    @staticmethod
    def get_departed_hotkeys_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/departed_hotkeys.json"

    @staticmethod
    def get_perf_ledger_eliminations_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/perf_ledger_eliminations.json"

    @staticmethod
    def get_perf_ledgers_path(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/perf_ledgers.json"

    @staticmethod
    def get_plagiarism_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/plagiarism/"
    @staticmethod
    def get_plagiarism_raster_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/plagiarism/raster_vectors"
    
    @staticmethod
    def get_plagiarism_positions_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/plagiarism/positions"

    @staticmethod
    def get_plagiarism_scores_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/plagiarism/miners/"
    
    @staticmethod
    def get_plagiarism_score_file_location(hotkey, running_unit_tests=False) -> str:
        return f"{ValiBkpUtils.get_plagiarism_scores_dir(running_unit_tests=running_unit_tests)}{hotkey}.json"
    
    @staticmethod
    def get_challengeperiod_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/challengeperiod.json"

    @staticmethod
    def get_asset_selections_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/asset_selections.json"

    @staticmethod
    def get_last_order_timestamp_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/timestamp.json"

    @staticmethod
    def get_miner_account_sizes_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/miner_account_sizes.json"

    @staticmethod
    def get_secrets_dir():
        return ValiConfig.BASE_DIR + "/secrets.json"

    @staticmethod
    def get_taoshi_api_keys_file_location():
        return ValiConfig.BASE_DIR + "/config-development.json"

    @staticmethod
    def get_plagiarism_blocklist_file_location():
        return ValiConfig.BASE_DIR + "/miner_blocklist.json"
    
    @staticmethod
    def get_vali_bkp_dir() -> str:
        return ValiConfig.BASE_DIR + "/backups/"

    @staticmethod
    def get_vali_outputs_dir() -> str:
        return ValiConfig.BASE_DIR + "/runnable/"

    @staticmethod
    def get_miner_stats_dir(running_unit_tests=False) -> str:
        return ValiBkpUtils.get_vali_outputs_dir() + "minerstatistics.json"

    @staticmethod
    def get_restore_file_path() -> str:
        """Legacy method for backward compatibility. Use get_validator_checkpoint_path instead.
        Note: Returns uncompressed path for backward compatibility with existing checkpoint files."""
        return ValiConfig.BASE_DIR + "/validator_checkpoint.json"

    @staticmethod
    def get_vcp_output_path() -> str:
        """Get path for compressed validator checkpoint output file.
        
        Returns:
            Full path to compressed validator checkpoint output file (.gz)
        """
        return ValiBkpUtils.get_vali_outputs_dir() + "validator_checkpoint.json.gz"

    @staticmethod
    def get_miner_positions_output_path(suffix_dir: None | str = None) -> str:
        if suffix_dir is None:
            suffix = ''
        else:
            suffix = f"tiered_positions/{suffix_dir}/"
        ans = ValiConfig.BASE_DIR + f"/validation/outputs/{suffix}output.json"
        if suffix_dir is not None:
            ans += '.gz'
        return ans

    @staticmethod
    def get_meta_json_path() -> str:
        return ValiConfig.BASE_DIR + '/meta/meta.json'

    @staticmethod
    def get_vali_weights_dir() -> str:
        return ValiConfig.BASE_DIR + "/validation/weights/"

    @staticmethod
    def get_vali_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/"

    @staticmethod
    def get_vali_data_file() -> str:
        return "valirecords.json"

    @staticmethod
    def get_vali_weights_file() -> str:
        return "valiweights.json"

    @staticmethod
    def get_vali_predictions_dir() -> str:
        return ValiConfig.BASE_DIR + "/validation/predictions/"

    @staticmethod
    def get_slippage_model_parameters_file() -> str:
        return ValiConfig.BASE_DIR + "/vali_objects/utils/model_parameters/all_model_parameters.json"

    @staticmethod
    def get_slippage_estimates_file() -> str:
        return ValiConfig.BASE_DIR + "/vali_objects/utils/model_parameters/slippage_estimates.json"

    @staticmethod
    def get_slippage_model_features_file() -> str:
        return ValiConfig.BASE_DIR + "/vali_objects/utils/model_parameters/model_features.json"

    @staticmethod
    def get_response_filename(request_uuid: str) -> str:
        return str(request_uuid) + ".pickle"

    @staticmethod
    def get_cmw_filename(request_uuid: str) -> str:
        return str(request_uuid) + ".json"

    @staticmethod
    def make_dir(vali_dir: str) -> None:
        if not os.path.exists(vali_dir):
            os.makedirs(vali_dir)

    @staticmethod
    def get_write_type(is_pickle: bool, is_binary:bool) -> str:
        return "wb" if is_pickle or is_binary else "w"

    @staticmethod
    def get_read_type(is_pickle: bool) -> str:
        return "rb" if is_pickle else "r"

    @staticmethod
    def clear_tmp_dir():
        temp_dir = ValiBkpUtils.get_temp_file_path()
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))

    @staticmethod
    def write_to_dir(
        vali_file: str, vali_data: dict | object, is_pickle: bool = False, is_binary:bool = False
    ) -> None:
        temp_dir = ValiBkpUtils.get_temp_file_path()
        os.makedirs(os.path.dirname(vali_file), exist_ok=True)
        os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
        # Create uuid file name
        temp_file_path = temp_dir + str(uuid.uuid4())
        # Write to temp file first
        with open(temp_file_path, ValiBkpUtils.get_write_type(is_pickle, is_binary)) as f:
            if is_binary:
                f.write(vali_data)
            elif isinstance(vali_data, Position):
                f.write(vali_data.to_json_string())
            elif is_pickle:
                pickle.dump(vali_data, f)
            else:
                f.write(json.dumps(vali_data, cls=CustomEncoder))
        # Move the file from temp to the final location
        shutil.move(temp_file_path, vali_file)

    @staticmethod
    def write_file(
        vali_dir: str, vali_data: dict | object, is_pickle: bool = False, is_binary: bool = False
    ) -> None:
        ValiBkpUtils.write_to_dir(vali_dir, vali_data, is_pickle, is_binary=is_binary)

    @staticmethod
    def get_file(vali_file: str, is_pickle: bool = False) -> str | object:
        #bt.logging.info(f"attempting to read vali_file: {vali_file}")
        with open(vali_file, ValiBkpUtils.get_read_type(is_pickle)) as f:
            ans = pickle.load(f) if is_pickle else f.read()
            return ans


    @staticmethod
    def safe_load_dict_from_disk(filename, default_value):
        try:
            full_path = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + filename
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    return json.load(f)

            temp_filename = f"{filename}.tmp"
            full_path_temp = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + temp_filename
            if os.path.exists(full_path_temp):
                with open(full_path_temp, 'r') as f:
                    ans = json.load(f)
                    # Write to disk with the correct filename
                    ValiBkpUtils.safe_save_dict_to_disk(filename, ans, skip_temp_write=True)
        except Exception as e:
            bt.logging.error(f"Error loading {filename} from disk: {e}")

        return default_value

    @staticmethod
    def safe_save_dict_to_disk(filename, data, skip_temp_write=False):
        try:
            temp_filename = f"{filename}.tmp"
            full_path_temp = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + temp_filename
            full_path_orig = ValiBkpUtils.get_vali_dir(running_unit_tests=False) + filename
            if skip_temp_write:
                with open(full_path_orig, 'w') as f:
                    json.dump(data, f)
            else:
                with open(full_path_temp, 'w') as f:
                    json.dump(data, f)
                os.replace(full_path_temp, full_path_orig)
        except Exception as e:
            bt.logging.error(f"Error saving {filename} to disk: {e}")

    @staticmethod
    def get_all_files_in_dir(vali_dir: str) -> list[str]:
        """
        Put open positions first as they are prone to race conditions and we want to process them first.
        """
        open_files = []  # List to store file paths from "open" directories
        closed_files = []  # List to store file paths from all other directories

        for dirpath, dirnames, filenames in os.walk(vali_dir):
            for filename in filenames:
                if filename == '.DS_Store':
                    continue  # Skip .DS_Store files
                elif filename.endswith('.swp'):
                    continue
                filepath = os.path.join(dirpath, filename)
                if '/open/' in filepath:  # Check if file is in an "open" subdirectory
                    open_files.append(filepath)
                else:
                    closed_files.append(filepath)

        # Concatenate "open" and other directory files without sorting
        return open_files + closed_files
    
    @staticmethod
    def get_hotkeys_from_file_name(files: list[str]) -> list[str]:
        return [os.path.splitext(os.path.basename(path))[0] for path in files]
    
    @staticmethod
    def get_directories_in_dir(directory):
        return [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]

    @staticmethod
    def get_partitioned_miner_positions_dir(miner_hotkey, trade_pair_id, order_status=OrderStatus.ALL,
                                            running_unit_tests=False) -> str:

        base_dir = (f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}"
               f"{miner_hotkey}/positions/{trade_pair_id}/")

        # Decide the subdirectory based on the order_status
        status_dir = {
            OrderStatus.OPEN: "open/",
            OrderStatus.CLOSED: "closed/",
            OrderStatus.ALL: ""
        }[order_status]

        return f"{base_dir}{status_dir}"
