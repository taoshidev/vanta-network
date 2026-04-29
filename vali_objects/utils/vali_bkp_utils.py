# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import gzip
import json
import os
import shutil
from tempfile import NamedTemporaryFile
from multiprocessing.managers import DictProxy

import bittensor as bt
import numpy as np
import orjson
from pydantic import BaseModel

from vali_objects.vali_config import ValiConfig
from vali_objects.enums.misc import OrderStatus
from vali_objects.enums.order_type_enum import OrderType, StopCondition
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import TradePair, DynamicTradePair


def orjson_encoder(obj):
    if hasattr(obj, '__json__'):
        return obj.__json__()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    elif isinstance(obj, DictProxy):
        return dict(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (TradePair, DynamicTradePair, OrderType, ExecutionType, StopCondition)):
            return obj.__json__()
        elif isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
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
        """
        Get the path to api_keys.json with backwards compatibility.

        Checks vanta_api first, then falls back to vanta_api for backwards compatibility
        during the migration period.

        ptn_api is deprecated, and support will be removed in the future.
        """
        vanta_path = ValiConfig.BASE_DIR + "/vanta_api/api_keys.json"
        ptn_path = ValiConfig.BASE_DIR + "/ptn_api/api_keys.json"

        # Prefer vanta_api, but fall back to ptn_api if vanta doesn't exist
        if os.path.exists(vanta_path):
            return vanta_path
        elif os.path.exists(ptn_path):
            import bittensor as bt
            bt.logging.warning(
                "⚠️  Using api_keys.json from ptn_api/ (deprecated). "
                "Please run vanta_api/migrate_from_ptn.sh to migrate your files."
            )
            return ptn_path

        # Default to vanta_api path if neither exists
        return vanta_path

    @staticmethod
    def get_positions_override_dir(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/data/positions_overrides/"

    @staticmethod
    def get_miner_all_positions_dir(miner_hotkey, running_unit_tests=False) -> str:
        return f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}{miner_hotkey}/positions/"

    @staticmethod
    def get_miner_archived_positions_dir(miner_hotkey, running_unit_tests=False) -> str:
        return f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}{miner_hotkey}/archived_positions/"

    @staticmethod
    def archive_position(
        hotkey: str,
        position,
        running_unit_tests: bool = False
    ) -> bool:
        """Move a single position file to archived_positions/ directory on disk.

        Returns:
            True if the file was moved, False if the source file did not exist
        """
        dst_base = ValiBkpUtils.get_miner_archived_positions_dir(hotkey, running_unit_tests=running_unit_tests)
        positions_base = ValiBkpUtils.get_miner_all_positions_dir(hotkey, running_unit_tests=running_unit_tests)

        trade_pair_id = position.trade_pair.trade_pair_id
        order_status = OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED
        src_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
            hotkey, trade_pair_id, order_status, running_unit_tests=running_unit_tests
        )
        src_file = os.path.join(src_dir, position.position_uuid)
        if not os.path.exists(src_file):
            return False
        rel = os.path.relpath(src_file, positions_base)
        dst_file = os.path.join(dst_base, rel)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.move(src_file, dst_file)
        return True

    @staticmethod
    def get_miner_transactions_path(miner_hotkey: str, running_unit_tests=False) -> str:
        """Get path to miner's transactions.jsonl file."""
        return f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}{miner_hotkey}/transactions.jsonl"

    @staticmethod
    def append_transaction(file_path: str, transaction: dict) -> None:
        """Atomically append a transaction to NDJSON file with file locking."""
        import fcntl
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        json_line = json.dumps(transaction, separators=(',', ':')) + '\n'
        with open(file_path, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json_line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def read_transactions(file_path: str) -> list:
        """Read all transactions from NDJSON file. Returns [] if not exists."""
        if not os.path.exists(file_path):
            return []
        transactions = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    transactions.append(json.loads(line))
        return transactions

    @staticmethod
    def clear_transactions(file_path: str) -> None:
        """Remove transactions file if it exists."""
        if os.path.exists(file_path):
            os.remove(file_path)

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
        """Get current perf_ledgers path (compressed JSON format)."""
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/perf_ledgers.json.gz"

    @staticmethod
    def get_perf_ledgers_path_pkl(running_unit_tests=False) -> str:
        """Get .pkl path (for migration from bug that wrote .json.gz data with .pkl extension)."""
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/perf_ledgers.pkl"

    @staticmethod
    def get_perf_ledgers_path_legacy(running_unit_tests=False) -> str:
        """Get legacy uncompressed perf_ledgers path for migration."""
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/perf_ledgers.json"

    @staticmethod
    def migrate_perf_ledgers_to_compressed(running_unit_tests=False) -> bool:
        """
        Migrate perf_ledgers from .pkl or .json to .json.gz and delete old file.

        Handles three migration scenarios:
        1. .pkl file (created by bug - contains gzip JSON with wrong extension)
        2. .json file (legacy uncompressed format)
        3. Already migrated (.json.gz exists) - no action needed

        Returns:
            bool: True if migration occurred, False otherwise
        """
        new_path = ValiBkpUtils.get_perf_ledgers_path(running_unit_tests)

        # Priority 1: Check for .pkl file (from bug - most recent format issue)
        pkl_path = ValiBkpUtils.get_perf_ledgers_path_pkl(running_unit_tests)
        if os.path.exists(pkl_path):
            try:
                # The .pkl file contains gzip-compressed JSON (created by write_compressed_json)
                # despite having the wrong extension, so read it as compressed JSON
                data = ValiBkpUtils.read_compressed_json(pkl_path)

                # Write to correct .json.gz path
                ValiBkpUtils.write_compressed_json(new_path, data)

                # Delete the misnamed .pkl file after successful migration
                os.remove(pkl_path)
                bt.logging.info(f"Migrated perf_ledgers from {pkl_path} to {new_path}")
                return True

            except Exception as e:
                bt.logging.error(f"Failed to migrate perf_ledgers from .pkl: {e}")
                return False

        # Priority 2: Check for legacy .json file (original uncompressed format)
        legacy_path = ValiBkpUtils.get_perf_ledgers_path_legacy(running_unit_tests)
        if os.path.exists(legacy_path):
            try:
                # Read legacy uncompressed file
                with open(legacy_path, 'r') as f:
                    data = json.load(f)

                # Write to compressed format
                ValiBkpUtils.write_compressed_json(new_path, data)

                # Delete legacy file after successful migration
                os.remove(legacy_path)
                bt.logging.info(f"Migrated perf_ledgers from {legacy_path} to {new_path}")
                return True

            except Exception as e:
                bt.logging.error(f"Failed to migrate perf_ledgers from .json: {e}")
                return False

        # No migration needed - already using .json.gz or no file exists
        return False

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
    def get_entity_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/entities.json"

    @staticmethod
    def get_entity_collateral_cache_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/entity_collateral_cache.json"

    @staticmethod
    def get_entity_slash_tracking_file_location(running_unit_tests=False) -> str:
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/validation/entity_slash_tracking.json"

    @staticmethod
    def get_hl_backup_watermarks_path() -> str:
        return ValiConfig.BASE_DIR + "/validation/hl_backup_poll_watermarks.json"

    @staticmethod
    def get_hl_observed_szi_path() -> str:
        return ValiConfig.BASE_DIR + "/validation/hl_observed_szi.json"

    @staticmethod
    def get_secrets_dir():
        return ValiConfig.BASE_DIR + "/secrets.json"

    @staticmethod
    def get_taoshi_api_keys_file_location():
        return ValiConfig.BASE_DIR + "/config-development.json"

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
    def get_vcp_output_path(running_unit_tests=False) -> str:
        """Get path for compressed validator checkpoint output file.

        Args:
            running_unit_tests: If True, returns test-specific path

        Returns:
            Full path to compressed validator checkpoint output file (.gz)
        """
        suffix = "/tests" if running_unit_tests else ""
        return ValiConfig.BASE_DIR + f"{suffix}/runnable/validator_checkpoint.json.gz"

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
    def clear_tmp_dir():
        temp_dir = ValiBkpUtils.get_temp_file_path()
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))

    @staticmethod
    def clear_directory(directory: str) -> None:
        """
        Clear all contents of a directory. If the directory doesn't exist, do nothing.
        Useful for cleaning up test data before test runs.

        Args:
            directory: Full path to directory to clear
        """
        if os.path.exists(directory):
            shutil.rmtree(directory)
            bt.logging.debug(f"Cleared directory: {directory}")

    @staticmethod
    def clear_all_miner_directories(running_unit_tests=False):
        """
        Clear all miner directories from disk (for testing).

        This removes the entire miners/ directory and recreates it empty.
        CAUTION: This will delete all position data on disk!

        Args:
            running_unit_tests: If True, clears test directories; else production
        """
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)
        if os.path.exists(miner_dir):
            shutil.rmtree(miner_dir)
            bt.logging.info(f"Cleared all miner directories from {miner_dir}")
        # Recreate empty directory
        os.makedirs(miner_dir, exist_ok=True)

    @staticmethod
    def write_json_stream(stream, data) -> None:
        """
        Writes a JSON document to a stream in a more efficient manner than the built-in json
        module. In addition to using the orjson library, which is faster at encoding, it
        also attempts to break up documents that contain large collections into smaller,
        more manageable chunks to reduce peak memory usage. It relies on the assumption that
        most large collections contain relatively small elements (less than 1 MiB). This is
        an imperfect solution, but a sensible compromise between performance and memory
        usage.

        The non-streaming method causes memory issues for large documents because it
        encodes the entire document into memory before writing to the stream:

          stream.write(json.dumps(data))

        The streaming method is very slow for large documents because it iterates over
        every element in a very inefficient manner:

          json.dump(stream, data)

        Fixes to address the iteration performance issue in json.dump have been submitted to
        CPython for several years but have always been rejected because they would increase
        the complexity of the reference source code. The standard recommendation is to use
        alternative libraries when writing large documents to a stream.
        See: https://github.com/python/cpython/pull/130076
        """
        _LARGE_COLLECTION_SIZE = 32

        if isinstance(data, dict):
            large_collection = len(data) > _LARGE_COLLECTION_SIZE
            stream.write(b"{")
            first = True
            for key, value in data.items():
                if first:
                    first = False
                else:
                    stream.write(b",")

                # Add quotes around keys that are not strings
                key_is_not_str = not isinstance(key, str)
                if key_is_not_str:
                    stream.write(b'"')
                stream.write(orjson.dumps(key))
                if key_is_not_str:
                    stream.write(b'"')

                stream.write(b":")

                if large_collection:
                    stream.write(orjson.dumps(value, default=orjson_encoder, option=orjson.OPT_NON_STR_KEYS))
                else:
                    ValiBkpUtils.write_json_stream(stream, value)
            stream.write(b"}")

        elif isinstance(data, list) or isinstance(data, tuple):
            large_collection = len(data) > _LARGE_COLLECTION_SIZE
            stream.write(b"[")
            first = True
            for item in data:
                if first:
                    first = False
                else:
                    stream.write(b",")
                if large_collection:
                    stream.write(orjson.dumps(item, default=orjson_encoder, option=orjson.OPT_NON_STR_KEYS))
                else:
                    ValiBkpUtils.write_json_stream(stream, item)
            stream.write(b"]")

        else:
            stream.write(orjson.dumps(data, default=orjson_encoder, option=orjson.OPT_NON_STR_KEYS))

    @staticmethod
    def write_compressed_json(file_path: str, data: dict) -> None:
        """Write JSON data compressed with gzip (atomic write via temp file)."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with NamedTemporaryFile(mode="wb", delete=False) as temp_file:
            with gzip.open(temp_file, "wb") as gzip_stream:
                ValiBkpUtils.write_json_stream(gzip_stream, data)
        shutil.move(temp_file.name, file_path)

    @staticmethod
    def read_compressed_json(file_path: str) -> dict:
        """Read compressed JSON data."""
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def write_file(file_path: str, data: dict | object) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with NamedTemporaryFile(mode="wb", delete=False) as temp_file:
            ValiBkpUtils.write_json_stream(temp_file, data)
        shutil.move(temp_file.name, file_path)

    @staticmethod
    def get_file(vali_file: str) -> str | object:
        with open(vali_file, "r") as f:
            return f.read()

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

    @staticmethod
    def get_limit_orders_dir(miner_hotkey, trade_pair_id, status_str, running_unit_tests=False):
        base_dir = (f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}"
               f"{miner_hotkey}/limit_orders/{trade_pair_id}/")

        return f"{base_dir}{status_str}/"

    @staticmethod
    def get_limit_orders(miner_hotkey, unfilled_only=False, *, running_unit_tests=False):
        miner_limit_orders_dir = (f"{ValiBkpUtils.get_miner_dir(running_unit_tests=running_unit_tests)}"
                                  f"{miner_hotkey}/limit_orders/")

        if not os.path.exists(miner_limit_orders_dir):
            return []

        orders = []
        trade_pair_dirs = ValiBkpUtils.get_directories_in_dir(miner_limit_orders_dir)
        if unfilled_only:
            status_dirs = ["unfilled"]
        else:
            status_dirs = ["unfilled", "closed"]
        for trade_pair_id in trade_pair_dirs:
            for status in status_dirs:
                status_dir = ValiBkpUtils.get_limit_orders_dir(miner_hotkey, trade_pair_id, status, running_unit_tests)

                if not os.path.exists(status_dir):
                    continue

                try:
                    status_files = ValiBkpUtils.get_all_files_in_dir(status_dir)
                    for filename in status_files:
                        with open(filename, 'r') as f:
                            orders.append(json.load(f))

                except Exception as e:
                    bt.logging.error(f"Error accessing {status} directory {status_dir}: {e}")

        return orders
