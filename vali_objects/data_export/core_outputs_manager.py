# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
CoreOutputsManager - Business logic for checkpoint generation and core outputs.

This manager contains the heavy business logic for generating validator checkpoints,
managing positions data, and handling production file creation.

The CoreOutputsServer wraps this manager and exposes its methods via RPC.

This follows the same pattern as PerfLedgerManager/PerfLedgerServer and
EliminationManager/EliminationServer.

Usage:
    # Typically created by CoreOutputsServer
    manager = CoreOutputsManager(
        running_unit_tests=False
    )

    # Generate checkpoint
    checkpoint = manager.generate_request_core()
"""

import copy
import gzip
import json
import os
import hashlib
import bittensor as bt

from google.cloud import storage

from time_util.time_util import TimeUtil
from vali_objects.price_fetcher import LivePriceFetcherClient
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils, CustomEncoder
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.data_sync.validator_sync_base import AUTO_SYNC_ORDER_LAG_MS


# no filters,... , max filter
PERCENT_NEW_POSITIONS_TIERS = [100, 50, 30, 0]
assert sorted(PERCENT_NEW_POSITIONS_TIERS, reverse=True) == PERCENT_NEW_POSITIONS_TIERS, 'needs to be sorted for efficient pruning'


class CoreOutputsManager:
    """
    Business logic manager for checkpoint generation and core outputs.

    Contains the heavy business logic for generating validator checkpoints,
    while CoreOutputsServer wraps it and exposes methods via RPC.

    This follows the same pattern as PerfLedgerManager and EliminationManager.
    """

    def __init__(
        self,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize CoreOutputsManager.

        Args:
            running_unit_tests: Whether running in unit test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode
        self.live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests, connection_mode=connection_mode)

        # Create own RPC clients (forward compatibility - no parameter passing)
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
        from vali_objects.utils.elimination.elimination_client import EliminationClient
        from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
        from vali_objects.contract.contract_client import ContractClient
        from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        from entity_management.entity_client import EntityClient

        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connect_immediately=not running_unit_tests
        )
        self._challengeperiod_client = ChallengePeriodClient()
        self._elimination_client = EliminationClient()
        # PerfLedgerClient for perf ledger operations (forward compatibility)
        self._perf_ledger_client = PerfLedgerClient(connection_mode=connection_mode)
        # LimitOrderClient for limit order operations (forward compatibility)
        self._limit_order_client = LimitOrderClient(connection_mode=connection_mode)
        # Create own ContractClient (forward compatibility - no parameter passing)
        self._contract_client = ContractClient(connection_mode=connection_mode)
        # MinerAccountClient for miner account operations (forward compatibility)
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode)
        # AssetSelectionClient for asset selection operations (forward compatibility)
        self._asset_selection_client = AssetSelectionClient(connection_mode=connection_mode)
        # EntityClient for entity operations (forward compatibility)
        self._entity_client = EntityClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)

        # Manager uses regular dict (no IPC needed - managed by server)
        self.validator_checkpoint_cache = {}

        bt.logging.info(f"[COREOUTPUTS_MANAGER] CoreOutputsManager initialized")

    # ==================== Properties (Forward Compatibility) ====================

    @property
    def position_manager(self):
        """Get position manager client."""
        return self._position_client

    @property
    def elimination_manager(self):
        """Get elimination manager client."""
        return self._elimination_client

    @property
    def challengeperiod_manager(self):
        """Get challenge period client."""
        return self._challengeperiod_client

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    # ==================== Core Business Logic ====================

    def hash_string_to_int(self, s: str) -> int:
        """Hash string to integer using SHA-256."""
        hash_object = hashlib.sha256()
        hash_object.update(s.encode('utf-8'))
        hex_digest = hash_object.hexdigest()
        hash_int = int(hex_digest, 16)
        return hash_int

    def filter_new_positions_random_sample(
        self,
        percent_new_positions_keep: float,
        hotkey_to_positions: dict[str:[dict]],
        time_of_position_read_ms: int
    ) -> None:
        """Filter positions based on tier percentage."""
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

            new_orders = []
            for order in position_to_truncate.orders:
                if order.processed_ms < stale_date_threshold_ms:
                    new_orders.append(order)

            if len(new_orders):
                position_to_truncate.orders = new_orders
                position_to_truncate.rebuild_position_with_updated_orders(self.live_price_client)
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

    @staticmethod
    def cleanup_test_files():
        """
        Clean up files created by generate_request_core for testing.

        This removes:
        - Compressed validator checkpoint
        - Miner positions at all tier levels (100, 50, 30, 0)
        """
        # Remove compressed checkpoint from test directory
        try:
            compressed_path = ValiBkpUtils.get_vcp_output_path(running_unit_tests=True)
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
        except Exception as e:
            print(f"Error removing compressed checkpoint: {e}")

        # Remove miner positions at all tiers
        for tier in PERCENT_NEW_POSITIONS_TIERS:
            try:
                suffix_dir = None if tier == 100 else str(tier)
                positions_path = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=suffix_dir)
                if os.path.exists(positions_path):
                    os.remove(positions_path)
            except Exception as e:
                print(f"Error removing positions file for tier {tier}: {e}")

    def compress_dict(self, data: dict) -> bytes:
        """Compress dict to gzip bytes."""
        str_to_write = json.dumps(data, cls=CustomEncoder)
        compressed = gzip.compress(str_to_write.encode("utf-8"))
        return compressed

    def decompress_dict(self, compressed_data: bytes) -> dict:
        """Decompress gzip bytes to dict."""
        decompressed = gzip.decompress(compressed_data)
        data = json.loads(decompressed.decode("utf-8"))
        return data

    def store_checkpoint_in_memory(self, checkpoint_data: dict):
        """Store compressed validator checkpoint data in memory cache."""
        try:
            compressed_data = self.compress_dict(checkpoint_data)
            self.validator_checkpoint_cache['checkpoint'] = {
                'data': compressed_data,
                'timestamp_ms': TimeUtil.now_in_millis()
            }
        except Exception as e:
            bt.logging.error(f"Error storing checkpoint in memory: {e}")

    def get_compressed_checkpoint_from_memory(self) -> bytes | None:
        """
        Retrieve compressed validator checkpoint data directly from memory cache.

        Returns:
            Cached compressed gzip bytes of checkpoint JSON (None if cache not built yet)
        """
        try:
            cached_entry = self.validator_checkpoint_cache.get('checkpoint', {})
            if not cached_entry or 'data' not in cached_entry:
                return None

            return cached_entry['data']
        except Exception as e:
            bt.logging.error(f"Error retrieving compressed checkpoint from memory: {e}")
            return None

    def upload_checkpoint_to_gcloud(self, final_dict):
        """
        Upload a zipped, time lagged validator checkpoint to google cloud for auto restoration
        on other validators as well as transparency with the community.
        """
        datetime_now = TimeUtil.generate_start_timestamp(0)  # UTC
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
        blob_name = 'validator_checkpoint.json.gz'

        # Create a new blob and upload data
        blob = bucket.blob(blob_name)

        # Create a zip file in memory
        zip_buffer = self.compress_dict(final_dict)
        # Upload the content of the zip_buffer to Google Cloud Storage
        blob.upload_from_string(zip_buffer)
        bt.logging.info(f'Uploaded {blob_name} to {bucket_name}')

    def create_and_upload_production_files(
        self,
        eliminations,
        ord_dict_hotkey_position_map,
        time_now,
        youngest_order_processed_ms,
        oldest_order_processed_ms,
        challengeperiod_dict,
        miner_account_sizes_dict,
        limit_orders_dict,
        save_to_disk=True,
        upload_to_gcloud=True
    ):
        """Create and optionally upload production files."""
        perf_ledgers = self._perf_ledger_client.get_perf_ledgers(portfolio_only=False)

        # Get asset selections via RPC client (forward compatibility)
        asset_selections = {}
        try:
            asset_selections = self._asset_selection_client.get_all_miner_selections()
        except Exception as e:
            bt.logging.warning(f"Could not fetch asset selections: {e}")

        # Get entity data via RPC client (forward compatibility)
        entities_dict = {}
        try:
            entities_dict = self._entity_client.to_checkpoint_dict()
        except Exception as e:
            bt.logging.warning(f"Could not fetch entity data: {e}")

        final_dict = {
            'version': ValiConfig.VERSION,
            'created_timestamp_ms': time_now,
            'created_date': TimeUtil.millis_to_formatted_date_str(time_now),
            'challengeperiod': challengeperiod_dict,
            'miner_account_sizes': miner_account_sizes_dict,
            'eliminations': eliminations,
            'entities': entities_dict,
            'youngest_order_processed_ms': youngest_order_processed_ms,
            'oldest_order_processed_ms': oldest_order_processed_ms,
            'positions': ord_dict_hotkey_position_map,
            'perf_ledgers': perf_ledgers,
            'asset_selections': asset_selections,
            'limit_orders': limit_orders_dict
        }

        if save_to_disk:
            # Write compressed checkpoint only - saves disk space and bandwidth
            compressed_data = self.compress_dict(final_dict)

            # Write compressed file directly
            compressed_path = ValiBkpUtils.get_vcp_output_path(
                running_unit_tests=self.running_unit_tests
            )
            with open(compressed_path, 'wb') as f:
                f.write(compressed_data)

            # Store compressed checkpoint data in memory cache
            self.store_checkpoint_in_memory(final_dict)

            # Write positions data at different tiers
            for t in PERCENT_NEW_POSITIONS_TIERS:
                if t == 100:  # no filtering
                    # Write legacy location as well. no compression
                    ValiBkpUtils.write_file(
                        ValiBkpUtils.get_miner_positions_output_path(suffix_dir=None),
                        ord_dict_hotkey_position_map,
                    )
                else:
                    self.filter_new_positions_random_sample(t, ord_dict_hotkey_position_map, time_now)

                # "v2" add a tier. compress the data
                for hotkey, dat in ord_dict_hotkey_position_map.items():
                    dat['tier'] = t

                compressed_positions = self.compress_dict(ord_dict_hotkey_position_map)
                ValiBkpUtils.write_file(
                    ValiBkpUtils.get_miner_positions_output_path(suffix_dir=str(t)),
                    compressed_positions, is_binary=True
                )

        # Max filtering
        if upload_to_gcloud:
            self.upload_checkpoint_to_gcloud(final_dict)

    def generate_request_core(
        self,
        get_dash_data_hotkey: str | None = None,
        write_and_upload_production_files=False,
        create_production_files=True,
        save_production_files=False,
        upload_production_files=False
    ) -> dict:
        """
        Generate request core data and optionally create/save/upload production files.

        Args:
            get_dash_data_hotkey: Optional specific hotkey to query (for dashboard)
            write_and_upload_production_files: Legacy parameter - if True, creates/saves/uploads files
            create_production_files: If False, skips creating production file dicts
            save_production_files: If False, skips writing files to disk
            upload_production_files: If False, skips uploading to gcloud

        Returns:
            dict: Checkpoint data containing positions, challengeperiod, etc.
        """
        eliminations = self.elimination_manager.get_eliminations_from_memory()
        try:
            if not os.path.exists(ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)):
                raise FileNotFoundError
        except FileNotFoundError:
            raise Exception(
                f"directory for miners doesn't exist "
                f"[{ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)}]. Skip run for now."
            )

        if get_dash_data_hotkey:
            all_miner_hotkeys: list = [get_dash_data_hotkey]
        else:
            all_miner_hotkeys: list = ValiBkpUtils.get_directories_in_dir(
                ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
            )

        # Query positions
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

        # Get miner account sizes from miner account client
        miner_account_sizes_dict = {}
        if self._miner_account_client:
            miner_account_sizes_dict = self._miner_account_client.accounts_dict()

        # Handle legacy parameter
        if write_and_upload_production_files:
            create_production_files = True
            save_production_files = True
            upload_production_files = True

        if create_production_files:
            limit_orders_dict = {}
            if self._limit_order_client:
                limit_orders_dict = self._limit_order_client.get_all_limit_orders()

            if save_production_files or upload_production_files:
                self.create_and_upload_production_files(
                    eliminations, ord_dict_hotkey_position_map, time_now_ms,
                    youngest_order_processed_ms, oldest_order_processed_ms,
                    challengeperiod_dict, miner_account_sizes_dict, limit_orders_dict,
                    save_to_disk=save_production_files,
                    upload_to_gcloud=upload_production_files
                )

        checkpoint_dict = {
            'challengeperiod': challengeperiod_dict,
            'miner_account_sizes': miner_account_sizes_dict,
            'positions': unfiltered_positions
        }
        return checkpoint_dict
