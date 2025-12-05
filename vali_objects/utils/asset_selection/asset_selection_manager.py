# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
AssetSelectionManager - Business logic for asset class selection.

This manager contains all the business logic for managing asset class selections.
It does NOT handle RPC - that's the job of AssetSelectionServer.

Miners can select an asset class (forex, crypto, etc.) only once.
Once selected, the miner cannot trade any trade pair from a different asset class.
Asset selections are persisted to disk and loaded on startup.
"""
import threading
from typing import Dict

import asyncio
import bittensor as bt

import template.protocol
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

ASSET_CLASS_SELECTION_TIME_MS = 1758326340000


class AssetSelectionManager:
    """
    Manages asset class selection for miners (business logic only).

    Each miner can select an asset class (forex, crypto, etc.) only once.
    Once selected, the miner cannot trade any trade pair from a different asset class.
    Asset selections are persisted to disk and loaded on startup.

    This class contains NO RPC code - only business logic.
    For RPC access, use AssetSelectionServer (which wraps this manager).
    """

    def __init__(
        self,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        config=None
    ):
        """
        Initialize the AssetSelectionManager.

        Args:
            running_unit_tests: Whether the manager is being used in unit tests
            connection_mode: Connection mode (RPC vs LOCAL for tests)
            config: Validator config (for netuid, wallet) - optional, used to initialize wallet
        """
        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode
        self.is_mothership = 'ms' in ValiUtils.get_secrets(running_unit_tests=running_unit_tests)

        # FIX: Create lock immediately in __init__, not lazy!
        # This prevents the race condition where multiple threads could create separate lock instances
        self._asset_selection_lock = threading.RLock()

        # Create own MetagraphClient (forward compatibility - no parameter passing)
        from shared_objects.rpc.metagraph_client import MetagraphClient
        self._metagraph_client = MetagraphClient(connection_mode=connection_mode)

        # Initialize wallet directly
        if not running_unit_tests and config is not None:
            self.is_testnet = config.netuid == 116
            self._wallet = bt.wallet(config=config)
            bt.logging.info("[ASSET_MGR] Wallet initialized")
        else:
            self.is_testnet = False
            self._wallet = None

        # SOURCE OF TRUTH: Normal Python dict
        # Structure: miner_hotkey -> TradePairCategory
        self.asset_selections: Dict[str, TradePairCategory] = {}

        self.ASSET_SELECTIONS_FILE = ValiBkpUtils.get_asset_selections_file_location(
            running_unit_tests=running_unit_tests
        )
        self._load_asset_selections_from_disk()

        bt.logging.info(f"[ASSET_MGR] AssetSelectionManager initialized with {len(self.asset_selections)} selections")

    @property
    def asset_selection_lock(self):
        """Thread-safe lock for protecting asset_selections dict access"""
        return self._asset_selection_lock

    @property
    def wallet(self):
        """Get wallet."""
        return self._wallet

    @property
    def metagraph(self):
        """Get metagraph client (created internally)"""
        return self._metagraph_client

    # ==================== Persistence Methods ====================

    def _load_asset_selections_from_disk(self) -> None:
        """Load asset selections from disk into memory using ValiUtils pattern."""
        try:
            disk_data = ValiUtils.get_vali_json_file_dict(self.ASSET_SELECTIONS_FILE)
            parsed_selections = self._parse_asset_selections_dict(disk_data)

            # FIX: Protect clear + update with lock to prevent data loss from concurrent access
            with self._asset_selection_lock:
                self.asset_selections.clear()
                self.asset_selections.update(parsed_selections)

            bt.logging.info(f"[ASSET_MGR] Loaded {len(parsed_selections)} asset selections from disk")
        except Exception as e:
            bt.logging.error(f"[ASSET_MGR] Error loading asset selections from disk: {e}")

    def _save_asset_selections_to_disk(self) -> None:
        """
        Save asset selections from memory to disk using ValiBkpUtils pattern.

        IMPORTANT: Caller MUST hold self._asset_selection_lock before calling this method!
        This ensures thread-safe iteration over asset_selections and prevents concurrent writes.
        """
        try:
            selections_data = self._to_dict()
            ValiBkpUtils.write_file(self.ASSET_SELECTIONS_FILE, selections_data)
            bt.logging.debug(f"[ASSET_MGR] Saved {len(selections_data)} asset selections to disk")
        except Exception as e:
            bt.logging.error(f"[ASSET_MGR] Error saving asset selections to disk: {e}")

    def _to_dict(self) -> Dict:
        """
        Convert in-memory asset selections to disk format.

        IMPORTANT: Caller MUST hold self._asset_selection_lock before calling this method!
        This prevents RuntimeError from dict modification during iteration.
        """
        return {
            hotkey: asset_class.value
            for hotkey, asset_class in self.asset_selections.items()
        }

    @staticmethod
    def _parse_asset_selections_dict(json_dict: Dict) -> Dict[str, TradePairCategory]:
        """Parse disk format back to in-memory format."""
        parsed_selections = {}

        for hotkey, asset_class_str in json_dict.items():
            try:
                if asset_class_str:
                    # Convert string back to TradePairCategory enum
                    asset_class = TradePairCategory(asset_class_str)
                    parsed_selections[hotkey] = asset_class
            except ValueError as e:
                bt.logging.warning(f"[ASSET_MGR] Invalid asset selection for miner {hotkey}: {e}")
                continue

        return parsed_selections

    def broadcast_asset_selection_to_validators(self, hotkey: str, asset_selection: TradePairCategory):
        """
        Broadcast AssetSelection synapse to other validators.
        Runs in a separate thread to avoid blocking the main process.

        Args:
            hotkey: The miner's hotkey
            asset_selection: The TradePairCategory enum value
        """
        def run_broadcast():
            try:
                asyncio.run(self._async_broadcast_asset_selection(hotkey, asset_selection))
            except Exception as e:
                bt.logging.error(f"[ASSET_MGR] Failed to broadcast asset selection for {hotkey}: {e}")

        thread = threading.Thread(target=run_broadcast, daemon=True)
        thread.start()

    async def _async_broadcast_asset_selection(self, hotkey: str, asset_selection: TradePairCategory):
        """
        Asynchronously broadcast AssetSelection synapse to other validators.

        Args:
            hotkey: The miner's hotkey
            asset_selection: The TradePairCategory enum value
        """
        try:
            if not self.wallet:
                bt.logging.debug("[ASSET_MGR] No wallet configured, skipping broadcast")
                return

            if not self.metagraph:
                bt.logging.debug("[ASSET_MGR] No metagraph configured, skipping broadcast")
                return

            # Get other validators to broadcast to
            if self.is_testnet:
                validator_axons = [
                    n.axon_info for n in self.metagraph.get_neurons()
                    if n.axon_info.ip != ValiConfig.AXON_NO_IP
                    and n.axon_info.hotkey != self.wallet.hotkey.ss58_address
                ]
            else:
                validator_axons = [
                    n.axon_info for n in self.metagraph.get_neurons()
                    if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                    and n.axon_info.ip != ValiConfig.AXON_NO_IP
                    and n.axon_info.hotkey != self.wallet.hotkey.ss58_address
                ]

            if not validator_axons:
                bt.logging.debug("[ASSET_MGR] No other validators to broadcast AssetSelection to")
                return

            # Create AssetSelection synapse with the data
            asset_selection_data = {
                "hotkey": hotkey,
                "asset_selection": asset_selection.value if hasattr(asset_selection, 'value') else str(asset_selection)
            }

            asset_selection_synapse = template.protocol.AssetSelection(
                asset_selection=asset_selection_data
            )

            bt.logging.info(f"[ASSET_MGR] Broadcasting AssetSelection for {hotkey} to {len(validator_axons)} validators")

            # Send to other validators using dendrite
            async with bt.dendrite(wallet=self.wallet) as dendrite:
                responses = await dendrite.aquery(validator_axons, asset_selection_synapse)

                # Log results
                success_count = 0
                for response in responses:
                    if response.successfully_processed:
                        success_count += 1
                    elif response.error_message:
                        bt.logging.warning(
                            f"[ASSET_MGR] Failed to send AssetSelection to {response.axon.hotkey}: {response.error_message}"
                        )

                bt.logging.info(
                    f"[ASSET_MGR] AssetSelection broadcast completed: {success_count}/{len(responses)} validators updated"
                )

        except Exception as e:
            bt.logging.error(f"[ASSET_MGR] Error in async broadcast asset selection: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    # ==================== Query Methods ====================

    def is_valid_asset_class(self, asset_class: str) -> bool:
        """
        Validate if the provided asset class is valid.

        Args:
            asset_class: The asset class string to validate

        Returns:
            True if valid, False otherwise
        """
        valid_asset_classes = [category.value for category in TradePairCategory]
        return asset_class.lower() in [cls.lower() for cls in valid_asset_classes]

    def validate_order_asset_class(
        self,
        miner_hotkey: str,
        trade_pair_category: TradePairCategory,
        timestamp_ms: int = None
    ) -> bool:
        """
        Check if a miner is allowed to trade a specific asset class.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_category: The trade pair category to check
            timestamp_ms: Optional timestamp in milliseconds

        Returns:
            True if the miner can trade this asset class, False otherwise
        """
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()
        if timestamp_ms < ASSET_CLASS_SELECTION_TIME_MS:
            return True

        # FIX: Protect read with lock to prevent TOCTOU race
        # Without lock, could read empty dict during sync or get stale data
        with self._asset_selection_lock:
            selected_asset_class = self.asset_selections.get(miner_hotkey, None)
            if selected_asset_class is None:
                return False

            # Check if the selected asset class matches the trade pair category
            return selected_asset_class == trade_pair_category

    def get_asset_selections(self) -> Dict[str, TradePairCategory]:
        """
        Get the asset_selections dict (copy).

        Returns:
            Dict[str, TradePairCategory]: Dictionary mapping hotkey to TradePairCategory enum
        """
        # FIX: Protect dict copy with lock to prevent torn reads
        # Without lock, could see partial state if dict modified during copy
        with self._asset_selection_lock:
            return dict(self.asset_selections)

    def get_asset_selection(self, hotkey: str) -> TradePairCategory | None:
        with self._asset_selection_lock:
            return self.asset_selections.get(hotkey)

    def get_all_miner_selections(self) -> Dict[str, str]:
        """
        Get all miner asset selections as a dictionary.

        Returns:
            Dict[str, str]: Dictionary mapping miner hotkeys to their asset class selections (as strings).
                           Returns empty dict if no selections exist.
        """
        try:
            # Only need lock for the copy operation to get a consistent snapshot
            with self.asset_selection_lock:
                # Convert the dict to a regular dict
                selections_copy = dict(self.asset_selections)

            # Lock not needed here - working with local copy
            # Convert TradePairCategory objects to their string values
            return {
                hotkey: asset_class.value if hasattr(asset_class, 'value') else str(asset_class)
                for hotkey, asset_class in selections_copy.items()
            }
        except Exception as e:
            bt.logging.error(f"[ASSET_MGR] Error getting all miner selections: {e}")
            return {}

    # ==================== Mutation Methods ====================

    def process_asset_selection_request(self, asset_selection: str, miner: str) -> Dict[str, str]:
        """
        Process an asset selection request from a miner.

        Args:
            asset_selection: The asset class the miner wants to select
            miner: The miner's hotkey

        Returns:
            Dict containing success status and message

        Note:
            This method does NOT broadcast to validators - that's the server's job.
            The server will call this method and then handle broadcasting.
        """
        try:
            # Validate asset class (read-only, safe outside lock)
            if not self.is_valid_asset_class(asset_selection):
                valid_classes = [category.value for category in TradePairCategory]
                return {
                    'successfully_processed': False,
                    'error_message': f'Invalid asset class. Valid options are: {", ".join(valid_classes)}'
                }

            # Convert string to TradePairCategory
            asset_class = TradePairCategory(asset_selection.lower())

            # FIX: Move check inside lock for atomic check-then-set
            # This prevents race where multiple threads could all pass the check before any sets the value
            with self._asset_selection_lock:
                # Re-check inside lock (double-checked locking pattern)
                if miner in self.asset_selections:
                    current_selection = self.asset_selections.get(miner)
                    return {
                        'successfully_processed': False,
                        'error_message': f'Asset class already selected: {current_selection.value}. Cannot change selection.'
                    }

                # Atomic check-then-set: Both check and set now happen atomically
                self.asset_selections[miner] = asset_class
                self._save_asset_selections_to_disk()

            bt.logging.info(f"[ASSET_MGR] Miner {miner} selected asset class: {asset_selection}")

            return {
                'successfully_processed': True,
                'success_message': f'Miner {miner} successfully selected asset class: {asset_selection}',
                'asset_class': asset_class  # Return the enum for server to use in broadcast
            }

        except Exception as e:
            bt.logging.error(f"[ASSET_MGR] Error processing asset selection request for miner {miner}: {e}")
            return {
                'successfully_processed': False,
                'error_message': 'Internal server error processing asset selection request'
            }

    def sync_miner_asset_selection_data(self, asset_selection_data: Dict[str, str]) -> None:
        """
        Sync miner asset selection data from external source (backup/sync).

        Args:
            asset_selection_data: Dict mapping hotkey to asset class string
        """
        if not asset_selection_data:
            bt.logging.warning("[ASSET_MGR] asset_selection_data appears empty or invalid")
            return
        try:
            # Parse outside lock (can take time, doesn't need lock)
            synced_data = self._parse_asset_selections_dict(asset_selection_data)

            # FIX: Use atomic replacement instead of clear + update
            # This prevents readers from seeing empty dict during the clear-then-populate gap
            with self._asset_selection_lock:
                # Option 1: Atomic replacement (recommended for visibility)
                # Old data visible until new data ready
                self.asset_selections = synced_data

                # Option 2 (commented): Clear + update if dict identity must be preserved
                # self.asset_selections.clear()
                # self.asset_selections.update(synced_data)

                self._save_asset_selections_to_disk()

            bt.logging.info(f"[ASSET_MGR] Synced {len(synced_data)} miner asset selection records")
        except Exception as e:
            bt.logging.error(f"[ASSET_MGR] Failed to sync miner asset selection data: {e}")

    def receive_asset_selection_update(self, asset_selection_data: dict) -> bool:
        """
        Process an incoming asset selection update from another validator.

        Args:
            asset_selection_data: Dictionary containing hotkey, asset selection

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.is_mothership:
                return False

            with self.asset_selection_lock:
                # Extract data from the synapse
                hotkey = asset_selection_data.get("hotkey")
                asset_selection = asset_selection_data.get("")
                bt.logging.info(f"[ASSET_MGR] Processing asset selection for miner {hotkey}")

                if not all([hotkey, asset_selection is not None]):
                    bt.logging.warning(f"[ASSET_MGR] Invalid asset selection data received: {asset_selection_data}")
                    return False

                # Check if we already have this record (avoid duplicates)
                if hotkey in self.asset_selections:
                    bt.logging.debug(f"[ASSET_MGR] Asset selection for {hotkey} already exists")
                    return True

                # Parse the asset selection string to TradePairCategory
                try:
                    if isinstance(asset_selection, str):
                        asset_class = TradePairCategory(asset_selection.lower())
                    else:
                        # Already a TradePairCategory
                        asset_class = asset_selection
                except ValueError as e:
                    bt.logging.warning(f"[ASSET_MGR] Invalid asset class value: {asset_selection}: {e}")
                    return False

                # Add the new record
                self.asset_selections[hotkey] = asset_class

                # Save to disk
                self._save_asset_selections_to_disk()

                bt.logging.info(f"[ASSET_MGR] Updated miner asset selection for {hotkey}: {asset_selection}")
                return True

        except Exception as e:
            bt.logging.error(f"[ASSET_MGR] Error processing asset selection update: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False
