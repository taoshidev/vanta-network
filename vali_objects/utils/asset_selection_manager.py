import asyncio
import threading

import bittensor as bt
from typing import Dict, Optional

import template.protocol
from time_util.time_util import TimeUtil
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.vali_config import TradePairCategory, ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

ASSET_CLASS_SELECTION_TIME_MS = 1758326340000

class AssetSelectionManager:
    """
    Manages asset class selection for miners. Each miner can select an asset class (forex, crypto, etc.) 
    only once. Once selected, the miner cannot trade any trade pair from a different asset class.
    Asset selections are persisted to disk and loaded on startup.
    """

    def __init__(self, config=None, metagraph=None, ipc_manager=None, running_unit_tests=False):
        """
        Initialize the AssetSelectionManager.
        
        Args:
            running_unit_tests: Whether the manager is being used in unit tests
        """
        self.running_unit_tests = running_unit_tests
        self.metagraph = metagraph
        self.is_mothership = 'ms' in ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        self._asset_selection_lock = None

        if not self.running_unit_tests and config is not None:
            self.is_testnet = config.netuid == 116
            self.wallet = bt.wallet(config=config)
        else:
            self.is_testnet = False
            self.wallet = None

        if ipc_manager:
            self.asset_selections = ipc_manager.dict()
        else:
            self.asset_selections: Dict[str, TradePairCategory] = {}  # miner_hotkey -> TradePairCategory

        self.ASSET_SELECTIONS_FILE = ValiBkpUtils.get_asset_selections_file_location(running_unit_tests=running_unit_tests)
        self._load_asset_selections_from_disk()

    def receive_asset_selection(self, synapse: template.protocol.AssetSelection) -> template.protocol.AssetSelection:
        """
        receive miner's asset selection
        """
        try:
            # Process the collateral record through the contract manager
            sender_hotkey = synapse.dendrite.hotkey
            bt.logging.info(f"Received miner asset selection from validator hotkey [{sender_hotkey}].")
            success = self.receive_asset_selection_update(synapse.asset_selection)

            if success:
                synapse.successfully_processed = True
                synapse.error_message = ""
                bt.logging.info(f"Successfully processed AssetSelection synapse from {sender_hotkey}")
            else:
                synapse.successfully_processed = False
                synapse.error_message = "Failed to process miner's asset selection"
                bt.logging.warning(f"Failed to process AssetSelection synapse from {sender_hotkey}")

        except Exception as e:
            synapse.successfully_processed = False
            synapse.error_message = f"Error processing asset selection: {str(e)}"
            bt.logging.error(f"Exception in receive_asset_selection: {e}")

        return synapse

    @property
    def asset_selection_lock(self):
        if not self._asset_selection_lock:
            self._asset_selection_lock = threading.RLock()
        return self._asset_selection_lock

    def _load_asset_selections_from_disk(self) -> None:
        """Load asset selections from disk into memory using ValiUtils pattern."""
        try:
            disk_data = ValiUtils.get_vali_json_file_dict(self.ASSET_SELECTIONS_FILE)
            parsed_selections = self._parse_asset_selections_dict(disk_data)
            self.asset_selections.clear()
            self.asset_selections.update(parsed_selections)
            bt.logging.info(f"Loaded {len(self.asset_selections)} asset selections from disk")
        except Exception as e:
            bt.logging.error(f"Error loading asset selections from disk: {e}")

    def _save_asset_selections_to_disk(self) -> None:
        """Save asset selections from memory to disk using ValiBkpUtils pattern."""
        try:
            selections_data = self._to_dict()
            ValiBkpUtils.write_file(self.ASSET_SELECTIONS_FILE, selections_data)
            bt.logging.debug(f"Saved {len(self.asset_selections)} asset selections to disk")
        except Exception as e:
            bt.logging.error(f"Error saving asset selections to disk: {e}")

    def _to_dict(self) -> Dict:
        """Convert in-memory asset selections to disk format."""
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
                bt.logging.warning(f"Invalid asset selection for miner {hotkey}: {e}")
                continue
                
        return parsed_selections

    def sync_miner_asset_selection_data(self, asset_selection_data: Dict[str, str]):
        """Sync miner asset selection data from external source (backup/sync)"""
        if not asset_selection_data:
            bt.logging.warning("asset_selection_data appears empty or invalid")
            return
        try:
            with self.asset_selection_lock:
                synced_data = self._parse_asset_selections_dict(asset_selection_data)
                self.asset_selections.clear()
                self.asset_selections.update(synced_data)
                self._save_asset_selections_to_disk()
                bt.logging.info(f"Synced {len(self.asset_selections)} miner account size records")
        except Exception as e:
            bt.logging.error(f"Failed to sync miner account sizes data: {e}")


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

    def validate_order_asset_class(self, miner_hotkey: str, trade_pair_category: TradePairCategory, timestamp_ms: int=None) -> bool:
        """
        Check if a miner is allowed to trade a specific asset class.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_category: The trade pair category to check

        Returns:
            True if the miner can trade this asset class, False otherwise
        """
        import time

        # Time the timestamp operations
        ts_start = time.perf_counter()
        if timestamp_ms is None:
            timestamp_ms = TimeUtil.now_in_millis()
        ts_check = timestamp_ms < ASSET_CLASS_SELECTION_TIME_MS
        ts_ms = (time.perf_counter() - ts_start) * 1000

        if ts_check:
            bt.logging.info(f"[ASSET_TIMING] timestamp_check={ts_ms:.2f}ms, early_exit=True (pre-selection era)")
            return True

        # Time the IPC dict lookup
        ipc_start = time.perf_counter()
        selected_asset_class = self.asset_selections.get(miner_hotkey, None)
        ipc_ms = (time.perf_counter() - ipc_start) * 1000

        # Time the comparison
        compare_start = time.perf_counter()
        result = selected_asset_class == trade_pair_category if selected_asset_class is not None else False
        compare_ms = (time.perf_counter() - compare_start) * 1000

        total_ms = ts_ms + ipc_ms + compare_ms
        bt.logging.info(
            f"[ASSET_TIMING] timestamp_check={ts_ms:.2f}ms, ipc_lookup={ipc_ms:.2f}ms, "
            f"compare={compare_ms:.2f}ms, total={total_ms:.2f}ms, "
            f"selected={selected_asset_class}, requested={trade_pair_category}, result={result}"
        )

        return result

    def process_asset_selection_request(self, asset_selection: str, miner: str) -> Dict[str, str]:
        """
        Process an asset selection request from a miner.
        
        Args:
            asset_selection: The asset class the miner wants to select
            miner: The miner's hotkey
            
        Returns:
            Dict containing success status and message
        """
        try:
            # Validate asset class
            if not self.is_valid_asset_class(asset_selection):
                valid_classes = [category.value for category in TradePairCategory]
                return {
                    'successfully_processed': False,
                    'error_message': f'Invalid asset class. Valid options are: {", ".join(valid_classes)}'
                }

            # Check if miner has already selected an asset class
            if miner in self.asset_selections:
                current_selection = self.asset_selections.get(miner)
                return {
                    'successfully_processed': False,
                    'error_message': f'Asset class already selected: {current_selection.value}. Cannot change selection.'
                }

            # Convert string to TradePairCategory and set the asset selection
            asset_class = TradePairCategory(asset_selection.lower())
            self.asset_selections[miner] = asset_class
            self._save_asset_selections_to_disk()
            self._broadcast_asset_selection_to_validators(miner, asset_class)
            
            bt.logging.info(f"Miner {miner} selected asset class: {asset_selection}")
            
            return {
                'successfully_processed': True,
                'success_message': f'Miner {miner} successfully selected asset class: {asset_selection}'
            }
            
        except Exception as e:
            bt.logging.error(f"Error processing asset selection request for miner {miner}: {e}")
            return {
                'successfully_processed': False,
                'error_message': 'Internal server error processing asset selection request'
            }

    def _broadcast_asset_selection_to_validators(self, hotkey: str, asset_selection: str):
        """
        Broadcast AssetSelection synapse to other validators.
        Runs in a separate thread to avoid blocking the main process.
        """
        def run_broadcast():
            try:
                asyncio.run(self._async_broadcast_asset_selection(hotkey, asset_selection))
            except Exception as e:
                bt.logging.error(f"Failed to broadcast asset selection for {hotkey}: {e}")

        thread = threading.Thread(target=run_broadcast, daemon=True)
        thread.start()

    async def _async_broadcast_asset_selection(self, hotkey: str, asset_selection: str):
        """
        Asynchronously broadcast AssetSelection synapse to other validators.
        """
        try:
            # Get other validators to broadcast to
            if self.is_testnet:
                validator_axons = [n.axon_info for n in self.metagraph.get_neurons() if n.axon_info.ip != ValiConfig.AXON_NO_IP and n.axon_info.hotkey != self.wallet.hotkey.ss58_address]
            else:
                validator_axons = [n.axon_info for n in self.metagraph.get_neurons() if n.stake > bt.Balance(ValiConfig.STAKE_MIN) and n.axon_info.ip != ValiConfig.AXON_NO_IP and n.axon_info.hotkey != self.wallet.hotkey.ss58_address]

            if not validator_axons:
                bt.logging.debug("No other validators to broadcast CollateralRecord to")
                return

            # Create AssetSelection synapse with the data
            asset_selection_data = {
                "hotkey": hotkey,
                "asset_selection": asset_selection
            }

            asset_selection_synapse = template.protocol.AssetSelection(
                asset_selection=asset_selection_data
            )

            bt.logging.info(f"Broadcasting AssetSelection for {hotkey} to {len(validator_axons)} validators")

            # Send to other validators using dendrite
            async with bt.dendrite(wallet=self.wallet) as dendrite:
                responses = await dendrite.aquery(validator_axons, asset_selection_synapse)

                # Log results
                success_count = 0
                for response in responses:
                    if response.successfully_processed:
                        success_count += 1
                    elif response.error_message:
                        bt.logging.warning(f"Failed to send CollateralRecord to {response.axon.hotkey}: {response.error_message}")

                bt.logging.info(f"CollateralRecord broadcast completed: {success_count}/{len(responses)} validators updated")

        except Exception as e:
            bt.logging.error(f"Error in async broadcast collateral record: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

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
                # Convert the IPC dict to a regular dict
                selections_copy = dict(self.asset_selections)
            
            # Lock not needed here - working with local copy
            # Convert TradePairCategory objects to their string values
            return {
                hotkey: asset_class.value if hasattr(asset_class, 'value') else str(asset_class)
                for hotkey, asset_class in selections_copy.items()
            }
        except Exception as e:
            bt.logging.error(f"Error getting all miner selections: {e}")
            return {}

    def receive_asset_selection_update(self, asset_selection_data: dict) -> bool:
        """
        Process an incoming AssetSelection synapse and update miner asset selection.

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
                asset_selection = asset_selection_data.get("asset_selection")
                bt.logging.info(f"Processing asset selection for miner {hotkey}")

                if not all([hotkey, asset_selection is not None]):
                    bt.logging.warning(f"Invalid asset selection data received: {asset_selection_data}")
                    return False

                # Check if we already have this record (avoid duplicates)
                if hotkey in self.asset_selections:
                    bt.logging.debug(f"Asset selection for {hotkey} already exists")
                    return True

                # Add the new record
                self.asset_selections[hotkey] = asset_selection

                # Save to disk
                self._save_asset_selections_to_disk()

                bt.logging.info(f"Updated miner asset selection for {hotkey}: {asset_selection}")
                return True

        except Exception as e:
            bt.logging.error(f"Error processing collateral record update: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False
