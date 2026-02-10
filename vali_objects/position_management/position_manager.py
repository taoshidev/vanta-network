# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
import os
import traceback
from pickle import UnpicklingError

import bittensor as bt
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional

from time_util.time_util import TimeUtil, timeme
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePairCategory, ValiConfig, TradePair, RPCConnectionMode
from vali_objects.vali_dataclasses.order import Order
from vali_objects.enums.misc import OrderStatus
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.position_management.position_utils.position_splitter import PositionSplitter
from vali_objects.position_management.position_utils.position_filtering import PositionFiltering
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.position_management.position_utils.positions_to_snap import positions_to_snap
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient

TARGET_MS = 1769068800000 + (1000 * 60 * 60 * 6)  # + 6 hours


class PositionManager:
    """
    Core business logic for position management.

    This class manages position data in normal Python dicts (not IPC),
    providing efficient in-place mutations and selective disk writes.

    Data Structures:
    - hotkey_to_positions: Source of truth for all positions (open + closed)
    - hotkey_to_open_positions: Secondary index for O(1) lookups by trade_pair
    """

    def __init__(
        self,
        running_unit_tests: bool = False,
        is_backtesting: bool = False,
        load_from_disk: bool = None,
        split_positions_on_disk_load: bool = False,
        connection_mode = RPCConnectionMode.RPC
    ):
        """
        Initialize the PositionManager.

        Args:
            running_unit_tests: Whether running in unit test mode
            is_backtesting: Whether running in backtesting mode
            load_from_disk: Override disk loading behavior (None=auto, True=force load, False=skip)
            split_positions_on_disk_load: Whether to apply position splitting after loading from disk
            connection_mode: RPC or LOCAL mode for client connections
        """
        # SOURCE OF TRUTH: All positions (open + closed)
        # Structure: hotkey -> position_uuid -> Position
        # This enables O(1) lookups, inserts, updates, and deletes by position_uuid
        self.hotkey_to_positions: Dict[str, Dict[str, Position]] = {}

        # SECONDARY INDEX: Only open positions, indexed by trade_pair_id for O(1) lookups
        # Structure: hotkey -> trade_pair_id -> Position
        # Invariant: Must always be in sync with open positions in hotkey_to_positions
        # Benefits: O(1) lookup instead of O(N) scan for get_open_position_for_trade_pair
        self.hotkey_to_open_positions: Dict[str, Dict[str, Position]] = {}

        self.running_unit_tests = running_unit_tests
        self.is_backtesting = is_backtesting
        self.load_from_disk = load_from_disk
        self.split_positions_on_disk_load = split_positions_on_disk_load
        self.connection_mode = connection_mode

        # Statistics
        self.split_stats = defaultdict(self._default_split_stats)

        # RPC clients for internal communication
        # Import PerfLedgerClient here to avoid circular import (position_manager.py ← perf_ledger.py ← perf_ledger_server.py)
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient

        # Internal clients always use RPC mode to connect to their servers
        # The connection_mode parameter is for how OTHER components connect TO PositionManager
        self._elimination_client = EliminationClient(connection_mode=RPCConnectionMode.RPC)
        self._challenge_period_client = ChallengePeriodClient(connection_mode=RPCConnectionMode.RPC)
        self._perf_ledger_client = PerfLedgerClient(connection_mode=RPCConnectionMode.RPC)
        self._live_price_client = LivePriceFetcherClient(
            running_unit_tests=self.running_unit_tests,
            connection_mode=RPCConnectionMode.RPC
        )

        # Load positions from disk on startup
        self._load_positions_from_disk()

        # Apply position splitting if enabled (after loading)
        if self.split_positions_on_disk_load:
            self._apply_position_splitting_on_startup()

    def _default_split_stats(self):
        return {
            'n_positions_split': 0,
            'product_return_pre_split': 1.0,
            'product_return_post_split': 1.0
        }

    # ==================== Core Position Methods ====================

    def health_check(self) -> dict:
        """Health check endpoint."""
        total_positions = sum(len(positions_dict) for positions_dict in self.hotkey_to_positions.values())
        total_open = sum(len(d) for d in self.hotkey_to_open_positions.values())

        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "total_positions": total_positions,
            "total_open_positions": total_open,
            "num_hotkeys": len(self.hotkey_to_positions)
        }

    def get_positions_for_one_hotkey(
        self,
        hotkey: str,
        only_open_positions=False,
        acceptable_position_end_ms=None,
        sort_positions=False
    ):
        """
        Get positions for a specific hotkey.

        Args:
            hotkey: The miner's hotkey
            only_open_positions: Whether to return only open positions
            acceptable_position_end_ms: Minimum timestamp for positions (filters out older positions)
            sort_positions: Whether to sort positions by close_ms (closed first, then open)

        Returns:
            List of positions matching the filters
        """
        if hotkey not in self.hotkey_to_positions:
            return []

        positions_dict = self.hotkey_to_positions[hotkey]
        positions = list(positions_dict.values())  # Convert dict values to list

        # Filters
        if only_open_positions:
            positions = [p for p in positions if not p.is_closed_position]

        # Timestamp filtering
        if acceptable_position_end_ms is not None:
            positions = [p for p in positions if p.open_ms > acceptable_position_end_ms]

        # Sorting (closed positions first by close_ms, then open positions)
        if sort_positions:
            positions = sorted(positions, key=lambda p: (p.close_ms is None, p.close_ms or 0))

        return positions

    def _delete_position_from_memory(self, position: Position):
        hotkey = position.miner_hotkey
        position_uuid = position.position_uuid
        trade_pair_id = position.trade_pair.trade_pair_id
        if hotkey in self.hotkey_to_open_positions:
            existing_open = self.hotkey_to_open_positions[hotkey].get(trade_pair_id)
            # Only delete if it's a DIFFERENT position (same trade pair, different UUID)
            if existing_open and existing_open.position_uuid != position_uuid:
                # Delete from memory only (disk deletion handled by caller)
                if existing_open.position_uuid in self.hotkey_to_positions.get(hotkey, {}):
                    del self.hotkey_to_positions[hotkey][existing_open.position_uuid]
                self._remove_from_open_index(existing_open)
                bt.logging.info(
                    f"Deleted existing open position {existing_open.position_uuid} from memory for {hotkey}/{trade_pair_id}")


    def _save_miner_position_to_memory(self, position: Position, delete_open_position_if_exists: bool = True):
        """
        Save a single position efficiently with O(1) insert/update.
        Also maintains the open positions index for fast lookups.
        Note: Disk I/O is handled separately to maintain compatibility with existing format.

        Args:
            position: The position to save
            delete_open_position_if_exists: If True and position is closed, delete any existing
                open position for the same trade pair from memory (liquidation scenario)
        """
        hotkey = position.miner_hotkey
        position_uuid = position.position_uuid

        # Handle memory-side deletion of existing open position (liquidation scenario)
        if delete_open_position_if_exists and position.is_closed_position:
            trade_pair_id = position.trade_pair.trade_pair_id
            if hotkey in self.hotkey_to_open_positions:
                existing_open = self.hotkey_to_open_positions[hotkey].get(trade_pair_id)
                # Only delete if it's a DIFFERENT position (same trade pair, different UUID)
                if existing_open and existing_open.position_uuid != position_uuid:
                    # Delete from memory only (disk deletion handled by caller)
                    if existing_open.position_uuid in self.hotkey_to_positions.get(hotkey, {}):
                        del self.hotkey_to_positions[hotkey][existing_open.position_uuid]
                    self._remove_from_open_index(existing_open)
                    bt.logging.info(f"Deleted existing open position {existing_open.position_uuid} from memory for {hotkey}/{trade_pair_id}")

        if hotkey not in self.hotkey_to_positions:
            self.hotkey_to_positions[hotkey] = {}

        # Check if this position already exists (update vs insert)
        existing_position = self.hotkey_to_positions[hotkey].get(position_uuid)

        # Validate trade pair consistency for updates
        if existing_position:
            assert existing_position.trade_pair == position.trade_pair, \
                f"Trade pair mismatch for position {position_uuid}. Existing: {existing_position.trade_pair}, New: {position.trade_pair}"

        # Update the main data structure (source of truth)
        self.hotkey_to_positions[hotkey][position_uuid] = position

        # Maintain the open positions index
        if existing_position:
            # Position is being updated - handle state transitions
            was_open = existing_position.is_open_position
            is_now_open = not position.is_closed_position

            if was_open and not is_now_open:
                # Open -> Closed transition: remove from index
                self._remove_from_open_index(position)
            elif is_now_open and not was_open:
                # Closed -> Open transition: add to index (rare but possible)
                self._add_to_open_index(position)
            elif is_now_open:
                # Still open: update the index reference
                self._add_to_open_index(position)
        else:
            # New position being inserted
            if not position.is_closed_position:
                self._add_to_open_index(position)

        bt.logging.trace(f"Saved position {position_uuid} for {hotkey}")

    def delete_open_position_if_exists(self, position: Position) -> None:
        # See if we need to delete the open position file
        open_position = self.get_open_position_for_trade_pair(position.miner_hotkey,
                                                               position.trade_pair.trade_pair_id)
        if open_position:
            self.delete_position(open_position.miner_hotkey, open_position.position_uuid)

    def _read_positions_from_disk_for_tests(self, miner_hotkey: str, only_open_positions: bool = False) -> List[Position]:
        """
        Test helper method to read positions directly from disk, bypassing the RPC server.

        ⚠️ WARNING: This method is ONLY for tests that need to verify disk persistence.
        Production code should NEVER call this method - always use get_positions_for_one_hotkey() instead.

        The RPC server architecture dictates that only the server should read from disk normally.
        This helper exists solely to allow tests to verify that the server is correctly
        persisting data to disk.

        Args:
            miner_hotkey: The hotkey to read positions for
            only_open_positions: Whether to filter to only open positions

        Returns:
            List of positions loaded directly from disk files
        """
        miner_dir = ValiBkpUtils.get_miner_all_positions_dir(
            miner_hotkey,
            running_unit_tests=self.running_unit_tests
        )
        all_files = ValiBkpUtils.get_all_files_in_dir(miner_dir)
        positions = [self._get_position_from_disk(file) for file in all_files]

        if only_open_positions:
            positions = [position for position in positions if position.is_open_position]

        return positions

    def _get_position_from_disk(self, file) -> Position:
        # wrapping here to allow simpler error handling & original for other error handling
        # Note one position always corresponds to one file.
        file_string = None
        try:
            file_string = ValiBkpUtils.get_file(file)
            ans = Position.model_validate_json(file_string)
            if not ans.orders:
                bt.logging.warning(f"Anomalous position has no orders: {ans.to_dict()}")
            return ans
        except FileNotFoundError:
            raise ValiFileMissingException(f"Vali position file is missing {file}")
        except UnpicklingError as e:
            raise ValiBkpCorruptDataException(f"file_string is {file_string}, {e}")
        except UnicodeDecodeError as e:
            raise ValiBkpCorruptDataException(
                f" Error {e} for file {file} You may be running an old version of the software. Confirm with the team if you should delete your cache. file string {file_string[:2000] if file_string else None}")
        except Exception as e:
            raise ValiBkpCorruptDataException(f"Error {e} file_path {file} file_string: {file_string}")


    def verify_open_position_write(self, miner_dir, updated_position):
        # Get open position from memory for this hotkey and trade_pair
        open_position = self.get_open_position_for_trade_pair(
            updated_position.miner_hotkey,
            updated_position.trade_pair.trade_pair_id
        )

        # If no open position exists, this is the first time it's being saved
        if open_position is None:
            return

        # If an open position exists, verify it has the same position_uuid
        if open_position.position_uuid != updated_position.position_uuid:
            msg = (
                f"Attempted to write open position {updated_position.position_uuid} for miner {updated_position.miner_hotkey} "
                f"and trade_pair {updated_position.trade_pair.trade_pair_id} but found an existing open"
                f" position with a different position_uuid {open_position.position_uuid}.")
            raise ValiRecordsMisalignmentException(msg)



    def get_positions_for_hotkeys(
        self,
        hotkeys: List[str],
        only_open_positions=False,
        filter_eliminations: bool = False,
        acceptable_position_end_ms: int = None,
        sort_positions: bool = False
    ) -> Dict[str, List[Position]]:
        """
        Get positions for multiple hotkeys (bulk operation).
        This is much more efficient than calling get_positions_for_one_hotkey multiple times.

        Server-side filtering reduces RPC payload and client processing.

        Args:
            hotkeys: List of hotkeys to fetch positions for
            only_open_positions: Whether to return only open positions
            filter_eliminations: If True, fetch eliminations internally and filter them out
            acceptable_position_end_ms: Minimum timestamp for positions
            sort_positions: If True, sort positions by close_ms (closed first, then open)

        Returns:
            Dict mapping hotkey to list of positions
        """
        # Elimination filtering (fetch eliminations internally if requested)
        if filter_eliminations and self._elimination_client:
            # Fetch eliminations via EliminationClient
            eliminations_list = self._elimination_client.get_eliminations_from_memory()
            eliminated_hotkeys = set(x['hotkey'] for x in eliminations_list) if eliminations_list else set()
            # Filter out eliminated hotkeys
            hotkeys = [hk for hk in hotkeys if hk not in eliminated_hotkeys]

        result = {}
        for hotkey in hotkeys:
            if hotkey not in self.hotkey_to_positions:
                result[hotkey] = []
                continue

            positions_dict = self.hotkey_to_positions[hotkey]
            positions = list(positions_dict.values())  # Convert dict values to list

            # Filters
            if only_open_positions:
                positions = [p for p in positions if not p.is_closed_position]

            # Timestamp filtering
            if acceptable_position_end_ms is not None:
                positions = [p for p in positions if p.open_ms > acceptable_position_end_ms]

            # Sorting (closed positions first by close_ms, then open positions)
            if sort_positions:
                positions = sorted(positions, key=lambda p: p.close_ms if p.is_closed_position else float("inf"))

            result[hotkey] = positions

        return result

    def clear_all_miner_positions(self):
        """Clear all positions (for testing). Also clears the open positions index and split statistics."""
        self.hotkey_to_positions.clear()
        self.hotkey_to_open_positions.clear()
        self.split_stats.clear()
        bt.logging.info("Cleared all positions, open index, and split statistics")

    def clear_all_miner_positions_and_disk(self, hotkey=None):
        if not self.running_unit_tests:
            raise Exception("Only available in unit tests")
        if hotkey is None:
            """Clear all positions from memory AND disk (for testing)."""
            # Clear memory first
            self.clear_all_miner_positions()
            # Clear disk directories
            ValiBkpUtils.clear_all_miner_directories(running_unit_tests=self.running_unit_tests)
            bt.logging.info("Cleared all positions from memory and disk")
        else:
            if hotkey in self.hotkey_to_positions:
                del self.hotkey_to_positions[hotkey]
            if hotkey in self.hotkey_to_open_positions:
                del self.hotkey_to_open_positions[hotkey]
            for p in self.get_positions_for_one_hotkey(hotkey):
                self.delete_position(p.miner_hotkey, p.position_uuid)

    def delete_position(self, hotkey: str, position_uuid: str):
        """
        Delete a specific position with O(1) deletion.
        Also removes from open positions index if it was open.
        Handles Disk deletion too.
        Lock should be aquired by caller
        """
        positions_dict = self.hotkey_to_positions.get(hotkey, {})
        # O(1) direct deletion from dict
        if position_uuid in positions_dict:
            position = positions_dict[position_uuid]
            # Remove from open index if it's an open position
            if position.is_open_position:
                self._remove_from_open_index(position)

            del positions_dict[position_uuid]
            if not self.is_backtesting:
                self._delete_position_from_disk(position)
            bt.logging.info(f"Deleted position {position_uuid} for {hotkey}")
            return True

        return False

    def get_position(self, hotkey: str, position_uuid: str):
        """Get a specific position by UUID with O(1) lookup."""
        if hotkey not in self.hotkey_to_positions:
            return None

        positions_dict = self.hotkey_to_positions[hotkey]

        # O(1) direct dict access
        return positions_dict.get(position_uuid, None)

    @staticmethod
    def sort_by_close_ms(_position):
        """
        Sort key function for positions.
        Closed positions are sorted by close_ms (ascending).
        Open positions are sorted to the end (infinity).

        This is the canonical sorting method used throughout the codebase.
        """
        return (
            _position.close_ms if _position.is_closed_position else float("inf")
        )

    def get_open_position_for_trade_pair(self, hotkey: str, trade_pair_id: str) -> Optional[Position]:
        """
        Get the open position for a specific hotkey and trade pair.
        Uses O(1) index lookup instead of scanning - extremely fast!

        Args:
            hotkey: The miner's hotkey
            trade_pair_id: The trade pair ID to filter by

        Returns:
            The open position if found, None otherwise
        """
        # O(1) lookup using the secondary index!
        # This is MUCH faster than scanning through all positions
        if hotkey not in self.hotkey_to_open_positions:
            return None

        return self.hotkey_to_open_positions[hotkey].get(trade_pair_id, None)

    def compute_realtime_drawdown(self, hotkey: str) -> float:
        """
        Compute the realtime drawdown from positions.
        Bypasses perf ledger, since perf ledgers are refreshed in 5 min intervals and may be out of date.
        Used to enable realtime withdrawals based on drawdown.

        Returns proportion of portfolio value as drawdown. 1.0 -> 0% drawdown, 0.9 -> 10% drawdown
        """
        # 1. Get existing perf ledger to access historical max portfolio value
        existing_bundle = self._perf_ledger_client.get_perf_ledgers(
            portfolio_only=True,
            from_disk=False
        )
        portfolio_ledger = existing_bundle.get(hotkey)

        if not portfolio_ledger or not portfolio_ledger.cps:
            bt.logging.warning(f"No perf ledger found for {hotkey}")
            return 1.0

        # 2. Get historical max portfolio value from existing checkpoints
        portfolio_ledger.init_max_portfolio_value()  # Ensures max_return is set
        max_portfolio_value = portfolio_ledger.max_return

        # 3. Calculate current portfolio value with live prices
        current_portfolio_value = self._calculate_current_portfolio_value(hotkey)

        # 4. Calculate current drawdown
        if max_portfolio_value <= 0:
            return 1.0

        drawdown = min(1.0, current_portfolio_value / max_portfolio_value)

        print(f"Real-time drawdown for {hotkey}: "
              f"{(1 - drawdown) * 100:.2f}% "
              f"(current: {current_portfolio_value:.4f}, "
              f"max: {max_portfolio_value:.4f})")

        return drawdown

    def _calculate_current_portfolio_value(self, miner_hotkey: str) -> float:
        """
        Calculate current portfolio value with live prices.
        """
        positions = self.get_positions_for_one_hotkey(
            miner_hotkey,
            only_open_positions=False
        )

        if not positions:
            return 1.0  # No positions = starting value

        portfolio_return = 1.0
        now_ms = TimeUtil.now_in_millis()

        for position in positions:
            if position.is_open_position:
                # Get live price for open positions
                price_sources = self._live_price_client.get_sorted_price_sources_for_trade_pair(
                    position.trade_pair,
                    now_ms
                )

                if price_sources and price_sources[0]:
                    realtime_price = price_sources[0].close
                    # Calculate return with fees at this moment
                    position_return = position.get_open_position_return_with_fees(
                        realtime_price,
                        now_ms
                    )
                    portfolio_return *= position_return
                else:
                    # Fallback to last known return
                    portfolio_return *= position.return_at_close
            else:
                # Use stored return for closed positions
                portfolio_return *= position.return_at_close

        return portfolio_return

    def get_all_hotkeys(self):
        """Get all hotkeys that have positions."""
        return list(self.hotkey_to_positions.keys())

    def get_extreme_position_order_processed_on_disk_ms(self) -> tuple:
        """
        Get the minimum and maximum processed_ms timestamps across all orders in all positions.

        Returns:
            tuple: (min_time, max_time) in milliseconds
        """
        min_time = float("inf")
        max_time = 0

        for hotkey in self.hotkey_to_positions.keys():
            positions = list(self.hotkey_to_positions[hotkey].values())
            for p in positions:
                for o in p.orders:
                    min_time = min(min_time, o.processed_ms)
                    max_time = max(max_time, o.processed_ms)

        return min_time, max_time

    def calculate_net_portfolio_leverage(self, hotkey: str) -> float:
        """
        Calculate total leverage across all open positions for a hotkey.
        Since miners only trade one asset class at a time, this returns the raw leverage sum.

        Args:
            hotkey: The miner hotkey

        Returns:
            Total portfolio leverage (sum of abs(leverage) for each open position)
        """
        # Use O(1) open positions index for fast lookup
        if hotkey not in self.hotkey_to_open_positions:
            return 0.0

        portfolio_leverage = 0.0
        for position in self.hotkey_to_open_positions[hotkey].values():
            portfolio_leverage += abs(position.get_net_leverage())

        return portfolio_leverage

    def filtered_positions_for_scoring(
        self,
        hotkeys: List[str] = None,
        include_development_positions: bool = False
    ) -> tuple:
        """
        Filter the positions for a set of hotkeys for scoring purposes.
        Excludes development positions by default.

        Args:
            hotkeys: Optional list of hotkeys to filter. If None, uses all hotkeys with positions.
            include_development_positions: If True, include development hotkey positions.

        Returns:
            Tuple of (filtered_positions dict, hk_to_first_order_time dict)
        """
        if hotkeys is None:
            # Get all hotkeys that have positions
            hotkeys = list(self.hotkey_to_positions.keys())
            if not include_development_positions:
                hotkeys = [hk for hk in hotkeys if hk != ValiConfig.DEVELOPMENT_HOTKEY]
        else:
            # Hotkeys were provided explicitly - filter them if needed
            if not include_development_positions:
                hotkeys = [hk for hk in hotkeys if hk != ValiConfig.DEVELOPMENT_HOTKEY]

        hk_to_first_order_time = {}
        filtered_positions = {}

        for hotkey in hotkeys:
            if hotkey not in self.hotkey_to_positions:
                continue

            # Get positions and sort by close_ms
            positions_dict = self.hotkey_to_positions[hotkey]
            miner_positions = sorted(
                positions_dict.values(),
                key=lambda p: p.close_ms if p.is_closed_position else float("inf")
            )

            if miner_positions:
                hk_to_first_order_time[hotkey] = min([p.orders[0].processed_ms for p in miner_positions])
                filtered_positions[hotkey] = PositionFiltering.filter_positions_for_duration(miner_positions)

        return filtered_positions, hk_to_first_order_time

    def close_open_orders_for_suspended_trade_pairs(self, live_price_fetcher=None) -> int:
        """
        Close all open positions for suspended trade pairs (SPX, DJI, NDX, VIX).

        Args:
            live_price_fetcher: Optional price fetcher to use. If None, uses internal client.
                               Pass a mock price fetcher for testing.

        Returns:
            Number of positions closed
        """
        tps_to_eliminate = [TradePair.SPX, TradePair.DJI, TradePair.NDX, TradePair.VIX]
        if not tps_to_eliminate:
            return 0

        # Use provided price fetcher or internal client
        price_fetcher = live_price_fetcher or self._live_price_client
        if not price_fetcher:
            bt.logging.warning("No price fetcher available for close_open_orders_for_suspended_trade_pairs")
            return 0

        # Get all positions
        all_positions = self.get_positions_for_all_miners(sort_positions=True)

        # Get eliminations
        eliminations = []
        if self._elimination_client:
            eliminations = self._elimination_client.get_eliminations_from_memory() or []
        eliminated_hotkeys = set(x['hotkey'] for x in eliminations)
        bt.logging.info(f"Found {len(eliminations)} eliminations on disk.")

        n_positions_closed = 0
        for hotkey, positions in all_positions.items():
            if hotkey in eliminated_hotkeys:
                continue
            # Closing all open positions for the specified trade pair
            for position in positions:
                if position.is_closed_position:
                    continue
                if position.trade_pair in tps_to_eliminate:
                    price_sources = price_fetcher.get_sorted_price_sources_for_trade_pair(
                        position.trade_pair, TARGET_MS
                    )
                    if not price_sources:
                        bt.logging.warning(
                            f"No price sources for {position.trade_pair.trade_pair_id}, skipping"
                        )
                        continue

                    live_price = price_sources[0].parse_appropriate_price(
                        TARGET_MS, position.trade_pair.is_forex, OrderType.FLAT, position
                    )
                    flat_order = Order(
                        price=live_price,
                        price_sources=price_sources,
                        processed_ms=TARGET_MS,
                        order_uuid=position.position_uuid[::-1],
                        trade_pair=position.trade_pair,
                        order_type=OrderType.FLAT,
                        leverage=0,
                        src=OrderSource.DEPRECATION_FLAT
                    )

                    position.add_order(flat_order, price_fetcher)
                    self.save_miner_position(position, delete_open_position_if_exists=True, validate=False)
                    n_positions_closed += 1
                    bt.logging.info(
                        f"Closed deprecated trade pair position {position.position_uuid} "
                        f"for {hotkey} ({position.trade_pair.trade_pair_id})"
                    )

        return n_positions_closed

    def close_all_positions(
        self,
        hotkey: str,
        close_time_ms: int,
        order_source: OrderSource,
        live_price_fetcher=None
    ) -> int:
        """
        Close all open positions for a specific hotkey.

        Args:
            hotkey: Hotkey whose positions should be closed
            close_time_ms: Timestamp for closing positions
            order_source: OrderSource enum value (e.g., SUBACCOUNT_PROMOTION, ELIMINATION_FLAT)
            live_price_fetcher: Optional price fetcher for accurate closing prices

        Returns:
            int: Number of positions successfully closed
        """
        # Get all positions for this hotkey
        open_positions = self.get_positions_for_one_hotkey(hotkey, only_open_positions=True, sort_positions=True)

        if not open_positions:
            bt.logging.info(f"No open positions to close for {hotkey}")
            return 0

        # Use provided price fetcher or internal client
        price_fetcher = live_price_fetcher or self._live_price_client
        if not price_fetcher:
            bt.logging.warning(f"No price fetcher available for closing positions for {hotkey}")
            return 0

        bt.logging.info(
            f"Closing {len(open_positions)} positions for {hotkey} "
            f"with order source {order_source.name}"
        )

        closed_count = 0
        for position in open_positions:
            try:
                # Generate FLAT order using existing pattern
                flat_order = Position.generate_fake_flat_order(
                    position=position,
                    elimination_time_ms=close_time_ms,
                    price_fetcher_client=price_fetcher,
                    src=order_source
                )

                # Add order to position (automatically marks as closed)
                position.add_order(flat_order, price_fetcher)

                # Save updated position
                self.save_miner_position(position)

                bt.logging.info(f"Closed open {position.trade_pair.trade_pair_id} position {position.position_uuid} for {hotkey}")
                closed_count += 1

            except Exception as e:
                bt.logging.error(
                    f"Failed to close position {position.position_uuid} for {hotkey}: {e}"
                )
                bt.logging.error(traceback.format_exc())

        bt.logging.info(
            f"Closed {closed_count}/{len(open_positions)} positions for {hotkey}"
        )
        return closed_count

    # ==================== Pre-run Setup Methods ====================

    @timeme
    def pre_run_setup(self, perform_order_corrections: bool = True) -> None:
        """
        Run pre-run setup operations.
        This is called once at validator startup.

        Handles perf ledger wiping internally via PerfLedgerClient.

        Args:
            perform_order_corrections: Whether to run order corrections
        """
        miners_to_wipe_perf_ledger = []

        if perform_order_corrections:
            try:
                miners_to_wipe_perf_ledger = self._apply_order_corrections()
            except Exception as e:
                bt.logging.error(f"Error applying order corrections: {e}")
                traceback.print_exc()

        # Wipe perf ledgers internally using PerfLedgerClient
        if miners_to_wipe_perf_ledger and self._perf_ledger_client:
            try:
                self._perf_ledger_client.wipe_miners_perf_ledgers(miners_to_wipe_perf_ledger)
                bt.logging.info(f"Wiped perf ledgers for {len(miners_to_wipe_perf_ledger)} miners")
            except Exception as e:
                bt.logging.error(f"Error wiping perf ledgers: {e}")
                traceback.print_exc()

    @timeme
    def _apply_order_corrections(self) -> List[str]:
        """
        Apply order corrections to positions.
        This is our mechanism for manually synchronizing validator orders in situations
        where a bug prevented an order from filling.

        Returns:
            List of miner hotkeys that need their perf ledgers wiped
        """
        now_ms = TimeUtil.now_in_millis()
        if now_ms > TARGET_MS:
            return []

        # Get all positions sorted
        hotkey_to_positions = self.get_positions_for_all_miners(sort_positions=True)

        n_corrections = 0
        n_attempts = 0
        unique_corrections = set()

        # Wipe miners only once when dynamic challenge period launches
        miners_to_wipe = []
        miners_to_promote = []
        position_uuids_to_delete = []
        wipe_positions = False
        reopen_force_closed_orders = False
        miners_to_wipe_perf_ledger = []

        current_eliminations = self._elimination_client.get_eliminations_from_memory() if self._elimination_client else []

        if now_ms < TARGET_MS:
            # # temp slippage correction
            # SLIPPAGE_V2_TIME_MS = 1759431540000
            # n_slippage_corrections = 0
            # for hotkey, positions in hotkey_to_positions.items():
            #     for position in positions:
            #         needs_save = False
            #         for order in position.orders:
            #             if (order.trade_pair.is_forex and SLIPPAGE_V2_TIME_MS < order.processed_ms):
            #                 old_slippage = order.slippage
            #                 order.slippage = PriceSlippageModel.calculate_slippage(order.bid, order.ask, order)
            #                 if old_slippage != order.slippage:
            #                     needs_save = True
            #                     n_slippage_corrections += 1
            #                     bt.logging.info(
            #                         f"Updated forex slippage for order {order}: "
            #                         f"{old_slippage:.6f} -> {order.slippage:.6f}")
            #
            #         if needs_save:
            #             position.rebuild_position_with_updated_orders(self._live_price_client)
            #             self.save_miner_position(position, validate=False)
            # bt.logging.info(f"Applied {n_slippage_corrections} forex slippage corrections")

            # All miners that wanted their challenge period restarted
            miners_to_wipe = []
            position_uuids_to_delete = []
            miners_to_promote = []

            for p in positions_to_snap:
                try:
                    pos = Position(**p)
                    hotkey = pos.miner_hotkey
                    # if this hotkey is eliminated, log an error and continue
                    if any(e['hotkey'] == hotkey for e in current_eliminations):
                        bt.logging.error(f"Hotkey {hotkey} is eliminated. Skipping position {pos}.")
                        continue
                    if pos.is_open_position:
                        self.delete_open_position_if_exists(pos)
                    self.save_miner_position(pos, validate=False)
                    print(f"Added position {pos.position_uuid} for trade pair {pos.trade_pair.trade_pair_id} for hk {pos.miner_hotkey}")
                except Exception as e:
                    print(f"Error adding position {p} {e}")

        # Don't accidentally promote eliminated miners
        for e in current_eliminations:
            if e['hotkey'] in miners_to_promote:
                miners_to_promote.remove(e['hotkey'])

        # Promote miners that would have passed challenge period
        if self._challenge_period_client:
            for miner in miners_to_promote:
                if self._challenge_period_client.has_miner(miner):
                    if self._challenge_period_client.get_miner_bucket(miner) != MinerBucket.MAINCOMP:
                        self._challenge_period_client.promote_challengeperiod_in_memory([miner], now_ms)
            self._challenge_period_client._write_challengeperiod_from_memory_to_disk()

        # Wipe miners_to_wipe below
        for k in miners_to_wipe:
            if k not in hotkey_to_positions:
                hotkey_to_positions[k] = []

        n_eliminations_before = len(current_eliminations)
        if self._elimination_client:
            for e in current_eliminations:
                if e['hotkey'] in miners_to_wipe:
                    self._elimination_client.delete_eliminations([e['hotkey']])
                    print(f"Removed elimination for hotkey {e['hotkey']}")
        n_eliminations_after = len(self._elimination_client.get_eliminations_from_memory()) if self._elimination_client else 0
        print(f'    n_eliminations_before {n_eliminations_before} n_eliminations_after {n_eliminations_after}')

        update_perf_ledgers = False
        for miner_hotkey, positions in hotkey_to_positions.items():
            n_attempts += 1
            self.dedupe_positions(positions, miner_hotkey)
            if miner_hotkey in miners_to_wipe:
                update_perf_ledgers = True
                miners_to_wipe_perf_ledger.append(miner_hotkey)
                bt.logging.info(f"Resetting hotkey {miner_hotkey}")
                n_corrections += 1
                unique_corrections.update([p.position_uuid for p in positions])
                for pos in positions:
                    if wipe_positions:
                        self.delete_position(pos.miner_hotkey, pos.position_uuid)
                    elif pos.position_uuid in position_uuids_to_delete:
                        print(f'Deleting position {pos.position_uuid} for trade pair {pos.trade_pair.trade_pair_id} for hk {pos.miner_hotkey}')
                        self.delete_position(pos.miner_hotkey, pos.position_uuid)
                    elif reopen_force_closed_orders:
                        if any((o.src in (1, 3)) for o in pos.orders):
                            pos.orders = [o for o in pos.orders if (o.src not in (1, 3))]
                            pos.rebuild_position_with_updated_orders(self._live_price_client)
                            self.save_miner_position(pos, validate=False)
                            print(f'Removed eliminated orders from position {pos}')

                if self._challenge_period_client and self._challenge_period_client.has_miner(miner_hotkey):
                    self._challenge_period_client.remove_miner(miner_hotkey)
                    print(f'Removed challengeperiod status for {miner_hotkey}')

                if self._challenge_period_client:
                    self._challenge_period_client._write_challengeperiod_from_memory_to_disk()

        bt.logging.warning(
            f"Applied {n_corrections} order corrections out of {n_attempts} attempts. unique positions corrected: {len(unique_corrections)}")

        return miners_to_wipe_perf_ledger

    def get_positions_for_all_miners(self, sort_positions: bool = False) -> Dict[str, List[Position]]:
        """
        Get all positions for all miners.

        Args:
            sort_positions: If True, sort positions by close_ms (closed first, then open)

        Returns:
            Dict mapping hotkey to list of positions
        """
        result = {}
        for hotkey, positions_dict in self.hotkey_to_positions.items():
            positions = list(positions_dict.values())
            if sort_positions:
                positions = sorted(positions, key=lambda p: p.close_ms if p.is_closed_position else float("inf"))
            result[hotkey] = positions
        return result

    def save_miner_position(self, position: Position, delete_open_position_if_exists=True, validate=True) -> None:
        """
        Save a position with full memory and disk cleanup.

        Args:
            position: The position to save
            delete_open_position_if_exists: If True and position is closed, delete any existing open position for the same trade pair
            validate: If True, perform validation checks (expensive disk reads). Should be True for external calls, False for internal operations.
        """
        # 1. Handle deletion of existing open position if needed
        if position.is_closed_position and delete_open_position_if_exists:
            open_pos = self.get_open_position_for_trade_pair(position.miner_hotkey, position.trade_pair.trade_pair_id)
            if open_pos and open_pos.position_uuid == position.position_uuid:
                self.delete_position(open_pos.miner_hotkey, open_pos.position_uuid)

        # 2. Validate if needed (only for open positions)
        if position.is_open_position and validate and not self.is_backtesting:
            miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                position.miner_hotkey,
                position.trade_pair.trade_pair_id,
                order_status=OrderStatus.OPEN,
                running_unit_tests=self.running_unit_tests
            )
            self.verify_open_position_write(miner_dir, position)

        # 3. Save to memory (don't delete again since we already did it in step 1)
        self._save_miner_position_to_memory(position, delete_open_position_if_exists=False)

        # 4. Save to disk
        if not self.is_backtesting:
            self._write_position_to_disk(position)


    def _delete_position_from_disk(self, position: Position) -> None:
        """Delete a position file from disk. Lock should be aquired by caller"""
        try:
            # Try both open and closed directories
            miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                position.miner_hotkey,
                position.trade_pair.trade_pair_id,
                order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                running_unit_tests=self.running_unit_tests
            )
            file_path = miner_dir + position.position_uuid
            if os.path.exists(file_path):
                os.remove(file_path)
                bt.logging.info(f"Deleted position from disk: {file_path}")
        except Exception as e:
            bt.logging.error(f"Error deleting position {position.position_uuid} from disk: {e}")

    def dedupe_positions(self, positions: List[Position], miner_hotkey: str) -> None:
        """Internal method to deduplicate positions for a miner."""
        positions_by_trade_pair = defaultdict(list)
        n_positions_deleted = 0
        n_orders_deleted = 0
        n_positions_rebuilt_with_new_orders = 0

        for position in positions:
            positions_by_trade_pair[position.trade_pair].append(deepcopy(position))

        for trade_pair, tp_positions in positions_by_trade_pair.items():
            position_uuid_to_dedupe = {}
            for p in tp_positions:
                if p.position_uuid in position_uuid_to_dedupe:
                    # Replace if it has more orders
                    if len(p.orders) > len(position_uuid_to_dedupe[p.position_uuid].orders):
                        old_position = position_uuid_to_dedupe[p.position_uuid]
                        self.delete_position(old_position.miner_hotkey, old_position.position_uuid)
                        position_uuid_to_dedupe[p.position_uuid] = p
                        n_positions_deleted += 1
                    else:
                        self.delete_position(p.miner_hotkey, p.position_uuid)
                        n_positions_deleted += 1
                else:
                    position_uuid_to_dedupe[p.position_uuid] = p

            for position in position_uuid_to_dedupe.values():
                order_uuid_to_dedup = {}
                new_orders = []
                any_orders_deleted = False
                for order in position.orders:
                    if order.order_uuid in order_uuid_to_dedup:
                        n_orders_deleted += 1
                        any_orders_deleted = True
                    else:
                        new_orders.append(order)
                        order_uuid_to_dedup[order.order_uuid] = order
                if any_orders_deleted:
                    position.orders = new_orders
                    position.rebuild_position_with_updated_orders(self._live_price_client)
                    self.save_miner_position(position, delete_open_position_if_exists=False, validate=False)
                    n_positions_rebuilt_with_new_orders += 1

        if n_positions_deleted or n_orders_deleted or n_positions_rebuilt_with_new_orders:
            bt.logging.warning(
                f"Hotkey {miner_hotkey}: Deleted {n_positions_deleted} duplicate positions and {n_orders_deleted} "
                f"duplicate orders across {n_positions_rebuilt_with_new_orders} positions.")

    # ==================== Compaction Methods ====================

    @staticmethod
    def strip_old_price_sources(position: Position, time_now_ms: int) -> int:
        """Strip price_sources from orders older than 1 week to save disk space."""
        n_removed = 0
        one_week_ago_ms = time_now_ms - 1000 * 60 * 60 * 24 * 7
        for o in position.orders:
            if o.processed_ms < one_week_ago_ms:
                if o.price_sources:
                    o.price_sources = []
                    n_removed += 1
        return n_removed

    @timeme
    def compact_price_sources(self):
        """
        Compact price_sources by removing old price data from closed positions.
        Runs directly on in-memory positions - no RPC overhead!
        """
        time_now = TimeUtil.now_in_millis()
        cutoff_time_ms = time_now - 10 * ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS  # Generous bound
        n_price_sources_removed = 0

        # Direct access to in-memory positions
        for hotkey, positions_dict in self.hotkey_to_positions.items():
            for position in positions_dict.values():
                if position.is_open_position:
                    continue  # Don't modify open positions as we don't want to deal with locking
                elif any(o.processed_ms > cutoff_time_ms for o in position.orders):
                    continue  # Could be subject to retro price correction and we don't want to deal with locking

                n = self.strip_old_price_sources(position, time_now)
                if n:
                    n_price_sources_removed += n
                    # Save to disk
                    self._write_position_to_disk(position)

        bt.logging.info(f'Removed {n_price_sources_removed} price sources from old data.')

    # ==================== Index Management ====================

    def _validate_no_duplicate_open_position(self, position: Position):
        """
        Validate that no other open position exists for the same trade pair.
        Call this BEFORE saving to main dict to ensure atomic validation.

        Raises:
            ValiRecordsMisalignmentException: If another open position already exists for this trade pair
        """
        hotkey = position.miner_hotkey
        trade_pair_id = position.trade_pair.trade_pair_id

        if hotkey not in self.hotkey_to_open_positions:
            return  # No open positions for this hotkey, safe to proceed

        if trade_pair_id in self.hotkey_to_open_positions[hotkey]:
            existing_pos = self.hotkey_to_open_positions[hotkey][trade_pair_id]
            if existing_pos.position_uuid != position.position_uuid:
                error_msg = (
                    f"Data corruption: Multiple open positions for miner {hotkey} and trade_pair {trade_pair_id}. "
                    f"Existing position UUID: {existing_pos.position_uuid}, "
                    f"New position UUID: {position.position_uuid}. "
                    f"Please restore cache."
                )
                bt.logging.error(error_msg)
                raise ValiRecordsMisalignmentException(error_msg)

    def _add_to_open_index(self, position: Position):
        """
        Add an open position to the secondary index for O(1) lookups.
        Only call this for positions that are definitely open.

        Note: Duplicate validation is now done in _validate_no_duplicate_open_position()
        which is called before saving to main dict. This method assumes validation passed.
        """
        hotkey = position.miner_hotkey
        trade_pair_id = position.trade_pair.trade_pair_id

        if hotkey not in self.hotkey_to_open_positions:
            self.hotkey_to_open_positions[hotkey] = {}

        self.hotkey_to_open_positions[hotkey][trade_pair_id] = position
        bt.logging.trace(f"Added to open index: {hotkey}/{trade_pair_id}")

    def _remove_from_open_index(self, position: Position):
        """
        Remove a position from the open positions index.
        Safe to call even if position isn't in the index.
        """
        hotkey = position.miner_hotkey
        trade_pair_id = position.trade_pair.trade_pair_id

        if hotkey not in self.hotkey_to_open_positions:
            return

        if trade_pair_id in self.hotkey_to_open_positions[hotkey]:
            # Only remove if it's the same position (by UUID)
            if self.hotkey_to_open_positions[hotkey][trade_pair_id].position_uuid == position.position_uuid:
                del self.hotkey_to_open_positions[hotkey][trade_pair_id]
                bt.logging.trace(f"Removed from open index: {hotkey}/{trade_pair_id}")

                # Cleanup empty dicts
                if not self.hotkey_to_open_positions[hotkey]:
                    del self.hotkey_to_open_positions[hotkey]

    def _rebuild_open_index(self):
        """
        Rebuild the entire open positions index from scratch.
        Used after bulk operations like loading from disk or position splitting.
        Detects and logs duplicate open positions for the same miner/trade_pair.
        """
        self.hotkey_to_open_positions.clear()

        for hotkey, positions_dict in self.hotkey_to_positions.items():
            for position in positions_dict.values():
                if not position.is_closed_position:
                    trade_pair_id = position.trade_pair.trade_pair_id
                    # Check for duplicate open positions
                    if hotkey in self.hotkey_to_open_positions and trade_pair_id in self.hotkey_to_open_positions[hotkey]:
                        existing_position = self.hotkey_to_open_positions[hotkey][trade_pair_id]
                        bt.logging.error(
                            f"Found duplicate open positions for miner {hotkey} and trade_pair {trade_pair_id}. "
                            f"Existing position UUID: {existing_position.position_uuid}, "
                            f"New position UUID: {position.position_uuid}. "
                            f"This indicates data corruption - please investigate."
                        )
                    self._add_to_open_index(position)

        total_open = sum(len(d) for d in self.hotkey_to_open_positions.values())
        bt.logging.debug(f"Rebuilt open index: {total_open} open positions across {len(self.hotkey_to_open_positions)} hotkeys")

    # ==================== Disk I/O Methods ====================

    @timeme
    def _load_positions_from_disk(self):
        """Load all positions from disk on startup."""

        # Check if we should skip disk loading
        should_skip = False
        if self.load_from_disk is False:
            # Explicitly disabled
            should_skip = True
        elif self.load_from_disk is True:
            # Explicitly enabled - load even in test mode
            should_skip = False
        elif self.running_unit_tests or self.is_backtesting:
            # Auto mode: skip in test/backtesting mode
            should_skip = True

        if should_skip:
            bt.logging.debug("Skipping disk load in test/backtesting mode")
            return

        # Get base miner directory
        base_dir = Path(ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests))
        if not base_dir.exists():
            bt.logging.info("No positions directory found, starting fresh")
            return

        # Iterate through all miner hotkey directories
        for hotkey_dir in base_dir.iterdir():
            if not hotkey_dir.is_dir():
                continue

            hotkey = hotkey_dir.name

            # Get all position files for this hotkey (both open and closed)
            all_files = ValiBkpUtils.get_all_files_in_dir(
                ValiBkpUtils.get_miner_all_positions_dir(hotkey, running_unit_tests=self.running_unit_tests)
            )

            if not all_files:
                continue

            positions_dict = {}  # Build dict directly keyed by position_uuid

            for position_file in all_files:
                try:
                    file_string = ValiBkpUtils.get_file(position_file)
                    position = Position.model_validate_json(file_string)
                    positions_dict[position.position_uuid] = position
                except Exception as e:
                    bt.logging.error(f"Error loading position file {position_file} for {hotkey}: {e}")

            if positions_dict:
                self.hotkey_to_positions[hotkey] = positions_dict
                bt.logging.debug(f"Loaded {len(positions_dict)} positions for {hotkey}")

        total_positions = sum(len(positions_dict) for positions_dict in self.hotkey_to_positions.values())
        bt.logging.success(
            f"Loaded {total_positions} positions for {len(self.hotkey_to_positions)} hotkeys from disk"
        )

        # Rebuild the open positions index after loading
        self._rebuild_open_index()


    @timeme
    def _apply_position_splitting_on_startup(self):
        """
        Apply position splitting to all loaded positions.
        This runs on startup if split_positions_on_disk_load is enabled.
        """
        from vali_objects.price_fetcher.live_price_server import LivePriceFetcherServer
        from vali_objects.utils.vali_utils import ValiUtils

        bt.logging.info("Applying position splitting on startup...")

        # Early exit if no positions to split (avoids loading secrets unnecessarily)
        if not self.hotkey_to_positions:
            bt.logging.info("No positions to split")
            return

        # Create live_price_fetcher for splitting logic
        secrets = ValiUtils.get_secrets(running_unit_tests=self.running_unit_tests)
        live_price_fetcher = LivePriceFetcherServer(secrets=secrets, disable_ws=True)

        total_hotkeys = len(self.hotkey_to_positions)
        hotkeys_with_splits = 0
        total_positions_split = 0

        for hotkey, positions_dict in list(self.hotkey_to_positions.items()):
            split_positions = {}  # Dict instead of list for O(1) operations
            positions_split_for_hotkey = 0

            for position in positions_dict.values():  # Iterate over dict values
                try:
                    # Split the position
                    new_positions, split_info = PositionSplitter.split_position_on_flat(position, live_price_fetcher)

                    # Add all resulting positions to the dict by UUID
                    for new_pos in new_positions:
                        split_positions[new_pos.position_uuid] = new_pos

                    # Count if this position was actually split
                    if len(new_positions) > 1:
                        positions_split_for_hotkey += 1

                except Exception as e:
                    bt.logging.error(f"Failed to split position {position.position_uuid} for hotkey {hotkey}: {e}")
                    bt.logging.error(f"Position details: {len(position.orders)} orders, trade_pair={position.trade_pair}")
                    traceback.print_exc()
                    # Keep the original position if splitting fails
                    split_positions[position.position_uuid] = position

            # Update positions for this hotkey (now assigning dict instead of list)
            self.hotkey_to_positions[hotkey] = split_positions

            if positions_split_for_hotkey > 0:
                hotkeys_with_splits += 1
                total_positions_split += positions_split_for_hotkey

        bt.logging.info(
            f"Position splitting complete: {total_positions_split} positions split across "
            f"{hotkeys_with_splits}/{total_hotkeys} hotkeys"
        )

        # Rebuild the open positions index after splitting
        self._rebuild_open_index()

    # ==================== Public Splitting Methods ====================

    def split_position_on_flat(self, position: Position, track_stats: bool = False) -> tuple[list[Position], dict]:
        """
        Public method to split a position on FLAT orders or implicit flats.
        Uses internal LivePriceFetcherClient.

        Args:
            position: The position to split
            track_stats: Whether to track splitting statistics for this miner

        Returns:
            Tuple of (list of split positions, split_info dict)
        """
        # Perform the split
        result_positions, split_info = PositionSplitter.split_position_on_flat(
            position,
            self._live_price_client,
            track_stats=track_stats
        )

        # Track statistics if requested and split actually happened
        if track_stats and len(result_positions) > 1:
            hotkey = position.miner_hotkey
            stats = self.split_stats[hotkey]

            # Update split count
            stats['n_positions_split'] += 1

            # Track pre-split return
            if position.is_closed_position:
                stats['product_return_pre_split'] *= position.return_at_close

            # Track post-split returns
            for pos in result_positions:
                if pos.is_closed_position:
                    stats['product_return_post_split'] *= pos.return_at_close

        return result_positions, split_info

    def get_split_stats(self, hotkey: str) -> dict:
        """
        Get position splitting statistics for a miner.

        Args:
            hotkey: The miner hotkey

        Returns:
            Dict with splitting statistics
        """
        return dict(self.split_stats.get(hotkey, self._default_split_stats()))

    def apply_stock_split(self, trade_pair_id: str, stock_split_ratio: float, execution_date: str):
        _cnt = 0
        for _, positions_dict in self.hotkey_to_open_positions.items():
            open_position = positions_dict.get(trade_pair_id, None)
            if open_position and open_position.apply_stock_split(stock_split_ratio, execution_date):
                self._write_position_to_disk(open_position)
                _cnt += 1

        bt.logging.info(f"Applied {trade_pair_id} stock split (ratio: {stock_split_ratio}, date: {execution_date}) to {_cnt} positions")

    # ==================== Bracket Order Attachment Methods ====================

    def attach_bracket_order_to_position(self, miner_hotkey: str, trade_pair_id: str, order_dict: dict) -> bool:
        """
        Attach a bracket order to a position. Called by LimitOrderManager.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_id: The trade pair ID
            order_dict: The order as a dictionary

        Returns:
            True if successfully attached, False if no open position found
        """
        position = self.get_open_position_for_trade_pair(miner_hotkey, trade_pair_id)
        if not position:
            return False
        position.add_unfilled_order(order_dict)
        return True

    def remove_bracket_order_from_position(self, miner_hotkey: str, trade_pair_id: str, order_uuid: str) -> bool:
        """
        Remove a bracket order from a position. Called by LimitOrderManager.

        Args:
            miner_hotkey: The miner's hotkey
            trade_pair_id: The trade pair ID
            order_uuid: The UUID of the order to remove

        Returns:
            True if found and removed, False otherwise
        """
        position = self.get_open_position_for_trade_pair(miner_hotkey, trade_pair_id)
        if not position:
            return False
        return position.remove_unfilled_order(order_uuid)

    # ==================== Disk I/O Methods ====================

    def _write_position_to_disk(self, position: Position):
        """Write a single position to disk."""
        try:
            miner_dir = ValiBkpUtils.get_partitioned_miner_positions_dir(
                position.miner_hotkey,
                position.trade_pair.trade_pair_id,
                order_status=OrderStatus.OPEN if position.is_open_position else OrderStatus.CLOSED,
                running_unit_tests=self.running_unit_tests
            )
            ValiBkpUtils.write_file(miner_dir + position.position_uuid, position)
            bt.logging.trace(f"Wrote position {position.position_uuid} for {position.miner_hotkey} to disk")

        except Exception as e:
            bt.logging.error(f"Error writing position {position.position_uuid} to disk: {e}")

