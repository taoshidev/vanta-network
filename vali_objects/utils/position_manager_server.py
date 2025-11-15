"""
Position Manager Server - RPC server for managing position data.

This server process manages position data in a normal Python dict (not IPC),
providing efficient in-place mutations and selective disk writes.
"""
import os
import bittensor as bt
import traceback
from collections import defaultdict
from pathlib import Path
from multiprocessing.managers import BaseManager
from typing import List, Dict, Optional

from time_util.time_util import TimeUtil, timeme
from vali_objects.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException


def cleanup_stale_position_manager_server(port: int = 50002):
    """
    Kill any existing process using the PositionManager RPC server port.

    This should be called before starting a new server to avoid "Address already in use" errors.

    Args:
        port: The port to cleanup (default: 50002)
    """
    import signal
    import subprocess
    import time

    if os.name != 'posix':
        bt.logging.debug("Port cleanup only supported on POSIX systems")
        return

    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')

            for pid_str in pids:
                try:
                    pid = int(pid_str)

                    # Check if it's a PositionManager server process
                    cmd_result = subprocess.run(
                        ['ps', '-p', str(pid), '-o', 'comm='],
                        capture_output=True,
                        text=True,
                        timeout=1
                    )

                    if cmd_result.returncode == 0:
                        process_name = cmd_result.stdout.strip()

                        # Only kill if it looks like our server
                        if 'PositionManagerServer' in process_name or 'python' in process_name:
                            bt.logging.warning(
                                f"Killing stale PositionManager server process "
                                f"(PID: {pid}, port: {port})"
                            )

                            try:
                                os.kill(pid, signal.SIGTERM)
                                time.sleep(0.5)

                                # Force kill if still alive
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass  # Already dead

                            except ProcessLookupError:
                                pass  # Process already terminated

                except (ValueError, subprocess.TimeoutExpired) as e:
                    bt.logging.trace(f"Error checking process {pid_str}: {e}")

    except FileNotFoundError:
        bt.logging.trace("lsof command not available, skipping port cleanup")
    except subprocess.TimeoutExpired:
        bt.logging.warning("Port cleanup timed out")
    except Exception as e:
        bt.logging.warning(f"Error during port cleanup: {e}")


class PositionManagerServer:
    """Server process that manages position data in a normal Python dict."""

    def __init__(self, running_unit_tests=False, is_backtesting=False, load_from_disk=None, split_positions_on_disk_load=False):
        """
        Initialize the server with a normal Python dict for positions.

        Args:
            running_unit_tests: Whether running in unit test mode
            is_backtesting: Whether running in backtesting mode
            load_from_disk: Override disk loading behavior (None=auto, True=force load, False=skip)
            split_positions_on_disk_load: Whether to apply position splitting after loading from disk
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

        # Statistics
        self.split_stats = defaultdict(self._default_split_stats)
        self.recalibrated_position_uuids = set()

        # Load positions from disk on startup
        self._load_positions_from_disk()

        # Apply position splitting if enabled (after loading)
        if self.split_positions_on_disk_load:
            self._apply_position_splitting_on_startup()

        bt.logging.success("PositionManagerServer initialized with normal Python dict")

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

    def _add_to_open_index(self, position: Position):
        """
        Add an open position to the secondary index for O(1) lookups.
        Only call this for positions that are definitely open.

        Raises:
            ValiRecordsMisalignmentException: If another open position already exists for this trade pair
        """
        hotkey = position.miner_hotkey
        trade_pair_id = position.trade_pair.trade_pair_id

        if hotkey not in self.hotkey_to_open_positions:
            self.hotkey_to_open_positions[hotkey] = {}

        # Check for duplicates (data corruption - this should NEVER happen)
        if trade_pair_id in self.hotkey_to_open_positions[hotkey]:
            existing_pos = self.hotkey_to_open_positions[hotkey][trade_pair_id]
            if existing_pos.position_uuid != position.position_uuid:
                # Data corruption detected - raise exception instead of silently continuing
                error_msg = (
                    f"Data corruption: Multiple open positions for miner {hotkey} and trade_pair {trade_pair_id}. "
                    f"Existing position UUID: {existing_pos.position_uuid}, "
                    f"New position UUID: {position.position_uuid}. "
                    f"Please restore cache."
                )
                bt.logging.error(error_msg)
                raise ValiRecordsMisalignmentException(error_msg)

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
        """
        self.hotkey_to_open_positions.clear()

        for hotkey, positions_dict in self.hotkey_to_positions.items():
            for position in positions_dict.values():
                if not position.is_closed_position:
                    self._add_to_open_index(position)

        total_open = sum(len(d) for d in self.hotkey_to_open_positions.values())
        bt.logging.debug(f"Rebuilt open index: {total_open} open positions across {len(self.hotkey_to_open_positions)} hotkeys")

    def _default_split_stats(self):
        return {
            'original_leverage': 0.0,
            'final_leverage': 0.0,
            'num_splits': 0
        }

    # ========================================================================
    # RPC Methods (called by client via RPC)
    # ========================================================================

    def get_positions_for_one_hotkey_rpc(self, hotkey: str, only_open_positions=False):
        """Get positions for a specific hotkey."""
        if hotkey not in self.hotkey_to_positions:
            return []

        positions_dict = self.hotkey_to_positions[hotkey]
        positions = list(positions_dict.values())  # Convert dict values to list

        if only_open_positions:
            return [p for p in positions if not p.is_closed_position]
        return positions

    def save_miner_position_rpc(self, position: Position):
        """
        Save a single position efficiently with O(1) insert/update.
        Also maintains the open positions index for fast lookups.
        Note: Disk I/O is handled by the client to maintain compatibility with existing format.
        """
        hotkey = position.miner_hotkey
        position_uuid = position.position_uuid

        if hotkey not in self.hotkey_to_positions:
            self.hotkey_to_positions[hotkey] = {}

        # Check if this position already exists (update vs insert)
        existing_position = self.hotkey_to_positions[hotkey].get(position_uuid)

        # Update the main data structure (source of truth)
        self.hotkey_to_positions[hotkey][position_uuid] = position

        # Maintain the open positions index
        if existing_position:
            # Position is being updated - handle state transitions
            was_open = not existing_position.is_closed_position
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

    def get_all_miner_positions_rpc(self, only_open_positions=False):
        """Get all positions across all miners."""
        all_positions = []
        for positions_dict in self.hotkey_to_positions.values():
            positions = positions_dict.values()  # Get all positions from nested dict
            if only_open_positions:
                all_positions.extend([p for p in positions if not p.is_closed_position])
            else:
                all_positions.extend(positions)
        return all_positions

    def get_positions_for_hotkeys_rpc(self, hotkeys: List[str], only_open_positions=False) -> Dict[str, List[Position]]:
        """
        Get positions for multiple hotkeys in a single RPC call (bulk operation).
        This is much more efficient than calling get_positions_for_one_hotkey_rpc multiple times.

        Args:
            hotkeys: List of hotkeys to fetch positions for
            only_open_positions: Whether to return only open positions

        Returns:
            Dict mapping hotkey to list of positions
        """
        result = {}
        for hotkey in hotkeys:
            if hotkey not in self.hotkey_to_positions:
                result[hotkey] = []
                continue

            positions_dict = self.hotkey_to_positions[hotkey]
            positions = list(positions_dict.values())  # Convert dict values to list

            if only_open_positions:
                result[hotkey] = [p for p in positions if not p.is_closed_position]
            else:
                result[hotkey] = positions

        return result

    def clear_all_miner_positions_rpc(self):
        """Clear all positions (for testing). Also clears the open positions index."""
        self.hotkey_to_positions.clear()
        self.hotkey_to_open_positions.clear()
        bt.logging.info("Cleared all positions and open index")

    def delete_position_rpc(self, hotkey: str, position_uuid: str):
        """
        Delete a specific position with O(1) deletion.
        Also removes from open positions index if it was open.
        Note: Disk I/O is handled by the client.
        """
        if hotkey not in self.hotkey_to_positions:
            return False

        positions_dict = self.hotkey_to_positions[hotkey]

        # O(1) direct deletion from dict
        if position_uuid in positions_dict:
            position = positions_dict[position_uuid]

            # Remove from open index if it's an open position
            if not position.is_closed_position:
                self._remove_from_open_index(position)

            del positions_dict[position_uuid]
            bt.logging.info(f"Deleted position {position_uuid} for {hotkey}")
            return True

        return False

    def get_position_rpc(self, hotkey: str, position_uuid: str):
        """Get a specific position by UUID with O(1) lookup."""
        if hotkey not in self.hotkey_to_positions:
            return None

        positions_dict = self.hotkey_to_positions[hotkey]

        # O(1) direct dict access
        return positions_dict.get(position_uuid, None)

    def get_open_position_for_trade_pair_rpc(self, hotkey: str, trade_pair_id: str) -> Optional[Position]:
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

    def get_hotkeys_with_open_positions_rpc(self):
        """Get list of hotkeys that have open positions."""
        return [
            hotkey for hotkey, positions_dict in self.hotkey_to_positions.items()
            if any(not p.is_closed_position for p in positions_dict.values())
        ]

    def get_all_hotkeys_rpc(self):
        """Get all hotkeys that have positions."""
        return list(self.hotkey_to_positions.keys())

    def health_check_rpc(self) -> dict:
        """Health check endpoint for RPC monitoring"""
        total_positions = sum(len(positions_dict) for positions_dict in self.hotkey_to_positions.values())
        total_open = sum(len(d) for d in self.hotkey_to_open_positions.values())

        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "total_positions": total_positions,
            "total_open_positions": total_open,
            "num_hotkeys": len(self.hotkey_to_positions)
        }

    def init_cache_files_rpc(self):
        """Initialize cache files (create directories)."""
        positions_dir = ValiBkpUtils.get_miner_all_positions_dir()
        positions_dir.mkdir(parents=True, exist_ok=True)
        bt.logging.debug(f"Initialized cache directory: {positions_dir}")

    @timeme
    def compact_price_sources(self):
        """
        Compact price_sources by removing old price data from closed positions.
        Runs directly on server's in-memory positions - no RPC overhead!
        """
        time_now = TimeUtil.now_in_millis()
        cutoff_time_ms = time_now - 10 * ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS  # Generous bound
        n_price_sources_removed = 0

        # Direct access to in-memory positions - no RPC call needed!
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

    def run_compaction_daemon_forever(self):
        """
        Daemon that periodically compacts price_sources from old closed positions.
        Runs on server side with direct memory access - no RPC overhead!
        """
        import time

        bt.logging.info("Starting price source compaction daemon on server")
        while True:
            try:
                t0 = time.time()
                self.compact_price_sources()
                bt.logging.info(f'Compacted price sources in {time.time() - t0:.2f} seconds')
            except Exception as e:
                bt.logging.error(f"Error in run_compaction_daemon_forever: {traceback.format_exc()}")
                time.sleep(ValiConfig.PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS)
            time.sleep(ValiConfig.PRICE_SOURCE_COMPACTING_SLEEP_INTERVAL_SECONDS)

    # ========================================================================
    # Private Helper Methods (not exposed via RPC)
    # ========================================================================

    @timeme
    def _load_positions_from_disk(self):
        """Load all positions from disk on server startup."""

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

        try:
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

                positions = []
                for position_file in all_files:
                    try:
                        file_string = ValiBkpUtils.get_file(position_file)
                        position = Position.model_validate_json(file_string)
                        positions.append(position)
                    except Exception as e:
                        bt.logging.error(f"Error loading position file {position_file} for {hotkey}: {e}")

                if positions:
                    # Convert list to nested dict keyed by position_uuid
                    self.hotkey_to_positions[hotkey] = {
                        p.position_uuid: p for p in positions
                    }
                    bt.logging.debug(f"Loaded {len(positions)} positions for {hotkey}")

            total_positions = sum(len(positions_dict) for positions_dict in self.hotkey_to_positions.values())
            bt.logging.success(
                f"Loaded {total_positions} positions for {len(self.hotkey_to_positions)} hotkeys from disk"
            )

            # Rebuild the open positions index after loading
            self._rebuild_open_index()

        except Exception as e:
            bt.logging.error(f"Error loading positions from disk: {e}")

    @timeme
    def _apply_position_splitting_on_startup(self):
        """
        Apply position splitting to all loaded positions.
        This runs on server startup if split_positions_on_disk_load is enabled.
        """
        bt.logging.info("Applying position splitting on startup...")

        # Early exit if no positions to split (avoids loading secrets unnecessarily)
        if not self.hotkey_to_positions:
            bt.logging.info("No positions to split")
            return

        # Create live_price_fetcher for splitting logic
        secrets = ValiUtils.get_secrets(running_unit_tests=self.running_unit_tests)
        live_price_fetcher = LivePriceFetcher(secrets=secrets, disable_ws=True)

        total_hotkeys = len(self.hotkey_to_positions)
        hotkeys_with_splits = 0
        total_positions_split = 0

        for hotkey, positions_dict in list(self.hotkey_to_positions.items()):
            split_positions = {}  # Dict instead of list for O(1) operations
            positions_split_for_hotkey = 0

            for position in positions_dict.values():  # Iterate over dict values
                try:
                    # Split the position
                    new_positions, split_info = self._split_position_on_flat(position, live_price_fetcher)

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

    def _find_split_points(self, position: Position) -> list[int]:
        """
        Find all valid split points in a position where splitting should occur.
        Returns a list of order indices where splits should happen.
        """
        if len(position.orders) < 2:
            return []

        split_points = []
        cumulative_leverage = 0.0
        previous_sign = None

        for i, order in enumerate(position.orders):
            cumulative_leverage += order.leverage

            # Determine the sign of leverage
            if abs(cumulative_leverage) < 1e-9:
                current_sign = 0
            elif cumulative_leverage > 0:
                current_sign = 1
            else:
                current_sign = -1

            # Check for leverage sign flip
            leverage_flipped = False
            if previous_sign is not None and previous_sign != 0 and current_sign != 0 and previous_sign != current_sign:
                leverage_flipped = True

            # Check for explicit FLAT or implicit flat
            is_explicit_flat = order.order_type == OrderType.FLAT
            is_implicit_flat = (abs(cumulative_leverage) < 1e-9 or leverage_flipped) and not is_explicit_flat

            if is_explicit_flat or is_implicit_flat:
                # Don't split if this is the last order
                if i < len(position.orders) - 1:
                    orders_before = position.orders[:i+1]
                    orders_after = position.orders[i+1:]

                    # Check if both parts are valid
                    first_valid = (len(orders_before) >= 2 and
                                 orders_before[0].order_type != OrderType.FLAT)
                    second_valid = (len(orders_after) >= 1 and
                                  orders_after[0].order_type != OrderType.FLAT)

                    if first_valid and second_valid:
                        split_points.append(i)
                        cumulative_leverage = 0.0
                        previous_sign = 0
                        continue

            previous_sign = current_sign

        return split_points

    def _split_position_on_flat(self, position: Position, live_price_fetcher) -> tuple[list[Position], dict]:
        """
        Split a position into multiple positions based on FLAT orders or implicit flats.
        Returns tuple of (list of positions, split_info dict).
        """
        try:
            split_points = self._find_split_points(position)

            if not split_points:
                return [position], {'implicit_flat_splits': 0, 'explicit_flat_splits': 0}

            # Create order groups based on split points
            order_groups = []
            start_idx = 0

            for split_idx in split_points:
                order_group = position.orders[start_idx:split_idx + 1]
                order_groups.append(order_group)
                start_idx = split_idx + 1

            # Add remaining orders if any
            if start_idx < len(position.orders):
                order_groups.append(position.orders[start_idx:])

            # Update the original position with the first group
            position.orders = order_groups[0]
            position.rebuild_position_with_updated_orders(live_price_fetcher)

            positions = [position]

            # Create new positions for remaining groups
            for order_group in order_groups[1:]:
                new_position = Position(
                    miner_hotkey=position.miner_hotkey,
                    position_uuid=order_group[0].order_uuid,
                    open_ms=0,
                    trade_pair=position.trade_pair,
                    orders=order_group
                )
                new_position.rebuild_position_with_updated_orders(live_price_fetcher)
                positions.append(new_position)

            return positions, {'implicit_flat_splits': 0, 'explicit_flat_splits': 0}

        except Exception as e:
            bt.logging.error(f"Error during position splitting: {e}")
            return [position], {'implicit_flat_splits': 0, 'explicit_flat_splits': 0}

    def _write_position_to_disk(self, position: Position):
        """Write a single position to disk."""
        try:
            from vali_objects.vali_dataclasses.order import OrderStatus

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

    def _write_positions_for_hotkey_to_disk(self, hotkey: str):
        """Write positions for a specific hotkey to disk."""
        try:
            positions_dir = ValiBkpUtils.get_miner_all_positions_dir()
            positions_dir.mkdir(parents=True, exist_ok=True)

            filepath = positions_dir / f"{hotkey}.json"
            positions = self.hotkey_to_positions.get(hotkey, [])

            # Convert positions to dict format
            positions_data = [p.to_dict() for p in positions]

            ValiBkpUtils.write_file(str(filepath), ValiBkpUtils.get_file_contents_from_json(positions_data))
            bt.logging.trace(f"Wrote {len(positions)} positions for {hotkey} to disk")

        except Exception as e:
            bt.logging.error(f"Error writing positions for {hotkey} to disk: {e}")


def start_position_manager_server(address, authkey, running_unit_tests=False, is_backtesting=False, split_positions_on_disk_load=False, start_compaction_daemon=False, ready_event=None):
    """
    Start the PositionManager server process.

    Args:
        address: (host, port) tuple for RPC server
        authkey: Authentication key for RPC
        running_unit_tests: Whether running in test mode
        is_backtesting: Whether running in backtesting mode
        split_positions_on_disk_load: Whether to apply position splitting after loading from disk
        start_compaction_daemon: Whether to start the price source compaction daemon
        ready_event: Optional multiprocessing.Event to signal when server is ready
    """
    from setproctitle import setproctitle
    from threading import Thread
    setproctitle("vali_PositionManagerServer")

    bt.logging.info(f"Starting PositionManager server on {address}")

    # Create server instance
    server_instance = PositionManagerServer(
        running_unit_tests=running_unit_tests,
        is_backtesting=is_backtesting,
        split_positions_on_disk_load=split_positions_on_disk_load
    )

    # Start compaction daemon if requested (runs in background thread on server)
    if start_compaction_daemon:
        compaction_thread = Thread(target=server_instance.run_compaction_daemon_forever, daemon=True)
        compaction_thread.start()
        bt.logging.info("Started price source compaction daemon on server")

    # Register the PositionManagerServer class directly with BaseManager
    # BaseManager will proxy method calls to the server instance
    class PositionManagerManager(BaseManager):
        pass

    # Register PositionManagerServer with a callable that returns our existing instance
    PositionManagerManager.register('PositionManagerServer', callable=lambda: server_instance)

    # Start manager and serve the instance
    manager = PositionManagerManager(address=address, authkey=authkey)
    server = manager.get_server()

    bt.logging.success(f"PositionManager server ready on {address}")

    # Signal that server is ready to accept connections
    if ready_event:
        ready_event.set()
        bt.logging.debug("Server readiness event set")

    server.serve_forever()
