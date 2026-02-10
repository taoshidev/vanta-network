import gzip
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.utils.vali_bkp_utils import CustomEncoder
from vali_objects.vali_config import RPCConnectionMode
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtLedger, DebtCheckpoint
from entity_management import entity_utils


class DebtLedgerManager():
    """
    Business logic for debt ledger management.

    NO RPC infrastructure here - pure business logic only.
    Manages debt ledgers in a normal Python dict, builds/updates ledgers,
    and handles persistence.

    The server (DebtLedgerServer) wraps this with RPC infrastructure.
    """

    DEFAULT_CHECK_INTERVAL_SECONDS = 3600 * 12  # 12 hours

    def __init__(self, slack_webhook_url=None, running_unit_tests=False,
                 validator_hotkey=None, is_mainnet=True, connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
        """
        Initialize the manager with a normal Python dict for debt ledgers.

        Note: Creates its own PerfLedgerClient and ContractClient internally (forward compatibility).
        PenaltyLedgerManager creates its own AssetSelectionClient internally.

        Args:
            slack_webhook_url: Slack webhook URL for notifications
            running_unit_tests: Whether running in unit test mode
            validator_hotkey: Validator hotkey for notifications
            connection_mode: RPC connection mode (for creating clients)
        """
        from shared_objects.slack_notifier import SlackNotifier
        from vali_objects.vali_dataclasses.ledger.emission.emissions_ledger import EmissionsLedgerManager
        from vali_objects.vali_dataclasses.ledger.penalty.penalty_ledger import PenaltyLedgerManager

        self.running_unit_tests = running_unit_tests

        # SOURCE OF TRUTH: Normal Python dict (NOT IPC dict!)
        # Structure: hotkey -> DebtLedger
        self.debt_ledgers: Dict[str, DebtLedger] = {}

        # Create PerfLedgerClient internally for accessing perf ledger data
        # In test mode, don't connect via RPC
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
        self._perf_ledger_client = PerfLedgerClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.contract.contract_client import ContractClient
        self._contract_client = ContractClient(running_unit_tests=running_unit_tests)

        # Create EntityClient for entity miner aggregation
        from entity_management.entity_client import EntityClient
        self._entity_client = EntityClient(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests,
            connect_immediately=False
        )

        # Create ChallengePeriodClient for getting entity bucket status
        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
        self._challengeperiod_client = ChallengePeriodClient(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )

        # IMPORTANT: PenaltyLedgerManager runs WITHOUT its own daemon process (run_daemon=False)
        # because DebtLedgerServer itself is already a daemon process, and daemon processes
        # cannot spawn child processes. The DebtLedgerServer daemon thread calls
        # penalty_ledger_manager methods directly when needed.
        # PenaltyLedgerManager creates its own PositionManagerClient, ChallengePeriodClient, PerfLedgerClient, and AssetSelectionClient internally.
        self.penalty_ledger_manager = PenaltyLedgerManager(
            slack_webhook_url=slack_webhook_url,
            run_daemon=False,  # No daemon - already inside DebtLedgerServer daemon process
            running_unit_tests=running_unit_tests,
            validator_hotkey=validator_hotkey
        )

        self.emissions_ledger_manager = EmissionsLedgerManager(
            slack_webhook_url=slack_webhook_url,
            start_daemon=False,
            running_unit_tests=running_unit_tests,
            validator_hotkey=validator_hotkey
        )

        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)
        self.running_unit_tests = running_unit_tests
        self.is_mainnet = is_mainnet

        # Cache for pre-compressed debt ledgers (updated on each build)
        # Stores gzip-compressed JSON bytes for instant RPC access
        self._compressed_ledgers_cache: bytes = b''

        # Load from disk on startup
        self.load_data_from_disk()

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    # ========================================================================
    # PUBLIC DATA ACCESS METHODS (called by server via self._manager)
    # ========================================================================

    def get_ledger(self, hotkey: str) -> Optional[DebtLedger]:
        """
        Get debt ledger for a specific hotkey.

        Args:
            hotkey: The miner's hotkey

        Returns:
            DebtLedger instance, or None if not found
        """
        return self.debt_ledgers.get(hotkey)

    def get_all_ledgers(self) -> Dict[str, DebtLedger]:
        """
        Get all debt ledgers.

        Returns:
            Dict mapping hotkey to DebtLedger instance
        """
        return self.debt_ledgers

    def get_ledger_summary(self, hotkey: str) -> Optional[dict]:
        """
        Get summary stats for a specific ledger (avoids sending full checkpoint history).

        Args:
            hotkey: The miner's hotkey

        Returns:
            Summary dict with cumulative stats and latest checkpoint
        """
        ledger = self.debt_ledgers.get(hotkey)
        if not ledger:
            return None

        latest = ledger.get_latest_checkpoint()
        if not latest:
            return None

        return {
            'hotkey': hotkey,
            'total_checkpoints': len(ledger.checkpoints),
            'cumulative_emissions_alpha': ledger.get_cumulative_emissions_alpha(),
            'cumulative_emissions_tao': ledger.get_cumulative_emissions_tao(),
            'cumulative_emissions_usd': ledger.get_cumulative_emissions_usd(),
            'portfolio_return': ledger.get_current_portfolio_return(),
            'weighted_score': ledger.get_current_weighted_score(),
            'latest_checkpoint_ms': latest.timestamp_ms,
            'realized_pnl': latest.realized_pnl,
            'unrealized_pnl': latest.unrealized_pnl,
            'total_fees': latest.total_fees,
        }

    def get_all_summaries(self) -> Dict[str, dict]:
        """
        Get summary stats for all ledgers (efficient for UI/status checks).

        Returns:
            Dict mapping hotkey to summary dict
        """
        summaries = {}
        for hotkey in self.debt_ledgers:
            summary = self.get_ledger_summary(hotkey)
            if summary:
                summaries[hotkey] = summary
        return summaries

    def get_compressed_summaries(self) -> bytes:
        """
        Get pre-compressed debt ledger summaries as gzip bytes from cache.

        This method returns pre-compressed data that was cached during the last
        ledger build, providing instant RPC access without compression overhead.
        Similar to MinerStatisticsManager.get_compressed_statistics().

        Returns:
            Cached compressed gzip bytes of debt ledger summaries JSON (empty bytes if cache not built yet)
        """
        return self._compressed_ledgers_cache

    # ========================================================================
    # SUB-LEDGER ACCESS METHODS (delegate to sub-managers)
    # ========================================================================

    def get_emissions_ledger(self, hotkey: str):
        """
        Get emissions ledger for a specific hotkey.

        Args:
            hotkey: The miner's hotkey

        Returns:
            EmissionsLedger instance, or None if not found
        """
        return self.emissions_ledger_manager.get_ledger(hotkey)

    def get_all_emissions_ledgers(self):
        """
        Get all emissions ledgers.

        Returns:
            Dict mapping hotkey to EmissionsLedger instance
        """
        return self.emissions_ledger_manager.get_all_ledgers()

    def get_penalty_ledger(self, hotkey: str):
        """
        Get penalty ledger for a specific hotkey.

        Args:
            hotkey: The miner's hotkey

        Returns:
            PenaltyLedger instance, or None if not found
        """
        return self.penalty_ledger_manager.get_penalty_ledger(hotkey)

    def get_all_penalty_ledgers(self):
        """
        Get all penalty ledgers.

        Returns:
            Dict mapping hotkey to PenaltyLedger instance
        """
        return self.penalty_ledger_manager.get_all_penalty_ledgers()

    # ========================================================================
    # PERSISTENCE METHODS
    # ========================================================================

    def _update_compressed_ledgers_cache(self):
        """
        Update the pre-compressed debt ledgers cache for instant RPC access.

        This method is called after build_debt_ledgers() completes.
        Caches compressed gzip bytes for zero-latency RPC responses.
        Pattern matches MinerStatisticsManager.generate_request_minerstatistics().
        """

        try:
            # Get all summaries
            summaries = self.get_all_summaries()

            # Serialize to JSON using CustomEncoder (handles datetime, BaseModel, etc.)
            json_str = json.dumps(summaries, cls=CustomEncoder)

            # Compress with gzip and cache
            self._compressed_ledgers_cache = gzip.compress(json_str.encode('utf-8'))

            bt.logging.info(
                f"Updated compressed ledgers cache: {len(summaries)} ledgers, "
                f"{len(self._compressed_ledgers_cache)} bytes"
            )

        except Exception as e:
            bt.logging.error(f"Error updating compressed ledgers cache: {e}", exc_info=True)
            # Keep old cache on error (don't clear it)

    def _get_ledger_path(self) -> str:
        """Get path for debt ledger file."""
        from vali_objects.vali_config import ValiConfig
        suffix = "/tests" if self.running_unit_tests else ""
        base_path = ValiConfig.BASE_DIR + f"{suffix}/validation/debt_ledger.json"
        return base_path + ".gz"

    def save_to_disk(self, create_backup: bool = True):
        """
        Save debt ledgers to disk with atomic write (JSON format).

        Args:
            create_backup: Whether to create timestamped backup before overwrite
        """
        if not self.debt_ledgers:
            bt.logging.warning("No debt ledgers to save")
            return

        ledger_path = self._get_ledger_path()

        # Build data structure with JSON serialization
        data = {
            "format_version": "1.0",
            "last_update_ms": int(time.time() * 1000),
            "ledgers": {hotkey: ledger.to_dict() for hotkey, ledger in self.debt_ledgers.items()}
        }

        # Atomic write: temp file -> move
        self._write_compressed(ledger_path, data)

        bt.logging.info(f"Saved {len(self.debt_ledgers)} debt ledgers to {ledger_path}")

    def load_data_from_disk(self) -> int:
        """
        Load existing ledgers from disk (JSON format).

        Returns:
            Number of ledgers loaded
        """

        ledger_path = self._get_ledger_path()

        if not os.path.exists(ledger_path):
            bt.logging.info("No existing debt ledger file found")
            return 0

        # Load data
        data = self._read_compressed(ledger_path)

        # Extract metadata
        metadata = {
            "last_update_ms": data.get("last_update_ms"),
            "format_version": data.get("format_version", "1.0")
        }

        # Reconstruct ledgers from JSON
        for hotkey, ledger_dict in data.get("ledgers", {}).items():
            ledger = DebtLedger.from_dict(ledger_dict)
            self.debt_ledgers[hotkey] = ledger

        bt.logging.info(
            f"Loaded {len(self.debt_ledgers)} debt ledgers, "
            f"metadata: {metadata}, "
            f"last update: {TimeUtil.millis_to_formatted_date_str(metadata.get('last_update_ms', 0))}"
        )

        return len(self.debt_ledgers)

    def _write_compressed(self, path: str, data: dict):
        """Write JSON data compressed with gzip (atomic write via temp file)."""
        temp_path = path + ".tmp"
        with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        shutil.move(temp_path, path)

    def _read_compressed(self, path: str) -> dict:
        """Read compressed JSON data."""

        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)

    # ========================================================================
    # BUSINESS LOGIC - BUILD DEBT LEDGERS
    # ========================================================================

    def build_debt_ledgers(self, verbose: bool = False, delta_update: bool = True):
        """
        Build or update debt ledgers for all hotkeys using timestamp-based iteration.

        IMPORTANT: This method writes directly to self.debt_ledgers (normal Python dict).
        No IPC overhead! All mutations are in-place and fast.

        Iterates over TIMESTAMPS (perf ledger checkpoints), processing ALL hotkeys at each timestamp.
        Saves to disk after completion. Matches emissions ledger pattern.

        In order to create a debt checkpoint, we must have:
        - Corresponding emissions checkpoint for that timestamp
        - Corresponding penalty checkpoint for that timestamp
        - Corresponding perf checkpoint for that timestamp

        IMPORTANT: Builds candidate ledgers first, then atomically swaps them in to prevent race conditions
        where ledgers momentarily disappear during the build process.

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints since last update. If False, rebuild from scratch.
        """
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import TP_ID_PORTFOLIO

        # Build into candidate dict to prevent race conditions (don't clear existing ledgers yet)
        if delta_update:
            # Delta update: start with copies of existing ledgers
            candidate_ledgers = {}
            for hotkey, existing_ledger in self.debt_ledgers.items():
                # Create a new DebtLedger with copies of existing checkpoints
                candidate_ledgers[hotkey] = DebtLedger(hotkey, checkpoints=list(existing_ledger.checkpoints))
        else:
            # Full rebuild: start from scratch
            candidate_ledgers = {}
            bt.logging.info("Full rebuild mode: building new debt ledgers from scratch")

        # Read all perf ledgers from perf ledger client
        all_perf_ledgers: Dict[str, Dict[str, any]] = self._perf_ledger_client.get_perf_ledgers(
            portfolio_only=False
        )

        if not all_perf_ledgers:
            bt.logging.warning("No performance ledgers found")
            return

        # Pick a reference portfolio ledger (use the one with the most checkpoints for maximum coverage)
        reference_portfolio_ledger = None
        reference_hotkey = None
        max_checkpoints = 0

        for hotkey, ledger_dict in all_perf_ledgers.items():
            portfolio_ledger = ledger_dict.get(TP_ID_PORTFOLIO)
            if portfolio_ledger and portfolio_ledger.cps:
                if len(portfolio_ledger.cps) > max_checkpoints:
                    max_checkpoints = len(portfolio_ledger.cps)
                    reference_portfolio_ledger = portfolio_ledger
                    reference_hotkey = hotkey

        if not reference_portfolio_ledger:
            bt.logging.warning("No valid portfolio ledgers found with checkpoints")
            return

        bt.logging.info(
            f"Using portfolio ledger from {reference_hotkey[:16]}...{reference_hotkey[-8:]} "
            f"as reference ({len(reference_portfolio_ledger.cps)} checkpoints, "
            f"target_cp_duration_ms: {reference_portfolio_ledger.target_cp_duration_ms}ms)"
        )

        target_cp_duration_ms = reference_portfolio_ledger.target_cp_duration_ms

        # Determine which checkpoints to process based on delta update mode
        # Find the ledger with the MOST checkpoints (longest history) to use as reference
        # This prevents truncating history when new miners register with few checkpoints
        last_processed_ms = 0
        if delta_update and candidate_ledgers:
            reference_ledger = None
            max_checkpoint_count = 0
            max_last_processed_ms = 0

            # Find ledger with most checkpoints
            for ledger in candidate_ledgers.values():
                if ledger.checkpoints:
                    checkpoint_count = len(ledger.checkpoints)
                    ledger_last_ms = ledger.checkpoints[-1].timestamp_ms

                    if checkpoint_count > max_checkpoint_count:
                        max_checkpoint_count = checkpoint_count
                        reference_ledger = ledger
                        last_processed_ms = ledger_last_ms

                    # Track maximum timestamp for sanity check
                    if ledger_last_ms > max_last_processed_ms:
                        max_last_processed_ms = ledger_last_ms

            if last_processed_ms > 0:
                # Sanity check: reference ledger (most checkpoints) should have the maximum timestamp
                # This validates that the longest-running miner is up-to-date
                assert last_processed_ms == max_last_processed_ms, (
                    f"Reference ledger (most checkpoints: {max_checkpoint_count}) has timestamp "
                    f"{TimeUtil.millis_to_formatted_date_str(last_processed_ms)}, but max timestamp across "
                    f"all ledgers is {TimeUtil.millis_to_formatted_date_str(max_last_processed_ms)}. "
                    f"This indicates the reference ledger is behind, which would cause history truncation."
                )

                bt.logging.info(
                    f"Delta update mode: resuming from {TimeUtil.millis_to_formatted_date_str(last_processed_ms)} "
                    f"(reference ledger with {max_checkpoint_count} checkpoints)"
                )

        # Filter checkpoints to process
        perf_checkpoints_to_process = []
        for checkpoint in reference_portfolio_ledger.cps:
            # Skip active checkpoints (incomplete)
            if checkpoint.accum_ms != target_cp_duration_ms:
                continue

            checkpoint_ms = checkpoint.last_update_ms

            # Skip checkpoints we've already processed in delta update mode
            if delta_update and checkpoint_ms <= last_processed_ms:
                continue

            perf_checkpoints_to_process.append(checkpoint)

        if not perf_checkpoints_to_process:
            bt.logging.info("No new checkpoints to process")
            return

        bt.logging.info(
            f"Processing {len(perf_checkpoints_to_process)} checkpoints "
            f"(from {TimeUtil.millis_to_formatted_date_str(perf_checkpoints_to_process[0].last_update_ms)} "
            f"to {TimeUtil.millis_to_formatted_date_str(perf_checkpoints_to_process[-1].last_update_ms)})"
        )

        # Track all hotkeys we need to process (from perf ledgers)
        all_hotkeys_to_track = set(all_perf_ledgers.keys())

        # Optimization: Find earliest emissions timestamp across all hotkeys to skip early checkpoints
        earliest_emissions_ms = self.emissions_ledger_manager.get_earliest_emissions_timestamp()

        if earliest_emissions_ms:
            bt.logging.info(
                f"Earliest emissions data starts at {TimeUtil.millis_to_formatted_date_str(earliest_emissions_ms)}"
            )

        # Iterate over TIMESTAMPS processing ALL hotkeys at each timestamp
        checkpoint_count = 0
        for perf_checkpoint in perf_checkpoints_to_process:
            checkpoint_count += 1
            checkpoint_start_time = time.time()

            # Skip this entire timestamp if it's before the earliest emissions data
            if earliest_emissions_ms and perf_checkpoint.last_update_ms < earliest_emissions_ms:
                if verbose:
                    bt.logging.info(
                        f"Skipping checkpoint {checkpoint_count} at {TimeUtil.millis_to_formatted_date_str(perf_checkpoint.last_update_ms)} "
                        f"(before earliest emissions data)"
                    )
                continue

            hotkeys_processed_at_checkpoint = 0
            hotkeys_missing_data = []

            # Process ALL hotkeys at this timestamp
            for hotkey in all_hotkeys_to_track:
                # Get ledgers for this hotkey
                ledger_dict = all_perf_ledgers.get(hotkey)
                if not ledger_dict:
                    continue

                portfolio_ledger = ledger_dict.get(TP_ID_PORTFOLIO)
                if not portfolio_ledger or not portfolio_ledger.cps:
                    continue

                # CRITICAL FIX: Get THIS MINER'S checkpoint at the current timestamp,
                # not the reference checkpoint (which would use the same PnL for all miners)
                miner_perf_checkpoint = portfolio_ledger.get_checkpoint_at_time(
                    perf_checkpoint.last_update_ms,
                    target_cp_duration_ms
                )

                if not miner_perf_checkpoint:
                    continue  # This hotkey doesn't have a perf checkpoint at this timestamp

                # Get corresponding penalty checkpoint (efficient O(1) lookup)
                penalty_ledger = self.penalty_ledger_manager.get_penalty_ledger(hotkey)
                penalty_checkpoint = None
                if penalty_ledger:
                    penalty_checkpoint = penalty_ledger.get_checkpoint_at_time(miner_perf_checkpoint.last_update_ms, target_cp_duration_ms)

                # Get corresponding emissions checkpoint (efficient O(1) lookup)
                emissions_ledger = self.emissions_ledger_manager.get_ledger(hotkey)
                emissions_checkpoint = None
                if emissions_ledger:
                    emissions_checkpoint = emissions_ledger.get_checkpoint_at_time(miner_perf_checkpoint.last_update_ms, target_cp_duration_ms)

                # Check if this is a synthetic hotkey (subaccount)
                is_subaccount = entity_utils.is_synthetic_hotkey(hotkey)

                # Skip if we don't have penalty data
                if not penalty_checkpoint:
                    hotkeys_missing_data.append(hotkey)
                    continue

                # For non-synthetic hotkeys (regular miners, entity hotkeys), mainnet emissions are required
                # For synthetic hotkeys (subaccounts), emissions are optional (they don't receive on-chain emissions)
                if self.is_mainnet and not emissions_checkpoint and not is_subaccount:
                    hotkeys_missing_data.append(hotkey)
                    continue

                # Validate timestamps match
                if miner_perf_checkpoint.last_update_ms != perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Perf checkpoint timestamp mismatch for {hotkey}: "
                            f"expected {perf_checkpoint.last_update_ms}, got {miner_perf_checkpoint.last_update_ms}"
                        )
                    continue

                if penalty_checkpoint.last_processed_ms != miner_perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Penalty checkpoint timestamp mismatch for {hotkey}: "
                            f"expected {miner_perf_checkpoint.last_update_ms}, got {penalty_checkpoint.last_processed_ms}"
                        )
                    continue

                # Only validate emissions timestamp if we have an emissions checkpoint
                if emissions_checkpoint and emissions_checkpoint.chunk_end_ms != miner_perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Emissions checkpoint end time mismatch for {hotkey}: "
                            f"expected {miner_perf_checkpoint.last_update_ms}, got {emissions_checkpoint.chunk_end_ms}"
                        )
                    continue

                # Get or create debt ledger for this hotkey (from candidate ledgers)
                if hotkey in candidate_ledgers:
                    debt_ledger = candidate_ledgers[hotkey]
                else:
                    debt_ledger = DebtLedger(hotkey)

                # Skip if this hotkey already has a checkpoint at this timestamp (delta update safety check)
                if delta_update and debt_ledger.checkpoints:
                    last_checkpoint_ms = debt_ledger.checkpoints[-1].timestamp_ms
                    if miner_perf_checkpoint.last_update_ms <= last_checkpoint_ms:
                        if verbose:
                            bt.logging.info(
                                f"Skipping checkpoint for {hotkey} at {miner_perf_checkpoint.last_update_ms} "
                                f"(already processed, last checkpoint: {last_checkpoint_ms})"
                            )
                        continue

                # Create unified debt checkpoint combining all three sources
                # CRITICAL: Use miner_perf_checkpoint (this miner's data), not perf_checkpoint (reference miner's data)

                # Handle emissions data - use zero values for synthetic hotkeys (subaccounts)
                if emissions_checkpoint and self.is_mainnet:
                    chunk_emissions_alpha = emissions_checkpoint.chunk_emissions
                    chunk_emissions_tao = emissions_checkpoint.chunk_emissions_tao
                    chunk_emissions_usd = emissions_checkpoint.chunk_emissions_usd
                    avg_alpha_to_tao_rate = emissions_checkpoint.avg_alpha_to_tao_rate
                    avg_tao_to_usd_rate = emissions_checkpoint.avg_tao_to_usd_rate
                    tao_balance_snapshot = emissions_checkpoint.tao_balance_snapshot
                    alpha_balance_snapshot = emissions_checkpoint.alpha_balance_snapshot
                else:
                    # Synthetic hotkeys (subaccounts) and testnet have zero emissions and zero rates
                    # Entity's real emissions will replace these during aggregation
                    chunk_emissions_alpha = 0.0
                    chunk_emissions_tao = 0.0
                    chunk_emissions_usd = 0.0
                    avg_alpha_to_tao_rate = 0.0
                    avg_tao_to_usd_rate = 0.0
                    tao_balance_snapshot = 0.0
                    alpha_balance_snapshot = 0.0

                debt_checkpoint = DebtCheckpoint(
                    timestamp_ms=miner_perf_checkpoint.last_update_ms,
                    # Emissions data (chunk only - cumulative calculated by summing)
                    chunk_emissions_alpha=chunk_emissions_alpha,
                    chunk_emissions_tao=chunk_emissions_tao,
                    chunk_emissions_usd=chunk_emissions_usd,
                    avg_alpha_to_tao_rate=avg_alpha_to_tao_rate,
                    avg_tao_to_usd_rate=avg_tao_to_usd_rate,
                    tao_balance_snapshot=tao_balance_snapshot,
                    alpha_balance_snapshot=alpha_balance_snapshot,
                    # Performance data - access attributes directly from THIS MINER'S PerfCheckpoint
                    portfolio_return=miner_perf_checkpoint.gain,  # Current portfolio multiplier
                    realized_pnl=miner_perf_checkpoint.realized_pnl,  # Realized PnL during this checkpoint period
                    unrealized_pnl=miner_perf_checkpoint.unrealized_pnl,  # Unrealized PnL during this checkpoint period
                    spread_fee_loss=miner_perf_checkpoint.spread_fee_loss,  # Spread fees during this checkpoint
                    carry_fee_loss=miner_perf_checkpoint.carry_fee_loss,  # Carry fees during this checkpoint
                    max_drawdown=miner_perf_checkpoint.mdd,  # Max drawdown
                    max_portfolio_value=miner_perf_checkpoint.mpv,  # Max portfolio value achieved
                    open_ms=miner_perf_checkpoint.open_ms,
                    accum_ms=miner_perf_checkpoint.accum_ms,
                    n_updates=miner_perf_checkpoint.n_updates,
                    # Penalty data
                    drawdown_penalty=penalty_checkpoint.drawdown_penalty,
                    risk_profile_penalty=penalty_checkpoint.risk_profile_penalty,
                    min_collateral_penalty=penalty_checkpoint.min_collateral_penalty,
                    risk_adjusted_performance_penalty=penalty_checkpoint.risk_adjusted_performance_penalty,
                    total_penalty=penalty_checkpoint.total_penalty,
                    challenge_period_status=penalty_checkpoint.challenge_period_status,
                )

                # Add checkpoint to candidate ledger (validates strict contiguity)
                debt_ledger.add_checkpoint(debt_checkpoint, target_cp_duration_ms)
                candidate_ledgers[hotkey] = debt_ledger  # Update candidate ledgers
                hotkeys_processed_at_checkpoint += 1

            # Log progress for this checkpoint
            checkpoint_elapsed = time.time() - checkpoint_start_time
            checkpoint_dt = datetime.fromtimestamp(perf_checkpoint.last_update_ms / 1000, tz=timezone.utc)
            bt.logging.info(
                f"Checkpoint {checkpoint_count}/{len(perf_checkpoints_to_process)} "
                f"({checkpoint_dt.strftime('%Y-%m-%d %H:%M UTC')}): "
                f"{hotkeys_processed_at_checkpoint} hotkeys processed, "
                f"{len(hotkeys_missing_data)} missing data, "
                f"{checkpoint_elapsed:.2f}s"
            )

        # Build completed successfully - atomically swap candidate ledgers into production
        # This prevents race conditions where ledgers momentarily disappear during build
        bt.logging.info(
            f"Build completed successfully: {checkpoint_count} checkpoints for {len(candidate_ledgers)} hotkeys. "
            f"Atomically updating debt ledgers..."
        )

        # Direct assignment to normal dict (no IPC overhead!)
        self.debt_ledgers = candidate_ledgers

        # Save to disk after atomic swap
        bt.logging.info(f"Saving {len(self.debt_ledgers)} debt ledgers to disk...")
        self.save_to_disk(create_backup=False)

        # Update compressed ledgers cache for instant RPC access (matches MinerStatisticsManager pattern)
        bt.logging.info("Updating compressed ledgers cache...")
        self._update_compressed_ledgers_cache()

        # Final summary
        bt.logging.info(
            f"Debt ledgers updated: {checkpoint_count} checkpoints processed, "
            f"{len(self.debt_ledgers)} hotkeys tracked "
            f"(target_cp_duration_ms: {target_cp_duration_ms}ms)"
        )

        # Aggregate entity debt ledgers after build completes
        bt.logging.info("Aggregating entity debt ledgers...")
        self.aggregate_entity_debt_ledgers(target_cp_duration_ms, verbose=verbose)

    def aggregate_entity_debt_ledgers(self, target_cp_duration_ms: int, verbose: bool = False):
        """
        Aggregate debt ledgers from all active subaccounts under their entity hotkeys.

        This method should be called after build_debt_ledgers() completes to ensure
        all subaccount ledgers are up-to-date before aggregation.

        For each entity:
        - Get all active subaccounts
        - Aggregate their debt ledgers timestamp by timestamp
        - Store aggregated ledger under entity_hotkey

        Aggregation rules:
        - Sum: emissions, realized PnL, fees, open_ms, n_updates, balances
        - Weighted average: portfolio_return (weighted by max_portfolio_value)
        - Worst case: max_drawdown (take minimum)
        - Max: max_portfolio_value (sum across subaccounts)
        - Ignore: penalties, unrealized pnl

        Args:
            target_cp_duration_ms: Target checkpoint duration in milliseconds
            verbose: Enable detailed logging
        """
        try:
            # Get all registered entities
            all_entities = self._entity_client.get_all_entities()

            if not all_entities:
                bt.logging.info("No entities registered - skipping entity aggregation")
                return

            bt.logging.info(f"Aggregating debt ledgers for {len(all_entities)} entities")

            entity_count = 0
            for entity_hotkey, entity_data in all_entities.items():
                # Get active subaccounts for this entity
                active_subaccounts = [sa for sa in entity_data.get('subaccounts', {}).values()
                                     if sa.get('status') == 'active']

                if not active_subaccounts:
                    if verbose:
                        bt.logging.info(f"Entity {entity_hotkey} has no active subaccounts - skipping")
                    continue

                # Get debt ledgers for all active subaccounts
                subaccount_ledgers = []
                for subaccount in active_subaccounts:
                    synthetic_hotkey = subaccount.get('synthetic_hotkey')
                    if not synthetic_hotkey:
                        continue

                    ledger = self.debt_ledgers.get(synthetic_hotkey)
                    if ledger and ledger.checkpoints:
                        subaccount_ledgers.append((synthetic_hotkey, ledger))

                if not subaccount_ledgers:
                    if verbose:
                        bt.logging.info(
                            f"Entity {entity_hotkey} has {len(active_subaccounts)} active subaccounts "
                            f"but no debt ledgers found - skipping"
                        )
                    continue

                # Collect all unique timestamps across all subaccount ledgers
                all_timestamps = set()
                for _, ledger in subaccount_ledgers:
                    for checkpoint in ledger.checkpoints:
                        all_timestamps.add(checkpoint.timestamp_ms)

                if not all_timestamps:
                    continue

                # Sort timestamps chronologically
                sorted_timestamps = sorted(all_timestamps)

                entity_emissions_ledger = self.emissions_ledger_manager.get_ledger(entity_hotkey)

                # Create aggregated checkpoints for each timestamp
                aggregated_checkpoints = []
                for timestamp_ms in sorted_timestamps:
                    # Collect checkpoints from all subaccounts at this timestamp
                    checkpoints_at_time = []
                    for synthetic_hotkey, ledger in subaccount_ledgers:
                        checkpoint = ledger.get_checkpoint_at_time(timestamp_ms, target_cp_duration_ms)
                        if checkpoint:
                            checkpoints_at_time.append(checkpoint)

                    if not checkpoints_at_time:
                        continue

                    # Get entity emissions checkpoint for this timestamp
                    entity_emissions_cp = entity_emissions_ledger.get_checkpoint_at_time(timestamp_ms, target_cp_duration_ms) if entity_emissions_ledger else None

                    # Aggregate fields across all subaccounts at this timestamp
                    # Sum additive fields
                    # Use entity emissions if available, otherwise zero (for newly registered entities)
                    chunk_emissions_alpha = getattr(entity_emissions_cp, "chunk_emissions", 0.0)
                    chunk_emissions_tao = getattr(entity_emissions_cp, "chunk_emissions_tao", 0.0)
                    chunk_emissions_usd = getattr(entity_emissions_cp, "chunk_emissions_usd", 0.0)
                    alpha_to_tao_rate = getattr(entity_emissions_cp, "avg_alpha_to_tao_rate", 0.0)
                    tao_to_usd_rate = getattr(entity_emissions_cp, "avg_tao_to_usd_rate", 0.0)
                    tao_balance = getattr(entity_emissions_cp, "tao_balance_snapshot", 0.0)
                    alpha_balance = getattr(entity_emissions_cp, "alpha_balance_snapshot", 0.0)
                    agg_realized_pnl = sum(max(cp.realized_pnl, 0.0) for cp in checkpoints_at_time)     # sum positive realized pnl
                    agg_unrealized_pnl = 0.0    # ignore unrealized pnl
                    agg_spread_fee = sum(cp.spread_fee_loss for cp in checkpoints_at_time)
                    agg_carry_fee = sum(cp.carry_fee_loss for cp in checkpoints_at_time)
                    agg_max_portfolio_value = sum(cp.max_portfolio_value for cp in checkpoints_at_time)
                    agg_open_ms = sum(cp.open_ms for cp in checkpoints_at_time)
                    agg_n_updates = sum(cp.n_updates for cp in checkpoints_at_time)

                    # Weighted average for portfolio_return (weighted by max_portfolio_value)
                    total_weight = sum(cp.max_portfolio_value for cp in checkpoints_at_time)
                    if total_weight > 0:
                        agg_portfolio_return = sum(
                            cp.portfolio_return * cp.max_portfolio_value
                            for cp in checkpoints_at_time
                        ) / total_weight
                    else:
                        # If no weight, use simple average
                        agg_portfolio_return = sum(cp.portfolio_return for cp in checkpoints_at_time) / len(checkpoints_at_time)

                    # Worst case for max_drawdown (minimum = worst drawdown)
                    agg_max_drawdown = min(cp.max_drawdown for cp in checkpoints_at_time)

                    # Use entity's actual challenge period bucket status, not aggregate of subaccounts
                    bucket = self._challengeperiod_client.get_miner_bucket(entity_hotkey)
                    entity_bucket = str(bucket.value) if bucket else "unknown"

                    # Use accum_ms from first checkpoint (should be same for all at this timestamp)
                    agg_accum_ms = checkpoints_at_time[0].accum_ms

                    # Create aggregated checkpoint
                    aggregated_checkpoint = DebtCheckpoint(
                        timestamp_ms=timestamp_ms,
                        # Emissions
                        chunk_emissions_alpha=chunk_emissions_alpha,
                        chunk_emissions_tao=chunk_emissions_tao,
                        chunk_emissions_usd=chunk_emissions_usd,
                        avg_alpha_to_tao_rate=alpha_to_tao_rate,
                        avg_tao_to_usd_rate=tao_to_usd_rate,
                        tao_balance_snapshot=tao_balance,
                        alpha_balance_snapshot=alpha_balance,
                        # Performance
                        portfolio_return=agg_portfolio_return,
                        realized_pnl=agg_realized_pnl,
                        unrealized_pnl=agg_unrealized_pnl,
                        spread_fee_loss=agg_spread_fee,
                        carry_fee_loss=agg_carry_fee,
                        max_drawdown=agg_max_drawdown,
                        max_portfolio_value=agg_max_portfolio_value,
                        open_ms=agg_open_ms,
                        accum_ms=agg_accum_ms,
                        n_updates=agg_n_updates,
                        # Penalties (Ignored for entity miner)
                        drawdown_penalty=1.0,
                        risk_profile_penalty=1.0,
                        min_collateral_penalty=1.0,
                        risk_adjusted_performance_penalty=1.0,
                        total_penalty=1.0,
                        challenge_period_status=entity_bucket,
                    )

                    aggregated_checkpoints.append(aggregated_checkpoint)

                if not aggregated_checkpoints:
                    continue

                # Create aggregated debt ledger for entity
                entity_ledger = DebtLedger(entity_hotkey, checkpoints=aggregated_checkpoints)

                # Store in debt_ledgers dict
                self.debt_ledgers[entity_hotkey] = entity_ledger
                entity_count += 1

                if verbose:
                    bt.logging.info(
                        f"Aggregated {len(aggregated_checkpoints)} checkpoints for entity {entity_hotkey} "
                        f"from {len(subaccount_ledgers)} active subaccounts"
                    )

            bt.logging.info(
                f"Entity aggregation completed: {entity_count} entities aggregated "
                f"({len(all_entities) - entity_count} skipped with no data)"
            )

        except Exception as e:
            bt.logging.error(f"Error aggregating entity debt ledgers: {e}", exc_info=True)
