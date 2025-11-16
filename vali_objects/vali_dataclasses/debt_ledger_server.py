"""
Debt Ledger Manager Server - RPC server for managing debt ledger data.

This server process manages debt ledgers in a normal Python dict (not IPC),
runs a daemon thread that continuously builds/updates ledgers, and serves
read-only RPC requests from clients.
"""
import os
import bittensor as bt
import signal
import time
import gzip
import json
import shutil
from datetime import datetime, timezone
from multiprocessing.managers import BaseManager
from typing import Dict, Optional

from ptn_api.slack_notifier import SlackNotifier
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.emissions_ledger import EmissionsLedgerManager
from vali_objects.vali_dataclasses.penalty_ledger import PenaltyLedgerManager
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO


class DebtLedgerManagerServer:
    """
    Server process that manages debt ledgers in a normal Python dict.

    Responsibilities:
    - Hold debt ledgers in normal Python dict (not IPC)
    - Run daemon thread that continuously builds/updates ledgers
    - Serve read-only RPC requests from clients
    - Handle persistence (save/load from disk)
    """

    DEFAULT_CHECK_INTERVAL_SECONDS = 3600 * 12  # 12 hours

    def __init__(self, perf_ledger_manager, position_manager, contract_manager,
                 asset_selection_manager, challengeperiod_manager=None,
                 slack_webhook_url=None, ipc_manager=None, running_unit_tests=False,
                 validator_hotkey=None):
        """
        Initialize the server with a normal Python dict for debt ledgers.

        Args:
            perf_ledger_manager: Performance ledger manager instance
            position_manager: Position manager instance
            contract_manager: Contract manager instance
            asset_selection_manager: Asset selection manager instance
            challengeperiod_manager: Challenge period manager instance
            slack_webhook_url: Slack webhook URL for notifications
            ipc_manager: IPC manager (for sub-ledger managers)
            running_unit_tests: Whether running in unit test mode
            validator_hotkey: Validator hotkey for notifications
        """
        # SOURCE OF TRUTH: Normal Python dict (NOT IPC dict!)
        # Structure: hotkey -> DebtLedger
        self.debt_ledgers: Dict[str, 'DebtLedger'] = {}

        self.perf_ledger_manager = perf_ledger_manager

        # IMPORTANT: PenaltyLedgerManager runs WITHOUT its own daemon process (run_daemon=False)
        # because DebtLedgerManagerServer itself is already a daemon process, and daemon processes
        # cannot spawn child processes. The DebtLedgerManagerServer daemon thread calls
        # penalty_ledger_manager methods directly when needed.
        self.penalty_ledger_manager = PenaltyLedgerManager(
            position_manager=position_manager,
            perf_ledger_manager=perf_ledger_manager,
            contract_manager=contract_manager,
            asset_selection_manager=asset_selection_manager,
            challengeperiod_manager=challengeperiod_manager,
            slack_webhook_url=slack_webhook_url,
            run_daemon=False,  # No daemon - already inside DebtLedgerManagerServer daemon process
            running_unit_tests=running_unit_tests,
            validator_hotkey=validator_hotkey,
            ipc_manager=ipc_manager
        )

        self.emissions_ledger_manager = EmissionsLedgerManager(
            slack_webhook_url=slack_webhook_url,
            start_daemon=False,
            ipc_manager=ipc_manager,
            perf_ledger_manager=perf_ledger_manager,
            running_unit_tests=running_unit_tests,
            validator_hotkey=validator_hotkey
        )

        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)
        self.running_unit_tests = running_unit_tests
        self.running = False

        # Load from disk on startup
        self.load_data_from_disk()

        bt.logging.success("DebtLedgerManagerServer initialized with normal Python dict")

    # ========================================================================
    # RPC METHODS (called by client via RPC)
    # ========================================================================

    def get_ledger_rpc(self, hotkey: str):
        """
        Get debt ledger for a specific hotkey.

        Args:
            hotkey: The miner's hotkey

        Returns:
            DebtLedger instance, or None if not found (pickled automatically by RPC)
        """
        return self.debt_ledgers.get(hotkey)

    def get_all_ledgers_rpc(self):
        """
        Get all debt ledgers.

        Returns:
            Dict mapping hotkey to DebtLedger instance (pickled automatically by RPC)
        """
        return self.debt_ledgers

    def get_ledger_summary_rpc(self, hotkey: str) -> Optional[dict]:
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
            'net_pnl': latest.net_pnl,
            'total_fees': latest.total_fees,
        }

    def get_all_summaries_rpc(self) -> Dict[str, dict]:
        """
        Get summary stats for all ledgers (efficient for UI/status checks).

        Returns:
            Dict mapping hotkey to summary dict
        """
        summaries = {}
        for hotkey in self.debt_ledgers:
            summary = self.get_ledger_summary_rpc(hotkey)
            if summary:
                summaries[hotkey] = summary
        return summaries

    def health_check_rpc(self) -> dict:
        """Health check endpoint for RPC monitoring."""
        return {
            "status": "ok",
            "timestamp_ms": TimeUtil.now_in_millis(),
            "total_ledgers": len(self.debt_ledgers),
            "daemon_running": self.running
        }

    # ========================================================================
    # PERSISTENCE METHODS
    # ========================================================================

    def _get_ledger_path(self) -> str:
        """Get path for debt ledger file."""
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
        from vali_objects.vali_dataclasses.debt_ledger import DebtLedger

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
    # DAEMON MODE (runs in background thread)
    # ========================================================================

    def run_daemon_forever(self, check_interval_seconds: Optional[int] = None, verbose: bool = False):
        """
        Run as daemon - continuously update debt ledgers forever.

        This runs in a background thread and is the ONLY writer to self.debt_ledgers.
        All updates happen server-side with direct dict access (no IPC overhead).

        Checks for new performance/emissions/penalty data at regular intervals and performs full rebuilds.
        Handles graceful shutdown on SIGINT/SIGTERM.

        Features:
        - Full rebuilds (debt ledgers are derived from emissions + penalties + performance)
        - Periodic refresh (default: every 12 hours)
        - Graceful shutdown
        - Automatic retry on failures

        Args:
            check_interval_seconds: How often to check for new checkpoints (default: 12 hours)
            verbose: Enable detailed logging
        """
        if check_interval_seconds is None:
            check_interval_seconds = self.DEFAULT_CHECK_INTERVAL_SECONDS

        self.running = True

        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            bt.logging.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        bt.logging.info("=" * 80)
        bt.logging.info("Debt Ledger Manager - Daemon Mode")
        bt.logging.info("=" * 80)
        bt.logging.info(f"Check Interval: {check_interval_seconds}s ({check_interval_seconds / 3600:.1f} hours)")
        bt.logging.info(f"Update Sequence: Penalty -> Emissions -> Debt (all in this thread)")
        bt.logging.info(f"Full Rebuild Mode: Enabled (debt ledgers derived from emissions + penalties + perf)")
        bt.logging.info(f"Slack Notifications: {'Enabled' if self.slack_notifier.webhook_url else 'Disabled'}")
        bt.logging.info("=" * 80)

        # Track consecutive failures for exponential backoff
        consecutive_failures = 0
        initial_backoff_seconds = 300  # Start with 5 minutes
        max_backoff_seconds = 3600  # Max 1 hour
        backoff_multiplier = 2

        time.sleep(120)  # Initial delay to stagger large ipc reads

        # Main loop
        while self.running:
            try:
                bt.logging.info("="*80)
                bt.logging.info("Starting coordinated ledger update cycle...")
                bt.logging.info("="*80)
                start_time = time.time()

                # IMPORTANT: Update sub-ledgers FIRST in correct order before building debt ledgers
                # This ensures debt ledgers have the latest data from all sources

                # Step 1: Update penalty ledgers (runs in this thread, not separate daemon)
                bt.logging.info("Step 1/3: Updating penalty ledgers...")
                penalty_start = time.time()
                self.penalty_ledger_manager.build_penalty_ledgers(force_rebuild=False)
                bt.logging.info(f"Penalty ledgers updated in {time.time() - penalty_start:.2f}s")

                # Step 2: Update emissions ledgers
                bt.logging.info("Step 2/3: Updating emissions ledgers...")
                emissions_start = time.time()
                self.emissions_ledger_manager.build_delta_update()
                bt.logging.info(f"Emissions ledgers updated in {time.time() - emissions_start:.2f}s")

                # Step 3: Build debt ledgers (combines data from penalty + emissions + perf)
                # IMPORTANT: Debt ledgers ALWAYS do full rebuilds (never delta updates)
                # since they're derived from three sources that can change retroactively
                bt.logging.info("Step 3/3: Building debt ledgers (full rebuild)...")
                debt_start = time.time()
                self.build_debt_ledgers(verbose=verbose, delta_update=False)
                bt.logging.info(f"Debt ledgers built in {time.time() - debt_start:.2f}s")

                elapsed = time.time() - start_time
                bt.logging.info("="*80)
                bt.logging.info(f"Complete update cycle finished in {elapsed:.2f}s")
                bt.logging.info("="*80)

                # Success - reset failure counter
                if consecutive_failures > 0:
                    bt.logging.info(f"Recovered after {consecutive_failures} failure(s)")
                    # Send recovery alert with VM/git/hotkey context
                    self.slack_notifier.send_ledger_recovery_alert("Debt Ledger", consecutive_failures)

                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1

                # Calculate backoff for logging
                backoff_seconds = min(
                    initial_backoff_seconds * (backoff_multiplier ** (consecutive_failures - 1)),
                    max_backoff_seconds
                )

                bt.logging.error(
                    f"Error in daemon loop (failure #{consecutive_failures}): {e}",
                    exc_info=True
                )

                # Send Slack alert with VM/git/hotkey context
                self.slack_notifier.send_ledger_failure_alert(
                    "Debt Ledger",
                    consecutive_failures,
                    e,
                    backoff_seconds
                )

            # Calculate sleep time and sleep
            if self.running:
                if consecutive_failures > 0:
                    # Exponential backoff
                    backoff_seconds = min(
                        initial_backoff_seconds * (backoff_multiplier ** (consecutive_failures - 1)),
                        max_backoff_seconds
                    )
                    next_check_time = time.time() + backoff_seconds
                    next_check_str = datetime.fromtimestamp(next_check_time, tz=timezone.utc).strftime(
                        '%Y-%m-%d %H:%M:%S UTC')
                    bt.logging.warning(
                        f"Retrying after {consecutive_failures} failure(s). "
                        f"Backoff: {backoff_seconds}s. Next attempt at: {next_check_str}"
                    )
                else:
                    # Normal interval
                    next_check_time = time.time() + check_interval_seconds
                    next_check_str = datetime.fromtimestamp(next_check_time, tz=timezone.utc).strftime(
                        '%Y-%m-%d %H:%M:%S UTC')
                    bt.logging.info(f"Next check at: {next_check_str}")

                # Sleep in small intervals to allow graceful shutdown
                while self.running and time.time() < next_check_time:
                    time.sleep(10)

        bt.logging.info("Debt Ledger Manager daemon stopped")

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
        from vali_objects.vali_dataclasses.debt_ledger import DebtLedger, DebtCheckpoint

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

        # Read all perf ledgers from perf ledger manager
        all_perf_ledgers: Dict[str, Dict[str, any]] = self.perf_ledger_manager.get_perf_ledgers(
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

                if not perf_checkpoint:
                    continue  # This hotkey doesn't have a perf checkpoint at this timestamp

                # Get corresponding penalty checkpoint (efficient O(1) lookup)
                penalty_ledger = self.penalty_ledger_manager.get_penalty_ledger(hotkey)
                penalty_checkpoint = None
                if penalty_ledger:
                    penalty_checkpoint = penalty_ledger.get_checkpoint_at_time(perf_checkpoint.last_update_ms, target_cp_duration_ms)

                # Get corresponding emissions checkpoint (efficient O(1) lookup)
                emissions_ledger = self.emissions_ledger_manager.get_ledger(hotkey)
                emissions_checkpoint = None
                if emissions_ledger:
                    emissions_checkpoint = emissions_ledger.get_checkpoint_at_time(perf_checkpoint.last_update_ms, target_cp_duration_ms)

                # Skip if we don't have both penalty and emissions data
                if not penalty_checkpoint or not emissions_checkpoint:
                    hotkeys_missing_data.append(hotkey)
                    continue

                # Validate timestamps match
                if penalty_checkpoint.last_processed_ms != perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Penalty checkpoint timestamp mismatch for {hotkey}: "
                            f"expected {perf_checkpoint.last_update_ms}, got {penalty_checkpoint.last_processed_ms}"
                        )
                    continue

                if emissions_checkpoint.chunk_end_ms != perf_checkpoint.last_update_ms:
                    if verbose:
                        bt.logging.warning(
                            f"Emissions checkpoint end time mismatch for {hotkey}: "
                            f"expected {perf_checkpoint.last_update_ms}, got {emissions_checkpoint.chunk_end_ms}"
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
                    if perf_checkpoint.last_update_ms <= last_checkpoint_ms:
                        if verbose:
                            bt.logging.info(
                                f"Skipping checkpoint for {hotkey} at {perf_checkpoint.last_update_ms} "
                                f"(already processed, last checkpoint: {last_checkpoint_ms})"
                            )
                        continue

                # Create unified debt checkpoint combining all three sources
                debt_checkpoint = DebtCheckpoint(
                    timestamp_ms=perf_checkpoint.last_update_ms,
                    # Emissions data (chunk only - cumulative calculated by summing)
                    chunk_emissions_alpha=emissions_checkpoint.chunk_emissions,
                    chunk_emissions_tao=emissions_checkpoint.chunk_emissions_tao,
                    chunk_emissions_usd=emissions_checkpoint.chunk_emissions_usd,
                    avg_alpha_to_tao_rate=emissions_checkpoint.avg_alpha_to_tao_rate,
                    avg_tao_to_usd_rate=emissions_checkpoint.avg_tao_to_usd_rate,
                    tao_balance_snapshot=emissions_checkpoint.tao_balance_snapshot,
                    alpha_balance_snapshot=emissions_checkpoint.alpha_balance_snapshot,
                    # Performance data - access attributes directly from PerfCheckpoint
                    portfolio_return=perf_checkpoint.gain,  # Current portfolio multiplier
                    pnl_gain=perf_checkpoint.pnl_gain,  # PnL gain during this checkpoint period
                    pnl_loss=perf_checkpoint.pnl_loss,  # PnL loss during this checkpoint period
                    spread_fee_loss=perf_checkpoint.spread_fee_loss,  # Spread fees during this checkpoint
                    carry_fee_loss=perf_checkpoint.carry_fee_loss,  # Carry fees during this checkpoint
                    max_drawdown=perf_checkpoint.mdd,  # Max drawdown
                    max_portfolio_value=perf_checkpoint.mpv,  # Max portfolio value achieved
                    open_ms=perf_checkpoint.open_ms,
                    accum_ms=perf_checkpoint.accum_ms,
                    n_updates=perf_checkpoint.n_updates,
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

        # Final summary
        bt.logging.info(
            f"Debt ledgers updated: {checkpoint_count} checkpoints processed, "
            f"{len(self.debt_ledgers)} hotkeys tracked "
            f"(target_cp_duration_ms: {target_cp_duration_ms}ms)"
        )


def start_debt_ledger_manager_server(
    address, authkey, perf_ledger_manager, position_manager, contract_manager,
    asset_selection_manager, challengeperiod_manager=None, slack_webhook_url=None,
    running_unit_tests=False, validator_hotkey=None, ipc_manager=None, ready_event=None
):
    """
    Start the DebtLedgerManager server process.

    Args:
        address: (host, port) tuple for RPC server
        authkey: Authentication key for RPC
        perf_ledger_manager: Performance ledger manager instance
        position_manager: Position manager instance
        contract_manager: Contract manager instance
        asset_selection_manager: Asset selection manager instance
        challengeperiod_manager: Challenge period manager instance
        slack_webhook_url: Slack webhook URL for notifications
        running_unit_tests: Whether running in test mode
        validator_hotkey: Validator hotkey for notifications
        ipc_manager: IPC manager (for sub-ledger managers)
        ready_event: Optional multiprocessing.Event to signal when server is ready
    """
    from setproctitle import setproctitle
    from threading import Thread
    setproctitle("vali_DebtLedgerManagerServer")

    bt.logging.info(f"Starting DebtLedgerManager server on {address}")

    # Create server instance
    server_instance = DebtLedgerManagerServer(
        perf_ledger_manager=perf_ledger_manager,
        position_manager=position_manager,
        contract_manager=contract_manager,
        asset_selection_manager=asset_selection_manager,
        challengeperiod_manager=challengeperiod_manager,
        slack_webhook_url=slack_webhook_url,
        running_unit_tests=running_unit_tests,
        validator_hotkey=validator_hotkey,
        ipc_manager=ipc_manager
    )

    # Start daemon thread (runs build_debt_ledgers forever)
    daemon_thread = Thread(
        target=server_instance.run_daemon_forever,
        args=(),
        kwargs={'verbose': False},
        daemon=True
    )
    daemon_thread.start()
    bt.logging.info("Started DebtLedgerManagerServer daemon thread")

    # Register the DebtLedgerManagerServer with BaseManager
    class DebtLedgerManagerRPC(BaseManager):
        pass

    DebtLedgerManagerRPC.register('DebtLedgerManagerServer', callable=lambda: server_instance)

    # Start manager and serve the instance
    manager = DebtLedgerManagerRPC(address=address, authkey=authkey)
    server = manager.get_server()

    bt.logging.success(f"DebtLedgerManager server ready on {address}")

    # Signal that server is ready to accept connections
    if ready_event:
        ready_event.set()
        bt.logging.debug("Server readiness event set")

    server.serve_forever()
