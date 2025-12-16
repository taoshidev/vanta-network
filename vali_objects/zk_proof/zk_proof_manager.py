"""
ZK Proof Manager - Self-contained background worker for daily ZK proof generation.

This manager runs as a simple background thread (no RPC) and generates ZK proofs
for all active miners once per day at midnight UTC. Proofs are saved to ~/.pop/
and uploaded to api.omron.ai for external verification.

Architecture: Follows the APIManager pattern - self-contained with built-in scheduling.
"""

import threading
import time
import traceback
from datetime import datetime, timezone
import bittensor as bt

from proof_of_portfolio import prove_async
from time_util.time_util import TimeUtil
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.metrics import Metrics
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import TP_ID_PORTFOLIO


class ZKProofManager:
    """
    Manages ZK proof generation for all miners with built-in daily scheduling.

    Self-contained background thread that:
    - Generates ZK proofs daily at midnight UTC
    - Saves results to ~/.pop/
    - Uploads proofs to api.omron.ai
    - Handles errors gracefully without crashing

    Not an RPC server - just a simple background worker for external verification.
    """

    def __init__(self, position_manager, perf_ledger, wallet):
        """
        Initialize ZK Proof Manager.

        Args:
            position_manager: PositionManagerClient for getting miner positions
            perf_ledger: PerfLedgerClient for getting performance ledgers
            wallet: Bittensor wallet for proof signing
        """
        self.position_manager = position_manager
        self.perf_ledger = perf_ledger
        self.wallet = wallet

        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.contract.contract_server import ContractClient
        self._contract_client = ContractClient(running_unit_tests=False)

        # Thread management
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

        # Timing configuration
        self.proof_generation_hour = 1  # Generate proofs at midnight UTC (00:00)
        self.last_proof_date = None  # Track last generation date to avoid duplicates

        bt.logging.info("ZKProofManager initialized")

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    def start(self):
        """Start background thread for daily proof generation."""
        if self._running:
            bt.logging.warning("ZKProofManager already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ZKProofManager"
        )
        self._thread.start()

        # Verify thread started
        time.sleep(0.1)
        if self._thread.is_alive():
            bt.logging.success(
                f"ZKProofManager started - will generate proofs daily at "
                f"{self.proof_generation_hour:02d}:00 UTC"
            )
        else:
            bt.logging.error("ZKProofManager thread failed to start")
            self._running = False

    def _run(self):
        """
        Main loop - checks hourly if it's time to generate proofs.

        Runs continuously in background, checking every hour if we should
        generate proofs. Proofs are generated once per day when the current
        hour matches proof_generation_hour (default: midnight UTC).
        """
        bt.logging.info("ZKProofManager thread running")

        while not self._stop_event.is_set():
            try:
                self._check_and_generate_daily_proofs()
            except Exception as e:
                bt.logging.error(f"ZKProofManager error in main loop: {e}")
                bt.logging.error(traceback.format_exc())

            # Check every hour if it's time to generate proofs
            # Using wait() instead of sleep() for graceful shutdown
            self._stop_event.wait(3600)  # 3600 seconds = 1 hour

        bt.logging.info("ZKProofManager thread stopped")

    def _check_and_generate_daily_proofs(self):
        """
        Check if it's time to generate daily proofs, and do so if needed.

        Proofs are generated when:
        1. We haven't generated proofs today yet (last_proof_date != today)
        2. Current hour matches target hour (proof_generation_hour)
        """
        now = datetime.now(timezone.utc)
        today = now.date()

        # Check if we should generate proofs
        should_generate = (
            self.last_proof_date != today and
            now.hour == self.proof_generation_hour
        )

        if should_generate:
            bt.logging.info(f"Starting daily ZK proof generation for {today}")
            self.generate_daily_proofs()
            self.last_proof_date = today

    def generate_daily_proofs(self):
        """
        Generate ZK proofs for all active miners.

        This can also be called manually for testing/debugging.
        Iterates over all miners with positions and generates a proof for each.
        """
        try:
            time_now = TimeUtil.now_in_millis()
            miner_hotkeys = self.position_manager.get_all_hotkeys()

            if not miner_hotkeys:
                bt.logging.info("No active miners found for ZK proof generation")
                return

            bt.logging.info(f"Generating ZK proofs for {len(miner_hotkeys)} miners")

            success_count = 0
            for hotkey in miner_hotkeys:
                try:
                    self.generate_proof_for_miner(hotkey, time_now)
                    success_count += 1
                except Exception as e:
                    bt.logging.error(f"ZK proof failed for {hotkey[:8]}: {e}")
                    bt.logging.error(traceback.format_exc())
                    continue

            bt.logging.success(
                f"Daily ZK proof generation completed: {success_count}/{len(miner_hotkeys)} successful"
            )

        except Exception as e:
            bt.logging.error(f"Daily ZK proof generation failed: {e}")
            bt.logging.error(traceback.format_exc())
            raise

    def generate_proof_for_miner(self, hotkey: str, time_now: int):
        """
        Generate ZK proof for a single miner.

        This method contains the core ZK proof generation logic extracted from
        miner_statistics_manager.py lines 976-1111.

        Args:
            hotkey: Miner's hotkey
            time_now: Current timestamp in milliseconds
        """
        bt.logging.info(f"Generating ZK proof for {hotkey}...")

        # Get portfolio ledger
        filtered_ledger = self.perf_ledger.filtered_ledger_for_scoring(hotkeys=[hotkey])
        raw_ledger_dict = filtered_ledger.get(hotkey, {})
        portfolio_ledger = raw_ledger_dict.get(TP_ID_PORTFOLIO)

        if not portfolio_ledger:
            bt.logging.debug(f"No portfolio ledger for {hotkey}, skipping ZK proof")
            return

        # Get positions
        positions = self.position_manager.get_positions_for_one_hotkey(hotkey)

        # Get account size
        account_size = self._get_account_size(hotkey, time_now)

        # Prepare miner data for proof generation
        try:
            # Calculate daily returns and PnL
            ptn_daily_returns = LedgerUtils.daily_return_log(portfolio_ledger)
            daily_pnl = LedgerUtils.daily_pnl(portfolio_ledger)

            # Calculate total PnL from checkpoints
            total_pnl = 0
            if portfolio_ledger and portfolio_ledger.cps:
                for cp in portfolio_ledger.cps:
                    total_pnl += cp.realized_pnl
                total_pnl += portfolio_ledger.cps[-1].unrealized_pnl

            # Calculate weighting distribution
            weights_float = Metrics.weighting_distribution(ptn_daily_returns)

            # Calculate augmented scores for ZK proof
            augmented_scores = self._calculate_simple_metrics(portfolio_ledger)

            if not augmented_scores:
                bt.logging.warning(
                    f"No augmented scores available for {hotkey[:8]}, using empty scores"
                )
                augmented_scores = {}

            # Construct miner data dictionary
            miner_data = {
                "daily_returns": ptn_daily_returns,
                "weights": weights_float,
                "total_pnl": total_pnl,
                "positions": {hotkey: {"positions": positions}},
                "perf_ledgers": {hotkey: portfolio_ledger},
            }

            bt.logging.info(
                f"ZK proof parameters for {hotkey}: "
                f"account_size=${account_size:,}, "
                f"daily_pnl_count={len(daily_pnl) if daily_pnl else 0}"
            )

            # Generate proof asynchronously
            zk_result = prove_async(
                miner_data=miner_data,
                daily_pnl=daily_pnl,
                hotkey=hotkey,
                vali_config=ValiConfig,
                use_weighting=True,  # Default to True for daily proofs
                bypass_confidence=False,  # Default to False
                account_size=account_size,
                augmented_scores=augmented_scores,  # Real scores calculated from daily returns
                wallet=self.wallet,
                verbose=False,  # Less verbose for automated daily runs
            )

            status = zk_result.get("status", "unknown")
            message = zk_result.get("message", "")
            bt.logging.info(f"ZK proof for {hotkey}: status={status}, message={message}")

        except Exception as e:
            bt.logging.error(
                f"Error in ZK proof generation for {hotkey}: "
                f"{type(e).__name__}: {str(e)}"
            )
            bt.logging.error(traceback.format_exc())
            raise

    def _get_account_size(self, hotkey: str, time_now: int):
        """
        Get account size for a miner from contract manager.

        Args:
            hotkey: Miner's hotkey
            time_now: Current timestamp in milliseconds

        Returns:
            int: Account size in USD (defaults to MIN_CAPITAL if not found)
        """
        try:
            account_size = self.contract_manager.get_miner_account_size(
                hotkey, time_now, most_recent=True
            )
            if account_size is not None:
                return account_size
            else:
                return ValiConfig.MIN_CAPITAL
        except Exception as e:
            bt.logging.warning(
                f"Error getting account size for {hotkey}: {e}, "
            )

    def _calculate_simple_metrics(self, portfolio_ledger) -> dict:
        """
        Calculate simplified metrics for ZK proof generation.

        These are basic calculations without penalties or asset class weighting.
        Sufficient for daily ZK proof generation.

        Args:
            portfolio_ledger: Performance ledger for the miner

        Returns:
            dict: Augmented scores in the format expected by prove_async
                  {"metric_name": {"value": float}, ...}
        """
        try:
            # Get daily returns
            daily_returns = LedgerUtils.daily_return_log(portfolio_ledger)

            if not daily_returns or len(daily_returns) == 0:
                bt.logging.warning("No daily returns available for metric calculation")
                return {}

            # Calculate each metric directly using Metrics class
            # Note: calmar requires both daily_returns and ledger
            calmar = Metrics.calmar(daily_returns, portfolio_ledger)
            sharpe = Metrics.sharpe(daily_returns)
            sortino = Metrics.sortino(daily_returns)
            omega = Metrics.omega(daily_returns)

            # Format for prove_async
            augmented_scores = {
                "calmar": {"value": calmar},
                "sharpe": {"value": sharpe},
                "sortino": {"value": sortino},
                "omega": {"value": omega},
            }

            bt.logging.debug(
                f"Calculated metrics: calmar={calmar:.4f}, sharpe={sharpe:.4f}, "
                f"sortino={sortino:.4f}, omega={omega:.4f}"
            )

            return augmented_scores

        except Exception as e:
            bt.logging.error(f"Error calculating simple metrics: {e}")
            bt.logging.error(traceback.format_exc())
            return {}

    def stop(self):
        """
        Stop the background thread gracefully.

        Sets the stop event and waits for the thread to finish.
        """
        if not self._running:
            return

        bt.logging.info("Stopping ZKProofManager...")
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

            if self._thread.is_alive():
                bt.logging.warning("ZKProofManager thread did not stop within timeout")
            else:
                bt.logging.success("ZKProofManager stopped")

        self._running = False
