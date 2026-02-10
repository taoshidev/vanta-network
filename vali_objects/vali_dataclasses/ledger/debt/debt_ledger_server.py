"""
Debt Ledger Server - RPC server wrapper for DebtLedgerManager.

This server wraps DebtLedgerManager with RPC infrastructure, following the
established Server/Manager pattern (like PerfLedgerServer/PerfLedgerManager).

Architecture:
- DebtLedgerManager: Pure business logic (in debt_ledger.py)
- DebtLedgerServer: Lightweight RPC wrapper (this file)

The server maintains self._manager and delegates all business logic to it.
"""
import bittensor as bt
import time
from typing import Dict, Optional

from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc.rpc_server_base import RPCServerBase


class DebtLedgerServer(RPCServerBase):
    """
    RPC server wrapper for DebtLedgerManager.

    Responsibilities:
    - Provide RPC infrastructure (inherits from RPCServerBase)
    - Expose RPC methods that delegate to self._manager
    - Run daemon thread that calls self._manager.build_debt_ledgers()
    - Handle graceful shutdown

    The actual business logic lives in DebtLedgerManager (debt_ledger.py).
    """
    service_name = ValiConfig.RPC_DEBTLEDGER_SERVICE_NAME
    service_port = ValiConfig.RPC_DEBTLEDGER_PORT

    def __init__(self, slack_webhook_url=None, running_unit_tests=False,
                 validator_hotkey=None, start_server=True, start_daemon=True,
                 is_backtesting=False, is_mainnet=True, connection_mode=RPCConnectionMode.RPC):
        """
        Initialize the server with RPC infrastructure.

        Args:
            slack_webhook_url: Slack webhook URL for notifications
            running_unit_tests: Whether running in unit test mode
            validator_hotkey: Validator hotkey for notifications
            start_server: Whether to start RPC server
            start_daemon: Whether to start daemon thread
            is_backtesting: Whether running in backtesting mode (unused, for compatibility)
            connection_mode: RPC connection mode
        """
        self.is_backtesting = is_backtesting
        # Create the manager first (needed before RPCServerBase init for daemon)
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_manager import DebtLedgerManager
        self._manager = DebtLedgerManager(
            slack_webhook_url=slack_webhook_url,
            running_unit_tests=running_unit_tests,
            validator_hotkey=validator_hotkey,
            is_mainnet=is_mainnet,
            connection_mode=connection_mode
        )

        # Initialize RPCServerBase with standard daemon pattern
        # Check interval: 12 hours (matching DEFAULT_CHECK_INTERVAL_SECONDS)
        # hang_timeout_s: Dynamically set to 2x interval to prevent false alarms during normal sleep
        # Backoff values auto-calculated: 300s initial (5 min), 3600s max (1 hour) for heavyweight daemon
        daemon_interval_s = self._manager.DEFAULT_CHECK_INTERVAL_SECONDS  # 12 hours (43200s)
        hang_timeout_s = daemon_interval_s * 2.0  # 24 hours (2x interval)

        super().__init__(
            service_name=ValiConfig.RPC_DEBTLEDGER_SERVICE_NAME,
            port=ValiConfig.RPC_DEBTLEDGER_PORT,
            connection_mode=connection_mode,
            slack_notifier=self._manager.slack_notifier,  # Use manager's slack_notifier for daemon alerts
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s,
            daemon_stagger_s=120.0    # Stagger startup by 2 minutes to avoid IPC contention
        )

        self.running_unit_tests = running_unit_tests

    # ========================================================================
    # PROPERTIES (forward to manager)
    # ========================================================================

    @property
    def contract_manager(self):
        """Get contract client from manager."""
        return self._manager.contract_manager

    @property
    def debt_ledgers(self):
        """Get debt ledgers dict from manager (for backward compatibility)."""
        return self._manager.debt_ledgers

    @property
    def penalty_ledger_manager(self):
        """Get penalty ledger manager from manager (for backward compatibility)."""
        return self._manager.penalty_ledger_manager

    @property
    def emissions_ledger_manager(self):
        """Get emissions ledger manager from manager (for backward compatibility)."""
        return self._manager.emissions_ledger_manager

    # ========================================================================
    # RPCServerBase ABSTRACT METHODS
    # ========================================================================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work - update all ledgers.

        This method is called by RPCServerBase's standard daemon loop.
        Updates penalty → emissions → debt ledgers in sequence.

        Note: Exception handling, exponential backoff, and startup stagger are handled by the base class.
              Exceptions will bubble up to RPCServerBase._daemon_loop() for proper retry logic.
        """
        if self._is_shutdown():
            return

        bt.logging.info("="*80)
        bt.logging.info("Starting coordinated ledger update cycle...")
        bt.logging.info("="*80)
        start_time = time.time()

        # IMPORTANT: Update sub-ledgers FIRST in correct order before building debt ledgers
        # This ensures debt ledgers have the latest data from all sources

        # Step 1: Update penalty ledgers
        bt.logging.info("Step 1/3: Updating penalty ledgers...")
        penalty_start = time.time()
        self._manager.penalty_ledger_manager.build_penalty_ledgers(delta_update=True)
        bt.logging.info(f"Penalty ledgers updated in {time.time() - penalty_start:.2f}s")

        # Step 2: Update emissions ledgers
        bt.logging.info("Step 2/3: Updating emissions ledgers...")
        emissions_start = time.time()
        self._manager.emissions_ledger_manager.build_delta_update()
        bt.logging.info(f"Emissions ledgers updated in {time.time() - emissions_start:.2f}s")

        # Step 3: Build debt ledgers (full rebuild)
        bt.logging.info("Step 3/3: Building debt ledgers (full rebuild)...")
        debt_start = time.time()
        self._manager.build_debt_ledgers(verbose=False, delta_update=False)
        bt.logging.info(f"Debt ledgers built in {time.time() - debt_start:.2f}s")

        elapsed = time.time() - start_time
        bt.logging.info("="*80)
        bt.logging.info(f"Complete update cycle finished in {elapsed:.2f}s")
        bt.logging.info("="*80)

    # ========================================================================
    # RPC METHODS (delegate to manager)
    # ========================================================================

    def get_ledger_rpc(self, hotkey: str):
        """
        Get debt ledger for a specific hotkey (RPC method).

        Args:
            hotkey: The miner's hotkey

        Returns:
            DebtLedger instance, or None if not found (pickled automatically by RPC)
        """
        return self._manager.get_ledger(hotkey)

    def get_all_ledgers_rpc(self):
        """
        Get all debt ledgers (RPC method).

        Returns:
            Dict mapping hotkey to DebtLedger instance (pickled automatically by RPC)
        """
        return self._manager.get_all_ledgers()

    def get_ledger_summary_rpc(self, hotkey: str) -> Optional[dict]:
        """
        Get summary stats for a specific ledger (RPC method).

        Args:
            hotkey: The miner's hotkey

        Returns:
            Summary dict with cumulative stats and latest checkpoint
        """
        return self._manager.get_ledger_summary(hotkey)

    def get_all_summaries_rpc(self) -> Dict[str, dict]:
        """
        Get summary stats for all ledgers (RPC method).

        Returns:
            Dict mapping hotkey to summary dict
        """
        return self._manager.get_all_summaries()

    def get_compressed_summaries_rpc(self) -> bytes:
        """
        Get pre-compressed debt ledger summaries as gzip bytes from cache (RPC method).

        Returns:
            Cached compressed gzip bytes of debt ledger summaries JSON
        """
        return self._manager.get_compressed_summaries()

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "total_ledgers": len(self._manager.debt_ledgers)
        }

    # ========================================================================
    # EMISSIONS LEDGER RPC METHODS (delegate to manager's sub-manager)
    # ========================================================================

    def get_emissions_ledger_rpc(self, hotkey: str):
        """
        Get emissions ledger for a specific hotkey (RPC method).

        Args:
            hotkey: The miner's hotkey

        Returns:
            EmissionsLedger instance, or None if not found
        """
        return self._manager.get_emissions_ledger(hotkey)

    def get_all_emissions_ledgers_rpc(self):
        """
        Get all emissions ledgers (RPC method).

        Returns:
            Dict mapping hotkey to EmissionsLedger instance
        """
        return self._manager.get_all_emissions_ledgers()

    def set_emissions_ledger_rpc(self, hotkey: str, emissions_ledger):
        """
        Set emissions ledger for a specific hotkey (RPC method - test-only).

        Args:
            hotkey: The miner's hotkey
            emissions_ledger: EmissionsLedger instance
        """
        self._manager.emissions_ledger_manager.emissions_ledgers[hotkey] = emissions_ledger
        return True

    # ========================================================================
    # PENALTY LEDGER RPC METHODS (delegate to manager's sub-manager)
    # ========================================================================

    def get_penalty_ledger_rpc(self, hotkey: str):
        """
        Get penalty ledger for a specific hotkey (RPC method).

        Args:
            hotkey: The miner's hotkey

        Returns:
            PenaltyLedger instance, or None if not found
        """
        return self._manager.get_penalty_ledger(hotkey)

    def get_all_penalty_ledgers_rpc(self):
        """
        Get all penalty ledgers (RPC method).

        Returns:
            Dict mapping hotkey to PenaltyLedger instance
        """
        return self._manager.get_all_penalty_ledgers()

    def build_penalty_ledgers_rpc(self, verbose: bool = False, delta_update: bool = True):
        """
        Build penalty ledgers (RPC method for testing/manual use).

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints. If False, rebuild from scratch.
        """
        return self._manager.penalty_ledger_manager.build_penalty_ledgers(verbose=verbose, delta_update=delta_update)

    def build_emissions_ledgers_rpc(self, delta_update: bool = True):
        """
        Build emissions ledgers (RPC method for testing/manual use ONLY).

        IMPORTANT: This method will raise RuntimeError if called in production.
        Only available when running_unit_tests=True.

        Args:
            delta_update: If True, only process new data. If False, rebuild from scratch.

        Raises:
            RuntimeError: If called in production (running_unit_tests=False)
        """
        return self._manager.emissions_ledger_manager.build_emissions_ledgers(delta_update=delta_update)

    # ========================================================================
    # MANUAL BUILD (for testing/manual use)
    # ========================================================================

    def build_debt_ledgers(self, verbose: bool = False, delta_update: bool = True):
        """
        Build or update debt ledgers (delegates to manager).

        This method is exposed for manual/testing use.
        The daemon calls this automatically at regular intervals.

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints. If False, rebuild from scratch.
        """
        return self._manager.build_debt_ledgers(verbose=verbose, delta_update=delta_update)

    def build_debt_ledgers_rpc(self, verbose: bool = False, delta_update: bool = True):
        """
        RPC wrapper for build_debt_ledgers.

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints. If False, rebuild from scratch.
        """
        return self.build_debt_ledgers(verbose=verbose, delta_update=delta_update)
