import bittensor as bt

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class DebtLedgerClient(RPCClientBase):
    """
    Lightweight RPC client for DebtLedgerServer.

    Can be created in ANY process. No server ownership.
    Forward compatibility - consumers create their own client instance.

    Example:
        client = DebtLedgerClient()
        ledgers = client.get_all_debt_ledgers()
    """

    def __init__(
            self,
            port: int = None,
            connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
            connect_immediately: bool = False,
            running_unit_tests: bool = False
    ):
        """
        Initialize DebtLedger client.

        Args:
            port: Port number of the DebtLedger server (default: ValiConfig.RPC_DEBTLEDGER_PORT)
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - bypass RPC, use set_direct_server()
                - RPC (1): Normal RPC mode - connect via network
            connect_immediately: If True, connect in __init__. If False, call connect() later.
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_DEBTLEDGER_SERVICE_NAME,
            port=port or ValiConfig.RPC_DEBTLEDGER_PORT,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Client Methods ====================

    def get_ledger(self, hotkey: str):
        """
        Get debt ledger for a specific hotkey.

        Args:
            hotkey: The miner's hotkey

        Returns:
            DebtLedger instance, or None if not found
        """
        try:
            return self._server.get_ledger_rpc(hotkey)
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get ledger failed: {e}")
            return None

    def get_compressed_summaries_rpc(self) -> bytes | None:
        """
        Get pre-compressed debt ledger summaries as gzip bytes from cache.

        Returns:
            Cached compressed gzip bytes of debt ledger summaries JSON
        """
        try:
            return self._server.get_compressed_summaries_rpc()
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get compressed summaries failed: {e}")
            return None

    def get_all_ledgers(self):
        """
        Get all debt ledgers.

        Returns:
            Dict mapping hotkey to DebtLedger instance
        """
        try:
            return self._server.get_all_ledgers_rpc()
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get all ledgers failed: {e}")
            return {}

    def get_all_debt_ledgers(self):
        """
        Get all debt ledgers (alias for get_all_ledgers for backward compatibility).

        Returns:
            Dict mapping hotkey to DebtLedger instance
        """
        return self.get_all_ledgers()

    def get_ledger_summary(self, hotkey: str):
        """
        Get summary stats for a specific ledger.

        Args:
            hotkey: The miner's hotkey

        Returns:
            Summary dict with cumulative stats and latest checkpoint
        """
        try:
            return self._server.get_ledger_summary_rpc(hotkey)
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get ledger summary failed: {e}")
            return None


    def get_all_summaries(self):
        """
        Get summary stats for all ledgers.

        Returns:
            Dict mapping hotkey to summary dict
        """
        try:
            return self._server.get_all_summaries_rpc()
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get all summaries failed: {e}")
            return {}

    def get_compressed_summaries(self):
        """
        Get pre-compressed debt ledger summaries as gzip bytes from cache.

        Returns:
            Cached compressed gzip bytes of debt ledger summaries JSON
        """
        try:
            return self._server.get_compressed_summaries_rpc()
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get compressed summaries failed: {e}")
            return None

    def health_check(self):
        """
        Health check endpoint for monitoring.

        Returns:
            dict: Health status, or None if server unavailable
        """
        try:
            return self._server.health_check_rpc()
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Health check failed: {e}")
            return None

    def build_debt_ledgers(self, verbose: bool = False, delta_update: bool = True):
        """
        Build or update debt ledgers (RPC method for testing/manual use).

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints. If False, rebuild from scratch.
        """
        try:
            return self._server.build_debt_ledgers_rpc(verbose=verbose, delta_update=delta_update)
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Build debt ledgers failed: {e}")
            return None

    # ==================== Emissions Ledger Methods ====================

    def get_emissions_ledger(self, hotkey: str):
        """
        Get emissions ledger for a specific hotkey.

        Args:
            hotkey: The miner's hotkey

        Returns:
            EmissionsLedger instance, or None if not found
        """
        try:
            return self._server.get_emissions_ledger_rpc(hotkey)
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get emissions ledger failed: {e}")
            return None

    def get_all_emissions_ledgers(self):
        """
        Get all emissions ledgers.

        Returns:
            Dict mapping hotkey to EmissionsLedger instance
        """
        try:
            return self._server.get_all_emissions_ledgers_rpc()
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get all emissions ledgers failed: {e}")
            return {}

    def set_emissions_ledger(self, hotkey: str, emissions_ledger):
        """
        Set emissions ledger for a specific hotkey (test-only).

        Args:
            hotkey: The miner's hotkey
            emissions_ledger: EmissionsLedger instance
        """
        try:
            return self._server.set_emissions_ledger_rpc(hotkey, emissions_ledger)
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Set emissions ledger failed: {e}")
            return None

    def build_emissions_ledgers(self, delta_update: bool = True):
        """
        Build emissions ledgers (RPC method for testing/manual use ONLY).

        IMPORTANT: This method will raise RuntimeError if called in production.
        Only available when running_unit_tests=True.

        Args:
            delta_update: If True, only process new data. If False, rebuild from scratch.

        Raises:
            RuntimeError: If called in production (running_unit_tests=False)
        """
        try:
            return self._server.build_emissions_ledgers_rpc(delta_update=delta_update)
        except Exception as e:
            bt.logging.error(f"DebtLedgerClient: Build emissions ledgers failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ==================== Penalty Ledger Methods ====================

    def get_penalty_ledger(self, hotkey: str):
        """
        Get penalty ledger for a specific hotkey.

        Args:
            hotkey: The miner's hotkey

        Returns:
            PenaltyLedger instance, or None if not found
        """
        try:
            return self._server.get_penalty_ledger_rpc(hotkey)
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get penalty ledger failed: {e}")
            return None

    def get_all_penalty_ledgers(self):
        """
        Get all penalty ledgers.

        Returns:
            Dict mapping hotkey to PenaltyLedger instance
        """
        try:
            return self._server.get_all_penalty_ledgers_rpc()
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Get all penalty ledgers failed: {e}")
            return {}

    def build_penalty_ledgers(self, verbose: bool = False, delta_update: bool = True):
        """
        Build penalty ledgers (RPC method for testing/manual use).

        Args:
            verbose: Enable detailed logging
            delta_update: If True, only process new checkpoints. If False, rebuild from scratch.
        """
        try:
            return self._server.build_penalty_ledgers_rpc(verbose=verbose, delta_update=delta_update)
        except Exception as e:
            bt.logging.debug(f"DebtLedgerClient: Build penalty ledgers failed: {e}")
            return None
