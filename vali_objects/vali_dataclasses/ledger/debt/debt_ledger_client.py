
from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from shared_objects.log import logger


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
            logger.debug(f"DebtLedgerClient: Get ledger failed: {e}")
            return None

    def get_dashboard(self, hotkey: str, checkpoints_time_ms: int) -> dict | None:
        return self._server.get_dashboard_rpc(hotkey, checkpoints_time_ms)

    def get_compressed_summaries_rpc(self) -> bytes | None:
        """
        Get pre-compressed debt ledger summaries as gzip bytes from cache.

        Returns:
            Cached compressed gzip bytes of debt ledger summaries JSON
        """
        try:
            return self._server.get_compressed_summaries_rpc()
        except Exception as e:
            logger.debug(f"DebtLedgerClient: Get compressed summaries failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get all ledgers failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get ledger summary failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get all summaries failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get compressed summaries failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Health check failed: {e}")
            return None

    def get_sealed_weeks(self, hotkey: str) -> dict:
        """
        Settled payout-week records for a hotkey, keyed by Monday 00:00 UTC.

        A sealed week is replayed verbatim by the payout paths instead of being recomputed, so a
        ledger rebuild cannot move it between paid and withheld. An empty dict on failure means the
        caller recomputes, which is the pre-seal behavior.

        Args:
            hotkey: The miner's hotkey

        Returns:
            Dict mapping week_start_ms to SealedWeek, empty on error
        """
        try:
            return self._server.get_sealed_weeks_rpc(hotkey)
        except Exception as e:
            logger.debug(f"DebtLedgerClient: Get sealed weeks failed: {e}")
            return {}

    def unseal_week(self, hotkey: str, week_start_ms: int) -> bool:
        """
        Drop one settled payout-week record so the next build reseals it.

        Args:
            hotkey: The miner's hotkey
            week_start_ms: Monday 00:00 UTC of the week to unseal

        Returns:
            True if a record was removed
        """
        try:
            return self._server.unseal_week_rpc(hotkey, week_start_ms)
        except Exception as e:
            logger.debug(f"DebtLedgerClient: Unseal week failed: {e}")
            return False

    def record_settled_segment(
        self,
        hotkey: str,
        week_start_ms: int,
        segment_start_ms: int,
        segment_end_ms: int,
        bucket: str,
        payout_usd: float,
        gross_payout_usd: float,
        weekly_penalty: float,
        payout_scale: float,
    ) -> bool:
        """
        Pin a payout settled early by an account switch.

        The caller runs this mid-wipe, so there is no second chance: everything this record
        describes is deleted moments later. A failure is therefore logged at error level rather
        than swallowed quietly like the reads on this client.

        Args:
            hotkey: The miner's hotkey
            week_start_ms: Monday 00:00 UTC of the week the segment falls in
            segment_start_ms: Start of the settled stretch
            segment_end_ms: The account switch; names the record for a later correction
            bucket: The bucket being wound down
            payout_usd: Settled payout, what both payout paths add
            gross_payout_usd: Pre-penalty payout, for audit
            weekly_penalty: The penalty applied
            payout_scale: The standard/pro ratio applied

        Returns:
            True if a new record was written, False if this week and bucket were already
            settled - a retried switch - or on error
        """
        try:
            return self._server.record_settled_segment_rpc(
                hotkey,
                week_start_ms,
                segment_start_ms,
                segment_end_ms,
                bucket,
                payout_usd,
                gross_payout_usd,
                weekly_penalty,
                payout_scale,
            )
        except Exception as e:
            logger.error(
                f"DebtLedgerClient: Record settled segment failed for {hotkey} "
                f"(${payout_usd:.2f} ending {segment_end_ms}): {e}"
            )
            return False

    def get_settled_segments(self, hotkey: str) -> list:
        """
        Segments settled early for a hotkey, oldest first.

        These are payouts whose source data was destroyed by an account switch, so the recorded
        dollars are all that remains of them.

        Args:
            hotkey: The miner's hotkey

        Returns:
            List of SettledSegment, empty on error
        """
        try:
            return self._server.get_settled_segments_rpc(hotkey)
        except Exception as e:
            logger.debug(f"DebtLedgerClient: Get settled segments failed: {e}")
            return []

    def amend_settled_segment(self, hotkey: str, segment_end_ms: int, payout_usd: float) -> bool:
        """
        Correct a settled segment's amount. Deliberate corrections only.

        Args:
            hotkey: The miner's hotkey
            segment_end_ms: The account switch that identifies the record
            payout_usd: The corrected payout

        Returns:
            True if a record was amended
        """
        try:
            return self._server.amend_settled_segment_rpc(hotkey, segment_end_ms, payout_usd)
        except Exception as e:
            logger.debug(f"DebtLedgerClient: Amend settled segment failed: {e}")
            return False

    def remove_settled_segment(self, hotkey: str, segment_end_ms: int) -> bool:
        """
        Drop a settled segment entirely. Deliberate corrections only.

        Args:
            hotkey: The miner's hotkey
            segment_end_ms: The account switch that identifies the record

        Returns:
            True if a record was removed
        """
        try:
            return self._server.remove_settled_segment_rpc(hotkey, segment_end_ms)
        except Exception as e:
            logger.debug(f"DebtLedgerClient: Remove settled segment failed: {e}")
            return False

    def get_weekly_seals_checkpoint_dict(self) -> dict:
        """
        Every sealed week and settled segment, JSON-ready, for the validator checkpoint.

        An empty dict on failure means the checkpoint ships without seal records, which peers
        treat as "nothing to merge" - the pre-autosync behavior.

        Returns:
            Dict with 'sealed' and 'settled' maps keyed by hotkey, empty on error
        """
        try:
            return self._server.get_weekly_seals_checkpoint_dict_rpc()
        except Exception as e:
            logger.warning(f"DebtLedgerClient: Get weekly seals checkpoint dict failed: {e}")
            return {}

    def sync_weekly_seals(self, weekly_seals_dict: dict) -> dict:
        """
        Merge a checkpoint's weekly seal records so validators agree on what was sealed.

        Args:
            weekly_seals_dict: Dict with 'sealed' and 'settled' maps from the checkpoint

        Returns:
            dict: Sync statistics, empty on error
        """
        try:
            return self._server.sync_weekly_seals_rpc(weekly_seals_dict)
        except Exception as e:
            logger.error(f"DebtLedgerClient: Sync weekly seals failed: {e}")
            return {}

    def clear_weekly_seals_for_test(self) -> bool:
        """
        Drop every seal record, in memory and on disk. Unit tests only.

        Returns:
            True once cleared
        """
        return self._server.clear_weekly_seals_for_test_rpc()

    def delete_debt_ledger(self, hotkey: str) -> bool:
        """
        Delete the debt ledger for a specific hotkey.

        Called on subaccount promotion so the funded period starts with a clean ledger.

        Args:
            hotkey: The miner's hotkey

        Returns:
            True if deleted, False if not found or on error
        """
        try:
            return self._server.delete_debt_ledger_rpc(hotkey)
        except Exception as e:
            logger.debug(f"DebtLedgerClient: Delete debt ledger failed: {e}")
            return False

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
            logger.debug(f"DebtLedgerClient: Build debt ledgers failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get emissions ledger failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get all emissions ledgers failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Set emissions ledger failed: {e}")
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
            logger.error(f"DebtLedgerClient: Build emissions ledgers failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get penalty ledger failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Get all penalty ledgers failed: {e}")
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
            logger.debug(f"DebtLedgerClient: Build penalty ledgers failed: {e}")
            return None
