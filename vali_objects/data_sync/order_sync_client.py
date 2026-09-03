# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
OrderSyncClient — cross-process drop-in for OrderSyncState (spec R2.1).

Exposes the SAME interface as OrderSyncState (is_sync_waiting / begin_order / wait_for_orders /
begin_sync / mark_sync_complete) but backed by CommonDataServer over RPC, so order processing
(vanta-orders, after R1) and position sync (core) coordinate ACROSS process boundaries. In a
single process, OrderSyncState is still used; this adapter is for the split.

Cutover (happens in R1, NOT here): vanta-orders instantiates this for the order side
(is_sync_waiting + begin_order); core's PositionSyncer instantiates it for the sync side
(wait_for_orders / begin_sync). Both sides MUST switch together — if only sync switched, it would
wait on a counter the order side never increments and miss in-flight orders (a correctness race).

Fail-open (spec R2.3): the ORDER side must never be blocked because coordination is unreachable.
If CommonDataServer is down (core restarting), sync cannot run anyway, so is_sync_waiting()
degrades to False and begin/end degrade to no-ops — orders keep flowing. The SYNC side does NOT
fail open: if it cannot confirm orders are quiesced it must not proceed with a checkpoint-wide
rewrite, so wait_for_orders errors propagate to the caller.
"""
import threading

import bittensor as bt

from shared_objects.rpc.common_data_client import CommonDataClient
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class OrderSyncClient:
    def __init__(
        self,
        common_data_client: CommonDataClient = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False,
    ):
        # Reuse a shared CommonDataClient if provided, else create one.
        self._client = common_data_client or CommonDataClient(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests,
        )

    # ==================== Order side (fail-open) ====================

    def is_sync_waiting(self) -> bool:
        """
        True if a sync is waiting for orders to drain. Fail-open: if coordination is unreachable
        (core down), returns False so orders are not blocked (sync cannot run while core is down).
        """
        try:
            return self._client.is_sync_waiting()
        except Exception as e:
            bt.logging.warning(f"OrderSyncClient.is_sync_waiting unreachable, assuming not waiting: {e}")
            return False

    def begin_order(self, label: str = None) -> "OrderSyncClient._OrderAdmission":
        """
        Atomically gate + register one order. Returns an _OrderAdmission context manager:
          - `admission.rejected is True`  => a sync is in progress; do NOT process the order,
            reject it with should_retry=True. (Nothing was registered; __exit__ is a no-op.)
          - `admission.rejected is False` => process normally; the registration is ended on exit.
        Fail-open (R2.3): if coordination is unreachable (core down), returns rejected=False,
        token=None — the order proceeds UNREGISTERED (sync can't run while core is down anyway).
        This distinguishes "syncing" (server returned None → reject) from "unreachable" (RPC raised
        → proceed), which a bare boolean could not.
        `label` (order_uuid + context) makes a reaped/stuck registration identifiable in a disaster.
        """
        try:
            token = self._client.begin_order(label)  # int => admitted, None => sync in progress
            if token is None:
                return self._OrderAdmission(self._client, token=None, rejected=True)
            return self._OrderAdmission(self._client, token=token, rejected=False)
        except Exception as e:
            bt.logging.warning(f"OrderSyncClient.begin_order failed, proceeding without sync gate: {e}")
            return self._OrderAdmission(self._client, token=None, rejected=False)

    def get_order_count(self) -> int:
        return self._client.get_order_count()

    # ==================== Sync side (fail-closed) ====================

    def wait_for_orders(self, timeout_seconds: float = None) -> bool:
        """Block until in-flight orders drain to 0; returns False on timeout. Errors propagate."""
        return self._client.wait_for_orders(timeout_seconds)

    def mark_sync_complete(self) -> None:
        self._client.mark_sync_complete()

    def begin_sync(self, timeout_seconds: float = None) -> "OrderSyncClient._SyncContext":
        return self._SyncContext(self._client, timeout_seconds)

    # ==================== Context managers ====================

    class _OrderAdmission:
        """
        Outcome of begin_order, usable as a context manager around order processing.
          .rejected: True => a sync is in progress; caller must reject (should_retry), skip work.
          .token:    the in-flight registration token (None if rejected or fail-open unregistered).
        __exit__ ends the registration (if any); the gate decision (whether to process) is the
        caller's, based on .rejected.
        """
        def __init__(self, client: CommonDataClient, token, rejected: bool):
            self.client = client
            self.token = token
            self.rejected = rejected

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.token is not None:
                try:
                    self.client.end_order(self.token)
                except Exception as e:
                    # If this fails (core down), the registration self-heals via the R2.4 TTL reap.
                    bt.logging.warning(f"OrderSyncClient.end_order failed (registration reaped after TTL): {e}")
            return False  # never suppress the order's own exception

    class _SyncContext:
        """
        Waits for orders to drain on enter (raising on timeout), clears sync_waiting on exit.

        While held, a daemon heartbeat thread renews the gate's server-side lease every
        SYNC_LEASE_RENEW_INTERVAL_S. The gate lives on CommonDataServer in another process; if
        THIS process (core) is hard-killed mid-sync, the heartbeat dies with it and the server
        auto-clears the gate after SYNC_WAITING_LEASE_MS — instead of rejecting every order
        network-wide until the next successful sync. Renewal errors are swallowed (a blip must
        not kill the sync); a dead server means orders aren't being gated anyway.
        """
        def __init__(self, client: CommonDataClient, timeout_seconds: float = None):
            self.client = client
            self.timeout_seconds = timeout_seconds
            self.acquired = False
            self._stop_heartbeat = threading.Event()
            self._heartbeat_thread = None

        def _heartbeat(self):
            while not self._stop_heartbeat.wait(ValiConfig.SYNC_LEASE_RENEW_INTERVAL_S):
                try:
                    if not self.client.renew_sync_lease():
                        bt.logging.error(
                            "OrderSyncClient: sync gate no longer held (lease expired or cleared) "
                            "— the in-progress sync has lost its exclusion window"
                        )
                        return
                except Exception as e:
                    bt.logging.warning(f"OrderSyncClient: sync-lease renewal failed (continuing): {e}")

        def __enter__(self):
            self.acquired = self.client.wait_for_orders(self.timeout_seconds)
            if not self.acquired:
                raise TimeoutError("Timeout waiting for orders to complete")
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat, name="sync-lease-heartbeat", daemon=True
            )
            self._heartbeat_thread.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._stop_heartbeat.set()
            self.client.mark_sync_complete()
            return False
