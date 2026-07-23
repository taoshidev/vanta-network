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
import bittensor as bt

from shared_objects.rpc.common_data_client import CommonDataClient
from vali_objects.vali_config import RPCConnectionMode


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

    def begin_order(self, label: str = None) -> "OrderSyncClient._OrderContext":
        """
        Context manager bracketing one order; fail-open on registration (see class docstring).
        `label` (order_uuid + context) makes a reaped/stuck registration identifiable in a disaster.
        """
        return self._OrderContext(self._client, label)

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

    class _OrderContext:
        """Registers the order on enter, deregisters (by token) on exit. Fail-open on enter."""
        def __init__(self, client: CommonDataClient, label: str = None):
            self.client = client
            self.label = label
            self.token = None

        def __enter__(self):
            try:
                self.token = self.client.begin_order(self.label)
            except Exception as e:
                # R2.3: coordination unavailable must NOT block the order. Proceed unregistered.
                bt.logging.warning(f"OrderSyncClient.begin_order failed, proceeding without sync gate: {e}")
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
        """Waits for orders to drain on enter (raising on timeout), clears sync_waiting on exit."""
        def __init__(self, client: CommonDataClient, timeout_seconds: float = None):
            self.client = client
            self.timeout_seconds = timeout_seconds
            self.acquired = False

        def __enter__(self):
            self.acquired = self.client.wait_for_orders(self.timeout_seconds)
            if not self.acquired:
                raise TimeoutError("Timeout waiting for orders to complete")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.client.mark_sync_complete()
            return False
