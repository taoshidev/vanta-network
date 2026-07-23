# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
CommonDataServer - Centralized RPC server for shared validator state.

This server manages cross-process shared data that was previously managed via IPC Manager:
- shutdown_dict: Global shutdown flag for graceful termination
- sync_in_progress: Flag to pause daemon processes during position sync
- sync_epoch: Counter incremented each sync cycle to detect stale data

Architecture:
- CommonDataServer: RPC server that manages shared state (runs in validator process)
- CommonDataClient: Lightweight RPC client for consumers to access/modify state

Forward Compatibility Pattern:
All consumers create their own CommonDataClient internally:
    self._common_data_client = CommonDataClient(connection_mode=connection_mode)

Usage in validator.py:
    # Start CommonDataServer early in initialization
    self.common_data_server = CommonDataServer(
        slack_notifier=self.slack_notifier,
        start_server=True,
        connection_mode=RPCConnectionMode.RPC
    )

    # Pass to consumers (they create their own clients internally)
    self.elimination_server = EliminationServer(connection_mode=RPCConnectionMode.RPC)
    # EliminationServer creates its own CommonDataClient internally

Usage in consumers:
    class EliminationServer(RPCServerBase):
        def __init__(self, ..., connection_mode=RPCConnectionMode.RPC):
            # Forward compatibility: create own CommonDataClient
            self._common_data_client = CommonDataClient(
                connection_mode=connection_mode
            )

        @property
        def shutdown_dict(self):
            return self._common_data_client.get_shutdown_dict()

        @property
        def sync_in_progress(self):
            return self._common_data_client.get_sync_in_progress()
"""
import threading
import time
from collections import namedtuple
import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc.rpc_server_base import RPCServerBase

# One in-flight order registration. `label` is a free-form debug string (the caller passes the
# order_uuid + context, e.g. "<hotkey>:<trade_pair>:<uuid>") so a reaped/stuck registration is
# identifiable in a disaster — a bare token tells you a count is stuck but not WHICH order.
_InflightOrder = namedtuple("_InflightOrder", ["registered_at_ms", "label"])


class CommonDataServer(RPCServerBase):
    """
    RPC server for shared validator state management.

    Manages:
    - shutdown_dict: Global shutdown flag (dict used as truthy check)
    - sync_in_progress: Boolean flag for sync state
    - sync_epoch: Integer counter for sync cycles

    Inherits from RPCServerBase for RPC server lifecycle.
    No daemon needed - this is a simple state server.
    """
    service_name = ValiConfig.RPC_COMMONDATA_SERVICE_NAME
    service_port = ValiConfig.RPC_COMMONDATA_PORT

    def __init__(
        self,
        slack_notifier=None,
        start_server: bool = True,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize CommonDataServer.

        Args:
            slack_notifier: Optional SlackNotifier for alerts
            start_server: Whether to start RPC server immediately
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        # Initialize shared state
        self.running_unit_tests = running_unit_tests
        self._shutdown_dict = {}
        self._sync_in_progress = False
        self._sync_epoch = 0
        self._state_lock = threading.Lock()

        # Cross-process order/sync coordination (spec R2.1/R2.4) — the RPC port of OrderSyncState.
        # _order_condition shares _state_lock, so order-coordination state and the
        # shutdown/sync_in_progress/sync_epoch state above are all guarded by ONE lock. The
        # condition's wait/notify works across the server's per-connection worker threads exactly
        # like the in-memory threading.Condition did within one process.
        #
        # In-flight orders are tracked as {token: registered_at_ms} rather than a bare count so a
        # producer that crashes mid-order (never calls end_order) cannot block sync forever: any
        # registration older than _inflight_ttl_ms is reaped (R2.4). The in-memory OrderSyncState
        # got this for free from process death; across processes we must reap explicitly.
        self._inflight_orders = {}          # token(int) -> _InflightOrder(registered_at_ms, label)
        self._next_order_token = 0
        self._inflight_ttl_ms = ValiConfig.ORDER_INFLIGHT_TTL_MS  # instance attr so tests can shrink it
        self._last_backlog_warn_ms = 0      # throttle for the soft-cap backlog alert
        self._sync_waiting = False
        self._last_sync_start_ms = 0
        self._last_sync_complete_ms = 0
        self._order_condition = threading.Condition(self._state_lock)

        # Initialize RPCServerBase (no daemon needed for this simple state server)
        super().__init__(
            service_name=ValiConfig.RPC_COMMONDATA_SERVICE_NAME,
            port=ValiConfig.RPC_COMMONDATA_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # No daemon needed
            connection_mode=connection_mode
        )

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """No daemon needed for this simple state server."""
        pass

    # ==================== Shutdown Dict RPC Methods ====================

    def get_shutdown_dict_rpc(self) -> dict:
        """Get the shutdown dict (truthy if shutting down)."""
        with self._state_lock:
            return dict(self._shutdown_dict)

    def is_shutdown_rpc(self) -> bool:
        """Check if shutdown is in progress (bool for easier use)."""
        with self._state_lock:
            return bool(self._shutdown_dict)

    def set_shutdown_rpc(self, value: bool = True) -> None:
        """
        Set shutdown state.

        Args:
            value: If True, sets shutdown_dict[True] = True (triggers shutdown)
                   If False, clears shutdown_dict
        """
        with self._state_lock:
            if value:
                self._shutdown_dict[True] = True
                bt.logging.warning("[COMMON_DATA] Shutdown flag set")
            else:
                self._shutdown_dict.clear()
                bt.logging.info("[COMMON_DATA] Shutdown flag cleared")

    # ==================== Sync In Progress RPC Methods ====================

    def get_sync_in_progress_rpc(self) -> bool:
        """Get sync_in_progress flag."""
        with self._state_lock:
            return self._sync_in_progress

    def set_sync_in_progress_rpc(self, value: bool) -> None:
        """Set sync_in_progress flag."""
        with self._state_lock:
            old_value = self._sync_in_progress
            self._sync_in_progress = value
            if old_value != value:
                bt.logging.info(f"[COMMON_DATA] sync_in_progress: {old_value} -> {value}")

    # ==================== Sync Epoch RPC Methods ====================

    def get_sync_epoch_rpc(self) -> int:
        """Get current sync epoch."""
        with self._state_lock:
            return self._sync_epoch

    def increment_sync_epoch_rpc(self) -> int:
        """
        Increment sync epoch and return new value.

        Returns:
            New sync epoch value after increment
        """
        with self._state_lock:
            old_epoch = self._sync_epoch
            self._sync_epoch += 1
            bt.logging.info(f"[COMMON_DATA] Incrementing sync epoch {old_epoch} -> {self._sync_epoch}")
            return self._sync_epoch

    def set_sync_epoch_rpc(self, value: int) -> None:
        """Set sync epoch to specific value."""
        with self._state_lock:
            self._sync_epoch = value

    # ============ Order/Sync Coordination RPC Methods (cross-process OrderSyncState, R2.1) ============

    # Server-side poll cap for wait_for_orders: an abandoned registration expires by WALL CLOCK,
    # not by an end_order notify, so the waiter must re-evaluate periodically to reap it even when
    # no end_order ever arrives. Small enough to be responsive, large enough to stay cheap (sync
    # is infrequent).
    _WAIT_POLL_SECONDS = 1.0
    _REAP_LOG_SAMPLE = 10       # max labels to log when reaping (avoid a giant log line)
    _IN_FLIGHT_SAMPLE = 50      # max entries returned by get_order_sync_state (avoid a giant payload)
    _INFLIGHT_SOFT_CAP = 10_000  # live count above this = "orders not draining" alert (no eviction)
    _BACKLOG_WARN_THROTTLE_MS = 30_000

    def _reap_and_count_locked(self) -> int:
        """
        Drop in-flight registrations older than the TTL and return the live count.
        MUST be called with _order_condition/_state_lock held.

        Registrations are inserted in non-decreasing timestamp order (monotonic clock, insertion
        under the lock) and dict preserves insertion order, so all expired entries form a PREFIX.
        We walk from the front and stop at the first live entry — O(reaped), not O(total in-flight)
        — which makes it cheap enough to call on the hot path (every begin_order).
        """
        cutoff = TimeUtil.now_in_millis() - self._inflight_ttl_ms
        stale_tokens = []
        for tok, e in self._inflight_orders.items():
            if e.registered_at_ms <= cutoff:
                stale_tokens.append(tok)
            else:
                break  # everything after this is newer (insertion == time order)
        if stale_tokens:
            # Log a COUNT + oldest age + a bounded SAMPLE of labels — never the full list (a
            # backlog of thousands must not produce a multi-MB log line). The sample is the oldest
            # (most-likely-stuck) orders, the disaster breadcrumb for an operator.
            oldest_age_ms = TimeUtil.now_in_millis() - self._inflight_orders[stale_tokens[0]].registered_at_ms
            sample = [self._inflight_orders[t].label for t in stale_tokens[:self._REAP_LOG_SAMPLE]]
            for tok in stale_tokens:
                del self._inflight_orders[tok]
            truncated = " (…truncated)" if len(stale_tokens) > self._REAP_LOG_SAMPLE else ""
            bt.logging.warning(
                f"[COMMON_DATA] Reaped {len(stale_tokens)} abandoned in-flight order registration(s) "
                f"(older than {self._inflight_ttl_ms}ms, oldest ~{oldest_age_ms}ms) — producer likely "
                f"crashed/stalled mid-order. Sample of reaped orders: {sample}{truncated}"
            )
        return len(self._inflight_orders)

    def begin_order_rpc(self, label: str = None) -> int:
        """
        Register an in-flight order. Returns an opaque token to pass back to end_order_rpc.
        `label` is a free-form debug string (caller passes order_uuid + context) surfaced on reap
        and via get_order_sync_state_rpc, so a stuck/abandoned order is identifiable.

        Reaps on every call so the map stays bounded by ~(TTL × arrival-rate) even during an
        inflow-with-no-drain stall (nothing else may be calling end/wait/count). If the live count
        crosses the soft cap, emit a throttled alert — orders aren't draining — but do NOT evict
        live entries (that would undercount and let sync rewrite positions mid-order).
        """
        with self._order_condition:
            count = self._reap_and_count_locked()
            if count >= self._INFLIGHT_SOFT_CAP:
                now_ms = TimeUtil.now_in_millis()
                if now_ms - self._last_backlog_warn_ms > self._BACKLOG_WARN_THROTTLE_MS:
                    self._last_backlog_warn_ms = now_ms
                    bt.logging.error(
                        f"[COMMON_DATA] Abnormal in-flight order backlog: {count} live registrations "
                        f"(soft cap {self._INFLIGHT_SOFT_CAP}). Orders are not draining — the state "
                        f"tier is likely slow/down. Not evicting (would corrupt sync coordination)."
                    )
            self._next_order_token += 1
            token = self._next_order_token
            self._inflight_orders[token] = _InflightOrder(TimeUtil.now_in_millis(), label)
            return token

    def end_order_rpc(self, token: int = None) -> int:
        """
        Deregister the in-flight order identified by `token` and wake any sync waiter when the live
        count reaches 0. Returns the new live count. Unknown/None token is a no-op (idempotent —
        e.g. the entry was already reaped after a crash-and-restart, or a spurious end).
        """
        with self._order_condition:
            if token is not None:
                self._inflight_orders.pop(token, None)
            count = self._reap_and_count_locked()
            if count == 0:
                self._order_condition.notify_all()
            return count

    def get_order_count_rpc(self) -> int:
        """Current number of live (non-expired) in-flight orders."""
        with self._order_condition:
            return self._reap_and_count_locked()

    def is_sync_waiting_rpc(self) -> bool:
        """True if a sync is waiting for orders to drain (order early-reject reads this)."""
        with self._order_condition:
            return self._sync_waiting

    def set_sync_waiting_rpc(self, value: bool) -> None:
        """Set the sync_waiting flag directly (mark_sync_complete_rpc clears it)."""
        with self._order_condition:
            self._sync_waiting = value

    def wait_for_orders_rpc(self, timeout_seconds: float = None) -> bool:
        """
        Set sync_waiting=True, then block (server-side) until in-flight orders drain to 0.
        Returns True if drained, False on timeout (sync_waiting is cleared on timeout). Called by
        PositionSyncer (core) before a sync. Stale registrations are reaped on each poll, so a
        crashed producer cannot block indefinitely even with timeout_seconds=None (worst case is
        one TTL window). A caller-supplied timeout still bounds the total wait.
        """
        with self._order_condition:
            self._sync_waiting = True
            self._last_sync_start_ms = TimeUtil.now_in_millis()
            deadline_ms = None if timeout_seconds is None else self._last_sync_start_ms + int(timeout_seconds * 1000)
            while self._reap_and_count_locked() > 0:
                if deadline_ms is not None:
                    remaining_s = (deadline_ms - TimeUtil.now_in_millis()) / 1000.0
                    if remaining_s <= 0:
                        self._sync_waiting = False
                        return False
                    self._order_condition.wait(timeout=min(self._WAIT_POLL_SECONDS, remaining_s))
                else:
                    self._order_condition.wait(timeout=self._WAIT_POLL_SECONDS)
            return True

    def mark_sync_complete_rpc(self) -> None:
        """Clear sync_waiting (sync finished; orders may resume)."""
        with self._order_condition:
            self._sync_waiting = False
            self._last_sync_complete_ms = TimeUtil.now_in_millis()

    def get_order_sync_state_rpc(self) -> dict:
        """
        Snapshot of the order/sync coordination state (for logging/debug). `in_flight_sample` lists
        up to _IN_FLIGHT_SAMPLE live registrations (oldest first — the most-likely-stuck) with
        their debug label and age; `in_flight_count` is the true total. Query during an incident to
        see what is holding sync open. The sample is bounded so this call can't return a huge
        payload during a backlog.
        """
        with self._order_condition:
            count = self._reap_and_count_locked()
            now_ms = TimeUtil.now_in_millis()
            # dict is in insertion (== time) order, so the first N are the oldest.
            sample = []
            for e in self._inflight_orders.values():
                if len(sample) >= self._IN_FLIGHT_SAMPLE:
                    break
                sample.append({"label": e.label, "age_ms": now_ms - e.registered_at_ms})
            return {
                "n_orders_being_processed": count,
                "sync_waiting": self._sync_waiting,
                "last_sync_start_ms": self._last_sync_start_ms,
                "last_sync_complete_ms": self._last_sync_complete_ms,
                "in_flight_count": count,
                "in_flight_sample": sample,
                "in_flight_sample_truncated": count > self._IN_FLIGHT_SAMPLE,
            }

    # ==================== Test State Cleanup ====================

    def clear_test_state_rpc(self) -> None:
        """
        Clear ALL test-sensitive state (for test isolation).

        This includes:
        - shutdown_dict (prevents false shutdown in tests)
        - sync_in_progress (prevents daemons from incorrectly pausing)
        - sync_epoch (resets stale data detection counter)

        Should be called by ServerOrchestrator.clear_all_test_data() to ensure
        complete test isolation when servers are shared across tests.
        """
        with self._state_lock:
            self._shutdown_dict.clear()
            self._sync_in_progress = False
            self._sync_epoch = 0
            # Reset order/sync coordination too, else in-flight registrations leak across tests.
            self._inflight_orders.clear()
            self._next_order_token = 0
            self._sync_waiting = False
            self._last_sync_start_ms = 0
            self._last_sync_complete_ms = 0
            bt.logging.debug("[COMMON_DATA] Test state cleared (shutdown, sync_in_progress, sync_epoch, order coordination reset)")

    # ==================== Combined State RPC Methods ====================

    def get_all_state_rpc(self) -> dict:
        """
        Get all shared state in a single RPC call (reduces round trips).

        Returns:
            dict with keys: shutdown_dict, sync_in_progress, sync_epoch
        """
        with self._state_lock:
            return {
                "shutdown_dict": dict(self._shutdown_dict),
                "sync_in_progress": self._sync_in_progress,
                "sync_epoch": self._sync_epoch,
                "n_orders_being_processed": self._reap_and_count_locked(),
                "sync_waiting": self._sync_waiting,
                "timestamp_ms": TimeUtil.now_in_millis()
            }

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        with self._state_lock:
            return {
                "is_shutdown": bool(self._shutdown_dict),
                "sync_in_progress": self._sync_in_progress,
                "sync_epoch": self._sync_epoch,
                "n_orders_being_processed": self._reap_and_count_locked(),
                "sync_waiting": self._sync_waiting
            }


# ==================== Server Entry Point ====================

def start_common_data_server(
    slack_notifier=None,
    server_ready=None
):
    """
    Entry point for starting CommonDataServer in a separate process.

    Args:
        slack_notifier: Optional SlackNotifier for alerts
        server_ready: Event to signal when server is ready
    """
    from setproctitle import setproctitle
    setproctitle("vali_CommonDataServerProcess")

    # Create server
    server = CommonDataServer(
        slack_notifier=slack_notifier,
        start_server=True,
        connection_mode=RPCConnectionMode.RPC
    )

    bt.logging.success(f"CommonDataServer ready on port {ValiConfig.RPC_COMMONDATA_PORT}")

    if server_ready:
        server_ready.set()

    # Block until shutdown
    while not server.is_shutdown_rpc():
        time.sleep(1)

    server.shutdown()
    bt.logging.info("CommonDataServer process exiting")
