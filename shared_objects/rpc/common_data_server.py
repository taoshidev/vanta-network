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
from collections import deque, namedtuple

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.rpc.rpc_server_base import RPCServerBase
from shared_objects.log import logger

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

        # Authoritative order-UUID dedup (spec R2.6) — replaces the process-local UUIDTracker so
        # dedup holds across a vanta-orders restart AND across multiple overlapping instances.
        # deque = FIFO eviction order; set = O(1) membership; capacity-bounded like UUIDTracker.
        self._order_uuids_deque = deque()
        self._order_uuids_set = set()
        self._order_uuid_capacity = ValiConfig.ORDER_UUID_DEDUP_CAPACITY
        # Two-phase claims: check_and_add records a PROVISIONAL claim (uuid -> claimed_at_ms);
        # confirm promotes it to the permanent FIFO set post-commit, release drops it on apply
        # failure. A claimant hard-killed mid-apply (SIGKILL/OOM — release never runs) leaves a
        # provisional claim that simply expires after ORDER_UUID_CLAIM_TTL_MS, so the miner's
        # retry of a never-committed order is not permanently rejected as a duplicate.
        self._order_uuid_provisional = {}   # uuid -> claimed_at_ms
        self._order_uuid_claim_ttl_ms = ValiConfig.ORDER_UUID_CLAIM_TTL_MS  # instance attr so tests can shrink it

        self._sync_waiting = False
        self._last_sync_start_ms = 0
        self._last_sync_complete_ms = 0
        self._order_condition = threading.Condition(self._state_lock)
        # INVARIANT (load-bearing): the condition MUST wrap _state_lock, not a lock of its own.
        # The R2.5a order/sync TOCTOU fix depends on begin_order_rpc's `_sync_waiting` check and
        # wait_for_orders_rpc's `_sync_waiting = True` running under the SAME mutex. A future edit
        # that gives the condition an independent lock would silently reopen that race. Use a real
        # raise (not assert) so the guard survives `python -O`, which strips asserts.
        if self._order_condition._lock is not self._state_lock:
            raise RuntimeError("_order_condition must share _state_lock (order/sync TOCTOU invariant)")

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
                logger.warning("[COMMON_DATA] Shutdown flag set")
            else:
                self._shutdown_dict.clear()
                logger.info("[COMMON_DATA] Shutdown flag cleared")

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
                logger.info(f"[COMMON_DATA] sync_in_progress: {old_value} -> {value}")

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
            logger.info(f"[COMMON_DATA] Incrementing sync epoch {old_epoch} -> {self._sync_epoch}")
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
            logger.warning(
                f"[COMMON_DATA] Reaped {len(stale_tokens)} abandoned in-flight order registration(s) "
                f"(older than {self._inflight_ttl_ms}ms, oldest ~{oldest_age_ms}ms) — producer likely "
                f"crashed/stalled mid-order. Sample of reaped orders: {sample}{truncated}"
            )
        return len(self._inflight_orders)

    def begin_order_rpc(self, label: str = None):
        """
        Atomically gate + register an in-flight order. Returns an opaque token to pass back to
        end_order_rpc, or **None if a sync is in progress** (sync_waiting set) — in which case the
        caller must REJECT the order (should_retry) and NOT process it.

        Returning None here is the fix for the check-then-register TOCTOU: an advisory
        is_sync_waiting() pre-check followed by a separate register has a gap in which a sync can
        start and reach its position rewrite before the order increments the count. Because setting
        sync_waiting (wait_for_orders_rpc) and this check share _order_condition, an order either
        registers BEFORE sync (so wait_for_orders waits for it) or is refused here — it can never
        slip in and apply during the rewrite window (sync_waiting stays set until mark_sync_complete).

        `label` is a free-form debug string (order_uuid + context) surfaced on reap and via
        get_order_sync_state_rpc. Reaps on every call so the map stays bounded by ~(TTL ×
        arrival-rate) even during an inflow-with-no-drain stall. If the live count crosses the soft
        cap, emit a throttled alert — but do NOT evict live entries (would undercount).
        """
        with self._order_condition:
            count = self._reap_and_count_locked()
            if self._sync_waiting:
                return None  # sync quiescing/running — refuse; caller rejects with should_retry
            if count >= self._INFLIGHT_SOFT_CAP:
                now_ms = TimeUtil.now_in_millis()
                if now_ms - self._last_backlog_warn_ms > self._BACKLOG_WARN_THROTTLE_MS:
                    self._last_backlog_warn_ms = now_ms
                    logger.error(
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

    # ============ Order UUID Dedup RPC Methods (authoritative, cross-process/-instance, R2.6) ============

    def _add_order_uuid_locked(self, uuid) -> None:
        """Add uuid to the dedup set with FIFO capacity eviction. Caller holds _state_lock.

        Evicting the oldest deque entry `discard`s it from the set (a no-op if that entry was
        already released — a lazy tombstone, see release_order_uuid_rpc), which reclaims tombstoned
        slots on the normal FIFO cycle.
        """
        if uuid in self._order_uuids_set:
            return
        if len(self._order_uuids_deque) >= self._order_uuid_capacity:
            oldest = self._order_uuids_deque.popleft()
            self._order_uuids_set.discard(oldest)
        self._order_uuids_deque.append(uuid)
        self._order_uuids_set.add(uuid)

    def check_and_add_order_uuid_rpc(self, uuid) -> bool:
        """
        Atomic claim. Returns True if `uuid` was NOT already present (now claimed — the caller may
        apply the order) or False if it was (duplicate — reject). This is the authoritative dedup
        that replaces the process-local UUIDTracker; being server-side, one claim wins even across
        two overlapping vanta-orders instances or a placer retry.

        Call BEFORE applying the order. On apply FAILURE call release_order_uuid_rpc, else a
        transient failure (the R4.1 retry case) would permanently block the retry and LOSE the
        order. A falsy uuid is not dedup-able -> returns True without recording (nothing to claim).
        """
        if not uuid:
            return True
        now_ms = TimeUtil.now_in_millis()
        with self._state_lock:
            if uuid in self._order_uuids_set:
                return False
            claimed_at = self._order_uuid_provisional.get(uuid)
            if claimed_at is not None:
                if now_ms - claimed_at <= self._order_uuid_claim_ttl_ms:
                    return False  # live claim held by another (or a hung) apply
                # Expired provisional: the claimant died mid-apply without release; let this
                # retry re-claim rather than rejecting an order that never committed.
                logger.warning(
                    f"[COMMON_DATA] Reclaiming expired provisional order-uuid claim {uuid} "
                    f"(claimed {(now_ms - claimed_at) / 1000:.0f}s ago, ttl {self._order_uuid_claim_ttl_ms / 1000:.0f}s)"
                )
            self._order_uuid_provisional[uuid] = now_ms
            return True

    def confirm_order_uuid_rpc(self, uuid) -> None:
        """
        Promote a provisional claim to the permanent dedup set — call after the order COMMITTED.
        A confirmed uuid never expires via the claim TTL (only via FIFO capacity eviction), so a
        late duplicate of a committed order is still rejected. No-op-safe: confirming an unknown
        uuid just records it (covers a confirm racing the TTL reap).
        """
        if not uuid:
            return
        with self._state_lock:
            self._order_uuid_provisional.pop(uuid, None)
            self._add_order_uuid_locked(uuid)

    def release_order_uuid_rpc(self, uuid) -> None:
        """
        Undo a claim (call when the order apply FAILED after a successful check_and_add) so the
        placer's retry can re-claim and succeed. No-op if absent.

        Lazy tombstone: drop `uuid` from the authoritative membership set (O(1)) and leave its stale
        deque entry in place — it is reclaimed on the normal FIFO eviction in _add_order_uuid_locked
        (which `discard`s the popped uuid, a no-op once tombstoned). This avoids an O(n) deque
        rebuild on a path that clusters precisely during deploys (the scenario this dedup serves).
        The membership set — not the deque — is authoritative for check_and_add/exists, so a
        tombstoned uuid is immediately re-claimable.

        Precise semantics of a released-then-RE-CLAIMED uuid: the re-claim appends a second deque
        entry (the tombstone is not removed), so the uuid appears twice. When the stale (first)
        entry later reaches the front, its `discard` drops the uuid from the set even though the
        re-claim is still logically live — i.e. a re-claimed uuid can be evicted EARLIER than the
        full capacity window (and the duplicate entry costs one slot until it too cycles out). This
        is safe because dedup only needs to cover the retry/replay window (seconds–minutes) and a
        placer retry re-claims within seconds, long before ~100k later insertions cycle the stale
        entry out. If strict claimed-for-capacity is ever required, carry a tombstone set and skip
        the re-append instead.
        """
        if not uuid:
            return
        with self._state_lock:
            self._order_uuid_provisional.pop(uuid, None)
            self._order_uuids_set.discard(uuid)

    def order_uuid_exists_rpc(self, uuid) -> bool:
        """Read-only membership check (early-reject fast path also served by the client-local cache)."""
        if not uuid:
            return False
        now_ms = TimeUtil.now_in_millis()
        with self._state_lock:
            if uuid in self._order_uuids_set:
                return True
            claimed_at = self._order_uuid_provisional.get(uuid)
            return claimed_at is not None and now_ms - claimed_at <= self._order_uuid_claim_ttl_ms

    def seed_order_uuids_rpc(self, uuids) -> int:
        """
        Bulk-add uuids from committed position history at boot (the rebuild UUIDTracker did at
        validator.py:233). CommonDataServer holds no position client, so the boot sequence extracts
        the uuids and pushes them in here. Returns the resulting set size.
        """
        with self._state_lock:
            for u in uuids:
                if u:
                    self._add_order_uuid_locked(u)
            return len(self._order_uuids_set)

    def order_uuid_count_rpc(self) -> int:
        """Current number of tracked order uuids (debug/health)."""
        with self._state_lock:
            return len(self._order_uuids_set)

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
            # Reset the order-UUID dedup set (else dedup state leaks across tests).
            self._order_uuids_deque.clear()
            self._order_uuids_set.clear()
            self._order_uuid_provisional.clear()
            logger.debug("[COMMON_DATA] Test state cleared (shutdown, sync_in_progress, sync_epoch, order coordination reset)")

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
                "sync_waiting": self._sync_waiting,
                "order_uuid_count": len(self._order_uuids_set)
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

    logger.info(f"CommonDataServer ready on port {ValiConfig.RPC_COMMONDATA_PORT}")

    if server_ready:
        server_ready.set()

    # Block until shutdown
    while not server.is_shutdown_rpc():
        time.sleep(1)

    server.shutdown()
    logger.info("CommonDataServer process exiting")
