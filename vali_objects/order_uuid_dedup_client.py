# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
OrderUuidDedupClient — cross-process/-instance order-UUID dedup (spec R2.6).

Successor to the process-local UUIDTracker. The AUTHORITATIVE claim (check_and_add) lives on
CommonDataServer, so dedup holds across a vanta-orders restart AND across multiple overlapping
instances (only one instance can claim a given uuid). A small LOCAL set caches known uuids to
short-circuit the obvious-duplicate early-reject WITHOUT an RPC — it is advisory only: a local
miss just means "ask the server," and the server's check_and_add is the real decision.

Intended R1 order-handler flow (replacing UUIDTracker's exists()/add()):
    if dedup.exists(uuid):                 # cheap local early-reject for obvious dupes
        reject("already processed")
    ... run business validation ...
    if not dedup.check_and_add(uuid):      # authoritative claim, right before applying
        reject("already processed")
    try:
        apply_order(...)                   # the state write
    except Exception:
        dedup.release(uuid)                # transient failure -> let the placer's retry re-claim
        raise
Claiming BEFORE the apply (not recording after) is what makes overlap safe; releasing on failure
is what keeps R4.1's transient-error retries from being lost. A falsy uuid is not dedup-able.
"""
from collections import deque

import bittensor as bt

from shared_objects.rpc.common_data_client import CommonDataClient
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class OrderUuidDedupClient:
    def __init__(
        self,
        common_data_client: CommonDataClient = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        running_unit_tests: bool = False,
    ):
        self._client = common_data_client or CommonDataClient(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests,
        )
        # Advisory fast-path cache; the server is the source of truth. FIFO-bounded (deque for
        # eviction order + set for O(1) membership), mirroring the server's dedup set, so a
        # long-lived vanta-orders process cannot leak one entry per order forever. A local miss is
        # harmless — the caller falls through to the authoritative check_and_add.
        self._local_capacity = ValiConfig.ORDER_UUID_DEDUP_CAPACITY
        self._local_order = deque()
        self._local = set()

    def _remember_local(self, uuid) -> None:
        """Record uuid in the bounded local cache with FIFO capacity eviction."""
        if not uuid or uuid in self._local:
            return
        if len(self._local_order) >= self._local_capacity:
            oldest = self._local_order.popleft()
            self._local.discard(oldest)
        self._local_order.append(uuid)
        self._local.add(uuid)

    def add_initial_uuids(self, hk_to_positions) -> None:
        """
        Seed from committed position history at boot (mirrors UUIDTracker.add_initial_uuids): push
        every order uuid to the authoritative server set and warm the local cache. Server seeding
        failure is non-fatal — the local cache still helps and check_and_add stays authoritative.
        """
        uuids = []
        try:
            for positions in hk_to_positions.values():
                for p in positions:
                    for o in p.orders:
                        if o.order_uuid:
                            uuids.append(o.order_uuid)
        except Exception as e:
            bt.logging.warning(f"OrderUuidDedupClient: failed to extract seed uuids: {e}")
            return
        try:
            total = self._client.seed_order_uuids(uuids)
            bt.logging.info(f"OrderUuidDedupClient: seeded {len(uuids)} order uuids (server set size {total})")
        except Exception as e:
            bt.logging.warning(f"OrderUuidDedupClient: server seed failed, using local cache only: {e}")
        for u in uuids:
            self._remember_local(u)

    def exists(self, uuid) -> bool:
        """
        Fast-path early-reject: True if this uuid is locally known to have been processed. A local
        miss returns False (the caller then proceeds to the authoritative check_and_add). Local-only
        by design — no RPC on the hot path for the common not-a-duplicate case.
        """
        if not uuid:
            return False
        return uuid in self._local

    def check_and_add(self, uuid) -> bool:
        """
        Authoritative atomic claim. True => newly claimed (apply the order); False => duplicate
        (reject). Warms the local cache either way (the uuid is now known). Falsy uuid => True
        (nothing to dedup on).
        """
        if not uuid:
            return True
        claimed = self._client.check_and_add_order_uuid(uuid)
        self._remember_local(uuid)
        return claimed

    def release(self, uuid) -> None:
        """Undo a claim after an apply failure (server-authoritative), and drop it from the cache."""
        if not uuid:
            return
        try:
            self._client.release_order_uuid(uuid)
        finally:
            # Lazy tombstone (mirrors the server): drop from the membership set only; the stale
            # deque entry is reclaimed on the normal FIFO eviction in _remember_local. O(1), and
            # keeps the deque bounded by capacity without an O(n) remove on the failure path.
            self._local.discard(uuid)
