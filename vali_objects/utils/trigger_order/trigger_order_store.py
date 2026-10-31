"""
Open trigger orders: in memory, on disk, and copied onto their bound position.

Every change to the set of open orders goes through this class, so memory, disk and the position
copy can't drift apart. Only open orders are held in memory. Fired orders are dropped (the fill
lives on the Position), along with the rest of their oco_group; cancelled orders are written to
closed/. A bound oco_group is copied onto its position as one legacy bracket, keyed by the group.

Disk layout: {miner_dir}{hotkey}/trigger_orders/{trade_pair_id}/{open|closed}/{order_uuid}
"""
import json
import os
import threading

from shared_objects.log import logger
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.utils.trigger_order.legacy_converter import legacy_uuid, to_bracket_dict
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.trigger_order import TriggerOrder


class TriggerOrderStore:
    def __init__(self, running_unit_tests=False, position_client=None):
        self.running_unit_tests = running_unit_tests
        self._position_client = position_client  # None disables position copies
        self._orders: dict[TradePair, dict[str, dict[str, TriggerOrder]]] = {}
        self._where: dict[tuple[str, str], TradePair] = {}  # (hotkey, uuid) -> trade pair
        self._locks: dict[tuple[str, str], threading.RLock] = {}
        self._load_from_disk()

    def lock(self, hotkey: str, trade_pair: TradePair) -> threading.RLock:
        """Per (hotkey, trade pair) lock. Callers hold it across read-check-write sequences."""
        return self._locks.setdefault((hotkey, trade_pair.trade_pair_id), threading.RLock())

    # ==================== Reads ====================

    def get(self, hotkey: str, order_uuid: str) -> TriggerOrder | None:
        trade_pair = self._where.get((hotkey, order_uuid))
        if trade_pair is None:
            return None
        return self._orders.get(trade_pair, {}).get(hotkey, {}).get(order_uuid)

    def select(self, hotkey: str | None = None, trade_pair: TradePair | None = None) -> list[TriggerOrder]:
        """Open orders matching the filters, oldest first."""
        trade_pairs = [trade_pair] if trade_pair else list(self._orders)
        orders = []
        for tp in trade_pairs:
            by_hotkey = self._orders.get(tp, {})
            hotkeys = [hotkey] if hotkey else list(by_hotkey)
            for hk in hotkeys:
                orders.extend(list(by_hotkey.get(hk, {}).values()))
        return sorted(orders, key=lambda o: o.processed_ms)

    def trade_pairs(self) -> list[TradePair]:
        return [tp for tp, by_hotkey in list(self._orders.items()) if any(by_hotkey.values())]

    def hotkeys(self, trade_pair: TradePair) -> list[str]:
        return [hk for hk, orders in list(self._orders.get(trade_pair, {}).items()) if orders]

    def read_closed(self, hotkey: str) -> list[TriggerOrder]:
        return self._read_dir(hotkey, "closed")

    # ==================== Writes ====================

    def add(self, order: TriggerOrder) -> bool:
        """Insert an open order. Returns False if the uuid is already open (replay)."""
        with self.lock(order.miner_hotkey, order.trade_pair):
            if self.get(order.miner_hotkey, order.order_uuid):
                return False
            self._put(order)
            return True

    def replace(self, order: TriggerOrder, copy_to_position: bool = True) -> None:
        """Overwrite an open order with the same uuid and trade pair."""
        with self.lock(order.miner_hotkey, order.trade_pair):
            existing = self.get(order.miner_hotkey, order.order_uuid)
            if existing is None or existing.trade_pair != order.trade_pair:
                raise KeyError(f"No open trigger order [{order.order_uuid}] on {order.trade_pair.trade_pair_id}")
            self._put(order, copy_to_position)

    def remove(self, hotkey: str, order_uuid: str, closed_ms: int, reason: OrderSource) -> TriggerOrder | None:
        """Cancel an open order and write it to closed/."""
        order = self.get(hotkey, order_uuid)
        if order is None:
            return None
        with self.lock(hotkey, order.trade_pair):
            if self._pop(order) is None:
                return None
            order.closed_ms = closed_ms
            order.close_reason = reason
            self._write(order, "closed")
            self._sync_position(order)
        logger.info(f"[TRIGGER] Cancelled [{order_uuid}] {order.trade_pair.trade_pair_id} for [{hotkey}]: {reason.name}")
        return order

    def discard(self, hotkey: str, order_uuid: str) -> TriggerOrder | None:
        """Remove an open order without a closed record: replaced by an edit, or deleted."""
        order = self.get(hotkey, order_uuid)
        if order is None:
            return None
        with self.lock(hotkey, order.trade_pair):
            if self._pop(order) is None:
                return None
            self._sync_position(order)
        return order

    def fire(self, order: TriggerOrder, children: list[TriggerOrder], fired_ms: int) -> None:
        """Close a fired order, drop the rest of its oco_group, and open its children in one step."""
        with self.lock(order.miner_hotkey, order.trade_pair):
            siblings = [o for o in self._group(order) if o.order_uuid != order.order_uuid]
            if self._pop(order) is None:
                raise KeyError(f"Trigger order [{order.order_uuid}] is no longer open")
            order.closed_ms = fired_ms
            for sibling in siblings:
                self._pop(sibling)
            self._sync_position(order)
            if siblings:
                logger.info(f"[TRIGGER] [{order.order_uuid}] fired, dropped oco_group {order.oco_group} {[o.order_uuid for o in siblings]}")
            for child in children:
                if not self.add(child):
                    logger.warning(f"[TRIGGER] Child [{child.order_uuid}] already open, skipping")

    def reopen(self, hotkey: str, reason: OrderSource) -> int:
        """Reopen orders that were cancelled with reason. Returns the number reopened."""
        reopened = 0
        for order in self.read_closed(hotkey):
            if order.close_reason != reason:
                continue
            with self.lock(hotkey, order.trade_pair):
                self._delete(order, "closed")
                order.closed_ms = None
                order.close_reason = None
                if self.add(order):
                    reopened += 1
        return reopened

    def load(self, orders: list[TriggerOrder]) -> None:
        """
        Write orders as they are, overwriting any with the same uuid: cancelled ones (close_reason
        set) to closed/, the rest as open. Used by sync, migration and tests.
        """
        for order in orders:
            if order.close_reason is not None:
                self._write(order, "closed")
                continue
            with self.lock(order.miner_hotkey, order.trade_pair):
                self._put(order)

    def clear(self) -> None:
        """Drop everything from memory. Disk is untouched."""
        self._orders.clear()
        self._where.clear()

    def copy_all_to_positions(self) -> int:
        """Re-copy every bound order onto its position, e.g. once the position server is up."""
        copied = 0
        seen = set()
        for order in self.select():
            key = (order.miner_hotkey, legacy_uuid(order))
            if order.position_uuid and key not in seen:
                seen.add(key)
                copied += self._sync_position(order)
        return copied

    # ==================== Internals ====================

    def _put(self, order: TriggerOrder, copy_to_position: bool = True) -> None:
        self._write(order, "open")
        self._orders.setdefault(order.trade_pair, {}).setdefault(order.miner_hotkey, {})[order.order_uuid] = order
        self._where[(order.miner_hotkey, order.order_uuid)] = order.trade_pair
        if copy_to_position:
            self._sync_position(order)

    def _pop(self, order: TriggerOrder) -> TriggerOrder | None:
        """Drop an order from memory and open/. Callers sync the position copy afterwards."""
        popped = self._orders.get(order.trade_pair, {}).get(order.miner_hotkey, {}).pop(order.order_uuid, None)
        if popped is None:
            return None
        self._where.pop((order.miner_hotkey, order.order_uuid), None)
        self._delete(order, "open")
        return popped

    def _group(self, order: TriggerOrder) -> list[TriggerOrder]:
        """The open orders sharing order's oco_group, or order alone (if open) when it has none."""
        if order.oco_group is None:
            return [o for o in [self.get(order.miner_hotkey, order.order_uuid)] if o]
        return [
            o for o in self.select(hotkey=order.miner_hotkey, trade_pair=order.trade_pair)
            if o.oco_group == order.oco_group
        ]

    def _sync_position(self, order: TriggerOrder) -> bool:
        """Make the position copy for order's group match its open bound legs."""
        if not order.position_uuid or not self._position_client:
            return False
        hotkey, trade_pair_id = order.miner_hotkey, order.trade_pair.trade_pair_id
        legs = [o for o in self._group(order) if o.position_uuid]
        try:
            if legs:
                return self._position_client.attach_bracket_order_to_position(hotkey, trade_pair_id, to_bracket_dict(legs))
            self._position_client.remove_bracket_order_from_position(hotkey, trade_pair_id, legacy_uuid(order))
        except Exception as e:
            logger.error(f"[TRIGGER] Error syncing [{legacy_uuid(order)}] to position: {e}")
        return False

    def _dir(self, hotkey: str, trade_pair_id: str, status: str) -> str:
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        return f"{miner_dir}{hotkey}/trigger_orders/{trade_pair_id}/{status}/"

    def _write(self, order: TriggerOrder, status: str) -> None:
        try:
            path = self._dir(order.miner_hotkey, order.trade_pair.trade_pair_id, status) + order.order_uuid
            ValiBkpUtils.write_file(path, order.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"[TRIGGER] Error writing [{order.order_uuid}] to disk: {e}")

    def _delete(self, order: TriggerOrder, status: str) -> None:
        path = self._dir(order.miner_hotkey, order.trade_pair.trade_pair_id, status) + order.order_uuid
        if os.path.exists(path):
            os.remove(path)

    def _read_dir(self, hotkey: str, status: str) -> list[TriggerOrder]:
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        base = f"{miner_dir}{hotkey}/trigger_orders/"
        if not os.path.exists(base):
            return []
        orders = []
        for trade_pair_id in ValiBkpUtils.get_directories_in_dir(base):
            status_dir = self._dir(hotkey, trade_pair_id, status)
            if not os.path.exists(status_dir):
                continue
            for filename in os.listdir(status_dir):
                try:
                    with open(status_dir + filename) as f:
                        orders.append(TriggerOrder.model_validate(json.load(f)))
                except Exception as e:
                    logger.error(f"[TRIGGER] Error reading {status_dir}{filename}: {e}")
        return orders

    def _load_from_disk(self) -> None:
        miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        if not os.path.exists(miner_dir):
            return
        for hotkey in ValiBkpUtils.get_directories_in_dir(miner_dir):
            for order in self._read_dir(hotkey, "open"):
                self._orders.setdefault(order.trade_pair, {}).setdefault(hotkey, {})[order.order_uuid] = order
                self._where[(hotkey, order.order_uuid)] = order.trade_pair
        logger.info(f"[TRIGGER] Loaded {len(self._where)} open trigger orders from disk")
