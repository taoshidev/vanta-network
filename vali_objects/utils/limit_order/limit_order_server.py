# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
LimitOrderServer - RPC server for limit, stop-limit and bracket orders.

Runs in its own process; clients connect with LimitOrderClient. Orders are held as TriggerOrders
(vali_objects/utils/trigger_order); this server converts the legacy Order shapes the API and
callers use in both directions, so the RPC surface is unchanged.
"""
from shared_objects.rpc.common_data_client import CommonDataClient
from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.utils.trigger_order.legacy_converter import (
    migrate_legacy_disk,
    sync_legacy_orders,
    to_legacy_orders,
)
from vali_objects.vali_config import RPCConnectionMode, TradePair, ValiConfig


class LimitOrderServer(RPCServerBase):
    """
    RPC server for limit order management. Public methods ending in _rpc are exposed to
    LimitOrderClient; the daemon fires orders whose trigger has crossed.
    """
    service_name = ValiConfig.RPC_LIMITORDERMANAGER_SERVICE_NAME
    service_port = ValiConfig.RPC_LIMITORDERMANAGER_PORT

    def __init__(
        self,
        running_unit_tests=False,
        slack_notifier=None,
        start_server=True,
        start_daemon=True,
        serve=True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        self.running_unit_tests = running_unit_tests
        self._common_data_client = CommonDataClient(connect_immediately=False)

        # Create the manager before RPCServerBase.__init__ so RPC calls never see it missing
        from vali_objects.utils.trigger_order.trigger_order_manager import TriggerOrderManager
        self._manager = TriggerOrderManager(running_unit_tests=running_unit_tests, connection_mode=connection_mode)
        migrate_legacy_disk(self._manager.store, running_unit_tests)

        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_LIMITORDERMANAGER_SERVICE_NAME,
            port=ValiConfig.RPC_LIMITORDERMANAGER_PORT,
            connection_mode=connection_mode,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # started below, after full initialization
            daemon_interval_s=0.01,
            hang_timeout_s=120.0
        )

        if start_daemon:
            self.start_daemon()

    def run_daemon_iteration(self) -> None:
        self._manager.tick()

    def get_health_check_details(self) -> dict:
        return self._manager.health_check()

    # ==================== Placement ====================

    def process_limit_order_rpc(self, miner_hotkey, order, is_edit=False):
        return self._manager.place_legacy(miner_hotkey, order, is_edit)

    def place_rpc(self, orders):
        return self._manager.place(*orders)

    # ==================== Cancellation ====================

    def cancel_limit_order_rpc(self, miner_hotkey, trade_pair_id, order_uuid, now_ms, execution_type=None, order_src=None):
        """
        Cancel by uuid (comma-separated for several) or "ALL" (optionally per trade pair and
        execution_type). Each uuid matches one order exactly; a bracket's legs are separate orders.
        """
        cancel_all = bool(order_uuid) and order_uuid.strip().upper() == "ALL"
        if cancel_all:
            trade_pair = TradePair.from_trade_pair_id(trade_pair_id) if trade_pair_id else None
            targets = [
                o for o in self._manager.select(hotkey=miner_hotkey, trade_pair=trade_pair)
                if execution_type is None or o.execution_type == execution_type
            ]
        else:
            uuids = {u.strip() for u in order_uuid.split(',')} if order_uuid else set()
            targets = [o for o in self._manager.select(hotkey=miner_hotkey) if o.order_uuid in uuids]

        if not targets and not cancel_all:
            raise SignalException(f"No unfilled limit orders found for {miner_hotkey} (uuid={order_uuid})")
        for o in targets:
            self._manager.cancel(miner_hotkey, now_ms, order_uuid=o.order_uuid, reason=order_src)

        return {
            "status": "cancelled",
            "order_uuid": order_uuid if order_uuid else "all",
            "miner_hotkey": miner_hotkey,
            "cancelled_ms": now_ms,
            "num_cancelled": len(targets),
        }

    def cancel_entry_orders_rpc(self, miner_hotkey, now_ms, order_src=None):
        """Cancel every order that would open or add to a position; brackets and exits stay."""
        cancelled = self._manager.cancel(miner_hotkey, now_ms, entry_only=True, reason=order_src)
        return {
            "status": "cancelled",
            "miner_hotkey": miner_hotkey,
            "cancelled_ms": now_ms,
            "num_cancelled": len(cancelled),
        }

    def delete_all_limit_orders_for_hotkey_rpc(self, miner_hotkey):
        """Delete every open order for a hotkey without a cancel record. Returns the number deleted."""
        orders = self._manager.select(hotkey=miner_hotkey)
        for o in orders:
            self._manager.store.discard(miner_hotkey, o.order_uuid)
        return len(orders)

    def restore_cancelled_limit_orders_rpc(self, miner_hotkey: str) -> int:
        """Reopen orders cancelled by elimination."""
        return self._manager.reopen(miner_hotkey, OrderSource.ELIMINATION_CANCELLED)

    # ==================== Reads (legacy Order shapes) ====================

    def get_limit_order_by_uuid_rpc(self, miner_hotkey, order_uuid):
        order = self._manager.get(miner_hotkey, order_uuid)
        return to_legacy_orders([order])[0].to_python_dict() if order else None

    def get_limit_orders_for_hotkey_rpc(self, miner_hotkey):
        return [o.to_python_dict() for o in to_legacy_orders(self._manager.select(hotkey=miner_hotkey))]

    def get_limit_orders_for_trade_pair_rpc(self, trade_pair_id):
        trade_pair = TradePair.from_trade_pair_id(trade_pair_id)
        return self._by_hotkey(self._manager.select(trade_pair=trade_pair))

    def get_all_limit_orders_rpc(self):
        result = {}
        for order in self._manager.select():
            result.setdefault(order.trade_pair.trade_pair_id, []).append(order)
        return {trade_pair_id: self._by_hotkey(orders) for trade_pair_id, orders in result.items()}

    def to_dashboard_dict_rpc(self, miner_hotkey, status_filter=None):
        """
        Without status_filter: a list of open order dicts (None if empty). With it: {status: [dicts]}
        for 'unfilled' and 'cancelled' (filled orders live on positions), None if all empty.
        """
        orders = []
        if not status_filter or "unfilled" in status_filter:
            orders.extend(self._manager.select(hotkey=miner_hotkey))
        if not status_filter:
            return [o.to_python_dict() for o in to_legacy_orders(orders)] or None

        if "cancelled" in status_filter:
            orders.extend(self._manager.store.read_closed(miner_hotkey))
        status_set = {s.upper() for s in status_filter}
        result = {s.lower(): [] for s in status_set}
        for legacy in to_legacy_orders(orders):
            status = OrderSource.status(legacy.src)
            if status in status_set:
                result[status.lower()].append(legacy.to_python_dict())
        return result if any(result.values()) else None

    def get_dashboard_rpc(self, hotkey: str, limit_orders_time_ms: int) -> dict | None:
        """Open orders placed after limit_orders_time_ms, plus uuids closed since the last call."""
        snapshot_time_ms = limit_orders_time_ms
        open_orders = {}
        for legacy in to_legacy_orders(self._manager.select(hotkey=hotkey)):
            if legacy.processed_ms > limit_orders_time_ms:
                snapshot_time_ms = max(snapshot_time_ms, legacy.processed_ms)
                open_orders[legacy.order_uuid] = legacy.to_dashboard(include_trade_pair=True)
        closed_orders = list(self._manager._closed.pop(hotkey, {}))

        if not open_orders and not closed_orders:
            return None
        dashboard = {}
        if open_orders:
            dashboard["open_orders"] = open_orders
        if closed_orders:
            dashboard["closed_orders"] = closed_orders
        dashboard["limit_orders_time_ms"] = snapshot_time_ms
        return dashboard

    # ==================== Sync ====================

    def sync_limit_orders_rpc(self, sync_data):
        """Load a checkpoint's legacy orders: {trade_pair_id: {hotkey: [order dicts]}}."""
        sync_legacy_orders(self._manager.store, sync_data)

    # ==================== Helpers ====================

    @staticmethod
    def _by_hotkey(orders) -> dict:
        grouped = {}
        for order in orders:
            grouped.setdefault(order.miner_hotkey, []).append(order)
        return {hotkey: [o.to_python_dict() for o in to_legacy_orders(group)] for hotkey, group in grouped.items()}

    # ==================== Test-only RPC methods ====================

    def _require_tests(self, name):
        if not self.running_unit_tests:
            raise Exception(f'{name} can only be called in unit test mode')

    def clear_limit_orders_rpc(self):
        """Clear all orders from memory. Disk is untouched."""
        self._require_tests('clear_limit_orders_rpc')
        self._manager.store.clear()
        self._manager._last_fill_ms.clear()
        self._manager._closed.clear()
        self._manager._market_order_client.clear_order_cooldown_cache()

    def check_and_fill_limit_orders_rpc(self, call_id=None):
        """Run one daemon tick. call_id only defeats RPC caching in tests."""
        self._require_tests('check_and_fill_limit_orders_rpc')
        result = self._manager.tick()
        return {'checked': result['checked'], 'filled': result['fired'], 'timestamp_ms': result['timestamp_ms']}

    def set_limit_orders_dict_rpc(self, orders_dict):
        """Replace the open orders with {trade_pair_id: {hotkey: [legacy order dicts]}}."""
        self._require_tests('set_limit_orders_dict_rpc')
        self._manager.store.clear()
        sync_legacy_orders(self._manager.store, orders_dict)

    def get_last_fill_time_rpc(self):
        self._require_tests('get_last_fill_time_rpc')
        result = {}
        for (hotkey, trade_pair_id), fill_ms in self._manager._last_fill_ms.items():
            result.setdefault(trade_pair_id, {})[hotkey] = fill_ms
        return result

    def set_last_fill_time_rpc(self, trade_pair_id, hotkey, fill_time):
        self._require_tests('set_last_fill_time_rpc')
        self._manager._last_fill_ms[(hotkey, trade_pair_id)] = fill_time

    def count_unfilled_orders_for_hotkey_rpc(self, miner_hotkey):
        self._require_tests('count_unfilled_orders_for_hotkey_rpc')
        return len(self._manager.select(hotkey=miner_hotkey))

    def get_position_for_rpc(self, hotkey, order):
        self._require_tests('get_position_for_rpc')
        return self._manager._open_position(hotkey, order.trade_pair)

