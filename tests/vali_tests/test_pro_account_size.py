"""
The network-owned pro account size.

Covers:
  * ValiConfig.PRO_ACCOUNT_SIZE: the size a pro promotion grants, bounded by MAX_PRO_ACCOUNT_SIZE at import.
  * EntityManager.apply_bucket_account_size: entering the pro track with no size grants the network size,
    an explicit size is still honoured (and capped), a recorded size is kept, a failed or rejected move
    changes nothing, and a standard account never picks up a pro size (so payout_scale stays 1.0).
  * subaccount_info.default_pro_account_size: present with the same name in the same place in the v1
    (GET /entity/subaccount/<hotkey>) and v2 (GET /v2/entity/subaccount/<hotkey>, websocket) dashboards,
    for standard and pro subaccounts, next to an untouched subaccount_info.pro_account_size.

EntityManager is built with object.__new__ and only the attributes these methods touch, so no RPC servers
are started.
"""
import json
import re
import threading
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

import vali_objects.vali_config as vali_config_module
from entity_management.entity_manager import EntityData, EntityManager, SubaccountInfo
from entity_management.entity_utils import create_subaccount_dashboard
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig

NOW_MS = 1_748_000_000_000
ENTITY_HOTKEY = "entity_alpha"
STANDARD_SIZE = 100_000.0
GRANTED_SIZE = 500_000.0  # deliberately different from ValiConfig.PRO_ACCOUNT_SIZE
FIELD = "default_pro_account_size"


def _bare_manager():
    """An EntityManager with in-memory entities, a mocked account client, and no disk writes."""
    manager = object.__new__(EntityManager)
    manager.is_backtesting = True
    manager.running_unit_tests = True
    manager.entities = {}
    manager._entities_lock = threading.RLock()
    manager._entity_locks = {}
    manager._miner_account_client = MagicMock()
    manager._miner_account_client.set_miner_account_size.return_value = {"account_size": "set"}
    manager._miner_account_client.get_account.return_value = None
    manager._challenge_period_client = MagicMock()
    manager._challenge_period_client.has_miner.return_value = False
    manager._challenge_period_client.get_drawdown_stats.return_value = None
    manager._debt_ledger_client = MagicMock()
    manager._debt_ledger_client.get_ledger.return_value = None
    manager._position_client = MagicMock()
    manager._position_client.get_positions_for_one_hotkey.return_value = []
    manager._limit_order_client = MagicMock()
    manager._limit_order_client.to_dashboard_dict.return_value = None
    manager._statistics_client = MagicMock()
    manager._statistics_client.get_miner_statistics_for_hotkey.return_value = None
    manager._elimination_client = MagicMock()
    manager._elimination_client.get_elimination.return_value = None
    return manager


def _add_standard(manager, subaccount_id=0, account_size=STANDARD_SIZE):
    info = SubaccountInfo(
        subaccount_id=subaccount_id,
        subaccount_uuid=f"uuid-{subaccount_id}",
        synthetic_hotkey=f"{ENTITY_HOTKEY}_{subaccount_id}",
        created_at_ms=NOW_MS,
        account_size=account_size,
        asset_class="forex",
    )
    entity = manager.entities.setdefault(
        ENTITY_HOTKEY, EntityData(entity_hotkey=ENTITY_HOTKEY, registered_at_ms=NOW_MS)
    )
    entity.subaccounts[subaccount_id] = info
    return info.synthetic_hotkey


def _add_pro(manager, subaccount_id=1, pro_size=GRANTED_SIZE):
    """A subaccount already promoted onto a pro account of pro_size."""
    hotkey = _add_standard(manager, subaccount_id)
    info = manager.get_subaccount_info_for_synthetic(hotkey)
    info.standard_account_size = STANDARD_SIZE
    info.pro_account_size = pro_size
    info.account_size = pro_size
    info.account_type = AccountType.PRO.value
    return hotkey


def _no_section_clients():
    """Dashboard section clients that have nothing to report, so only subaccount_info is built."""
    clients = {name: MagicMock() for name in (
        "challenge_period", "elimination", "miner_account", "position", "limit_order", "debt_ledger",
        "statistics",
    )}
    clients["challenge_period"].get_dashboard.return_value = None
    clients["challenge_period"].get_drawdown_stats.return_value = None
    clients["challenge_period"].get_pro_stats.return_value = None
    for name in ("elimination", "miner_account", "position", "limit_order", "debt_ledger", "statistics"):
        clients[name].get_dashboard.return_value = None
    return clients


# ═══════════════════════════════════════════════════════════════════════════════
# ValiConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestProAccountSizeConfig(unittest.TestCase):

    def test_network_grants_one_million_today(self):
        self.assertEqual(ValiConfig.PRO_ACCOUNT_SIZE, 1_000_000)

    def test_grant_is_positive_and_within_the_cap(self):
        self.assertGreater(ValiConfig.PRO_ACCOUNT_SIZE, 0)
        self.assertLessEqual(ValiConfig.PRO_ACCOUNT_SIZE, ValiConfig.MAX_PRO_ACCOUNT_SIZE)

    def test_import_fails_for_a_grant_outside_the_cap(self):
        """Editing PRO_ACCOUNT_SIZE above MAX_PRO_ACCOUNT_SIZE (or to a non-positive value) breaks import."""
        path = vali_config_module.__file__
        with open(path) as f:
            source = f.read()
        assignment = re.compile(r"^(    PRO_ACCOUNT_SIZE = ).*$", re.M)
        self.assertEqual(len(assignment.findall(source)), 1)

        for bad in ("MAX_PRO_ACCOUNT_SIZE + 1", "0", "-1"):
            with self.subTest(PRO_ACCOUNT_SIZE=bad):
                tampered = assignment.sub(lambda m: m.group(1) + bad, source)
                with self.assertRaisesRegex(ValueError, "PRO_ACCOUNT_SIZE"):
                    exec(compile(tampered, path, "exec"), {"__name__": "vali_config_probe", "__file__": path})

        # The untouched source imports cleanly, so the failures above come from the guard
        namespace = {"__name__": "vali_config_probe", "__file__": path}
        exec(compile(source, path, "exec"), namespace)
        self.assertEqual(namespace["ValiConfig"].PRO_ACCOUNT_SIZE, ValiConfig.PRO_ACCOUNT_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
# EntityManager.apply_bucket_account_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyBucketAccountSizeDefault(unittest.TestCase):

    def setUp(self):
        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)
        self.set_size = self.manager._miner_account_client.set_miner_account_size

    def _info(self, hotkey):
        return self.manager.get_subaccount_info_for_synthetic(hotkey)

    def test_transition_without_a_size_records_the_network_size(self):
        success, message = self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION
        )

        self.assertTrue(success, message)
        info = self._info(self.standard)
        self.assertEqual(info.pro_account_size, ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)
        self.assertEqual(info.account_type, AccountType.PRO.value)
        # TRANSITION keeps trading the standard account
        self.assertEqual(info.account_size, STANDARD_SIZE)
        self.set_size.assert_not_called()

    def test_direct_pro_challenge_without_a_size_trades_the_network_size(self):
        success, message = self.manager.apply_bucket_account_size(self.standard, MinerBucket.PRO_CHALLENGE_DIRECT)

        self.assertTrue(success, message)
        info = self._info(self.standard)
        self.assertEqual(info.pro_account_size, ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertEqual(info.account_size, ValiConfig.PRO_ACCOUNT_SIZE)
        self.set_size.assert_called_once()
        self.assertEqual(self.set_size.call_args.kwargs["account_size"], ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertEqual(
            self.set_size.call_args.kwargs["collateral_balance_theta"],
            ValiConfig.PRO_ACCOUNT_SIZE / ValiConfig.ENTITY_COST_PER_THETA,
        )

    def test_default_is_read_when_the_promotion_happens(self):
        """The network can change the size; a promotion grants whatever it is at that moment."""
        changed = ValiConfig.PRO_ACCOUNT_SIZE - 1
        with patch.object(ValiConfig, "PRO_ACCOUNT_SIZE", changed):
            self.assertTrue(self.manager.apply_bucket_account_size(
                self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION)[0])
        self.assertEqual(self._info(self.standard).pro_account_size, changed)

    def test_explicit_size_is_honoured(self):
        success, message = self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION, GRANTED_SIZE
        )

        self.assertTrue(success, message)
        self.assertEqual(self._info(self.standard).pro_account_size, GRANTED_SIZE)

    def test_explicit_size_above_the_cap_is_rejected_and_changes_nothing(self):
        success, message = self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_DIRECT, ValiConfig.MAX_PRO_ACCOUNT_SIZE + 1
        )

        self.assertFalse(success)
        self.assertIn("exceeds maximum", message)
        info = self._info(self.standard)
        self.assertIsNone(info.pro_account_size)
        self.assertIsNone(info.standard_account_size)
        self.assertEqual(info.account_type, AccountType.STANDARD.value)
        self.assertEqual(info.account_size, STANDARD_SIZE)
        self.set_size.assert_not_called()

    def test_recorded_size_is_kept_over_the_network_size(self):
        """Once granted, a size survives later pro moves even if the network size has changed since."""
        pro = _add_pro(self.manager)

        with patch.object(ValiConfig, "PRO_ACCOUNT_SIZE", GRANTED_SIZE + 1):
            success, message = self.manager.apply_bucket_account_size(pro, MinerBucket.PRO_FUNDED)

        self.assertTrue(success, message)
        info = self._info(pro)
        self.assertEqual(info.pro_account_size, GRANTED_SIZE)
        self.assertEqual(info.account_size, GRANTED_SIZE)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)

    def test_transition_then_promotion_grants_the_size_recorded_at_transition(self):
        """The admin move records the network size; the miner's own promotion (no size) trades it."""
        self.assertTrue(self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_TRANSITION)[0])

        success, message = self.manager.apply_bucket_account_size(
            self.standard, MinerBucket.PRO_CHALLENGE_FROM_STANDARD
        )

        self.assertTrue(success, message)
        info = self._info(self.standard)
        self.assertEqual(info.account_size, ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)
        self.assertAlmostEqual(self.manager.get_payout_scale(self.standard),
                               STANDARD_SIZE / ValiConfig.PRO_ACCOUNT_SIZE)

    def test_failed_account_resize_leaves_a_standard_account_untouched(self):
        """A pro move whose account resize fails leaves the subaccount standard, with no pro size."""
        self.set_size.return_value = None

        for size in (None, GRANTED_SIZE):
            with self.subTest(pro_account_size=size):
                success, message = self.manager.apply_bucket_account_size(
                    self.standard, MinerBucket.PRO_CHALLENGE_DIRECT, size
                )

                self.assertFalse(success)
                self.assertIn("Failed to set account size", message)
                info = self._info(self.standard)
                self.assertIsNone(info.pro_account_size)
                self.assertIsNone(info.standard_account_size)
                self.assertEqual(info.account_type, AccountType.STANDARD.value)
                self.assertEqual(info.account_size, STANDARD_SIZE)
                self.assertEqual(self.manager.get_payout_scale(self.standard), 1.0)

    def test_failed_account_resize_is_not_written_to_disk(self):
        self.set_size.return_value = None
        with patch.object(self.manager, "_write_entities_from_memory_to_disk") as write:
            self.manager.apply_bucket_account_size(self.standard, MinerBucket.PRO_CHALLENGE_DIRECT)
        write.assert_not_called()

    def test_standard_buckets_never_record_a_pro_size(self):
        for bucket in (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_ALPHA):
            with self.subTest(bucket=bucket):
                # Even a stray explicit size is ignored for a standard bucket
                success, message = self.manager.apply_bucket_account_size(self.standard, bucket, GRANTED_SIZE)
                self.assertTrue(success, message)
                info = self._info(self.standard)
                self.assertIsNone(info.pro_account_size)
                self.assertIsNone(info.standard_account_size)
                self.assertEqual(info.account_type, AccountType.STANDARD.value)
                self.assertEqual(info.account_size, STANDARD_SIZE)
                self.assertEqual(self.manager.get_payout_scale(self.standard), 1.0)
        self.set_size.assert_not_called()

    def test_standard_account_payout_scale_is_unaffected_by_the_network_size(self):
        """Publishing a network size must not give a never-promoted account a payout scale."""
        for size in (ValiConfig.PRO_ACCOUNT_SIZE, ValiConfig.PRO_ACCOUNT_SIZE - 1):
            with self.subTest(PRO_ACCOUNT_SIZE=size), patch.object(ValiConfig, "PRO_ACCOUNT_SIZE", size):
                self.assertEqual(self.manager.get_payout_scale(self.standard), 1.0)
                self.assertEqual(self.manager.get_subaccount_dashboard(self.standard)[FIELD], size)
                self.assertIsNone(self._info(self.standard).pro_account_size)

    def test_demotion_restores_the_standard_size_and_keeps_the_granted_one(self):
        pro = _add_pro(self.manager)

        success, message = self.manager.apply_bucket_account_size(pro, MinerBucket.SUBACCOUNT_FUNDED)

        self.assertTrue(success, message)
        info = self._info(pro)
        self.assertEqual(info.account_size, STANDARD_SIZE)
        self.assertEqual(info.account_type, AccountType.STANDARD.value)
        self.assertEqual(info.pro_account_size, GRANTED_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboards: subaccount_info.default_pro_account_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestDashboardDefaultProAccountSize(unittest.TestCase):

    def setUp(self):
        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)
        self.pro = _add_pro(self.manager)

    def _v1_info(self, hotkey):
        return self.manager.get_subaccount_dashboard_data(hotkey)["subaccount_info"]

    def _v2_info(self, hotkey):
        clients = _no_section_clients()
        dashboard = create_subaccount_dashboard(
            hotkey,
            self.manager.get_subaccount_dashboard(hotkey),
            clients["challenge_period"], clients["elimination"], clients["miner_account"],
            clients["position"], clients["limit_order"], clients["debt_ledger"], clients["statistics"],
            0, 0, 0, 0,
        )
        return dashboard["subaccount_info"]

    def test_v1_publishes_the_network_size_for_standard_and_pro(self):
        for hotkey in (self.standard, self.pro):
            with self.subTest(hotkey=hotkey):
                self.assertEqual(self._v1_info(hotkey)[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)

    def test_v2_publishes_the_network_size_for_standard_and_pro(self):
        for hotkey in (self.standard, self.pro):
            with self.subTest(hotkey=hotkey):
                self.assertEqual(self._v2_info(hotkey)[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)

    def test_v2_keeps_pro_account_size_as_the_granted_size(self):
        standard = self._v2_info(self.standard)
        self.assertIsNone(standard["pro_account_size"])
        self.assertEqual(standard["account_type"], AccountType.STANDARD.value)

        pro = self._v2_info(self.pro)
        self.assertEqual(pro["pro_account_size"], GRANTED_SIZE)
        self.assertEqual(pro["account_type"], AccountType.PRO.value)
        self.assertNotEqual(pro["pro_account_size"], pro[FIELD])

    def test_v1_and_v2_carry_the_same_value_in_the_same_place(self):
        for hotkey in (self.standard, self.pro):
            with self.subTest(hotkey=hotkey):
                self.assertEqual(self._v1_info(hotkey)[FIELD], self._v2_info(hotkey)[FIELD])

    def test_dashboards_follow_a_change_to_the_network_size(self):
        changed = ValiConfig.PRO_ACCOUNT_SIZE - 1
        with patch.object(ValiConfig, "PRO_ACCOUNT_SIZE", changed):
            self.assertEqual(self._v1_info(self.standard)[FIELD], changed)
            self.assertEqual(self._v2_info(self.standard)[FIELD], changed)
            # A granted size does not follow it
            self.assertEqual(self._v2_info(self.pro)["pro_account_size"], GRANTED_SIZE)

    def test_hl_subaccounts_carry_the_field_too(self):
        hotkey = _add_standard(self.manager, subaccount_id=2)
        self.manager.get_subaccount_info_for_synthetic(hotkey).hl_address = "0x" + "ab" * 20
        self.assertEqual(self._v1_info(hotkey)[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertEqual(self._v2_info(hotkey)[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)


class TestDashboardEndpointsDefaultProAccountSize(unittest.TestCase):
    """The field on the wire, through the real validator REST handlers and websocket frame builder."""

    def setUp(self):
        from vanta_api.validator_rest_server import ValidatorRestServer

        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)
        self.pro = _add_pro(self.manager)

        server = object.__new__(ValidatorRestServer)
        server._get_api_key_safe = MagicMock(return_value="key")
        server.is_valid_api_key = MagicMock(return_value=True)
        server.can_access_tier = MagicMock(return_value=True)
        server._entity_client = MagicMock()
        server._entity_client.get_subaccount_dashboard.side_effect = self.manager.get_subaccount_dashboard
        server._entity_client.get_subaccount_dashboard_data.side_effect = self.manager.get_subaccount_dashboard_data
        clients = _no_section_clients()
        server._challenge_period_client = clients["challenge_period"]
        server._elimination_client = clients["elimination"]
        server._miner_account_client = clients["miner_account"]
        server._position_client = clients["position"]
        server._limit_order_client = clients["limit_order"]
        server._debt_ledger_client = clients["debt_ledger"]
        server._statistics_client = clients["statistics"]
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.route("/entity/subaccount/<synthetic_hotkey>", methods=["GET"])(server.get_subaccount_dashboard)
        app.route("/v2/entity/subaccount/<synthetic_hotkey>", methods=["GET"])(server.v2_get_subaccount_dashboard)
        self.client = app.test_client()

    def _get(self, path):
        resp = self.client.get(path)
        self.assertEqual(resp.status_code, 200, resp.data)
        return json.loads(resp.data)["dashboard"]["subaccount_info"]

    def test_v1_endpoint(self):
        standard = self._get(f"/entity/subaccount/{self.standard}")
        pro = self._get(f"/entity/subaccount/{self.pro}")
        self.assertEqual(standard[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertEqual(pro[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)

    def test_v2_endpoint(self):
        standard = self._get(f"/v2/entity/subaccount/{self.standard}")
        pro = self._get(f"/v2/entity/subaccount/{self.pro}")
        self.assertEqual(standard[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertIsNone(standard["pro_account_size"])
        self.assertEqual(pro[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)
        self.assertEqual(pro["pro_account_size"], GRANTED_SIZE)

    def test_websocket_frames_carry_the_field_on_every_update(self):
        """The websocket rebuilds subaccount_info in full on each frame, including incremental ones
        where the watermarked sections have nothing new and are left out."""
        from vanta_api.websocket_server import DashboardSubscription, WebSocketServer, WebSocketServerClient

        server = object.__new__(WebSocketServer)
        server._event_loop = MagicMock()
        server._entity_client = MagicMock()
        server._entity_client.get_subaccount_dashboard.side_effect = self.manager.get_subaccount_dashboard
        clients = _no_section_clients()
        server._challenge_period_client = clients["challenge_period"]
        server._elimination_client = clients["elimination"]
        server._miner_account_client = clients["miner_account"]
        server._position_client = clients["position"]
        server._limit_order_client = clients["limit_order"]
        server._debt_ledger_client = clients["debt_ledger"]
        server._statistics_client = clients["statistics"]
        server._send_serialized = MagicMock()

        for hotkey, granted in ((self.standard, None), (self.pro, GRANTED_SIZE)):
            ws_client = WebSocketServerClient(client_id=1, websocket=MagicMock(), api_key="k", tier=200)
            subscription = DashboardSubscription()
            for frame in ("initial", "incremental"):
                with self.subTest(hotkey=hotkey, frame=frame), \
                        patch("vanta_api.websocket_server.asyncio.run_coroutine_threadsafe"):
                    if frame == "incremental":
                        subscription.positions_time_ms = NOW_MS
                        subscription.checkpoints_time_ms = NOW_MS
                    server._send_serialized.reset_mock()
                    server._send_dashboard_update(hotkey, ws_client, subscription)
                    _client, serialized = server._send_serialized.call_args.args
                    info = json.loads(serialized)["data"]["dashboard"]["subaccount_info"]
                    self.assertEqual(info[FIELD], ValiConfig.PRO_ACCOUNT_SIZE)
                    self.assertEqual(info["pro_account_size"], granted)


if __name__ == "__main__":
    unittest.main()
