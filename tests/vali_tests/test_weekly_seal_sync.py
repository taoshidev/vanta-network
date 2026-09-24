"""
Autosync of the weekly seal ledger.

Covers the two halves of the guarantee: the records survive a round trip through the same
ValiBkpUtils path every other validator object is written with, and they travel in the validator
checkpoint so every validator ends up agreeing on what was sealed.
"""
import os
from unittest.mock import patch

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.data_export.core_outputs_manager import CoreOutputsManager
from vali_objects.data_sync.auto_sync import PositionSyncer
from vali_objects.data_sync.order_sync_state import OrderSyncState
from vali_objects.data_sync.validator_sync_base import AUTO_SYNC_ORDER_LAG_MS
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode
from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import (
    SealedWeek,
    SettledSegment,
    WeeklySealLedger,
)

MS_IN_WEEK = 1000 * 60 * 60 * 24 * 7
WEEK_0 = TimeUtil.ms_at_start_of_week(1700000000000)
WEEK_1 = WEEK_0 + MS_IN_WEEK


def sealed_week_dict(week_start_ms, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK"):
    return SealedWeek(
        week_start_ms=week_start_ms,
        weekly_penalty=weekly_penalty,
        payout_scale=payout_scale,
        track=track,
        first_earning_ms=None,
        sealed_ms=week_start_ms + 1,
    ).to_dict()


def settled_segment_dict(week_start_ms, bucket="PRO_CHALLENGE_TRANSITION", payout_usd=100.0,
                         gross_payout_usd=120.0, segment_end_ms=None):
    return SettledSegment(
        week_start_ms=week_start_ms,
        segment_start_ms=week_start_ms,
        segment_end_ms=segment_end_ms if segment_end_ms is not None else week_start_ms + 5000,
        bucket=bucket,
        payout_usd=payout_usd,
        gross_payout_usd=gross_payout_usd,
        weekly_penalty=1.0,
        payout_scale=1.0,
        recorded_ms=week_start_ms + 6000,
    ).to_dict()


class TestWeeklySealLedgerPersistence(TestBase):
    """Disk round trip and merge semantics, on the ledger object itself."""

    def setUp(self):
        self.ledger = WeeklySealLedger(running_unit_tests=True)
        self.ledger.clear_for_test()

    def tearDown(self):
        self.ledger.clear_for_test()

    def test_saves_to_the_vali_bkp_utils_path(self):
        self.assertEqual(
            self.ledger._get_path(),
            ValiBkpUtils.get_weekly_seal_ledger_file_location(running_unit_tests=True),
        )

    def test_disk_round_trip(self):
        self.ledger.seal("hk1", WEEK_0, weekly_penalty=0.0, payout_scale=0.5,
                         track="OFF_TRACK", first_earning_ms=WEEK_0 + 10)
        self.ledger.record_settled(
            "hk1", week_start_ms=WEEK_0, segment_start_ms=WEEK_0, segment_end_ms=WEEK_0 + 5000,
            bucket="PRO_CHALLENGE_TRANSITION", payout_usd=42.5, gross_payout_usd=50.0,
            weekly_penalty=0.0, payout_scale=0.5,
        )
        self.ledger.save_to_disk()

        self.assertTrue(os.path.exists(self.ledger._get_path()))
        reloaded = WeeklySealLedger(running_unit_tests=True)
        self.assertEqual(reloaded.to_checkpoint_dict(), self.ledger.to_checkpoint_dict())

        week = reloaded.get_sealed("hk1")[WEEK_0]
        self.assertEqual(week.payout_scale, 0.5)
        self.assertEqual(week.track, "OFF_TRACK")
        self.assertEqual(week.first_earning_ms, WEEK_0 + 10)
        self.assertEqual(reloaded.get_settled("hk1")[0].payout_usd, 42.5)

    def test_checkpoint_dict_keys_weeks_as_strings(self):
        """JSON has no integer keys, so the checkpoint has to survive a string round trip."""
        self.ledger.seal("hk1", WEEK_0, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK")
        checkpoint = self.ledger.to_checkpoint_dict()
        self.assertEqual(list(checkpoint['sealed']['hk1']), [str(WEEK_0)])

        peer = WeeklySealLedger(running_unit_tests=True)
        peer.sealed.clear()
        peer.settled.clear()
        peer.sync_from_checkpoint(checkpoint)
        self.assertIn(WEEK_0, peer.get_sealed("hk1"))

    def test_sync_adds_records_this_validator_is_missing(self):
        stats = self.ledger.sync_from_checkpoint({
            'sealed': {'hk1': {str(WEEK_0): sealed_week_dict(WEEK_0)}},
            'settled': {'hk1': [settled_segment_dict(WEEK_0)]},
        })
        self.assertEqual(stats['sealed_added'], 1)
        self.assertEqual(stats['settled_added'], 1)
        self.assertIn(WEEK_0, self.ledger.get_sealed("hk1"))
        self.assertEqual(self.ledger.get_settled("hk1")[0].payout_usd, 100.0)

    def test_sync_adopts_the_checkpoints_verdict_on_a_disagreement(self):
        """A validator that classified a closed week differently is the divergence being repaired."""
        self.ledger.seal("hk1", WEEK_0, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK")
        stats = self.ledger.sync_from_checkpoint({
            'sealed': {'hk1': {str(WEEK_0): sealed_week_dict(WEEK_0, weekly_penalty=0.0,
                                                             track="OFF_TRACK")}},
        })
        self.assertEqual(stats['sealed_replaced'], 1)
        self.assertEqual(stats['sealed_added'], 0)
        week = self.ledger.get_sealed("hk1")[WEEK_0]
        self.assertEqual(week.weekly_penalty, 0.0)
        self.assertEqual(week.track, "OFF_TRACK")

    def test_sync_leaves_an_agreeing_record_alone(self):
        self.ledger.seal("hk1", WEEK_0, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK")
        local_sealed_ms = self.ledger.get_sealed("hk1")[WEEK_0].sealed_ms
        stats = self.ledger.sync_from_checkpoint({
            'sealed': {'hk1': {str(WEEK_0): sealed_week_dict(WEEK_0)}},
        })
        self.assertEqual(stats['sealed_replaced'], 0)
        self.assertEqual(self.ledger.get_sealed("hk1")[WEEK_0].sealed_ms, local_sealed_ms)

    def test_sync_keeps_records_only_this_validator_has(self):
        """The checkpoint is time-lagged; a week sealed since it was written must not be dropped."""
        self.ledger.seal("hk1", WEEK_1, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK")
        self.ledger.record_settled(
            "hk1", week_start_ms=WEEK_1, segment_start_ms=WEEK_1, segment_end_ms=WEEK_1 + 1,
            bucket="PRO_FUNDED", payout_usd=7.0, gross_payout_usd=8.0,
            weekly_penalty=1.0, payout_scale=1.0,
        )
        self.ledger.sync_from_checkpoint({
            'sealed': {'hk1': {str(WEEK_0): sealed_week_dict(WEEK_0)}},
            'settled': {'hk1': [settled_segment_dict(WEEK_0)]},
        })
        self.assertEqual(sorted(self.ledger.get_sealed("hk1")), [WEEK_0, WEEK_1])
        self.assertEqual(
            sorted(s.week_start_ms for s in self.ledger.get_settled("hk1")), [WEEK_0, WEEK_1]
        )

    def test_sync_is_idempotent(self):
        checkpoint = {
            'sealed': {'hk1': {str(WEEK_0): sealed_week_dict(WEEK_0)}},
            'settled': {'hk1': [settled_segment_dict(WEEK_0)]},
        }
        self.ledger.sync_from_checkpoint(checkpoint)
        stats = self.ledger.sync_from_checkpoint(checkpoint)
        self.assertEqual(
            stats,
            {'sealed_added': 0, 'sealed_replaced': 0, 'settled_added': 0,
             'settled_replaced': 0, 'errors': 0},
        )
        self.assertEqual(len(self.ledger.get_settled("hk1")), 1)

    def test_settled_segments_are_keyed_on_week_and_bucket(self):
        """A retried account switch has a fresh segment_end_ms; it must not pay the week twice."""
        self.ledger.record_settled(
            "hk1", week_start_ms=WEEK_0, segment_start_ms=WEEK_0, segment_end_ms=WEEK_0 + 1,
            bucket="PRO_CHALLENGE_TRANSITION", payout_usd=100.0, gross_payout_usd=120.0,
            weekly_penalty=1.0, payout_scale=1.0,
        )
        stats = self.ledger.sync_from_checkpoint({
            'settled': {'hk1': [settled_segment_dict(WEEK_0, segment_end_ms=WEEK_0 + 999)]},
        })
        self.assertEqual(stats['settled_added'], 0)
        self.assertEqual(stats['settled_replaced'], 0)
        self.assertEqual(len(self.ledger.get_settled("hk1")), 1)

    def test_a_different_bucket_in_the_same_week_is_its_own_record(self):
        self.ledger.sync_from_checkpoint({
            'settled': {'hk1': [
                settled_segment_dict(WEEK_0, bucket="PRO_CHALLENGE_TRANSITION"),
                settled_segment_dict(WEEK_0, bucket="PRO_FUNDED", segment_end_ms=WEEK_0 + 9000),
            ]},
        })
        self.assertEqual(len(self.ledger.get_settled("hk1")), 2)

    def test_settled_segments_stay_sorted(self):
        self.ledger.sync_from_checkpoint({
            'settled': {'hk1': [
                settled_segment_dict(WEEK_1, bucket="PRO_FUNDED", segment_end_ms=WEEK_1 + 10),
                settled_segment_dict(WEEK_0, bucket="PRO_FUNDED", segment_end_ms=WEEK_0 + 10),
            ]},
        })
        ends = [s.segment_end_ms for s in self.ledger.get_settled("hk1")]
        self.assertEqual(ends, sorted(ends))

    def test_sync_skips_malformed_records(self):
        stats = self.ledger.sync_from_checkpoint({
            'sealed': {'hk1': {'not-a-week': sealed_week_dict(WEEK_0),
                               str(WEEK_1): {'missing': 'fields'}}},
            'settled': {'hk1': ['not-a-segment']},
        })
        self.assertEqual(stats['errors'], 3)
        self.assertEqual(self.ledger.get_sealed("hk1"), {})
        self.assertEqual(self.ledger.get_settled("hk1"), [])

    def test_sync_tolerates_junk_input(self):
        empty = {'sealed_added': 0, 'sealed_replaced': 0, 'settled_added': 0,
                 'settled_replaced': 0, 'errors': 0}
        for junk in (None, [], "", {}, {'sealed': None, 'settled': None}):
            self.assertEqual(self.ledger.sync_from_checkpoint(junk), empty)

    def test_sync_persists_what_it_merged(self):
        self.ledger.sync_from_checkpoint({
            'sealed': {'hk1': {str(WEEK_0): sealed_week_dict(WEEK_0)}},
        })
        self.assertIn(WEEK_0, WeeklySealLedger(running_unit_tests=True).get_sealed("hk1"))

    def test_sync_does_not_write_when_nothing_changed(self):
        with patch.object(WeeklySealLedger, 'save_to_disk') as mock_save:
            self.ledger.sync_from_checkpoint({'sealed': {}, 'settled': {}})
            mock_save.assert_not_called()

    def test_a_synced_week_is_not_resealed_by_a_local_rebuild(self):
        """The point of syncing: the peer's verdict wins over what this validator would recompute."""
        self.ledger.sync_from_checkpoint({
            'sealed': {'hk1': {str(WEEK_0): sealed_week_dict(WEEK_0, weekly_penalty=0.0,
                                                             track="OFF_TRACK")}},
        })
        self.assertFalse(
            self.ledger.seal("hk1", WEEK_0, weekly_penalty=1.0, payout_scale=1.0, track="ON_TRACK")
        )
        self.assertEqual(self.ledger.get_sealed("hk1")[WEEK_0].track, "OFF_TRACK")


class TestWeeklySealAutoSync(TestBase):
    """The checkpoint path end to end, against the running debt ledger server."""

    orchestrator = None
    debt_ledger_client = None
    position_syncer = None

    HOTKEY = "seal_sync_test_miner"

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=ValiUtils.get_secrets(running_unit_tests=True),
        )
        cls.debt_ledger_client = cls.orchestrator.get_client('debt_ledger')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.order_sync = OrderSyncState()
        cls.position_syncer = PositionSyncer(
            order_sync=cls.order_sync,
            running_unit_tests=True,
        )

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.debt_ledger_client.clear_weekly_seals_for_test()
        self.metagraph_client.set_hotkeys([self.HOTKEY])

    def tearDown(self):
        self.debt_ledger_client.clear_weekly_seals_for_test()

    def candidate_data(self, weekly_seals=None):
        data = {
            'positions': {},
            'eliminations': [],
            'created_timestamp_ms': TimeUtil.now_in_millis() + AUTO_SYNC_ORDER_LAG_MS,
        }
        if weekly_seals is not None:
            data['weekly_seals'] = weekly_seals
        return data

    def test_autosync_merges_weekly_seals(self):
        self.position_syncer.sync_positions(
            shadow_mode=False,
            candidate_data=self.candidate_data({
                'sealed': {self.HOTKEY: {str(WEEK_0): sealed_week_dict(WEEK_0, weekly_penalty=0.0,
                                                                       track="OFF_TRACK")}},
                'settled': {self.HOTKEY: [settled_segment_dict(WEEK_0)]},
            }),
            disk_positions={},
        )

        sealed = self.debt_ledger_client.get_sealed_weeks(self.HOTKEY)
        self.assertIn(WEEK_0, sealed)
        self.assertEqual(sealed[WEEK_0].track, "OFF_TRACK")
        self.assertEqual(sealed[WEEK_0].weekly_penalty, 0.0)

        settled = self.debt_ledger_client.get_settled_segments(self.HOTKEY)
        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0].payout_usd, 100.0)

        self.assertEqual(self.position_syncer.global_stats['weekly_seals_added'], 1)
        self.assertEqual(self.position_syncer.global_stats['settled_segments_added'], 1)

    def test_autosync_repairs_a_diverged_validator(self):
        self.debt_ledger_client.sync_weekly_seals({
            'sealed': {self.HOTKEY: {str(WEEK_0): sealed_week_dict(WEEK_0, weekly_penalty=1.0,
                                                                   track="ON_TRACK")}},
        })
        self.position_syncer.sync_positions(
            shadow_mode=False,
            candidate_data=self.candidate_data({
                'sealed': {self.HOTKEY: {str(WEEK_0): sealed_week_dict(WEEK_0, weekly_penalty=0.0,
                                                                       track="OFF_TRACK")}},
            }),
            disk_positions={},
        )
        sealed = self.debt_ledger_client.get_sealed_weeks(self.HOTKEY)
        self.assertEqual(sealed[WEEK_0].weekly_penalty, 0.0)
        self.assertEqual(self.position_syncer.global_stats['weekly_seals_replaced'], 1)

    def test_shadow_mode_does_not_merge(self):
        self.position_syncer.sync_positions(
            shadow_mode=True,
            candidate_data=self.candidate_data({
                'sealed': {self.HOTKEY: {str(WEEK_0): sealed_week_dict(WEEK_0)}},
            }),
            disk_positions={},
        )
        self.assertEqual(self.debt_ledger_client.get_sealed_weeks(self.HOTKEY), {})

    def test_checkpoint_without_weekly_seals_is_a_no_op(self):
        """Older checkpoints predate the key; syncing against one must not touch local seals."""
        self.debt_ledger_client.sync_weekly_seals({
            'sealed': {self.HOTKEY: {str(WEEK_0): sealed_week_dict(WEEK_0)}},
        })
        self.position_syncer.sync_positions(
            shadow_mode=False, candidate_data=self.candidate_data(), disk_positions={},
        )
        self.assertIn(WEEK_0, self.debt_ledger_client.get_sealed_weeks(self.HOTKEY))

    def test_the_checkpoint_carries_the_seal_records(self):
        self.debt_ledger_client.sync_weekly_seals({
            'sealed': {self.HOTKEY: {str(WEEK_0): sealed_week_dict(WEEK_0)}},
            'settled': {self.HOTKEY: [settled_segment_dict(WEEK_0)]},
        })

        manager = CoreOutputsManager(running_unit_tests=True, connection_mode=RPCConnectionMode.RPC)
        written = []
        with patch.object(ValiBkpUtils, 'write_compressed_json',
                          side_effect=lambda path, data: written.append((path, data))):
            manager.create_and_upload_production_files(
                eliminations=[],
                ord_dict_hotkey_position_map={},
                time_now=TimeUtil.now_in_millis(),
                youngest_order_processed_ms=0,
                oldest_order_processed_ms=0,
                challengeperiod_dict={},
                miner_account_sizes_dict={},
                limit_orders_dict={},
                save_to_disk=True,
                upload_to_gcloud=False,
            )

        self.assertTrue(written)
        checkpoint = written[0][1]
        self.assertIn('weekly_seals', checkpoint)
        self.assertIn(str(WEEK_0), checkpoint['weekly_seals']['sealed'][self.HOTKEY])
        self.assertEqual(len(checkpoint['weekly_seals']['settled'][self.HOTKEY]), 1)

    def test_checkpoint_survives_a_full_round_trip_through_a_peer(self):
        """Export from one validator, import into another, and the records match."""
        self.debt_ledger_client.sync_weekly_seals({
            'sealed': {self.HOTKEY: {str(WEEK_0): sealed_week_dict(WEEK_0, weekly_penalty=0.0,
                                                                   payout_scale=0.5,
                                                                   track="OFF_TRACK")}},
            'settled': {self.HOTKEY: [settled_segment_dict(WEEK_0)]},
        })
        exported = self.debt_ledger_client.get_weekly_seals_checkpoint_dict()

        peer = WeeklySealLedger(running_unit_tests=True)
        peer.sealed.clear()
        peer.settled.clear()
        with patch.object(WeeklySealLedger, 'save_to_disk'):
            peer.sync_from_checkpoint(exported)

        week = peer.get_sealed(self.HOTKEY)[WEEK_0]
        self.assertEqual((week.weekly_penalty, week.payout_scale, week.track),
                         (0.0, 0.5, "OFF_TRACK"))
        self.assertEqual(peer.get_settled(self.HOTKEY)[0].payout_usd, 100.0)
