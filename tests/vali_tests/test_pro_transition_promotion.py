"""
Unit tests for the miner-initiated promotion out of PRO_CHALLENGE_TRANSITION, and for the admin endpoint
that sets the pro account size.

The admin sets the pro account size when offering the pro track; a miner never chooses one.

Covers:
  * ChallengePeriodManager.promote_pro_transition: the bucket guard, the account switch that closes
    positions / cancels limit orders / restarts the ledgers, and the size recorded when the admin offered
    the transition (through a real EntityManager), including the organic promotions and a re-offer after a
    demotion.
  * The admin endpoint POST /admin/miner-bucket/<hotkey>: a size is required to enter the pro track and must
    be a finite number within [$200,000, $1,000,000]; PRO_FUNDED may re-set it; a bad size is refused before
    anything moves (the literals NaN and Infinity, which Flask's JSON parser accepts, included).
  * The validator HTTP endpoint: any request carrying pro_account_size is a 400 that never reaches the
    challenge period client and never consumes its nonce; coldkey signature bound to every remaining field,
    nonce + timestamp replay protection, field validation, and subaccount ownership.
  * The gateway HTTP endpoint: the payload it signs and forwards never carries a size, a request with one is
    refused before signing, and validator errors are passed through.
"""
import contextlib
import json
import unittest
import uuid
from unittest.mock import MagicMock, patch

import pytest
from bittensor_wallet import Keypair
from flask import Flask

from time_util.time_util import TimeUtil
from vali_objects.challenge_period.challengeperiod_manager import ChallengePeriodManager
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.vali_config import ValiConfig
from tests.vali_tests.test_pro_account_size import (
    GRANTED_SIZE,
    REQUIRED,
    STANDARD_SIZE,
    _add_demoted,
    _add_pro,
    _add_standard as _add_standard_subaccount,
    _bare_manager as _bare_entity_manager,
)

NOW_MS = 1_748_000_000_000
HOTKEY = "entity_0"

_CLIENT_PATHS = [
    "vali_objects.challenge_period.challengeperiod_manager.PerfLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.PositionManagerClient",
    "vali_objects.challenge_period.challengeperiod_manager.LimitOrderClient",
    "vali_objects.challenge_period.challengeperiod_manager.EliminationClient",
    "vali_objects.challenge_period.challengeperiod_manager.PlagiarismClient",
    "vali_objects.challenge_period.challengeperiod_manager.MinerAccountClient",
    "vali_objects.challenge_period.challengeperiod_manager.CommonDataClient",
    "vali_objects.challenge_period.challengeperiod_manager.AssetSelectionClient",
    "vali_objects.challenge_period.challengeperiod_manager.DebtLedgerClient",
    "vali_objects.challenge_period.challengeperiod_manager.EntityClient",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — ChallengePeriodManager.promote_pro_transition
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def manager():
    with contextlib.ExitStack() as stack:
        for path in _CLIENT_PATHS:
            stack.enter_context(patch(path))
        mgr = ChallengePeriodManager(is_backtesting=True)
        mgr._entity_client.apply_bucket_account_size.return_value = (True, "account size set")
        with patch.object(mgr, "_sync_buckets_to_accounts"):
            yield mgr


def _in_transition(manager, bucket=MinerBucket.PRO_CHALLENGE_TRANSITION):
    manager.set_miner_bucket(HOTKEY, bucket, NOW_MS)
    return manager


def test_promotion_moves_the_bucket_and_sends_no_size(manager):
    """The miner does not choose a size: the entity manager keeps the one recorded at the transition."""
    _in_transition(manager)

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS)

    assert success, message
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, None
    )


def test_promotion_winds_down_the_standard_account(manager):
    _in_transition(manager)

    assert manager.promote_pro_transition(HOTKEY, NOW_MS)[0]

    manager._position_client.close_all_positions.assert_called_once_with(
        hotkey=HOTKEY, close_time_ms=NOW_MS, order_source=OrderSource.SUBACCOUNT_PROMOTION
    )
    manager._position_client.archive_positions_for_hotkey.assert_called_once_with(HOTKEY, archive_all=True)
    manager._limit_order_client.cancel_limit_order.assert_called_once_with(HOTKEY, None, "ALL", NOW_MS)
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_called_once_with([HOTKEY])
    manager._debt_ledger_client.delete_debt_ledger.assert_called_once_with(HOTKEY)
    manager._miner_account_client.reset_account.assert_called_once_with(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    )


@pytest.mark.parametrize(
    "challenge_bucket",
    [MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_FROM_STANDARD],
)
def test_pro_funded_always_starts_fresh(manager, challenge_bucket):
    """A pro funded account starts from scratch, so challenge-period gains are never payable."""
    manager.set_miner_bucket(HOTKEY, challenge_bucket, NOW_MS)

    assert manager.promote_hotkeys([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_FUNDED
    manager._position_client.close_all_positions.assert_called_once_with(
        hotkey=HOTKEY, close_time_ms=NOW_MS, order_source=OrderSource.SUBACCOUNT_PROMOTION
    )
    manager._position_client.archive_positions_for_hotkey.assert_called_once_with(HOTKEY, archive_all=True)
    manager._limit_order_client.cancel_limit_order.assert_called_once_with(HOTKEY, None, "ALL", NOW_MS)
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_called_once_with([HOTKEY])
    manager._debt_ledger_client.delete_debt_ledger.assert_called_once_with(HOTKEY)


def test_entity_rejection_blocks_the_promotion(manager):
    """When the entity manager cannot size the pro account, nothing is wound down."""
    _in_transition(manager)
    manager._entity_client.apply_bucket_account_size.return_value = (False, REQUIRED)

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS)

    assert not success
    assert REQUIRED in message
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    manager._position_client.close_all_positions.assert_not_called()


def _with_real_entity_manager(manager, hotkey_factory=_add_standard_subaccount):
    """Swap the mocked entity client for a real in-memory EntityManager holding one standard
    SUBACCOUNT_FUNDED subaccount. Returns (entity_manager, synthetic_hotkey)."""
    entity_manager = _bare_entity_manager()
    hotkey = hotkey_factory(entity_manager)
    manager._entity_client = entity_manager
    manager.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS)
    return entity_manager, hotkey


def _admin_move_to_transition(manager, entity_manager, hotkey, pro_account_size=GRANTED_SIZE):
    """What POST /admin/miner-bucket does: size the subaccount, then move the bucket."""
    assert entity_manager.apply_bucket_account_size(hotkey, MinerBucket.PRO_CHALLENGE_TRANSITION, pro_account_size)[0]
    assert manager.admin_set_bucket(hotkey, MinerBucket.PRO_CHALLENGE_TRANSITION, NOW_MS)[0]


def test_promotion_trades_the_size_the_admin_set(manager):
    entity_manager, hotkey = _with_real_entity_manager(manager)
    _admin_move_to_transition(manager, entity_manager, hotkey, pro_account_size=250_000)

    success, message = manager.promote_pro_transition(hotkey, NOW_MS)

    assert success, message
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 250_000
    assert info.account_size == 250_000
    assert info.standard_account_size == STANDARD_SIZE
    assert entity_manager.get_payout_scale(hotkey) == pytest.approx(STANDARD_SIZE / 250_000)


def test_promotion_without_a_recorded_size_is_rejected(manager):
    """A subaccount that never had a size set (nothing to fall back on) is not promoted onto a pro
    account of unknown size."""
    entity_manager, hotkey = _with_real_entity_manager(manager)
    manager.set_miner_bucket(hotkey, MinerBucket.PRO_CHALLENGE_TRANSITION, NOW_MS)

    success, message = manager.promote_pro_transition(hotkey, NOW_MS)

    assert not success
    assert REQUIRED in message
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size is None
    assert info.account_size == STANDARD_SIZE
    manager._position_client.close_all_positions.assert_not_called()


def test_organic_promotion_keeps_the_recorded_size(manager):
    """The end-of-week auto promotion and the pro funded promotion send no size either."""
    entity_manager, hotkey = _with_real_entity_manager(manager)
    _admin_move_to_transition(manager, entity_manager, hotkey)

    assert manager.promote_hotkeys([hotkey], NOW_MS)
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    assert entity_manager.get_subaccount_info_for_synthetic(hotkey).account_size == GRANTED_SIZE

    assert manager.promote_hotkeys([hotkey], NOW_MS + 1000)
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_FUNDED
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == GRANTED_SIZE
    assert info.account_size == GRANTED_SIZE


def test_reoffer_after_a_demotion_needs_a_new_size(manager):
    """A demoted subaccount keeps its old pro size on record, but re-entering the track needs a size."""
    entity_manager, hotkey = _with_real_entity_manager(manager, hotkey_factory=_add_demoted)
    assert entity_manager.get_subaccount_info_for_synthetic(hotkey).pro_account_size == GRANTED_SIZE

    success, message = entity_manager.apply_bucket_account_size(hotkey, MinerBucket.PRO_CHALLENGE_TRANSITION)
    assert not success
    assert REQUIRED in message

    _admin_move_to_transition(manager, entity_manager, hotkey, pro_account_size=200_000)
    assert manager.promote_pro_transition(hotkey, NOW_MS)[0]
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 200_000
    assert info.account_size == 200_000


def test_a_failed_promotion_rolls_back_the_pro_sizing(manager):
    """The pro account is sized before the bucket moves, so a failed move must put the sizing back:
    PRO_CHALLENGE_FROM_STANDARD trades the pro size, and a subaccount left holding it while still in
    PRO_CHALLENGE_TRANSITION is supposed to be trading the standard account."""
    entity_manager, hotkey = _with_real_entity_manager(manager)
    _admin_move_to_transition(manager, entity_manager, hotkey)
    before = entity_manager.get_subaccount_info_for_synthetic(hotkey).model_copy()
    assert before.account_size == STANDARD_SIZE

    with patch.object(manager, "admin_set_bucket", return_value=(False, "bucket move failed")):
        success, message = manager.promote_pro_transition(hotkey, NOW_MS)

    assert not success
    assert "bucket move failed" in message
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_size == STANDARD_SIZE
    assert info.standard_account_size == before.standard_account_size
    assert info.pro_account_size == GRANTED_SIZE
    assert info.account_type == before.account_type
    resized_to = [c.kwargs['account_size']
                  for c in entity_manager._miner_account_client.set_miner_account_size.call_args_list]
    assert resized_to[-1] == STANDARD_SIZE


@pytest.mark.parametrize("bucket", [
    MinerBucket.SUBACCOUNT_CHALLENGE,
    MinerBucket.SUBACCOUNT_FUNDED,
    MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
    MinerBucket.PRO_FUNDED,
    MinerBucket.ELIMINATED,
])
def test_only_transition_subaccounts_can_promote(manager, bucket):
    _in_transition(manager, bucket)

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS)

    assert not success
    assert bucket.value in message
    assert manager.miner_states[HOTKEY].current_bucket == bucket
    manager._entity_client.apply_bucket_account_size.assert_not_called()
    manager._position_client.close_all_positions.assert_not_called()


def test_unknown_hotkey_is_rejected(manager):
    success, message = manager.promote_pro_transition("not_a_miner_0", NOW_MS)

    assert not success
    assert "not found" in message
    manager._entity_client.apply_bucket_account_size.assert_not_called()


def test_promotion_is_not_repeatable(manager):
    _in_transition(manager)
    assert manager.promote_pro_transition(HOTKEY, NOW_MS)[0]

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS + 1000)

    assert not success
    assert MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value in message


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Validator endpoints
# ═══════════════════════════════════════════════════════════════════════════════

PRO_TRANSITION_SIGNED_FIELDS = ("entity_coldkey", "entity_hotkey", "synthetic_hotkey", "nonce", "timestamp")
SIZE_REFUSED = "pro_account_size is not accepted"
# Every kind of size a miner might send: in range, the bounds, out of range, wrong type, null, and the
# non-finite floats json.dumps writes as the literals NaN / Infinity / -Infinity.
EXPLICIT_SIZES = (
    GRANTED_SIZE, ValiConfig.MIN_PRO_ACCOUNT_SIZE, ValiConfig.MAX_PRO_ACCOUNT_SIZE,
    ValiConfig.MIN_PRO_ACCOUNT_SIZE - 1, ValiConfig.MAX_PRO_ACCOUNT_SIZE + 1, 1, 0, -1,
    "500000", True, None, float("nan"), float("inf"), float("-inf"),
)


def _validator_pro_transition_client():
    """Flask test client for the validator endpoint with mocked clients and ownership check."""
    from vanta_api.nonce_manager import NonceManager
    from vanta_api.validator_rest_server import ValidatorRestServer

    server = object.__new__(ValidatorRestServer)
    server._entity_client = MagicMock()
    server._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "pro_account_size": GRANTED_SIZE, "account_size": GRANTED_SIZE,
    }
    server._challenge_period_client = MagicMock()
    server._challenge_period_client.promote_pro_transition.return_value = (True, "promoted")
    server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
    server.nonce_manager = NonceManager()
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/entity/subaccount/pro-transition", methods=["POST"])(server.promote_pro_transition)
    return server, app.test_client()


class TestValidatorProTransitionEndpoint(unittest.TestCase):
    """POST /entity/subaccount/pro-transition on the validator, with mocked RPC clients."""

    def setUp(self):
        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.synthetic = f"{self.hotkey.ss58_address}_0"
        self.server, self.client = _validator_pro_transition_client()

    def _body(self, signed=None, drop=(), **tampered):
        """A correctly signed request. `signed` overrides or adds fields before signing (what a gateway
        would have sent), `drop` removes them before signing, `tampered` overrides after signing."""
        fields = {
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey.ss58_address,
            "synthetic_hotkey": self.synthetic,
            "nonce": uuid.uuid4().hex,
            "timestamp": TimeUtil.now_in_millis(),
        }
        fields.update(signed or {})
        for field in drop:
            fields.pop(field, None)
        message = json.dumps(fields, sort_keys=True).encode("utf-8")
        body = {**fields, "signature": self.coldkey.sign(message).hex(), "version": "2.2.1"}
        body.update(tampered)
        return body

    def _post(self, body):
        # json.dumps writes NaN / Infinity as bare literals, exactly what request.get_json() parses back
        resp = self.client.post("/entity/subaccount/pro-transition", data=json.dumps(body),
                                content_type="application/json")
        return resp.status_code, json.loads(resp.data)

    def _promote(self):
        return self.server._challenge_period_client.promote_pro_transition

    def test_valid_request_reaches_the_challenge_period_client_without_a_size(self):
        """The promotion keeps the size the admin recorded: the endpoint never passes one."""
        status, data = self._post(self._body())
        self.assertEqual(status, 200)
        self.assertEqual(data['bucket'], MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value)
        self.assertEqual(data['pro_account_size'], GRANTED_SIZE)
        self._promote().assert_called_once()
        self.assertEqual(len(self._promote().call_args.args), 2)
        hotkey, timestamp = self._promote().call_args.args
        self.assertEqual(self._promote().call_args.kwargs, {})
        self.assertEqual(hotkey, self.synthetic)
        self.assertLess(abs(timestamp - TimeUtil.now_in_millis()), 60_000)

    def _assert_size_refused_without_consuming_the_nonce(self, body):
        """`body` carries a pro_account_size: it is a 400, and its nonce still works without the size."""
        self._promote().reset_mock()
        status, data = self._post(body)
        self.assertEqual(status, 400, data)
        self.assertIn(SIZE_REFUSED, data['error'])
        self.assertIn("/admin/miner-bucket", data['error'])
        self._promote().assert_not_called()

        retry = self._body(signed={"nonce": body["nonce"], "timestamp": body["timestamp"]})
        self.assertNotIn("pro_account_size", retry)
        status, data = self._post(retry)
        self.assertEqual(status, 200, data)
        self._promote().assert_called_once()

    def test_any_signed_size_is_400(self):
        """A correctly signed size is refused whatever its value: a miner never chooses its pro size."""
        for size in EXPLICIT_SIZES:
            with self.subTest(pro_account_size=size):
                self._assert_size_refused_without_consuming_the_nonce(
                    self._body(signed={"pro_account_size": size})
                )

    def test_a_size_added_to_a_signed_request_is_400(self):
        """A size bolted onto a request signed without one is refused on the size, not the signature."""
        for size in (ValiConfig.MAX_PRO_ACCOUNT_SIZE, 1, float("nan")):
            with self.subTest(pro_account_size=size):
                self._assert_size_refused_without_consuming_the_nonce(self._body(pro_account_size=size))

    def test_size_is_refused_before_the_signature_and_ownership_checks(self):
        other = Keypair.create_from_uri("//Charlie")
        self.server._verify_coldkey_owns_hotkey.return_value = False
        status, data = self._post(self._body(signed={"pro_account_size": GRANTED_SIZE},
                                             signature=other.sign(b"anything").hex()))
        self.assertEqual(status, 400)
        self.assertIn(SIZE_REFUSED, data['error'])
        self.server._verify_coldkey_owns_hotkey.assert_not_called()
        self._promote().assert_not_called()

    def test_manager_rejection_is_a_400(self):
        self._promote().return_value = (
            False, "entity_0 is in SUBACCOUNT_FUNDED, not PRO_CHALLENGE_TRANSITION"
        )
        status, data = self._post(self._body())
        self.assertEqual(status, 400)
        self.assertIn("not PRO_CHALLENGE_TRANSITION", data['error'])

    def test_bad_signature_is_401(self):
        other = Keypair.create_from_uri("//Charlie")
        status, _ = self._post(self._body(signature=other.sign(b"anything").hex()))
        self.assertEqual(status, 401)
        self._promote().assert_not_called()

    def test_coldkey_not_owning_hotkey_is_403(self):
        self.server._verify_coldkey_owns_hotkey.return_value = False
        status, _ = self._post(self._body())
        self.assertEqual(status, 403)
        self._promote().assert_not_called()

    def test_subaccount_of_another_entity_is_403(self):
        other = Keypair.create_from_uri("//Charlie")
        status, data = self._post(self._body(signed={"synthetic_hotkey": f"{other.ss58_address}_0"}))
        self.assertEqual(status, 403)
        self.assertIn("does not belong to entity", data['error'])
        self._promote().assert_not_called()

    def test_non_subaccount_hotkey_is_400(self):
        status, data = self._post(self._body(signed={"synthetic_hotkey": self.hotkey.ss58_address}))
        self.assertEqual(status, 400)
        self.assertIn("not a subaccount", data['error'])
        self._promote().assert_not_called()

    def test_missing_and_malformed_fields_are_400(self):
        for field in ("synthetic_hotkey", "nonce", "timestamp", "signature"):
            with self.subTest(missing=field):
                body = self._body()
                del body[field]
                status, data = self._post(body)
                self.assertEqual(status, 400)
                self.assertIn(field, data['error'])
        for bad in ({"nonce": ""}, {"nonce": 123}, {"timestamp": "now"}, {"timestamp": 1.5}, {"timestamp": True},
                    {"synthetic_hotkey": ""}):
            with self.subTest(signed=bad):
                status, _ = self._post(self._body(signed=bad))
                self.assertEqual(status, 400)
        self._promote().assert_not_called()

    # ==================== replay protection ====================

    def test_replayed_request_is_rejected(self):
        body = self._body()
        self.assertEqual(self._post(body)[0], 200)
        status, data = self._post(body)
        self.assertEqual(status, 401)
        self.assertIn("Nonce already used", data['error'])
        self._promote().assert_called_once()

    def test_signature_is_bound_to_every_signed_field(self):
        for field, value in (
            ("entity_coldkey", Keypair.create_from_uri("//Charlie").ss58_address),
            ("synthetic_hotkey", f"{self.hotkey.ss58_address}_1"),
            ("nonce", uuid.uuid4().hex),
            ("timestamp", TimeUtil.now_in_millis() + 30_000),
        ):
            with self.subTest(tampered=field):
                body = self._body()
                self.assertNotEqual(body[field], value)
                body[field] = value
                status, _ = self._post(body)
                self.assertEqual(status, 401)
        self._promote().assert_not_called()

    def test_signature_covers_exactly_the_signed_fields(self):
        """The validator rebuilds only PRO_TRANSITION_SIGNED_FIELDS: a signature over anything more fails."""
        body = self._body(signed={"extra": "field"})
        del body["extra"]
        status, _ = self._post(body)
        self.assertEqual(status, 401)
        self._promote().assert_not_called()

    def test_expired_or_future_timestamp_is_rejected(self):
        now = TimeUtil.now_in_millis()
        for timestamp in (now - 6 * 60 * 1000, now + 2 * 60 * 1000):
            with self.subTest(timestamp=timestamp):
                status, _ = self._post(self._body(signed={"timestamp": timestamp}))
                self.assertEqual(status, 401)
        self._promote().assert_not_called()

    def test_nonce_is_consumed_only_after_signature_and_ownership_pass(self):
        body = self._body()
        other = Keypair.create_from_uri("//Charlie")
        self.assertEqual(self._post({**body, "signature": other.sign(b"anything").hex()})[0], 401)
        self.server._verify_coldkey_owns_hotkey.return_value = False
        self.assertEqual(self._post(body)[0], 403)
        self.server._verify_coldkey_owns_hotkey.return_value = True
        self.assertEqual(self._post(body)[0], 200)


class TestAdminMinerBucketProSize(unittest.TestCase):
    """POST /admin/miner-bucket/<hotkey> sizing a pro move through a real in-memory EntityManager."""

    def setUp(self):
        from vanta_api.validator_rest_server import ValidatorRestServer

        self.entity_manager = _bare_entity_manager()
        self.hotkey = _add_standard_subaccount(self.entity_manager)
        self.server = object.__new__(ValidatorRestServer)
        self.server._get_api_key_safe = MagicMock(return_value="key")
        self.server.is_valid_api_key = MagicMock(return_value=True)
        self.server.can_access_tier = MagicMock(return_value=True)
        self.server._entity_client = self.entity_manager
        self.server._challenge_period_client = MagicMock()
        self.server._challenge_period_client.can_admin_set_bucket.return_value = (True, "")
        self.server._challenge_period_client.admin_set_bucket.return_value = (True, "moved")
        self.server._miner_account_client = MagicMock()
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/admin/miner-bucket/<hotkey>", methods=["POST"])(self.server.set_miner_bucket_admin)
        self.client = app.test_client()

    def _post(self, body, hotkey=None):
        resp = self.client.post(f"/admin/miner-bucket/{hotkey or self.hotkey}", json=body)
        return resp.status_code, json.loads(resp.data)

    def _post_raw(self, raw_json, hotkey=None):
        resp = self.client.post(f"/admin/miner-bucket/{hotkey or self.hotkey}", data=raw_json,
                                content_type="application/json")
        return resp.status_code, json.loads(resp.data)

    def _info(self, hotkey=None):
        return self.entity_manager.get_subaccount_info_for_synthetic(hotkey or self.hotkey)

    def _assert_nothing_moved(self, hotkey=None):
        info = self._info(hotkey)
        self.assertIsNone(info.pro_account_size)
        self.assertIsNone(info.standard_account_size)
        self.assertEqual(info.account_size, STANDARD_SIZE)
        self.assertEqual(info.account_type, "standard")
        self.entity_manager._miner_account_client.set_miner_account_size.assert_not_called()
        self.server._challenge_period_client.admin_set_bucket.assert_not_called()
        self.server._miner_account_client.set_miner_bucket.assert_not_called()

    def test_entering_the_track_without_a_size_is_400(self):
        for bucket in (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_DIRECT):
            with self.subTest(bucket=bucket):
                status, data = self._post({"bucket": bucket.value})
                self.assertEqual(status, 400)
                self.assertEqual(data['error'], REQUIRED)
                self._assert_nothing_moved()

    def test_entering_the_track_with_a_size_records_it(self):
        status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_TRANSITION.value,
                                   "pro_account_size": 450_000})
        self.assertEqual(status, 200, data)
        info = self._info()
        self.assertEqual(info.pro_account_size, 450_000)
        self.assertEqual(info.standard_account_size, STANDARD_SIZE)
        self.assertEqual(info.account_size, STANDARD_SIZE)  # TRANSITION keeps the standard account
        self.server._challenge_period_client.admin_set_bucket.assert_called_once()

    def test_direct_pro_challenge_trades_the_size(self):
        status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value,
                                   "pro_account_size": 450_000})
        self.assertEqual(status, 200, data)
        self.assertEqual(self._info().account_size, 450_000)

    def test_range_bounds_are_inclusive(self):
        for i, (size, accepted) in enumerate(((199_999, False), (200_000, True),
                                              (1_000_000, True), (1_000_001, False)), start=10):
            hotkey = _add_standard_subaccount(self.entity_manager, subaccount_id=i)
            self.entity_manager._miner_account_client.reset_mock()
            self.server._challenge_period_client.reset_mock()
            self.server._challenge_period_client.can_admin_set_bucket.return_value = (True, "")
            self.server._challenge_period_client.admin_set_bucket.return_value = (True, "moved")
            self.server._miner_account_client.reset_mock()
            with self.subTest(pro_account_size=size):
                status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value,
                                           "pro_account_size": size}, hotkey=hotkey)
                if accepted:
                    self.assertEqual(status, 200, data)
                    self.assertEqual(self._info(hotkey).pro_account_size, size)
                else:
                    self.assertEqual(status, 400, data)
                    self.assertIn("outside the allowed range", data['error'])
                    self._assert_nothing_moved(hotkey)

    def test_non_finite_sizes_are_rejected_before_anything_moves(self):
        """Flask's request.get_json() parses these literals, and NaN passes a plain range check."""
        for bucket in (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_DIRECT,
                       MinerBucket.SUBACCOUNT_CHALLENGE):
            for literal in ("NaN", "Infinity", "-Infinity", "1e999"):
                with self.subTest(bucket=bucket, pro_account_size=literal):
                    status, data = self._post_raw(
                        f'{{"bucket": "{bucket.value}", "pro_account_size": {literal}}}'
                    )
                    self.assertEqual(status, 400, data)
                    self.assertIn("must be a finite number", data['error'])
                    self._assert_nothing_moved()
                    self.server._challenge_period_client.can_admin_set_bucket.assert_not_called()

    def test_non_numeric_sizes_are_rejected_before_anything_moves(self):
        for size in ("500000", True, False, [500_000]):
            with self.subTest(pro_account_size=size):
                status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value,
                                           "pro_account_size": size})
                self.assertEqual(status, 400, data)
                self.assertIn("must be a number", data['error'])
                self._assert_nothing_moved()
                self.server._challenge_period_client.can_admin_set_bucket.assert_not_called()

    def test_pro_funded_may_re_set_the_size(self):
        pro = _add_pro(self.entity_manager, pro_size=GRANTED_SIZE)

        status, data = self._post({"bucket": MinerBucket.PRO_FUNDED.value,
                                   "pro_account_size": 1_000_001}, hotkey=pro)
        self.assertEqual(status, 400, data)
        self.assertEqual(self._info(pro).pro_account_size, GRANTED_SIZE)

        status, data = self._post({"bucket": MinerBucket.PRO_FUNDED.value,
                                   "pro_account_size": 900_000}, hotkey=pro)
        self.assertEqual(status, 200, data)
        info = self._info(pro)
        self.assertEqual(info.pro_account_size, 900_000)
        self.assertEqual(info.account_size, 900_000)

    def test_pro_funded_without_a_size_keeps_the_recorded_one(self):
        pro = _add_pro(self.entity_manager, pro_size=GRANTED_SIZE)
        status, data = self._post({"bucket": MinerBucket.PRO_FUNDED.value}, hotkey=pro)
        self.assertEqual(status, 200, data)
        info = self._info(pro)
        self.assertEqual(info.pro_account_size, GRANTED_SIZE)
        self.assertEqual(info.account_size, GRANTED_SIZE)

    def test_reoffer_after_a_demotion_requires_a_size(self):
        demoted = _add_demoted(self.entity_manager)

        status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_TRANSITION.value}, hotkey=demoted)
        self.assertEqual(status, 400, data)
        self.assertEqual(data['error'], REQUIRED)
        self.assertEqual(self._info(demoted).account_type, "standard")
        self.server._challenge_period_client.admin_set_bucket.assert_not_called()

        status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_TRANSITION.value,
                                   "pro_account_size": 350_000}, hotkey=demoted)
        self.assertEqual(status, 200, data)
        self.assertEqual(self._info(demoted).pro_account_size, 350_000)

    def test_standard_bucket_never_records_a_pro_size(self):
        status, data = self._post({"bucket": MinerBucket.SUBACCOUNT_CHALLENGE.value,
                                   "pro_account_size": 450_000})
        self.assertEqual(status, 200, data)
        self.assertIsNone(self._info().pro_account_size)
        self.assertEqual(self.entity_manager.get_payout_scale(self.hotkey), 1.0)

    def test_a_failed_bucket_move_leaves_no_pro_marking(self):
        """The sizing is committed before the bucket moves, so a failed move must roll it back: a
        subaccount left marked pro would trade the pro size in a standard bucket, and the next
        size-less re-offer would take the "recorded" branch instead of demanding a size."""
        for i, bucket in enumerate((MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_DIRECT),
                                   start=20):
            with self.subTest(bucket=bucket):
                hotkey = _add_standard_subaccount(self.entity_manager, subaccount_id=i)
                self.entity_manager._miner_account_client.reset_mock()
                self.server._challenge_period_client.admin_set_bucket.return_value = (
                    False, f"{hotkey} account size update failed, bucket unchanged")

                status, data = self._post({"bucket": bucket.value, "pro_account_size": 500_000}, hotkey=hotkey)
                self.assertEqual(status, 400, data)

                info = self._info(hotkey)
                self.assertEqual(info.account_type, "standard")
                self.assertIsNone(info.pro_account_size)
                self.assertIsNone(info.standard_account_size)
                self.assertEqual(info.account_size, STANDARD_SIZE)
                self.server._miner_account_client.set_miner_bucket.assert_not_called()
                # PRO_CHALLENGE_DIRECT trades the pro size, so its resize was committed too and the
                # live account must be put back; TRANSITION keeps the standard account and never moved it
                sizes = [call.kwargs['account_size']
                         for call in self.entity_manager._miner_account_client.set_miner_account_size.call_args_list]
                self.assertEqual(sizes[-1] if sizes else STANDARD_SIZE, STANDARD_SIZE)

                # behavioural backstop: the offer did not happen, so a size is still required
                self.server._challenge_period_client.admin_set_bucket.return_value = (True, "moved")
                status, data = self._post({"bucket": bucket.value}, hotkey=hotkey)
                self.assertEqual(status, 400, data)
                self.assertEqual(data['error'], REQUIRED)

    def test_a_bucket_move_that_raises_leaves_no_pro_marking(self):
        """The 500 path: an exception inside admin_set_bucket (its wind-down or its disk write) must
        not leave the sizing write landed either."""
        hotkey = _add_standard_subaccount(self.entity_manager, subaccount_id=30)
        self.server._challenge_period_client.admin_set_bucket.side_effect = RuntimeError("disk full")

        status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value,
                                   "pro_account_size": 500_000}, hotkey=hotkey)

        self.assertEqual(status, 500, data)
        info = self._info(hotkey)
        self.assertEqual(info.account_type, "standard")
        self.assertIsNone(info.pro_account_size)
        self.assertEqual(info.account_size, STANDARD_SIZE)

    def test_a_successful_move_is_never_rolled_back(self):
        """set_miner_bucket runs after the bucket moved; if it raises, the sizing must stand."""
        hotkey = _add_standard_subaccount(self.entity_manager, subaccount_id=31)
        self.server._miner_account_client.set_miner_bucket.side_effect = RuntimeError("bookkeeping blew up")

        status, data = self._post({"bucket": MinerBucket.PRO_CHALLENGE_DIRECT.value,
                                   "pro_account_size": 500_000}, hotkey=hotkey)

        self.assertEqual(status, 500, data)
        info = self._info(hotkey)
        self.assertEqual(info.account_type, "pro")
        self.assertEqual(info.pro_account_size, 500_000)
        self.assertEqual(info.account_size, 500_000)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Gateway endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestGatewayProTransitionEndpoint(unittest.TestCase):
    """POST /api/promote-pro-transition on the gateway, with the validator call patched."""

    def setUp(self):
        from vanta_api.entity_miner_rest_server import EntityMinerRestServer

        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.synthetic = f"{self.hotkey.ss58_address}_0"
        self.gw = object.__new__(EntityMinerRestServer)
        self.gw._coldkey = self.coldkey
        self.gw._hotkey = self.hotkey
        self.gw._validator_url = "http://validator.test"
        self.gw._get_api_key_safe = MagicMock(return_value="key")
        self.gw.is_valid_api_key = MagicMock(return_value=True)
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/api/promote-pro-transition", methods=["POST"])(self.gw.promote_pro_transition_endpoint)
        self.client = app.test_client()

    def _post(self, body):
        # json.dumps writes NaN / Infinity as bare literals, exactly what request.get_json() parses back
        resp = self.client.post("/api/promote-pro-transition", data=json.dumps(body),
                                content_type="application/json")
        return resp.status_code, json.loads(resp.data)

    def _forward(self, body=None):
        """Post to the gateway with the validator call patched; returns (status, data, payload sent)."""
        validator_resp = MagicMock(status_code=200)
        validator_resp.json.return_value = {"status": "success", "pro_account_size": GRANTED_SIZE}
        with patch("requests.post", return_value=validator_resp) as post:
            status, data = self._post(body if body is not None else {"synthetic_hotkey": self.synthetic})
        self.assertEqual(post.call_args.args[0], "http://validator.test/entity/subaccount/pro-transition")
        return status, data, post.call_args.kwargs['json']

    def _assert_signed(self, payload, fields):
        signed = json.dumps({k: payload[k] for k in fields}, sort_keys=True).encode("utf-8")
        self.assertTrue(Keypair(ss58_address=self.coldkey.ss58_address).verify(
            signed, bytes.fromhex(payload['signature'])))

    def test_forwards_a_signed_payload_with_no_size(self):
        status, data, payload = self._forward()
        self.assertEqual(status, 200)
        self.assertEqual(data['pro_account_size'], GRANTED_SIZE)
        self.assertEqual(payload['entity_coldkey'], self.coldkey.ss58_address)
        self.assertEqual(payload['entity_hotkey'], self.hotkey.ss58_address)
        self.assertEqual(payload['synthetic_hotkey'], self.synthetic)
        self.assertIsInstance(payload['nonce'], str)
        self.assertTrue(payload['nonce'])
        self.assertIsInstance(payload['timestamp'], int)
        self.assertLess(abs(payload['timestamp'] - TimeUtil.now_in_millis()), 60_000)
        # No size is signed or sent: the payload is exactly the signed fields plus signature and version
        self.assertEqual(set(payload), set(PRO_TRANSITION_SIGNED_FIELDS) | {"signature", "version"})
        self._assert_signed(payload, PRO_TRANSITION_SIGNED_FIELDS)

    def test_any_size_is_refused_before_signing(self):
        """The admin owns the pro size, so the gateway never signs or forwards a miner-chosen one."""
        self.gw._coldkey = MagicMock()
        for size in EXPLICIT_SIZES:
            with self.subTest(pro_account_size=size), patch("requests.post") as post:
                status, data = self._post({"synthetic_hotkey": self.synthetic, "pro_account_size": size})
                self.assertEqual(status, 400, data)
                self.assertIn(SIZE_REFUSED, data['message'])
                self.assertIn("/admin/miner-bucket", data['message'])
                post.assert_not_called()
        self.gw._coldkey.sign.assert_not_called()

    def test_each_request_gets_a_fresh_nonce(self):
        first = self._forward()[2]
        second = self._forward()[2]
        self.assertNotEqual(first['nonce'], second['nonce'])
        self.assertNotEqual(first['signature'], second['signature'])

    def test_gateway_payload_is_accepted_by_the_validator_endpoint(self):
        payload = self._forward()[2]
        server, client = _validator_pro_transition_client()
        resp = client.post("/entity/subaccount/pro-transition", json=payload)
        self.assertEqual(resp.status_code, 200, resp.data)
        promote = server._challenge_period_client.promote_pro_transition
        promote.assert_called_once()
        self.assertEqual(promote.call_args.args[0], self.synthetic)
        self.assertEqual(len(promote.call_args.args), 2)
        self.assertEqual(promote.call_args.kwargs, {})
        # The same bytes a second time are a replay
        self.assertEqual(client.post("/entity/subaccount/pro-transition", json=payload).status_code, 401)

    def test_validator_error_is_passed_through(self):
        validator_resp = MagicMock(status_code=400)
        validator_resp.json.return_value = {"error": "not PRO_CHALLENGE_TRANSITION"}
        with patch("requests.post", return_value=validator_resp):
            status, data = self._post({"synthetic_hotkey": self.synthetic})
        self.assertEqual(status, 400)
        self.assertIn("not PRO_CHALLENGE_TRANSITION", data['message'])

    def test_invalid_input_never_reaches_validator(self):
        with patch("requests.post") as post:
            for body in (
                {"pro_account_size": GRANTED_SIZE},
                {"synthetic_hotkey": ""},
                {"synthetic_hotkey": 123},
            ):
                with self.subTest(body=body):
                    status, _ = self._post(body)
                    self.assertEqual(status, 400)
            post.assert_not_called()

    def test_bad_api_key_is_401(self):
        self.gw.is_valid_api_key.return_value = False
        with patch("requests.post") as post:
            status, _ = self._post({"synthetic_hotkey": self.synthetic})
        self.assertEqual(status, 401)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
