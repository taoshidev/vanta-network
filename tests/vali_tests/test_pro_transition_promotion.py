"""
Unit tests for the miner-initiated promotion out of PRO_CHALLENGE_TRANSITION.

Covers:
  * ChallengePeriodManager.promote_pro_transition: the bucket guard, the account switch that closes
    positions / cancels limit orders / restarts the ledgers, the granted pro account size sent with
    the request, and the fallback to the size recorded when the subaccount entered the transition.
  * The validator HTTP endpoint: coldkey signature bound to the target subaccount and the requested
    size, nonce + timestamp replay protection, field validation, and subaccount ownership.
  * The gateway HTTP endpoint: the signed payload forwarded to the validator, with and without a
    pro_account_size, and validator errors passed through.
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


def test_promotion_moves_the_bucket_and_records_the_size(manager):
    _in_transition(manager)

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS, 500_000)

    assert success, message
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 500_000
    )


def test_promotion_winds_down_the_standard_account(manager):
    _in_transition(manager)

    assert manager.promote_pro_transition(HOTKEY, NOW_MS, 500_000)[0]

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


def test_no_size_sent_keeps_the_recorded_one(manager):
    """A request without a size promotes on the size recorded when the miner entered the transition."""
    _in_transition(manager)

    assert manager.promote_pro_transition(HOTKEY, NOW_MS)[0]

    manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, None
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


def test_unset_size_blocks_the_promotion(manager):
    """The entity manager rejects a pro bucket with no size on either side; nothing is wound down."""
    _in_transition(manager)
    manager._entity_client.apply_bucket_account_size.return_value = (
        False, "pro_account_size is required to enter the pro track"
    )

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS)

    assert not success
    assert "pro_account_size is required" in message
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    manager._position_client.close_all_positions.assert_not_called()


@pytest.mark.parametrize("bucket", [
    MinerBucket.SUBACCOUNT_CHALLENGE,
    MinerBucket.SUBACCOUNT_FUNDED,
    MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
    MinerBucket.PRO_FUNDED,
    MinerBucket.ELIMINATED,
])
def test_only_transition_subaccounts_can_promote(manager, bucket):
    _in_transition(manager, bucket)

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS, 500_000)

    assert not success
    assert bucket.value in message
    assert manager.miner_states[HOTKEY].current_bucket == bucket
    manager._entity_client.apply_bucket_account_size.assert_not_called()
    manager._position_client.close_all_positions.assert_not_called()


def test_unknown_hotkey_is_rejected(manager):
    success, message = manager.promote_pro_transition("not_a_miner_0", NOW_MS, 500_000)

    assert not success
    assert "not found" in message
    manager._entity_client.apply_bucket_account_size.assert_not_called()


def test_promotion_is_not_repeatable(manager):
    _in_transition(manager)
    assert manager.promote_pro_transition(HOTKEY, NOW_MS, 500_000)[0]

    success, message = manager.promote_pro_transition(HOTKEY, NOW_MS + 1000, 500_000)

    assert not success
    assert MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value in message


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Validator endpoint
# ═══════════════════════════════════════════════════════════════════════════════

PRO_TRANSITION_SIGNED_FIELDS = ("entity_coldkey", "entity_hotkey", "synthetic_hotkey", "nonce", "timestamp")


def _validator_pro_transition_client():
    """Flask test client for the validator endpoint with mocked clients and ownership check."""
    from vanta_api.nonce_manager import NonceManager
    from vanta_api.validator_rest_server import ValidatorRestServer

    server = object.__new__(ValidatorRestServer)
    server._entity_client = MagicMock()
    server._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "pro_account_size": 500_000, "account_size": 500_000,
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
        """A correctly signed request. `signed` overrides fields before signing (what a gateway would
        have sent), `drop` removes them before signing, `tampered` overrides after signing."""
        fields = {
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey.ss58_address,
            "synthetic_hotkey": self.synthetic,
            "pro_account_size": 500_000,
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
        resp = self.client.post("/entity/subaccount/pro-transition", json=body)
        return resp.status_code, json.loads(resp.data)

    def _promote_call(self):
        return self.server._challenge_period_client.promote_pro_transition.call_args.args

    def test_valid_request_reaches_the_challenge_period_client(self):
        status, data = self._post(self._body())
        self.assertEqual(status, 200)
        self.assertEqual(data['bucket'], MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value)
        self.assertEqual(data['pro_account_size'], 500_000)
        hotkey, _timestamp, size = self._promote_call()
        self.assertEqual(hotkey, self.synthetic)
        self.assertEqual(size, 500_000)

    def test_size_is_optional(self):
        """No size in the payload promotes on the size already recorded for the subaccount."""
        status, _ = self._post(self._body(drop=("pro_account_size",)))
        self.assertEqual(status, 200)
        self.assertIsNone(self._promote_call()[2])

    def test_manager_rejection_is_a_400(self):
        self.server._challenge_period_client.promote_pro_transition.return_value = (
            False, "entity_0 is in SUBACCOUNT_FUNDED, not PRO_CHALLENGE_TRANSITION"
        )
        status, data = self._post(self._body())
        self.assertEqual(status, 400)
        self.assertIn("not PRO_CHALLENGE_TRANSITION", data['error'])

    def test_bad_signature_is_401(self):
        other = Keypair.create_from_uri("//Charlie")
        status, _ = self._post(self._body(signature=other.sign(b"anything").hex()))
        self.assertEqual(status, 401)
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_coldkey_not_owning_hotkey_is_403(self):
        self.server._verify_coldkey_owns_hotkey.return_value = False
        status, _ = self._post(self._body())
        self.assertEqual(status, 403)
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_subaccount_of_another_entity_is_403(self):
        other = Keypair.create_from_uri("//Charlie")
        status, data = self._post(self._body(signed={"synthetic_hotkey": f"{other.ss58_address}_0"}))
        self.assertEqual(status, 403)
        self.assertIn("does not belong to entity", data['error'])
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_non_subaccount_hotkey_is_400(self):
        status, data = self._post(self._body(signed={"synthetic_hotkey": self.hotkey.ss58_address}))
        self.assertEqual(status, 400)
        self.assertIn("not a subaccount", data['error'])
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

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
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_invalid_sizes_are_400(self):
        for bad in (0, -1, "500000", True, ValiConfig.MAX_PRO_ACCOUNT_SIZE + 1):
            with self.subTest(pro_account_size=bad):
                status, data = self._post(self._body(signed={"pro_account_size": bad}))
                self.assertEqual(status, 400)
                self.assertIn("pro_account_size", data['error'])
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_max_size_is_allowed(self):
        status, _ = self._post(self._body(signed={"pro_account_size": ValiConfig.MAX_PRO_ACCOUNT_SIZE}))
        self.assertEqual(status, 200)
        self.assertEqual(self._promote_call()[2], ValiConfig.MAX_PRO_ACCOUNT_SIZE)

    # ==================== replay protection ====================

    def test_replayed_request_is_rejected(self):
        body = self._body()
        self.assertEqual(self._post(body)[0], 200)
        status, data = self._post(body)
        self.assertEqual(status, 401)
        self.assertIn("Nonce already used", data['error'])
        self.server._challenge_period_client.promote_pro_transition.assert_called_once()

    def test_signature_is_bound_to_every_signed_field(self):
        for field, value in (
            ("pro_account_size", 750_000),
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
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_a_size_cannot_be_added_to_a_signed_request(self):
        """A signature made without a size does not cover one bolted on in transit."""
        body = self._body(drop=("pro_account_size",))
        body["pro_account_size"] = ValiConfig.MAX_PRO_ACCOUNT_SIZE
        status, _ = self._post(body)
        self.assertEqual(status, 401)
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_expired_or_future_timestamp_is_rejected(self):
        now = TimeUtil.now_in_millis()
        for timestamp in (now - 6 * 60 * 1000, now + 2 * 60 * 1000):
            with self.subTest(timestamp=timestamp):
                status, _ = self._post(self._body(signed={"timestamp": timestamp}))
                self.assertEqual(status, 401)
        self.server._challenge_period_client.promote_pro_transition.assert_not_called()

    def test_nonce_is_consumed_only_after_signature_and_ownership_pass(self):
        body = self._body()
        other = Keypair.create_from_uri("//Charlie")
        self.assertEqual(self._post({**body, "signature": other.sign(b"anything").hex()})[0], 401)
        self.server._verify_coldkey_owns_hotkey.return_value = False
        self.assertEqual(self._post(body)[0], 403)
        self.server._verify_coldkey_owns_hotkey.return_value = True
        self.assertEqual(self._post(body)[0], 200)


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
        resp = self.client.post("/api/promote-pro-transition", json=body)
        return resp.status_code, json.loads(resp.data)

    def _forward(self, body=None):
        """Post to the gateway with the validator call patched; returns (status, data, payload sent)."""
        validator_resp = MagicMock(status_code=200)
        validator_resp.json.return_value = {"status": "success", "pro_account_size": 500_000}
        with patch("requests.post", return_value=validator_resp) as post:
            status, data = self._post(body if body is not None
                                      else {"synthetic_hotkey": self.synthetic, "pro_account_size": 500_000})
        self.assertEqual(post.call_args.args[0], "http://validator.test/entity/subaccount/pro-transition")
        return status, data, post.call_args.kwargs['json']

    def _assert_signed(self, payload, fields):
        signed = json.dumps({k: payload[k] for k in fields}, sort_keys=True).encode("utf-8")
        self.assertTrue(Keypair(ss58_address=self.coldkey.ss58_address).verify(
            signed, bytes.fromhex(payload['signature'])))

    def test_forwards_signed_payload_to_validator(self):
        status, data, payload = self._forward()
        self.assertEqual(status, 200)
        self.assertEqual(data['pro_account_size'], 500_000)
        self.assertEqual(payload['entity_coldkey'], self.coldkey.ss58_address)
        self.assertEqual(payload['entity_hotkey'], self.hotkey.ss58_address)
        self.assertEqual(payload['synthetic_hotkey'], self.synthetic)
        self.assertEqual(payload['pro_account_size'], 500_000)
        self.assertIsInstance(payload['nonce'], str)
        self.assertTrue(payload['nonce'])
        self.assertIsInstance(payload['timestamp'], int)
        self.assertLess(abs(payload['timestamp'] - TimeUtil.now_in_millis()), 60_000)
        self._assert_signed(payload, PRO_TRANSITION_SIGNED_FIELDS + ("pro_account_size",))

    def test_omitted_size_is_left_out_of_the_payload(self):
        _status, _data, payload = self._forward({"synthetic_hotkey": self.synthetic})
        self.assertNotIn('pro_account_size', payload)
        self._assert_signed(payload, PRO_TRANSITION_SIGNED_FIELDS)

    def test_each_request_gets_a_fresh_nonce(self):
        first = self._forward()[2]
        second = self._forward()[2]
        self.assertNotEqual(first['nonce'], second['nonce'])
        self.assertNotEqual(first['signature'], second['signature'])

    def test_gateway_payload_is_accepted_by_the_validator_endpoint(self):
        for body in ({"synthetic_hotkey": self.synthetic, "pro_account_size": 500_000},
                     {"synthetic_hotkey": self.synthetic}):
            with self.subTest(body=body):
                payload = self._forward(body)[2]
                server, client = _validator_pro_transition_client()
                resp = client.post("/entity/subaccount/pro-transition", json=payload)
                self.assertEqual(resp.status_code, 200, resp.data)
                server._challenge_period_client.promote_pro_transition.assert_called_once()
                self.assertEqual(
                    server._challenge_period_client.promote_pro_transition.call_args.args[2],
                    body.get("pro_account_size"),
                )
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
                {"synthetic_hotkey": self.synthetic, "pro_account_size": 0},
                {"synthetic_hotkey": self.synthetic, "pro_account_size": "500000"},
                {"synthetic_hotkey": self.synthetic, "pro_account_size": True},
                {"synthetic_hotkey": self.synthetic, "pro_account_size": ValiConfig.MAX_PRO_ACCOUNT_SIZE + 1},
                {"pro_account_size": 500_000},
                {"synthetic_hotkey": ""},
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
