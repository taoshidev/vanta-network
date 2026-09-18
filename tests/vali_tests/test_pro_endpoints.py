"""
Every HTTP surface a pro account is driven through, exercised against the real Flask handlers with
mocked RPC clients.

  * POST /api/promote on the gateway and POST /entity/subaccount/promote on the validator: the one
    promotion path on the network. The gateway signs the request with the entity coldkey and
    forwards it; the validator proves the entity owns the subaccount before anything moves. Covers
    the payload that is signed, the size validation, coldkey signature binding, ownership, and
    nonce + timestamp replay protection.
  * GET /subaccounts/<synthetic_hotkey>/limits: the Vanta-native counterpart to
    /hl-traders/<hl_address>/limits, which pro accounts cannot reach because they have no
    Hyperliquid address. Covers the flat pro curve and its caps, the correlated-exposure block, and
    the entity-collateral headroom (whose "unknown balance" case must not read as "no headroom").
  * GET /trade-pairs: the pro universe and the flat per-pair leverage table clients size orders
    against, plus the correlated-exposure limits published alongside it.
  * The subaccount dashboard payloads (v1, v2, hl-traders and the websocket frames) that carry the
    granted pro_account_size.

The rules behind these payloads are tested elsewhere: promotion and elimination in
test_challengeperiod_pro.py, account sizing in test_pro_account_size.py, and correlated exposure
enforcement in test_correlated_pair_limits.py.
"""
import json
import unittest
import uuid
from unittest.mock import MagicMock, patch

from bittensor_wallet import Keypair
from flask import Flask

from entity_management.entity_utils import create_subaccount_dashboard
from tests.vali_tests.test_pro_account_size import (
    GRANTED_SIZE,
    INVALID_SIZES,
    NOW_MS,
    STANDARD_SIZE,
    _add_pro,
    _add_standard,
    _bare_manager,
)
from time_util.time_util import TimeUtil
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount
from vali_objects.utils.leverage_utils import (
    get_all_correlation_group_limits,
    get_grandfathered_class_leverage,
    get_grandfathered_portfolio_leverage,
    get_grandfathered_positional_leverage,
    get_legacy_leverage_tier,
    get_legacy_tier_positional_leverage,
    get_pro_positional_leverage,
    get_standard_positional_leverage,
)
from vali_objects.vali_config import TradePair, TradePairCategory, ValiConfig
from vanta_api.validator_rest_server import ValidatorRestServer

PROMOTE_SIGNED_FIELDS = ("entity_coldkey", "entity_hotkey", "synthetic_hotkey", "nonce", "timestamp")
PRO_ACCOUNT_SIZE = 400_000.0
REMOVED_FIELD = "default_pro_account_size"


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
# Section 1 — POST /entity/subaccount/promote (validator)
# ═══════════════════════════════════════════════════════════════════════════════

def _validator_promote_client():
    """Flask test client for the validator endpoint with mocked clients and ownership check."""
    from vanta_api.nonce_manager import NonceManager

    server = object.__new__(ValidatorRestServer)
    server._get_api_key_safe = MagicMock(return_value="key")
    server.is_valid_api_key = MagicMock(return_value=True)
    server.can_access_tier = MagicMock(return_value=True)
    server._entity_client = MagicMock()
    server._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "pro_account_size": GRANTED_SIZE, "account_size": GRANTED_SIZE,
    }
    server._challenge_period_client = MagicMock()
    server._challenge_period_client.promote_subaccount.return_value = (True, "promoted")
    server._challenge_period_client.get_miner_bucket.return_value = MinerBucket.PRO_CHALLENGE_TRANSITION
    server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
    server.nonce_manager = NonceManager()
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/entity/subaccount/promote", methods=["POST"])(server.promote_subaccount)
    return server, app.test_client()


class TestValidatorPromoteEndpoint(unittest.TestCase):
    """POST /entity/subaccount/promote on the validator, with mocked RPC clients."""

    def setUp(self):
        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.synthetic = f"{self.hotkey.ss58_address}_0"
        self.server, self.client = _validator_promote_client()

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
        resp = self.client.post("/entity/subaccount/promote", data=json.dumps(body),
                                content_type="application/json")
        return resp.status_code, json.loads(resp.data)

    def _promote(self):
        return self.server._challenge_period_client.promote_subaccount

    # ==================== the size it forwards ====================

    def test_valid_request_forwards_the_requested_size(self):
        status, data = self._post(self._body(signed={"pro_account_size": GRANTED_SIZE}))
        self.assertEqual(status, 200, data)
        self.assertEqual(data['bucket'], MinerBucket.PRO_CHALLENGE_TRANSITION.value)
        self.assertEqual(data['pro_account_size'], GRANTED_SIZE)
        self._promote().assert_called_once()
        hotkey, timestamp, size = self._promote().call_args.args
        self.assertEqual(self._promote().call_args.kwargs, {})
        self.assertEqual(hotkey, self.synthetic)
        self.assertEqual(size, GRANTED_SIZE)
        self.assertLess(abs(timestamp - TimeUtil.now_in_millis()), 60_000)

    def test_a_request_without_a_size_forwards_none(self):
        """A hop within the pro track may omit the size; the recorded one is kept."""
        status, data = self._post(self._body())
        self.assertEqual(status, 200, data)
        self.assertIsNone(self._promote().call_args.args[2])

    def test_the_range_bounds_are_inclusive(self):
        for size in (STANDARD_SIZE, GRANTED_SIZE, ValiConfig.MAX_PRO_ACCOUNT_SIZE):
            with self.subTest(pro_account_size=size):
                status, data = self._post(self._body(signed={"pro_account_size": size}))
                self.assertEqual(status, 200, data)
                self.assertEqual(self._promote().call_args.args[2], size)

    def test_an_invalid_size_is_400_before_the_signature_and_ownership_checks(self):
        """Refused on the size, and without consuming the nonce, so a corrected retry still works."""
        for size, fragment in INVALID_SIZES:
            with self.subTest(pro_account_size=size):
                self._promote().reset_mock()
                self.server._verify_coldkey_owns_hotkey.reset_mock()
                body = self._body(signed={"pro_account_size": size})
                status, data = self._post(body)
                self.assertEqual(status, 400, data)
                self.assertIn(fragment, data['error'])
                self.server._verify_coldkey_owns_hotkey.assert_not_called()
                self._promote().assert_not_called()

                retry = self._body(signed={"nonce": body["nonce"], "timestamp": body["timestamp"],
                                           "pro_account_size": GRANTED_SIZE})
                self.assertEqual(self._post(retry)[0], 200)
                self._promote().assert_called_once()

    def test_the_size_is_part_of_what_is_signed(self):
        """One bolted on after signing, or stripped off it, breaks the signature."""
        status, _ = self._post(self._body(pro_account_size=GRANTED_SIZE))
        self.assertEqual(status, 401)

        body = self._body(signed={"pro_account_size": GRANTED_SIZE})
        del body["pro_account_size"]
        self.assertEqual(self._post(body)[0], 401)
        self._promote().assert_not_called()

    # ==================== auth, ownership and validation ====================

    def test_manager_rejection_is_a_400(self):
        self._promote().return_value = (False, "entity_0 cannot be promoted out of PRO_FUNDED")
        status, data = self._post(self._body())
        self.assertEqual(status, 400)
        self.assertIn("cannot be promoted out of PRO_FUNDED", data['error'])

    def test_invalid_api_key_is_401_and_below_tier_200_is_403(self):
        self.server.is_valid_api_key.return_value = False
        self.assertEqual(self._post(self._body())[0], 401)

        self.server.is_valid_api_key.return_value = True
        self.server.can_access_tier.return_value = False
        status, data = self._post(self._body())
        self.assertEqual(status, 403)
        self.assertIn("tier 200", data['error'])
        self.server.can_access_tier.assert_called_with("key", 200)
        self._promote().assert_not_called()

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

    def test_unknown_subaccount_is_404(self):
        self.server._entity_client.get_subaccount_info_for_synthetic.return_value = None
        status, data = self._post(self._body())
        self.assertEqual(status, 404)
        self.assertIn("not found", data['error'])
        self._promote().assert_not_called()

    def test_missing_and_malformed_fields_are_400(self):
        for field in ("synthetic_hotkey", "nonce", "timestamp", "signature"):
            with self.subTest(missing=field):
                body = self._body()
                del body[field]
                status, data = self._post(body)
                self.assertEqual(status, 400)
                self.assertIn(field, data['error'])
        for bad in ({"nonce": ""}, {"nonce": 123}, {"timestamp": "now"}, {"timestamp": 1.5},
                    {"timestamp": True}, {"synthetic_hotkey": ""}):
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
            ("pro_account_size", ValiConfig.MAX_PRO_ACCOUNT_SIZE),
        ):
            with self.subTest(tampered=field):
                body = self._body(signed={"pro_account_size": GRANTED_SIZE})
                self.assertNotEqual(body[field], value)
                body[field] = value
                status, _ = self._post(body)
                self.assertEqual(status, 401)
        self._promote().assert_not_called()

    def test_signature_covers_exactly_the_signed_fields(self):
        """The validator rebuilds only PROMOTE_SIGNED_FIELDS plus the size: more than that fails."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — POST /api/promote (gateway)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGatewayPromoteEndpoint(unittest.TestCase):
    """POST /api/promote on the gateway, with the validator call patched."""

    def setUp(self):
        from vanta_api.entity_miner_rest_server import EntityMinerRestServer

        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.synthetic = f"{self.hotkey.ss58_address}_0"
        self.gw = object.__new__(EntityMinerRestServer)
        self.gw._coldkey = self.coldkey
        self.gw._hotkey = self.hotkey
        self.gw._validator_url = "http://validator.test"
        self.gw._api_key = "validator-key"
        self.gw._get_api_key_safe = MagicMock(return_value="key")
        self.gw.is_valid_api_key = MagicMock(return_value=True)
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/api/promote", methods=["POST"])(self.gw.promote_endpoint)
        self.client = app.test_client()

    def _post(self, body):
        # json.dumps writes NaN / Infinity as bare literals, exactly what request.get_json() parses back
        resp = self.client.post("/api/promote", data=json.dumps(body), content_type="application/json")
        return resp.status_code, json.loads(resp.data)

    def _forward(self, body=None):
        """Post to the gateway with the validator call patched; returns (status, data, payload sent)."""
        validator_resp = MagicMock(status_code=200)
        validator_resp.json.return_value = {"status": "success", "pro_account_size": GRANTED_SIZE}
        if body is None:
            body = {"synthetic_hotkey": self.synthetic, "pro_account_size": GRANTED_SIZE}
        with patch("requests.post", return_value=validator_resp) as post:
            status, data = self._post(body)
        self.assertEqual(post.call_args.args[0], "http://validator.test/entity/subaccount/promote")
        return status, data, post.call_args.kwargs['json']

    def _assert_signed(self, payload, fields):
        signed = json.dumps({k: payload[k] for k in fields}, sort_keys=True).encode("utf-8")
        self.assertTrue(Keypair(ss58_address=self.coldkey.ss58_address).verify(
            signed, bytes.fromhex(payload['signature'])))

    def test_forwards_a_signed_payload_carrying_the_size(self):
        status, data, payload = self._forward()
        self.assertEqual(status, 200)
        self.assertEqual(data['pro_account_size'], GRANTED_SIZE)
        self.assertEqual(payload['entity_coldkey'], self.coldkey.ss58_address)
        self.assertEqual(payload['entity_hotkey'], self.hotkey.ss58_address)
        self.assertEqual(payload['synthetic_hotkey'], self.synthetic)
        self.assertEqual(payload['pro_account_size'], GRANTED_SIZE)
        self.assertIsInstance(payload['nonce'], str)
        self.assertTrue(payload['nonce'])
        self.assertIsInstance(payload['timestamp'], int)
        self.assertLess(abs(payload['timestamp'] - TimeUtil.now_in_millis()), 60_000)
        self.assertEqual(set(payload),
                         set(PROMOTE_SIGNED_FIELDS) | {"pro_account_size", "signature", "version"})
        self._assert_signed(payload, PROMOTE_SIGNED_FIELDS + ("pro_account_size",))

    def test_an_omitted_size_is_neither_signed_nor_sent(self):
        """Omitted must not become a signed null: the validator rebuilds the dict without the key."""
        _, _, payload = self._forward({"synthetic_hotkey": self.synthetic})
        self.assertNotIn("pro_account_size", payload)
        self.assertEqual(set(payload), set(PROMOTE_SIGNED_FIELDS) | {"signature", "version"})
        self._assert_signed(payload, PROMOTE_SIGNED_FIELDS)

    def test_an_invalid_size_is_refused_before_signing(self):
        self.gw._coldkey = MagicMock()
        for size, fragment in INVALID_SIZES:
            with self.subTest(pro_account_size=size), patch("requests.post") as post:
                status, data = self._post({"synthetic_hotkey": self.synthetic, "pro_account_size": size})
                self.assertEqual(status, 400, data)
                self.assertIn(fragment, data['message'])
                post.assert_not_called()
        self.gw._coldkey.sign.assert_not_called()

    def test_forwards_the_validator_api_key(self):
        """The validator endpoint needs tier 200, so the gateway sends its own key."""
        validator_resp = MagicMock(status_code=200)
        validator_resp.json.return_value = {"status": "success"}
        with patch("requests.post", return_value=validator_resp) as post:
            self._post({"synthetic_hotkey": self.synthetic})
        self.assertEqual(post.call_args.kwargs['headers']["Authorization"], "Bearer validator-key")

    def test_each_request_gets_a_fresh_nonce(self):
        first = self._forward()[2]
        second = self._forward()[2]
        self.assertNotEqual(first['nonce'], second['nonce'])
        self.assertNotEqual(first['signature'], second['signature'])

    def test_gateway_payload_is_accepted_by_the_validator_endpoint(self):
        """The contract between the two halves, with and without a size, replay included."""
        for body in ({"synthetic_hotkey": self.synthetic, "pro_account_size": GRANTED_SIZE},
                     {"synthetic_hotkey": self.synthetic}):
            with self.subTest(body=body):
                payload = self._forward(body)[2]
                server, client = _validator_promote_client()
                resp = client.post("/entity/subaccount/promote", json=payload)
                self.assertEqual(resp.status_code, 200, resp.data)
                promote = server._challenge_period_client.promote_subaccount
                promote.assert_called_once()
                self.assertEqual(promote.call_args.args[0], self.synthetic)
                self.assertEqual(promote.call_args.args[2], body.get("pro_account_size"))
                # The same bytes a second time are a replay
                self.assertEqual(client.post("/entity/subaccount/promote", json=payload).status_code, 401)

    def test_validator_error_is_passed_through(self):
        validator_resp = MagicMock(status_code=400)
        validator_resp.json.return_value = {"error": "cannot be promoted out of PRO_FUNDED"}
        with patch("requests.post", return_value=validator_resp):
            status, data = self._post({"synthetic_hotkey": self.synthetic})
        self.assertEqual(status, 400)
        self.assertIn("cannot be promoted out of PRO_FUNDED", data['message'])

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


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — GET /subaccounts/<synthetic_hotkey>/limits
# ═══════════════════════════════════════════════════════════════════════════════

LIMITS_HOTKEY = "5GhDr3xy" + "a" * 40 + "_0"  # {entity_hotkey}_{subaccount_id}


def _pro_account(exposures=None, bucket=MinerBucket.PRO_FUNDED):
    account = MinerAccount(
        miner_hotkey=LIMITS_HOTKEY,
        asset_class=MinerAssetClass.ALL_MARKETS,
        miner_bucket=bucket,
    )
    account.collateral_records = [CollateralRecord(PRO_ACCOUNT_SIZE, PRO_ACCOUNT_SIZE / 5000, 0, True)]
    account.correlated_exposure_by_group = exposures or {}
    return account


class TestProSubaccountLimitsEndpoint(unittest.TestCase):

    def setUp(self):
        self.server = object.__new__(ValidatorRestServer)
        self.server._entity_client = MagicMock()
        self.server._miner_account_client = MagicMock()
        self.server._entity_collateral_client = MagicMock()
        # Auth is exercised elsewhere; these tests are about the payload.
        self.server._get_access_error_response = lambda *a, **kw: None

        self.server._entity_client.get_subaccount_dashboard.return_value = {"account_type": "pro"}
        self.server._miner_account_client.get_account.return_value = _pro_account()
        self.server._entity_collateral_client.get_entity_collateral_headroom.return_value = 100.0
        self.server._entity_collateral_client.compute_subaccount_margin_requirement.return_value = 1_234.0

        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.route("/subaccounts/<synthetic_hotkey>/limits", methods=["GET"])(
            self.server.get_subaccount_limits
        )
        self.app.route("/trade-pairs", methods=["GET"])(self.server.get_allowed_trade_pairs)
        self.client = self.app.test_client()

    def _get(self, hotkey=LIMITS_HOTKEY):
        resp = self.client.get(f"/subaccounts/{hotkey}/limits")
        return resp.status_code, json.loads(resp.data)

    def test_pro_account_reports_the_flat_pro_curve_and_its_caps(self):
        status, data = self._get()

        self.assertEqual(status, 200)
        self.assertTrue(data["is_pro"])
        self.assertEqual(data["tier_curve"], "pro")
        # The pro curve is flat, so there is no tier for a client to key a table on.
        self.assertIsNone(data["tier"])
        self.assertEqual(data["portfolio_multiplier"], ValiConfig.PRO_PORTFOLIO_LEVERAGE)
        self.assertAlmostEqual(
            data["max_portfolio_usd"], PRO_ACCOUNT_SIZE * ValiConfig.PRO_PORTFOLIO_LEVERAGE
        )
        self.assertAlmostEqual(
            data["max_asset_class_usd"]["crypto"],
            PRO_ACCOUNT_SIZE * ValiConfig.PRO_CLASS_LEVERAGE[TradePairCategory.CRYPTO],
        )

    def test_correlated_exposure_is_reported_with_its_room(self):
        account = _pro_account({"currency:EUR": [PRO_ACCOUNT_SIZE * 20, 0.0]})
        self.server._miner_account_client.get_account.return_value = account

        _, data = self._get()

        group = data["correlation_limits"]["groups"]["currency:EUR"]
        limit = ValiConfig.PRO_CURRENCY_EXPOSURE_LIMITS["EUR"]
        self.assertEqual(group["limit_multiplier"], limit)
        self.assertAlmostEqual(group["long_room_usd"], PRO_ACCOUNT_SIZE * (limit - 20))
        self.assertAlmostEqual(group["short_room_usd"], PRO_ACCOUNT_SIZE * limit)

    def test_non_pro_account_gets_no_correlation_block(self):
        self.server._miner_account_client.get_account.return_value = _pro_account(
            bucket=MinerBucket.SUBACCOUNT_FUNDED)

        _, data = self._get()

        self.assertNotIn("correlation_limits", data)
        self.assertNotEqual(data["tier_curve"], "pro")

    def test_per_pair_caps_are_published_resolved_on_the_pro_curve(self):
        _, data = self._get()

        caps = data["positional_leverage"]
        self.assertEqual(caps["BTCUSDC"], get_pro_positional_leverage(TradePair.BTCUSDC))
        self.assertEqual(len(caps), sum(1 for tp in TradePair if MinerAssetClass.ALL_MARKETS.can_trade(tp, is_pro=True)))

    def test_pre_tier_standard_account_reports_tier_0_and_its_resolved_caps(self):
        # No stored leverage_tier on a $400K funded standard account: tier 0 against legacy tier 3.
        self.server._entity_client.get_subaccount_dashboard.return_value = {"account_type": "standard"}
        self.server._miner_account_client.get_account.return_value = _pro_account(bucket=MinerBucket.SUBACCOUNT_FUNDED)

        _, data = self._get()

        self.assertEqual(data["tier_curve"], "standard")
        legacy_tier = get_legacy_leverage_tier(MinerBucket.SUBACCOUNT_FUNDED, PRO_ACCOUNT_SIZE)
        self.assertEqual(legacy_tier, 3)
        # Reported as minus the legacy tier, the key of its floor rows in /trade-pairs
        self.assertEqual(data["tier"], -3)
        caps = data["positional_leverage"]
        self.assertEqual(caps["NVDA"], get_grandfathered_positional_leverage(legacy_tier, TradePair.NVDA))
        self.assertEqual(caps["NVDA"], 1.5)     # legacy 0.5 x 3 beats Base 0.5
        self.assertEqual(caps["EURUSD"], 10.0)  # Base 10 beats legacy 2.5 x 3
        self.assertEqual(len(caps), sum(1 for tp in TradePair if MinerAssetClass.ALL_MARKETS.can_trade(tp)))
        self.assertAlmostEqual(data["max_asset_class_usd"]["indices"], PRO_ACCOUNT_SIZE * 8.0)  # legacy 8 beats Base 3
        self.assertAlmostEqual(data["max_portfolio_usd"], PRO_ACCOUNT_SIZE * 18.0)  # legacy 18 beats Base 15

    def test_pre_tier_account_tier_keys_its_floor_rows_in_trade_pairs(self):
        # The UI contract: the `tier` limits reports keys the same rows in /trade-pairs that the
        # limits payload resolves, for per-pair, class and portfolio caps alike.
        self.server._entity_client.get_subaccount_dashboard.return_value = {"account_type": "standard"}
        self.server._miner_account_client.get_account.return_value = _pro_account(bucket=MinerBucket.SUBACCOUNT_FUNDED)
        _, limits = self._get()
        key = str(limits["tier"])
        self.assertEqual(key, "-3")

        pairs = json.loads(self.client.get("/trade-pairs").data)
        by_id = {entry["trade_pair_id"]: entry for entry in pairs["allowed"] + pairs["disabled"]}
        self.assertGreater(len(limits["positional_leverage"]), 1000)
        for pair_id, cap in limits["positional_leverage"].items():
            self.assertEqual(by_id[pair_id]["standard_positional_leverage_by_tier"][key], cap, pair_id)
        for category, usd in limits["max_asset_class_usd"].items():
            self.assertAlmostEqual(pairs["standard_leverage_tiers"]["class"][key][category] * PRO_ACCOUNT_SIZE, usd)
        self.assertAlmostEqual(
            pairs["standard_leverage_tiers"]["portfolio"][key]["all_markets"] * PRO_ACCOUNT_SIZE, limits["max_portfolio_usd"]
        )

    def test_per_pair_caps_for_a_stored_tier_and_a_legacy_account(self):
        stored = _pro_account(bucket=MinerBucket.SUBACCOUNT_FUNDED)
        stored.leverage_tier = 2
        self.server._miner_account_client.get_account.return_value = stored
        _, data = self._get()
        self.assertEqual((data["tier_curve"], data["tier"]), ("standard", 2))
        self.assertEqual(data["positional_leverage"]["BTCUSDC"], get_standard_positional_leverage(2, TradePair.BTCUSDC))

        hl = _pro_account(bucket=MinerBucket.SUBACCOUNT_FUNDED)
        hl.asset_class = MinerAssetClass.HL_ALL
        hl.hl_address = "0x" + "a" * 40
        self.server._miner_account_client.get_account.return_value = hl
        _, data = self._get()
        self.assertEqual((data["tier_curve"], data["tier"]), ("legacy", 3))
        self.assertEqual(data["positional_leverage"]["BTCUSDC"], get_legacy_tier_positional_leverage(3, TradePair.BTCUSDC))
        self.assertEqual(data["positional_leverage"]["BTCUSDC"], 1.5)

    def test_unknown_entity_balance_is_not_reported_as_zero_headroom(self):
        self.server._entity_collateral_client.get_entity_collateral_headroom.return_value = None

        _, data = self._get()

        self.assertIsNone(data["entity_collateral"]["headroom_theta"])
        self.assertIsNone(data["entity_collateral"]["headroom_usd"])

    def test_headroom_is_converted_to_usd(self):
        _, data = self._get()
        self.assertEqual(
            data["entity_collateral"]["headroom_usd"], 100.0 * ValiConfig.ENTITY_COLLATERAL_CPT_RISK
        )

    def test_non_subaccount_hotkey_is_rejected(self):
        status, _ = self._get("not_a_subaccount")
        self.assertEqual(status, 400)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — GET /trade-pairs: the pro universe and its leverage table
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradePairsEndpointProUniverse(unittest.TestCase):
    """The public table a pro client sizes orders against. No API key required."""

    def setUp(self):
        server = object.__new__(ValidatorRestServer)
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/trade-pairs", methods=["GET"])(server.get_allowed_trade_pairs)
        self.client = app.test_client()

    def _get(self, query=""):
        resp = self.client.get(f"/trade-pairs{query}")
        self.assertEqual(resp.status_code, 200, resp.data)
        return json.loads(resp.data)

    def test_per_pair_pro_leverage_matches_the_published_table(self):
        """One pair per category against the spec values, read off the payload a client gets."""
        expected = {
            'BTCUSDC': 5.0, 'ADAUSDC': 1.5, 'TRXUSDC': 1.0,
            'EURUSD': 20.0, 'NZDJPY': 10.0,
            'GOLDUSDC': 8.0, 'SILVERUSDC': 5.0,
            'SP500USDC': 10.0, 'EWYUSDC': 5.0,
            'NVDA': 2.0,
        }
        data = self._get()
        by_id = {entry['trade_pair_id']: entry for entry in data['allowed'] + data['disabled']}

        for trade_pair_id, leverage in expected.items():
            with self.subTest(trade_pair=trade_pair_id):
                self.assertEqual(by_id[trade_pair_id]['pro_positional_leverage'], leverage)
                # The order path resolves the same number from the same function
                self.assertEqual(
                    get_pro_positional_leverage(TradePair.from_trade_pair_id(trade_pair_id)), leverage
                )

    def test_tier_0_floor_rows_are_published_under_negative_keys(self):
        data = self._get()
        by_id = {entry['trade_pair_id']: entry for entry in data['allowed'] + data['disabled']}
        floor_keys = {"-1", "-2", "-3", "-4"}

        for legacy_tier in (1, 2, 3, 4):
            key = str(-legacy_tier)
            for trade_pair_id in ('BTCUSDC', 'ADAUSDC', 'EURUSD', 'EURNZD', 'SP500USDC', 'EWYUSDC', 'GOLDUSDC', 'NVDA'):
                with self.subTest(key=key, trade_pair=trade_pair_id):
                    tp = TradePair.from_trade_pair_id(trade_pair_id)
                    self.assertEqual(by_id[trade_pair_id]['standard_positional_leverage_by_tier'][key],
                                     get_grandfathered_positional_leverage(legacy_tier, tp))
            class_row = data['standard_leverage_tiers']['class'][key]
            portfolio_row = data['standard_leverage_tiers']['portfolio'][key]
            self.assertEqual(class_row, {cat.value: get_grandfathered_class_leverage(legacy_tier, cat)
                                         for cat in ValiConfig.STANDARD_CLASS_LEVERAGE_BY_TIER[1]})
            self.assertEqual(portfolio_row, {ac.value: get_grandfathered_portfolio_leverage(legacy_tier, ac)
                                             for ac in ValiConfig.STANDARD_PORTFOLIO_LEVERAGE_BY_TIER[1]})

        # the tiers a client could select are still exactly 1 to 3 next to the floor keys
        self.assertEqual(set(by_id['BTCUSDC']['standard_positional_leverage_by_tier']), {"1", "2", "3"} | floor_keys)
        self.assertEqual(set(data['standard_leverage_tiers']['class']), {"1", "2", "3"} | floor_keys)
        self.assertEqual(set(data['standard_leverage_tiers']['portfolio']), {"1", "2", "3"} | floor_keys)
        # spot values: old funded (-2) keeps NVDA 1.0 and the indices class cap 6.0, old challenge (-1) NVDA 0.5
        self.assertEqual(by_id['NVDA']['standard_positional_leverage_by_tier']['-2'], 1.0)
        self.assertEqual(by_id['NVDA']['standard_positional_leverage_by_tier']['-1'], 0.5)
        self.assertEqual(by_id['EURUSD']['standard_positional_leverage_by_tier']['-1'], 10.0)
        self.assertEqual(data['standard_leverage_tiers']['class']['-2']['indices'], 6.0)
        self.assertEqual(data['standard_leverage_tiers']['portfolio']['-2']['all_markets'], 15.0)

    def test_is_pro_restricts_the_allowed_list_to_the_pro_universe(self):
        default = self._get()
        pro = self._get("?is_pro=true")

        self.assertFalse(default['is_pro'])
        self.assertTrue(pro['is_pro'])
        pro_allowed = {entry['trade_pair_id'] for entry in pro['allowed']}
        self.assertEqual(pro_allowed, {tp.trade_pair_id for tp in TradePair
                                       if tp.is_pro and not tp.is_blocked})
        # Every pair still appears somewhere, so a client can see why one was withheld
        self.assertEqual(len(pro['allowed']) + len(pro['disabled']), len(list(TradePair)))
        self.assertLess(len(pro['allowed']), len(default['allowed']))
        for entry in pro['disabled']:
            self.assertFalse(entry['is_pro'] and not TradePair.from_trade_pair_id(
                entry['trade_pair_id']).is_blocked)

    def test_the_pro_block_carries_every_cap_a_pro_account_is_sized_against(self):
        block = self._get()['pro']

        self.assertEqual(block['portfolio_leverage'], ValiConfig.PRO_PORTFOLIO_LEVERAGE)
        self.assertEqual(block['default_positional_leverage'], ValiConfig.PRO_DEFAULT_POSITIONAL_LEVERAGE)
        self.assertEqual(block['class_leverage'],
                         {cat.value: cap for cat, cap in ValiConfig.PRO_CLASS_LEVERAGE.items()})
        self.assertEqual(set(block['allowed_trade_pair_ids']),
                         {tp.trade_pair_id for tp in TradePair if tp.is_pro and not tp.is_blocked})

    def test_the_correlated_exposure_limits_are_published_with_their_basis(self):
        """Gross per side against the balance, not the account size - a client that assumes
        otherwise sizes every correlated order wrong."""
        block = self._get()['pro']

        self.assertEqual(block['basis'], 'gross_per_side')
        self.assertEqual(block['denominator'], 'balance')
        self.assertEqual(block['correlation_limits'], get_all_correlation_group_limits())
        self.assertEqual(block['currency_limits'], dict(ValiConfig.PRO_CURRENCY_EXPOSURE_LIMITS))
        self.assertEqual(block['sector_limit'], ValiConfig.PRO_SECTOR_EXPOSURE_LIMIT)
        self.assertEqual(block['us_index_limit'], ValiConfig.PRO_US_INDEX_EXPOSURE_LIMIT)
        self.assertEqual(set(block['us_index_trade_pair_ids']), set(ValiConfig.PRO_US_INDEX_TRADE_PAIR_IDS))

    def test_each_pair_reports_the_legs_a_long_position_contributes_to(self):
        """`correlation_legs` is what the trade box has to replicate to size an order."""
        by_id = {entry['trade_pair_id']: entry for entry in self._get()['allowed'] + self._get()['disabled']}

        eurusd = by_id['EURUSD']['correlation_legs']
        self.assertEqual({leg['group']: leg['direction'] for leg in eurusd},
                         {'currency:EUR': 1.0, 'currency:USD': -1.0})
        # The group keys are the published contract: a client keys its own book on these strings
        self.assertEqual([leg['group'] for leg in by_id['SP500USDC']['correlation_legs']], ['index:us'])
        self.assertEqual(by_id['NVDA']['correlation_legs'][0]['group'].split(':')[0], 'sector')
        self.assertEqual(by_id['BTCUSDC']['correlation_legs'], [])
        self.assertLessEqual({leg['group'] for entry in by_id.values()
                              for leg in entry['correlation_legs']},
                             set(get_all_correlation_group_limits()))


class TestProLeverageCurveReporting(unittest.TestCase):
    """A pro account trades its own flat curve, and the dashboard has to say so.

    `leverage_tier` alone is None for pro, so without this block a client cannot tell which
    table in /trade-pairs to size against. The pro curve has no tier dimension, so `tier` is
    None there -- a client must not fall back to a tiered table for it.
    """

    @staticmethod
    def _account(bucket, account_size):
        account = MinerAccount(
            miner_hotkey="entity_alpha_0",
            asset_class=MinerAssetClass.ALL_MARKETS,
            miner_bucket=bucket,
            leverage_tier=1,
        )
        account.collateral_records = [CollateralRecord(account_size, account_size / 5000, 0, True)]
        return account

    def test_curve_and_tier_by_bucket(self):
        cases = [
            # Every pro bucket gets the flat pro curve, regardless of account size.
            (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 400_000, "pro", None),
            (MinerBucket.PRO_CHALLENGE_DIRECT, 400_000, "pro", None),
            (MinerBucket.PRO_FUNDED, 400_000, "pro", None),
            (MinerBucket.PRO_FUNDED, 1_000_000, "pro", None),
            # Transition still trades the standard account, so it keeps the standard curve.
            (MinerBucket.PRO_CHALLENGE_TRANSITION, 100_000, "standard", 1),
            (MinerBucket.SUBACCOUNT_CHALLENGE, 100_000, "standard", 1),
        ]
        for bucket, account_size, curve, tier in cases:
            with self.subTest(bucket=bucket, account_size=account_size):
                account = self._account(bucket, account_size)
                limits = account.leverage_limits()
                self.assertEqual(limits["tier_curve"], curve)
                self.assertEqual(limits["tier"], tier)
                self.assertEqual(limits["is_pro"], bucket.is_pro)
                self.assertEqual(account.to_dashboard()["is_pro"], bucket.is_pro)
                if bucket.is_pro:
                    self.assertEqual(limits["portfolio_multiplier"], ValiConfig.PRO_PORTFOLIO_LEVERAGE)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — the subaccount dashboard payloads
#
# subaccount_info carries the size actually granted to this subaccount and no network-wide
# default_pro_account_size, in every payload that builds it.
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubaccountInfoProAccountSize(unittest.TestCase):
    """Off the wire: the two builders every payload goes through."""

    def setUp(self):
        self.manager = _bare_manager()
        self.standard = _add_standard(self.manager)
        self.pro = _add_pro(self.manager)
        self.hl = _add_standard(self.manager, subaccount_id=2)
        self.manager.get_subaccount_info_for_synthetic(self.hl).hl_address = "0x" + "ab" * 20

    def test_v1_subaccount_info_carries_no_network_default(self):
        for hotkey in (self.standard, self.pro, self.hl):
            with self.subTest(hotkey=hotkey):
                info = self.manager.get_subaccount_dashboard_data(hotkey)["subaccount_info"]
                self.assertNotIn(REMOVED_FIELD, info)

    def test_v2_subaccount_info_keeps_the_granted_size(self):
        clients = _no_section_clients()
        for hotkey, granted in ((self.standard, None), (self.pro, GRANTED_SIZE), (self.hl, None)):
            with self.subTest(hotkey=hotkey):
                dashboard = create_subaccount_dashboard(
                    hotkey,
                    self.manager.get_subaccount_dashboard(hotkey),
                    clients["challenge_period"], clients["elimination"], clients["miner_account"],
                    clients["position"], clients["limit_order"], clients["debt_ledger"],
                    clients["statistics"],
                    0, 0, 0, 0,
                )
                info = dashboard["subaccount_info"]
                self.assertNotIn(REMOVED_FIELD, info)
                self.assertEqual(info["pro_account_size"], granted)


class TestDashboardEndpointsProAccountSize(unittest.TestCase):
    """On the wire, through the real validator REST handlers and websocket frame builder."""

    def setUp(self):
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
        self.server = server
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.route("/entity/subaccount/<synthetic_hotkey>", methods=["GET"])(server.get_subaccount_dashboard)
        app.route("/v2/entity/subaccount/<synthetic_hotkey>", methods=["GET"])(server.v2_get_subaccount_dashboard)
        app.route("/hl-traders/<hl_address>", methods=["GET"])(server.get_hl_trader)
        self.client = app.test_client()

    def _get(self, path):
        resp = self.client.get(path)
        self.assertEqual(resp.status_code, 200, resp.data)
        return json.loads(resp.data)["dashboard"]["subaccount_info"]

    def test_v1_and_hl_trader_endpoints(self):
        for hotkey in (self.standard, self.pro):
            with self.subTest(hotkey=hotkey):
                self.assertNotIn(REMOVED_FIELD, self._get(f"/entity/subaccount/{hotkey}"))

        self.server._entity_client.get_synthetic_hotkey_for_hl_address.return_value = self.standard
        info = self._get("/hl-traders/0x" + "ab" * 20)
        self.assertEqual(info["synthetic_hotkey"], self.standard)
        self.assertNotIn(REMOVED_FIELD, info)

    def test_v2_endpoint_keeps_the_granted_size(self):
        for hotkey, granted in ((self.standard, None), (self.pro, GRANTED_SIZE)):
            with self.subTest(hotkey=hotkey):
                info = self._get(f"/v2/entity/subaccount/{hotkey}")
                self.assertNotIn(REMOVED_FIELD, info)
                self.assertEqual(info["pro_account_size"], granted)

    def test_websocket_frames(self):
        """The websocket rebuilds subaccount_info in full on each frame, including incremental ones."""
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
                    self.assertNotIn(REMOVED_FIELD, info)
                    self.assertEqual(info["pro_account_size"], granted)


if __name__ == "__main__":
    unittest.main()
