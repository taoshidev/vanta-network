"""
Subaccount intraday drawdown threshold (daily loss limit): one of ValiConfig.SUBACCOUNT_INTRADAY_DRAWDOWN_VALUES,
chosen at creation, that replaces the bucket's intraday drawdown threshold in every standard and pro bucket except
PRO_FUNDED, which always runs the pro defaults. A subaccount that chooses none keeps each bucket's default threshold.
The other rule (static or EOD) is unchanged.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from bittensor_wallet import Keypair
from flask import Flask

from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager is a pytest fixture
    DAILY_MS,
    HOTKEY,
    MIDNIGHT_MS,
    NOW_MS,
    _elimination_kwargs,
    _eod_breach,
    _run_refresh,
    manager,
)
from vali_objects.challenge_period.challengeperiod_manager import DrawdownStats, MinerBucketState
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.vali_config import ValiConfig

SUBACCOUNT_BUCKETS = (
    MinerBucket.SUBACCOUNT_CHALLENGE,
    MinerBucket.SUBACCOUNT_FUNDED,
    MinerBucket.PRO_CHALLENGE_TRANSITION,
    MinerBucket.PRO_CHALLENGE_DIRECT,
    MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
    MinerBucket.PRO_FUNDED,
)
OVERRIDE_BUCKETS = tuple(b for b in SUBACCOUNT_BUCKETS if b != MinerBucket.PRO_FUNDED)
CRITERIA = (DrawdownCriteria.STATIC, DrawdownCriteria.TRAILING)
DAY_OPEN = 1.10  # up on the starting balance, so the static rule never binds
HL_ADDRESS = "0x" + "c" * 40


def _threshold_state(bucket: MinerBucket, intraday_drawdown_threshold: float | None,
                     criteria: DrawdownCriteria = DrawdownCriteria.STATIC) -> MinerBucketState:
    return MinerBucketState(HOTKEY, [BucketEntry(bucket, NOW_MS - DAILY_MS)], drawdown_criteria=criteria,
                            intraday_drawdown_threshold_override=intraday_drawdown_threshold)


def _below_day_open(drop: float) -> DrawdownStats:
    """Live equity `drop` below the day's open, with the EOD mark and starting balance both clear."""
    equity = DAY_OPEN * (1 - drop)
    return DrawdownStats(current_equity=equity, current_balance=equity, daily_open_equity=DAY_OPEN,
                         eod_hwm=DAY_OPEN, last_eod_equity=DAY_OPEN, last_eod_checked_ms=MIDNIGHT_MS)


def _seed(manager, bucket: MinerBucket, drawdown: DrawdownStats, intraday_drawdown_threshold: float | None,
          criteria: DrawdownCriteria = DrawdownCriteria.STATIC) -> None:
    manager.set_miner_bucket(HOTKEY, bucket, NOW_MS - DAILY_MS, drawdown_criteria=criteria,
                             intraday_drawdown_threshold=intraday_drawdown_threshold)
    manager.miner_states[HOTKEY].drawdown = drawdown


# ── Validation ────────────────────────────────────────────────────────────────

# ── Threshold resolution ──────────────────────────────────────────────────────

@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("bucket", OVERRIDE_BUCKETS)
@pytest.mark.parametrize("threshold", ValiConfig.SUBACCOUNT_INTRADAY_DRAWDOWN_VALUES)
def test_the_chosen_threshold_is_the_intraday_threshold_in_every_bucket(threshold, bucket, criteria):
    state = _threshold_state(bucket, threshold, criteria)
    assert state.intraday_drawdown_threshold == threshold
    assert state.intraday_drawdown_threshold_pct == pytest.approx(threshold * 100)


@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("threshold", ValiConfig.SUBACCOUNT_INTRADAY_DRAWDOWN_VALUES)
def test_pro_funded_ignores_the_chosen_threshold(threshold, criteria):
    state = _threshold_state(MinerBucket.PRO_FUNDED, threshold, criteria)
    assert state.intraday_drawdown_threshold == ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD


@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("bucket", SUBACCOUNT_BUCKETS)
def test_the_chosen_threshold_leaves_the_eod_threshold_alone(bucket, criteria):
    assert (_threshold_state(bucket, 0.03, criteria).eod_drawdown_threshold
            == _threshold_state(bucket, None, criteria).eod_drawdown_threshold)


def test_no_chosen_threshold_keeps_each_bucket_default():
    assert _threshold_state(MinerBucket.SUBACCOUNT_FUNDED, None).intraday_drawdown_threshold == \
        ValiConfig.SUBACCOUNT_STATIC_INTRADAY_DRAWDOWN_THRESHOLD
    assert _threshold_state(MinerBucket.PRO_FUNDED, None).intraday_drawdown_threshold == \
        ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD
    assert _threshold_state(MinerBucket.SUBACCOUNT_CHALLENGE, None, DrawdownCriteria.TRAILING).intraday_drawdown_threshold == \
        ValiConfig.CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD

    # A trailing account registered before the V0 cutoff keeps its legacy funded threshold
    assert NOW_MS < ValiConfig.FUNDED_V0_CUTOFF_MS
    legacy = MinerBucketState(HOTKEY, [
        BucketEntry(MinerBucket.SUBACCOUNT_CHALLENGE, NOW_MS - 10 * DAILY_MS),
        BucketEntry(MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - DAILY_MS),
    ])
    assert legacy.intraday_drawdown_threshold == ValiConfig.FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V0


def test_the_chosen_threshold_carries_through_every_promotion_until_pro_funded(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, NOW_MS - 4 * DAILY_MS,
                             drawdown_criteria=DrawdownCriteria.STATIC, intraday_drawdown_threshold=0.03)
    pro_default = ValiConfig.PRO_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD
    for offset, (bucket, expected) in enumerate(((MinerBucket.SUBACCOUNT_FUNDED, 0.03),
                                                 (MinerBucket.PRO_CHALLENGE_TRANSITION, 0.03),
                                                 (MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 0.03),
                                                 (MinerBucket.PRO_FUNDED, pro_default),
                                                 (MinerBucket.ELIMINATED, pro_default))):
        manager.set_miner_bucket(HOTKEY, bucket, NOW_MS - (3 - offset) * DAILY_MS)
        assert manager.miner_states[HOTKEY].current_bucket == bucket
        assert manager.miner_states[HOTKEY].intraday_drawdown_threshold == expected


def test_the_chosen_threshold_is_write_once(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, NOW_MS - DAILY_MS, intraday_drawdown_threshold=0.03)
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS, intraday_drawdown_threshold=0.05)
    assert manager.miner_states[HOTKEY].intraday_drawdown_threshold_override == 0.03


def test_the_dashboard_reports_the_chosen_threshold_as_the_intraday_threshold(manager):
    manager.set_miner_bucket(HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE, NOW_MS - DAILY_MS,
                             drawdown_criteria=DrawdownCriteria.STATIC, intraday_drawdown_threshold=0.03)
    assert manager.get_drawdown_stats(HOTKEY)["intraday_drawdown_threshold"] == 0.03


# ── Checkpoint ────────────────────────────────────────────────────────────────

def test_the_chosen_threshold_round_trips_through_the_checkpoint():
    state = _threshold_state(MinerBucket.PRO_CHALLENGE_FROM_STANDARD, 0.03)
    restored = MinerBucketState.from_checkpoint_dict(HOTKEY, state.to_checkpoint_dict())
    assert restored.intraday_drawdown_threshold_override == 0.03
    assert restored.intraday_drawdown_threshold == 0.03


def test_a_checkpoint_written_before_the_field_loads_with_no_limit():
    data = _threshold_state(MinerBucket.SUBACCOUNT_FUNDED, None).to_checkpoint_dict()
    data.pop("intraday_drawdown_threshold_override")
    restored = MinerBucketState.from_checkpoint_dict(HOTKEY, data)
    assert restored.intraday_drawdown_threshold_override is None
    assert restored.intraday_drawdown_threshold == ValiConfig.SUBACCOUNT_STATIC_INTRADAY_DRAWDOWN_THRESHOLD


# ── Elimination ───────────────────────────────────────────────────────────────

INTRADAY_REASON = {
    MinerBucket.SUBACCOUNT_CHALLENGE: EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN,
    MinerBucket.SUBACCOUNT_FUNDED: EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN,
    MinerBucket.PRO_CHALLENGE_DIRECT: EliminationReason.FAILED_PRO_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN,
    MinerBucket.PRO_FUNDED: EliminationReason.FAILED_PRO_FUNDED_PERIOD_INTRADAY_DRAWDOWN,
}


@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("bucket", tuple(b for b in INTRADAY_REASON if b != MinerBucket.PRO_FUNDED))
@pytest.mark.parametrize("threshold", ValiConfig.SUBACCOUNT_INTRADAY_DRAWDOWN_VALUES)
def test_a_drop_just_past_the_chosen_threshold_eliminates(manager, threshold, bucket, criteria):
    _seed(manager, bucket, _below_day_open(threshold + 0.001), threshold, criteria)
    _run_refresh(manager, HOTKEY)
    assert manager.get_miner_bucket(HOTKEY) == MinerBucket.ELIMINATED
    assert _elimination_kwargs(manager)["reason"] == INTRADAY_REASON[bucket]


@pytest.mark.parametrize("criteria", CRITERIA)
@pytest.mark.parametrize("bucket", tuple(INTRADAY_REASON))
@pytest.mark.parametrize("threshold", ValiConfig.SUBACCOUNT_INTRADAY_DRAWDOWN_VALUES)
def test_a_drop_just_inside_the_chosen_threshold_survives(manager, threshold, bucket, criteria):
    _seed(manager, bucket, _below_day_open(threshold - 0.001), threshold, criteria)
    _run_refresh(manager, HOTKEY)
    assert manager.get_miner_bucket(HOTKEY) == bucket


def test_the_pro_eod_rule_still_binds_with_a_chosen_threshold(manager):
    _seed(manager, MinerBucket.PRO_FUNDED, _eod_breach(), 0.03, DrawdownCriteria.TRAILING)
    _run_refresh(manager, HOTKEY)
    assert _elimination_kwargs(manager)["reason"] == EliminationReason.FAILED_PRO_FUNDED_PERIOD_EOD_DRAWDOWN


def test_the_static_rule_still_binds_with_a_chosen_threshold(manager):
    breach = 1.0 - ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD - 0.001
    _seed(manager, MinerBucket.SUBACCOUNT_FUNDED,
          DrawdownStats(current_equity=breach, daily_open_equity=breach, eod_hwm=1.0, last_eod_equity=breach),
          0.03)
    _run_refresh(manager, HOTKEY)
    assert _elimination_kwargs(manager)["reason"] == EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN


# ── REST: validator /entity/create-subaccount ─────────────────────────────────


@pytest.fixture
def keys():
    return Keypair.create_from_uri("//Alice"), Keypair.create_from_uri("//Bob")


@pytest.fixture
def validator(keys):
    """Flask test client for the validator create endpoint with a mocked entity client."""
    from vanta_api.validator_rest_server import ValidatorRestServer

    server = object.__new__(ValidatorRestServer)
    server._entity_client = MagicMock()
    server._entity_client.create_subaccount.return_value = (True, {"synthetic_hotkey": "hk_0"}, "created")
    server._entity_client.create_hl_subaccount.return_value = (True, {"synthetic_hotkey": "hk_0"}, "created")
    server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/entity/create-subaccount", methods=["POST"])(server.create_subaccount)
    return server, app.test_client()


def _signed_create_body(keys, hl: bool = False, unsigned: dict | None = None, **options) -> dict:
    """A create body as the gateway sends it: the legacy field set plus any options, all signed.
    `unsigned` is merged in after signing, to model a request altered in transit."""
    coldkey, hotkey = keys
    signed = {"account_size": 100_000.0, "asset_class": "hl_all" if hl else "crypto",
              "entity_coldkey": coldkey.ss58_address, "entity_hotkey": hotkey.ss58_address, **options}
    if hl:
        signed["hl_address"] = HL_ADDRESS
    message = json.dumps(signed, sort_keys=True).encode("utf-8")
    return {**signed, "signature": coldkey.sign(message).hex(), "version": ValiConfig.VANTA_CLI_MINIMUM_VERSION,
            **(unsigned or {})}


INSTANT_FUNDED = {"bucket": MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value, "pro_account_size": 500_000,
                  "eod_hwm_threshold": 0.08, "intraday_drawdown_threshold": 0.03}


@pytest.mark.parametrize("hl", [False, True])
def test_validator_forwards_the_chosen_threshold(validator, keys, hl):
    server, client = validator
    resp = client.post("/entity/create-subaccount", json=_signed_create_body(keys, hl=hl, intraday_drawdown_threshold=0.03))
    assert resp.status_code == 200, resp.data
    create = server._entity_client.create_hl_subaccount if hl else server._entity_client.create_subaccount
    assert create.call_args.kwargs["intraday_drawdown_threshold"] == 0.03


@pytest.mark.parametrize("hl", [False, True])
def test_validator_forwards_none_when_omitted(validator, keys, hl):
    server, client = validator
    resp = client.post("/entity/create-subaccount", json=_signed_create_body(keys, hl=hl))
    assert resp.status_code == 200, resp.data
    create = server._entity_client.create_hl_subaccount if hl else server._entity_client.create_subaccount
    assert create.call_args.kwargs["intraday_drawdown_threshold"] is None


def test_validator_forwards_signed_instant_funded_options(validator, keys):
    server, client = validator
    resp = client.post("/entity/create-subaccount", json=_signed_create_body(keys, **INSTANT_FUNDED))
    assert resp.status_code == 200, resp.data
    kwargs = server._entity_client.create_subaccount.call_args.kwargs
    assert {name: kwargs[name] for name in INSTANT_FUNDED} == INSTANT_FUNDED
    assert "payout_scale" not in kwargs


@pytest.mark.parametrize("name", sorted(INSTANT_FUNDED))
def test_validator_rejects_an_option_added_after_signing(validator, keys, name):
    """Every creation option is covered by the signature: adding one in transit fails verification."""
    server, client = validator
    signed = {k: v for k, v in INSTANT_FUNDED.items() if k != name}
    body = _signed_create_body(keys, unsigned={name: INSTANT_FUNDED[name]}, **signed)
    resp = client.post("/entity/create-subaccount", json=body)
    assert resp.status_code == 401, resp.data
    server._entity_client.create_subaccount.assert_not_called()


def test_validator_rejects_a_pro_account_size_changed_after_signing(validator, keys):
    server, client = validator
    body = _signed_create_body(keys, unsigned={"pro_account_size": 1_000_000}, **INSTANT_FUNDED)
    resp = client.post("/entity/create-subaccount", json=body)
    assert resp.status_code == 401, resp.data
    server._entity_client.create_subaccount.assert_not_called()


def test_validator_ignores_payout_scale(validator, keys):
    server, client = validator
    body = _signed_create_body(keys, unsigned={"payout_scale": 2.0}, **INSTANT_FUNDED)
    resp = client.post("/entity/create-subaccount", json=body)
    assert resp.status_code == 200, resp.data
    assert "payout_scale" not in server._entity_client.create_subaccount.call_args.kwargs


@pytest.mark.parametrize("bad", [0.04, 0.07 - 0.04, 0.02, 0.06, 0.035, 3, 5, "0.03", True, [0.03]])
def test_validator_rejects_an_invalid_limit(validator, keys, bad):
    server, client = validator
    resp = client.post("/entity/create-subaccount", json=_signed_create_body(keys, intraday_drawdown_threshold=bad))
    assert resp.status_code == 400
    assert "intraday_drawdown_threshold" in json.loads(resp.data)["error"]
    server._entity_client.create_subaccount.assert_not_called()


# ── REST: gateway /api/create-subaccount ──────────────────────────────────────

@pytest.fixture
def gateway(keys):
    from vanta_api.entity_miner_rest_server import EntityMinerRestServer

    coldkey, hotkey = keys
    gw = object.__new__(EntityMinerRestServer)
    gw._coldkey = coldkey
    gw._hotkey = hotkey
    gw._validator_url = "http://validator.test"
    gw._get_api_key_safe = MagicMock(return_value="key")
    gw.is_valid_api_key = MagicMock(return_value=True)
    gw._max_hl_traders = None
    gw._hl_to_synthetic = {}
    gw._set_hl_mapping = MagicMock()
    gw._save_hl_mappings = MagicMock()
    gw.slack_notifier = None
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/api/create-subaccount", methods=["POST"])(gw.create_subaccount_endpoint)
    return app.test_client()


def _gateway_post(client, body):
    validator_resp = MagicMock(status_code=200)
    validator_resp.json.return_value = {"status": "success", "subaccount": {"synthetic_hotkey": "hk_0"}}
    with patch("requests.post", return_value=validator_resp) as post:
        resp = client.post("/api/create-subaccount", json=body)
    return resp, post


def _assert_signed_over(keys, payload, *option_names):
    signed = {k: payload[k] for k in ("account_size", "asset_class", "entity_coldkey", "entity_hotkey",
                                      *option_names)}
    if "hl_address" in payload:
        signed["hl_address"] = payload["hl_address"]
    message = json.dumps(signed, sort_keys=True).encode("utf-8")
    assert Keypair(ss58_address=keys[0].ss58_address).verify(message, bytes.fromhex(payload["signature"]))


@pytest.mark.parametrize("hl", [False, True])
def test_gateway_signs_the_chosen_threshold(gateway, keys, hl):
    body = {"account_size": 100_000, "intraday_drawdown_threshold": 0.03}
    body.update({"hl_address": HL_ADDRESS} if hl else {"asset_class": "crypto"})
    resp, post = _gateway_post(gateway, body)
    assert resp.status_code == 200, resp.data
    payload = post.call_args.kwargs["json"]
    assert payload["intraday_drawdown_threshold"] == 0.03
    _assert_signed_over(keys, payload, "intraday_drawdown_threshold")


def test_gateway_signs_instant_funded_options(gateway, keys):
    resp, post = _gateway_post(gateway, {"account_size": 100_000, "asset_class": "crypto",
                                         "payout_scale": 2.0, **INSTANT_FUNDED})
    assert resp.status_code == 200, resp.data
    payload = post.call_args.kwargs["json"]
    assert {name: payload[name] for name in INSTANT_FUNDED} == INSTANT_FUNDED
    assert "payout_scale" not in payload
    _assert_signed_over(keys, payload, *INSTANT_FUNDED)


def test_gateway_signs_only_the_legacy_field_set_without_options(gateway, keys):
    """A request with no creation options signs exactly what a legacy gateway signs, so either
    gateway verifies against either validator."""
    resp, post = _gateway_post(gateway, {"account_size": 100_000, "asset_class": "crypto"})
    assert resp.status_code == 200, resp.data
    payload = post.call_args.kwargs["json"]
    for name in INSTANT_FUNDED:
        assert name not in payload
    _assert_signed_over(keys, payload)


@pytest.mark.parametrize("bad", [0.04, 0.07 - 0.04, 0.02, 0.06, 0.035, 3, 5, "0.03", True, [0.03]])
def test_gateway_rejects_an_invalid_limit(gateway, bad):
    resp, post = _gateway_post(gateway, {"account_size": 100_000, "asset_class": "crypto", "intraday_drawdown_threshold": bad})
    assert resp.status_code == 400
    assert "intraday_drawdown_threshold" in json.loads(resp.data)["message"]
    post.assert_not_called()


# ── REST: Hyperliquid Instant Funded ──────────────────────────────────────────

HL_INSTANT_FUNDED = {"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "eod_hwm_threshold": 0.05,
                     "intraday_drawdown_threshold": 0.03}


def test_validator_forwards_hl_instant_funded_options(validator, keys):
    server, client = validator
    resp = client.post("/entity/create-subaccount", json=_signed_create_body(keys, hl=True, **HL_INSTANT_FUNDED))
    assert resp.status_code == 200, resp.data
    kwargs = server._entity_client.create_hl_subaccount.call_args.kwargs
    assert {name: kwargs[name] for name in HL_INSTANT_FUNDED} == HL_INSTANT_FUNDED


@pytest.mark.parametrize("options,fragment", [
    ({"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "pro_account_size": 500_000}, "Hyperliquid"),
    ({"bucket": MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value, "pro_account_size": 500_000}, "Hyperliquid"),
    ({"eod_hwm_threshold": 0.05}, "eod_hwm_threshold"),
])
def test_validator_rejects_hl_options_outside_instant_funded(validator, keys, options, fragment):
    server, client = validator
    resp = client.post("/entity/create-subaccount", json=_signed_create_body(keys, hl=True, **options))
    assert resp.status_code == 400
    assert fragment in json.loads(resp.data)["error"]
    server._entity_client.create_hl_subaccount.assert_not_called()


def test_gateway_signs_hl_instant_funded_options(gateway, keys):
    resp, post = _gateway_post(gateway, {"account_size": 100_000, "hl_address": HL_ADDRESS, **HL_INSTANT_FUNDED})
    assert resp.status_code == 200, resp.data
    payload = post.call_args.kwargs["json"]
    assert {name: payload[name] for name in HL_INSTANT_FUNDED} == HL_INSTANT_FUNDED
    _assert_signed_over(keys, payload, *HL_INSTANT_FUNDED)


@pytest.mark.parametrize("options", [{"eod_hwm_threshold": 0.05},
                                     {"bucket": MinerBucket.SUBACCOUNT_FUNDED.value, "pro_account_size": 500_000}])
def test_gateway_rejects_hl_options_outside_instant_funded(gateway, options):
    resp, post = _gateway_post(gateway, {"account_size": 100_000, "hl_address": HL_ADDRESS, **options})
    assert resp.status_code == 400
    post.assert_not_called()
