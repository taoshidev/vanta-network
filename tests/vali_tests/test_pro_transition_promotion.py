"""
Unit tests for the entity-initiated promotion up the pro track.

There is one promotion path on the network: the gateway's POST /api/promote signs the request with the
entity coldkey and forwards it to the validator's POST /entity/subaccount/promote. The entity picks the
pro account size it is buying; the validator proves the entity owns the subaccount before anything moves.

Covers:
  * ChallengePeriodManager.promote_subaccount: the three hops it allows and the buckets it refuses, the
    account switch that closes positions / cancels limit orders / restarts the ledgers, and the sizing
    (through a real EntityManager) including a re-offer after a demotion and the rollback of a failed move.
  * The validator HTTP endpoint: coldkey signature bound to every field including the size, subaccount
    ownership, nonce + timestamp replay protection, field and size validation, and the size it forwards.
  * The gateway HTTP endpoint: the payload it signs and forwards, a bad size refused before signing, and
    validator errors passed through.
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
    INVALID_SIZES,
    REQUIRED,
    STANDARD_SIZE,
    _add_demoted,
    _add_pro,
    _add_standard as _add_standard_subaccount,
    _bare_manager as _bare_entity_manager,
)

NOW_MS = 1_748_000_000_000
HOTKEY = "entity_0"

# The only bucket moves an entity can ask for, source -> target.
HOPS = (
    (MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.PRO_CHALLENGE_DIRECT),
    (MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.PRO_CHALLENGE_TRANSITION),
    (MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD),
)
# Buckets with nowhere to promote to: the end of the pro track, the standard buckets off it, and
# the regular (non-subaccount) miner buckets.
NO_PROMOTION = (
    MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
    MinerBucket.PRO_CHALLENGE_DIRECT,
    MinerBucket.PRO_FUNDED,
    MinerBucket.SUBACCOUNT_ALPHA,
    MinerBucket.ELIMINATED,
    MinerBucket.MAINCOMP,
    MinerBucket.CHALLENGE,
)

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
# Section 1 — ChallengePeriodManager.promote_subaccount
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


def _in_bucket(manager, bucket):
    manager.set_miner_bucket(HOTKEY, bucket, NOW_MS)
    return manager


@pytest.mark.parametrize("source,target", HOPS)
def test_each_hop_moves_to_its_own_target(manager, source, target):
    _in_bucket(manager, source)

    success, message = manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)

    assert success, message
    assert manager.miner_states[HOTKEY].current_bucket == target
    manager._entity_client.apply_bucket_account_size.assert_any_call(HOTKEY, target, GRANTED_SIZE)


def test_an_omitted_size_is_passed_through_as_none(manager):
    """A hop within the track may leave the size out; the entity manager falls back to the recorded one."""
    _in_bucket(manager, MinerBucket.PRO_CHALLENGE_TRANSITION)

    assert manager.promote_subaccount(HOTKEY, NOW_MS)[0]

    manager._entity_client.apply_bucket_account_size.assert_any_call(
        HOTKEY, MinerBucket.PRO_CHALLENGE_FROM_STANDARD, None
    )


def test_promotion_winds_down_the_standard_account(manager):
    _in_bucket(manager, MinerBucket.PRO_CHALLENGE_TRANSITION)

    assert manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)[0]

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


def test_entering_the_transition_wipes_nothing_and_only_sweeps_entry_orders(manager):
    """PRO_CHALLENGE_TRANSITION is a wind-down week on the standard account, so this hop keeps the
    subaccount's positions, limit orders and ledgers. Only the resting orders that could open or
    increase a position are cancelled; closes and reductions stay for the week."""
    _in_bucket(manager, MinerBucket.SUBACCOUNT_FUNDED)

    assert manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)[0]

    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    manager._position_client.close_all_positions.assert_not_called()
    manager._position_client.archive_positions_for_hotkey.assert_not_called()
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_not_called()
    manager._debt_ledger_client.delete_debt_ledger.assert_not_called()
    manager._miner_account_client.reset_account.assert_not_called()
    # the blanket "cancel everything" sweep belongs to an account switch; this hop is not one
    manager._limit_order_client.cancel_limit_order.assert_not_called()
    manager._limit_order_client.cancel_entry_orders.assert_called_once_with(
        HOTKEY, NOW_MS, OrderSource.PRO_TRANSITION_CANCELLED
    )


@pytest.mark.parametrize("source,target", [h for h in HOPS if h[1] != MinerBucket.PRO_CHALLENGE_TRANSITION])
def test_only_the_hops_onto_a_pro_account_wipe_trading_state(manager, source, target):
    """The counterpart: both hops that land on a pro account do run the full wind-down."""
    _in_bucket(manager, source)

    assert manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)[0]

    assert target.switches_account
    manager._position_client.close_all_positions.assert_called_once_with(
        hotkey=HOTKEY, close_time_ms=NOW_MS, order_source=OrderSource.SUBACCOUNT_PROMOTION
    )
    manager._limit_order_client.cancel_limit_order.assert_called_once_with(HOTKEY, None, "ALL", NOW_MS)
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_called_once_with([HOTKEY])
    manager._debt_ledger_client.delete_debt_ledger.assert_called_once_with(HOTKEY)
    manager._limit_order_client.cancel_entry_orders.assert_not_called()


@pytest.mark.parametrize(
    "challenge_bucket",
    [MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_FROM_STANDARD],
)
def test_pro_funded_keeps_the_account_it_passed_on(manager, challenge_bucket):
    """Passing the pro challenge keeps the same account.

    Balance, equity, positions and the ledgers all carry over, so all-time calmar keeps the ratio
    the miner passed with. Challenge-period gains are kept out of the payout by the payout paths
    reading each checkpoint's own bucket, not by wiping the history.
    """
    manager.set_miner_bucket(HOTKEY, challenge_bucket, NOW_MS)

    assert manager.promote_hotkeys([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_FUNDED
    assert not MinerBucket.PRO_FUNDED.switches_account
    manager._position_client.close_all_positions.assert_not_called()
    manager._position_client.archive_positions_for_hotkey.assert_not_called()
    manager._limit_order_client.cancel_limit_order.assert_not_called()
    manager._perf_ledger_client.wipe_miners_perf_ledgers.assert_not_called()
    manager._debt_ledger_client.delete_debt_ledger.assert_not_called()
    manager._miner_account_client.reset_account.assert_not_called()


def test_entity_rejection_blocks_the_promotion(manager):
    """When the entity manager cannot size the pro account, nothing is wound down."""
    _in_bucket(manager, MinerBucket.PRO_CHALLENGE_TRANSITION)
    manager._entity_client.apply_bucket_account_size.return_value = (False, REQUIRED)

    success, message = manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)

    assert not success
    assert REQUIRED in message
    assert manager.miner_states[HOTKEY].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    manager._position_client.close_all_positions.assert_not_called()


@pytest.mark.parametrize("bucket", NO_PROMOTION)
def test_buckets_off_the_promotion_path_are_refused(manager, bucket):
    _in_bucket(manager, bucket)

    success, message = manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)

    assert not success
    assert bucket.value in message
    assert manager.miner_states[HOTKEY].current_bucket == bucket
    manager._entity_client.apply_bucket_account_size.assert_not_called()
    manager._position_client.close_all_positions.assert_not_called()


def test_unknown_hotkey_is_rejected(manager):
    success, message = manager.promote_subaccount("not_a_miner_0", NOW_MS, GRANTED_SIZE)

    assert not success
    assert "not found" in message
    manager._entity_client.apply_bucket_account_size.assert_not_called()


def test_promotion_stops_at_the_end_of_the_track(manager):
    """Each hop is single use: the target of the last one has no promotion of its own."""
    _in_bucket(manager, MinerBucket.PRO_CHALLENGE_TRANSITION)
    assert manager.promote_subaccount(HOTKEY, NOW_MS, GRANTED_SIZE)[0]

    success, message = manager.promote_subaccount(HOTKEY, NOW_MS + 1000, GRANTED_SIZE)

    assert not success
    assert MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value in message


# ==================== sizing through a real EntityManager ====================

def _with_real_entity_manager(manager, bucket=MinerBucket.SUBACCOUNT_FUNDED,
                              hotkey_factory=_add_standard_subaccount):
    """Swap the mocked entity client for a real in-memory EntityManager holding one standard
    subaccount. Returns (entity_manager, synthetic_hotkey)."""
    entity_manager = _bare_entity_manager()
    hotkey = hotkey_factory(entity_manager)
    manager._entity_client = entity_manager
    manager.set_miner_bucket(hotkey, bucket, NOW_MS)
    return entity_manager, hotkey


def test_transition_records_the_size_without_trading_it(manager):
    """Entering the transition records the pro size but keeps trading the standard account."""
    entity_manager, hotkey = _with_real_entity_manager(manager)

    success, message = manager.promote_subaccount(hotkey, NOW_MS, 250_000)

    assert success, message
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 250_000
    assert info.standard_account_size == STANDARD_SIZE
    assert info.account_size == STANDARD_SIZE


def test_promotion_out_of_the_transition_trades_the_pro_size(manager):
    entity_manager, hotkey = _with_real_entity_manager(manager)
    assert manager.promote_subaccount(hotkey, NOW_MS, 250_000)[0]

    success, message = manager.promote_subaccount(hotkey, NOW_MS + 1000)

    assert success, message
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 250_000
    assert info.account_size == 250_000
    assert info.standard_account_size == STANDARD_SIZE
    assert entity_manager.get_payout_scale(hotkey) == pytest.approx(STANDARD_SIZE / 250_000)


def test_direct_pro_challenge_trades_the_size_immediately(manager):
    """The hop off the standard challenge goes straight onto the pro account."""
    entity_manager, hotkey = _with_real_entity_manager(manager, bucket=MinerBucket.SUBACCOUNT_CHALLENGE)

    assert manager.promote_subaccount(hotkey, NOW_MS, 450_000)[0]

    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_DIRECT
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 450_000
    assert info.account_size == 450_000


def test_entering_the_track_without_a_size_is_rejected(manager):
    """A subaccount that never had a size set (nothing to fall back on) is not promoted onto a pro
    account of unknown size."""
    entity_manager, hotkey = _with_real_entity_manager(manager)

    success, message = manager.promote_subaccount(hotkey, NOW_MS)

    assert not success
    assert REQUIRED in message
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.SUBACCOUNT_FUNDED
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size is None
    assert info.account_size == STANDARD_SIZE
    manager._position_client.close_all_positions.assert_not_called()


def test_organic_promotion_keeps_the_recorded_size(manager):
    """The end-of-week auto promotion and the pro funded promotion send no size either."""
    entity_manager, hotkey = _with_real_entity_manager(manager)
    assert manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)[0]

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

    success, message = manager.promote_subaccount(hotkey, NOW_MS)
    assert not success
    assert REQUIRED in message

    assert manager.promote_subaccount(hotkey, NOW_MS, 200_000)[0]
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.pro_account_size == 200_000


def test_an_out_of_range_size_never_reaches_the_account(manager):
    """The REST layer checks the range first, but the entity manager is the last line of defence."""
    entity_manager, hotkey = _with_real_entity_manager(manager)

    for size, fragment in INVALID_SIZES:
        success, message = manager.promote_subaccount(hotkey, NOW_MS, size)
        assert not success, size
        assert fragment in message, size
        assert manager.miner_states[hotkey].current_bucket == MinerBucket.SUBACCOUNT_FUNDED
        assert entity_manager.get_subaccount_info_for_synthetic(hotkey).pro_account_size is None


def test_promotion_target_is_never_the_bucket_the_miner_is_already_in(manager):
    """Why promote_subaccount needs no "already in that bucket" precheck before it commits the
    sizing: no hop is a self-loop, so that rejection is unreachable here by construction."""
    for source, target in HOPS:
        assert source != target


def test_an_unaffordable_promotion_fee_is_refused_before_anything_is_written(manager):
    """The fee is the one predictable failure that lives inside the committing call, so it has to be
    checked before the write, not compensated after it."""
    entity_manager, hotkey = _with_real_entity_manager(manager, bucket=MinerBucket.SUBACCOUNT_CHALLENGE)
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    info.reg_fee_theta = 1.0  # collateral-exempt subaccounts (0.0) never pay the promotion fee
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 0.5

    success, message = manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)

    assert not success
    assert "Insufficient collateral" in message
    assert manager.miner_states[hotkey].current_bucket == MinerBucket.SUBACCOUNT_CHALLENGE
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_type == "standard"
    assert info.pro_account_size is None
    assert info.standard_account_size is None
    assert info.account_size == STANDARD_SIZE
    assert info.pro_fee_theta == 0.0
    assert info.pro_fee_theta_pending == 0.0
    # neither the live account nor the collateral reservation was touched
    entity_manager._miner_account_client.set_miner_account_size.assert_not_called()
    entity_manager._entity_collateral_client.offset_collateral_cache.assert_not_called()
    manager._position_client.close_all_positions.assert_not_called()


def test_a_failed_promotion_rolls_back_the_pro_sizing(manager):
    """The pro account is sized before the bucket moves, so a failed move must put the sizing back:
    PRO_CHALLENGE_FROM_STANDARD trades the pro size, and a subaccount left holding it while still in
    PRO_CHALLENGE_TRANSITION is supposed to be trading the standard account."""
    entity_manager, hotkey = _with_real_entity_manager(manager)
    assert manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)[0]
    before = entity_manager.get_subaccount_info_for_synthetic(hotkey).model_copy()
    assert before.account_size == STANDARD_SIZE

    with patch.object(manager, "admin_set_bucket", return_value=(False, "bucket move failed")):
        success, message = manager.promote_subaccount(hotkey, NOW_MS)

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


def test_a_crash_before_the_bucket_moves_rolls_the_sizing_back(manager):
    """_switch_account runs before the bucket entry, so a crash there means no move happened."""
    entity_manager, hotkey = _with_real_entity_manager(manager)
    assert manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)[0]

    with patch.object(manager, "_switch_account", side_effect=RuntimeError("rpc down")):
        with pytest.raises(RuntimeError):
            manager.promote_subaccount(hotkey, NOW_MS + 1000)

    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_TRANSITION
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_size == STANDARD_SIZE  # TRANSITION still trades the standard account


def test_a_crash_after_the_bucket_moves_keeps_the_sizing(manager):
    """The disk write and the entry-order sweep run after the bucket entry has landed. Rolling the
    sizing back there would leave a pro bucket trading the standard size."""
    entity_manager, hotkey = _with_real_entity_manager(manager)
    assert manager.promote_subaccount(hotkey, NOW_MS, GRANTED_SIZE)[0]

    with patch.object(manager, "_save_to_disk", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError):
            manager.promote_subaccount(hotkey, NOW_MS + 1000)

    assert manager.miner_states[hotkey].current_bucket == MinerBucket.PRO_CHALLENGE_FROM_STANDARD
    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_type == "pro"
    assert info.account_size == GRANTED_SIZE
    assert info.pro_account_size == GRANTED_SIZE


def test_a_failed_first_hop_leaves_no_pro_marking(manager):
    """A subaccount left marked pro would trade the pro size in a standard bucket, and the next
    size-less attempt would take the "recorded" branch instead of demanding a size."""
    entity_manager, hotkey = _with_real_entity_manager(manager, bucket=MinerBucket.SUBACCOUNT_CHALLENGE)

    with patch.object(manager, "admin_set_bucket", return_value=(False, "bucket move failed")):
        assert not manager.promote_subaccount(hotkey, NOW_MS, 500_000)[0]

    info = entity_manager.get_subaccount_info_for_synthetic(hotkey)
    assert info.account_type == "standard"
    assert info.pro_account_size is None
    assert info.standard_account_size is None
    assert info.account_size == STANDARD_SIZE

    # behavioural backstop: the promotion did not happen, so a size is still required
    success, message = manager.promote_subaccount(hotkey, NOW_MS)
    assert not success
    assert REQUIRED in message


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Validator endpoint
# ═══════════════════════════════════════════════════════════════════════════════

PROMOTE_SIGNED_FIELDS = ("entity_coldkey", "entity_hotkey", "synthetic_hotkey", "nonce", "timestamp")


def _validator_promote_client():
    """Flask test client for the validator endpoint with mocked clients and ownership check."""
    from vanta_api.nonce_manager import NonceManager
    from vanta_api.validator_rest_server import ValidatorRestServer

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

    def test_a_size_added_to_a_signed_request_is_401(self):
        """The size is part of what is signed, so one bolted on afterwards breaks the signature."""
        status, _ = self._post(self._body(pro_account_size=GRANTED_SIZE))
        self.assertEqual(status, 401)
        self._promote().assert_not_called()

    def test_a_size_stripped_from_a_signed_request_is_401(self):
        body = self._body(signed={"pro_account_size": GRANTED_SIZE})
        del body["pro_account_size"]
        status, _ = self._post(body)
        self.assertEqual(status, 401)
        self._promote().assert_not_called()

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
# Section 3 — Gateway endpoint
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


class TestProLeverageCurveReporting(unittest.TestCase):
    """A pro account trades its own flat curve, and the dashboard has to say so.

    `leverage_tier` alone is None for pro, so without this block a client cannot tell which
    table in /trade-pairs to size against. The pro curve has no tier dimension, so `tier` is
    None there -- a client must not fall back to a tiered table for it.
    """

    @staticmethod
    def _account(bucket, account_size):
        from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
        from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount

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


if __name__ == "__main__":
    unittest.main()
