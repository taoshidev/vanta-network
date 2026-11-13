"""
The PRO_CHALLENGE_FROM_STANDARD payout multiplier fixed on an account with its pro size: the one of the multiple an
auto-promotion grants (ValiConfig.AUTO_PROMOTION_PAYOUT_MULTIPLIERS), else the lowest. Standard miners always take
it; subaccounts from ValiConfig.SUBACCOUNT_PROMOTION_RULES_START_MS on, older ones keep PRO_TRANSITION_PAYOUT_MULTIPLIER.
"""
from unittest.mock import MagicMock, patch

import pytest

from entity_management.entity_client import EntityClient
from entity_management.entity_manager import EntityData, EntityManager
from entity_management.entity_server import EntityServer
from entity_management.entity_utils import is_on_promotion_rules
from tests.vali_tests.test_auto_promotion import (
    HELD_SINCE_MS,
    _auto_promote,
    _candidate,
    _EntityClient,
    _stats,
)
from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager and hop_manager are fixtures
    DAILY_MS,
    NOW_MS,
    hop_manager,
    manager,
)
from tests.vali_tests.test_pro_account_size import NOW_MS as CREATED_MS
from tests.vali_tests.test_pro_account_size import ENTITY_HOTKEY, STANDARD_SIZE, _add_standard, _bare_manager
from tests.vali_tests.test_standard_miner_pipeline import SIZE, record
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.standard_miner import PipelineAccountInfo, bucket_sizing
from vali_objects.vali_config import RPCConnectionMode, ValiConfig

TRANSITION = MinerBucket.PRO_CHALLENGE_TRANSITION
GROW = MinerBucket.PRO_CHALLENGE_FROM_STANDARD
LOWEST = min(ValiConfig.AUTO_PROMOTION_PAYOUT_MULTIPLIERS.values())


def _starting(start_ms):
    return patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS", start_ms)


@pytest.fixture
def entity_manager():
    return _bare_manager()


@pytest.fixture
def on_the_rules():
    """The switch set to the moment the test subaccounts are created."""
    with _starting(CREATED_MS):
        yield


def _info(entity_manager, hotkey):
    return entity_manager.get_subaccount_info_for_synthetic(hotkey)


# ── The table and the switch ──────────────────────────────────────────────────

def test_every_multiple_a_track_grants_has_a_multiplier():
    granted = {ValiConfig.AUTO_PROMOTION_FAST_TRACK_MULTIPLIER,
               *(multiple for _, multiple in ValiConfig.AUTO_PROMOTION_STEADY_TRACK_CALMAR_MULTIPLIERS)}
    assert granted <= set(ValiConfig.AUTO_PROMOTION_PAYOUT_MULTIPLIERS)


@pytest.mark.parametrize("multiple,expected", [(3, 1.3), (5, 1.5), (10, 2.0), (None, 1.3), (4, 1.3)])
def test_the_multiplier_of_each_multiple_and_the_lowest_for_any_other_grant(multiple, expected):
    assert ValiConfig.grow_payout_multiplier(multiple) == expected


def test_the_switch_is_off_until_it_is_set():
    assert ValiConfig.SUBACCOUNT_PROMOTION_RULES_START_MS is None
    assert ValiConfig.SUBACCOUNT_PROMOTION_RULES_START_MS_TESTNET is None
    assert not is_on_promotion_rules(CREATED_MS) and not is_on_promotion_rules(CREATED_MS, is_testnet=True)


def test_a_subaccount_is_on_the_rules_from_its_networks_start():
    with _starting(CREATED_MS), patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS_TESTNET",
                                             CREATED_MS + 10):
        assert is_on_promotion_rules(CREATED_MS) and is_on_promotion_rules(CREATED_MS + 1)
        assert not is_on_promotion_rules(CREATED_MS - 1)
        assert not is_on_promotion_rules(None)
        assert not is_on_promotion_rules(CREATED_MS, is_testnet=True)
        assert is_on_promotion_rules(CREATED_MS + 10, is_testnet=True)


# ── The entity manager's sizing ───────────────────────────────────────────────

def test_the_first_grant_fixes_the_multiplier_and_the_moves_after_it_keep_it(entity_manager, on_the_rules):
    hotkey = _add_standard(entity_manager)
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    assert _info(entity_manager, hotkey).payout_scale == 1.3

    assert entity_manager.apply_bucket_account_size(hotkey, GROW, payout_scale=2.0)[0]
    assert entity_manager.apply_bucket_account_size(hotkey, GROW, 5 * STANDARD_SIZE, payout_scale=2.0)[0]
    info = _info(entity_manager, hotkey)
    assert info.payout_scale == 1.3
    assert entity_manager.get_payout_scale(hotkey) == pytest.approx(1.3 * STANDARD_SIZE / info.pro_account_size)


@pytest.mark.parametrize("start_ms", [None, CREATED_MS + 1])
def test_a_subaccount_off_the_rules_keeps_the_old_multiplier(entity_manager, start_ms):
    hotkey = _add_standard(entity_manager)
    with _starting(start_ms):
        assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    assert _info(entity_manager, hotkey).payout_scale is None
    assert entity_manager.get_payout_scale(hotkey) == pytest.approx(ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER / 3)


def test_any_first_grant_fixes_one_the_lowest_unless_one_is_given(entity_manager, on_the_rules):
    """An operator's grant names no multiplier, and the direct pro challenge never reads the one it fixes."""
    unpriced = _add_standard(entity_manager, subaccount_id=0)
    assert entity_manager.apply_bucket_account_size(unpriced, TRANSITION, 3 * STANDARD_SIZE)[0]
    direct = _add_standard(entity_manager, subaccount_id=1)
    assert entity_manager.apply_bucket_account_size(direct, MinerBucket.PRO_CHALLENGE_DIRECT, 3 * STANDARD_SIZE)[0]
    # Moved into the transition without a pro size, then sized on the hop into Grow
    late = _add_standard(entity_manager, subaccount_id=2)
    assert entity_manager.apply_bucket_account_size(late, GROW, 3 * STANDARD_SIZE)[0]
    assert [_info(entity_manager, hotkey).payout_scale for hotkey in (unpriced, direct, late)] == [LOWEST] * 3


def test_an_account_that_already_has_a_pro_size_keeps_its_multiplier(entity_manager, on_the_rules):
    """Promoted before its validator read the switch, it stays on what it was promoted with, resized or not."""
    hotkey = _add_standard(entity_manager)
    info = _info(entity_manager, hotkey)
    info.account_type, info.standard_account_size, info.pro_account_size = "pro", STANDARD_SIZE, 3 * STANDARD_SIZE
    assert entity_manager.apply_bucket_account_size(hotkey, GROW, payout_scale=1.5)[0]
    assert entity_manager.apply_bucket_account_size(hotkey, GROW, 5 * STANDARD_SIZE, payout_scale=1.5)[0]
    assert _info(entity_manager, hotkey).payout_scale is None


def test_a_validator_reads_its_own_networks_switch(entity_manager):
    entity_manager.is_testnet = True
    mainnet_start = _add_standard(entity_manager, subaccount_id=0)
    with _starting(CREATED_MS):
        assert entity_manager.apply_bucket_account_size(mainnet_start, TRANSITION, 3 * STANDARD_SIZE)[0]
    testnet_start = _add_standard(entity_manager, subaccount_id=1)
    with patch.object(ValiConfig, "SUBACCOUNT_PROMOTION_RULES_START_MS_TESTNET", CREATED_MS):
        assert entity_manager.apply_bucket_account_size(testnet_start, TRANSITION, 3 * STANDARD_SIZE)[0]
    assert _info(entity_manager, mainnet_start).payout_scale is None
    assert _info(entity_manager, testnet_start).payout_scale == LOWEST


def test_a_multiplier_already_set_is_never_replaced(entity_manager, on_the_rules):
    hotkey = _add_standard(entity_manager)
    _info(entity_manager, hotkey).payout_scale = ValiConfig.GROW_DIRECT_CREATION_PAYOUT_MULTIPLIER
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    assert _info(entity_manager, hotkey).payout_scale == ValiConfig.GROW_DIRECT_CREATION_PAYOUT_MULTIPLIER


def test_a_rolled_back_promotion_takes_its_multiplier_back(entity_manager, on_the_rules):
    hotkey = _add_standard(entity_manager)
    snapshot = entity_manager.snapshot_bucket_account_size(hotkey)
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    assert entity_manager.restore_bucket_account_size(hotkey, snapshot)[0]
    assert _info(entity_manager, hotkey).payout_scale is None


def test_the_multiplier_travels_with_the_pro_sizing(entity_manager, on_the_rules):
    hotkey = _add_standard(entity_manager)
    peer = _info(entity_manager, hotkey).model_copy()
    standard = peer.model_copy()
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    promoted = _info(entity_manager, hotkey)

    assert EntityManager.adopt_pro_sizing(peer, promoted)  # a checkpoint fills a peer that missed the broadcast
    assert peer.payout_scale == 1.3
    assert EntityManager.adopt_pro_sizing(peer, standard, allow_clear=True)  # a broadcast of a rollback clears it
    assert peer.payout_scale is None


def test_a_sync_fills_a_missing_multiplier_on_a_pro_record_and_never_replaces_one(entity_manager, on_the_rules):
    hotkey = _add_standard(entity_manager)
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    promoted = _info(entity_manager, hotkey)
    missing = promoted.model_copy(update={"payout_scale": None})
    assert EntityManager.adopt_pro_sizing(missing, promoted)
    assert missing.payout_scale == 1.3
    instant_funded = promoted.model_copy(update={"payout_scale": ValiConfig.GROW_DIRECT_CREATION_PAYOUT_MULTIPLIER})
    assert not EntityManager.adopt_pro_sizing(instant_funded, promoted)
    assert not EntityManager.adopt_pro_sizing(promoted.model_copy(), missing.model_copy(update={"payout_scale": None}))
    assert instant_funded.payout_scale == ValiConfig.GROW_DIRECT_CREATION_PAYOUT_MULTIPLIER


def test_a_promotion_after_a_move_back_keeps_the_first_multiplier(entity_manager, on_the_rules):
    hotkey = _add_standard(entity_manager)
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    assert entity_manager.apply_bucket_account_size(hotkey, MinerBucket.SUBACCOUNT_FUNDED)[0]
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 10 * STANDARD_SIZE, payout_scale=2.0)[0]
    assert _info(entity_manager, hotkey).payout_scale == 1.3


def test_the_payloads_report_the_multiplier(entity_manager, on_the_rules):
    hotkey = _add_standard(entity_manager)
    assert entity_manager.apply_bucket_account_size(hotkey, TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.3)[0]
    assert entity_manager.get_subaccount_dashboard(hotkey)["payout_scale"] == 1.3
    assert entity_manager.get_subaccount_dashboard_data(hotkey)["subaccount_info"]["payout_scale"] == 1.3


def test_creation_into_a_pro_bucket_with_the_switch_on(entity_manager):
    """Instant Funded keeps its own 1.0; the direct pro challenge fixes the lowest, which it never reads."""
    entity_manager._entity_collateral_client = MagicMock()
    entity_manager._entity_collateral_client.get_cached_collateral.return_value = 10_000.0
    entity_manager._uuid_to_hotkey, entity_manager._hl_address_to_synthetic, entity_manager._client_ref_index = {}, {}, {}
    entity_manager.entities[ENTITY_HOTKEY] = EntityData(entity_hotkey=ENTITY_HOTKEY, registered_at_ms=CREATED_MS)
    with _starting(0):
        instant, grow_info, message = entity_manager.create_subaccount_ex(
            ENTITY_HOTKEY, account_size=STANDARD_SIZE, asset_class="crypto", bucket=GROW.value,
            pro_account_size=5 * STANDARD_SIZE, eod_hwm_threshold=0.05)[:3]
        assert instant, message
        direct, direct_info, message = entity_manager.create_subaccount_ex(
            ENTITY_HOTKEY, account_size=STANDARD_SIZE, asset_class="crypto",
            bucket=MinerBucket.PRO_CHALLENGE_DIRECT.value, pro_account_size=5 * STANDARD_SIZE)[:3]
        assert direct, message
    assert grow_info.payout_scale == ValiConfig.GROW_DIRECT_CREATION_PAYOUT_MULTIPLIER
    assert direct_info.payout_scale == LOWEST


def test_a_multiplier_reaches_the_entity_manager_through_the_client_and_server():
    server = EntityServer.__new__(EntityServer)  # only its forwarding is under test
    server._manager = MagicMock()
    client = EntityClient(connection_mode=RPCConnectionMode.LOCAL, running_unit_tests=True)
    client.set_direct_server(server)
    client.apply_bucket_account_size("entity_alpha_0", TRANSITION, 3 * STANDARD_SIZE, payout_scale=1.5)
    server._manager.apply_bucket_account_size.assert_called_once_with(
        "entity_alpha_0", TRANSITION, 3 * STANDARD_SIZE, waive_promotion_fee=False, payout_scale=1.5)


# ── The promotions, through a real entity manager ─────────────────────────────

def _real_entity(hop_manager):
    entity_manager = _bare_manager()
    entity_manager._entity_collateral_client = MagicMock()
    hotkey = _add_standard(entity_manager)
    hop_manager._entity_client = _EntityClient(entity_manager)
    return entity_manager, hotkey


@pytest.mark.parametrize("calmar,equity,balance,held,multiple", [
    (0.0, 1.12, 1.0, True, 3),
    (1.5, 1.035, 1.035, False, 5),
    (2.0, 1.035, 1.035, False, 10),
])
def test_an_auto_promotion_fixes_the_multiplier_of_the_multiple_it_grants(hop_manager, calmar, equity, balance,
                                                                          held, multiple):
    entity_manager, hotkey = _real_entity(hop_manager)
    _candidate(hop_manager, hotkey, stats=_stats(calmar=calmar), equity=equity, balance=balance,
               held_since_ms=HELD_SINCE_MS if held else None)
    with _starting(CREATED_MS):
        assert _auto_promote(hop_manager, hotkeys=(hotkey,)) == [hotkey]
    info = _info(entity_manager, hotkey)
    assert (info.pro_account_size, info.payout_scale) == (multiple * STANDARD_SIZE,
                                                          ValiConfig.grow_payout_multiplier(multiple))


def test_an_auto_promotion_of_an_older_subaccount_keeps_the_old_multiplier(hop_manager):
    entity_manager, hotkey = _real_entity(hop_manager)
    _candidate(hop_manager, hotkey, stats=_stats(days=20), equity=1.12, balance=1.0, held_since_ms=HELD_SINCE_MS)
    with _starting(CREATED_MS + 1):
        assert _auto_promote(hop_manager, hotkeys=(hotkey,)) == [hotkey]
    assert _info(entity_manager, hotkey).payout_scale is None
    assert entity_manager.get_payout_scale(hotkey) == pytest.approx(ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER / 3)


def test_a_failed_promotion_takes_its_multiplier_back_and_a_retry_fixes_its_own(hop_manager):
    entity_manager, hotkey = _real_entity(hop_manager)
    hop_manager.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - DAILY_MS)
    with _starting(CREATED_MS):
        with patch.object(hop_manager, "admin_set_bucket", return_value=(False, "the bucket move failed")):
            assert not hop_manager.promote_subaccount(hotkey, NOW_MS, 3 * STANDARD_SIZE, auto_promotion_multiple=3)[0]
        assert _info(entity_manager, hotkey).payout_scale is None
        assert hop_manager.promote_subaccount(hotkey, NOW_MS, 5 * STANDARD_SIZE, auto_promotion_multiple=5)[0]
    assert _info(entity_manager, hotkey).payout_scale == ValiConfig.grow_payout_multiplier(5)


@pytest.mark.parametrize("start_ms,expected", [(CREATED_MS, LOWEST), (None, None)])
def test_an_entitys_own_promotion_fixes_the_lowest_multiplier_whatever_the_size(hop_manager, start_ms, expected):
    entity_manager, hotkey = _real_entity(hop_manager)
    hop_manager.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_FUNDED, NOW_MS - DAILY_MS)
    with _starting(start_ms):
        assert hop_manager.promote_subaccount(hotkey, NOW_MS, 10 * STANDARD_SIZE)[0]
        assert hop_manager.promote_subaccount(hotkey, NOW_MS + DAILY_MS)[0]
    assert hop_manager.miner_states[hotkey].current_bucket == GROW
    assert _info(entity_manager, hotkey).payout_scale == expected


# ── A standard miner's record ─────────────────────────────────────────────────

def test_a_standard_miners_first_pro_size_fixes_its_multiplier():
    granted = bucket_sizing(record(), TRANSITION, 3 * SIZE, payout_scale=1.3)
    assert granted.payout_scale == 1.3
    assert bucket_sizing(record(), TRANSITION, 3 * SIZE).payout_scale == LOWEST
    assert bucket_sizing(granted, GROW).payout_scale == 1.3
    assert bucket_sizing(granted, GROW, 5 * SIZE, payout_scale=1.5).payout_scale == 1.3
    assert bucket_sizing(granted, MinerBucket.SUBACCOUNT_FUNDED).payout_scale == 1.3
    assert bucket_sizing(record(), MinerBucket.SUBACCOUNT_FUNDED, payout_scale=1.3).payout_scale is None
    assert bucket_sizing(record(pro_account_size=3 * SIZE), GROW, 5 * SIZE, payout_scale=1.5).payout_scale is None


def test_a_standard_miner_is_paid_the_multiplier_on_its_record():
    paid = PipelineAccountInfo.from_standard_miner(record(pro_account_size=5 * SIZE, payout_scale=1.5))
    assert paid.payout_scale == 1.5
    unset = PipelineAccountInfo.from_standard_miner(record(pro_account_size=5 * SIZE))
    assert unset.payout_scale == LOWEST
