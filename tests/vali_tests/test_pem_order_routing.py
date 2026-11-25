"""
How the validator takes in an order a trader signed and a permissionless entity miner relayed
(Validator._receive_signal_sync with an envelope, docs/pem_envelope.md): the checks on the envelope, the relay
and the trader, that the order is processed as the trader with the relay recorded, the one-use-per-payload
replay guard, and that orders without an envelope are unchanged. The order processor is mocked; the uuid tracker
is the validator's own.
"""
import contextlib
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import bittensor as bt
from bittensor_wallet import Keypair

from neurons.validator import Validator
from shared_objects.pem_envelope import build_payload, sign_payload
from shared_objects.rate_limiter import RateLimiter
from template.protocol import SendSignal
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.uuid_tracker import UUIDTracker
from vali_objects.vali_config import ValiConfig, TradePair

TRADER = Keypair.create_from_uri("//Alice")
OTHER_TRADER = Keypair.create_from_uri("//Dave")
PEM = Keypair.create_from_uri("//Bob")
OTHER_PEM = Keypair.create_from_uri("//Charlie")
MARKET = {"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1}
NETUID = 8


class TestPemOrderRouting(unittest.TestCase):

    def setUp(self):
        self.registered = {TRADER.ss58_address, OTHER_TRADER.ss58_address, PEM.ss58_address, OTHER_PEM.ss58_address}
        self.pems = {PEM.ss58_address}

        v = object.__new__(Validator)
        v.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator_hotkey"))
        v.config = SimpleNamespace(netuid=NETUID)
        v.subtensor_ops_manager = Mock()
        v.subtensor_ops_manager.is_hotkey_registered_cached.side_effect = lambda hk: hk in self.registered
        v.entity_client = Mock()
        v.entity_client.is_permissionless_entity_miner.side_effect = lambda hk: hk in self.pems
        v.order_rate_limiter = RateLimiter()
        v.position_inspector_rate_limiter = RateLimiter()
        v.order_sync = Mock()
        v.order_sync.is_sync_waiting.return_value = False
        v.order_sync.begin_order.side_effect = lambda: contextlib.nullcontext(SimpleNamespace(rejected=False))
        v.uuid_tracker = UUIDTracker()
        v.order_processor = Mock()
        v.order_processor.validate.return_value = (True, "", None)
        v.order_processor.process_vanta_signal.return_value = Mock(
            get_response_json=Mock(return_value="{}"), should_track_uuid=True, order_for_logging=None)
        self.validator = v
        self.process = v.order_processor.process_vanta_signal

    def _envelope(self, signal=MARKET, trader=TRADER, relay=PEM, netuid=NETUID, order_uuid=None, timestamp_ms=None):
        payload = build_payload(trader.ss58_address, relay.ss58_address, signal, netuid, order_uuid, timestamp_ms)
        return payload, sign_payload(payload, trader)

    def _send(self, payload=None, signature=None, sender=PEM, signal=None, subaccount_id=None, miner_order_uuid=""):
        synapse = SendSignal(signal=signal or {}, origin_payload=payload, origin_signature=signature,
                             subaccount_id=subaccount_id, miner_order_uuid=miner_order_uuid)
        synapse.dendrite = bt.TerminalInfo(hotkey=sender.ss58_address if isinstance(sender, Keypair) else sender)
        return self.validator._receive_signal_sync(synapse)

    def assertRejected(self, synapse, reason):
        self.assertFalse(synapse.successfully_processed)
        self.assertIn(reason, synapse.error_message)
        self.assertFalse(synapse.should_retry)
        self.process.assert_not_called()

    # ==================== Accepted ====================

    def test_a_relayed_order_is_processed_as_the_trader_with_the_relay_recorded(self):
        payload, signature = self._envelope(order_uuid="order-1")
        synapse = self._send(payload, signature)
        self.assertTrue(synapse.successfully_processed, synapse.error_message)

        self.validator.order_processor.validate.assert_called_once()
        self.assertEqual(self.validator.order_processor.validate.call_args.kwargs["hotkey"], TRADER.ss58_address)
        kwargs = self.process.call_args.kwargs
        self.assertEqual(kwargs["hotkey"], TRADER.ss58_address)
        self.assertEqual(kwargs["relay_hotkey"], PEM.ss58_address)
        self.assertEqual(kwargs["order_uuid"], "order-1")
        self.assertEqual(kwargs["signal"].trade_pair, TradePair.BTCUSDC)
        self.assertEqual(kwargs["signal"].execution_type, ExecutionType.MARKET)

    def test_the_synapse_signal_may_repeat_the_envelopes(self):
        payload, signature = self._envelope()
        self.assertTrue(self._send(payload, signature, signal=dict(MARKET)).successfully_processed)

    def test_an_order_without_an_envelope_is_unchanged(self):
        synapse = self._send(signal=MARKET, sender=TRADER, miner_order_uuid="direct-1")
        self.assertTrue(synapse.successfully_processed, synapse.error_message)
        kwargs = self.process.call_args.kwargs
        self.assertEqual(kwargs["hotkey"], TRADER.ss58_address)
        self.assertIsNone(kwargs["relay_hotkey"])
        self.validator.entity_client.is_permissionless_entity_miner.assert_not_called()

    def test_a_subaccount_order_is_unchanged(self):
        synapse = self._send(signal=MARKET, sender=OTHER_PEM, subaccount_id=3, miner_order_uuid="sub-1")
        self.assertTrue(synapse.successfully_processed, synapse.error_message)
        kwargs = self.process.call_args.kwargs
        self.assertEqual(kwargs["hotkey"], f"{OTHER_PEM.ss58_address}_3")
        self.assertIsNone(kwargs["relay_hotkey"])

    # ==================== The envelope ====================

    def test_an_envelope_with_a_subaccount_id_is_refused(self):
        payload, signature = self._envelope()
        self.assertRejected(self._send(payload, signature, subaccount_id=0), "subaccount_id")

    def test_a_bad_or_missing_signature_is_refused(self):
        payload, signature = self._envelope()
        other_payload, _ = self._envelope(order_uuid="another")
        for p, s in ((payload, sign_payload(payload, PEM)), (other_payload, signature), (payload, None),
                     (None, signature), (payload, "zz")):
            with self.subTest(payload=p, signature=s):
                self.assertRejected(self._send(p, s), "Invalid order envelope")

    def test_the_sender_must_be_the_signed_relay(self):
        payload, signature = self._envelope(relay=PEM)
        self.pems.add(OTHER_PEM.ss58_address)
        self.assertRejected(self._send(payload, signature, sender=OTHER_PEM), "is not the sender")

    def test_the_netuid_must_match(self):
        payload, signature = self._envelope(netuid=116)
        self.assertRejected(self._send(payload, signature), "netuid 116")

    def test_a_stale_or_future_timestamp_is_refused(self):
        now = TimeUtil.now_in_millis()
        for ts in (now - ValiConfig.PEM_ENVELOPE_MAX_AGE_MS - 5_000, now + ValiConfig.PEM_ENVELOPE_MAX_FUTURE_MS + 5_000):
            with self.subTest(timestamp_ms=ts):
                payload, signature = self._envelope(timestamp_ms=ts)
                self.assertRejected(self._send(payload, signature), "window")

    def test_a_synapse_signal_that_differs_is_refused(self):
        payload, signature = self._envelope()
        self.assertRejected(self._send(payload, signature, signal={**MARKET, "leverage": 0.5}), "differs")

    # ==================== The relay and the trader ====================

    def test_the_relay_must_be_a_permissionless_entity_miner(self):
        payload, signature = self._envelope(relay=OTHER_PEM)
        self.assertRejected(self._send(payload, signature, sender=OTHER_PEM), "not registered as a permissionless_entity_miner")

    def test_the_trader_must_be_registered(self):
        self.registered.discard(TRADER.ss58_address)
        payload, signature = self._envelope()
        self.assertRejected(self._send(payload, signature), "is not registered on netuid")

    def test_a_failed_relay_lookup_asks_for_a_retry(self):
        self.validator.entity_client.is_permissionless_entity_miner.side_effect = ConnectionRefusedError("core down")
        payload, signature = self._envelope()
        synapse = self._send(payload, signature)
        self.assertFalse(synapse.successfully_processed)
        self.assertTrue(synapse.should_retry)
        self.process.assert_not_called()

    def test_the_traders_account_rules_apply(self):
        self.validator.order_processor.validate.return_value = (False, "Miner has been eliminated", None)
        payload, signature = self._envelope()
        synapse = self._send(payload, signature)
        self.assertFalse(synapse.successfully_processed)
        self.assertIn("eliminated", synapse.error_message)
        self.assertEqual(self.validator.order_processor.validate.call_args.kwargs["hotkey"], TRADER.ss58_address)

    # ==================== Rate limit ====================

    def test_refused_envelopes_do_not_spend_the_traders_rate_limit(self):
        payload, signature = self._envelope()
        bad = sign_payload(payload, PEM)
        for _ in range(self.validator.order_rate_limiter.max_requests_per_window + 5):
            self.assertFalse(self._send(payload, bad).successfully_processed)
        self.assertTrue(self._send(payload, signature).successfully_processed)

    def test_the_rate_limit_is_the_traders_not_the_relays(self):
        for i in range(self.validator.order_rate_limiter.max_requests_per_window):
            payload, signature = self._envelope(order_uuid=f"a-{i}")
            self.assertTrue(self._send(payload, signature).successfully_processed)
        payload, signature = self._envelope(order_uuid="a-over")
        self.assertIn("Rate limited", self._send(payload, signature).error_message)
        # The same relay still sends for another trader
        payload, signature = self._envelope(trader=OTHER_TRADER, order_uuid="b-0")
        self.assertTrue(self._send(payload, signature).successfully_processed)

    # ==================== Replay ====================

    def test_a_signed_payload_is_accepted_once(self):
        for signal in (MARKET, {"execution_type": "FLAT_ALL"},
                       {"execution_type": "LIMIT_CANCEL", "trade_pair": "BTCUSDC"}):
            with self.subTest(execution_type=signal["execution_type"]):
                order_uuid = "ALL" if signal["execution_type"] != "MARKET" else None
                payload, signature = self._envelope(signal=signal, order_uuid=order_uuid)
                self.assertTrue(self._send(payload, signature).successfully_processed)
                calls = self.process.call_count
                # Re-sent as is, and with a fresh signature over the same payload. A MARKET resend is caught
                # by its order uuid first ("already been processed"); the others only by the payload claim
                expected = "already been processed" if signal["execution_type"] == "MARKET" else "already been received"
                for s in (signature, sign_payload(payload, TRADER)):
                    synapse = self._send(payload, s)
                    self.assertFalse(synapse.successfully_processed)
                    self.assertIn(expected, synapse.error_message)
                self.assertEqual(self.process.call_count, calls)

    def test_a_new_flat_all_of_all_is_a_new_payload(self):
        first = self._envelope(signal={"execution_type": "FLAT_ALL"}, order_uuid="ALL", timestamp_ms=TimeUtil.now_in_millis() - 1)
        second = self._envelope(signal={"execution_type": "FLAT_ALL"}, order_uuid="ALL")
        self.assertTrue(self._send(*first).successfully_processed)
        self.assertTrue(self._send(*second).successfully_processed)
        self.assertEqual(self.process.call_count, 2)

    def test_a_failed_apply_releases_the_payload_for_a_retry(self):
        self.process.side_effect = [ConnectionRefusedError("market order server bouncing"), self.process.return_value]
        payload, signature = self._envelope()
        failed = self._send(payload, signature)
        self.assertFalse(failed.successfully_processed)
        self.assertTrue(failed.should_retry)
        retried = self._send(payload, signature)
        self.assertTrue(retried.successfully_processed, retried.error_message)

    def test_a_used_order_uuid_is_refused_and_frees_the_payload(self):
        first, signature = self._envelope(order_uuid="dup")
        self.assertTrue(self._send(first, signature).successfully_processed)
        second, signature = self._envelope(order_uuid="dup", timestamp_ms=TimeUtil.now_in_millis() + 1)
        synapse = self._send(second, signature)
        self.assertFalse(synapse.successfully_processed)
        self.assertIn("already been processed", synapse.error_message)
        self.assertFalse(self.validator.uuid_tracker.exists("envelope:" + hashlib.sha256(second.encode()).hexdigest()))


if __name__ == "__main__":
    unittest.main()
