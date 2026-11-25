"""
The permissionless entity miner's relay (docs/pem_envelope.md): POST /api/pem/submit-order on the entity miner
REST server, against the real Flask handler with the order placer mocked, and PropNetOrderPlacer putting the
envelope on the SendSignal it sends to validators.
"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bittensor_wallet import Keypair
from flask import Flask

from miner_objects.prop_net_order_placer import PropNetOrderPlacer
from shared_objects.pem_envelope import build_payload, sign_payload
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.vali_config import ValiConfig, TradePair
from vali_objects.vali_dataclasses.order_signal import Signal
from vanta_api.entity_miner_rest_server import EntityMinerRestServer

TRADER = Keypair.create_from_uri("//Alice")
PEM = Keypair.create_from_uri("//Bob")
OTHER_PEM = Keypair.create_from_uri("//Charlie")
MARKET = {"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1}
API_KEY = "a-key"
RELAYED = {"success": True, "order_uuid": "order-1", "order_json": "{}", "error_message": "",
           "processing_time": 0.1, "message": "Order successfully processed by Vanta Network"}


def _client(order_placer):
    gw = object.__new__(EntityMinerRestServer)
    gw.order_placer = order_placer
    gw.is_valid_api_key = lambda key: key == API_KEY
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/api/pem/submit-order", methods=["POST"])(gw.pem_submit_order_endpoint)
    return app.test_client()


class TestPemSubmitOrderEndpoint(unittest.TestCase):

    def setUp(self):
        self.placer = MagicMock()
        self.placer.wallet.hotkey.ss58_address = PEM.ss58_address
        self.placer.config.netuid = 8
        self.placer.process_a_signal_for_rest.return_value = dict(RELAYED)
        self.client = _client(self.placer)
        self.relay = self.placer.process_a_signal_for_rest

    def _envelope(self, signal=MARKET, relay=PEM, netuid=8, order_uuid="order-1", timestamp_ms=None):
        payload = build_payload(TRADER.ss58_address, relay.ss58_address, signal, netuid, order_uuid, timestamp_ms)
        return {"origin_payload": payload, "origin_signature": sign_payload(payload, TRADER)}

    def _post(self, body, api_key=API_KEY):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = self.client.post("/api/pem/submit-order", json=body, headers=headers)
        return response.status_code, response.get_json()

    def assertRefused(self, body, reason, status=400):
        code, reply = self._post(body)
        self.assertEqual(code, status, reply)
        self.assertFalse(reply.get("success", False))
        self.assertIn(reason, reply["error"])
        self.relay.assert_not_called()

    def test_a_valid_envelope_is_relayed_unchanged(self):
        body = self._envelope()
        code, reply = self._post(body)
        self.assertEqual(code, 200, reply)
        self.assertTrue(reply["success"])
        self.assertEqual(reply["origin_hotkey"], TRADER.ss58_address)
        self.assertEqual(reply["relay_hotkey"], PEM.ss58_address)

        kwargs = self.relay.call_args.kwargs
        self.assertEqual(kwargs["origin_payload"], body["origin_payload"])
        self.assertEqual(kwargs["origin_signature"], body["origin_signature"])
        self.assertEqual(kwargs["order_uuid"], "order-1")
        self.assertEqual(kwargs["signal"].trade_pair, TradePair.BTCUSDC)
        self.assertEqual(kwargs["signal"].execution_type, ExecutionType.MARKET)
        self.assertNotIn("subaccount_id", kwargs)

    def test_a_validator_rejection_is_400_with_the_validators_reply(self):
        self.relay.return_value = {**RELAYED, "success": False, "error_message": "Miner has been eliminated"}
        code, reply = self._post(self._envelope())
        self.assertEqual(code, 400)
        self.assertEqual(reply["error_message"], "Miner has been eliminated")

    def test_an_error_while_relaying_is_500(self):
        self.relay.side_effect = RuntimeError("no validators")
        code, reply = self._post(self._envelope())
        self.assertEqual(code, 500)
        self.assertIn("no validators", reply["error"])

    def test_without_a_valid_api_key(self):
        for key in (None, "wrong"):
            with self.subTest(key=key):
                code, _ = self._post(self._envelope(), api_key=key)
                self.assertEqual(code, 401)
        self.relay.assert_not_called()

    def test_without_an_order_placer(self):
        code, _ = _client(None).post("/api/pem/submit-order", json=self._envelope(),
                                     headers={"Authorization": API_KEY}).status_code, None
        self.assertEqual(code, 503)

    def test_a_body_that_is_not_an_object(self):
        for body in ([1, 2], "text", 42):
            with self.subTest(body=body):
                self.assertRefused(body, "JSON object")

    def test_a_bad_envelope(self):
        body = self._envelope()
        other = self._envelope(order_uuid="another")
        for bad in ({"origin_payload": body["origin_payload"]},
                    {"origin_signature": body["origin_signature"]},
                    {"origin_payload": body["origin_payload"], "origin_signature": other["origin_signature"]},
                    {"origin_payload": body["origin_payload"], "origin_signature": sign_payload(body["origin_payload"], PEM)}):
            with self.subTest(body=bad):
                self.assertRefused(bad, "Invalid order envelope")

    def test_an_envelope_for_another_relay(self):
        self.assertRefused(self._envelope(relay=OTHER_PEM), "not this miner's")

    def test_an_envelope_for_another_netuid(self):
        self.assertRefused(self._envelope(netuid=116), "netuid 116")

    def test_a_stale_or_future_envelope(self):
        now = TimeUtil.now_in_millis()
        for ts in (now - ValiConfig.PEM_ENVELOPE_MAX_AGE_MS - 5_000, now + ValiConfig.PEM_ENVELOPE_MAX_FUTURE_MS + 5_000):
            with self.subTest(timestamp_ms=ts):
                self.assertRefused(self._envelope(timestamp_ms=ts), "window")

    def test_an_invalid_signal(self):
        self.assertRefused(self._envelope(signal={"execution_type": "MARKET", "trade_pair": "BTCUSDC",
                                                  "order_type": "LONG", "leverage": 0.1, "value": 100.0}),
                           "Invalid signal data")


class TestOrderPlacerCarriesTheEnvelope(unittest.TestCase):

    def setUp(self):
        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=PEM.ss58_address))
        config = SimpleNamespace(netuid=8, write_failed_signal_logs=False)
        self.placer = PropNetOrderPlacer(wallet, MagicMock(), config, is_testnet=False, running_unit_tests=True)
        self.sent = []

        async def send(synapse, mothership_axon, other_axons):
            self.sent.append(synapse)
            return {"success": True, "order_json": "{}", "error_message": ""}

        self.patches = [patch.object(self.placer, "_get_mothership_and_other_axons", return_value=(None, [])),
                        patch.object(self.placer, "_send_order", side_effect=send),
                        patch.object(self.placer, "write_signal_to_processed_directory"),
                        patch.object(self.placer, "write_signal_to_failure_directory")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.placer.executor.shutdown(wait=False)

    def test_the_envelope_rides_on_send_signal_with_an_empty_signal(self):
        payload = build_payload(TRADER.ss58_address, PEM.ss58_address, MARKET, 8, "order-1")
        signature = sign_payload(payload, TRADER)
        result = self.placer.process_a_signal_for_rest("order-1", Signal(**MARKET), origin_payload=payload,
                                                       origin_signature=signature)
        self.assertTrue(result["success"], result)
        synapse = self.sent[0]
        self.assertEqual(synapse.origin_payload, payload)
        self.assertEqual(synapse.origin_signature, signature)
        self.assertEqual(synapse.signal, {})
        self.assertEqual(synapse.miner_order_uuid, "order-1")
        self.assertIsNone(synapse.subaccount_id)

    def test_an_order_without_an_envelope_is_unchanged(self):
        result = self.placer.process_a_signal_for_rest("order-2", Signal(**MARKET))
        self.assertTrue(result["success"], result)
        synapse = self.sent[0]
        self.assertIsNone(synapse.origin_payload)
        self.assertIsNone(synapse.origin_signature)
        self.assertEqual(synapse.signal["order_type"], "LONG")


if __name__ == "__main__":
    unittest.main()
