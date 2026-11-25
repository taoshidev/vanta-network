"""
The order envelope a trader signs for a permissionless entity miner (shared_objects/pem_envelope.py): the test
vectors in docs/pem_envelope.md, signing and verification, tampering, the structural checks, and the SendSignal
fields that carry it.
"""
import json
import unittest

from bittensor_wallet import Keypair

from shared_objects.pem_envelope import (
    EnvelopeError, MAX_ORDER_UUID_LENGTH, MAX_PAYLOAD_LENGTH, PAYLOAD_VERSION, build_payload, parse_payload,
    sign_payload, verify_envelope,
)
from template.protocol import SendSignal

TRADER = Keypair.create_from_uri("//Alice")
PEM = Keypair.create_from_uri("//Bob")

# docs/pem_envelope.md "Test vectors": (inputs, expected payload, a valid signature by //Alice)
VECTORS = [
    (
        dict(netuid=8, miner_order_uuid="0b3c1f4e-7a2d-4c1e-9f6a-2d8e5b7c9a01", timestamp_ms=1795000000000,
             signal={"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1}),
        '{"miner_order_uuid":"0b3c1f4e-7a2d-4c1e-9f6a-2d8e5b7c9a01","netuid":8,'
        '"origin_hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",'
        '"relay_hotkey":"5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",'
        '"signal":{"execution_type":"MARKET","leverage":0.1,"order_type":"LONG","trade_pair":"BTCUSDC"},'
        '"timestamp_ms":1795000000000,"v":1}',
        "3608e544c6cbbfb67d92c4f2739a2c7a9750d05b118108abe7ab94961aa77670"
        "deb617f4ba22b260596383da8d0bde17d89fef2f84eba8d0fdce199b87968185",
    ),
    (
        dict(netuid=8, miner_order_uuid="5e2a9c7d-3b1f-4d8e-a6c2-7f0b9e1d4c32", timestamp_ms=1795000060000,
             signal={"execution_type": "LIMIT", "trade_pair": "ETHUSDC", "order_type": "SHORT", "value": 2500.5,
                     "limit_price": 3125.75}),
        '{"miner_order_uuid":"5e2a9c7d-3b1f-4d8e-a6c2-7f0b9e1d4c32","netuid":8,'
        '"origin_hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",'
        '"relay_hotkey":"5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",'
        '"signal":{"execution_type":"LIMIT","limit_price":3125.75,"order_type":"SHORT","trade_pair":"ETHUSDC","value":2500.5},'
        '"timestamp_ms":1795000060000,"v":1}',
        "b218bcf6e59bfaaff8e3bf93c92621b98e160dcb57738ffc540589c4078b8944"
        "c4863e45cc5c87e0090ef50654903b01f885a60436fc328f529dc36781a0cb82",
    ),
    (
        dict(netuid=116, miner_order_uuid="9d4b2e6f-1c8a-4f3d-b7e5-0a2c6d8f1b43", timestamp_ms=1795000120000,
             signal={"execution_type": "FLAT_ALL"}),
        '{"miner_order_uuid":"9d4b2e6f-1c8a-4f3d-b7e5-0a2c6d8f1b43","netuid":116,'
        '"origin_hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",'
        '"relay_hotkey":"5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",'
        '"signal":{"execution_type":"FLAT_ALL"},"timestamp_ms":1795000120000,"v":1}',
        "f0f2aa6c593ece2d90a196d92bde9dfb36c2fb355417da142bc54d6f7dfdcd74"
        "86d3fd639cb6c6a72247c885cd63aa696b032ffa6e1516c46ee6634bdb1ee182",
    ),
]


def _fields(**overrides):
    fields = {"v": 1, "netuid": 8, "origin_hotkey": TRADER.ss58_address, "relay_hotkey": PEM.ss58_address,
              "miner_order_uuid": "an-order", "timestamp_ms": 1795000000000,
              "signal": {"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1}}
    fields.update(overrides)
    return fields


class TestTestVectors(unittest.TestCase):

    def test_the_payloads_are_reproduced_exactly(self):
        for inputs, expected_payload, _ in VECTORS:
            with self.subTest(uuid=inputs["miner_order_uuid"]):
                payload = build_payload(TRADER.ss58_address, PEM.ss58_address, **inputs)
                self.assertEqual(payload, expected_payload)

    def test_the_documented_signatures_verify(self):
        for inputs, expected_payload, signature in VECTORS:
            with self.subTest(uuid=inputs["miner_order_uuid"]):
                parsed = verify_envelope(expected_payload, signature)
                self.assertEqual(parsed.origin_hotkey, TRADER.ss58_address)
                self.assertEqual(parsed.relay_hotkey, PEM.ss58_address)
                self.assertEqual(parsed.netuid, inputs["netuid"])
                self.assertEqual(parsed.miner_order_uuid, inputs["miner_order_uuid"])
                self.assertEqual(parsed.timestamp_ms, inputs["timestamp_ms"])
                self.assertEqual(parsed.signal, inputs["signal"])
                self.assertEqual(parsed.v, PAYLOAD_VERSION)


class TestSignAndVerify(unittest.TestCase):

    def setUp(self):
        self.payload = build_payload(TRADER.ss58_address, PEM.ss58_address, _fields()["signal"], 8)

    def test_a_signed_payload_verifies(self):
        self.assertEqual(verify_envelope(self.payload, sign_payload(self.payload, TRADER)).origin_hotkey,
                         TRADER.ss58_address)

    def test_signatures_are_randomized_and_each_verifies(self):
        first, second = sign_payload(self.payload, TRADER), sign_payload(self.payload, TRADER)
        self.assertNotEqual(first, second)
        verify_envelope(self.payload, first)
        verify_envelope(self.payload, second)

    def test_a_0x_prefix_is_accepted(self):
        verify_envelope(self.payload, "0x" + sign_payload(self.payload, TRADER))

    def test_a_browser_wallet_bytes_wrapped_signature_is_accepted(self):
        wrapped = TRADER.sign(b"<Bytes>" + self.payload.encode("utf-8") + b"</Bytes>").hex()
        verify_envelope(self.payload, wrapped)

    def test_build_payload_fills_a_uuid_and_the_current_time(self):
        parsed = parse_payload(self.payload)
        self.assertTrue(parsed.miner_order_uuid)
        self.assertGreater(parsed.timestamp_ms, 1_700_000_000_000)
        other = parse_payload(build_payload(TRADER.ss58_address, PEM.ss58_address, {"execution_type": "FLAT_ALL"}, 8))
        self.assertNotEqual(parsed.miner_order_uuid, other.miner_order_uuid)

    def test_a_signature_by_another_key_is_refused(self):
        for key in (PEM, Keypair.create_from_uri("//Charlie")):
            with self.subTest(key=key.ss58_address):
                with self.assertRaisesRegex(EnvelopeError, "not origin_hotkey's signature"):
                    verify_envelope(self.payload, sign_payload(self.payload, key))

    def test_any_change_to_the_signed_payload_is_refused(self):
        signature = sign_payload(self.payload, TRADER)
        fields = json.loads(self.payload)
        for name, value in (("relay_hotkey", Keypair.create_from_uri("//Charlie").ss58_address),
                            ("netuid", 116), ("timestamp_ms", fields["timestamp_ms"] + 1),
                            ("miner_order_uuid", "another-order"),
                            ("signal", {**fields["signal"], "leverage": 0.5})):
            with self.subTest(field=name):
                tampered = json.dumps({**fields, name: value}, sort_keys=True, separators=(",", ":"))
                with self.assertRaises(EnvelopeError):
                    verify_envelope(tampered, signature)
        # Re-serializing the same fields differently changes the bytes, so the signature no longer matches
        with self.assertRaises(EnvelopeError):
            verify_envelope(json.dumps(fields, sort_keys=True), signature)

    def test_a_malformed_signature_is_refused(self):
        good = sign_payload(self.payload, TRADER)
        for signature in ("", "not hex", "0x", good[:-2], good + "00", "ab" * 200, None, 42):
            with self.subTest(signature=signature):
                with self.assertRaises(EnvelopeError):
                    verify_envelope(self.payload, signature)


class TestPayloadStructure(unittest.TestCase):

    def assertRefused(self, payload, reason):
        with self.assertRaisesRegex(EnvelopeError, reason):
            parse_payload(payload)

    def _dump(self, fields):
        return json.dumps(fields, separators=(",", ":"))

    def test_a_well_formed_payload_parses_in_any_key_order(self):
        fields = _fields()
        reordered = dict(reversed(list(fields.items())))
        self.assertEqual(parse_payload(json.dumps(reordered, indent=2)).signal, fields["signal"])

    def test_not_a_json_object(self):
        for payload in ("", "not json", "[1, 2]", "42", "null", None, 42):
            with self.subTest(payload=payload):
                with self.assertRaises(EnvelopeError):
                    parse_payload(payload)

    def test_too_long(self):
        self.assertRefused(self._dump(_fields(signal={"pad": "x" * MAX_PAYLOAD_LENGTH})), "longer than")

    def test_a_repeated_key(self):
        payload = self._dump(_fields())[:-1] + ',"relay_hotkey":"' + TRADER.ss58_address + '"}'
        self.assertRefused(payload, "repeats a key")

    def test_missing_and_unknown_fields(self):
        for name in ("v", "netuid", "origin_hotkey", "relay_hotkey", "miner_order_uuid", "timestamp_ms", "signal"):
            with self.subTest(missing=name):
                fields = _fields()
                del fields[name]
                self.assertRefused(self._dump(fields), "missing")
        self.assertRefused(self._dump(_fields(max_take=0.1)), "unknown fields")

    def test_the_version(self):
        for v in (0, 2, "1", True, 1.0, 1.5):
            with self.subTest(v=v):
                self.assertRefused(self._dump(_fields(v=v)), "version")

    def test_netuid_and_timestamp_are_non_negative_integers(self):
        for name in ("netuid", "timestamp_ms"):
            for value in ("8", 8.0, True, -1, None):
                with self.subTest(field=name, value=value):
                    self.assertRefused(self._dump(_fields(**{name: value})), name)

    def test_hotkeys_are_distinct_ss58_addresses(self):
        for name in ("origin_hotkey", "relay_hotkey"):
            for value in ("not an address", "", 42, None, f"{PEM.ss58_address}_0"):
                with self.subTest(field=name, value=value):
                    self.assertRefused(self._dump(_fields(**{name: value})), name)
        self.assertRefused(self._dump(_fields(relay_hotkey=TRADER.ss58_address)), "must differ")

    def test_the_order_uuid(self):
        for value in ("", None, 42, "x" * (MAX_ORDER_UUID_LENGTH + 1)):
            with self.subTest(value=value):
                self.assertRefused(self._dump(_fields(miner_order_uuid=value)), "miner_order_uuid")
        parse_payload(self._dump(_fields(miner_order_uuid="x" * MAX_ORDER_UUID_LENGTH)))

    def test_the_signal_is_a_non_empty_object(self):
        for value in ({}, [], "MARKET", None):
            with self.subTest(value=value):
                self.assertRefused(self._dump(_fields(signal=value)), "signal")


class TestSendSignalFields(unittest.TestCase):

    def test_the_envelope_rides_on_send_signal(self):
        payload = build_payload(TRADER.ss58_address, PEM.ss58_address, {"execution_type": "FLAT_ALL"}, 8)
        synapse = SendSignal(origin_payload=payload, origin_signature=sign_payload(payload, TRADER))
        verify_envelope(synapse.origin_payload, synapse.origin_signature)
        self.assertIsNone(SendSignal().origin_payload)
        self.assertIsNone(SendSignal().origin_signature)

    def test_the_body_hash_is_unchanged_so_existing_miners_are_unaffected(self):
        self.assertEqual(SendSignal.required_hash_fields, ["signal"])
        signal = {"execution_type": "MARKET", "trade_pair": "BTCUSDC", "order_type": "LONG", "leverage": 0.1}
        payload = build_payload(TRADER.ss58_address, PEM.ss58_address, signal, 8)
        plain = SendSignal(signal=signal)
        enveloped = SendSignal(signal=signal, origin_payload=payload, origin_signature=sign_payload(payload, TRADER))
        self.assertEqual(plain.body_hash, enveloped.body_hash)


if __name__ == "__main__":
    unittest.main()
