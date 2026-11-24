# Copyright (c) 2026 Taoshi Inc
"""
Axon request signature verification.

bittensor's Axon.default_verify only checks a signature that is present, so a request with none passes under any
hotkey it claims. ValidatorBase.wire_axon attaches a verify_fn to every synapse that refuses an unsigned request
before running the default check. These tests wire a real bt.Axon (never served or started) and run its attached
verify functions, which is what the axon middleware calls on each request.

The validator broadcasts are also blacklisted unless they come from the mothership, and the mothership takes none.
"""
import asyncio
import socket
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bittensor as bt
from bittensor_wallet import Keypair

import template.protocol as protocol
from neurons.validator_base import ValidatorBase
from vali_objects.vali_config import ValiConfig

SYNAPSE_TYPES = [
    protocol.SendSignal,
    protocol.GetPositions,
    protocol.CollateralRecord,
    protocol.AssetSelection,
    protocol.SubaccountRegistration,
    protocol.EntityEndpointUpdate,
    protocol.StandardMinerOpening,
    protocol.StandardMinerRecord,
    protocol.AccountSizeLocks,
]
MINER_TYPES = [protocol.SendSignal, protocol.GetPositions]
BROADCAST_TYPES = [t for t in SYNAPSE_TYPES if t not in MINER_TYPES]


class _EntityClientStub:
    def receive_entity_endpoint_synapse(
        self, synapse: protocol.EntityEndpointUpdate
    ) -> protocol.EntityEndpointUpdate:
        return synapse


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wallet(keypair: Keypair):
    return SimpleNamespace(hotkey=keypair, coldkeypub=keypair, coldkey=keypair, name="test", hotkey_str="test")


class TestAxonSignatureVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.validator_keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        cls.sender_keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

        validator = object.__new__(ValidatorBase)
        validator.wallet = _wallet(cls.validator_keypair)
        validator.config = SimpleNamespace(
            axon=SimpleNamespace(port=_free_port(), external_port=None),
            subtensor=SimpleNamespace(chain_endpoint="test"),
            netuid=8,
        )
        validator.metagraph_client = Mock()
        validator.subtensor = None
        validator._entity_client = _EntityClientStub()
        with patch.object(bt.Axon, "serve"), patch.object(bt.Axon, "start"):
            validator.wire_axon()
        cls.validator = validator
        cls.axon = validator.axon

    def _verify(self, synapse):
        asyncio.run(self.axon.verify_fns[synapse.name](synapse))

    def _claim(self, synapse_cls, hotkey: str, signature):
        synapse = synapse_cls()
        synapse.dendrite.hotkey = hotkey
        synapse.dendrite.nonce = time.time_ns()
        synapse.dendrite.uuid = "uuid-" + synapse.name
        synapse.dendrite.version = 10_000_000
        synapse.dendrite.signature = signature
        return synapse

    def _signed(self, synapse_cls):
        dendrite = bt.Dendrite(wallet=self.sender_keypair)
        target = bt.AxonInfo(
            version=1, ip="127.0.0.1", port=self.validator.config.axon.port, ip_type=4,
            hotkey=self.validator_keypair.ss58_address, coldkey=self.validator_keypair.ss58_address,
        )
        synapse = dendrite.preprocess_synapse_for_request(target, synapse_cls())
        # On the wire the dendrite sends body_hash as the computed_body_hash header, which the axon reads back
        synapse.computed_body_hash = synapse.to_headers()["computed_body_hash"]
        return synapse

    def test_every_synapse_has_the_signature_check(self):
        for synapse_cls in SYNAPSE_TYPES:
            name = synapse_cls.__name__
            self.assertIn(name, self.axon.verify_fns)
            self.assertNotEqual(self.axon.verify_fns[name], self.axon.default_verify, name)

    def test_unsigned_request_claiming_the_mothership_is_refused(self):
        for synapse_cls in SYNAPSE_TYPES:
            for signature in (None, ""):
                with self.subTest(synapse=synapse_cls.__name__, signature=signature):
                    synapse = self._claim(synapse_cls, ValiConfig.MOTHERSHIP_HOTKEY, signature)
                    with self.assertRaisesRegex(Exception, "Unsigned"):
                        self._verify(synapse)

    def test_default_verify_alone_accepts_the_unsigned_request(self):
        # The hole the verify_fn closes: if bittensor starts refusing these itself, this test tells us
        synapse = self._claim(protocol.SubaccountRegistration, ValiConfig.MOTHERSHIP_HOTKEY, None)
        asyncio.run(self.axon.default_verify(synapse))

    def test_wrong_signature_is_refused(self):
        for synapse_cls in SYNAPSE_TYPES:
            with self.subTest(synapse=synapse_cls.__name__):
                synapse = self._claim(synapse_cls, ValiConfig.MOTHERSHIP_HOTKEY, "0x" + "00" * 64)
                with self.assertRaisesRegex(Exception, "Signature mismatch"):
                    self._verify(synapse)

    def test_signature_from_another_key_is_refused(self):
        # A sender signing with its own key while claiming the mothership's hotkey
        synapse = self._signed(protocol.SubaccountRegistration)
        synapse.dendrite.hotkey = ValiConfig.MOTHERSHIP_HOTKEY
        with self.assertRaisesRegex(Exception, "Signature mismatch"):
            self._verify(synapse)

    def test_request_signed_by_a_dendrite_passes(self):
        for synapse_cls in SYNAPSE_TYPES:
            with self.subTest(synapse=synapse_cls.__name__):
                synapse = self._signed(synapse_cls)
                self.assertEqual(synapse.dendrite.hotkey, self.sender_keypair.ss58_address)
                self._verify(synapse)

    def test_replayed_signed_request_is_refused(self):
        synapse = self._signed(protocol.SendSignal)
        self._verify(synapse)
        with self.assertRaisesRegex(Exception, "Nonce is too old"):
            self._verify(synapse)

    # ==================== Broadcast blacklist ====================

    def _blacklisted(self, synapse_cls, sender_hotkey: str) -> bool:
        synapse = synapse_cls()
        synapse.dendrite.hotkey = sender_hotkey
        blacklisted, _ = self.axon.blacklist_fns[synapse.name](synapse)
        return blacklisted

    def test_broadcast_from_the_mothership_is_taken(self):
        for synapse_cls in BROADCAST_TYPES:
            with self.subTest(synapse=synapse_cls.__name__):
                self.assertFalse(self._blacklisted(synapse_cls, ValiConfig.MOTHERSHIP_HOTKEY))

    def test_broadcast_from_another_registered_hotkey_is_refused(self):
        # Registered in the metagraph (the Mock says so) but not the mothership: another validator or a miner
        for synapse_cls in BROADCAST_TYPES:
            with self.subTest(synapse=synapse_cls.__name__):
                self.assertTrue(self._blacklisted(synapse_cls, self.sender_keypair.ss58_address))

    def test_broadcast_from_the_testnet_mothership_is_refused_on_mainnet(self):
        for synapse_cls in BROADCAST_TYPES:
            with self.subTest(synapse=synapse_cls.__name__):
                self.assertTrue(self._blacklisted(synapse_cls, ValiConfig.MOTHERSHIP_HOTKEY_TESTNET))

    def test_testnet_takes_broadcasts_from_the_testnet_mothership_only(self):
        with patch.object(self.validator.config, "netuid", 116):
            for synapse_cls in BROADCAST_TYPES:
                with self.subTest(synapse=synapse_cls.__name__):
                    self.assertFalse(self._blacklisted(synapse_cls, ValiConfig.MOTHERSHIP_HOTKEY_TESTNET))
                    self.assertTrue(self._blacklisted(synapse_cls, ValiConfig.MOTHERSHIP_HOTKEY))

    def test_mothership_takes_no_broadcasts(self):
        # This validator made the mothership: refuses its own hotkey and everyone else's
        with patch.object(ValiConfig, "MOTHERSHIP_HOTKEY", self.validator_keypair.ss58_address):
            for synapse_cls in BROADCAST_TYPES:
                for sender in (self.validator_keypair.ss58_address, self.sender_keypair.ss58_address):
                    with self.subTest(synapse=synapse_cls.__name__, sender=sender):
                        self.assertTrue(self._blacklisted(synapse_cls, sender))

    def test_mothership_still_takes_miner_requests(self):
        with patch.object(ValiConfig, "MOTHERSHIP_HOTKEY", self.validator_keypair.ss58_address):
            for synapse_cls in MINER_TYPES:
                with self.subTest(synapse=synapse_cls.__name__):
                    self.assertFalse(self._blacklisted(synapse_cls, self.sender_keypair.ss58_address))

    def test_miner_requests_are_not_limited_to_the_mothership(self):
        for synapse_cls in MINER_TYPES:
            with self.subTest(synapse=synapse_cls.__name__):
                self.assertFalse(self._blacklisted(synapse_cls, self.sender_keypair.ss58_address))

    def test_unregistered_mothership_hotkey_is_still_refused(self):
        # The metagraph check still applies behind the mothership check
        with patch.object(self.validator.metagraph_client, "has_hotkey", return_value=False):
            self.assertTrue(self._blacklisted(protocol.SubaccountRegistration, ValiConfig.MOTHERSHIP_HOTKEY))


if __name__ == "__main__":
    unittest.main()
