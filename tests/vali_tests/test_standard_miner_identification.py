"""
Identification of standard miners: real hotkeys registered after their network's cutoff block,
which trade on the standard account pipeline. The metagraph server derives who registered after the
cutoff at each full sync, and core pushes a copy to the account server, whose order path never reads
the metagraph.
"""
from dataclasses import dataclass
from unittest import mock

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.miner_account.miner_account_manager import MinerAccountManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import RPCConnectionMode, ValiConfig

CUTOFF = 5_000
AT_CUTOFF = "miner_registered_at_cutoff"
NEW = "miner_registered_after_cutoff"
VALIDATOR = "validator_registered_after_cutoff"
HOTKEYS = [AT_CUTOFF, NEW, VALIDATOR]
BLOCKS = [CUTOFF, CUTOFF + 1, CUTOFF + 1]
STAKE_WEIGHTS = [0.0, 0.0, ValiConfig.STAKE_MIN]


@dataclass
class AxonStub:
    hotkey: str
    ip: str


@dataclass
class NeuronStub:
    uid: int
    hotkey: str
    axon_info: AxonStub


def neurons_for(hotkeys):
    return [NeuronStub(uid=i, hotkey=hk, axon_info=AxonStub(hotkey=hk, ip=f"10.0.0.{i + 1}"))
            for i, hk in enumerate(hotkeys)]


def account_manager():
    return MinerAccountManager(running_unit_tests=True, connection_mode=RPCConnectionMode.LOCAL, is_testnet=False)


class TestStandardMinerIdentification(TestBase):

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.account_client = cls.orchestrator.get_client('miner_account')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()

    def tearDown(self):
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def _sync(self, cutoff_block=CUTOFF, hotkeys=HOTKEYS, blocks=BLOCKS):
        """One full metagraph sync as core runs it."""
        self.metagraph_client.set_hotkeys(hotkeys, block_at_registration=blocks, stake_weight=STAKE_WEIGHTS[:len(hotkeys)],
                                          standard_miner_cutoff_block=cutoff_block)

    def test_cutoff_block_is_per_network(self):
        with mock.patch.object(ValiConfig, "STANDARD_MINER_CUTOFF_BLOCK", None), \
                mock.patch.object(ValiConfig, "STANDARD_MINER_CUTOFF_BLOCK_TESTNET", None):
            self.assertIsNone(ValiConfig.standard_miner_cutoff_block(is_testnet=False))
            self.assertIsNone(ValiConfig.standard_miner_cutoff_block(is_testnet=True))
        with mock.patch.object(ValiConfig, "STANDARD_MINER_CUTOFF_BLOCK", CUTOFF), \
                mock.patch.object(ValiConfig, "STANDARD_MINER_CUTOFF_BLOCK_TESTNET", CUTOFF + 10):
            self.assertEqual(ValiConfig.standard_miner_cutoff_block(is_testnet=False), CUTOFF)
            self.assertEqual(ValiConfig.standard_miner_cutoff_block(is_testnet=True), CUTOFF + 10)

    def test_nobody_is_after_the_cutoff_while_it_is_unset(self):
        self._sync(cutoff_block=None)
        self.assertEqual(self.metagraph_client.get_registered_after_cutoff(), {hk: False for hk in HOTKEYS})

    def test_registered_after_the_cutoff_whatever_the_stake(self):
        self._sync()
        self.assertIs(self.metagraph_client.is_registered_after_cutoff(NEW), True)
        self.assertIs(self.metagraph_client.is_registered_after_cutoff(VALIDATOR), True)
        self.assertIs(self.metagraph_client.is_registered_after_cutoff(AT_CUTOFF), False)
        self.assertIsNone(self.metagraph_client.is_registered_after_cutoff("unregistered_hotkey"))

    def test_an_unknown_block_answers_none(self):
        self._sync(blocks=BLOCKS[:2])
        self.assertIsNone(self.metagraph_client.is_registered_after_cutoff(VALIDATOR))
        self.assertIs(self.metagraph_client.is_registered_after_cutoff(NEW), True)

    def test_a_partial_update_keeps_the_answers_and_a_full_sync_replaces_them(self):
        self._sync()
        self.metagraph_client.set_block_at_registration(AT_CUTOFF, CUTOFF + 5)
        self.assertIs(self.metagraph_client.is_registered_after_cutoff(AT_CUTOFF), False)
        self.metagraph_client.update_metagraph(hotkeys=HOTKEYS, uids=[0, 1, 2], neurons=neurons_for(HOTKEYS))
        self.assertIs(self.metagraph_client.is_registered_after_cutoff(AT_CUTOFF), False)
        self._sync(cutoff_block=CUTOFF + 5)
        self.assertEqual(self.metagraph_client.get_registered_after_cutoff(), {hk: False for hk in HOTKEYS})
        self._sync(hotkeys=HOTKEYS[:2], blocks=BLOCKS[:2])
        self.assertIsNone(self.metagraph_client.is_registered_after_cutoff(VALIDATOR))

    def test_a_sync_without_stake_weights_carries_them_by_hotkey(self):
        self.metagraph_client.update_metagraph(hotkeys=HOTKEYS, uids=[0, 1, 2], neurons=neurons_for(HOTKEYS),
                                               block_at_registration=BLOCKS, stake_weight=STAKE_WEIGHTS,
                                               standard_miner_cutoff_block=CUTOFF)
        self.assertEqual([a.hotkey for a in self.metagraph_client.get_validator_axons()], [VALIDATOR])
        replacement = "miner_registered_in_the_validators_old_slot"
        swapped = [AT_CUTOFF, NEW, replacement]
        self.metagraph_client.update_metagraph(hotkeys=swapped, uids=[0, 1, 2], neurons=neurons_for(swapped),
                                               block_at_registration=BLOCKS, stake_weight=None,
                                               standard_miner_cutoff_block=CUTOFF)
        self.assertEqual(self.metagraph_client.get_stake_weight(), [0.0, 0.0, 0.0])
        self.assertEqual(self.metagraph_client.get_validator_axons(), [])
        self.assertIs(self.metagraph_client.is_registered_after_cutoff(replacement), True)
        self.assertIsNone(self.metagraph_client.is_registered_after_cutoff(VALIDATOR))

    def test_account_manager_answers_from_the_pushed_copy(self):
        manager = account_manager()
        registered_after = {AT_CUTOFF: False, NEW: True, VALIDATOR: True}
        with mock.patch.object(ValiConfig, "STANDARD_MINER_CUTOFF_BLOCK", CUTOFF):
            self.assertIsNone(manager.is_registered_after_cutoff(NEW))  # nothing pushed yet
            manager.set_registered_after_cutoff(CUTOFF + 1, registered_after)
            self.assertIsNone(manager.is_registered_after_cutoff(NEW))  # a copy made under another cutoff
            manager.set_registered_after_cutoff(CUTOFF, registered_after)
            self.assertEqual(manager.get_registered_after_cutoff(), (CUTOFF, registered_after))
            self.assertIs(manager.is_registered_after_cutoff(NEW), True)
            self.assertIs(manager.is_registered_after_cutoff(VALIDATOR), True)
            self.assertIs(manager.is_registered_after_cutoff(AT_CUTOFF), False)
            self.assertIsNone(manager.is_registered_after_cutoff("unregistered_hotkey"))
            self.assertIs(manager.is_registered_after_cutoff(ValiConfig.DEVELOPMENT_HOTKEY), False)
            self.assertIs(manager.is_registered_after_cutoff(f"{NEW}_1"), False)
            # The copy is kept on disk, so a restarted account server answers until core's next push
            restarted = account_manager()
            self.assertEqual(restarted.get_registered_after_cutoff(), (CUTOFF, registered_after))
            self.assertIs(restarted.is_registered_after_cutoff(NEW), True)
        # With the cutoff unset the answer is False whatever the copy
        self.assertIs(manager.is_registered_after_cutoff(NEW), False)

    def test_account_server_holds_the_pushed_copy(self):
        registered_after = {AT_CUTOFF: False, NEW: True, VALIDATOR: None}
        self.account_client.set_registered_after_cutoff(CUTOFF, registered_after)
        self.assertEqual(self.account_client.get_registered_after_cutoff(), (CUTOFF, registered_after))
        # The test-mode server runs with the cutoff unset, so it answers False whatever the copy
        self.assertIs(self.account_client.is_registered_after_cutoff(NEW), False)
