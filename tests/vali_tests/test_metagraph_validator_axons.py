"""
Validator selection by chain stake weight, served through the metagraph server.
"""
from dataclasses import dataclass

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig


@dataclass
class AxonStub:
    hotkey: str
    ip: str


@dataclass
class NeuronStub:
    uid: int
    hotkey: str
    axon_info: AxonStub
    validator_permit: bool = False


class TestMetagraphValidatorAxons(TestBase):
    """Validators are the UIDs whose chain stake weight reaches STAKE_MIN, never the permit holders:
    the chain may grant the permit to one UID only."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()

    def tearDown(self):
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    @staticmethod
    def _neuron(uid, hotkey, ip="1.2.3.4", validator_permit=False):
        return NeuronStub(uid=uid, hotkey=hotkey, axon_info=AxonStub(hotkey=hotkey, ip=ip),
                          validator_permit=validator_permit)

    def _publish(self, neurons, stake_weight):
        self.metagraph_client.update_metagraph(
            neurons=neurons, uids=[n.uid for n in neurons], hotkeys=[n.hotkey for n in neurons],
            stake_weight=stake_weight,
        )

    def _validator_hotkeys(self, exclude_hotkey=None):
        return [a.hotkey for a in self.metagraph_client.get_validator_axons(exclude_hotkey=exclude_hotkey)]

    def test_validators_are_selected_by_stake_weight_and_served_ip(self):
        neurons = [
            self._neuron(0, "mothership"),
            self._neuron(1, "staked"),
            self._neuron(2, "permit_low_stake", validator_permit=True),
            self._neuron(3, "staked_no_ip", ip=ValiConfig.AXON_NO_IP),
            self._neuron(4, "at_minimum"),
            self._neuron(5, "below_minimum"),
            self._neuron(6, "beyond_stake_list"),
        ]
        stake_weight = [50_000.0, 5_000.0, 2.0, 5_000.0, ValiConfig.STAKE_MIN, ValiConfig.STAKE_MIN - 0.5]
        self._publish(neurons, stake_weight)

        self.assertEqual(self.metagraph_client.get_stake_weight(), stake_weight)
        self.assertEqual(self._validator_hotkeys(), ["mothership", "staked", "at_minimum"])
        self.assertEqual(self._validator_hotkeys(exclude_hotkey="mothership"), ["staked", "at_minimum"])

    def test_validator_set_follows_each_update(self):
        neurons = [self._neuron(0, "a"), self._neuron(1, "b")]
        self._publish(neurons, [5_000.0, 0.0])
        self.assertEqual(self._validator_hotkeys(), ["a"])
        self._publish(neurons, [0.0, 5_000.0])
        self.assertEqual(self._validator_hotkeys(), ["b"])
