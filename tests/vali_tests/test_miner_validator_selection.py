"""
Which validators a miner sends its orders to: the mothership by its hotkey, the others by chain stake weight.
"""
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from miner_objects.position_inspector import PositionInspector
from miner_objects.prop_net_order_placer import PropNetOrderPlacer
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
    validator_trust: float = 0.0
    validator_permit: bool = False


class TestMinerValidatorSelection(TestBase):
    """Under Null consensus one validator holds the permit and the validator trust. The miner still sends every order
    to all of them, chosen by stake weight as the validators choose each other, and finds the mothership by hotkey."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()
        self.placer = self._placer(is_testnet=False)

    def tearDown(self):
        self.placer.shutdown()
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def _placer(self, is_testnet):
        config = MagicMock(netuid=116 if is_testnet else 8)
        inspector = PositionInspector(MagicMock(), self.metagraph_client, config, running_unit_tests=True)
        return PropNetOrderPlacer(MagicMock(), self.metagraph_client, config, is_testnet,
                                 position_inspector=inspector, running_unit_tests=True)

    @staticmethod
    def _neuron(uid, hotkey, ip="1.2.3.4", validator_trust=0.0, validator_permit=False):
        return NeuronStub(uid=uid, hotkey=hotkey, axon_info=AxonStub(hotkey=hotkey, ip=ip),
                          validator_trust=validator_trust, validator_permit=validator_permit)

    def _publish(self, neurons, stake_weight):
        self.metagraph_client.update_metagraph(
            neurons=neurons, uids=[n.uid for n in neurons], hotkeys=[n.hotkey for n in neurons],
            stake_weight=stake_weight,
        )

    def _targets(self, placer=None):
        mothership, others = (placer or self.placer)._get_mothership_and_other_axons()
        return (mothership.hotkey if mothership else None), [a.hotkey for a in others]

    def test_orders_go_to_the_mothership_and_to_every_validator_by_stake(self):
        neurons = [
            self._neuron(0, ValiConfig.MOTHERSHIP_HOTKEY, validator_trust=1.0, validator_permit=True),
            self._neuron(1, "staked_no_trust"),
            self._neuron(2, "permit_low_stake", validator_trust=0.9, validator_permit=True),
            self._neuron(3, "staked_no_ip", ip=ValiConfig.AXON_NO_IP),
            self._neuron(4, "miner"),
        ]
        self._publish(neurons, [50_000.0, 5_000.0, 2.0, 5_000.0, 0.0])

        self.assertEqual(self._targets(), (ValiConfig.MOTHERSHIP_HOTKEY, ["staked_no_trust"]))
        self.assertEqual([a.hotkey for a in self.placer.position_inspector.get_possible_validators()],
                         [ValiConfig.MOTHERSHIP_HOTKEY, "staked_no_trust"])

    def test_mothership_is_found_by_hotkey_outside_the_stake_rule(self):
        neurons = [self._neuron(0, ValiConfig.MOTHERSHIP_HOTKEY), self._neuron(1, "staked")]
        self._publish(neurons, [0.0, 5_000.0])
        self.assertEqual(self._targets(), (ValiConfig.MOTHERSHIP_HOTKEY, ["staked"]))
        # The position inspector queries the validators by the stake rule alone
        self.assertEqual([a.hotkey for a in self.placer.position_inspector.get_possible_validators()], ["staked"])

    def test_mothership_without_a_served_ip_is_not_a_target(self):
        neurons = [self._neuron(0, ValiConfig.MOTHERSHIP_HOTKEY, ip=ValiConfig.AXON_NO_IP), self._neuron(1, "staked")]
        self._publish(neurons, [50_000.0, 5_000.0])
        self.assertEqual(self._targets(), (None, ["staked"]))

    def test_an_empty_stake_list_leaves_the_mothership_alone_and_warns_once(self):
        # A sync whose stake query failed carries nothing over on a fresh miner: no validator passes the stake rule
        neurons = [self._neuron(0, ValiConfig.MOTHERSHIP_HOTKEY), self._neuron(1, "staked")]
        self._publish(neurons, None)
        with patch("miner_objects.prop_net_order_placer.logger") as log:
            self.assertEqual(self._targets(), (ValiConfig.MOTHERSHIP_HOTKEY, []))
            self.assertEqual(self.placer.position_inspector.get_possible_validators(), [])
            self.assertEqual(self._targets(), (ValiConfig.MOTHERSHIP_HOTKEY, []))
            log.warning.assert_called_once()

    def test_testnet_sends_to_every_validator_serving_an_ip_whatever_its_stake(self):
        placer = self._placer(is_testnet=True)
        try:
            neurons = [self._neuron(0, ValiConfig.MOTHERSHIP_HOTKEY_TESTNET), self._neuron(1, "low_stake"),
                       self._neuron(2, "no_ip", ip=ValiConfig.AXON_NO_IP), self._neuron(3, ValiConfig.MOTHERSHIP_HOTKEY)]
            self._publish(neurons, [50_000.0, 2.0, 5_000.0, 50_000.0])
            self.assertEqual(self._targets(placer),
                             (ValiConfig.MOTHERSHIP_HOTKEY_TESTNET, ["low_stake", ValiConfig.MOTHERSHIP_HOTKEY]))
        finally:
            placer.shutdown()
