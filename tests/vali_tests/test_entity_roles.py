"""
Entity roles: registering as an entity miner or a permissionless entity miner, adding the other role to a
registered entity, what each role allows, and how roles persist and sync. The registration fee itself is not
slashed in unit tests (EntityManager skips the collateral check when running_unit_tests).
"""
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.entity_role_enum import EntityRole
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig

EM = EntityRole.ENTITY_MINER.value
PEM = EntityRole.PERMISSIONLESS_ENTITY_MINER.value


class TestEntityRoles(TestBase):

    orchestrator = None
    entity_client = None
    metagraph_client = None
    challenge_period_client = None

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        # Avoid the {text}_{number} pattern, which reads as a synthetic hotkey
        self.ENTITY = "entity_alpha"
        self.OTHER = "entity_beta"
        self.metagraph_client.set_hotkeys([self.ENTITY, self.OTHER])

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    def _roles(self, hotkey):
        return self.entity_client.get_entity_data(hotkey)['roles']

    # ==================== Registration ====================

    def test_default_registration_is_an_entity_miner(self):
        success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY)
        self.assertTrue(success, message)
        data = self.entity_client.get_entity_data(self.ENTITY)
        self.assertEqual(data['roles'], [EM])
        self.assertIsNone(data['pem_take'])
        self.assertIsNone(data['pem_registered_at_ms'])
        self.assertFalse(self.entity_client.is_permissionless_entity_miner(self.ENTITY))

    def test_register_as_a_permissionless_entity_miner(self):
        success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=PEM, pem_take=0.15)
        self.assertTrue(success, message)
        data = self.entity_client.get_entity_data(self.ENTITY)
        self.assertEqual(data['roles'], [PEM])
        self.assertEqual(data['pem_take'], 0.15)
        self.assertIsNotNone(data['pem_registered_at_ms'])
        self.assertTrue(self.entity_client.is_permissionless_entity_miner(self.ENTITY))
        # Same bucket as an entity miner: dust weight, cannot trade
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.ENTITY), MinerBucket.ENTITY)

    def test_an_entity_miner_adds_the_permissionless_role(self):
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY)[0])
        registered_at_ms = self.entity_client.get_entity_data(self.ENTITY)['registered_at_ms']

        success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=PEM, pem_take=0.1)
        self.assertTrue(success, message)
        data = self.entity_client.get_entity_data(self.ENTITY)
        self.assertEqual(data['roles'], [EM, PEM])
        self.assertEqual(data['pem_take'], 0.1)
        self.assertEqual(data['registered_at_ms'], registered_at_ms)
        self.assertTrue(self.entity_client.is_permissionless_entity_miner(self.ENTITY))

    def test_a_permissionless_entity_miner_adds_the_entity_miner_role(self):
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=PEM, pem_take=0.2)[0])
        success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY)
        self.assertTrue(success, message)
        data = self.entity_client.get_entity_data(self.ENTITY)
        self.assertEqual(data['roles'], [PEM, EM])
        self.assertEqual(data['pem_take'], 0.2)

    def test_a_role_cannot_be_registered_twice(self):
        for role, take in ((EM, None), (PEM, 0.1)):
            with self.subTest(role=role):
                self.orchestrator.clear_all_test_data()
                self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=role, pem_take=take)[0])
                success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=role, pem_take=take)
                self.assertFalse(success)
                self.assertIn("already registered", message.lower())
                self.assertEqual(self._roles(self.ENTITY), [role])

    def test_an_unknown_role_is_refused(self):
        for role in ("rem", "pem", "", None, "ENTITY_MINER"):
            with self.subTest(role=role):
                success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=role)
                self.assertFalse(success)
                self.assertIn("Invalid role", message)
        self.assertIsNone(self.entity_client.get_entity_data(self.ENTITY))

    def test_the_take_is_bounded(self):
        for take in (None, 0.0, 0.05, ValiConfig.PEM_MIN_TAKE - 1e-9, 1, 1.5, -0.2, float('nan'), float('inf'), True, "0.2"):
            with self.subTest(take=take):
                success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=PEM, pem_take=take)
                self.assertFalse(success)
                self.assertIn("pem_take", message)
        self.assertIsNone(self.entity_client.get_entity_data(self.ENTITY))
        for take in (ValiConfig.PEM_MIN_TAKE, 0.5, 0.99):
            with self.subTest(take=take):
                self.orchestrator.clear_all_test_data()
                success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=PEM, pem_take=take)
                self.assertTrue(success, message)

    def test_a_take_is_refused_for_the_entity_miner_role(self):
        success, message = self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=EM, pem_take=0.1)
        self.assertFalse(success)
        self.assertIn("pem_take", message)
        self.assertIsNone(self.entity_client.get_entity_data(self.ENTITY))

    def test_the_cap_counts_entities_not_roles(self):
        hotkeys = [f"entity_cap{chr(ord('a') + i)}" for i in range(ValiConfig.MAX_REGISTERED_ENTITIES)]
        for hotkey in hotkeys:
            self.assertTrue(self.entity_client.register_entity(entity_hotkey=hotkey)[0])

        # A new entity is refused at the cap, in either role
        for role, take in ((EM, None), (PEM, 0.1)):
            success, message = self.entity_client.register_entity(entity_hotkey="entity_overflow", role=role, pem_take=take)
            self.assertFalse(success)
            self.assertIn("Maximum number of entities", message)

        # A registered entity still adds a role
        success, message = self.entity_client.register_entity(entity_hotkey=hotkeys[0], role=PEM, pem_take=0.1)
        self.assertTrue(success, message)

    # ==================== What each role allows ====================

    def test_only_an_entity_miner_creates_subaccounts(self):
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=PEM, pem_take=0.1)[0])
        success, info, message = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY, account_size=10_000, asset_class="crypto")
        self.assertFalse(success)
        self.assertIsNone(info)
        self.assertIn(EM, message)
        self.assertEqual(self.entity_client.get_entity_data(self.ENTITY)['subaccounts'], {})

        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY)[0])
        success, info, message = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY, account_size=10_000, asset_class="crypto")
        self.assertTrue(success, message)
        self.assertEqual(info['synthetic_hotkey'], f"{self.ENTITY}_0")

    def test_an_unregistered_hotkey_is_not_a_permissionless_entity_miner(self):
        self.assertFalse(self.entity_client.is_permissionless_entity_miner(self.ENTITY))

    # ==================== Persistence and sync ====================

    def test_roles_round_trip_through_the_checkpoint(self):
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY)[0])
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY, role=PEM, pem_take=0.3)[0])
        checkpoint = self.entity_client.to_checkpoint_dict()
        self.assertEqual(checkpoint[self.ENTITY]['roles'], [EM, PEM])
        self.assertEqual(checkpoint[self.ENTITY]['pem_take'], 0.3)

        self.orchestrator.clear_all_test_data()
        stats = self.entity_client.sync_entity_data(checkpoint)
        self.assertEqual(stats['entities_added'], 1)
        data = self.entity_client.get_entity_data(self.ENTITY)
        self.assertEqual(data['roles'], [EM, PEM])
        self.assertEqual(data['pem_take'], 0.3)
        self.assertTrue(self.entity_client.is_permissionless_entity_miner(self.ENTITY))

    def test_a_record_from_before_roles_is_an_entity_miner(self):
        legacy = {self.ENTITY: {'entity_hotkey': self.ENTITY, 'subaccounts': {}, 'next_subaccount_id': 0,
                                'registered_at_ms': TimeUtil.now_in_millis()}}
        self.entity_client.sync_entity_data(legacy)
        data = self.entity_client.get_entity_data(self.ENTITY)
        self.assertEqual(data['roles'], [EM])
        self.assertIsNone(data['pem_take'])

    def test_sync_adds_a_role_to_a_known_entity(self):
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY)[0])
        incoming = self.entity_client.to_checkpoint_dict()
        incoming[self.ENTITY]['roles'] = [EM, PEM]
        incoming[self.ENTITY]['pem_take'] = 0.25
        incoming[self.ENTITY]['pem_registered_at_ms'] = 123

        stats = self.entity_client.sync_entity_data(incoming)
        self.assertEqual(stats['entities_added'], 0)
        data = self.entity_client.get_entity_data(self.ENTITY)
        self.assertEqual(data['roles'], [EM, PEM])
        self.assertEqual(data['pem_take'], 0.25)
        self.assertEqual(data['pem_registered_at_ms'], 123)

        # Syncing again changes nothing, and a role is never removed by sync
        incoming[self.ENTITY]['roles'] = [EM]
        self.entity_client.sync_entity_data(incoming)
        self.assertEqual(self._roles(self.ENTITY), [EM, PEM])
