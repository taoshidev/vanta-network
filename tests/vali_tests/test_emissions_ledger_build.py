"""
Smoke test of the emissions ledger build in test mode (mocked chain data) over more hotkeys than the
256 UIDs a subnet held historically.
"""
import os

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import generate_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.emission.emissions_ledger import EmissionsLedgerManager


class TestEmissionsLedgerBuild(TestBase):
    """The build must not cap how many hotkeys it tracks: every hotkey it discovers gets an
    emissions ledger, however many there are. Test mode discovers them from the perf ledgers."""

    NUM_HOTKEYS = 300
    NUM_CHECKPOINTS = 4

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')

    def setUp(self):
        super().setUp()
        self.orchestrator.clear_all_test_data()
        self.hotkeys = [f"test_miner_{i}" for i in range(self.NUM_HOTKEYS)]

        # Complete checkpoints on checkpoint boundaries that all end before now, so the build
        # processes every one of them
        duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        start_ms = (TimeUtil.now_in_millis() // duration_ms - self.NUM_CHECKPOINTS - 1) * duration_ms
        ledgers = {
            hotkey: generate_ledger(value=0.01, start_time=start_ms, nterms=self.NUM_CHECKPOINTS)
            for hotkey in self.hotkeys
        }
        self.perf_ledger_client.save_perf_ledgers(ledgers)
        self.perf_ledger_client.re_init_perf_ledger_data()

        self.manager = EmissionsLedgerManager(running_unit_tests=True)
        self.manager.emissions_ledgers = {}

    def tearDown(self):
        ledger_path = self.manager._get_ledger_path()
        if os.path.exists(ledger_path):
            os.remove(ledger_path)
        self.orchestrator.clear_all_test_data()
        super().tearDown()

    def test_build_covers_more_than_256_hotkeys(self):
        self.manager.build_all_emissions_ledgers_optimized()

        self.assertEqual(len(self.manager.emissions_ledgers), self.NUM_HOTKEYS)
        for hotkey in self.hotkeys:
            ledger = self.manager.emissions_ledgers[hotkey]
            self.assertEqual(len(ledger.checkpoints), self.NUM_CHECKPOINTS, hotkey)
