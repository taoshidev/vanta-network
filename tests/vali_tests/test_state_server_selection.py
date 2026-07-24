# developer: Taoshi Inc
# Copyright (c) 2024 Taoshi Inc
"""
Hermetic unit tests for the vanta-state tiering logic (R6).

Pure selection + context logic ONLY — no server spawning, no RPC, no chain. Deliberately does not
touch the integration harness (which is brittle post order-execution refactor). Covers:
  - ServerOrchestrator._select_servers (include-set / exclude-set / mode filtering / validation)
  - The vanta-state include-set (VANTA_STATE_SERVERS) vs the core tier partition
  - NeuronContext.validator_hotkey_override (wallet-less identity)
"""
import types
import unittest

from shared_objects.rpc.server_orchestrator import (
    ServerOrchestrator,
    ServerMode,
    NeuronContext,
)


class TestStateServerSelection(unittest.TestCase):
    def setUp(self):
        self.orch = ServerOrchestrator.get_instance()

    def test_include_set_names_are_valid(self):
        """VANTA_STATE_SERVERS must be a subset of the registry (guards against a typo'd constant)."""
        unknown = set(ServerOrchestrator.VANTA_STATE_SERVERS) - set(ServerOrchestrator.SERVERS)
        self.assertEqual(unknown, set(), f"VANTA_STATE_SERVERS has unknown names: {unknown}")

    def test_state_include_set_selects_exactly_the_tier(self):
        selected = self.orch._select_servers(
            ServerMode.VALIDATOR,
            include_servers=set(ServerOrchestrator.VANTA_STATE_SERVERS),
        )
        self.assertEqual(set(selected), set(ServerOrchestrator.VANTA_STATE_SERVERS))
        # The wallet/chain + collateral + scoring servers must NOT be in the state tier.
        for core_only in ('subtensor_ops', 'contract', 'perf_ledger', 'weight_calculator', 'metagraph'):
            self.assertNotIn(core_only, selected)

    def test_core_exclude_set_is_the_complement(self):
        core = self.orch._select_servers(
            ServerMode.VALIDATOR,
            exclude_servers=set(ServerOrchestrator.VANTA_STATE_SERVERS),
        )
        # Core keeps the wallet/chain + collateral + scoring; none of the state tier.
        self.assertIn('subtensor_ops', core)
        self.assertIn('contract', core)
        for state_srv in ServerOrchestrator.VANTA_STATE_SERVERS:
            self.assertNotIn(state_srv, core)

    def test_state_and_core_partition_the_full_validator_tier(self):
        full = set(self.orch._select_servers(ServerMode.VALIDATOR))
        state = set(self.orch._select_servers(
            ServerMode.VALIDATOR, include_servers=set(ServerOrchestrator.VANTA_STATE_SERVERS)))
        core = set(self.orch._select_servers(
            ServerMode.VALIDATOR, exclude_servers=set(ServerOrchestrator.VANTA_STATE_SERVERS)))
        # No overlap, and together they cover the whole validator tier.
        self.assertEqual(state & core, set())
        self.assertEqual(state | core, full)

    def test_unknown_include_name_raises(self):
        with self.assertRaises(ValueError):
            self.orch._select_servers(ServerMode.VALIDATOR, include_servers={'not_a_server'})

    def test_unknown_exclude_name_raises(self):
        with self.assertRaises(ValueError):
            self.orch._select_servers(ServerMode.VALIDATOR, exclude_servers={'subtensor-ops'})  # typo'd

    def test_mode_filtering_still_applies_under_include(self):
        # A server that is not required in the mode is not started even if explicitly included.
        # hl_funding is required_in_validator but NOT required_in_testing.
        selected_testing = self.orch._select_servers(
            ServerMode.TESTING, include_servers={'hl_funding'})
        self.assertNotIn('hl_funding', selected_testing)


class TestValidatorHotkeyOverride(unittest.TestCase):
    def test_override_used_when_walletless(self):
        ctx = NeuronContext(wallet=None, validator_hotkey_override='5Fss58abc')
        self.assertEqual(ctx.validator_hotkey, '5Fss58abc')

    def test_wallet_wins_over_override(self):
        wallet = types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address='5Fwallet'))
        ctx = NeuronContext(wallet=wallet, validator_hotkey_override='5Foverride')
        self.assertEqual(ctx.validator_hotkey, '5Fwallet')

    def test_none_when_neither(self):
        ctx = NeuronContext(wallet=None, validator_hotkey_override=None)
        self.assertIsNone(ctx.validator_hotkey)


if __name__ == '__main__':
    unittest.main()
