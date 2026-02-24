# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Hyperliquid integration tests covering:
- HL subaccount creation and validation
- HL address reverse index lookups
- Broadcast with hl_address threading
- Receive registration with hl_address
- HL subaccount trade blocking (should_fail_early equivalent)
- HyperliquidTracker fill processing, dedup, coin mapping, leverage calculation
- Sync entity data with HL addresses
"""
import re
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair, TRADE_PAIR_ID_TO_TRADE_PAIR
from time_util.time_util import TimeUtil
from entity_management.entity_utils import is_synthetic_hotkey, parse_synthetic_hotkey
from entity_management.hyperliquid_tracker import HyperliquidTracker


# ==================== Valid/invalid HL addresses for testing ====================
VALID_HL_ADDRESS = "0x" + "a1b2c3d4" * 5       # 0x + 40 hex chars
VALID_HL_ADDRESS_2 = "0x" + "1234567890abcdef" * 2 + "12345678"  # 0x + 40 hex chars
VALID_HL_ADDRESS_3 = "0x" + "ff" * 20           # 0x + 40 hex chars
INVALID_HL_SHORT = "0xabc"
INVALID_HL_NO_PREFIX = "a1b2c3d4" * 5           # 40 hex chars, no 0x prefix
INVALID_HL_BAD_CHARS = "0x" + "zzzz" * 10


class TestHyperliquidSubaccounts(TestBase):
    """
    Tests for HL subaccount creation, lookups, and entity management integration.

    Uses ServerOrchestrator for full client/server architecture (same pattern
    as TestEntityManagement).
    """

    orchestrator = None
    entity_client = None
    metagraph_client = None

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.ENTITY_HOTKEY_1 = "entity_alpha"
        self.ENTITY_HOTKEY_2 = "entity_beta"
        self.metagraph_client.set_hotkeys([self.ENTITY_HOTKEY_1, self.ENTITY_HOTKEY_2])

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    # ==================== HL Subaccount Creation ====================

    def test_create_hl_subaccount_success(self):
        """Test successful HL subaccount creation."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )

        self.assertTrue(success, f"HL subaccount creation failed: {message}")
        self.assertIsNotNone(subaccount_info)
        self.assertEqual(subaccount_info['subaccount_id'], 0)
        # Asset class should be auto-set to "crypto" for HL subaccounts
        self.assertEqual(subaccount_info['asset_class'], 'crypto')

    def test_create_hl_subaccount_invalid_address_format(self):
        """Test HL subaccount creation fails with invalid address formats."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        for invalid_addr in [INVALID_HL_SHORT, INVALID_HL_NO_PREFIX, INVALID_HL_BAD_CHARS, "", "0x"]:
            success, _, message = self.entity_client.create_hl_subaccount(
                entity_hotkey=self.ENTITY_HOTKEY_1,
                account_size=50_000,
                hl_address=invalid_addr
            )
            self.assertFalse(success, f"Should reject invalid address: {invalid_addr}")
            self.assertIn("invalid", message.lower())

    def test_create_hl_subaccount_duplicate_address(self):
        """Test HL subaccount creation fails if address already registered."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # First creation should succeed
        success, _, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertTrue(success)

        # Second creation with same address should fail
        success, _, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertFalse(success)
        self.assertIn("already registered", message.lower())

    def test_create_hl_subaccount_duplicate_address_across_entities(self):
        """Test HL address uniqueness is enforced across different entities."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_2)

        # Register address on entity 1
        success, _, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertTrue(success)

        # Try same address on entity 2
        success, _, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_2,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertFalse(success)
        self.assertIn("already registered", message.lower())

    def test_create_hl_subaccount_unregistered_entity(self):
        """Test HL subaccount creation fails for unregistered entity."""
        success, _, message = self.entity_client.create_hl_subaccount(
            entity_hotkey="unregistered_entity",
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertFalse(success)
        self.assertIn("not registered", message.lower())

    def test_create_hl_subaccount_admin_flag(self):
        """Test HL subaccount creation with admin flag."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS,
            admin=True
        )

        self.assertTrue(success)
        self.assertEqual(subaccount_info['status'], 'admin')

    # ==================== Payout Address ====================

    def test_create_hl_subaccount_with_payout_address(self):
        """Test HL subaccount creation with a valid payout address."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        payout_addr = "0x" + "de" * 20

        success, subaccount_info, message = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS,
            payout_address=payout_addr
        )

        self.assertTrue(success, f"HL subaccount with payout_address failed: {message}")
        self.assertIsNotNone(subaccount_info)

        info = self.entity_client.get_subaccount_info_for_synthetic(
            subaccount_info['synthetic_hotkey']
        )
        self.assertIsNotNone(info)
        self.assertEqual(info['payout_address'], payout_addr)
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)

    def test_create_hl_subaccount_without_payout_address(self):
        """Test HL subaccount creation without payout_address defaults to None."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertTrue(success)

        info = self.entity_client.get_subaccount_info_for_synthetic(
            subaccount_info['synthetic_hotkey']
        )
        self.assertIsNotNone(info)
        self.assertIsNone(info.get('payout_address'))

    def test_create_hl_subaccount_invalid_payout_address(self):
        """Test HL subaccount creation fails with invalid payout_address formats."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        for invalid_addr in ["0xabc", "not_an_address", "0x" + "zz" * 20, ""]:
            success, _, message = self.entity_client.create_hl_subaccount(
                entity_hotkey=self.ENTITY_HOTKEY_1,
                account_size=50_000,
                hl_address=VALID_HL_ADDRESS_2,
                payout_address=invalid_addr
            )
            self.assertFalse(success, f"Should reject invalid payout_address: {invalid_addr}")
            self.assertIn("payout_address", message.lower())

    def test_regular_subaccount_has_no_payout_address(self):
        """Test that regular (non-HL) subaccounts have None payout_address."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, _ = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            asset_class="crypto"
        )
        self.assertTrue(success)

        info = self.entity_client.get_subaccount_info_for_synthetic(
            subaccount_info['synthetic_hotkey']
        )
        self.assertIsNotNone(info)
        self.assertIsNone(info.get('payout_address'))

    # ==================== HL Address Reverse Index Lookups ====================

    def test_get_synthetic_hotkey_for_hl_address(self):
        """Test O(1) reverse lookup from HL address to synthetic hotkey."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertTrue(success)

        # Lookup by HL address
        synthetic_hotkey = self.entity_client.get_synthetic_hotkey_for_hl_address(VALID_HL_ADDRESS)
        self.assertEqual(synthetic_hotkey, subaccount_info['synthetic_hotkey'])

    def test_get_synthetic_hotkey_for_unknown_hl_address(self):
        """Test lookup returns None for unknown HL address."""
        result = self.entity_client.get_synthetic_hotkey_for_hl_address(VALID_HL_ADDRESS)
        self.assertIsNone(result)

    def test_get_all_active_hl_subaccounts(self):
        """Test listing all active HL subaccounts."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        # Create 2 HL subaccounts
        self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=30_000,
            hl_address=VALID_HL_ADDRESS_2
        )

        hl_subaccounts = self.entity_client.get_all_active_hl_subaccounts()
        self.assertEqual(len(hl_subaccounts), 2)
        addresses = {addr for addr, _ in hl_subaccounts}
        self.assertIn(VALID_HL_ADDRESS, addresses)
        self.assertIn(VALID_HL_ADDRESS_2, addresses)

    def test_get_all_active_hl_subaccounts_excludes_eliminated(self):
        """Test that eliminated HL subaccounts are not returned."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)

        success, subaccount_info, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertTrue(success)

        # Eliminate the subaccount
        self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            subaccount_id=subaccount_info['subaccount_id'],
            reason="test_elimination"
        )

        hl_subaccounts = self.entity_client.get_all_active_hl_subaccounts()
        self.assertEqual(len(hl_subaccounts), 0)

    def test_get_subaccount_info_for_synthetic_with_hl_address(self):
        """Test that subaccount info includes hl_address field."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertTrue(success)

        info = self.entity_client.get_subaccount_info_for_synthetic(
            subaccount_info['synthetic_hotkey']
        )
        self.assertIsNotNone(info)
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)

    def test_get_subaccount_info_for_synthetic_without_hl_address(self):
        """Test that regular subaccount info has None hl_address."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, _ = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            asset_class="crypto"
        )
        self.assertTrue(success)

        info = self.entity_client.get_subaccount_info_for_synthetic(
            subaccount_info['synthetic_hotkey']
        )
        self.assertIsNotNone(info)
        self.assertIsNone(info.get('hl_address'))

    # ==================== HL Address in Broadcast / Sync ====================

    def test_sync_populates_hl_address_and_reverse_index(self):
        """Test that sync_entity_data with hl_address populates reverse index."""
        now_ms = TimeUtil.now_in_millis()
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-hl',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 50_000,
                        'asset_class': 'crypto',
                        'hl_address': VALID_HL_ADDRESS
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }

        stats = self.entity_client.sync_entity_data(checkpoint_dict)
        self.assertEqual(stats['subaccounts_added'], 1)

        # Verify HL address reverse index was populated
        synthetic = self.entity_client.get_synthetic_hotkey_for_hl_address(VALID_HL_ADDRESS)
        self.assertEqual(synthetic, f"{self.ENTITY_HOTKEY_1}_0")

        # Verify subaccount info has hl_address
        info = self.entity_client.get_subaccount_info_for_synthetic(f"{self.ENTITY_HOTKEY_1}_0")
        self.assertIsNotNone(info)
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)

    def test_sync_without_hl_address_has_none(self):
        """Test that sync_entity_data without hl_address leaves it None."""
        now_ms = TimeUtil.now_in_millis()
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-regular',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 100_000,
                        'asset_class': 'crypto',
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }

        self.entity_client.sync_entity_data(checkpoint_dict)

        # HL address lookup should return None
        synthetic = self.entity_client.get_synthetic_hotkey_for_hl_address(VALID_HL_ADDRESS)
        self.assertIsNone(synthetic)

        info = self.entity_client.get_subaccount_info_for_synthetic(f"{self.ENTITY_HOTKEY_1}_0")
        self.assertIsNotNone(info)
        self.assertIsNone(info.get('hl_address'))

    def test_sync_idempotent_adds_hl_address(self):
        """Test that re-syncing can add hl_address to existing subaccount."""
        now_ms = TimeUtil.now_in_millis()

        # First sync: without hl_address
        checkpoint_no_hl = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-sync',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 50_000,
                        'asset_class': 'crypto',
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }
        self.entity_client.sync_entity_data(checkpoint_no_hl)

        info = self.entity_client.get_subaccount_info_for_synthetic(f"{self.ENTITY_HOTKEY_1}_0")
        self.assertIsNone(info.get('hl_address'))

        # Second sync: same UUID, now with hl_address
        checkpoint_with_hl = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-sync',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 50_000,
                        'asset_class': 'crypto',
                        'hl_address': VALID_HL_ADDRESS
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }
        self.entity_client.sync_entity_data(checkpoint_with_hl)

        # Verify HL address is now set
        info = self.entity_client.get_subaccount_info_for_synthetic(f"{self.ENTITY_HOTKEY_1}_0")
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)

        # Verify reverse index populated
        synthetic = self.entity_client.get_synthetic_hotkey_for_hl_address(VALID_HL_ADDRESS)
        self.assertEqual(synthetic, f"{self.ENTITY_HOTKEY_1}_0")

    # ==================== HL Trade Blocking ====================

    def test_hl_subaccount_blocked_from_direct_orders(self):
        """
        Test that HL-linked subaccounts are detected by subaccount_info check.

        This validates the data-level check used by should_fail_early() in validator.py:
        if subaccount_info.get('hl_address') -> reject direct trades.
        """
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, _ = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=50_000,
            hl_address=VALID_HL_ADDRESS
        )
        self.assertTrue(success)

        # Simulate what should_fail_early does
        info = self.entity_client.get_subaccount_info_for_synthetic(
            subaccount_info['synthetic_hotkey']
        )
        self.assertIsNotNone(info)
        # The validator checks this to block direct trades
        self.assertIsNotNone(info.get('hl_address'))
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)

    def test_regular_subaccount_not_blocked(self):
        """Test that regular subaccounts are NOT blocked from direct orders."""
        self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY_1)
        success, subaccount_info, _ = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY_1,
            account_size=100_000,
            asset_class="crypto"
        )
        self.assertTrue(success)

        info = self.entity_client.get_subaccount_info_for_synthetic(
            subaccount_info['synthetic_hotkey']
        )
        self.assertIsNotNone(info)
        # Regular subaccounts should NOT have hl_address
        self.assertIsNone(info.get('hl_address'))

    # ==================== Sync with HL Addresses ====================

    def test_sync_entity_data_with_hl_address(self):
        """Test that syncing entity data preserves hl_address and reverse index."""
        now_ms = TimeUtil.now_in_millis()
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-hl',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 50_000,
                        'asset_class': 'crypto',
                        'hl_address': VALID_HL_ADDRESS
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }

        stats = self.entity_client.sync_entity_data(checkpoint_dict)
        self.assertEqual(stats['entities_added'], 1)
        self.assertEqual(stats['subaccounts_added'], 1)

        # Verify subaccount has hl_address
        info = self.entity_client.get_subaccount_info_for_synthetic(f'{self.ENTITY_HOTKEY_1}_0')
        self.assertIsNotNone(info)
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)

    def test_sync_entity_data_with_payout_address(self):
        """Test that syncing entity data preserves payout_address."""
        now_ms = TimeUtil.now_in_millis()
        payout_addr = "0x" + "ab" * 20
        checkpoint_dict = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-hl-payout',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 50_000,
                        'asset_class': 'crypto',
                        'hl_address': VALID_HL_ADDRESS,
                        'payout_address': payout_addr
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }

        stats = self.entity_client.sync_entity_data(checkpoint_dict)
        self.assertEqual(stats['entities_added'], 1)

        info = self.entity_client.get_subaccount_info_for_synthetic(f'{self.ENTITY_HOTKEY_1}_0')
        self.assertIsNotNone(info)
        self.assertEqual(info['hl_address'], VALID_HL_ADDRESS)
        self.assertEqual(info['payout_address'], payout_addr)

    def test_sync_idempotent_adds_payout_address(self):
        """Test that re-syncing can add payout_address to existing subaccount."""
        now_ms = TimeUtil.now_in_millis()
        payout_addr = "0x" + "cd" * 20

        # First sync: without payout_address
        checkpoint_no_payout = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-sync-payout',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 50_000,
                        'asset_class': 'crypto',
                        'hl_address': VALID_HL_ADDRESS
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }
        self.entity_client.sync_entity_data(checkpoint_no_payout)

        info = self.entity_client.get_subaccount_info_for_synthetic(f'{self.ENTITY_HOTKEY_1}_0')
        self.assertIsNone(info.get('payout_address'))

        # Second sync: same UUID, now with payout_address
        checkpoint_with_payout = {
            self.ENTITY_HOTKEY_1: {
                'entity_hotkey': self.ENTITY_HOTKEY_1,
                'subaccounts': {
                    '0': {
                        'subaccount_id': 0,
                        'subaccount_uuid': 'uuid-sync-payout',
                        'synthetic_hotkey': f'{self.ENTITY_HOTKEY_1}_0',
                        'status': 'active',
                        'created_at_ms': now_ms,
                        'eliminated_at_ms': None,
                        'account_size': 50_000,
                        'asset_class': 'crypto',
                        'hl_address': VALID_HL_ADDRESS,
                        'payout_address': payout_addr
                    }
                },
                'next_subaccount_id': 1,
                'registered_at_ms': now_ms
            }
        }
        self.entity_client.sync_entity_data(checkpoint_with_payout)

        info = self.entity_client.get_subaccount_info_for_synthetic(f'{self.ENTITY_HOTKEY_1}_0')
        self.assertEqual(info['payout_address'], payout_addr)

    # ==================== HL Address Format Validation ====================

    def test_hl_address_regex_valid(self):
        """Test ValiConfig.HL_ADDRESS_REGEX matches valid addresses."""
        valid_addresses = [
            "0x" + "a" * 40,
            "0x" + "A" * 40,
            "0x" + "0123456789abcdef" * 2 + "01234567",
            "0x" + "aAbBcCdDeEfF0123" * 2 + "aAbBcCdD",
        ]
        for addr in valid_addresses:
            self.assertRegex(addr, ValiConfig.HL_ADDRESS_REGEX, f"Should match: {addr}")

    def test_hl_address_regex_invalid(self):
        """Test ValiConfig.HL_ADDRESS_REGEX rejects invalid addresses."""
        invalid_addresses = [
            "0x" + "a" * 39,         # too short
            "0x" + "a" * 41,         # too long
            "a" * 42,                # no 0x prefix
            "0x" + "g" * 40,         # non-hex chars
            "",                       # empty
            "0x",                     # prefix only
        ]
        for addr in invalid_addresses:
            self.assertNotRegex(addr, ValiConfig.HL_ADDRESS_REGEX, f"Should NOT match: {addr}")


class TestHyperliquidTracker(TestBase):
    """
    Tests for HyperliquidTracker fill processing, dedup, and signal conversion.

    Uses mocks for all external dependencies (no real WebSocket or RPC).
    """

    def setUp(self):
        """Create HyperliquidTracker with all mocked dependencies."""
        self.entity_client = MagicMock()
        self.elimination_client = MagicMock()
        self.price_fetcher_client = MagicMock()
        self.asset_selection_client = MagicMock()
        self.market_order_manager = MagicMock()
        self.limit_order_client = MagicMock()
        self.uuid_tracker = MagicMock()
        self.rate_limiter = MagicMock()

        self.tracker = HyperliquidTracker(
            entity_client=self.entity_client,
            elimination_client=self.elimination_client,
            price_fetcher_client=self.price_fetcher_client,
            asset_selection_client=self.asset_selection_client,
            market_order_manager=self.market_order_manager,
            limit_order_client=self.limit_order_client,
            uuid_tracker=self.uuid_tracker,
            rate_limiter=self.rate_limiter,
        )

    def _make_fill(self, coin="BTC", side="B", sz="1.0", px="50000.0", fill_hash="hash_1"):
        """Helper to create a fill dict."""
        return {
            "coin": coin,
            "side": side,
            "sz": sz,
            "px": px,
            "hash": fill_hash,
        }

    def _setup_successful_fill_mocks(self, synthetic_hotkey="entity_alpha_0", account_size=100_000):
        """Set up mocks for a successful fill processing scenario."""
        self.entity_client.get_synthetic_hotkey_for_hl_address.return_value = synthetic_hotkey
        self.entity_client.get_subaccount_info_for_synthetic.return_value = {
            "account_size": account_size,
            "status": "active",
            "hl_address": VALID_HL_ADDRESS,
        }
        self.rate_limiter.is_allowed.return_value = (True, 0)
        self.elimination_client.get_elimination_local_cache.return_value = None
        self.entity_client.validate_hotkey_for_orders.return_value = {
            "is_valid": True, "error_message": ""
        }
        self.price_fetcher_client.is_market_open.return_value = True

        # Mock USDC balance check to pass by default
        self.tracker._get_hl_usdc_balance = MagicMock(return_value=5000.0)

        # OrderProcessor mock
        mock_result = MagicMock()
        mock_result.should_track_uuid = True
        return mock_result

    # ==================== Coin Mapping ====================

    def test_hl_coin_to_trade_pair_mapping(self):
        """Test that HL_COIN_TO_TRADE_PAIR maps all supported coins correctly."""
        expected = {
            "BTC": "BTCUSD",
            "ETH": "ETHUSD",
            "SOL": "SOLUSD",
            "XRP": "XRPUSD",
            "DOGE": "DOGEUSD",
            "ADA": "ADAUSD",
        }
        self.assertEqual(ValiConfig.HL_COIN_TO_TRADE_PAIR, expected)

    def test_all_mapped_trade_pairs_exist(self):
        """Test that all trade pair IDs in the mapping resolve to actual TradePair objects."""
        for coin, trade_pair_id in ValiConfig.HL_COIN_TO_TRADE_PAIR.items():
            tp = TRADE_PAIR_ID_TO_TRADE_PAIR.get(trade_pair_id)
            self.assertIsNotNone(tp, f"Trade pair {trade_pair_id} for coin {coin} not found in TRADE_PAIR_ID_TO_TRADE_PAIR")

    # ==================== Fill Dedup ====================

    def test_record_hash_basic_dedup(self):
        """Test that duplicate fill hashes are detected."""
        self.tracker._record_hash("hash_1")
        self.assertIn("hash_1", self.tracker._processed_hashes)

        # Recording same hash again should not raise
        self.tracker._record_hash("hash_1")
        self.assertEqual(len(self.tracker._processed_hashes), 1)

    def test_record_hash_bounded_eviction(self):
        """Test that oldest hashes are evicted when MAX_DEDUP_HASHES is exceeded."""
        # Fill to max
        for i in range(HyperliquidTracker.MAX_DEDUP_HASHES):
            self.tracker._record_hash(f"hash_{i}")

        self.assertEqual(len(self.tracker._processed_hashes), HyperliquidTracker.MAX_DEDUP_HASHES)

        # Add one more - should evict oldest
        self.tracker._record_hash("hash_overflow")
        self.assertEqual(len(self.tracker._processed_hashes), HyperliquidTracker.MAX_DEDUP_HASHES)
        self.assertNotIn("hash_0", self.tracker._processed_hashes)
        self.assertIn("hash_overflow", self.tracker._processed_hashes)

    def test_handle_user_fills_dedup_skips_processed(self):
        """Test that _handle_user_fills skips already-processed fill hashes."""
        # Pre-record a hash
        self.tracker._record_hash("existing_hash")

        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": False,
                "user": VALID_HL_ADDRESS,
                "fills": [{"hash": "existing_hash", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}]
            }
        }

        # _process_fill should NOT be called since hash is duplicate
        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            mock_process.assert_not_called()

    def test_handle_user_fills_skips_snapshot(self):
        """Test that snapshot fills are recorded for dedup but not processed."""
        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": True,
                "user": VALID_HL_ADDRESS,
                "fills": [{"hash": "snap_hash", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}]
            }
        }

        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            mock_process.assert_not_called()

        # But hash should still be recorded for future dedup
        self.assertIn("snap_hash", self.tracker._processed_hashes)

    def test_handle_user_fills_processes_new_fill(self):
        """Test that new fills (non-snapshot, new hash) are processed."""
        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": False,
                "user": VALID_HL_ADDRESS,
                "fills": [{"hash": "new_hash", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}]
            }
        }

        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            mock_process.assert_called_once_with(
                VALID_HL_ADDRESS,
                {"hash": "new_hash", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}
            )

    # ==================== Message Routing ====================

    def test_handle_message_routes_user_fills(self):
        """Test that userFills messages are routed to _handle_user_fills."""
        msg = {"channel": "userFills", "data": {"user": VALID_HL_ADDRESS, "fills": []}}

        with patch.object(self.tracker, '_handle_user_fills') as mock_handler:
            self.tracker._handle_message(msg)
            mock_handler.assert_called_once_with(msg)

    def test_handle_message_ignores_pong(self):
        """Test that pong messages are silently ignored."""
        msg = {"channel": "pong"}

        with patch.object(self.tracker, '_handle_user_fills') as mock_handler:
            self.tracker._handle_message(msg)
            mock_handler.assert_not_called()

    def test_handle_message_ignores_unknown_channel(self):
        """Test that unknown channels are ignored."""
        msg = {"channel": "unknown_channel", "data": {}}

        with patch.object(self.tracker, '_handle_user_fills') as mock_handler:
            self.tracker._handle_message(msg)
            mock_handler.assert_not_called()

    # ==================== Fill Processing ====================

    def test_process_fill_unsupported_coin(self):
        """Test that unsupported coins are silently skipped."""
        fill = self._make_fill(coin="UNKNOWN_COIN")
        # Should not raise
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        # No order processed
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_no_synthetic_hotkey(self):
        """Test that fills for unknown HL addresses are skipped."""
        self.entity_client.get_synthetic_hotkey_for_hl_address.return_value = None

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_no_subaccount_info(self):
        """Test that fills for subaccounts with no info are skipped."""
        self.entity_client.get_synthetic_hotkey_for_hl_address.return_value = "entity_alpha_0"
        self.entity_client.get_subaccount_info_for_synthetic.return_value = None

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_zero_account_size(self):
        """Test that fills with zero account size are skipped."""
        self.entity_client.get_synthetic_hotkey_for_hl_address.return_value = "entity_alpha_0"
        self.entity_client.get_subaccount_info_for_synthetic.return_value = {
            "account_size": 0, "status": "active"
        }

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_rate_limited(self):
        """Test that rate-limited fills are skipped."""
        mock_result = self._setup_successful_fill_mocks()
        self.rate_limiter.is_allowed.return_value = (False, 5.0)

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_eliminated_miner(self):
        """Test that fills for eliminated miners are skipped."""
        mock_result = self._setup_successful_fill_mocks()
        self.elimination_client.get_elimination_local_cache.return_value = {"reason": "mdd"}

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_invalid_hotkey(self):
        """Test that fills with invalid hotkey validation are skipped."""
        mock_result = self._setup_successful_fill_mocks()
        self.entity_client.validate_hotkey_for_orders.return_value = {
            "is_valid": False, "error_message": "not active"
        }

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_market_closed(self):
        """Test that fills are skipped when market is closed."""
        mock_result = self._setup_successful_fill_mocks()
        self.price_fetcher_client.is_market_open.return_value = False

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_zero_size(self):
        """Test that fills with zero size are skipped."""
        mock_result = self._setup_successful_fill_mocks()

        fill = self._make_fill(sz="0")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_zero_price(self):
        """Test that fills with zero price are skipped."""
        mock_result = self._setup_successful_fill_mocks()

        fill = self._make_fill(px="0")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_process_fill_buy_side_maps_to_long(self, mock_order_processor):
        """Test that buy-side fills are converted to LONG orders."""
        mock_result = self._setup_successful_fill_mocks()
        mock_order_processor.process_order.return_value = mock_result

        fill = self._make_fill(side="B")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        call_args = mock_order_processor.process_order.call_args
        signal = call_args.kwargs['signal']
        self.assertEqual(signal['order_type'], 'LONG')

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_process_fill_sell_side_maps_to_short(self, mock_order_processor):
        """Test that sell-side fills are converted to SHORT orders."""
        mock_result = self._setup_successful_fill_mocks()
        mock_order_processor.process_order.return_value = mock_result

        fill = self._make_fill(side="A")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        call_args = mock_order_processor.process_order.call_args
        signal = call_args.kwargs['signal']
        self.assertEqual(signal['order_type'], 'SHORT')

    def test_process_fill_unknown_side(self):
        """Test that fills with unknown side are skipped."""
        mock_result = self._setup_successful_fill_mocks()

        fill = self._make_fill(side="X")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)
        self.assertEqual(self.tracker._fills_processed, 0)

    # ==================== USDC Balance Check ====================

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_process_fill_sufficient_balance_proceeds(self, mock_order_processor):
        """Test that fills proceed when USDC balance meets minimum."""
        mock_result = self._setup_successful_fill_mocks()
        mock_order_processor.process_order.return_value = mock_result

        fill = self._make_fill()
        with patch.object(self.tracker, '_get_hl_usdc_balance', return_value=5000.0):
            self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        self.assertEqual(self.tracker._fills_processed, 1)

    def test_process_fill_insufficient_balance_rejected(self):
        """Test that fills are rejected when USDC balance is below minimum."""
        self._setup_successful_fill_mocks()

        fill = self._make_fill()
        with patch.object(self.tracker, '_get_hl_usdc_balance', return_value=500.0):
            self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_exactly_minimum_balance_proceeds(self):
        """Test that fills proceed when USDC balance is exactly the minimum."""
        mock_result = self._setup_successful_fill_mocks()

        fill = self._make_fill()
        with patch.object(self.tracker, '_get_hl_usdc_balance', return_value=float(ValiConfig.HL_MIN_USDC_BALANCE)), \
             patch('entity_management.hyperliquid_tracker.OrderProcessor') as mock_op:
            mock_op.process_order.return_value = mock_result
            self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        self.assertEqual(self.tracker._fills_processed, 1)

    def test_process_fill_balance_query_fails_rejected(self):
        """Test that fills are rejected when USDC balance query returns None."""
        self._setup_successful_fill_mocks()

        fill = self._make_fill()
        with patch.object(self.tracker, '_get_hl_usdc_balance', return_value=None):
            self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        self.assertEqual(self.tracker._fills_processed, 0)

    def test_process_fill_zero_balance_rejected(self):
        """Test that fills are rejected when USDC balance is zero."""
        self._setup_successful_fill_mocks()

        fill = self._make_fill()
        with patch.object(self.tracker, '_get_hl_usdc_balance', return_value=0.0):
            self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        self.assertEqual(self.tracker._fills_processed, 0)

    @patch('entity_management.hyperliquid_tracker.requests')
    def test_get_hl_usdc_balance_success(self, mock_requests):
        """Test _get_hl_usdc_balance returns withdrawable amount on success."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"withdrawable": "2500.50"}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        balance = self.tracker._get_hl_usdc_balance(VALID_HL_ADDRESS)

        self.assertAlmostEqual(balance, 2500.50)
        mock_requests.post.assert_called_once_with(
            ValiConfig.HL_MAINNET_INFO,
            json={"type": "clearinghouseState", "user": VALID_HL_ADDRESS},
            timeout=5,
        )

    @patch('entity_management.hyperliquid_tracker.requests')
    def test_get_hl_usdc_balance_api_error(self, mock_requests):
        """Test _get_hl_usdc_balance returns None on API error."""
        mock_requests.post.side_effect = Exception("connection timeout")

        balance = self.tracker._get_hl_usdc_balance(VALID_HL_ADDRESS)

        self.assertIsNone(balance)

    @patch('entity_management.hyperliquid_tracker.requests')
    def test_get_hl_usdc_balance_missing_withdrawable(self, mock_requests):
        """Test _get_hl_usdc_balance returns 0 when withdrawable is missing."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        balance = self.tracker._get_hl_usdc_balance(VALID_HL_ADDRESS)

        self.assertEqual(balance, 0.0)

    # ==================== Leverage Calculation ====================

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_leverage_calculation_basic(self, mock_order_processor):
        """Test leverage = (fill_sz * fill_px) / account_size."""
        account_size = 100_000
        mock_result = self._setup_successful_fill_mocks(account_size=account_size)
        mock_order_processor.process_order.return_value = mock_result

        # fill_sz=1.0, fill_px=50000 => raw leverage = 50000/100000 = 0.5
        fill = self._make_fill(sz="1.0", px="50000.0")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        call_args = mock_order_processor.process_order.call_args
        signal = call_args.kwargs['signal']
        self.assertAlmostEqual(signal['leverage'], 0.5, places=4)

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_leverage_clamped_to_min(self, mock_order_processor):
        """Test leverage is clamped to CRYPTO_MIN_LEVERAGE when raw value is too small."""
        account_size = 100_000
        mock_result = self._setup_successful_fill_mocks(account_size=account_size)
        mock_order_processor.process_order.return_value = mock_result

        # Tiny fill: raw leverage = (0.001 * 50) / 100000 = 0.0000005 < CRYPTO_MIN_LEVERAGE
        fill = self._make_fill(sz="0.001", px="50.0")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        call_args = mock_order_processor.process_order.call_args
        signal = call_args.kwargs['signal']
        self.assertEqual(signal['leverage'], ValiConfig.CRYPTO_MIN_LEVERAGE)

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_leverage_clamped_to_max(self, mock_order_processor):
        """Test leverage is clamped to CRYPTO_MAX_LEVERAGE when raw value is too large."""
        account_size = 10_000
        mock_result = self._setup_successful_fill_mocks(account_size=account_size)
        mock_order_processor.process_order.return_value = mock_result

        # Large fill: raw leverage = (10 * 50000) / 10000 = 50 > CRYPTO_MAX_LEVERAGE
        fill = self._make_fill(sz="10.0", px="50000.0")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        call_args = mock_order_processor.process_order.call_args
        signal = call_args.kwargs['signal']
        self.assertEqual(signal['leverage'], ValiConfig.CRYPTO_MAX_LEVERAGE)

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_process_fill_signal_structure(self, mock_order_processor):
        """Test the full signal structure passed to OrderProcessor."""
        mock_result = self._setup_successful_fill_mocks(account_size=100_000)
        mock_order_processor.process_order.return_value = mock_result

        fill = self._make_fill(coin="ETH", side="B", sz="2.0", px="3000.0")
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        call_args = mock_order_processor.process_order.call_args
        signal = call_args.kwargs['signal']

        self.assertEqual(signal['order_type'], 'LONG')
        self.assertEqual(signal['trade_pair'], {'trade_pair_id': 'ETHUSD'})
        self.assertEqual(signal['execution_type'], 'MARKET')
        # leverage = (2.0 * 3000.0) / 100000 = 0.06
        self.assertAlmostEqual(signal['leverage'], 0.06, places=4)

        # Verify miner_hotkey
        self.assertEqual(call_args.kwargs['miner_hotkey'], 'entity_alpha_0')
        self.assertEqual(call_args.kwargs['miner_repo_version'], 'hl_tracker')

    @patch('entity_management.hyperliquid_tracker.OrderProcessor')
    def test_process_fill_increments_counter_and_tracks_uuid(self, mock_order_processor):
        """Test that successful fill processing increments counter and tracks UUID."""
        mock_result = self._setup_successful_fill_mocks()
        mock_result.should_track_uuid = True
        mock_order_processor.process_order.return_value = mock_result

        fill = self._make_fill()
        self.tracker._process_fill(VALID_HL_ADDRESS, fill)

        self.assertEqual(self.tracker._fills_processed, 1)
        self.assertIsNotNone(self.tracker._last_fill_time)
        self.uuid_tracker.add.assert_called_once()

    # ==================== Status ====================

    def test_get_status_initial(self):
        """Test initial tracker status."""
        status = self.tracker.get_status()
        self.assertFalse(status['connected'])
        self.assertEqual(status['subscribed_addresses'], 0)
        self.assertEqual(status['fills_processed'], 0)
        self.assertIsNone(status['last_fill_time'])

    # ==================== Multiple Fills ====================

    def test_handle_user_fills_multiple(self):
        """Test processing multiple fills in a single message."""
        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": False,
                "user": VALID_HL_ADDRESS,
                "fills": [
                    {"hash": "hash_a", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"},
                    {"hash": "hash_b", "coin": "ETH", "side": "A", "sz": "10", "px": "3000"},
                    {"hash": "hash_c", "coin": "SOL", "side": "B", "sz": "100", "px": "200"},
                ]
            }
        }

        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            self.assertEqual(mock_process.call_count, 3)

    def test_handle_user_fills_empty_fills(self):
        """Test that messages with empty fills list are handled gracefully."""
        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": False,
                "user": VALID_HL_ADDRESS,
                "fills": []
            }
        }

        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            mock_process.assert_not_called()

    def test_handle_user_fills_no_user(self):
        """Test that messages without user field are handled gracefully."""
        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": False,
                "fills": [{"hash": "h1", "coin": "BTC"}]
            }
        }

        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            mock_process.assert_not_called()

    def test_handle_user_fills_skip_no_hash(self):
        """Test that fills without hash are skipped."""
        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": False,
                "user": VALID_HL_ADDRESS,
                "fills": [{"coin": "BTC", "side": "B", "sz": "1", "px": "50000"}]  # No hash
            }
        }

        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            mock_process.assert_not_called()

    def test_handle_user_fills_uses_tid_as_fallback_hash(self):
        """Test that tid is used as fill hash when hash field is missing."""
        msg = {
            "channel": "userFills",
            "data": {
                "isSnapshot": False,
                "user": VALID_HL_ADDRESS,
                "fills": [{"tid": "tid_hash_1", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}]
            }
        }

        with patch.object(self.tracker, '_process_fill') as mock_process:
            self.tracker._handle_user_fills(msg)
            mock_process.assert_called_once()

        self.assertIn("tid_hash_1", self.tracker._processed_hashes)


if __name__ == '__main__':
    unittest.main()
