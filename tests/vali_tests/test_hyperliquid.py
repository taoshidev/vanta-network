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
import asyncio
import re
import time
import unittest
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestHyperliquidTrackerBackupPoll(TestBase):
    """
    Tests for the backup REST poll feature: _make_proxied_session, _fetch_fills_by_time,
    _backup_poll_cycle, get_status backup_poll stats, and ValiConfig backup constants.
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

    # ==================== Init State ====================

    def test_backup_poll_init_state(self):
        """Test that backup poll state is properly initialized."""
        self.assertEqual(self.tracker._last_poll_time_ms, {})
        self.assertIsNone(self.tracker._backup_poll_task)
        self.assertEqual(self.tracker._backup_fills_caught, 0)
        self.assertEqual(self.tracker._backup_polls_total, 0)
        self.assertEqual(self.tracker._proxy_index_rest, 0)

    # ==================== ValiConfig Backup Constants ====================

    def test_backup_config_constants_exist(self):
        """Test that backup poll config constants are defined in ValiConfig."""
        self.assertEqual(ValiConfig.HL_BACKUP_POLL_INTERVAL_S, 30.0)
        self.assertEqual(ValiConfig.HL_BACKUP_POLL_LOOKBACK_MS, 120_000)
        self.assertEqual(ValiConfig.HL_BACKUP_POLL_RATE_BUDGET, 600)

    # ==================== get_status Backup Section ====================

    def test_get_status_includes_backup_poll(self):
        """Test that get_status includes backup_poll section with correct initial values."""
        status = self.tracker.get_status()
        self.assertIn('backup_poll', status)
        bp = status['backup_poll']
        self.assertEqual(bp['fills_caught'], 0)
        self.assertEqual(bp['total_polls'], 0)
        self.assertEqual(bp['tracked_addresses'], 0)

    def test_get_status_backup_poll_reflects_state(self):
        """Test that get_status backup_poll section reflects updated state."""
        self.tracker._backup_fills_caught = 5
        self.tracker._backup_polls_total = 42
        self.tracker._last_poll_time_ms = {"0xaaa": 1000, "0xbbb": 2000}

        status = self.tracker.get_status()
        bp = status['backup_poll']
        self.assertEqual(bp['fills_caught'], 5)
        self.assertEqual(bp['total_polls'], 42)
        self.assertEqual(bp['tracked_addresses'], 2)

    # ==================== _make_proxied_session ====================

    def test_make_proxied_session_no_proxy_returns_plain(self):
        """Test that _make_proxied_session returns a plain session when no proxy configured."""
        self.tracker._proxy_base_url = None
        session = self.tracker._make_proxied_session()
        self.assertIsNotNone(session)
        self.assertEqual(session.proxies, {})
        session.close()

    def test_make_proxied_session_no_ports_returns_plain(self):
        """Test that _make_proxied_session returns a plain session when proxy is set but no ports."""
        self.tracker._proxy_base_url = "socks5://user:pass@host"
        self.tracker._available_ports = []
        self.tracker._shards = {}
        session = self.tracker._make_proxied_session()
        self.assertEqual(session.proxies, {})
        session.close()

    def test_make_proxied_session_with_available_ports(self):
        """Test that _make_proxied_session sets proxy from available ports."""
        self.tracker._proxy_base_url = "socks5://user:pass@host"
        self.tracker._available_ports = [10001, 10002]
        self.tracker._shards = {}

        session = self.tracker._make_proxied_session()
        # Port should be 10001 (first in sorted set, index 0 % 2 = 0)
        expected_url = "socks5://user:pass@host:10001"
        self.assertEqual(session.proxies.get("http"), expected_url)
        self.assertEqual(session.proxies.get("https"), expected_url)
        session.close()

    def test_make_proxied_session_round_robin(self):
        """Test that successive calls rotate through ports."""
        self.tracker._proxy_base_url = "socks5://user:pass@host"
        self.tracker._available_ports = [10001, 10002, 10003]
        self.tracker._shards = {}

        ports_used = []
        for _ in range(6):
            session = self.tracker._make_proxied_session()
            proxy = session.proxies.get("http", "")
            port = int(proxy.rsplit(":", 1)[-1]) if proxy else None
            ports_used.append(port)
            session.close()

        # Should round-robin: 10001, 10002, 10003, 10001, 10002, 10003
        self.assertEqual(ports_used, [10001, 10002, 10003, 10001, 10002, 10003])

    def test_make_proxied_session_includes_healthy_shard_ports(self):
        """Test that ports from healthy shards are included in the pool."""
        self.tracker._proxy_base_url = "socks5://user:pass@host"
        self.tracker._available_ports = [10001]

        # Create a mock shard with a port
        mock_shard = MagicMock()
        mock_shard.healthy = True
        mock_shard.port = 10005
        self.tracker._shards = {0: mock_shard}

        session = self.tracker._make_proxied_session()
        proxy = session.proxies.get("http", "")
        port = int(proxy.rsplit(":", 1)[-1])
        # Sorted set of [10001, 10005], first call -> 10001
        self.assertEqual(port, 10001)
        session.close()

        session2 = self.tracker._make_proxied_session()
        proxy2 = session2.proxies.get("http", "")
        port2 = int(proxy2.rsplit(":", 1)[-1])
        # Second call -> 10005
        self.assertEqual(port2, 10005)
        session2.close()

    def test_make_proxied_session_skips_unhealthy_shard_ports(self):
        """Test that unhealthy shard ports are NOT included in the pool."""
        self.tracker._proxy_base_url = "socks5://user:pass@host"
        self.tracker._available_ports = [10001]

        mock_shard = MagicMock()
        mock_shard.healthy = False
        mock_shard.port = 10005
        self.tracker._shards = {0: mock_shard}

        session = self.tracker._make_proxied_session()
        proxy = session.proxies.get("http", "")
        port = int(proxy.rsplit(":", 1)[-1])
        # Only 10001 in pool (unhealthy shard excluded)
        self.assertEqual(port, 10001)
        session.close()

    # ==================== _fetch_fills_by_time ====================

    @patch('entity_management.hyperliquid_tracker.requests.Session')
    def test_fetch_fills_by_time_success(self, mock_session_cls):
        """Test successful REST fill fetch returns list of fills."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"hash": "h1", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"},
            {"hash": "h2", "coin": "ETH", "side": "A", "sz": "2", "px": "3000"},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.tracker._fetch_fills_by_time(VALID_HL_ADDRESS, 1000)
            )
        finally:
            loop.close()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["hash"], "h1")
        self.assertEqual(self.tracker._backup_polls_total, 1)

    @patch('entity_management.hyperliquid_tracker.requests.Session')
    def test_fetch_fills_by_time_non_list_response(self, mock_session_cls):
        """Test that non-list response is converted to empty list."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "something"}
        mock_resp.raise_for_status = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.tracker._fetch_fills_by_time(VALID_HL_ADDRESS, 1000)
            )
        finally:
            loop.close()

        self.assertEqual(result, [])
        self.assertEqual(self.tracker._backup_polls_total, 1)

    @patch('entity_management.hyperliquid_tracker.requests.Session')
    def test_fetch_fills_by_time_exception_returns_none(self, mock_session_cls):
        """Test that REST errors return None."""
        mock_session = MagicMock()
        mock_session.post.side_effect = Exception("connection timeout")
        mock_session_cls.return_value = mock_session

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.tracker._fetch_fills_by_time(VALID_HL_ADDRESS, 1000)
            )
        finally:
            loop.close()

        self.assertIsNone(result)
        self.assertEqual(self.tracker._backup_polls_total, 1)

    # ==================== _backup_poll_cycle ====================

    def test_backup_poll_cycle_processes_missed_fill(self):
        """Test that backup poll catches a fill that WS missed (not in _processed_hashes)."""
        # Set up tracker state: one tracked address
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        # Stop after first iteration
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        missed_fill = {"hash": "missed_hash_1", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[missed_fill]) as mock_fetch, \
             patch.object(self.tracker, '_process_fill') as mock_process, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

            mock_process.assert_called_once_with(VALID_HL_ADDRESS, missed_fill)
            self.assertEqual(self.tracker._backup_fills_caught, 1)
            self.assertIn("missed_hash_1", self.tracker._processed_hashes)

    def test_backup_poll_cycle_skips_already_processed(self):
        """Test that backup poll skips fills already in _processed_hashes."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        # Pre-record the hash (simulating WS already processed it)
        self.tracker._record_hash("existing_hash")

        fill = {"hash": "existing_hash", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[fill]), \
             patch.object(self.tracker, '_process_fill') as mock_process, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

            mock_process.assert_not_called()
            self.assertEqual(self.tracker._backup_fills_caught, 0)

    def test_backup_poll_cycle_skips_fill_without_hash(self):
        """Test that fills without hash or tid are skipped."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        fill_no_hash = {"coin": "BTC", "side": "B", "sz": "1", "px": "50000"}

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[fill_no_hash]), \
             patch.object(self.tracker, '_process_fill') as mock_process, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

            mock_process.assert_not_called()
            self.assertEqual(self.tracker._backup_fills_caught, 0)

    def test_backup_poll_cycle_uses_tid_as_fallback(self):
        """Test that backup poll uses tid as hash when hash field is missing."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        fill_with_tid = {"tid": "tid_123", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[fill_with_tid]), \
             patch.object(self.tracker, '_process_fill') as mock_process, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

            mock_process.assert_called_once_with(VALID_HL_ADDRESS, fill_with_tid)
            self.assertIn("tid_123", self.tracker._processed_hashes)
            self.assertEqual(self.tracker._backup_fills_caught, 1)

    def test_backup_poll_cycle_advances_watermark_on_success(self):
        """Test that watermark advances after successful poll."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[]), \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

        # Watermark should be set for the address
        self.assertIn(VALID_HL_ADDRESS, self.tracker._last_poll_time_ms)
        self.assertGreater(self.tracker._last_poll_time_ms[VALID_HL_ADDRESS], 0)

    def test_backup_poll_cycle_no_watermark_advance_on_failure(self):
        """Test that watermark does NOT advance when fetch returns None."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=None), \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

        # Watermark should NOT be set
        self.assertNotIn(VALID_HL_ADDRESS, self.tracker._last_poll_time_ms)

    def test_backup_poll_cycle_cleans_stale_watermarks(self):
        """Test that watermarks for no-longer-tracked addresses are cleaned up."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        # Pre-set a watermark for an address no longer tracked
        stale_addr = "0x" + "dead" * 10
        self.tracker._last_poll_time_ms = {stale_addr: 999}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[]), \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

        # Stale address watermark should be removed
        self.assertNotIn(stale_addr, self.tracker._last_poll_time_ms)
        # Active address should have a watermark
        self.assertIn(VALID_HL_ADDRESS, self.tracker._last_poll_time_ms)

    def test_backup_poll_cycle_skips_when_no_tracked_addresses(self):
        """Test that backup poll cycle does nothing when no addresses are tracked."""
        self.tracker._address_to_shard = {}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, True])

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock) as mock_fetch, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

            mock_fetch.assert_not_called()

    def test_backup_poll_cycle_handles_process_fill_exception(self):
        """Test that exception in _process_fill doesn't crash the poll cycle."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        fill = {"hash": "crash_hash", "coin": "BTC", "side": "B", "sz": "1", "px": "50000"}

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[fill]), \
             patch.object(self.tracker, '_process_fill', side_effect=Exception("boom")), \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                # Should not raise
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

        # Fill was not successfully processed, so counter should NOT increment
        self.assertEqual(self.tracker._backup_fills_caught, 0)
        # But hash should still be recorded for dedup
        self.assertIn("crash_hash", self.tracker._processed_hashes)

    def test_backup_poll_cycle_uses_lookback_for_new_address(self):
        """Test that first poll for an address uses HL_BACKUP_POLL_LOOKBACK_MS."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[]) as mock_fetch, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

            # Check that start_time_ms was approximately now - lookback
            call_args = mock_fetch.call_args
            start_ms = call_args[0][1]  # second positional arg
            expected_approx = int(time.time() * 1000) - ValiConfig.HL_BACKUP_POLL_LOOKBACK_MS
            # Allow 5 seconds of tolerance
            self.assertAlmostEqual(start_ms, expected_approx, delta=5000)

    def test_backup_poll_cycle_uses_existing_watermark(self):
        """Test that subsequent polls for an address use the stored watermark."""
        self.tracker._address_to_shard = {VALID_HL_ADDRESS: 0}
        self.tracker._last_poll_time_ms = {VALID_HL_ADDRESS: 1234567890000}
        self.tracker._stop_event = MagicMock()
        self.tracker._stop_event.is_set = MagicMock(side_effect=[False, False, True])

        with patch.object(self.tracker, '_fetch_fills_by_time', new_callable=AsyncMock,
                          return_value=[]) as mock_fetch, \
             patch('asyncio.sleep', new_callable=AsyncMock):

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.tracker._backup_poll_cycle())
            finally:
                loop.close()

            call_args = mock_fetch.call_args
            start_ms = call_args[0][1]
            self.assertEqual(start_ms, 1234567890000)

    # ==================== get_account_state uses proxied session ====================

    @patch('entity_management.hyperliquid_tracker.requests.Session')
    def test_fetch_hl_account_state_uses_proxied_session(self, mock_session_cls):
        """Test that _fetch_hl_account_state creates a proxied session and closes it."""
        mock_session = MagicMock()
        perp_resp = MagicMock()
        perp_resp.json.return_value = {
            "crossMarginSummary": {"accountValue": "10000"},
            "assetPositions": []
        }
        spot_resp = MagicMock()
        spot_resp.json.return_value = {"balances": []}
        mids_resp = MagicMock()
        mids_resp.json.return_value = {}

        mock_session.post.side_effect = [perp_resp, spot_resp, mids_resp]

        with patch.object(self.tracker, '_make_proxied_session', return_value=mock_session):
            result = self.tracker._fetch_hl_account_state(VALID_HL_ADDRESS)

        # Session should be closed after use
        mock_session.close.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(result['total_portfolio_value'], 10000.0)


class TestProxyPortHealth(TestBase):
    """Tests for proxy port health management with exponential backoff probing."""

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

    def _setup_proxy_ports(self, ports):
        """Configure tracker with proxy ports and initialize health records."""
        from entity_management.hyperliquid_tracker import _PortHealthRecord
        self.tracker._proxy_base_url = "socks5://user:pass@proxy.example.com"
        self.tracker._available_ports = list(ports)
        self.tracker._port_health = {p: _PortHealthRecord(p) for p in ports}

    def test_health_records_initialized_on_proxy_config_load(self):
        """Health records should be created for all ports during _load_proxy_config."""
        from entity_management.hyperliquid_tracker import _PortHealthRecord
        with patch('entity_management.hyperliquid_tracker.ValiUtils') as mock_vali:
            mock_vali.get_secrets.return_value = {
                ValiConfig.HL_PROXY_SECRET_KEY: "socks5://user:pass@host",
                ValiConfig.HL_PROXY_PORTS_SECRET_KEY: "10001-10003",
            }
            with patch('entity_management.hyperliquid_tracker.SocksProxy', create=True):
                self.tracker._load_proxy_config()

        self.assertEqual(len(self.tracker._port_health), 3)
        for port in [10001, 10002, 10003]:
            self.assertIn(port, self.tracker._port_health)
            rec = self.tracker._port_health[port]
            self.assertTrue(rec.healthy)
            self.assertEqual(rec.rest_consecutive_failures, 0)
            self.assertEqual(rec.consecutive_probe_failures, 0)

    def test_unhealthy_ports_property_backward_compat(self):
        """_unhealthy_ports property should return set of unhealthy port numbers."""
        self._setup_proxy_ports([10001, 10002, 10003])

        # All healthy initially
        self.assertEqual(self.tracker._unhealthy_ports, set())

        # Mark one unhealthy
        self.tracker._port_health[10002].mark_unhealthy()
        self.assertEqual(self.tracker._unhealthy_ports, {10002})

        # Mark another unhealthy
        self.tracker._port_health[10001].mark_unhealthy()
        self.assertEqual(self.tracker._unhealthy_ports, {10001, 10002})

        # Recover one
        self.tracker._port_health[10002].mark_healthy()
        self.assertEqual(self.tracker._unhealthy_ports, {10001})

    def test_exponential_backoff_schedule(self):
        """Cooldown should follow 300, 600, 1200, 2400, 3600, 3600 (capped)."""
        from entity_management.hyperliquid_tracker import _PortHealthRecord
        rec = _PortHealthRecord(10001)
        rec.mark_unhealthy()

        expected_cooldowns = [300.0, 600.0, 1200.0, 2400.0, 3600.0, 3600.0]
        for i, expected in enumerate(expected_cooldowns):
            rec.consecutive_probe_failures = i
            self.assertEqual(rec.cooldown_seconds(), expected,
                             f"Cooldown at attempt {i} should be {expected}")

    def test_probe_respects_cooldown_timing(self):
        """is_probe_due should return False during cooldown, True after."""
        from entity_management.hyperliquid_tracker import _PortHealthRecord
        rec = _PortHealthRecord(10001)

        # Healthy port — never due
        self.assertFalse(rec.is_probe_due())

        # Mark unhealthy just now
        rec.mark_unhealthy()
        # With 300s cooldown and unhealthy_since = now, should NOT be due
        self.assertFalse(rec.is_probe_due())

        # Simulate unhealthy_since was 301s ago
        rec.unhealthy_since = time.time() - 301
        self.assertTrue(rec.is_probe_due())

    def test_probe_recovers_port_on_success(self):
        """Successful probe should mark port healthy and return it to available pool."""
        self._setup_proxy_ports([10001, 10002])
        self.tracker._port_health[10002].mark_unhealthy()
        self.tracker._port_health[10002].unhealthy_since = time.time() - 400  # past cooldown
        self.tracker._available_ports.remove(10002)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch('entity_management.hyperliquid_tracker.requests.Session') as mock_session_cls:
            mock_session = MagicMock()
            mock_session.post.return_value = mock_resp
            mock_session_cls.return_value = mock_session

            self.tracker._probe_unhealthy_ports()

        self.assertTrue(self.tracker._port_health[10002].healthy)
        self.assertIn(10002, self.tracker._available_ports)

    def test_probe_increments_failure_on_error(self):
        """Failed probe should increment consecutive_probe_failures."""
        self._setup_proxy_ports([10001])
        self.tracker._port_health[10001].mark_unhealthy()
        self.tracker._port_health[10001].unhealthy_since = time.time() - 400  # past cooldown

        with patch('entity_management.hyperliquid_tracker.requests.Session') as mock_session_cls:
            mock_session = MagicMock()
            mock_session.post.side_effect = Exception("connection refused")
            mock_session_cls.return_value = mock_session

            self.tracker._probe_unhealthy_ports()

        self.assertFalse(self.tracker._port_health[10001].healthy)
        self.assertEqual(self.tracker._port_health[10001].consecutive_probe_failures, 1)

    def test_make_proxied_session_excludes_unhealthy_ports(self):
        """_make_proxied_session should skip unhealthy ports in round-robin."""
        self._setup_proxy_ports([10001, 10002, 10003])
        self.tracker._port_health[10002].mark_unhealthy()

        # Collect ports used across multiple calls
        used_ports = set()
        for _ in range(10):
            session = self.tracker._make_proxied_session()
            port = getattr(session, '_hl_proxy_port', None)
            if port is not None:
                used_ports.add(port)

        self.assertNotIn(10002, used_ports)
        self.assertTrue(used_ports.issubset({10001, 10003}))

    def test_make_proxied_session_all_unhealthy_falls_back_to_direct(self):
        """When all ports are unhealthy, fall back to direct (no proxy)."""
        self._setup_proxy_ports([10001, 10002])
        self.tracker._port_health[10001].mark_unhealthy()
        self.tracker._port_health[10002].mark_unhealthy()

        session = self.tracker._make_proxied_session()
        self.assertIsNone(getattr(session, '_hl_proxy_port', None))
        self.assertEqual(session.proxies, {})

    def test_rest_failure_tracking_triggers_unhealthy(self):
        """Port should be marked unhealthy after HL_PORT_REST_FAILURE_THRESHOLD consecutive failures."""
        self._setup_proxy_ports([10001])
        threshold = ValiConfig.HL_PORT_REST_FAILURE_THRESHOLD

        for i in range(threshold - 1):
            self.tracker._report_rest_proxy_failure(10001)
            self.assertTrue(self.tracker._port_health[10001].healthy,
                            f"Should still be healthy after {i+1} failures")

        self.tracker._report_rest_proxy_failure(10001)
        self.assertFalse(self.tracker._port_health[10001].healthy)

    def test_rest_success_resets_failure_counter(self):
        """Successful REST call should reset the failure counter."""
        self._setup_proxy_ports([10001])

        # Accumulate failures just below threshold
        for _ in range(ValiConfig.HL_PORT_REST_FAILURE_THRESHOLD - 1):
            self.tracker._report_rest_proxy_failure(10001)
        self.assertEqual(
            self.tracker._port_health[10001].rest_consecutive_failures,
            ValiConfig.HL_PORT_REST_FAILURE_THRESHOLD - 1
        )

        # Success resets counter
        self.tracker._report_rest_proxy_success(10001)
        self.assertEqual(self.tracker._port_health[10001].rest_consecutive_failures, 0)
        self.assertTrue(self.tracker._port_health[10001].healthy)

    def test_rest_failure_with_none_port_is_noop(self):
        """_report_rest_proxy_failure with None port should be a no-op."""
        self._setup_proxy_ports([10001])
        # Should not raise
        self.tracker._report_rest_proxy_failure(None)
        self.tracker._report_rest_proxy_success(None)
        self.assertTrue(self.tracker._port_health[10001].healthy)

    def test_get_status_includes_port_health(self):
        """get_status should include port_health list with per-port details."""
        self._setup_proxy_ports([10001, 10002])
        self.tracker._port_health[10002].mark_unhealthy()
        self.tracker._port_health[10002].rest_consecutive_failures = 2

        status = self.tracker.get_status()
        self.assertIn("port_health", status)
        self.assertEqual(len(status["port_health"]), 2)

        # Find the unhealthy port entry
        unhealthy_entry = [e for e in status["port_health"] if e["port"] == 10002][0]
        self.assertFalse(unhealthy_entry["healthy"])
        self.assertEqual(unhealthy_entry["rest_failures"], 2)
        self.assertIn("next_probe_in_s", unhealthy_entry)

        # Find the healthy port entry
        healthy_entry = [e for e in status["port_health"] if e["port"] == 10001][0]
        self.assertTrue(healthy_entry["healthy"])
        self.assertNotIn("next_probe_in_s", healthy_entry)

    def test_mark_unhealthy_idempotent(self):
        """Calling mark_unhealthy multiple times shouldn't reset unhealthy_since."""
        from entity_management.hyperliquid_tracker import _PortHealthRecord
        rec = _PortHealthRecord(10001)
        rec.mark_unhealthy()
        first_timestamp = rec.unhealthy_since
        self.assertIsNotNone(first_timestamp)

        # Small delay to ensure time difference
        rec.consecutive_probe_failures = 3
        rec.mark_unhealthy()
        # unhealthy_since should NOT be reset, probe failures should NOT be reset
        self.assertEqual(rec.unhealthy_since, first_timestamp)
        self.assertEqual(rec.consecutive_probe_failures, 3)

    def test_mark_healthy_resets_all_state(self):
        """mark_healthy should reset all failure counters and timestamps."""
        from entity_management.hyperliquid_tracker import _PortHealthRecord
        rec = _PortHealthRecord(10001)
        rec.mark_unhealthy()
        rec.consecutive_probe_failures = 5
        rec.rest_consecutive_failures = 3
        rec.last_probe_time = time.time()

        rec.mark_healthy()
        self.assertTrue(rec.healthy)
        self.assertIsNone(rec.unhealthy_since)
        self.assertIsNone(rec.last_probe_time)
        self.assertEqual(rec.consecutive_probe_failures, 0)
        self.assertEqual(rec.rest_consecutive_failures, 0)

    def test_teardown_empty_shards_uses_unhealthy_ports_property(self):
        """_teardown_empty_shards should not return unhealthy ports to available pool."""
        self._setup_proxy_ports([10001, 10002])

        # Simulate: port 10001 was consumed by a shard, so remove from available
        self.tracker._available_ports.remove(10001)

        # Create a shard with an unhealthy port and no addresses
        shard = MagicMock()
        shard.addresses = set()
        shard.port = 10001
        shard.task = None
        self.tracker._shards = {0: shard}

        # Mark port 10001 unhealthy
        self.tracker._port_health[10001].mark_unhealthy()

        self.tracker._teardown_empty_shards()

        # Port should NOT be returned to available pool since it's unhealthy
        self.assertNotIn(10001, self.tracker._available_ports)


if __name__ == '__main__':
    unittest.main()
