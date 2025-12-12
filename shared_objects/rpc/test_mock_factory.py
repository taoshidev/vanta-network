# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Test Mock Factory - Centralized creation of mock configs and wallets for unit testing.

This module provides utilities for creating minimal mock objects needed by RPC servers
during unit tests. Instead of having mock creation logic scattered across ServerOrchestrator
or individual servers, all mock creation is centralized here.

Usage in server __init__:
    from shared_objects.rpc.test_mock_factory import TestMockFactory

    def __init__(self, config=None, wallet=None, running_unit_tests=False, **kwargs):
        # Create mocks if running tests and parameters not provided
        if running_unit_tests:
            config = TestMockFactory.create_mock_config_if_needed(config)
            wallet = TestMockFactory.create_mock_wallet_if_needed(wallet)

        # Use config and wallet normally
        self.config = config
        self.wallet = wallet
"""

from types import SimpleNamespace
from typing import Optional, Any


class TestMockFactory:
    """
    Factory for creating mock objects used in unit testing.

    Provides standardized mock configs, wallets, and other test objects
    needed by RPC servers when running_unit_tests=True.
    """

    @staticmethod
    def create_mock_config(
        netuid: int = 116,
        network: str = "test",
        **additional_attrs
    ) -> SimpleNamespace:
        """
        Create a minimal mock config for unit testing.

        Args:
            netuid: Network UID (default: 116 for testnet)
            network: Network name (default: "test")
            **additional_attrs: Additional attributes to add to the config

        Returns:
            SimpleNamespace: Mock config with required attributes

        Example:
            config = TestMockFactory.create_mock_config(
                netuid=116,
                slack_error_webhook_url="https://hooks.slack.com/test"
            )
        """
        config = SimpleNamespace(
            netuid=netuid,
            subtensor=SimpleNamespace(network=network)
        )

        # Add any additional attributes
        for key, value in additional_attrs.items():
            setattr(config, key, value)

        return config

    @staticmethod
    def create_mock_config_if_needed(
        config: Optional[Any],
        netuid: int = 116,
        network: str = "test",
        **additional_attrs
    ) -> Any:
        """
        Create a mock config only if config is None.

        Args:
            config: Existing config or None
            netuid: Network UID (default: 116 for testnet)
            network: Network name (default: "test")
            **additional_attrs: Additional attributes to add to the config

        Returns:
            The existing config if provided, or a new mock config

        Example:
            # Only creates mock if config is None
            config = TestMockFactory.create_mock_config_if_needed(
                config,
                netuid=116
            )
        """
        if config is None:
            return TestMockFactory.create_mock_config(netuid, network, **additional_attrs)
        return config

    @staticmethod
    def create_mock_wallet(
        hotkey: str = "test_hotkey_address",
        coldkey: str = "test_coldkey_address",
        name: str = "test_wallet"
    ) -> SimpleNamespace:
        """
        Create a minimal mock wallet for unit testing.

        Args:
            hotkey: Hotkey SS58 address (default: "test_hotkey_address")
            coldkey: Coldkey SS58 address (default: "test_coldkey_address")
            name: Wallet name (default: "test_wallet")

        Returns:
            SimpleNamespace: Mock wallet with hotkey and coldkey

        Example:
            wallet = TestMockFactory.create_mock_wallet(
                hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        """
        return SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address=hotkey),
            coldkey=SimpleNamespace(ss58_address=coldkey),
            name=name
        )

    @staticmethod
    def create_mock_wallet_if_needed(
        wallet: Optional[Any],
        hotkey: str = "test_hotkey_address",
        coldkey: str = "test_coldkey_address",
        name: str = "test_wallet"
    ) -> Any:
        """
        Create a mock wallet only if wallet is None.

        Args:
            wallet: Existing wallet or None
            hotkey: Hotkey SS58 address (default: "test_hotkey_address")
            coldkey: Coldkey SS58 address (default: "test_coldkey_address")
            name: Wallet name (default: "test_wallet")

        Returns:
            The existing wallet if provided, or a new mock wallet

        Example:
            # Only creates mock if wallet is None
            wallet = TestMockFactory.create_mock_wallet_if_needed(wallet)
        """
        if wallet is None:
            return TestMockFactory.create_mock_wallet(hotkey, coldkey, name)
        return wallet

    @staticmethod
    def create_mock_hotkey(hotkey: str = "test_validator_hotkey") -> str:
        """
        Create a mock hotkey string for unit testing.

        Args:
            hotkey: Hotkey string (default: "test_validator_hotkey")

        Returns:
            str: Mock hotkey

        Example:
            hotkey = TestMockFactory.create_mock_hotkey("my_test_hotkey")
        """
        return hotkey

    @staticmethod
    def create_mock_hotkey_if_needed(
        hotkey: Optional[str],
        default_hotkey: str = "test_validator_hotkey"
    ) -> str:
        """
        Create a mock hotkey only if hotkey is None.

        Args:
            hotkey: Existing hotkey or None
            default_hotkey: Default hotkey string (default: "test_validator_hotkey")

        Returns:
            The existing hotkey if provided, or a mock hotkey

        Example:
            # Only creates mock if hotkey is None
            hotkey = TestMockFactory.create_mock_hotkey_if_needed(hotkey)
        """
        if hotkey is None:
            return default_hotkey
        return hotkey
