# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ValidatorBroadcastBase - Shared base class for validator state broadcasting.

This base class provides common functionality for broadcasting state updates
between validators in the Vanta Network. It implements:
1. Sender verification (only mothership can broadcast)
2. Receiver verification (only accept broadcasts from mothership)
3. Threaded async broadcasting with logging
4. Axon selection based on stake/testnet mode

Usage:
    class MyManager(ValidatorBroadcastBase):
        def __init__(self, ...):
            super().__init__(
                running_unit_tests=running_unit_tests,
                is_mothership=is_mothership,
                vault_wallet=vault_wallet,
                metagraph_client=metagraph_client,
                is_testnet=is_testnet
            )

        def my_broadcast_method(self, data):
            def create_synapse():
                return template.protocol.MySynapse(data=data)

            self._broadcast_to_validators(
                synapse_factory=create_synapse,
                broadcast_name="MyUpdate"
            )
"""
import threading
from typing import Callable, Optional
import bittensor as bt

import template.protocol
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


class ValidatorBroadcastBase:
    """
    Base class for validator managers that need to broadcast state updates.

    Provides:
    - Sender verification (is_mothership check)
    - Receiver verification (sender_hotkey validation)
    - Async broadcast with threading
    - Common logging patterns
    """

    def __init__(
        self,
        *,
        running_unit_tests: bool = False,
        is_testnet: bool = False,
        config=None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize ValidatorBroadcastBase.

        Args:
            running_unit_tests: Whether running in test mode
            is_testnet: Whether running on testnet
            config: Bittensor config (used to get hotkey for axon filtering)
            connection_mode: RPC connection mode (for lazy client initialization)
        """
        self.running_unit_tests = running_unit_tests
        self.is_testnet = is_testnet
        self._config = config

        # Get hotkey for filtering out self from broadcasts and derive is_mothership
        if self.running_unit_tests:
            self._hotkey = None
            self.is_mothership = False
            self.wallet = None
        else:
            self.wallet = bt.wallet(config=config)
            self._hotkey = self.wallet.hotkey.ss58_address
            # Derive is_mothership using centralized utility
            from vali_objects.utils.vali_utils import ValiUtils
            self.is_mothership = ValiUtils.is_mothership_wallet(self.wallet)

        # Create metagraph client with connect_immediately=False to defer connection
        from shared_objects.rpc.metagraph_client import MetagraphClient
        self._metagraph_client = MetagraphClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        # Create SubtensorOpsClient for broadcasting (deferred connection)
        from shared_objects.subtensor_ops.subtensor_ops_client import SubtensorOpsClient
        self._subtensor_ops_client = SubtensorOpsClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False
        )

    # ==================== Sender Methods (Mothership) ====================

    def _broadcast_to_validators(
        self,
        synapse_factory: Callable[[], template.protocol.bt.Synapse],
        broadcast_name: str,
        context: Optional[dict] = None
    ) -> None:
        """
        Broadcast a synapse to other validators in a background thread.

        This method should only be called by the mothership validator.
        It spawns a thread that runs the broadcast via SubtensorOpsClient RPC.

        Args:
            synapse_factory: Function that creates the synapse to broadcast
            broadcast_name: Human-readable name for logging (e.g., "CollateralRecord")
            context: Optional dict with context info for logging (e.g., {"hotkey": "5..."})
        """
        if self.running_unit_tests:
            bt.logging.debug(f"[BROADCAST] Running unit tests, skipping {broadcast_name} broadcast")
            return

        if not self.is_mothership:
            bt.logging.debug(f"[BROADCAST] Not mothership, skipping {broadcast_name} broadcast")
            return

        def run_broadcast():
            try:
                self._do_broadcast_via_rpc(
                    synapse_factory=synapse_factory,
                    broadcast_name=broadcast_name,
                    context=context
                )
            except Exception as e:
                context_str = f" ({context})" if context else ""
                bt.logging.error(f"[BROADCAST] Failed to broadcast {broadcast_name}{context_str}: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())

        thread = threading.Thread(target=run_broadcast, daemon=True)
        thread.start()

    def _do_broadcast_via_rpc(
        self,
        synapse_factory: Callable[[], template.protocol.bt.Synapse],
        broadcast_name: str,
        context: Optional[dict] = None
    ) -> None:
        """
        Broadcast synapse to other validators via SubtensorOpsClient RPC.

        This method delegates the actual broadcast to SubtensorOpsManager which has
        access to the wallet and subtensor objects. This allows ValidatorBroadcastBase
        to work in separate processes without direct subtensor access.

        Args:
            synapse_factory: Function that creates the synapse to broadcast
            broadcast_name: Human-readable name for logging
            context: Optional context dict for logging
        """
        try:
            if not self._metagraph_client:
                bt.logging.debug(f"[BROADCAST] No metagraph client configured, skipping {broadcast_name} broadcast")
                return

            # Get other validators to broadcast to
            validator_axons = self._get_validator_axons()

            if not validator_axons:
                bt.logging.debug(f"[BROADCAST] No other validators to broadcast {broadcast_name} to")
                return

            # Create synapse using factory
            synapse = synapse_factory()

            # Validate synapse is picklable for RPC transmission
            synapse = self._serialize_synapse(synapse)

            context_str = f" for {context}" if context else ""
            bt.logging.info(
                f"[BROADCAST] Broadcasting {broadcast_name}{context_str} to {len(validator_axons)} validators via RPC"
            )

            # Broadcast via RPC client
            result = self._subtensor_ops_client.broadcast_to_validators_rpc(
                synapse=synapse,
                validator_axons_list=validator_axons
            )

            if result.get("success"):
                success_count = result.get("success_count", 0)
                total_count = result.get("total_count", 0)
                errors = result.get("errors", [])

                bt.logging.info(
                    f"[BROADCAST] {broadcast_name} broadcast completed: "
                    f"{success_count}/{total_count} validators updated"
                )

                # Log any errors
                for error in errors:
                    bt.logging.warning(f"[BROADCAST] Broadcast error: {error}")
            else:
                errors = result.get("errors", ["Unknown error"])
                bt.logging.error(f"[BROADCAST] Broadcast failed: {errors}")

        except Exception as e:
            bt.logging.error(f"[BROADCAST] Error in RPC broadcast {broadcast_name}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    def _serialize_synapse(self, synapse: template.protocol.bt.Synapse) -> template.protocol.bt.Synapse:
        """
        Validate that synapse is picklable for RPC transmission.

        Args:
            synapse: The synapse object to serialize

        Returns:
            The synapse object (if picklable)

        Raises:
            TypeError: If synapse is not picklable
        """
        import pickle

        # Verify synapse is picklable
        try:
            pickle.dumps(synapse)
        except (TypeError, AttributeError) as e:
            raise TypeError(f"Synapse {synapse.__class__.__name__} is not picklable: {e}")

        return synapse

    def _get_validator_axons(self) -> list:
        """
        Get list of validator axons to broadcast to (excluding self).

        Returns:
            List of axon_info objects for other validators
        """
        if self.is_testnet:
            # Testnet: All validators with valid IP (no stake requirement)
            validator_axons = [
                n.axon_info for n in self._metagraph_client.get_neurons()
                if n.axon_info.ip != ValiConfig.AXON_NO_IP
                and (not self._hotkey or n.axon_info.hotkey != self._hotkey)
            ]
        else:
            # Mainnet: Validators with minimum stake
            validator_axons = [
                n.axon_info for n in self._metagraph_client.get_neurons()
                if n.stake > bt.Balance(ValiConfig.STAKE_MIN)
                and n.axon_info.ip != ValiConfig.AXON_NO_IP
                and (not self._hotkey or n.axon_info.hotkey != self._hotkey)
            ]

        return validator_axons

    # ==================== Receiver Methods (All Validators) ====================

    def verify_broadcast_sender(
        self,
        sender_hotkey: Optional[str],
        broadcast_name: str
    ) -> bool:
        """
        Verify that a broadcast sender is authorized (must be mothership).

        This should be called by all receive_*_update methods to validate
        that the broadcast came from the authorized mothership validator.

        IMPORTANT: The sender_hotkey parameter should ALWAYS come from
        synapse.dendrite.hotkey, which is cryptographically verified by
        Bittensor's authentication system. Never accept sender identity
        from user-provided parameters.

        Args:
            sender_hotkey: Hotkey from synapse.dendrite.hotkey (Bittensor-authenticated)
            broadcast_name: Name of broadcast type for logging

        Returns:
            bool: True if sender is authorized, False otherwise
        """
        # SECURITY: Only process if receiver is not the mothership
        # (mothership doesn't process its own broadcasts)
        if self.is_mothership:
            bt.logging.debug(f"[BROADCAST] Mothership ignoring own {broadcast_name} broadcast")
            return False

        # SECURITY: Verify sender is the authorized mothership
        if not ValiConfig.MOTHERSHIP_HOTKEY:
            bt.logging.warning(
                f"[SECURITY] MOTHERSHIP_HOTKEY not configured in ValiConfig. Cannot verify {broadcast_name} broadcast."
            )
            return False

        if not sender_hotkey:
            bt.logging.warning(
                f"[SECURITY] No sender_hotkey provided for {broadcast_name} broadcast."
            )
            return False

        if sender_hotkey != ValiConfig.MOTHERSHIP_HOTKEY:
            bt.logging.warning(
                f"[SECURITY] Rejected {broadcast_name} broadcast from unauthorized validator: {sender_hotkey}. "
                f"Only mothership ({ValiConfig.MOTHERSHIP_HOTKEY}) can broadcast."
            )
            return False

        # Sender is authorized
        return True

    def _handle_incoming_broadcast(
        self,
        synapse_data: dict,
        sender_hotkey: Optional[str],
        broadcast_name: str,
        update_callback: Callable[[dict], None]
    ) -> bool:
        """
        Common handler for processing incoming broadcasts from other validators.

        This method provides a unified pattern for all receive_*_update methods:
        1. Verify the sender is authorized (mothership)
        2. Call the update callback to modify local state
        3. Handle logging and error handling consistently

        Example usage in a manager:
            def receive_my_update(self, data: dict, sender_hotkey: str = None) -> bool:
                def update_state(data):
                    # Extract fields
                    hotkey = data.get("hotkey")
                    value = data.get("value")

                    # Validate
                    if not hotkey or not value:
                        raise ValueError("Missing required fields")

                    # Update local state
                    self.my_dict[hotkey] = value
                    self._save_to_disk()

                return self._handle_incoming_broadcast(
                    synapse_data=data,
                    sender_hotkey=sender_hotkey,
                    broadcast_name="MyUpdate",
                    update_callback=update_state
                )

        Args:
            synapse_data: Dictionary containing the broadcast data
            sender_hotkey: The hotkey of the validator that sent this broadcast
                          (from synapse.dendrite.hotkey)
            broadcast_name: Name of the broadcast type (for logging, e.g., "AssetSelection")
            update_callback: Function that takes synapse_data and updates local state.
                           Should raise an exception if validation fails.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # SECURITY: Verify sender using shared verification
            if not self.verify_broadcast_sender(sender_hotkey, broadcast_name):
                return False

            # Call the update callback to modify local state
            update_callback(synapse_data)

            bt.logging.info(
                f"[{broadcast_name.upper()}] Successfully processed broadcast from {sender_hotkey}"
            )
            return True

        except Exception as e:
            bt.logging.error(f"[{broadcast_name.upper()}] Error processing broadcast: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False
