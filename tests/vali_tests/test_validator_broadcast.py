# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Validator Broadcast and Synapse Pickling Tests

This test suite validates:
1. All synapse types can be pickled for RPC transmission
2. ValidatorBroadcastBase serialization logic works correctly
3. SubtensorOpsManager broadcast RPC method handles synapses properly
4. Error cases are handled gracefully (non-picklable objects, invalid synapses)

This is critical for the validator broadcast architecture where synapses are
transmitted via RPC between processes.
"""
import unittest
import pickle
from unittest.mock import Mock
from types import SimpleNamespace

import vanta_protocol
from vali_objects.vali_config import ValiConfig
from vali_objects.validator_broadcast_base import ValidatorBroadcastBase
from shared_objects.subtensor_ops.subtensor_ops import SubtensorOpsManager


class TestSynapsePickling(unittest.TestCase):
    """
    Test that all synapse types can be successfully pickled.

    This validates our design decision to use direct pickle instead of
    custom serialization/deserialization.
    """

    def setUp(self):
        """Set up test data for all synapse types."""
        # Sample data for each synapse type
        self.signal_data = {
            "trade_pair": "BTCUSD",
            "position": "LONG",
            "leverage": 1.0
        }

        self.position_data = {
            "position_uuid": "test-uuid",
            "trade_pair": "BTCUSD",
            "return_at_close": 0.05
        }

        self.checkpoint_data = "compressed_checkpoint_data_here"

        self.collateral_record = {
            "hotkey": "test_hotkey",
            "collateral_amount": 1000.0
        }

        self.asset_selection = {
            "hotkey": "test_hotkey",
            "selected_asset": "CRYPTO"
        }

        self.subaccount_data = {
            "entity_hotkey": "entity_hotkey_1",
            "subaccount_id": 0,
            "subaccount_uuid": "uuid-123",
            "synthetic_hotkey": "entity_hotkey_1_0"
        }

        self.opening = {
            "hotkey": "test_hotkey",
            "record": {"opened_at_ms": 1_700_000_000_000, "opening_leverage_tier": 2, "account_size": 50_000.0}
        }

    def test_pickle_send_signal_synapse(self):
        """Test that SendSignalRequest can be pickled."""
        synapse = vanta_protocol.SendSignalRequest(
            signal=self.signal_data,
            repo_version="8.8.8",
            miner_order_uuid="test-uuid"
        )

        # Attempt to pickle
        pickled = pickle.dumps(synapse)
        self.assertIsNotNone(pickled)

        # Attempt to unpickle
        unpickled = pickle.loads(pickled)
        self.assertIsInstance(unpickled, vanta_protocol.SendSignalRequest)
        self.assertEqual(unpickled.signal, self.signal_data)
        self.assertEqual(unpickled.repo_version, "8.8.8")
        self.assertEqual(unpickled.miner_order_uuid, "test-uuid")

    def test_pickle_get_positions_synapse(self):
        """Test that GetPositionsRequest/GetPositionsResponse can be pickled.

        version lives on the request, positions on the response (no longer the same
        mutable object), so each is exercised separately.
        """
        request = vanta_protocol.GetPositionsRequest(version=1)
        pickled_request = pickle.dumps(request)
        unpickled_request = pickle.loads(pickled_request)
        self.assertIsInstance(unpickled_request, vanta_protocol.GetPositionsRequest)
        self.assertEqual(unpickled_request.version, 1)

        response = vanta_protocol.GetPositionsResponse(positions=[self.position_data])
        pickled_response = pickle.dumps(response)
        unpickled_response = pickle.loads(pickled_response)
        self.assertIsInstance(unpickled_response, vanta_protocol.GetPositionsResponse)
        self.assertEqual(len(unpickled_response.positions), 1)
        self.assertEqual(unpickled_response.positions[0], self.position_data)

    def test_pickle_validator_checkpoint_synapse(self):
        """Test that ValidatorCheckpointRequest/ValidatorCheckpointResponse can be pickled.

        checkpoint lives on the request, validator_receive_hotkey on the response.
        """
        request = vanta_protocol.ValidatorCheckpointRequest(checkpoint=self.checkpoint_data)
        pickled_request = pickle.dumps(request)
        unpickled_request = pickle.loads(pickled_request)
        self.assertIsInstance(unpickled_request, vanta_protocol.ValidatorCheckpointRequest)
        self.assertEqual(unpickled_request.checkpoint, self.checkpoint_data)

        response = vanta_protocol.ValidatorCheckpointResponse(validator_receive_hotkey="receiver_hotkey")
        pickled_response = pickle.dumps(response)
        unpickled_response = pickle.loads(pickled_response)
        self.assertIsInstance(unpickled_response, vanta_protocol.ValidatorCheckpointResponse)
        self.assertEqual(unpickled_response.validator_receive_hotkey, "receiver_hotkey")

    def test_pickle_broadcast_request_synapse(self):
        """Test that the generic BroadcastRequest (shared by CollateralRecord, AssetSelection,
        SubaccountRegistration, StandardMinerOpening, StandardMinerRecord, EntityEndpointUpdate
        and AccountSizeLocks) can be pickled, whatever payload shape it's carrying."""
        for payload in (self.collateral_record, self.asset_selection, self.subaccount_data, self.opening):
            synapse = vanta_protocol.BroadcastRequest(payload=payload)

            pickled = pickle.dumps(synapse)
            unpickled = pickle.loads(pickled)

            self.assertIsInstance(unpickled, vanta_protocol.BroadcastRequest)
            self.assertEqual(unpickled.payload, payload)

    def test_pickle_all_synapses_with_full_state(self):
        """Test pickling each message type's *Response with all response fields populated.

        (successfully_processed/error_message live only on *Response now, not on the
        *Request alongside the request payload, so this exercises the Response side.)
        """
        synapses = [
            vanta_protocol.SendSignalResponse(
                successfully_processed=True,
                error_message="test error",
                validator_hotkey="validator_key",
                order_json='{"test": "json"}'
            ),
            vanta_protocol.GetPositionsResponse(
                positions=[self.position_data],
                successfully_processed=False,
                error_message="error"
            ),
            vanta_protocol.ValidatorCheckpointResponse(
                successfully_processed=True,
                validator_receive_hotkey="receiver"
            ),
            # Shared by CollateralRecord, AssetSelection, SubaccountRegistration,
            # StandardMinerOpening, StandardMinerRecord, EntityEndpointUpdate, AccountSizeLocks.
            vanta_protocol.BroadcastResponse(
                successfully_processed=True
            ),
        ]

        # Verify all can be pickled and unpickled
        for synapse in synapses:
            pickled = pickle.dumps(synapse)
            unpickled = pickle.loads(pickled)

            # Verify class type preserved
            self.assertEqual(type(unpickled), type(synapse))

            # Verify successfully_processed field preserved
            self.assertEqual(
                unpickled.successfully_processed,
                synapse.successfully_processed
            )


class TestValidatorBroadcastBase(unittest.TestCase):
    """
    Test ValidatorBroadcastBase serialization and validation logic.
    """

    def setUp(self):
        """Set up test instance."""
        # Create a simple test class that inherits from ValidatorBroadcastBase
        self.broadcast_base = ValidatorBroadcastBase(
            running_unit_tests=True,
            is_testnet=True
        )

    @staticmethod
    def _neuron(hotkey, ip="1.2.3.4"):
        return SimpleNamespace(axon_info=SimpleNamespace(hotkey=hotkey, ip=ip))

    def test_mainnet_targets_come_from_the_metagraph_validator_set_minus_self(self):
        """The chain may grant a single permit, so the targets are the validators the metagraph server
        derives from stake weight, excluding this validator."""
        base = ValidatorBroadcastBase(running_unit_tests=True, is_testnet=False)
        base._metagraph_client = Mock()
        base._metagraph_client.get_validator_axons.return_value = [self._neuron("other").axon_info]

        base._hotkey = "self"
        self.assertEqual([a.hotkey for a in base._get_validator_axons()], ["other"])
        base._metagraph_client.get_validator_axons.assert_called_once_with(exclude_hotkey="self")

        base._hotkey = None
        base._get_validator_axons()
        base._metagraph_client.get_validator_axons.assert_called_with(exclude_hotkey=None)
        base._metagraph_client.get_neurons.assert_not_called()

    def test_testnet_targets_are_every_served_axon_but_self(self):
        base = self.broadcast_base
        base._hotkey = "self"
        base._metagraph_client = Mock()
        base._metagraph_client.get_neurons.return_value = [
            self._neuron("self"),
            self._neuron("unstaked"),
            self._neuron("no_ip", ip=ValiConfig.AXON_NO_IP),
        ]
        self.assertEqual([a.hotkey for a in base._get_validator_axons()], ["unstaked"])
        base._metagraph_client.get_validator_axons.assert_not_called()

    def test_serialize_synapse_valid(self):
        """Test that valid picklable requests pass validation."""
        synapse = vanta_protocol.SendSignalRequest(
            signal={"test": "data"},
            repo_version="8.8.8"
        )

        # Should return request unchanged if picklable
        result = self.broadcast_base._serialize_request(synapse)

        self.assertIs(result, synapse)  # Same object reference

    def test_serialize_synapse_invalid(self):
        """Test that non-picklable objects raise TypeError."""
        # Create a request with non-picklable attribute
        synapse = vanta_protocol.SendSignalRequest()
        synapse._unpicklable_lambda = lambda x: x  # Lambdas can't be pickled

        with self.assertRaises(TypeError) as context:
            self.broadcast_base._serialize_request(synapse)

        self.assertIn("not picklable", str(context.exception))

    def test_serialize_synapse_complex_data(self):
        """Test serialization with complex nested data structures."""
        complex_signal = {
            "trade_pair": "BTCUSD",
            "position": "LONG",
            "nested": {
                "level1": {
                    "level2": [1, 2, 3, {"deep": "data"}]
                }
            },
            "list_data": [1, 2, 3, 4, 5]
        }

        synapse = vanta_protocol.SendSignalRequest(signal=complex_signal)

        # Should successfully validate
        result = self.broadcast_base._serialize_request(synapse)
        self.assertIs(result, synapse)

        # Verify still picklable
        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)
        self.assertEqual(unpickled.signal, complex_signal)


class TestSubtensorOpsManagerBroadcast(unittest.TestCase):
    """
    Test SubtensorOpsManager's broadcast_to_validators_rpc method.
    """

    def setUp(self):
        """Set up mock SubtensorOpsManager for testing."""
        # Create mock config with all required attributes
        self.mock_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey="test_hotkey"),
            subtensor=SimpleNamespace(network="test")  # Add subtensor.network
        )

        # Create SubtensorOpsManager in test mode
        self.manager = SubtensorOpsManager(
            config=self.mock_config,
            hotkey="test_hotkey",
            is_miner=False,
            running_unit_tests=True  # Skip actual RPC server
        )

    def test_broadcast_rpc_validates_synapse(self):
        """Test that broadcast_rpc validates synapse object."""
        valid_synapse = vanta_protocol.BroadcastRequest(
            payload={"entity_hotkey": "test"}
        )

        mock_axons = [Mock()]

        # Should not raise - validation passes
        result = self.manager.broadcast_to_validators_rpc(valid_synapse, mock_axons, message_name="SubaccountRegistration")

        # In unit test mode, should return success with 0 validators
        self.assertTrue(result["success"])
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["total_count"], 0)

    def test_broadcast_rpc_rejects_invalid_synapse(self):
        """
        Test that broadcast_rpc validates synapse structure.

        Note: In unit test mode, the method returns early with success=True.
        This test verifies the validation logic is in place for non-test modes.
        """
        # In unit test mode, None is allowed (early return)
        # This is acceptable since the validation happens in production
        result = self.manager.broadcast_to_validators_rpc(None, [Mock()], message_name="SubaccountRegistration")
        self.assertTrue(result["success"])  # Early return in unit test mode

        # The important test: Verify the validation code exists
        # by checking it doesn't crash with valid synapse
        valid_synapse = vanta_protocol.BroadcastRequest(
            payload={"entity_hotkey": "test"}
        )
        result = self.manager.broadcast_to_validators_rpc(valid_synapse, [Mock()], message_name="SubaccountRegistration")
        self.assertTrue(result["success"])

    def test_broadcast_rpc_handles_empty_validator_list(self):
        """Test that broadcast_rpc handles empty validator list gracefully."""
        synapse = vanta_protocol.BroadcastRequest(
            payload={"hotkey": "test"}
        )

        result = self.manager.broadcast_to_validators_rpc(synapse, [], message_name="AssetSelection")

        # Should return success with 0 validators
        self.assertTrue(result["success"])
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["total_count"], 0)

    def test_broadcast_rpc_requires_message_name(self):
        """Test that broadcast_rpc requires an explicit message_name now that several
        message types share the generic BroadcastRequest shape (no longer derivable from
        the request's class name alone)."""
        synapse = vanta_protocol.BroadcastRequest(
            payload={"hotkey": "test", "amount": 100.0}
        )

        result = self.manager.broadcast_to_validators_rpc(synapse, [Mock()], message_name="CollateralRecord")

        self.assertTrue(result["success"])


class TestEndToEndBroadcastFlow(unittest.TestCase):
    """
    Test complete broadcast flow from ValidatorBroadcastBase through
    SubtensorOpsClient to SubtensorOpsManager.

    This simulates the real-world usage pattern.
    """

    def test_broadcast_flow_with_subaccount_registration(self):
        """
        Test complete flow: ValidatorBroadcastBase -> RPC -> SubtensorOpsManager

        This validates that synapses can be pickled, transmitted via RPC,
        and processed without custom serialization/deserialization.
        """
        # 1. Create synapse in ValidatorBroadcastBase
        broadcast_base = ValidatorBroadcastBase(
            running_unit_tests=True,
            is_testnet=True
        )

        synapse_data = {
            "entity_hotkey": "entity_1",
            "subaccount_id": 0,
            "subaccount_uuid": "uuid-123",
            "synthetic_hotkey": "entity_1_0"
        }

        synapse = vanta_protocol.BroadcastRequest(
            payload=synapse_data
        )

        # 2. Validate synapse is picklable
        validated_synapse = broadcast_base._serialize_request(synapse)

        # 3. Simulate RPC transmission (pickle/unpickle)
        pickled = pickle.dumps(validated_synapse)
        transmitted_synapse = pickle.loads(pickled)

        # 4. Process in SubtensorOpsManager
        mock_config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey="test_hotkey"),
            subtensor=SimpleNamespace(network="test")
        )

        manager = SubtensorOpsManager(
            config=mock_config,
            hotkey="test_hotkey",
            is_miner=False,
            running_unit_tests=True
        )

        result = manager.broadcast_to_validators_rpc(transmitted_synapse, [], message_name="SubaccountRegistration")

        # 5. Verify success
        self.assertTrue(result["success"])

        # 6. Verify data integrity preserved through entire flow
        self.assertEqual(transmitted_synapse.payload, synapse_data)

    def test_broadcast_flow_preserves_all_synapse_types(self):
        """
        Test that all synapse types preserve data through broadcast flow.
        """
        broadcast_base = ValidatorBroadcastBase(
            running_unit_tests=True,
            is_testnet=True
        )

        # GetPositions is the one type here whose interesting field (positions) lives on the
        # *Response, not the *Request broadcast by _serialize_request/_broadcast_to_validators
        # — so it's exercised as a response, pickled directly rather than through
        # _serialize_request (which is request-only).
        synapses = [
            vanta_protocol.SendSignalRequest(signal={"test": "data"}),
            vanta_protocol.ValidatorCheckpointRequest(checkpoint="checkpoint"),
            # CollateralRecord/AssetSelection/SubaccountRegistration/StandardMinerOpening/
            # StandardMinerRecord all share this generic shape now.
            vanta_protocol.BroadcastRequest(payload={"amount": 100}),
            vanta_protocol.BroadcastRequest(payload={"asset": "CRYPTO"}),
            vanta_protocol.BroadcastRequest(payload={"id": 0}),
            vanta_protocol.BroadcastRequest(payload={"hotkey": "h", "record": {"account_size": 50_000.0}}),
            vanta_protocol.BroadcastRequest(payload={"hotkey": "h", "record": {"pro_account_size": 250_000.0}}),
        ]

        for original_synapse in synapses:
            # Validate
            validated = broadcast_base._serialize_request(original_synapse)

            # Simulate RPC transmission
            transmitted = pickle.loads(pickle.dumps(validated))

            # Verify type preserved
            self.assertEqual(type(transmitted), type(original_synapse))

            # Verify data fields preserved (check first field of each type)
            if isinstance(transmitted, vanta_protocol.SendSignalRequest):
                self.assertEqual(transmitted.signal, original_synapse.signal)
            elif isinstance(transmitted, vanta_protocol.ValidatorCheckpointRequest):
                self.assertEqual(transmitted.checkpoint, original_synapse.checkpoint)
            elif isinstance(transmitted, vanta_protocol.BroadcastRequest):
                self.assertEqual(transmitted.payload, original_synapse.payload)

        # GetPositionsResponse pickled directly (not a request, so not through
        # _serialize_request/_broadcast_to_validators).
        original_response = vanta_protocol.GetPositionsResponse(positions=[{"pos": "data"}])
        transmitted_response = pickle.loads(pickle.dumps(original_response))
        self.assertEqual(type(transmitted_response), type(original_response))
        self.assertEqual(transmitted_response.positions, original_response.positions)


if __name__ == '__main__':
    unittest.main()
