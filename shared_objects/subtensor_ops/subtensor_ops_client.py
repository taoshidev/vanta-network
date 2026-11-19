from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig


class SubtensorOpsClient(RPCClientBase):
    """
    RPC client for calling functions on the local subtensor.

    Used by WeightCalculatorServer and ValidatorBroadcastBase to send requests
    to SubtensorOpsManager running in a separate process.

    Supports:
    - Weight setting requests
    - Validator broadcast requests

    Usage:
        client = SubtensorOpsClient()
        result = client.set_weights_rpc(uids=[1,2,3], weights=[0.3,0.3,0.4], version_key=200)
        result = client.broadcast_to_validators_rpc(synapse_dict, validator_axons)
    """

    # set_weights waits for inclusion, finalization and revealed execution under the subtensor
    # lock; see ValiConfig.RPC_CHAIN_CALL_TIMEOUT_S.
    RPC_CALL_TIMEOUT_S = ValiConfig.RPC_CHAIN_CALL_TIMEOUT_S

    def __init__(self, running_unit_tests=False, connect_immediately=True):
        self.running_unit_tests = running_unit_tests
        if self.running_unit_tests:
            return  # Don't want to connect to local subtensor for tests
        super().__init__(
            service_name=ValiConfig.RPC_WEIGHT_SETTER_SERVICE_NAME,
            port=ValiConfig.RPC_WEIGHT_SETTER_PORT,
            connect_immediately=connect_immediately
        )

    def set_weights_rpc(self, uids: list, weights: list, version_key: int) -> dict:
        """
        Send weight setting request to SubtensorOpsManager.

        Args:
            uids: List of UIDs to set weights for
            weights: List of weights corresponding to UIDs
            version_key: Subnet version key

        Returns:
            dict: {"success": bool, "error": str or None}
        """
        return self.call("set_weights_rpc", uids, weights, version_key)

    def broadcast_to_validators_rpc(self, request, validator_axons_list: list, message_name: str = None) -> dict:
        """
        Send broadcast request to SubtensorOpsManager.

        This allows processes without direct subtensor/wallet access to broadcast
        messages to validators using the SubtensorOpsManager's wallet and the
        signed-request HTTP client.

        Args:
            request: One of vanta_protocol's *Request models (must be picklable)
            validator_axons_list: List of axon_info objects to broadcast to
            message_name: The MESSAGE_TYPES/HTTP-path key for this broadcast (e.g.
                "CollateralRecord") — required since several message types share the
                generic BroadcastRequest shape and can no longer be told apart by class.

        Returns:
            dict: {
                "success": bool,
                "success_count": int,
                "total_count": int,
                "errors": list of error messages
            }
        """
        return self.call("broadcast_to_validators_rpc", request, validator_axons_list, message_name)
