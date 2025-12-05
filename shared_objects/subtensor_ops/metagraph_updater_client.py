from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig


class WeightSetterClient(RPCClientBase):
    """
    RPC client for calling set_weights_rpc on the local subtensor.

    Used by WeightCalculatorServer to send weight setting requests
    to MetagraphUpdater running in a separate process.

    Usage:
        client = MetagraphUpdaterClient()
        result = client.set_weights_rpc(uids=[1,2,3], weights=[0.3,0.3,0.4], version_key=200)
    """

    def __init__(self, running_unit_tests=False, connect_immediately=True):
        super().__init__(
            service_name=ValiConfig.RPC_WEIGHT_SETTER_SERVICE_NAME,
            port=ValiConfig.RPC_WEIGHT_SETTER_PORT,
            connect_immediately=False
        )
        self.running_unit_tests = running_unit_tests

    def set_weights_rpc(self, uids: list, weights: list, version_key: int) -> dict:
        """
        Send weight setting request to MetagraphUpdater.

        Args:
            uids: List of UIDs to set weights for
            weights: List of weights corresponding to UIDs
            version_key: Subnet version key

        Returns:
            dict: {"success": bool, "error": str or None}
        """
        return self.call("set_weights_rpc", uids, weights, version_key)
