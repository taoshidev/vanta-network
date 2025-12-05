from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import ValiConfig


class WeightCalculatorClient(RPCClientBase):
    """
    RPC client for WeightCalculatorServer.

    Provides access to weight calculation results and daemon control.

    Usage:
        client = WeightCalculatorClient()
        client.start_daemon()  # Start weight calculation daemon
        results = client.get_checkpoint_results_rpc()
    """

    def __init__(self, running_unit_tests=False):
        super().__init__(
            service_name=ValiConfig.RPC_WEIGHT_CALCULATOR_SERVICE_NAME,
            port=ValiConfig.RPC_WEIGHT_CALCULATOR_PORT,
            connect_immediately=True
        )
        self.running_unit_tests = running_unit_tests

    def get_checkpoint_results_rpc(self) -> list:
        """Get latest checkpoint results from server."""
        return self.call("get_checkpoint_results_rpc")

    def get_transformed_list_rpc(self) -> list:
        """Get latest transformed weight list from server."""
        return self.call("get_transformed_list_rpc")
