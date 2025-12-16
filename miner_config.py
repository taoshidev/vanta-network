import os
from vali_objects.vali_config import ValiConfig


BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

class MinerConfig:
    HIGH_V_TRUST_THRESHOLD = 0.75
    STAKE_MIN = 1000  # Do not change from int
    AXON_NO_IP = "0.0.0.0"

    DASHBOARD_API_PORT = 41511
    BASE_DIR = base_directory = BASE_DIR

    @staticmethod
    def get_miner_received_signals_dir() -> str:
        return ValiConfig.BASE_DIR + "/mining/received_signals/"

    @staticmethod
    def get_miner_processed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + "/mining/processed_signals/"

    @staticmethod
    def get_miner_failed_signals_dir() -> str:
        return ValiConfig.BASE_DIR + "/mining/failed_signals/"

    @staticmethod
    def get_position_file_location() -> str:
        return ValiConfig.BASE_DIR + f"/mining/positions.json"
