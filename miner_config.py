import os
from vali_objects.vali_config import ValiConfig


BASE_DIR = base_directory = os.path.dirname(os.path.abspath(__file__))

class MinerConfig:
    HIGH_V_TRUST_THRESHOLD = 0.75
    STAKE_MIN = 1000  # Do not change from int
    AXON_NO_IP = "0.0.0.0"

    DASHBOARD_API_PORT = 41511
    BASE_DIR = base_directory = BASE_DIR

    # Miner API Server configuration
    MINER_REST_HOST = "0.0.0.0"
    MINER_REST_PORT = 8088

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

    @staticmethod
    def get_secrets_file_path() -> str:
        """Get path to miner API keys file."""
        return ValiConfig.BASE_DIR + "/mining/miner_secrets.json"

    # USDC Payment Configuration (Base chain)
    USDC_CONTRACT_ADDRESS_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    PAYMENT_MIN_USDC_AMOUNT = 1.0
    PAYMENT_GAS_BUFFER_MULTIPLIER = 1.2
    PAYMENT_CONFIRMATION_TIMEOUT_S = 120
    PAYMENT_MAX_RETRIES = 3
    PAYMENT_RETRY_DELAY_S = 5.0
    HYPERSCALED_API_URL = "https://hyperscaled.trade"

    @staticmethod
    def get_payment_ledger_file_path() -> str:
        """Get path to payment ledger file."""
        return ValiConfig.BASE_DIR + "/mining/payment_ledger.json"
