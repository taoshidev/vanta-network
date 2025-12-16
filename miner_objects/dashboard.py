import socket

import bittensor as bt
import uvicorn
import template.protocol
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from miner_config import MinerConfig
from shared_objects.rate_limiter import RateLimiter
# Note: Using async with bt.dendrite() for automatic cleanup - no need for manual DendriteUtils

origins = [
    "*",
    "http://localhost",
]

class Dashboard:
    def __init__(self, wallet, metagraph, config, is_testnet):
        self.wallet = wallet
        self.metagraph = metagraph
        self.config = config
        self.is_testnet = is_testnet
        self.port = self.get_next_unused_port(MinerConfig.DASHBOARD_API_PORT, MinerConfig.DASHBOARD_API_PORT+100)

        self.dash_rate_limiter = RateLimiter(max_requests_per_window=1, rate_limit_window_duration_seconds=60)

        self.app = FastAPI()
        self._add_cors_middleware()
        self._setup_routes()
        self.miner_data = {"timestamp": 0, "statistics": {"data": []}, "positions": {}}

    def _add_cors_middleware(self):
        # allow the connection from the frontend
        origins.append(f"http://localhost:{self.port}")

        # Set up CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],  # allow all HTTP methods
            allow_headers=["*"],  # allow all headers
        )

    def _setup_routes(self):
        @self.app.get("/miner")
        async def get_miner():
            return self.wallet.hotkey.ss58_address

        @self.app.get("/miner-data")
        async def get_miner_data():
            allowed, wait_time = self.dash_rate_limiter.is_allowed(self.wallet.hotkey.ss58_address)
            if not allowed:
                bt.logging.info(f"Rate limited. Please wait {wait_time} seconds before refreshing.")
                return self.miner_data

            success = await self.refresh_validator_dash_data()
            if not success:
                bt.logging.info("No data received from validator. Setting and returning empty data.")

            return self.miner_data

    def get_next_unused_port(self, start, stop):
        """
        finds a free port in the range [start, stop). raises an OSError if a port is unable to be found, and aborts dashboard startup.
        """
        for port in range(start, stop):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    self.write_env_file(port)
                    return port  # found a free port
        raise OSError(f"All ports from [{start}, {stop}) in use. Aborting dashboard.")

    def write_env_file(self, api_port):
        """
        write the miner_url to the miner_dashboard .env file
        """
        env_file_path = MinerConfig.BASE_DIR + "/miner_objects/miner_dashboard/.env"
        with open(env_file_path, 'w') as env_file:
            env_file.write(f'VITE_MINER_URL=http://127.0.0.1:{api_port}\n')

    def run(self):
        uvicorn.run(self.app, host="127.0.0.1", port=self.port)

