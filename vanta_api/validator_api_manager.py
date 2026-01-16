import json
import os
import time

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vanta_api.validator_rest_server import ValidatorRestServer
from shared_objects.slack_notifier import SlackNotifier
from vanta_api.websocket_server import WebSocketServer


class ValidatorAPIManager:
    """Manages API services and processes."""

    def __init__(self, refresh_interval=15,
                 slack_webhook_url=None,
                 validator_hotkey=None,
                 api_host=None,
                 api_rest_port=None,
                 api_ws_port=None):
        """Initialize API management with server configurations.

        Uses spawn_process() for process management with:
        - Automatic health monitoring
        - Auto-restart on failure
        - Slack notifications

        Note: Both servers inherit from RPCServerBase and use spawn_process()
        Server endpoints default to ValiConfig values but can be overridden:
        - VantaRestServer: Flask HTTP on api_host:api_rest_port (default: 127.0.0.1:48888), RPC on port 50022
        - WebSocketServer: WebSocket on api_host:api_ws_port (default: localhost:8765), RPC on port 50014

        Args:
            refresh_interval: How often to check for API key changes (seconds)
            slack_webhook_url: Slack webhook URL for health alerts (optional)
            validator_hotkey: Validator hotkey for identification in alerts (optional)
            api_host: Host address for API servers (default: ValiConfig.REST_API_HOST)
            api_rest_port: Port for REST API (default: ValiConfig.REST_API_PORT)
            api_ws_port: Port for WebSocket (default: ValiConfig.VANTA_WEBSOCKET_PORT)
        """
        from vali_objects.vali_config import ValiConfig

        self.refresh_interval = refresh_interval

        # Store API configuration (use ValiConfig defaults if not provided)
        self.api_host = api_host if api_host is not None else ValiConfig.REST_API_HOST
        self.api_rest_port = api_rest_port if api_rest_port is not None else ValiConfig.REST_API_PORT
        self.api_ws_port = api_ws_port if api_ws_port is not None else ValiConfig.VANTA_WEBSOCKET_PORT

        # Initialize Slack notifier
        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)

        # Process handles (set in run())
        self.rest_handle = None
        self.ws_handle = None

        # Get default API keys file path
        self.api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        # Verify API keys file exists
        if not os.path.exists(self.api_keys_file):
            print(f"WARNING: API keys file '{self.api_keys_file}' not found!")
        else:
            print(f"API keys file found at: {self.api_keys_file}")
            # Check if it's a valid JSON file
            try:
                with open(self.api_keys_file, "r") as f:
                    keys = json.load(f)
                print(f"API keys file contains {len(keys)} keys")
            except Exception as e:
                print(f"ERROR reading API keys file: {e}")


    def run(self):
        """Main entry point to run REST API and WebSocket server with automatic restart capability.

        Uses spawn_process() for:
        - Automatic health monitoring
        - Auto-restart on failure
        - Slack notifications
        """
        print("Starting API services with spawn_process()...")

        # Spawn REST server using spawn_process() with configured host/port
        print(f"Spawning REST API server at http://{self.api_host}:{self.api_rest_port}...")
        self.rest_handle = ValidatorRestServer.spawn_process(
            api_keys_file=self.api_keys_file,
            refresh_interval=self.refresh_interval,
            slack_notifier=self.slack_notifier,
            health_check_interval_s=10.0,  # Check every 10 seconds
            enable_auto_restart=True,
            # Pass host/port configuration to REST server
            flask_host=self.api_host,
            flask_port=self.api_rest_port
        )
        print(f"REST API server spawned (PID: {self.rest_handle.pid})")

        # Spawn WebSocket server using spawn_process() with configured host/port
        print(f"Spawning WebSocket server at ws://{self.api_host}:{self.api_ws_port}...")
        self.ws_handle = WebSocketServer.spawn_process(
            api_keys_file=self.api_keys_file,
            refresh_interval=self.refresh_interval,
            slack_notifier=self.slack_notifier,
            health_check_interval_s=10.0,  # Check every 10 seconds
            enable_auto_restart=True,
            # Pass host/port configuration to WebSocket server
            websocket_host=self.api_host,
            websocket_port=self.api_ws_port
        )
        print(f"WebSocket server spawned (PID: {self.ws_handle.pid})")
        print("Both servers running with automatic health monitoring and restart")

        # Keep main thread alive - health monitoring happens in background threads
        try:
            while True:
                time.sleep(60)  # Just keep alive

        except KeyboardInterrupt:
            print("\nShutting down API services due to keyboard interrupt...")

            # Stop both servers gracefully
            if self.rest_handle:
                print(f"Stopping REST server (PID: {self.rest_handle.pid})...")
                self.rest_handle.stop()

            if self.ws_handle:
                print(f"Stopping WebSocket server (PID: {self.ws_handle.pid})...")
                self.ws_handle.stop()

            print("API services shutdown complete.")


if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the API services")
    args = parser.parse_args()

    # Note: Server endpoints are hardcoded in ValiConfig (well-known network endpoints)
    from vali_objects.vali_config import ValiConfig
    print(f"API services will run on well-known network endpoints:")
    print(f"  REST API: http://{ValiConfig.REST_API_HOST}:{ValiConfig.REST_API_PORT}")
    print(f"  Vanta WebSocket: ws://{ValiConfig.VANTA_WEBSOCKET_HOST}:{ValiConfig.VANTA_WEBSOCKET_PORT}")

    # Create and run the API manager
    # WebSocket notifications now use RPC instead of multiprocessing.Queue
    api_manager = ValidatorAPIManager()
    api_manager.run()
