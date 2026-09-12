#!/usr/bin/env python3
"""
Production entrypoint for the standalone Validator WebSocket server (PM2 app: vanta-ws).

This is the REAL prod entrypoint — NOT the test `__main__` at the bottom of
websocket_server.py (which defaults send_test_positions=True and writes fake API keys).

Running it as its own PM2 app means a WS deploy no longer restarts the validator core.

Startup shape differs from REST (intentionally):
  - The WS constructor starts ONLY the notifier RPC server (:50014). The actual
    websocket serving loop lives in run() (asyncio). So here we DO call server.run()
    (unlike vanta-rest, whose Flask server is already started by its constructor).
  - The WS constructor has a fixed signature and takes no slack_notifier, so the
    server itself runs without one (matching the in-validator spawn path, which filters
    slack_notifier out). We keep a local SlackNotifier only for a crash alert here.

Shutdown isolation: same host-global-flag concern as vanta-rest — we set our own
ShutdownCoordinator namespace before importing anything that binds it, so a core
restart can't tear us down and our restart can't signal core.

PM2 owns supervision/restart; spawn_process()/enable_auto_restart are not used.

Usage (PM2 runs, roughly):
    python vanta_api/run_ws_server.py
"""

import os

# Isolate this app's shutdown lifecycle from vanta-core. Must be set before the imports
# below (ShutdownCoordinator binds its segment name at import time). setdefault lets
# run.sh override via the PM2 environment.
os.environ.setdefault("VANTA_SHUTDOWN_SHM_NAME", "vanta_ws_shutdown")

import argparse  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

import bittensor as bt  # noqa: E402

from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator  # noqa: E402
from shared_objects.slack_notifier import SlackNotifier  # noqa: E402
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils  # noqa: E402
from vali_objects.vali_config import RPCConnectionMode, ValiConfig  # noqa: E402
from vanta_api.server_readiness import start_readiness_watchdog  # noqa: E402
from vanta_api.websocket_server import WebSocketServer  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone Validator WebSocket server (vanta-ws).")
    parser.add_argument("--netuid", type=int, default=None,
                        help="Accepted for CLI symmetry with run_rest_server.py; the WS server has no "
                             "mainnet-specific behavior, so this is informational only.")
    parser.add_argument("--api-keys-file", type=str, default=None,
                        help="Path to the API keys JSON file. Default: ValiBkpUtils.get_api_keys_file_path().")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the WebSocket server. Default: 127.0.0.1 — parity with "
                             "the in-core spawn's --api-host default, so the split never widens the "
                             "bind surface an operator didn't ask for. Pass 0.0.0.0 for external reachability.")
    parser.add_argument("--port", type=int, default=None,
                        help=f"Port for the WebSocket server. Default: ValiConfig.VANTA_WEBSOCKET_PORT ({ValiConfig.VANTA_WEBSOCKET_PORT}).")
    parser.add_argument("--refresh-interval", type=int, default=15,
                        help="Seconds between API-key file refresh checks. Default: 15.")
    parser.add_argument("--slack-webhook-url", type=str, default=None,
                        help="Slack webhook for crash alerts. If omitted, SlackNotifier falls back to the "
                             "SLACK_WEBHOOK_URL env var (same source as vanta-core).")
    parser.add_argument("--validator-hotkey", type=str, default=None,
                        help="Label used to identify this server in Slack alerts. No wallet is required.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    bt.logging.enable_info()

    api_keys_file = args.api_keys_file or ValiBkpUtils.get_api_keys_file_path()
    port = args.port if args.port is not None else ValiConfig.VANTA_WEBSOCKET_PORT
    alert_hotkey = args.validator_hotkey or f"vanta-ws@{socket.gethostname()}"

    ShutdownCoordinator.initialize(reset_on_attach=True)

    if not os.path.exists(api_keys_file):
        bt.logging.warning(f"[vanta-ws] API keys file not found at {api_keys_file} — "
                           f"the server will start but reject authenticated subscriptions until it exists.")

    slack_notifier = SlackNotifier(hotkey=alert_hotkey, webhook_url=args.slack_webhook_url)

    # PM2 stops processes with SIGINT by default; the WS run() loop already handles
    # KeyboardInterrupt (graceful shutdown). Convert SIGTERM to the same path so both
    # stop signals shut down cleanly.
    def _sigterm_to_keyboard_interrupt(signum, _frame):
        bt.logging.info(f"[vanta-ws] Received signal {signum} — raising KeyboardInterrupt for graceful shutdown.")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

    bt.logging.info(
        f"[vanta-ws] Starting standalone WebSocket server: host={args.host} port={port} "
        f"api_keys_file={api_keys_file} shutdown_ns={os.environ.get('VANTA_SHUTDOWN_SHM_NAME')}"
    )

    try:
        server = WebSocketServer(
            api_keys_file=api_keys_file,
            refresh_interval=args.refresh_interval,
            send_test_positions=False,  # PROD: never emit synthetic test positions
            connection_mode=RPCConnectionMode.RPC,
            start_server=True,
            websocket_host=args.host,
            websocket_port=port,
        )
        # Alert via Slack if we never become healthy (front door bound + core reachable) within the
        # grace window. Started before run() blocks; the front door binds inside run(), so the grace
        # window covers that. No stop_event — the daemon thread exits with the process.
        start_readiness_watchdog(
            app_name="vanta-ws",
            slack_notifier=slack_notifier,
            front_door_host=args.host,
            front_door_port=port,
            core_probe_ports=[ValiConfig.RPC_POSITIONMANAGER_PORT],
        )
        # Starts the asyncio websocket loop and BLOCKS. Handles KeyboardInterrupt
        # internally with a graceful shutdown (which signals only OUR namespace).
        server.run()
        return 0
    except KeyboardInterrupt:
        bt.logging.info("[vanta-ws] Interrupt received — shutdown complete.")
        return 0
    except Exception as e:
        bt.logging.error(f"[vanta-ws] FATAL: {type(e).__name__}: {e}")
        bt.logging.error(traceback.format_exc())
        try:
            slack_notifier.send_message(f"🔴 vanta-ws crashed: {type(e).__name__}: {e}", level="error")
        except Exception:
            pass
        raise
    finally:
        # Unlink our OWN shutdown-coordinator segment so it doesn't leak (the resource_tracker
        # "leaked shared_memory objects" warning). Safe: this app owns its namespace.
        ShutdownCoordinator.cleanup()


if __name__ == "__main__":
    sys.exit(main())
