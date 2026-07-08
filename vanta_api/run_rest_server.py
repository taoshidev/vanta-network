#!/usr/bin/env python3
"""
Production entrypoint for the standalone Validator REST server (PM2 app: vanta-rest).

PM2 owns process supervision and restart, so `spawn_process()` / `enable_auto_restart`
(the in-validator supervision model) are intentionally NOT used here.

Usage (PM2 runs, roughly):
    python vanta_api/run_rest_server.py --netuid 8
"""

import os

# Isolate this app's shutdown lifecycle from vanta-core. ShutdownCoordinator binds its segment
# name at import time, so this MUST be set before the imports below (which transitively import it).
# setdefault lets run.sh override via the PM2 environment. See shutdown_coordinator.py for why.
os.environ.setdefault("VANTA_SHUTDOWN_SHM_NAME", "vanta_rest_shutdown")

import argparse  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import bittensor as bt  # noqa: E402

from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator  # noqa: E402
from shared_objects.slack_notifier import SlackNotifier  # noqa: E402
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils  # noqa: E402
from vali_objects.vali_config import RPCConnectionMode, ValiConfig  # noqa: E402
from vanta_api.server_readiness import start_readiness_watchdog  # noqa: E402
from vanta_api.validator_rest_server import ValidatorRestServer  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone Validator REST server (vanta-rest).")
    parser.add_argument("--netuid", type=int, default=8,
                        help="Subnet netuid; is_mainnet is derived as (netuid == 8). Default: 8.")
    parser.add_argument("--api-keys-file", type=str, default=None,
                        help="Path to the API keys JSON file. Default: ValiBkpUtils.get_api_keys_file_path().")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the Flask HTTP server. Default: 0.0.0.0 (external reachability).")
    parser.add_argument("--port", type=int, default=None,
                        help=f"Port for the Flask HTTP server. Default: ValiConfig.REST_API_PORT ({ValiConfig.REST_API_PORT}).")
    parser.add_argument("--refresh-interval", type=int, default=15,
                        help="Seconds between API-key file refresh checks. Default: 15.")
    parser.add_argument("--slack-webhook-url", type=str, default=None,
                        help="Slack webhook for health alerts. If omitted, SlackNotifier falls back to the "
                             "SLACK_WEBHOOK_URL env var (same source as vanta-core).")
    parser.add_argument("--validator-hotkey", type=str, default=None,
                        help="Label used to identify this server in Slack alerts. No wallet is required; "
                             "pass the validator's ss58 to match core alerts, else a hostname label is used.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    bt.logging.enable_info()

    is_mainnet = (args.netuid == 8)
    api_keys_file = args.api_keys_file or ValiBkpUtils.get_api_keys_file_path()
    port = args.port if args.port is not None else ValiConfig.REST_API_PORT
    alert_hotkey = args.validator_hotkey or f"vanta-rest@{socket.gethostname()}"

    # Initialize our own (isolated) shutdown namespace; clear any stale flag from a
    # previous run of THIS app, mirroring how the validator main process initializes.
    ShutdownCoordinator.initialize(reset_on_attach=True)

    if not os.path.exists(api_keys_file):
        # Unlike the test __main__, a prod server does NOT fabricate keys — surface the misconfig.
        bt.logging.warning(f"[vanta-rest] API keys file not found at {api_keys_file} — "
                            f"the server will start but reject authenticated requests until it exists.")

    # webhook_url=None lets SlackNotifier fall back to the SLACK_WEBHOOK_URL env var.
    slack_notifier = SlackNotifier(hotkey=alert_hotkey, webhook_url=args.slack_webhook_url)

    bt.logging.info(
        f"[vanta-rest] Starting standalone REST server: host={args.host} port={port} "
        f"netuid={args.netuid} is_mainnet={is_mainnet} api_keys_file={api_keys_file} "
        f"shutdown_ns={os.environ.get('VANTA_SHUTDOWN_SHM_NAME')}"
    )

    server = None
    stop = threading.Event()

    # Keep SIGINT default: it raises KeyboardInterrupt, which breaks the blocking retry sleeps
    # during a slow startup — swallowing it into an Event makes the process unkillable
    # mid-construction. Route SIGTERM (PM2's stop signal) to the same path.
    def _sigterm_to_keyboard_interrupt(signum, _frame):
        bt.logging.info(f"[vanta-rest] Received signal {signum} — interrupting for graceful shutdown.")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

    try:
        # The constructor starts Flask (daemon thread) + the RPC health server, so do NOT call
        # server.run() afterward — it would re-bind the HTTP port.
        server = ValidatorRestServer(
            api_keys_file=api_keys_file,
            refresh_interval=args.refresh_interval,
            connection_mode=RPCConnectionMode.RPC,
            start_server=True,
            flask_host=args.host,
            flask_port=port,
            is_mainnet=is_mainnet,
            slack_notifier=slack_notifier,  # flows via **kwargs to RPCServerBase health/hang alerts
        )
        bt.logging.success(f"[vanta-rest] REST server up on {args.host}:{port}. Blocking until signal.")

        # Alert via Slack if we never become healthy (front door bound + core reachable) within
        # the grace window — makes a stuck spin-up observable despite the lazy-client tolerance.
        start_readiness_watchdog(
            app_name="vanta-rest",
            slack_notifier=slack_notifier,
            front_door_host=args.host,
            front_door_port=port,
            core_probe_ports=[ValiConfig.RPC_POSITIONMANAGER_PORT, ValiConfig.RPC_COREOUTPUTS_PORT],
            stop_event=stop,
        )

        # Block until interrupted or our own namespace signals shutdown. Nothing sets `stop` here —
        # signals arrive as KeyboardInterrupt (caught below); `stop` is only the watchdog's kill
        # switch (set in finally). stop.wait() is just an interruptible sleep.
        while not ShutdownCoordinator.is_shutdown():
            stop.wait(1.0)
        return 0
    except KeyboardInterrupt:
        bt.logging.info("[vanta-rest] Interrupt received — shutting down.")
        return 0
    except Exception as e:
        bt.logging.error(f"[vanta-rest] FATAL: {type(e).__name__}: {e}")
        bt.logging.error(traceback.format_exc())
        try:
            slack_notifier.send_message(f"🔴 vanta-rest crashed: {type(e).__name__}: {e}", level="error")
        except Exception:
            pass
        raise
    finally:
        stop.set()  # stop the readiness watchdog thread
        if server is not None:
            try:
                server.shutdown()  # isolated namespace — safe, does not touch core
            except Exception as e:
                bt.logging.warning(f"[vanta-rest] error during shutdown: {e}")
        # Unlink our OWN coordinator segment so it doesn't leak (resource_tracker warning);
        # safe because this app owns its namespace.
        ShutdownCoordinator.cleanup()


if __name__ == "__main__":
    sys.exit(main())
