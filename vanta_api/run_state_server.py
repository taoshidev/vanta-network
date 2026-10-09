#!/usr/bin/env python3
"""
Production entrypoint for the vanta-state tier (PM2 app: vanta-state).

vanta-state hosts the order-WRITE-critical RPC servers (ServerOrchestrator.VANTA_STATE_SERVERS)
as their own process so a core restart cannot kill them. subtensor_ops (wallet/chain),
contract/collateral, metagraph, and all scoring stay in core — this tier holds NO wallet and NO
chain signer (identity comes in as the validator hotkey ss58 STRING via --validator-hotkey; see
NeuronContext.validator_hotkey_override).

PM2 owns process supervision/restart, so spawn_process()/enable_auto_restart (the in-validator
supervision model) are intentionally NOT used here.

Start order matters: run.sh launches vanta-state FIRST, then core. Core reaches these servers via
its directly-instantiated RPC clients (they connect by service-name+port regardless of which
process spawned the server).

Usage (PM2 runs, roughly):
    python vanta_api/run_state_server.py --netuid 8 --wallet.name <w> --wallet.hotkey <hk> \
        --validator-hotkey <ss58> --serve
"""

import os

# Isolate this app's shutdown lifecycle from vanta-core. ShutdownCoordinator binds its segment name
# at import time, so this MUST be set before the imports below (which transitively import it).
# Without this, a core SIGTERM would flip the shared flag and kill vanta-state — defeating the whole
# extraction. setdefault lets run.sh override via the PM2 environment.
os.environ.setdefault("VANTA_SHUTDOWN_SHM_NAME", "vanta_state_shutdown")

import signal  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import bittensor as bt  # noqa: E402

from neurons.validator_base import ValidatorBase  # noqa: E402  (static get_config for config parity)
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, NeuronContext  # noqa: E402
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator  # noqa: E402
from shared_objects.slack_notifier import SlackNotifier  # noqa: E402
from vali_objects.utils.vali_utils import ValiUtils  # noqa: E402
from vali_objects.vali_config import ValiConfig  # noqa: E402
from vanta_api.server_readiness import start_readiness_watchdog  # noqa: E402

# Daemons for the vanta-state tier's servers. The core-tier daemons (perf_ledger, elimination,
# challenge_period, debt_ledger, mdd_checker, core_outputs, miner_statistics, weight_calculator,
# entity) are started by core (neurons/validator.py under --split-state). These four are the state
# servers that have deferred daemons; the rest of the include-set (common_data, position_lock,
# live_price_fetcher, market_order) either have no deferred daemon or start it at spawn.
STATE_DAEMONS = ['miner_account', 'position_manager', 'limit_order', 'entity_collateral']

# Backstop that force-exits if graceful shutdown hangs. Kept under this tier's PM2 kill_timeout
# (10s, set in run.sh) so that we SIGKILL our own server subprocesses before PM2 SIGKILLs us; a
# SIGKILL of this process alone would orphan them and leak their RPC ports. The normal path
# (shutdown_all_servers: 4s grace + 1s reap) finishes well inside it.
GRACEFUL_SHUTDOWN_DEADLINE_S = 9


def main() -> int:
    # Reuse the validator's config parser for exact parity with core (netuid, wallet.*, serve,
    # subtensor.*, slack, --validator-hotkey). Wallet args are parsed as STRINGS only — no wallet is
    # loaded here.
    config = ValidatorBase.get_config()
    bt.logging.enable_info()

    is_mainnet = (config.netuid == 8)
    validator_hotkey = getattr(config, 'validator_hotkey', None)
    if not validator_hotkey:
        # Required: without it, miner_account's ValidatorBroadcastBase would fall back to loading a
        # wallet from config — defeating the wallet-less guarantee. Fail loud rather than silently
        # pull a keypair into vanta-state.
        bt.logging.error("[vanta-state] --validator-hotkey <ss58> is required (wallet-less identity). Exiting.")
        return 1

    alert_hotkey = validator_hotkey or f"vanta-state@{socket.gethostname()}"

    # Initialize our own (isolated) shutdown namespace in a fresh segment, mirroring the validator
    # main process. Never reset a previous run's segment in place: orphaned children of a killed
    # run still read it, and a reset would bring them back to serving on our ports.
    ShutdownCoordinator.initialize_fresh()

    # Secrets are needed by live_price_fetcher (API keys). Same source as core.
    secrets = ValiUtils.get_secrets()
    if secrets is None:
        bt.logging.warning("[vanta-state] No secrets found (validation/miner_secrets.json) — "
                           "live_price_fetcher may fail to start.")

    # webhook_url=None lets SlackNotifier fall back to the SLACK_WEBHOOK_URL env var.
    slack_notifier = SlackNotifier(hotkey=alert_hotkey, webhook_url=getattr(config, 'slack_webhook_url', None))

    bt.logging.info(
        f"[vanta-state] Starting state tier: netuid={config.netuid} is_mainnet={is_mainnet} "
        f"validator_hotkey={validator_hotkey} shutdown_ns={os.environ.get('VANTA_SHUTDOWN_SHM_NAME')}"
    )

    # WALLET-LESS context: wallet=None + validator_hotkey_override supplies identity to
    # miner_account's ValidatorBroadcastBase without loading a keypair.
    context = NeuronContext(
        slack_notifier=slack_notifier,
        config=config,
        wallet=None,
        secrets=secrets,
        is_mainnet=is_mainnet,
        validator_hotkey_override=validator_hotkey,
    )

    orchestrator = ServerOrchestrator.get_instance()
    stop = threading.Event()

    # Arms the backstop alarm, then sets the shared shutdown flag, which the spawned state
    # subprocesses poll to leave their serve loop and shut down gracefully. The main loop also
    # polls the flag every second, so this handler deliberately does NOT call stop.set():
    # Event.set() takes a non-reentrant lock that the interrupted main thread may already hold
    # inside stop.wait(). SIGINT is handled the same way, so it no longer raises KeyboardInterrupt;
    # startup checks the flag between phases instead.
    def _signal_handler(signum, _frame):
        if ShutdownCoordinator.is_shutdown():
            return
        signal.alarm(GRACEFUL_SHUTDOWN_DEADLINE_S)  # arm the backstop before anything can block
        bt.logging.info(f"[vanta-state] Received signal {signum} — initiating graceful shutdown.")
        ShutdownCoordinator.signal_shutdown(f"vanta-state received signal {signum}")

    def _alarm_handler(_signum, _frame):
        # Can fire anywhere, including inside shutdown_all_servers, so nothing here takes an
        # orchestrator lock. Leave via os._exit, not sys.exit: our server subprocesses ignore
        # SIGTERM, so multiprocessing's exit hook (terminate + join() with no timeout) would hang
        # on any survivor until PM2's SIGKILL orphaned it.
        bt.logging.error("[vanta-state] Graceful shutdown exceeded deadline — killing server "
                         "subprocesses and force-exiting.")
        ServerOrchestrator.force_kill_child_processes()
        ShutdownCoordinator.cleanup()
        os._exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGALRM, _alarm_handler)

    try:
        # Run on-disk state migrations BEFORE any server loads that state. vanta-state starts
        # FIRST (run.sh ordering) and its servers (position_manager, limit_order, miner_account)
        # load exactly the files migrations rewrite — if core ran them instead (as the monolith
        # does), it would migrate the files AFTER this tier already loaded pre-migration data,
        # and this tier's next save would clobber the migrated file while migrations_completed.txt
        # marks it done forever. Core skips migrations under --split-state for this reason.
        from runnable.run_migrations import main as run_migrations
        bt.logging.info("[vanta-state] Checking for pending migrations (state tier owns them under --split-state)...")
        if not run_migrations():
            bt.logging.error("[vanta-state] Migration failed. Starting state servers without executing migrations")
        else:
            bt.logging.info("[vanta-state] Migrations completed successfully.")
        if ShutdownCoordinator.is_shutdown():
            return 0  # stopped during migrations; don't start servers just to tear them down

        # Start the include-set (scoped start: skips the global RPC-port kill so we never take down
        # core's subtensor_ops or other core-held ports).
        orchestrator.start_state_servers(context)
        if ShutdownCoordinator.is_shutdown():
            return 0
        # pre_run_setup + daemons run HERE (position_manager lives in this tier now), not in core.
        # BOOT ORDER: vanta-state starts BEFORE core, so core-tier servers (elimination, perf_ledger)
        # may not be up yet. pre_run_setup's one-time order-corrections path can touch them, but it is
        # (a) date-gated (a no-op past TARGET_MS) and (b) wrapped in try/except inside pre_run_setup,
        # so an absent core degrades to "corrections skipped + logged", never a boot crash. If order
        # corrections are ever re-enabled with a future TARGET_MS, they will no-op until core is up
        # and re-apply on a later boot — acceptable for a one-time migration mechanism.
        orchestrator.call_pre_run_setup(perform_order_corrections=True)
        if ShutdownCoordinator.is_shutdown():
            return 0
        orchestrator.start_server_daemons(STATE_DAEMONS)
        bt.logging.success("[vanta-state] State servers up and daemons started. Blocking until signal.")

        # Alert via Slack if we never become healthy (our own listener bound + core reachable) within
        # the grace window. front door = position_manager RPC; core presence = subtensor_ops port.
        start_readiness_watchdog(
            app_name="vanta-state",
            slack_notifier=slack_notifier,
            front_door_host="127.0.0.1",
            front_door_port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            core_probe_ports=[ValiConfig.RPC_WEIGHT_SETTER_PORT],  # subtensor_ops (core) presence
            stop_event=stop,
        )

        # Block until interrupted or our own namespace signals shutdown.
        while not ShutdownCoordinator.is_shutdown():
            stop.wait(1.0)
        return 0
    except Exception as e:
        bt.logging.error(f"[vanta-state] FATAL: {type(e).__name__}: {e}")
        bt.logging.error(traceback.format_exc())
        try:
            slack_notifier.send_message(f"🔴 vanta-state crashed: {type(e).__name__}: {e}", level="error")
        except Exception:
            pass
        raise
    finally:
        stop.set()  # stop the readiness watchdog thread
        try:
            orchestrator.shutdown_all_servers()  # isolated namespace — does not touch core
        except Exception as e:
            bt.logging.warning(f"[vanta-state] error during shutdown: {e}")
        # Unlink our OWN coordinator segment so it doesn't leak; safe because this app owns its namespace.
        ShutdownCoordinator.cleanup()
        signal.alarm(0)  # teardown finished in time; cancel the backstop


if __name__ == "__main__":
    sys.exit(main())
