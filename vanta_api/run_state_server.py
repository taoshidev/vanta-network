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

    # Initialize our own (isolated) shutdown namespace; clear any stale flag from a previous run of
    # THIS app, mirroring how the validator main process initializes.
    ShutdownCoordinator.initialize(reset_on_attach=True)

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

    # Keep SIGINT default (raises KeyboardInterrupt, breaking blocking sleeps during a slow startup).
    # Route SIGTERM (PM2's stop signal) to the same path so shutdown is graceful.
    def _sigterm_to_keyboard_interrupt(signum, _frame):
        bt.logging.info(f"[vanta-state] Received signal {signum} — interrupting for graceful shutdown.")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

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

        # Start the include-set (scoped start: skips the global RPC-port kill so we never take down
        # core's subtensor_ops or other core-held ports).
        orchestrator.start_state_servers(context)
        # pre_run_setup + daemons run HERE (position_manager lives in this tier now), not in core.
        # BOOT ORDER: vanta-state starts BEFORE core, so core-tier servers (elimination, perf_ledger)
        # may not be up yet. pre_run_setup's one-time order-corrections path can touch them, but it is
        # (a) date-gated (a no-op past TARGET_MS) and (b) wrapped in try/except inside pre_run_setup,
        # so an absent core degrades to "corrections skipped + logged", never a boot crash. If order
        # corrections are ever re-enabled with a future TARGET_MS, they will no-op until core is up
        # and re-apply on a later boot — acceptable for a one-time migration mechanism.
        orchestrator.call_pre_run_setup(perform_order_corrections=True)
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
    except KeyboardInterrupt:
        bt.logging.info("[vanta-state] Interrupt received — shutting down.")
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


if __name__ == "__main__":
    sys.exit(main())
