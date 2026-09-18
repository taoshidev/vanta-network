#!/usr/bin/env python3
"""
Production entrypoint for the vanta-orders tier (PM2 app: vanta-orders).

vanta-orders IS the order-reception tier: it serves the axon (the 6 miner-facing handlers) and runs
the Hyperliquid tracker CO-RESIDENT, and is a pure RPC CLIENT of vanta-state (order-write servers)
and core (metagraph, scoring, contract). It starts NO RPC servers and runs NO PositionSyncer/weight
loop — those stay in core. Moving order reception here means a CORE restart no longer drops orders.

v1 scope: single instance, no SO_REUSEPORT/overlap. Its OWN (rare) restarts blip order reception a
few seconds (same-port fast restart); a core restart is ~0s to orders.

PM2 owns supervision/restart. Start order (run.sh): vanta-state → core → vanta-orders (this app is
last so vanta-state + core's metagraph are up when it seeds dedup / primes the blacklist cache).

Usage (PM2 runs, roughly):
    python vanta_api/run_orders_server.py --netuid 8 --wallet.name <w> --wallet.hotkey <hk> \
        --axon.port <p> --subtensor.network <n> --orders-app
"""

import os

# Isolate this app's shutdown lifecycle from core (and vanta-state). ShutdownCoordinator binds its
# segment name at import time, so this MUST be set before importing neurons.validator (which
# transitively imports the coordinator). Without it, a core SIGTERM would flip the shared flag and
# kill vanta-orders — defeating the extraction. setdefault lets run.sh override via the environment.
os.environ.setdefault("VANTA_SHUTDOWN_SHM_NAME", "vanta_orders_shutdown")

import sys  # noqa: E402

# Force the orders-app role regardless of how we were invoked. Running this entrypoint WITHOUT
# --orders-app would construct a full monolith (starting every RPC server) and double-bind the whole
# port range — a dangerous footgun. Guarantee the flag is present.
if "--orders-app" not in sys.argv:
    sys.argv.append("--orders-app")

from neurons.validator import Validator  # noqa: E402  (after shutdown-namespace env is set)


def main() -> int:
    # Validator reads its role from config (--orders-app): starts no servers, wires the axon + HL,
    # and runs the lean order/keep-alive main loop (no PositionSyncer/weight loop).
    validator = Validator()
    validator.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
