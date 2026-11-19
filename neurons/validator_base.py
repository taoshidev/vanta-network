import argparse
import concurrent.futures
import os

import bittensor as bt
import logging
from shared_objects.log import logger
logger.setLevel(logging.INFO)

import vanta_protocol
from vanta_protocol import GetPositionsRequest, GetPositionsResponse, SendSignalRequest, SendSignalResponse
from shared_objects.bt_config import add_logging_args, add_subtensor_args, add_wallet_args, build_config
from shared_objects.http_auth import AuthVerifier
from shared_objects.http_server import HttpServer, add_http_server_args, build_app, register_handler
from shared_objects.locks.subtensor_lock import get_subtensor_lock
from shared_objects.net_utils import get_external_ip


class ValidatorBase:
    def __init__(self, wallet, config, metagraph_client, asset_selection_client, subtensor=None, slack_notifier=None):
        self.wallet = wallet
        self.config = config
        self.metagraph_client = metagraph_client
        self.slack_notifier = slack_notifier
        self._asset_selection_client = asset_selection_client
        self.subtensor = subtensor

        # Create own EntityClient (forward compatibility - no parameter passing)
        from entity_management.entity_client import EntityClient
        self._entity_client = EntityClient(running_unit_tests=False)

        # Create own MinerAccountClient for receiving collateral updates
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        self._miner_account_client = MinerAccountClient(running_unit_tests=False)

        # Create own ContractClient for receiving account openings
        from vali_objects.contract.contract_client import ContractClient
        self._contract_client = ContractClient(running_unit_tests=False)

        # Dedicated thread pool for concurrent synchronous requests
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)

        self.http_server = None
        # --no-axon (serve_axon=False): this process does not serve the signed-request HTTP server —
        # the vanta-orders app does. Core-split runs with serve_axon=False; the default monolith and
        # vanta-orders run with serve_axon=True. The orders app ALWAYS serves it (that is its whole
        # job), even if --no-axon leaked into its args. self.http_server stays None otherwise so
        # guards can check it. (Flag name kept as --no-axon/serve_axon for operator continuity even
        # though this no longer wires up a bt.Axon — see shared_objects/http_server/.)
        if getattr(self.config, 'serve_axon', True) or getattr(self.config, 'orders_app', False):
            self.wire_http_server()
        else:
            logger.info("[INIT] --no-axon: signed-request HTTP server not served here (runs in vanta-orders app)")

        # Each hotkey gets a unique identity (UID) in the network for differentiation.
        # (Unconditional — only needs metagraph_client, which is available regardless of the server.)
        my_subnet_uid = self.metagraph_client.get_hotkeys().index(self.wallet.hotkey.ss58_address)
        logger.info(f"Running validator on uid: {my_subnet_uid}")

    def _receive_signal_sync(self, request: SendSignalRequest, caller_hotkey: str) -> SendSignalResponse:
        """
        Abstract method - must be implemented by child class.
        Handles incoming trading signals from miners.
        """
        raise NotImplementedError("Child class must implement _receive_signal_sync()")

    def _get_positions(self, request: GetPositionsRequest, caller_hotkey: str) -> GetPositionsResponse:
        """
        Abstract method - must be implemented by child class.
        Handles position inspection requests from miners.
        """
        raise NotImplementedError("Child class must implement _get_positions()")

    def is_caller_authorized(self, caller_hotkey: str) -> None:
        """Registered-hotkey authorization check (replaces bt.Axon's blacklist_fn), raising
        shared_objects.http_auth.Forbidden if not registered. Default implementation calls
        MetagraphClient over RPC — override this in a subclass (see neurons/validator.py's
        Validator, which answers from an in-process cache) if that per-request RPC round-trip
        matters for your throughput; every signed-request route pays this check once."""
        from shared_objects.http_auth import require_registered_hotkey
        require_registered_hotkey(caller_hotkey, self.metagraph_client)

    @staticmethod
    def get_config():
        # Step 2: Set up the configuration parser
        # This function initializes the necessary command-line arguments.
        # Using command-line arguments allows users to customize various miner settings.
        # NOTE: staticmethod so the standalone vanta-state entrypoint (run_state_server.py) can build
        # an identical config for the state servers without constructing a Validator (or a wallet).
        parser = argparse.ArgumentParser()
        # Set autosync to store true if flagged, otherwise defaults to False.
        parser.add_argument("--autosync", action='store_true',
                            help="Automatically sync order data with a validator trusted by Taoshi.")
        # Set run_generate to store true if flagged, otherwise defaults to False.
        parser.add_argument("--start-generate", action='store_true', dest='start_generate',
                            help="Run the request output generator.")

        # API Server related arguments
        parser.add_argument("--serve", action='store_true',
                            help="Start the API server for REST and WebSocket endpoints")
        # Default spawn_api=True is backward-safe: a code update under an OLD run.sh keeps today's
        # in-core spawning; only the split run.sh passes --no-spawn-api. See validator.py's spawn
        # gate for why --serve must remain on alongside this.
        parser.add_argument("--no-spawn-api", action='store_false', dest='spawn_api', default=True,
                            help="Do not spawn the REST/WebSocket servers from the validator core "
                                 "(they run as separate PM2 apps). Requires run.sh to launch them.")
        # vanta-state split: when set, core does NOT host the order-write state servers — they run in
        # the separate vanta-state PM2 app (run_state_server.py) so a core restart doesn't kill them.
        # Core starts only its own tier (subtensor_ops + contract + scoring + metagraph) and reaches
        # the state servers via RPC clients. Default False = today's single-process behavior
        # (backward-safe under an old run.sh). Requires run.sh to launch vanta-state first.
        parser.add_argument("--split-state", action='store_true', dest='split_state', default=False,
                            help="Run the order-write state servers in a separate vanta-state PM2 app "
                                 "instead of in-core. Requires run.sh to launch vanta-state.")
        # vanta-orders split: when set, this process does NOT serve the signed-request HTTP server or
        # run the HL tracker — they run in the separate vanta-orders PM2 app (run_orders_server.py).
        # Core passes this so a core restart doesn't drop order reception. Default True (serve_axon) =
        # today's behavior (HTTP server + HL in-process). Backward-safe polarity mirrors --no-spawn-api.
        parser.add_argument("--no-axon", action='store_false', dest='serve_axon', default=True,
                            help="Do not serve the signed-request HTTP server or run the HL tracker in "
                                 "this process (they run in the vanta-orders PM2 app). Requires run.sh "
                                 "to launch it.")
        # vanta-orders app role: this process IS the order-reception tier — it serves the HTTP server +
        # runs HL and is a pure RPC CLIENT of vanta-state (order-write servers) and core (metagraph,
        # scoring). It starts NO RPC servers, runs NO PositionSyncer/weight loop. Set by
        # run_orders_server.py. Implies serve_axon=True. Default False = not the orders app.
        parser.add_argument("--orders-app", action='store_true', dest='orders_app', default=False,
                            help="Run as the vanta-orders app (HTTP server + HL + order path, client of "
                                 "vanta-state/core; starts no servers). Set by run_orders_server.py.")
        # Wallet-less identity for the vanta-state app (run_state_server.py). Core ignores it (it has
        # a wallet); vanta-state passes the validator's ss58 so miner_account's ValidatorBroadcastBase
        # gets its identity without loading a keypair. See NeuronContext.validator_hotkey_override.
        parser.add_argument("--validator-hotkey", type=str, default=None, dest='validator_hotkey',
                            help="Validator hotkey ss58 for the wallet-less vanta-state app; core ignores it.")
        parser.add_argument("--api-host", type=str, default="127.0.0.1",
                            help="Host address for the API server")
        parser.add_argument("--api-rest-port", type=int, default=48888,
                            help="Port for the REST API server")
        parser.add_argument("--api-ws-port", type=int, default=8765,
                            help="Port for the WebSocket server")

        # (developer): Adds your custom arguments to the parser.
        # Adds override arguments for network and netuid.
        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")

        # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
        add_subtensor_args(parser)
        # Logging arguments (--logging.debug, --logging.trace, --logging.logging_dir)
        add_logging_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        add_wallet_args(parser)

        # Add Slack webhook arguments
        parser.add_argument(
            "--slack-webhook-url",
            type=str,
            default=None,
            help="Slack webhook URL for general notifications (optional)"
        )
        parser.add_argument(
            "--slack-error-webhook-url",
            type=str,
            default=None,
            help="Slack webhook URL for error notifications (optional, defaults to general webhook if not provided)"
        )
        # Adds signed-request HTTP server args i.e. --http.port ... (replaces bt.Axon.add_args)
        add_http_server_args(parser)
        # Activating the parser to read any command-line inputs.
        # To print help message, run python3 template/miner.py --help
        config = build_config(parser)
        if config.logging.debug:
            logger.setLevel(logging.DEBUG)
        if config.logging.trace:
            logger.setLevel(logging.DEBUG)

        # Step 3: Set up logging directory
        # Logging captures events for diagnosis or understanding miner's behavior.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "validator",
            )
        )
        return config

    def wire_http_server(self):
        """Builds and starts the signed-request FastAPI server, replacing bt.Axon.

        Authentication (signature verification, dual-scheme: new btauth/1 + legacy
        bittensor v10 bridge) runs as middleware (shared_objects/http_server/app.py);
        authorization (registered-hotkey check, this subnet's blacklist_fn-equivalent)
        runs per-route via shared_objects/http_server/routes.py's register_handler.
        """
        http_port = self.config.http_port
        http_external_port = self.config.http_external_port or http_port
        logger.info(f"setting http port [{http_port}]")
        logger.info(f"setting http external port [{http_external_port}]")

        verifier = AuthVerifier(
            self_hotkey_ss58=self.wallet.hotkey.ss58_address,
            legacy_enabled=getattr(self.config, 'legacy_auth_enabled', True),
            initial_hotkeys=self.metagraph_client.get_hotkeys(),
        )
        app = build_app(verifier, vanta_protocol.LEGACY_HASH_FIELDS)

        # Each business_fn already has the (request, caller_hotkey) -> response shape
        # register_handler expects — no internal model, no request/response conversion.
        route_business_fns = {
            "SendSignal": self._receive_signal_sync,
            "GetPositions": self._get_positions,
            "CollateralRecord": self._miner_account_client.receive_collateral_record,
            "AssetSelection": self._asset_selection_client.receive_asset_selection,
            "SubaccountRegistration": self._entity_client.receive_subaccount_registration,
            "EntityEndpointUpdate": self._entity_client.receive_entity_endpoint_update,
            "StandardMinerOpening": self._contract_client.receive_standard_miner_opening,
            "StandardMinerRecord": self._miner_account_client.receive_standard_miner_record,
            "AccountSizeLocks": self._miner_account_client.receive_size_locks,
        }
        for message_name, business_fn in route_business_fns.items():
            request_model, response_model = vanta_protocol.MESSAGE_TYPES[message_name]
            register_handler(
                app, message_name, request_model, response_model,
                business_fn, self._thread_pool, self.is_caller_authorized,
            )

        self.http_server = HttpServer(app, host=self.config.http_host, port=http_port)
        self.http_server.start()
        logger.info(f"Started signed-request HTTP server on {self.config.http_host}:{http_port}")

        self.auth_verifier = verifier
        self._start_trusted_hotkey_refresh_loop()

        # Publish our IP:port on chain so other nodes can discover this server via the metagraph's
        # per-neuron axon field (replaces axon.serve()). Use subtensor lock to prevent WebSocket
        # concurrency errors with the metagraph_updater thread.
        public_ip = get_external_ip()
        logger.info(f"Publishing ServeAxon on network {self.config.subtensor.network} "
                    f"netuid {self.config.netuid}: {public_ip}:{http_external_port}")
        with get_subtensor_lock():
            result = self.subtensor.execute(
                bt.ServeAxon(netuid=self.config.netuid, ip=public_ip, port=http_external_port),
                self.wallet,
            )
        if not result.success:
            logger.error(f"ServeAxon failed: {result.error}")

    def _start_trusted_hotkey_refresh_loop(self, interval_s: float = 30.0):
        """bittensor.http_auth.configure_verifier's trusted_sender_schemes mapping is
        process-local, so each process that runs an HttpServer (core monolith or the
        vanta-orders split) must periodically re-sync it against the metagraph itself —
        a newly-registered hotkey otherwise can't be verified until this refreshes."""
        import threading
        import time

        def _loop():
            while True:
                time.sleep(interval_s)
                try:
                    self.auth_verifier.refresh_trusted_hotkeys(self.metagraph_client.get_hotkeys())
                except Exception as e:
                    logger.warning(f"[http_server] failed to refresh trusted hotkeys: {e}")

        threading.Thread(target=_loop, daemon=True, name="auth-trusted-hotkey-refresh").start()
