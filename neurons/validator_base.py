import argparse
import asyncio
import concurrent.futures
import os
from typing import Tuple

import bittensor as bt
bt.logging.enable_info()

import template
from time_util.time_util import timeme
from shared_objects.locks.subtensor_lock import get_subtensor_lock


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

        # Dedicated thread pool for concurrent synchronous requests
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)

        # --no-axon (serve_axon=False): this process does not serve the axon — the vanta-orders app
        # does. Core-split runs with serve_axon=False; the default monolith and vanta-orders run with
        # serve_axon=True. The orders app ALWAYS serves the axon (that is its whole job), even if
        # --no-axon leaked into its args. self.axon stays None otherwise so guards can check it.
        if getattr(self.config, 'serve_axon', True) or getattr(self.config, 'orders_app', False):
            self.wire_axon()
        else:
            self.axon = None
            bt.logging.info("[INIT] --no-axon: axon not served here (runs in vanta-orders app)")

        # Each hotkey gets a unique identity (UID) in the network for differentiation.
        # (Unconditional — only needs metagraph_client, which is available regardless of the axon.)
        my_subnet_uid = self.metagraph_client.get_hotkeys().index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    def _receive_signal_sync(self, synapse: template.protocol.SendSignal) -> template.protocol.SendSignal:
        """
        Abstract method - must be implemented by child class.
        Handles incoming trading signals from miners.
        """
        raise NotImplementedError("Child class must implement _receive_signal_sync()")

    async def receive_signal(self, synapse: template.protocol.SendSignal) -> template.protocol.SendSignal:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self._receive_signal_sync, synapse)

    def _get_positions(self, synapse: template.protocol.GetPositions) -> template.protocol.GetPositions:
        """
        Abstract method - must be implemented by child class.
        Handles position inspection requests from miners.
        """
        raise NotImplementedError("Child class must implement _get_positions()")

    async def get_positions(self, synapse: template.protocol.GetPositions) -> template.protocol.GetPositions:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self._get_positions, synapse)

    async def receive_collateral_record(self, synapse: template.protocol.CollateralRecord) -> template.protocol.CollateralRecord:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self._miner_account_client.receive_collateral_record, synapse)

    async def receive_asset_selection(self, synapse: template.protocol.AssetSelection) -> template.protocol.AssetSelection:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self._asset_selection_client.receive_asset_selection, synapse)

    async def receive_subaccount_registration(self, synapse: template.protocol.SubaccountRegistration) -> template.protocol.SubaccountRegistration:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, self._entity_client.receive_subaccount_registration, synapse)

    @timeme
    def blacklist_fn(self, synapse, metagraph) -> Tuple[bool, str]:
        miner_hotkey = synapse.dendrite.hotkey
        if not metagraph.has_hotkey(miner_hotkey):
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

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
        # vanta-orders split: when set, this process does NOT serve the axon or run the HL tracker —
        # they run in the separate vanta-orders PM2 app (run_orders_server.py). Core passes this so a
        # core restart doesn't drop order reception. Default True (serve_axon) = today's behavior
        # (axon + HL in-process). Backward-safe polarity mirrors --no-spawn-api.
        parser.add_argument("--no-axon", action='store_false', dest='serve_axon', default=True,
                            help="Do not serve the axon or run the HL tracker in this process "
                                 "(they run in the vanta-orders PM2 app). Requires run.sh to launch it.")
        # vanta-orders app role: this process IS the order-reception tier — it serves the axon + runs
        # HL and is a pure RPC CLIENT of vanta-state (order-write servers) and core (metagraph,
        # scoring). It starts NO RPC servers, runs NO PositionSyncer/weight loop. Set by
        # run_orders_server.py. Implies serve_axon=True. Default False = not the orders app.
        parser.add_argument("--orders-app", action='store_true', dest='orders_app', default=False,
                            help="Run as the vanta-orders app (axon + HL + order path, client of "
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
        bt.Subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.Wallet.add_args(parser)

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
        # Adds axon specific arguments i.e. --axon.port ...
        bt.Axon.add_args(parser)
        # Activating the parser to read any command-line inputs.
        # To print help message, run python3 template/miner.py --help
        config = bt.Config(parser)
        if config.logging.debug:
            bt.logging.enable_debug()
        if config.logging.trace:
            bt.logging.enable_trace()

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

    def wire_axon(self):
        bt.logging.info(f"setting port [{self.config.axon.port}]")
        bt.logging.info(f"setting external port [{self.config.axon.external_port}]")
        self.axon = bt.Axon(
            wallet=self.wallet, port=self.config.axon.port, external_port=self.config.axon.external_port
        )
        bt.logging.info(f"Axon {self.axon}")

        # Attach determines which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to axon.")

        def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_client)

        def gp_blacklist_fn(synapse: template.protocol.GetPositions) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_client)

        def cr_blacklist_fn(synapse: template.protocol.CollateralRecord) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_client)

        def as_blacklist_fn(synapse: template.protocol.AssetSelection) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_client)

        def sr_blacklist_fn(synapse: template.protocol.SubaccountRegistration) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_client)

        def eeu_blacklist_fn(synapse: template.protocol.EntityEndpointUpdate) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_client)

        self.axon.attach(
            forward_fn=self.receive_signal,
            blacklist_fn=rs_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.get_positions,
            blacklist_fn=gp_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.receive_collateral_record,
            blacklist_fn=cr_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.receive_asset_selection,
            blacklist_fn=as_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.receive_subaccount_registration,
            blacklist_fn=sr_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self._entity_client.receive_entity_endpoint_synapse,
            blacklist_fn=eeu_blacklist_fn
        )

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving attached axons on network:"
            f" {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        # Use subtensor lock to prevent WebSocket concurrency errors with metagraph_updater thread
        with get_subtensor_lock():
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Starts the miner's axon, making it active on the network.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
