import argparse
import asyncio
import concurrent.futures
import os
from typing import Tuple

import bittensor as bt
import logging
from shared_objects.log import logger
logger.setLevel(logging.INFO)

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

        self.wire_axon()

        # Each hotkey gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = self.metagraph_client.get_hotkeys().index(self.wallet.hotkey.ss58_address)
        logger.info(f"Running validator on uid: {my_subnet_uid}")

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
            logger.debug(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, synapse.dendrite.hotkey

        logger.debug(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, synapse.dendrite.hotkey

    def get_config(self):
        # Step 2: Set up the configuration parser
        # This function initializes the necessary command-line arguments.
        # Using command-line arguments allows users to customize various miner settings.
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
        # --no-spawn-api: stop the validator core from spawning the REST/WS servers as child
        # processes, so they can run as their own PM2 apps (vanta-rest / vanta-ws) with
        # independent deploy lifecycles. Defaults to SPAWNING (spawn_api=True) so a code update
        # under an old run.sh keeps today's behavior; the split run.sh passes --no-spawn-api.
        # NOTE: gates ONLY spawning. --serve stays on in core — it also gates the position-update
        # broadcasts that feed the (extracted) WS server (market_order_manager.py).
        parser.add_argument("--no-spawn-api", action='store_false', dest='spawn_api', default=True,
                            help="Do not spawn the REST/WebSocket servers from the validator core "
                                 "(they run as separate PM2 apps). Requires run.sh to launch them.")
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
        # Logging arguments (--logging.debug, --logging.trace, --logging.logging_dir)
        parser.add_argument("--logging.debug", action="store_true", default=False,
                            help="Turn on debugging information")
        parser.add_argument("--logging.trace", action="store_true", default=False,
                            help="Turn on trace level information")
        parser.add_argument("--logging.logging_dir", type=str, default=os.path.expanduser("~/.bittensor/miners"),
                            help="Logging default root directory.")
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

    def wire_axon(self):
        logger.info(f"setting port [{self.config.axon.port}]")
        logger.info(f"setting external port [{self.config.axon.external_port}]")
        self.axon = bt.Axon(
            wallet=self.wallet, port=self.config.axon.port, external_port=self.config.axon.external_port
        )
        logger.info(f"Axon {self.axon}")

        # Attach determines which functions are called when servicing a request.
        logger.info("Attaching forward function to axon.")

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
        logger.info(
            f"Serving attached axons on network:"
            f" {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        # Use subtensor lock to prevent WebSocket concurrency errors with metagraph_updater thread
        with get_subtensor_lock():
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Starts the miner's axon, making it active on the network.
        logger.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
