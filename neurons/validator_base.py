import argparse
import os
import time
import traceback
from typing import Tuple

import bittensor as bt
bt.logging.enable_info()

import template
from time_util.time_util import timeme


class ValidatorBase:
    def __init__(self, wallet, config, metagraph, p2p_syncer, asset_selection_client, subtensor=None, slack_notifier=None):
        self.wallet = wallet
        self.config = config
        self.metagraph_server = metagraph
        self.slack_notifier = slack_notifier
        self.p2p_syncer = p2p_syncer
        self.asset_selection_client = asset_selection_client
        self.subtensor = subtensor

        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.utils.contract_server import ContractClient
        self._contract_client = ContractClient(running_unit_tests=False)

        self.wire_axon()

        # Each hotkey gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = self.metagraph_server.get_hotkeys().index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    def receive_signal(self, synapse: template.protocol.SendSignal) -> template.protocol.SendSignal:
        """
        Abstract method - must be implemented by child class.
        Handles incoming trading signals from miners.
        """
        raise NotImplementedError("Child class must implement receive_signal()")

    def get_positions(self, synapse: template.protocol.GetPositions) -> template.protocol.GetPositions:
        """
        Abstract method - must be implemented by child class.
        Handles position inspection requests from miners.
        """
        raise NotImplementedError("Child class must implement get_positions()")

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
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.wallet.add_args(parser)

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
        bt.axon.add_args(parser)
        # Activating the parser to read any command-line inputs.
        # To print help message, run python3 template/miner.py --help
        config = bt.config(parser)
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
        self.axon = bt.axon(
            wallet=self.wallet, port=self.config.axon.port, external_port=self.config.axon.external_port
        )
        bt.logging.info(f"Axon {self.axon}")

        # Attach determines which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to axon.")

        def rs_blacklist_fn(synapse: template.protocol.SendSignal) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_server)

        def gp_blacklist_fn(synapse: template.protocol.GetPositions) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_server)

        def rc_blacklist_fn(synapse: template.protocol.ValidatorCheckpoint) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_server)

        def cr_blacklist_fn(synapse: template.protocol.CollateralRecord) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_server)

        def as_blacklist_fn(synapse: template.protocol.AssetSelection) -> Tuple[bool, str]:
            return self.blacklist_fn(synapse, self.metagraph_server)

        self.axon.attach(
            forward_fn=self.receive_signal,
            blacklist_fn=rs_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.get_positions,
            blacklist_fn=gp_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.p2p_syncer.receive_checkpoint,
            blacklist_fn=rc_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.contract_manager.receive_collateral_record,
            blacklist_fn=cr_blacklist_fn
        )
        self.axon.attach(
            forward_fn=self.asset_selection_client.receive_asset_selection,
            blacklist_fn=as_blacklist_fn
        )

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving attached axons on network:"
            f" {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Starts the miner's axon, making it active on the network.
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()