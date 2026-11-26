# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import argparse
import json
import logging
import os

from typing import Dict, List, Optional
from google.cloud import secretmanager

from vali_objects.exceptions.vali_bkp_file_missing_exception import (
    ValiFileMissingException,
)
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from shared_objects.log import logger
from shared_objects.bt_config import add_logging_args, add_subtensor_args, add_wallet_args, build_config
from shared_objects.http_server import add_http_server_args

class ValiUtils:
    @staticmethod
    def get_config():
        # Set up the configuration parser. Using command-line arguments allows users to customize
        # various validator settings.
        # NOTE: standalone function (not a Validator method) so the standalone vanta-state entrypoint
        # (run_state_server.py) can build an identical config for the state servers without
        # constructing a Validator (or a wallet).
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
        # today's behavior (HTTP server + HL in-process).
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
        parser.add_argument("--collateral.chain_endpoint", type=str, default=None,
                             help="ws:// or wss:// URL of a self-hosted subtensor node for collateral operations "
                                  "only (e.g. ws://10.0.0.5:9944). The public endpoints are used while it is unhealthy.")

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
        # To print help message, run python3 neurons/validator.py --help
        config = build_config(parser)
        if config.logging.debug:
            logger.setLevel(logging.DEBUG)
        if config.logging.trace:
            logger.setLevel(logging.DEBUG)

        # Set up logging directory. Logging captures events for diagnosis or understanding
        # the validator's behavior.
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

    @staticmethod
    def get_secrets(running_unit_tests=False, secrets_path=None) -> Dict:
        """
        return dict of secret names and values
        """
        # wrapping here to allow simpler error handling & original for other error handling
        if running_unit_tests:
            return {'polygon_apikey': "", 'tiingo_apikey': ""}

        ans = {}
        try:
            if secrets_path is None:
                secrets = ValiBkpUtils.get_file(ValiBkpUtils.get_secrets_dir())
            else:
                secrets = ValiBkpUtils.get_file(secrets_path)
            ans = json.loads(secrets)
            if running_unit_tests:
                for k in ['polygon_apikey', 'tiingo_apikey']:
                    if k not in ans:
                        ans[k] = ""
        except FileNotFoundError:
            raise ValiFileMissingException("Vali secrets file is missing")

        return ans

    @staticmethod
    def get_secret(secret_name: str, secrets_path: str=None) -> Optional[str]:
        """
        Get secret with fallback to local secrets

        Args:
            secret_name (str): name of secret

        Returns:
            str: secret or None if not found
        """
        secret = ValiUtils._get_gcp_secret(secret_name, secrets_path)
        if secret is not None:
            return secret

        secret = ValiUtils.get_secrets(secrets_path=secrets_path).get(secret_name)
        if secret is not None:
            logger.info(f"{secret_name} retrieved from local secrets file")
        return secret

    @staticmethod
    def _get_gcp_secret(secret_name: str, secrets_path: str=None) -> Optional[str]:
        """
        Get vault password from Google Cloud Secret Manager.

        Args:
            secret_name (str): name of secret

        Returns:
            str: secret or None if not found
        """
        try:
            gcp_secret_manager_client = secretmanager.SecretManagerServiceClient()
            secrets = ValiUtils.get_secrets(secrets_path=secrets_path)

            secret_path = gcp_secret_manager_client.secret_version_path(
                secrets.get('gcp_project_name'), secrets.get(secret_name), "latest"
            )
            response = gcp_secret_manager_client.access_secret_version(name=secret_path)
            secret = response.payload.data.decode()

            if secret:
                logger.info(f"{secret_name} retrieved from Google Cloud Secret Manager")
                return secret
            else:
                logger.debug(f"{secret_name} not found in Google Cloud Secret Manager")
                return None
        except Exception as e:
            logger.debug(f"Failed to retrieve {secret_name} from Google Cloud: {e}")

    @staticmethod
    def get_taoshi_ts_secrets():
        secrets = ValiBkpUtils.get_file(ValiBkpUtils.get_taoshi_api_keys_file_location())
        ans = json.loads(secrets)
        return ans

    @staticmethod
    def get_vali_json_file(vali_dir: str, key: str = None) -> List | Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_file(vali_dir)
            if key is not None:
                return json.loads(secrets)[key]
            else:
                return json.loads(secrets)
        except FileNotFoundError:
            print(f"no vali json file [{vali_dir}], continuing")
            return []
        
    @staticmethod
    def get_vali_json_file_dict(vali_dir: str, key: str = None) -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        try:
            secrets = ValiBkpUtils.get_file(vali_dir)
            if key is not None:
                return json.loads(secrets)[key]
            else:
                return json.loads(secrets)
        except FileNotFoundError:
            print(f"no vali json file [{vali_dir}], continuing")
            return {}

    @staticmethod
    def is_mothership_wallet(wallet, is_testnet=False) -> bool:
        """
        Determine if the given wallet is the mothership validator.

        This is the single source of truth for mothership identification.
        Compares the wallet's hotkey against the configured MOTHERSHIP_HOTKEY.

        Args:
            wallet: Bittensor wallet object with hotkey attribute

        Returns:
            bool: True if wallet's hotkey matches MOTHERSHIP_HOTKEY

        Examples:
            >>> from vali_objects.utils.vali_utils import ValiUtils
            >>> wallet = bt.wallet(config=config)
            >>> if ValiUtils.is_mothership_wallet(wallet):
            >>>     # This is the mothership validator
            >>>     logger.info("Running as mothership")
        """
        if not wallet or not hasattr(wallet, 'hotkey'):
            return False
        return ValiUtils.is_mothership_hotkey(wallet.hotkey.ss58_address, is_testnet)

    @staticmethod
    def is_mothership_hotkey(hotkey_ss58: str, is_testnet=False) -> bool:
        """
        Mothership check from a hotkey ss58 STRING (no wallet/keypair needed).

        Used by wallet-less processes (vanta-state's miner_account) that receive the validator
        hotkey as a string via NeuronContext.validator_hotkey_override rather than loading a wallet.
        is_mothership_wallet() delegates here.
        """
        from vali_objects.vali_config import ValiConfig
        if not hotkey_ss58:
            return False
        if is_testnet:
            return hotkey_ss58 == ValiConfig.MOTHERSHIP_HOTKEY_TESTNET
        return hotkey_ss58 == ValiConfig.MOTHERSHIP_HOTKEY
