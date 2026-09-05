import argparse
import asyncio
import concurrent.futures
import os
import threading
from typing import Tuple

import bittensor as bt
import logging
from shared_objects.log import logger
logger.setLevel(logging.INFO)

import template
from time_util.time_util import timeme
from shared_objects.locks.subtensor_lock import get_subtensor_lock
from shared_objects.bt_config import (
    add_subtensor_args, add_wallet_args, add_axon_args, add_logging_args,
    build_config, make_wallet,
)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal Axon replacement
#
# bt.Axon, bt.Dendrite, and bt.Synapse do not exist in bt11.
# LocalAxon provides a thin Flask-based HTTP server that:
#   • exposes /axon/<SynapseClass> POST routes for each registered handler
#   • verifies the caller's hotkey with bt.http_auth.verify
#   • sets synapse.dendrite.hotkey from the verified hotkey
#   • applies the blacklist function before calling the forward handler
#   • registers the validator endpoint on chain via bt.ServeAxon
# ──────────────────────────────────────────────────────────────────────────────

class LocalAxon:
    """Flask-based Axon replacement for bt11 migration."""

    def __init__(self, wallet, port: int, external_port: int | None = None):
        self.wallet = wallet
        self.port = port
        self.external_port = external_port or port
        self._handlers: dict[str, dict] = {}  # synapse_class_name → {forward_fn, blacklist_fn}
        self._app = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def attach(self, forward_fn, blacklist_fn=None, priority_fn=None):
        """Register a forward handler (and optional blacklist/priority) for a synapse type."""
        import inspect
        sig = inspect.signature(forward_fn)
        params = list(sig.parameters.values())
        if not params:
            logger.warning(f"[LocalAxon] Could not determine synapse type for {forward_fn}")
            return
        synapse_class = params[0].annotation
        if synapse_class is inspect.Parameter.empty or synapse_class is None:
            logger.warning(f"[LocalAxon] No type annotation on first param of {forward_fn}")
            return
        class_name = synapse_class.__name__ if hasattr(synapse_class, '__name__') else str(synapse_class)
        self._handlers[class_name] = {
            "forward_fn": forward_fn,
            "blacklist_fn": blacklist_fn,
            "synapse_class": synapse_class,
        }
        logger.info(f"[LocalAxon] Registered handler for {class_name}")

    # ------------------------------------------------------------------
    def _build_flask_app(self):
        from flask import Flask, request as flask_request, jsonify
        app = Flask(__name__)

        for class_name, info in self._handlers.items():
            # Capture via default arg to avoid closure-over-loop pitfall
            def make_route(cn=class_name, inf=info):
                def route_fn():
                    body = flask_request.get_data()
                    sender_hotkey = ""
                    try:
                        caller = bt.http_auth.verify(
                            dict(flask_request.headers), body,
                            method=flask_request.method,
                            path=flask_request.path,
                            self_hotkey_ss58=self.wallet.hotkey.ss58_address,
                            require_receiver=False,
                        )
                        sender_hotkey = caller.hotkey_ss58
                    except Exception as e:
                        logger.warning(f"[LocalAxon] Signature verification failed for {cn}: {e}")
                        sender_hotkey = flask_request.headers.get("x-bittensor-hotkey", "")

                    synapse_cls = inf["synapse_class"]
                    try:
                        import json as _json
                        synapse = synapse_cls.model_validate_json(body)
                    except Exception as e:
                        logger.error(f"[LocalAxon] Failed to deserialize {cn}: {e}")
                        return jsonify({"error": str(e)}), 400

                    synapse.dendrite = template.protocol._DendriteInfo(hotkey=sender_hotkey)

                    blacklist_fn = inf.get("blacklist_fn")
                    if blacklist_fn:
                        try:
                            blocked, reason = blacklist_fn(synapse)
                            if blocked:
                                logger.debug(f"[LocalAxon] Blacklisted {sender_hotkey}: {reason}")
                                return jsonify({"error": "blacklisted", "reason": reason}), 403
                        except Exception as e:
                            logger.error(f"[LocalAxon] Blacklist error for {cn}: {e}")

                    forward_fn = inf["forward_fn"]
                    try:
                        if asyncio.iscoroutinefunction(forward_fn):
                            loop = asyncio.new_event_loop()
                            result = loop.run_until_complete(forward_fn(synapse))
                            loop.close()
                        else:
                            result = forward_fn(synapse)
                    except Exception as e:
                        logger.error(f"[LocalAxon] Forward fn error for {cn}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        return jsonify({"error": str(e)}), 500

                    return result.model_dump_json(), 200, {"Content-Type": "application/json"}

                route_fn.__name__ = f"axon_{cn}"
                return route_fn

            app.add_url_rule(
                f"/axon/{class_name}",
                view_func=make_route(),
                methods=["POST"],
            )

        return app

    # ------------------------------------------------------------------
    def serve(self, netuid: int, subtensor):
        """Register this axon on chain via bt.ServeAxon."""
        try:
            import bittensor.utils.networking as net_utils
            ip = net_utils.get_external_ip()
        except Exception:
            ip = "0.0.0.0"

        try:
            result = subtensor.execute(
                bt.ServeAxon(netuid=netuid, ip=ip, port=self.external_port),
                self.wallet,
            )
            if result.success:
                logger.info(f"[LocalAxon] Registered on chain: {ip}:{self.external_port} netuid={netuid}")
            else:
                logger.warning(f"[LocalAxon] Chain registration failed: {getattr(result, 'error', result)}")
        except Exception as e:
            logger.error(f"[LocalAxon] Failed to register on chain: {e}")

    # ------------------------------------------------------------------
    def start(self):
        """Start the Flask server in a background daemon thread."""
        self._app = self._build_flask_app()
        import waitress

        def run():
            logger.info(f"[LocalAxon] Starting HTTP server on port {self.port}")
            waitress.serve(self._app, host="0.0.0.0", port=self.port)

        self._thread = threading.Thread(target=run, name="LocalAxon", daemon=True)
        self._thread.start()
        logger.info(f"[LocalAxon] Server thread started (port={self.port})")

    def __repr__(self):
        return f"LocalAxon(port={self.port}, handlers={list(self._handlers.keys())})"


# ──────────────────────────────────────────────────────────────────────────────

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
        parser = argparse.ArgumentParser()
        parser.add_argument("--autosync", action='store_true',
                            help="Automatically sync order data with a validator trusted by Taoshi.")
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

        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")

        # bt11-compatible argument helpers (replaces bt.Subtensor.add_args etc.)
        add_subtensor_args(parser)
        add_wallet_args(parser)
        add_axon_args(parser)
        add_logging_args(parser)

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

        args = parser.parse_args()
        config = build_config(args)

        if config.logging.debug:
            logger.setLevel(logging.DEBUG)
        if config.logging.trace:
            logger.setLevel(logging.DEBUG)

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
        self.axon = LocalAxon(
            wallet=self.wallet,
            port=self.config.axon.port,
            external_port=self.config.axon.external_port,
        )
        logger.info(f"Axon {self.axon}")

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

        logger.info(
            f"Serving attached axons on network:"
            f" {getattr(self.config.subtensor, 'chain_endpoint', None) or getattr(self.config.subtensor, 'network', 'finney')}"
            f" with netuid: {self.config.netuid}"
        )
        with get_subtensor_lock():
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        logger.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
