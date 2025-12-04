# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import json
import os
import argparse
import threading
import traceback
import time
import bittensor as bt
import subprocess

from miner_config import MinerConfig
from miner_objects.dashboard import Dashboard
from miner_objects.prop_net_order_placer import PropNetOrderPlacer
from miner_objects.position_inspector import PositionInspector
from shared_objects.slack_notifier import SlackNotifier
from shared_objects.metagraph.metagraph_updater import MetagraphUpdater
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from vali_objects.decoders.generalized_json_decoder import GeneralizedJSONDecoder
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils


class Miner:
    def __init__(self):
        self.config = self.get_config()
        assert self.config.netuid in (8, 116), "Taoshi runs on netuid 8 (mainnet) and 116 (testnet)"
        self.is_testnet = self.config.netuid == 116

        self.setup_logging_directory()
        self.wallet = bt.wallet(config=self.config)

        # Initialize Slack notifier
        self.slack_notifier = SlackNotifier(
            hotkey=self.wallet.hotkey.ss58_address,
            webhook_url=self.config.slack_webhook_url,
            error_webhook_url=self.config.slack_error_webhook_url,
            is_miner=True,
            enable_metrics=True,
            enable_daily_summary=True
        )

        # Start required servers using ServerOrchestrator (fixes connection errors)
        # This ensures servers are fully started before clients try to connect
        bt.logging.info("Initializing miner servers...")
        self.orchestrator = ServerOrchestrator.get_instance()

        # Start only the servers miners need (common_data, metagraph)
        # Servers start in dependency order and block until ready - no connection errors!
        self.orchestrator.start_all_servers(
            mode=ServerMode.MINER,
            secrets=None
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection errors)
        self.metagraph_client = self.orchestrator.get_client('metagraph')

        bt.logging.success("Miner servers initialized successfully")

        self.position_inspector = PositionInspector(self.wallet, self.metagraph_client, self.config)
        self.metagraph_updater = MetagraphUpdater(self.config, self.wallet.hotkey.ss58_address,
                                                  True, position_inspector=self.position_inspector,
                                                    slack_notifier=self.slack_notifier)
        self.prop_net_order_placer = PropNetOrderPlacer(
            self.wallet,
            self.metagraph_client,
            self.config,
            self.is_testnet,
            position_inspector=self.position_inspector,
            slack_notifier=self.slack_notifier
        )

        # Start the metagraph updater and wait for initial population
        self.metagraph_updater_thread = self.metagraph_updater.start_and_wait_for_initial_update(
            max_wait_time=60,
            slack_notifier=self.slack_notifier
        )
        
        self.check_miner_registration()
        self.my_subnet_uid = self.metagraph_client.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Running miner on netuid {self.config.netuid} with uid: {self.my_subnet_uid}")



        # Send startup notification with hotkey and IP
        self.slack_notifier.send_message(
            f"🚀 Miner starting on netuid {self.config.netuid} ({'testnet' if self.is_testnet else 'mainnet'})\n"
            f"UID: {self.my_subnet_uid}\n",
            level="info"
        )

        # Start position inspector loop in its own thread
        if self.config.run_position_inspector:
            self.position_inspector_thread = threading.Thread(target=self.position_inspector.run_update_loop_sync,
                                                              daemon=True)
            self.position_inspector_thread.start()
        else:
            self.position_inspector_thread = None
        # Dashboard
        # Start the miner data api in its own thread
        try:
            self.dashboard = Dashboard(self.wallet, self.metagraph_client, self.config, self.is_testnet)
            self.dashboard_api_thread = threading.Thread(target=self.dashboard.run, daemon=True)
            self.dashboard_api_thread.start()
        except OSError as e:
            bt.logging.info(
                f"Unable to start miner dashboard with error {e}. Restart miner and specify a new port if desired.")
            self.slack_notifier.send_message(
                f"⚠️ Failed to start dashboard: {str(e)}",
                level="warning"
            )
        # Initialize the dashboard process variable for the frontend
        self.dashboard_frontend_process = None

    def setup_logging_directory(self):
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

    def check_miner_registration(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph_client.hotkeys:
            error_msg = "Your miner is not registered. Please register and try again."
            bt.logging.error(error_msg)
            self.slack_notifier.send_message(f"❌ {error_msg}", level="error")
            exit()

    def load_signal_data(self, signal_file_path: str):
        """Loads the signal data from a file."""
        try:
            data = ValiBkpUtils.get_file(signal_file_path)
            return json.loads(data, cls=GeneralizedJSONDecoder)
        except json.JSONDecodeError as e:
            bt.logging.error(f"Failed to decode JSON from {signal_file_path}: {e}")
            self.slack_notifier.send_message(
                f"❌ Failed to decode signal file: {os.path.basename(signal_file_path)}\nError: {str(e)}",
                level="error"
            )
            return None

    def get_all_files_in_dir_no_duplicate_trade_pairs(self):
        # If there are duplicate trade pairs, only the most recent signal for that trade pair will be sent this round.
        all_files = ValiBkpUtils.get_all_files_in_dir(MinerConfig.get_miner_received_signals_dir())
        signals_dict = {}
        files_to_delete = []
        for f_name in all_files:
            try:
                bt.logging.info(f"Reading signal file {f_name}")
                signal = self.load_signal_data(f_name)
                if signal:
                    trade_pair_id = signal['trade_pair']['trade_pair_id']
                    time_of_signal_file = os.path.getmtime(f_name)
                    if trade_pair_id not in signals_dict or signals_dict[trade_pair_id][2] < time_of_signal_file:
                        signals_dict[trade_pair_id] = (signal, f_name, time_of_signal_file)
                        files_to_delete.append(f_name)
            except json.JSONDecodeError as e:
                bt.logging.error(f"Error decoding JSON from file {f_name}: {e}")

        # Delete files to prevent duplicate reading and conflicts
        for f_name in files_to_delete:
            bt.logging.info(f"Deleting signal file {f_name}")
            os.remove(f_name)

        # Return all signals as a list
        return [x[0] for x in signals_dict.values()], [x[1] for x in signals_dict.values()]

    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser()
        # Adds override arguments for network and netuid.
        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
        # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
        bt.logging.add_args(parser)
        # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
        bt.wallet.add_args(parser)
        # Adds an argument to allow setting write_failed_signal_logs from the command line
        parser.add_argument("--write_failed_signal_logs", type=bool, default=None,
                            help="Whether to write logs for failed signals. Default is True unless --subtensor.network is 'test'.")
        # Add argument so we can check if run_position_inspector is set which tells us to start the PI thread. Default false
        parser.add_argument("--run-position-inspector", action="store_true", help="Run the position inspector thread.")
        parser.add_argument(
            '--start-dashboard',
            action='store_true',
            help='Start the miner-dashboard along with the miner.'
        )
        # Add Slack configuration argument
        parser.add_argument(
            '--slack-webhook-url',
            type=str,
            default=None,
            help='Slack webhook URL for notifications'
        )
        parser.add_argument(
            '--slack-error-webhook-url',
            type=str,
            default=None,
            help='Slack webhook URL for error notifications'
        )

        # Parse the config (will take command-line arguments if provided)
        config = bt.config(parser)
        bt.logging.enable_info()
        if config.logging.debug:
            bt.logging.enable_debug()
        if config.logging.trace:
            bt.logging.enable_trace()

        # Determine the default value for write_failed_signal_logs based on the subtensor.network
        if config.write_failed_signal_logs is None:
            config.write_failed_signal_logs = False if config.subtensor.network == "test" else True

        # Set up logging directory
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "miner",
            )
        )
        return config

    def start_dashboard_frontend(self):
        """
        starts the miner dashboard. Allows the use of npm, yarn, or pnpm
        """
        try:
            dashboard_dir = "miner_objects/miner_dashboard"
            # Determine which package manager is available
            package_manager = None
            for pm in ['pnpm', 'yarn', 'npm']:
                if subprocess.run(['which', pm], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                    package_manager = pm
                    break

            if not package_manager:
                bt.logging.error("No package manager found. Please install npm, yarn, or pnpm.")
                return

            # Run 'install' command for the identified package manager
            subprocess.run([package_manager, "install"], cwd=dashboard_dir, check=True)
            bt.logging.info(f"Install completed using {package_manager}.")

            # Start the dashboard process
            if package_manager == 'npm':
                self.dashboard_frontend_process = subprocess.Popen(['npm', 'run', 'dev'], cwd=dashboard_dir)
            else:
                self.dashboard_frontend_process = subprocess.Popen([package_manager, 'dev'], cwd=dashboard_dir)
            bt.logging.info("Dashboard started.")
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Command '{e.cmd}' failed with return code {e.returncode}.")
        except Exception as e:
            bt.logging.error(f"Failed to start dashboard: {e}")

    def run(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info("Starting miner loop.")

        # Start the dashboard if the flag is set
        if self.config.start_dashboard:
            bt.logging.info("Starting miner dashboard.")
            self.start_dashboard_frontend()

        bt.logging.info("Waiting for signals...")
        while True:
            try:
                signals, signal_file_names = self.get_all_files_in_dir_no_duplicate_trade_pairs()
                n_signals = len(signals)
                self.prop_net_order_placer.send_signals(signals, signal_file_names, recently_acked_validators=
                self.position_inspector.get_recently_acked_validators())
                if n_signals == 0:
                    time.sleep(0.2)
            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.slack_notifier.send_message("🛑 Miner shutting down (keyboard interrupt)", level="warning")
                bt.logging.success("Miner killed by keyboard interrupt.")

                #self.slack_notifier.shutdown()

                # Shutdown the order placer's thread pool
                bt.logging.info("Shutting down order placer thread pool...")
                self.prop_net_order_placer.shutdown()

                if self.dashboard_frontend_process:
                    self.dashboard_frontend_process.terminate()
                    self.dashboard_frontend_process.wait()
                    bt.logging.info("Dashboard terminated.")
                self.metagraph_updater_thread.join()
                self.position_inspector.stop_update_loop()
                if self.position_inspector_thread:
                    self.position_inspector_thread.join()
                # dashboard api server
                if self.dashboard_api_thread is not None and self.dashboard_api_thread.is_alive():
                    self.dashboard_api_thread.join()
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                error_trace = traceback.format_exc()
                bt.logging.error(error_trace)
                self.slack_notifier.send_message(
                    f"❌ Unexpected error for hotkey in main loop:\n{str(e)}\n\nTraceback:\n{error_trace[:500]}",
                    level="error"
                )
                time.sleep(10)


if __name__ == "__main__":
    miner = Miner()
    miner.run()
