# developer: jbonilla
import os
import datetime

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from pathlib import Path


import bittensor as bt


class CacheController:
    ELIMINATIONS = "eliminations"
    MAX_DAILY_DRAWDOWN = 'MAX_DAILY_DRAWDOWN'
    MAX_TOTAL_DRAWDOWN = 'MAX_TOTAL_DRAWDOWN'

    def __init__(self, running_unit_tests=False, is_backtesting=False, connection_mode: RPCConnectionMode = RPCConnectionMode.RPC):
        self.running_unit_tests = running_unit_tests
        self.init_cache_files()
        self.is_backtesting = is_backtesting
        self._last_update_time_ms = 0
        self.DD_V2_TIME = TimeUtil.millis_to_datetime(1715359820000 + 1000 * 60 * 60 * 2)  # 5/10/24 TODO: Update before mainnet release

        # Create metagraph client internally (forward compatibility pattern)
        # connection_mode controls RPC behavior, running_unit_tests only controls file paths
        # Default to RPC mode - use LOCAL only when explicitly requested
        self._connection_mode = connection_mode

        # Create MetagraphClient - in LOCAL mode it won't connect via RPC
        # Tests can call set_direct_metagraph_server() to inject a server reference
        from shared_objects.rpc.metagraph_client import MetagraphClient
        self._metagraph_client : MetagraphClient = MetagraphClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)

    def get_last_update_time_ms(self):
        return self._last_update_time_ms

    def set_last_update_time(self, skip_message=False):
        # Log that the class has finished updating and the time it finished updating
        self._last_update_time_ms = TimeUtil.now_in_millis()
        delta_time_ms = self._last_update_time_ms - self.attempted_start_time_ms
        delta_time_s = delta_time_ms / 1000
        delta_time_s_formatted_3_decimals = "{:.3f}".format(delta_time_s)
        if not skip_message:
            bt.logging.success(f"Finished updating class {self.__class__.__name__} in {delta_time_s_formatted_3_decimals} seconds.")

    @staticmethod
    def get_directory_names(query_dir):
        """
        Returns a list of directory names contained in the specified directory.

        :param query_dir: Path to the directory to query.
        :return: List of directory names.
        """
        directory_names = [item for item in os.listdir(query_dir) if (Path(query_dir) / item).is_dir()]
        return directory_names

    # ----------------- Eliminations -----------------

    @staticmethod
    def generate_elimination_row(hotkey, dd, reason, t_ms=None, price_info=None, return_info=None):
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()
        ans = {'hotkey': hotkey, 'elimination_initiated_time_ms': t_ms, 'dd': dd, 'reason': reason}
        if price_info:
            ans['price_info'] = price_info
        if return_info:
            ans['return_info'] = return_info
        bt.logging.info(f"Created elimination row: {ans}")
        return ans

    def refresh_allowed(self, refresh_interval_ms):
        self.attempted_start_time_ms = TimeUtil.now_in_millis()

        if self.is_backtesting or self.running_unit_tests:
            return True

        return  self.attempted_start_time_ms - self.get_last_update_time_ms() > refresh_interval_ms


    def init_cache_files(self) -> None:
        ValiBkpUtils.make_dir(ValiBkpUtils.get_vali_dir(running_unit_tests=self.running_unit_tests))
        # Check if the get_miner_dir directory exists. If not, create it
        dir_path = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def get_last_modified_time_miner_directory(self, directory_path):
        try:
            # Get the last modification time of the directory
            timestamp = os.path.getmtime(directory_path)

            # Convert the timestamp to a datetime object
            last_modified_date = datetime.datetime.fromtimestamp(timestamp)

            return last_modified_date
        except FileNotFoundError:
            # Handle the case where the directory does not exist
            print(f"The directory {directory_path} was not found.")
            return None


    @staticmethod
    def calculate_drawdown(final, initial):
        # Ex we went from return of 1 to 0.9. Drawdown is -10% or in this case -0.1. Return 1 - 0.1 = 0.9
        # Ex we went from return of 0.9 to 1. Drawdown is +10% or in this case 0.1. Return 1 + 0.1 = 1.1 (not really a drawdown)
        return 1.0 + ((float(final) - float(initial)) / float(initial))

    def is_drawdown_beyond_mdd(self, dd, time_now=None) -> str | bool:
        # DD V2 5/10/24 - remove daily DD. Make anytime DD limit 95%.
        if time_now is None:
            time_now = TimeUtil.generate_start_timestamp(0)
        if time_now < self.DD_V2_TIME:
            if dd < ValiConfig.MAX_DAILY_DRAWDOWN and time_now.hour == 0 and time_now.minute < 5:
                return CacheController.MAX_DAILY_DRAWDOWN
            elif dd < ValiConfig.MAX_TOTAL_DRAWDOWN:
                return CacheController.MAX_TOTAL_DRAWDOWN
        else:
            if dd < ValiConfig.MAX_TOTAL_DRAWDOWN_V2:
                return CacheController.MAX_TOTAL_DRAWDOWN

        return False
