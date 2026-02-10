# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import json

from typing import Dict, List, Optional
import bittensor as bt
from google.cloud import secretmanager

from vali_objects.exceptions.vali_bkp_file_missing_exception import (
    ValiFileMissingException,
)
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

class ValiUtils:
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
            bt.logging.info(f"{secret_name} retrieved from local secrets file")
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
                bt.logging.info(f"{secret_name} retrieved from Google Cloud Secret Manager")
                return secret
            else:
                bt.logging.debug(f"{secret_name} not found in Google Cloud Secret Manager")
                return None
        except Exception as e:
            bt.logging.debug(f"Failed to retrieve {secret_name} from Google Cloud: {e}")

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
            >>>     bt.logging.info("Running as mothership")
        """
        from vali_objects.vali_config import ValiConfig

        if not wallet or not hasattr(wallet, 'hotkey'):
            return False

        hotkey = wallet.hotkey.ss58_address
        if is_testnet:
            return hotkey == ValiConfig.MOTHERSHIP_HOTKEY_TESTNET
        else:
            return hotkey == ValiConfig.MOTHERSHIP_HOTKEY
