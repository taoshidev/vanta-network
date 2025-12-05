# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import json

from typing import Dict, List

from vali_objects.exceptions.vali_bkp_file_missing_exception import (
    ValiFileMissingException,
)
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

class ValiUtils:
    @staticmethod
    def get_secrets(running_unit_tests=False) -> Dict:
        # wrapping here to allow simpler error handling & original for other error handling
        if running_unit_tests:
            return {'polygon_apikey': "", 'tiingo_apikey': ""}

        ans = {}
        try:
            secrets = ValiBkpUtils.get_file(ValiBkpUtils.get_secrets_dir())
            ans = json.loads(secrets)
            if running_unit_tests:
                for k in ['polygon_apikey', 'tiingo_apikey']:
                    if k not in ans:
                        ans[k] = ""
        except FileNotFoundError:
            raise ValiFileMissingException("Vali secrets file is missing")

        return ans

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
