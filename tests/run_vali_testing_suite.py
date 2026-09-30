# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import sys
import unittest

from vali_objects.vali_config import ValiConfig

if __name__ == '__main__':
    # Ensure the repo root is importable. This script's sys.path[0] is the tests/
    # dir, and an editable install that predates the py_modules=["miner_config"]
    # declaration in setup.py doesn't expose top-level modules — so tests that
    # import code doing `from miner_config import MinerConfig` (e.g. the
    # entity-miner gateway) otherwise fail with ModuleNotFoundError under this
    # runner. Guarded under __main__ so merely importing this file never
    # mutates sys.path.
    if ValiConfig.BASE_DIR not in sys.path:
        sys.path.insert(0, ValiConfig.BASE_DIR)

    # Create a test loader
    loader = unittest.TestLoader()

    if len(sys.argv) > 1:
        # Get the test file name from the command line argument
        test_file = sys.argv[1]
        suite = loader.discover(start_dir=ValiConfig.BASE_DIR + "/tests/vali_tests/", pattern=test_file)
    else:
        # Discover all test files in the specified directory
        start_dir = ValiConfig.BASE_DIR + "/tests/vali_tests/"
        suite = loader.discover(start_dir, pattern='test_*.py')

    # Create an instance of the custom test runner
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
