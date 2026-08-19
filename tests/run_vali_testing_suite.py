# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import sys
import unittest

from vali_objects.vali_config import ValiConfig

# Ensure the repo root is importable. This script's sys.path[0] is the tests/
# dir, and the editable install only exposes packages (dirs with __init__.py),
# not top-level modules like miner_config.py — so tests that import code which
# does `from miner_config import MinerConfig` (e.g. the entity-miner gateway)
# otherwise fail with ModuleNotFoundError under this runner.
if ValiConfig.BASE_DIR not in sys.path:
    sys.path.insert(0, ValiConfig.BASE_DIR)

if __name__ == '__main__':
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
