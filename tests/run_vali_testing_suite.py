# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

import sys
import unittest

from vali_objects.vali_config import ValiConfig

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
