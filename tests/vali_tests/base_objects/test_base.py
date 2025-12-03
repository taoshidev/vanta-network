# developer: Taoshidev
# Copyright © 2024 Taoshi Inc

import os
import unittest

class TestBase(unittest.TestCase):

    def setUp(self) -> None:
        if "vm" in os.environ:
            del os.environ["vm"]
        return

    def tearDown(self) -> None:
        pass
