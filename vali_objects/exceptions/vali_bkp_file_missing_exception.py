# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

class ValiFileMissingException(Exception):
    def __init__(self, message):
        super().__init__(self, message)