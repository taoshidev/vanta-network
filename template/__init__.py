# The MIT License (MIT)
# Copyright (c) 2024 Yuma Rao
# developer: Taoshidev
# Copyright (c) 2024 Taoshi Inc

__version__ = "2.0.0"
version_split = __version__.split(".")
__spec_version__ = (1000 * int(version_split[0])) + (10 * int(version_split[1])) + (1 * int(version_split[2]))

# Import all submodules.
from . import protocol  # noqa: E402, F401
