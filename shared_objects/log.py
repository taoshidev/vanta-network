import logging
import sys

logger = logging.getLogger("bittensor")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
