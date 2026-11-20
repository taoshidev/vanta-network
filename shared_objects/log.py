import logging
import sys

logger = logging.getLogger("bittensor")

# The bt.logging -> stdlib migration dropped bt.logging's implicit console handler, leaving this
# logger with none: every logger.info/debug call (the vast majority of this codebase's logging,
# including weight-setting progress) was silently swallowed by Python's WARNING-only handler of
# last resort. Each process (validator, miner, each RPC server subprocess) imports this module
# once, so guard against double-adding a handler on re-import within the same process.
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
    # This logger manages its own output; don't also hand records to the root logger's handlers.
    logger.propagate = False
