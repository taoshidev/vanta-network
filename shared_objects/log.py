import logging
import sys

logger = logging.getLogger("bittensor")

# The bt.logging -> stdlib migration dropped bt.logging's implicit console handler, leaving this
# logger with none: every logger.info/debug call (the vast majority of this codebase's logging,
# including weight-setting progress) was silently swallowed by Python's WARNING-only handler of
# last resort. Each process (validator, miner, each RPC server subprocess) imports this module
# once, so guard against double-adding a handler on re-import within the same process.
class _ColorTimeFormatter(logging.Formatter):
    _TIME_COLOR = "\033[90m"  # dim gray
    _RESET = "\033[0m"
    _LEVEL_COLORS = {
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
    }

    def formatTime(self, record, datefmt=None):
        return f"{self._TIME_COLOR}{super().formatTime(record, datefmt)}{self._RESET}"

    def format(self, record):
        message = super().format(record)
        color = self._LEVEL_COLORS.get(record.levelname)
        return f"{color}{message}{self._RESET}" if color else message


if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(_ColorTimeFormatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(_handler)
    # This logger manages its own output; don't also hand records to the root logger's handlers.
    logger.propagate = False
