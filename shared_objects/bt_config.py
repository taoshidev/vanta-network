"""
bt_config.py — bt10-compatible argparse helpers for bt11 migration.

bt10's bt.Subtensor.add_args / bt.Wallet.add_args / bt.Axon.add_args / bt.Config
are all removed in bt11. This module provides drop-in replacements that:

  1. Add the same CLI flags (--subtensor.network, --wallet.name, etc.)
  2. Return a nested-namespace config object with the same attribute access
     pattern as bt10's bt.Config (config.subtensor.network, etc.)
"""

import argparse
import os


class _Ns:
    """Simple dot-access namespace."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        pairs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Namespace({pairs})"


def add_subtensor_args(parser: argparse.ArgumentParser) -> None:
    """Add --subtensor.network and --subtensor.chain_endpoint flags."""
    parser.add_argument(
        "--subtensor.network",
        dest="subtensor_network",
        type=str,
        default="finney",
        help="Bittensor network name (finney, test, local) or ws:// URL",
    )
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="subtensor_chain_endpoint",
        type=str,
        default=None,
        help="Explicit wss:// chain endpoint (overrides --subtensor.network)",
    )


def add_wallet_args(parser: argparse.ArgumentParser) -> None:
    """Add --wallet.name, --wallet.hotkey, --wallet.path flags."""
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="default",
        help="Wallet coldkey name",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Wallet hotkey name",
    )
    parser.add_argument(
        "--wallet.path",
        dest="wallet_path",
        type=str,
        default=os.path.expanduser("~/.bittensor/wallets"),
        help="Path to wallet directory",
    )


def add_axon_args(parser: argparse.ArgumentParser) -> None:
    """Add --axon.port and --axon.external_port flags."""
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=8091,
        help="Port for the validator axon HTTP server",
    )
    parser.add_argument(
        "--axon.external_port",
        dest="axon_external_port",
        type=int,
        default=None,
        help="External port (if behind NAT)",
    )


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add --logging.debug, --logging.trace, --logging.logging_dir flags."""
    parser.add_argument(
        "--logging.debug",
        dest="logging_debug",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    parser.add_argument(
        "--logging.trace",
        dest="logging_trace",
        action="store_true",
        default=False,
        help="Enable trace logging",
    )
    parser.add_argument(
        "--logging.logging_dir",
        dest="logging_logging_dir",
        type=str,
        default=os.path.expanduser("~/.bittensor/miners"),
        help="Logging root directory",
    )


def build_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Convert a flat argparse Namespace (with _-separated dest names) into a
    nested Namespace that mirrors bt10's bt.Config attribute layout.

    Flat args produced by add_*_args helpers:
      args.subtensor_network, args.subtensor_chain_endpoint
      args.wallet_name, args.wallet_hotkey, args.wallet_path
      args.axon_port, args.axon_external_port
      args.logging_debug, args.logging_trace, args.logging_logging_dir

    Result:
      config.subtensor.network, config.subtensor.chain_endpoint
      config.wallet.name, config.wallet.hotkey, config.wallet.path
      config.axon.port, config.axon.external_port
      config.logging.debug, config.logging.trace, config.logging.logging_dir
      + all other flat args copied directly onto config
    """
    subtensor = _Ns(
        network=getattr(args, "subtensor_network", "finney"),
        chain_endpoint=getattr(args, "subtensor_chain_endpoint", None),
    )
    wallet = _Ns(
        name=getattr(args, "wallet_name", "default"),
        hotkey=getattr(args, "wallet_hotkey", "default"),
        path=getattr(args, "wallet_path", os.path.expanduser("~/.bittensor/wallets")),
    )
    axon = _Ns(
        port=getattr(args, "axon_port", 8091),
        external_port=getattr(args, "axon_external_port", None),
    )
    logging_ns = _Ns(
        debug=getattr(args, "logging_debug", False),
        trace=getattr(args, "logging_trace", False),
        logging_dir=getattr(args, "logging_logging_dir",
                            os.path.expanduser("~/.bittensor/miners")),
    )

    config = argparse.Namespace(**vars(args))
    config.subtensor = subtensor
    config.wallet = wallet
    config.axon = axon
    config.logging = logging_ns

    # Remove the flat aliases to keep the namespace clean
    _prefixes = {
        "subtensor_network", "subtensor_chain_endpoint",
        "wallet_name", "wallet_hotkey", "wallet_path",
        "axon_port", "axon_external_port",
        "logging_debug", "logging_trace", "logging_logging_dir",
    }
    for attr in _prefixes:
        if hasattr(config, attr):
            delattr(config, attr)

    return config


def make_subtensor(config) -> "bt.Subtensor":  # type: ignore[name-defined]
    """
    Create a bt.Subtensor from a nested config namespace.

    Tries chain_endpoint first (explicit ws:// URL), falls back to
    network name, then 'finney'.
    """
    import bittensor as bt

    subtensor_cfg = getattr(config, "subtensor", None)
    endpoint = getattr(subtensor_cfg, "chain_endpoint", None)
    network = getattr(subtensor_cfg, "network", None)

    # An explicit endpoint overrides the network name
    if endpoint:
        return bt.Subtensor(endpoint)
    return bt.Subtensor(network or "finney")


def make_wallet(config) -> "Wallet":  # type: ignore[name-defined]
    """Create a bittensor.wallet.Wallet from a nested config namespace."""
    from bittensor.wallet import Wallet

    wallet_cfg = getattr(config, "wallet", None)
    name = getattr(wallet_cfg, "name", "default")
    hotkey = getattr(wallet_cfg, "hotkey", "default")
    path = getattr(wallet_cfg, "path", os.path.expanduser("~/.bittensor/wallets"))
    return Wallet(name=name, hotkey=hotkey, path=path)
