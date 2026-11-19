"""Minimal replacement for bittensor v10's bt.Config/argparse integration.

bittensor v11 removes bt.Config, Subtensor.add_args, Wallet.add_args, Axon.add_args —
"Scripts own their own argument parsing; the SDK reads nothing from sys.argv"
(MIGRATION-GUIDE.md). 118+ call sites across this repo read config as a nested namespace
(`config.wallet.name`, `config.subtensor.network`, `config.logging.debug`, ...), so rather
than flatten every one of those call sites, this keeps that exact nested shape — just
built by hand from a flat argparse.Namespace instead of bt.Config's parser magic. CLI
flag spelling (--wallet.name, --subtensor.network, etc.) is preserved so operators' run
scripts and vanta_cli don't need to change.
"""
import argparse
import os
from types import SimpleNamespace
from typing import Optional, Sequence


def add_wallet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wallet.name", type=str, default="default", dest="wallet_name",
                         help="The name of the wallet to use.")
    parser.add_argument("--wallet.hotkey", type=str, default="default", dest="wallet_hotkey",
                         help="The hotkey of the wallet to use.")
    parser.add_argument("--wallet.path", type=str, default=os.path.expanduser("~/.bittensor/wallets"),
                         dest="wallet_path", help="The path to the wallet directory.")


def add_subtensor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subtensor.network", type=str, default="finney", dest="subtensor_network",
                         help="The subtensor network to connect to (finney/test/archive/local, or a ws:// URL).")
    parser.add_argument("--subtensor.chain_endpoint", type=str, default=None, dest="subtensor_chain_endpoint",
                         help="Override chain endpoint (ws:// URL). Takes precedence over --subtensor.network.")


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--logging.debug", action="store_true", default=False, dest="logging_debug",
                         help="Turn on debugging information")
    parser.add_argument("--logging.trace", action="store_true", default=False, dest="logging_trace",
                         help="Turn on trace level information")
    parser.add_argument("--logging.logging_dir", type=str, default=os.path.expanduser("~/.bittensor/miners"),
                         dest="logging_logging_dir", help="Logging default root directory.")


def build_config(parser: argparse.ArgumentParser, args: Optional[Sequence[str]] = None) -> SimpleNamespace:
    """Parses args and regroups the flat argparse.Namespace into the nested shape
    (`config.wallet.name`, `config.subtensor.network`, `config.logging.debug`, ...) that
    every call site in this repo already expects. Only regroups keys this module's
    add_*_args functions added — any other --foo.bar-style arguments a caller adds with
    dest='foo_bar' must be regrouped by the caller itself, same as before this helper existed.
    """
    ns = parser.parse_args(args=args)
    config = SimpleNamespace(**vars(ns))

    if hasattr(ns, "wallet_name"):
        config.wallet = SimpleNamespace(name=ns.wallet_name, hotkey=ns.wallet_hotkey, path=ns.wallet_path)
    if hasattr(ns, "subtensor_network"):
        config.subtensor = SimpleNamespace(network=ns.subtensor_network, chain_endpoint=ns.subtensor_chain_endpoint)
    if hasattr(ns, "logging_debug"):
        config.logging = SimpleNamespace(debug=ns.logging_debug, trace=ns.logging_trace,
                                          logging_dir=ns.logging_logging_dir)
    return config
