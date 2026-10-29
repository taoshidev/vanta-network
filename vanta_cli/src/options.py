"""Options shared by every vanta command, and the wallet lookup they feed."""
import os
from typing import Annotated, Optional

import typer
from bittensor_wallet import Wallet
from rich.prompt import Prompt

from vanta_cli.src.utils.console import err_console

EPILOG = "Made with [bold red]:heart:[/bold red] by Vanτa Neτwork"

DEFAULT_WALLET_PATH = "~/.bittensor/wallets"

# Wallet aliases match bittensor-cli v9 so existing invocations keep working.
WalletName = Annotated[Optional[str], typer.Option(
    "--wallet-name", "--name", "--wallet_name", "--wallet.name",
    help="Name of the wallet.",
)]
WalletPath = Annotated[Optional[str], typer.Option(
    "--wallet-path", "-p", "--wallet_path", "--wallet.path",
    help=f"Path where the wallets are located (default: {DEFAULT_WALLET_PATH}).",
)]
WalletHotkey = Annotated[Optional[str], typer.Option(
    "--hotkey", "-H", "--wallet-hotkey", "--wallet_hotkey", "--wallet.hotkey",
    help="Name of the wallet hotkey.",
)]
Network = Annotated[str, typer.Option(
    "--network", "--subtensor.network",
    help="The subtensor network to connect to.",
)]
Amount = Annotated[Optional[float], typer.Option(
    "--amount",
    help="Amount of Theta to use for collateral",
)]
PromptFlag = Annotated[bool, typer.Option(
    "--prompt/--no-prompt",
    help="Whether to prompt for confirmation",
)]


def wallet_ask(
    wallet_name: Optional[str],
    wallet_path: Optional[str],
    wallet_hotkey: Optional[str],
) -> Wallet:
    """
    Builds the wallet from the supplied values, prompting for a missing name or hotkey,
    and exits if the coldkey or hotkey does not exist on disk.
    """
    if not wallet_name:
        wallet_name = Prompt.ask("Enter the [blue]wallet name[/blue]", default="default")
    if not wallet_hotkey:
        wallet_hotkey = Prompt.ask("Enter the [blue]wallet hotkey[/blue]", default="default")
    wallet_path = os.path.expanduser(wallet_path or DEFAULT_WALLET_PATH)

    wallet = Wallet(name=wallet_name, hotkey=wallet_hotkey, path=wallet_path)

    if not os.path.isdir(os.path.join(wallet_path, wallet_name)):
        err_console.print(
            f"[red]Error: Wallet does not exist.\nPlease verify your wallet information: {wallet}[/red]"
        )
        raise typer.Exit(1)
    if not os.path.isfile(os.path.join(wallet_path, wallet_name, "hotkeys", wallet_hotkey)):
        err_console.print(
            f"[red]Error: Wallet '{wallet_name}' exists but the hotkey '{wallet_hotkey}' does not.\n"
            f"Please verify your wallet information: {wallet}[/red]"
        )
        raise typer.Exit(1)
    return wallet
