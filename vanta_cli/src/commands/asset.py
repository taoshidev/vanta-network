"""`vanta asset`: Asset selection commands."""
import getpass
import json

import typer
from rich.panel import Panel
from rich.prompt import IntPrompt

from vanta_cli.src.config import VANTA_API_BASE_URL_MAINNET, VANTA_API_BASE_URL_TESTNET
from vanta_cli.src.options import EPILOG, Network, PromptFlag, WalletHotkey, WalletName, WalletPath, wallet_ask
from vanta_cli.src.utils.api import make_api_request
from vanta_cli.src.utils.console import console

app = typer.Typer(no_args_is_help=True, help="Vanta Network - Asset selection commands", epilog=EPILOG)


@app.command("select", rich_help_panel="Asset class selection")
def select(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
    prompt: PromptFlag = True,
):
    """
    Select the asset class your miner trades
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    assets = ["crypto", "forex", "equities", "commodities"]

    for idx, asset in enumerate(assets, start=1):
        console.print(f"{idx}. {asset}")

    choice = IntPrompt.ask(
        "\nEnter the [bold]number[/bold] of the asset class you want to select",
        choices=[str(i) for i in range(1, len(assets) + 1)],
        show_choices=False,
    )
    asset = assets[choice - 1]

    console.print("[blue]Selecting miner asset class on Vanta Network[/blue]")

    # Load wallet and get keys
    password = getpass.getpass(prompt='Enter your password: ')

    coldkey = wallet.get_coldkey(password=password)
    hotkey = wallet.hotkey

    if prompt:
        confirm = typer.confirm(f"Are you sure you want to select asset class {asset} for miner {hotkey.ss58_address}?")
        if not confirm:
            console.print("[yellow]Asset Class Selection Cancelled[/yellow]")
            return False

    # Prepare withdrawal data for signing
    asset_selection_data = {
        "asset_selection": asset,
        "miner_coldkey": coldkey.ss58_address,
        "miner_hotkey": hotkey.ss58_address
    }

    # Create message to sign (sorted JSON)
    message = json.dumps(asset_selection_data, sort_keys=True)

    # Sign the message with coldkey
    signature = coldkey.sign(message.encode('utf-8')).hex()

    # Prepare payload for withdrawal (include signature)
    payload = {
        "asset_selection": asset,
        "miner_coldkey": coldkey.ss58_address,
        "miner_hotkey": hotkey.ss58_address,
        "signature": signature
    }

    # Determine which API base URL to use based on network
    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    # Make the API request
    console.print("\n[cyan]Sending asset selection request...[/cyan]")
    console.print(f"[dim]Using network: {network}[/dim]")

    try:
        response = make_api_request("/asset-selection", payload, base_url=base_url)

        if response is None:
            console.print("[red]Asset class selection failed[/red]")
            return False

        # Check if asset selection was successful
        if response.get("successfully_processed"):
            console.print(f"[green]✅ Asset selection successful! Selected: {asset}[/green]")

            success_panel = Panel.fit(
                f"🎉 Asset selection completed!\nSelected: {asset}",
                style="bold green",
                border_style="green"
            )
            console.print(success_panel)
            return True
        else:
            error_message = (
                response.get("error_message") or
                response.get("error") or
                "An unknown error occurred."
            )
            console.print(f"[red]❌ Asset selection failed: {error_message}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error during asset selection: {e}[/red]")
