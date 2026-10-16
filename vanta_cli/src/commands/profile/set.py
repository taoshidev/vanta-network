import getpass
import json
import typer

from rich.panel import Panel
from rich.text import Text

from bittensor_wallet import Wallet
from bittensor_cli.src.bittensor.utils import console

from vanta_cli.src.config import VANTA_API_BASE_URL_MAINNET, VANTA_API_BASE_URL_TESTNET
from vanta_cli.src.utils.api import make_api_request

async def set(
    wallet: Wallet,
    network: str,
    profile_key: str,
    value: str,
    prompt: bool,
):
    console.print("[blue]Updating profile on Vanta Network[/blue]")

    hotkey = wallet.hotkey
    if prompt:
        console.print(Panel.fit(
            f"hotkey: {hotkey.ss58_address}\n[dim]{profile_key}[/dim]: {value}",
            title="Profile Update",
            border_style="yellow"
        ))
        typer.confirm("Proceed?", abort=True)

    # Load wallet and get keys
    password = getpass.getpass(prompt='Enter your password: ')
    coldkey = wallet.get_coldkey(password=password)

    # Prepare data for signing
    profile_data = {
        profile_key: value,
        "miner_coldkey": coldkey.ss58_address,
        "miner_hotkey": hotkey.ss58_address
    }

    # Create message to sign (sorted JSON)
    message = json.dumps(profile_data, sort_keys=True)

    # Sign the message with coldkey
    signature = coldkey.sign(message.encode('utf-8')).hex()

    # Prepare payload (include signature)
    payload = {
        profile_key: value,
        "miner_coldkey": coldkey.ss58_address,
        "miner_hotkey": hotkey.ss58_address,
        "signature": signature
    }

    # Determine which API base URL to use based on network
    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    # Make the API request
    console.print("\n[cyan]Sending profile update request...[/cyan]")
    console.print(f"[dim]Using network: {network}[/dim]")

    try:
        response = make_api_request("/profile", payload, base_url=base_url)

        if response is None:
            console.print("[red]Profile update failed[/red]")
            return False

        if response.get("successfully_processed"):
            console.print(f"[green]Profile update successful![/green]")

            success_panel = Panel.fit(
                f"Profile updated!\n{profile_key}: {value}",
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
            console.print(f"[red]Profile update failed: {error_message}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Error during profile update: {e}[/red]")



