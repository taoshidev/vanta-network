"""`vanta account`: Trading account opening for a miner on the standard account rules."""
from datetime import datetime, timezone
import getpass
import json
import secrets
from typing import Annotated, Optional

import typer
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from vanta_cli.src.config import VANTA_API_BASE_URL_MAINNET, VANTA_API_BASE_URL_TESTNET
from vanta_cli.src.options import EPILOG, Network, PromptFlag, WalletHotkey, WalletName, WalletPath, wallet_ask
from vanta_cli.src.utils.api import make_api_request
from vanta_cli.src.utils.console import console

app = typer.Typer(no_args_is_help=True, help="Vanta Network - Trading account commands", epilog=EPILOG,
                  pretty_exceptions_show_locals=False)

# Higher tiers allow more leverage and lock more collateral per dollar of account size
LEVERAGE_TIERS = {1: "Base", 2: "Boost I", 3: "Boost II"}

LeverageTier = Annotated[Optional[int], typer.Option(
    "--tier",
    help="Leverage tier: 1 Base, 2 Boost I, 3 Boost II",
)]


def opening_request(coldkey, hotkey_ss58: str, leverage_tier: int) -> dict:
    """The signed request the validator verifies; its nonce and timestamp make it single use."""
    opening_data = {
        "leverage_tier": leverage_tier,
        "miner_coldkey": coldkey.ss58_address,
        "miner_hotkey": hotkey_ss58,
        "nonce": secrets.token_urlsafe(),
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    message = json.dumps(opening_data, sort_keys=True)
    signature = coldkey.sign(message.encode('utf-8')).hex()
    return {**opening_data, "signature": signature}


@app.command("open", rich_help_panel="Trading account")
def open_account(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
    leverage_tier: LeverageTier = None,
    prompt: PromptFlag = True,
):
    """
    Open your miner's trading account on the standard account rules, sized from its collateral at the chosen leverage tier
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    if leverage_tier is None:
        console.print("\nLeverage tiers (a higher tier allows more leverage but opens a smaller account from the same collateral):")
        for tier, name in LEVERAGE_TIERS.items():
            console.print(f"{tier}. {name}")

        leverage_tier = IntPrompt.ask(
            "\nEnter the [bold]number[/bold] of the leverage tier",
            choices=[str(tier) for tier in LEVERAGE_TIERS],
            show_choices=False,
        )
    elif leverage_tier not in LEVERAGE_TIERS:
        console.print(f"[red]--tier must be one of {list(LEVERAGE_TIERS)}[/red]")
        raise typer.Exit(1)

    title = Text("🔗 VANTA NETWORK 🔗", style="bold blue")
    subtitle = Text("Trading Account Opening", style="italic cyan")

    panel = Panel.fit(
        f"{title}\n{subtitle}",
        style="bold blue",
        border_style="bright_blue"
    )

    console.print(panel)
    console.print("[blue]Opening the trading account on Vanta Network[/blue]")

    password = getpass.getpass(prompt='Enter your password: ')

    try:
        coldkey = wallet.get_coldkey(password=password)
        hotkey = wallet.hotkey
    except Exception as e:
        console.print(f"[red]Failed to unlock wallet: {e}[/red]")
        return False

    console.print(f"[cyan]Miner hotkey:[/cyan] {hotkey.ss58_address}")
    console.print(f"[cyan]Leverage tier:[/cyan] {leverage_tier} ({LEVERAGE_TIERS[leverage_tier]})")
    console.print("[yellow]The account size is set from your collateral at today's theta price and never changes "
                  "afterwards, and the tier cannot be changed. A higher tier allows more leverage but opens a smaller "
                  "account from the same collateral.[/yellow]")

    if prompt:
        confirm = typer.confirm(f"Open the trading account of miner {hotkey.ss58_address} at tier {leverage_tier}?")
        if not confirm:
            console.print("[yellow]Account opening cancelled[/yellow]")
            return False

    payload = opening_request(coldkey, hotkey.ss58_address, leverage_tier)
    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    console.print("\n[cyan]Sending account opening request...[/cyan]")
    console.print(f"[dim]Using network: {network}[/dim]")

    try:
        response = make_api_request("/account/open", payload, base_url=base_url)

        if response is None:
            console.print("[red]Account opening failed[/red]")
            return False

        if response.get("successfully_processed"):
            account = response.get("account") or {}
            console.print(f"[green]✅ {response.get('success_message') or 'Account opened'}[/green]")

            tier = account.get('opening_leverage_tier')
            table = Table(title="Trading Account", show_header=True, header_style="bold green")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Account size", f"${account.get('account_size') or 0:,.2f}")
            table.add_row("Leverage tier", f"{tier} ({LEVERAGE_TIERS.get(tier, '')})")
            table.add_row("Asset class", str(account.get('asset_class')))
            table.add_row("Locked collateral", f"{account.get('locked_theta') or 0:.4f} Theta")
            table.add_row("Theta price at opening", f"${account.get('opening_price_usd') or 0:,.4f}")
            console.print(table)
            return True
        else:
            error_message = (
                response.get("error_message") or
                response.get("error") or
                "An unknown error occurred."
            )
            console.print(f"[red]❌ Account opening failed: {error_message}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error during account opening: {e}[/red]")
        return False
