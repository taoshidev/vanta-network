"""`vanta entity`: Entity management commands."""
import getpass
import json
from typing import Annotated, Optional

import typer
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt
from rich.table import Table

from vanta_cli.src.config import VANTA_API_BASE_URL_MAINNET, VANTA_API_BASE_URL_TESTNET
from vanta_cli.src.options import EPILOG, Network, PromptFlag, WalletHotkey, WalletName, WalletPath, wallet_ask
from vanta_cli.src.utils.api import make_api_request
from vanta_cli.src.utils.console import console

app = typer.Typer(no_args_is_help=True, help="Vanta Network - Entity management commands", epilog=EPILOG)


@app.command("register", rich_help_panel="Entity Management")
def register(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
    prompt: PromptFlag = True,
):
    """
    Register a new entity on the Vanta Network
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    console.print("[blue]Registering entity on Vanta Network[/blue]")

    # Determine base URL
    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    registration_fee = 1000 # Theta

    # Display configuration
    config_table = Table(title="Entity Registration Configuration", show_header=True, header_style="bold cyan")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Network", "Testnet" if network == "test" else "Mainnet")
    config_table.add_row("Registration Fee", f"{registration_fee} Theta")

    console.print(config_table)

    # Step 2: Get password
    password = getpass.getpass(prompt='Enter your wallet password: ')

    try:
        coldkey = wallet.get_coldkey(password=password)
        hotkey = wallet.hotkey
    except Exception as e:
        console.print(f"[red]Failed to unlock wallet: {e}[/red]")
        return False

    # Step 3: Confirm registration
    if prompt:
        confirm = typer.confirm(
            f"Register entity {hotkey.ss58_address} "
            f"(costs {registration_fee} Theta)?"
        )
        if not confirm:
            console.print("[yellow]Entity registration cancelled[/yellow]")
            return False

    # Step 4: Ensure sufficient collateral
    response = make_api_request(f"/collateral/balance/{hotkey.ss58_address}", method="GET", base_url=base_url)
    if not response or response.get("balance_theta") < registration_fee:
        console.print(f"[red]Insufficient collateral for entity registration: {response.get('balance_theta')}[/red]")
        return False


    # collateral_success, collateral_msg = ensure_sufficient_collateral(
    #     wallet=wallet,
    #     network=network,
    #     required_theta=registration_fee,
    #     purpose="entity registration",
    #     password=password,
    #     base_url=base_url,
    # )
    #
    # if not collateral_success:
    #     console.print(f"[red]Collateral check failed: {collateral_msg}[/red]")
    #     return False

    # Step 5: Prepare and sign registration request
    console.print("\n[cyan]Signing entity registration request...[/cyan]")

    registration_data = {
        "entity_coldkey": coldkey.ss58_address,
        "entity_hotkey": hotkey.ss58_address
    }

    # Create message to sign (sorted JSON)
    message = json.dumps(registration_data, sort_keys=True)

    # Sign the message with coldkey
    signature = coldkey.sign(message.encode('utf-8')).hex()

    # Prepare payload
    payload = {
        "entity_coldkey": coldkey.ss58_address,
        "entity_hotkey": hotkey.ss58_address,
        "signature": signature
    }

    # Step 6: Send registration request
    console.print("\n[cyan]Sending entity registration request...[/cyan]")

    try:
        response = make_api_request("/entity/register", payload, base_url=base_url)

        if response is None:
            console.print("[red]Entity registration failed - no response[/red]")
            return False

        # Check success
        if response.get("status") == "success":
            console.print(f"[green]{response.get('message')}[/green]")

            # Display success info
            success_table = Table(title="Entity Registered Successfully", show_header=True, header_style="bold green")
            success_table.add_column("Field", style="cyan")
            success_table.add_column("Value", style="green")

            success_table.add_row("Entity Hotkey", response.get('entity_hotkey'))
            success_table.add_row("Fee Charged", f"{registration_fee} Theta")

            console.print(success_table)

            success_panel = Panel.fit(
                f"🎉 Entity registration completed!\nYou can now create subaccounts using 'vanta entity create-subaccount'\nCreate your API key using 'vanta entity apikey'",
                style="bold green",
                border_style="green"
            )
            console.print(success_panel)
            return True
        else:
            error_message = response.get("error") or "Unknown error occurred"
            console.print(f"[red]Entity registration failed: {error_message}[/red]")
            return False

    except Exception as e:
        console.print(f"[red]Error during entity registration: {e}[/red]")
        return False


@app.command("create-subaccount", rich_help_panel="Entity Management")
def create_subaccount(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
    account_size: Annotated[Optional[float], typer.Option(
        "--account-size", help="Account size in USD"
    )] = None,
    asset_class: Annotated[Optional[str], typer.Option(
        "--asset-class", help="Asset class selection (crypto, forex, equities)"
    )] = None,
    prompt: PromptFlag = True,
):
    """
    Create a new subaccount for an entity
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    # Prompt for account_size if not provided
    if account_size is None:
        account_size = FloatPrompt.ask("Enter subaccount size in USD")

    # Prompt for asset_class if not provided
    if asset_class is None:
        assets = ["crypto", "forex", "equities"]
        console.print("\nAvailable asset classes:")
        for idx, asset in enumerate(assets, start=1):
            console.print(f"{idx}. {asset}")

        choice = IntPrompt.ask(
            "\nEnter the number of the asset class",
            choices=[str(i) for i in range(1, len(assets) + 1)],
            show_choices=False,
        )
        asset_class = assets[choice - 1]

    console.print("[blue]Creating subaccount on Vanta Network[/blue]")

    # Determine base URL
    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    max_account_size = 100_000
    cost_per_theta = 5000 if account_size > 10000 else 2500

    # Step 2: Validate account size
    if account_size > max_account_size:
        error_msg = f"Account size ${account_size:,.0f} exceeds maximum ${max_account_size:,.0f}"
        console.print(f"[red]{error_msg}[/red]")
        return {"status": "error", "message": error_msg}

    if account_size <= 0:
        error_msg = "Account size must be positive"
        console.print(f"[red]{error_msg}[/red]")
        return {"status": "error", "message": error_msg}

    # Step 3: Calculate required collateral
    required_theta = account_size / cost_per_theta

    # Display configuration
    config_table = Table(title="Subaccount Creation Configuration", show_header=True, header_style="bold cyan")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Network", "Testnet" if network == "test" else "Mainnet")
    config_table.add_row("Account Size", f"${account_size:,.2f}")
    config_table.add_row("Asset Class", asset_class)
    config_table.add_row("Cost per Theta", f"${cost_per_theta:,.0f}")
    config_table.add_row("Required Collateral", f"{required_theta:.4f} Theta")

    console.print(config_table)

    # Step 4: Get password
    password = getpass.getpass(prompt='Enter your wallet password: ')

    try:
        coldkey = wallet.get_coldkey(password=password)
        hotkey = wallet.hotkey
    except Exception as e:
        error_msg = f"Failed to unlock wallet: {e}"
        console.print(f"[red]{error_msg}[/red]")
        return {"status": "error", "message": error_msg}

    # Step 5: Confirm creation
    if prompt:
        confirm = typer.confirm(
            f"Create subaccount for entity {hotkey.ss58_address} with "
            f"${account_size:,.2f} account size and {asset_class} asset class "
            f"(costs {required_theta:.4f} Theta)?"
        )
        if not confirm:
            cancel_msg = "Subaccount creation cancelled"
            console.print(f"[yellow]{cancel_msg}[/yellow]")
            return {"status": "error", "message": cancel_msg}

    # Step 6: Ensure sufficient collateral
    response = make_api_request(f"/collateral/balance/{hotkey.ss58_address}", method="GET", base_url=base_url)
    if not response or response.get("balance_theta", 0) < required_theta:
        balance = response.get('balance_theta', 0) if response else 0
        error_msg = f"Insufficient collateral for subaccount creation: {balance:.4f} Theta available, {required_theta:.4f} Theta required"
        console.print(f"[red]{error_msg}[/red]")
        return {"status": "error", "message": error_msg}

    # Step 7: Prepare and sign subaccount creation request
    console.print("\n[cyan]Signing subaccount creation request...[/cyan]")

    subaccount_data = {
        "entity_coldkey": coldkey.ss58_address,
        "entity_hotkey": hotkey.ss58_address,
        "account_size": account_size,
        "asset_class": asset_class
    }

    # Create message to sign (sorted JSON)
    message = json.dumps(subaccount_data, sort_keys=True)

    # Sign the message with coldkey
    signature = coldkey.sign(message.encode('utf-8')).hex()

    # Prepare payload
    payload = {
        "entity_coldkey": coldkey.ss58_address,
        "entity_hotkey": hotkey.ss58_address,
        "account_size": account_size,
        "asset_class": asset_class,
        "signature": signature
    }

    # Step 8: Send subaccount creation request
    console.print("\n[cyan]Sending subaccount creation request...[/cyan]")

    try:
        response = make_api_request("/entity/create-subaccount", payload, base_url=base_url)

        if response is None:
            error_msg = "Subaccount creation failed - no response from API"
            console.print(f"[red]{error_msg}[/red]")
            return {"status": "error", "message": error_msg}

        # Check success
        if response.get("status") == "success":
            success_msg = response.get('message', 'Subaccount created successfully')
            subaccount = response.get('subaccount', {})

            console.print(f"[green]{success_msg}[/green]")

            # Display success info
            success_table = Table(title="Subaccount Created Successfully", show_header=True, header_style="bold green")
            success_table.add_column("Field", style="cyan")
            success_table.add_column("Value", style="green")

            success_table.add_row("Synthetic Hotkey", subaccount.get('synthetic_hotkey'))
            success_table.add_row("Subaccount ID", str(subaccount.get('subaccount_id')))
            success_table.add_row("Subaccount UUID", subaccount.get('subaccount_uuid'))
            success_table.add_row("Account Size", f"${subaccount.get('account_size'):,.2f}")
            success_table.add_row("Asset Class", subaccount.get('asset_class'))
            success_table.add_row("Status", subaccount.get('status'))
            success_table.add_row("Collateral Charged", f"{required_theta:.4f} Theta")

            console.print(success_table)

            success_panel = Panel.fit(
                f"🎉 Subaccount created successfully!\n"
                f"Synthetic Hotkey: {subaccount.get('synthetic_hotkey')}\n"
                f"Use this hotkey to place orders and track performance.",
                style="bold green",
                border_style="green"
            )
            console.print(success_panel)

            return {
                "status": "success",
                "message": success_msg,
                "subaccount": {
                    "synthetic_hotkey": subaccount.get('synthetic_hotkey'),
                    "subaccount_id": subaccount.get('subaccount_id'),
                    "subaccount_uuid": subaccount.get('subaccount_uuid'),
                    "account_size": subaccount.get('account_size'),
                    "asset_class": subaccount.get('asset_class'),
                    "status": subaccount.get('status')
                },
                "collateral_charged": required_theta
            }
        else:
            error_message = response.get("error") or "Unknown error occurred"
            console.print(f"[red]Subaccount creation failed: {error_message}[/red]")
            return {"status": "error", "message": f"Subaccount creation failed: {error_message}"}

    except Exception as e:
        error_msg = f"Error during subaccount creation: {e}"
        console.print(f"[red]{error_msg}[/red]")
        return {"status": "error", "message": error_msg}


@app.command("apikey", rich_help_panel="Entity Management")
def apikey(
    wallet_name: WalletName = None,
    wallet_path: WalletPath = None,
    wallet_hotkey: WalletHotkey = None,
    network: Network = "finney",
):
    """
    Request or retrieve an API key for a registered entity miner.

    The API key grants tier-200 WebSocket access, allowing the entity miner
    to receive real-time subaccount dashboard updates. The key is idempotent —
    calling this command again returns the same key if one has already been
    issued for the entity.
    """
    wallet = wallet_ask(wallet_name, wallet_path, wallet_hotkey)

    base_url = VANTA_API_BASE_URL_TESTNET if network == "test" else VANTA_API_BASE_URL_MAINNET

    password = getpass.getpass(prompt="Enter your wallet password: ")

    try:
        coldkey = wallet.get_coldkey(password=password)
        hotkey = wallet.hotkey
    except Exception as e:
        console.print(f"[red]Failed to unlock wallet: {e}[/red]")
        return False

    console.print("\n[cyan]Signing API key request...[/cyan]")

    request_data = {
        "entity_coldkey": coldkey.ss58_address,
        "entity_hotkey": hotkey.ss58_address,
    }
    message = json.dumps(request_data, sort_keys=True)
    signature = coldkey.sign(message.encode("utf-8")).hex()

    payload = {
        "entity_coldkey": coldkey.ss58_address,
        "entity_hotkey": hotkey.ss58_address,
        "signature": signature,
    }

    try:
        response = make_api_request("/request-api-key", payload, base_url=base_url)

        if response is None:
            console.print("[red]API key request failed - no response from validator[/red]")
            return False

        api_key = response.get("api_key")
        if api_key:
            result_table = Table(title="Entity API Key", show_header=True, header_style="bold cyan")
            result_table.add_column("Field", style="cyan")
            result_table.add_column("Value", style="green")
            result_table.add_row("Entity Hotkey", hotkey.ss58_address)
            result_table.add_row("API Key", api_key)
            console.print(result_table)

            console.print(Panel.fit(
                "Store this API key as 'validator_api_key' in your entity miner secrets file.\n"
                "It grants tier-200 WebSocket access to your subaccount dashboards.",
                style="bold green",
                border_style="green"
            ))
            return True
        else:
            error_message = response.get("error") or "Unknown error occurred"
            console.print(f"[red]API key request failed: {error_message}[/red]")
            return False

    except Exception as e:
        console.print(f"[red]Error requesting API key: {e}[/red]")
        return False
